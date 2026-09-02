from typing import Iterable, Optional

import torch
import torch.nn as nn
from torch import Tensor

from optimizers.polar_express import zeropower_via_polar_express

################################################
#                     Muon                     #
################################################
# In Muon, the goal is:
#   For 2D weight matrices (e.g., Linear weights of shape (out_features, in_features)),
#   build an SGD+momentum update g, then REPLACE it with its nearest orthogonal
#   matrix (orthogonalization), and apply that as the step. This helps keep layers
#   well-conditioned and discourages rank collapse.
# ---------
# Muon vs AdamW:
#   Muon uses SGD + momentum to form g, then orthogonalizes g (via the backend) and scales it.
#   Use Muon for 2D block weights (e.g., transformer.h.*.weight).
#   Keep embeddings, layer norms, biases, and the final lm_head on AdamW.
#   Train these disjoint  param sets with their respective optimizers in parallel.
#   To use it with 4D convolutional filters, flatten the last 3 dimensions.
# ---------
# The backend is simply the algorithm used to orthogonalize the 2D update matrix. Three choices are:
#   svd:
#       Exact, via SVD. If G = U S V^T, the projection of G (in Frobenius norm) onto the Stiefel
#       manifold (the set of all orthonormal k-frames, with a k-frame an ordered set of 
#       k linearly independent vectors in an n-dimensional vector space) is U V^T 
#       (i.e., the closest matrix with orthonormal columns/rows).
#   newtonschulz5:
#       Fast, iterative approximation (GPU-friendly) using a quintic Newton-Schulz iteration.
#       It avoids an SVD on every step, runs well in bfloat16, and plays nice with torch.compile.
#   polarexpress:
#       Polar Express sign method (fast iterative), fixed 5-step coefficients.
# ---------
# It is recommended to use Nesterov-style momentum in the internal SGD.
# ---------
# # QKV note:
#   If a fused projection has shape (3d, d), we split it into three (d, d) blocks and
#   orthogonalize each block separately. After orthogonalization we scale updates by
#   sqrt(max(m, n)) so update magnitudes are well-behaved.
# ---------
# Why @torch.compile on newtonschulz5?
#   The kernel is a small loop of matmuls; compiling fuses ops and reduces Python overhead,
#   generating an efficient GPU kernel.
#   SVD is a single library call and doesn’t benefit.
# -----------------------------------------------------------

# (external)            Muon blog: https://kellerjordan.github.io/posts/muon/
# (another external)    Muon blog: https://jeremybernste.in/writing/deriving-muon

# -- Start of (modified) source: https://github.com/tyler-romero/nanogpt-speedrun/blob/main/src/train_gpt2.py --
# -----------------------------------------------------------------------------
# Muon optimizer
# Reference: https://kellerjordan.github.io/posts/muon/
def zeropower_via_svd(G: Tensor, steps: Optional[int] = None) -> Tensor:
    U, S, V = G.svd()
    return U @ V.T

@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, muon_eps: float = 1e-7) -> Tensor:
    """
    In-place & buffer-reusing variant:
      - uses preallocated mm(out=...) to avoid new tensors each iter
      - uses in-place fused linear combination to reduce allocator pressure
      - keeps the 'transpose-if-tall' trick to minimize flops
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)

    # orient to have rows <= cols so A = X X^T is the smaller square
    transposed = G.size(0) > G.size(1)
    X = G.T if transposed else G
    # scale so top singular value <= 1 (Frobenius norm upper-bounds spectral norm)
    denom = X.norm() + muon_eps
    X = (X / denom).to(torch.bfloat16)

    m, n = X.shape  # m <= n
    # reusable buffers (compiled graph will persist them between calls)
    A = torch.empty((m, m), dtype=X.dtype, device=X.device)
    B = torch.empty_like(X)
    AB = torch.empty_like(X)

    for _ in range(steps):
        # A = X @ X^T
        torch.mm(X, X.T, out=A)
        # B = A @ X
        torch.mm(A, X, out=B)
        # AB = A @ B  (quintic NS5 term)
        torch.mm(A, B, out=AB)
        # X = a*X + b*B + c*AB  (in-place fused)
        X.mul_(a).add_(B, alpha=b).add_(AB, alpha=c)

    X = X.T if transposed else X
    return X.to(G.dtype)

zeropower_backends = dict(
    svd=zeropower_via_svd,
    newtonschulz5=zeropower_via_newtonschulz5,
    polarexpress=zeropower_via_polar_express,
)

class Muon(torch.optim.Optimizer):
    """
    Muon: MomentUm Orthogonalized by Newton-Schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer assumes that all parameters passed in are 2D.
    - It should not be used for the embedding layer, the final fully connected layer, or any {0,1}-D
    parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.
    - We believe it is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.
    - We have not yet tried this optimizer for training scenarios larger than NanoGPT (124M).

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        backend: The chosen backend for the orthogonalization step. (recommended: 'newtonschulz5')
        backend_steps: The number of iteration steps to use in the backend, if it is iterative.
    """

    def __init__(
            self,
            params: Iterable[nn.Parameter],
            lr: float = 3e-4,
            momentum: float = 0.95,
            nesterov: bool = True,
            backend: str = "newtonschulz5",
            backend_steps: int = 5,
            ) -> None:
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, backend=backend, backend_steps=backend_steps)
        super().__init__(params, defaults)

    def step(self) -> None:
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            zeropower_backend = zeropower_backends[group['backend']]
            for p in group['params']:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                if group['nesterov']:
                    g.add_(buf, alpha=momentum)
                if g.size(0) == 3 * g.size(1):  # split grouped QKV parameters
                    g = torch.cat([zeropower_backend(g1, steps=group['backend_steps']) for g1 in g.split(g.size(1))])
                    scale = g.size(1) ** 0.5
                else:
                    g = zeropower_backend(g, steps=group['backend_steps'])
                    scale = max(g.size(0), g.size(1)) ** 0.5  # scale so the update has ≈ unit mean-squared magnitude
                p.data.add_(g, alpha=-lr * scale)
# -- End of (modified) source: https://github.com/tyler-romero/nanogpt-speedrun/blob/main/src/train_gpt2.py --

