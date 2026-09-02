from typing import Any

import torch
from torch import Tensor

################################################
#            Fast tanh Soft Capping            #
################################################
# adapted from PyTorch FlexAttention "attention-gym" example
# -- Start of (modified) source: https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/mods/softcapping.py --
_PTX_TANH_LOWERED = False

try:
    # use inductor internals if present; otherwise we keep ptx unavailable
    from functools import partial
    from torch._inductor.virtualized import ops
    from torch._inductor.lowering import make_pointwise, register_lowering

    # avoid double-registrations when re-running in dev
    try:
        torch.ops.approx.tanh
        _ALREADY_REGISTERED = True
    except (AttributeError, RuntimeError):
        _ALREADY_REGISTERED = False

    if not _ALREADY_REGISTERED:
        @torch.library.custom_op("approx::tanh", mutates_args=())
        def _tanh_approx(inp: Tensor) -> Tensor:
            # eager fallback path
            return torch.tanh(inp)

        @_tanh_approx.register_fake
        def _(inp: Tensor) -> Tensor:
            return torch.tanh(inp)

        def _tanh_approx_lowering(inp: Tensor) -> Tensor:
            # inline ptx (f32)
            fn = partial(ops.inline_asm_elementwise, asm="tanh.approx.f32 $0, $1;")
            return make_pointwise(fn)(inp)

        register_lowering(torch.ops.approx.tanh)(_tanh_approx_lowering)

    _PTX_TANH_LOWERED = True
except Exception:
    # any failure leaves ptx unavailable
    _PTX_TANH_LOWERED = False

class _TanhApproxFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: Tensor) -> Tensor:
        # ptx op is f32; caller upcasts beforehand
        y = torch.ops.approx.tanh(x)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx: Any, grad_out: Tensor) -> Tensor:
        (y,) = ctx.saved_tensors
        # d/dx tanh(x) = 1 - tanh^2(x)
        return grad_out * (1 - y * y)

def _tanh_ptx(x: Tensor) -> Tensor:
    # expects float32
    return _TanhApproxFn.apply(x)

def generate_tanh_softcap(soft_cap: float, backend: str):
    """
    returns a score_mod closure for flexattention that applies tanh soft-capping.

    backend: "ptx" | "rational" | "exact"
      - "ptx": use inline ptx tanh.approx (requires inductor lowering to be available)
      - "rational": use polynomial approx (fast & portable)
      - "exact": use torch.tanh
    """
    sc = float(soft_cap)

    if backend == "clamp":
        def fn(score):
            # clamp to [-sc, sc] with no transcendental ops
            return torch.clamp(score, min=-sc, max=sc)
    elif backend == "exact":
        def fn(score):
            return sc * torch.tanh(score / sc)
    elif backend == "rational":
        def fn(score):
            z = (score / sc).to(torch.float32)
            x2 = z * z
            y = z * (27.0 + x2) / (27.0 + 9.0 * x2)
            return (y * sc).to(dtype=score.dtype)
    elif backend == "ptx":
        if not _PTX_TANH_LOWERED:
            raise RuntimeError(
                "tanh_backend='ptx' requested, but ptx lowering is unavailable. "
                "pick 'rational' or 'exact'."
            )
        def fn(score):
            z32 = (score / sc).to(torch.float32)
            y32 = _tanh_ptx(z32)
            return (y32.to(score.dtype) * sc)
    else:
        raise ValueError(f"unknown tanh backend: {backend}")

    def tanh_softcap(score, b, h, q_idx, kv_idx):
        return fn(score)

    tanh_softcap.__name__ = f"tanh_softcap_{backend}_{int(sc)}"
    return tanh_softcap
# -- End of (modified) source: https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/mods/softcapping.py --
