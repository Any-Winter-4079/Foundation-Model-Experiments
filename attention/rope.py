from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

################################################
#                      RoPE                    #
################################################
class Rotary(nn.Module):
    def __init__(self, head_size: int, base_theta: int, max_seq_len: int) -> None:
        super().__init__()
        assert head_size % 2 == 0, "RoPE needs even head_size"
        # head_size/2,
        # frequencies decay geometrically down to ~1/base_theta with ratio 1/base_theta^(2/head_size)
        inv_freq = 1.0 / (base_theta ** (torch.arange(0, head_size, 2, dtype=torch.float32) / head_size))
        # register_buffer registers as a non-trainable buffer that moves/casts with the model (to device/dtype)
        # persistent=False avoids taking checkpoint space (won't be in state_dict)
        # buffers are broadcast to all ranks (broadcast_buffers=True), so this stays in sync
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(max_seq_len, dtype=torch.float32)
        # max_seq_len, head_size/2
        freqs = torch.outer(t, self.inv_freq)
        # 1, 1, max_seq_len, head_size/2
        cos = freqs.cos()[None, None, :, :]
        sin = freqs.sin()[None, None, :, :]
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def get_cos_sin(
            self,
            seq_len: int,
            dtype: torch.dtype,
            device: Union[str, torch.device],
            position_ids: Optional[Tensor] = None,
            ) -> Tuple[Tensor, Tensor]:
        # slice to current length and cast on the fly to avoid
        # shape mutation of buffers
        if position_ids is not None:
            # per-sequence positions for batched generation with right-padding.
            # cos_cached is (1, 1, max_seq_len, head_size/2) — a lookup table where
            # row i holds the cos rotation vector for position i.
            # cos_cached[0, 0] collapses the leading 1-dims → (max_seq_len, head_size/2).
            # position_ids is (gpu_batch_size, seq_len) where each integer value picks
            # which row to fetch from the table. The (i, j) coordinates of position_ids
            # determine where in the output the result lands; the integer value at
            # position_ids[i, j] determines which table row is fetched.
            # example (max_seq_len=6, head_size/2=3):
            #   cos_cached[0, 0] (6, 3):
            #     row 0: [0.1, 0.2, 0.3]   ← rotation vector for position 0
            #     row 1: [0.4, 0.5, 0.6]   ← rotation vector for position 1
            #     row 2: [0.7, 0.8, 0.9]   ← rotation vector for position 2
            #     row 3: [1.0, 1.1, 1.2]   ← rotation vector for position 3
            #     row 4: [1.3, 1.4, 1.5]   ← rotation vector for position 4
            #     row 5: [1.6, 1.7, 1.8]   ← rotation vector for position 5
            #   seq 0: [tok_A, tok_B, tok_C, PAD,   PAD  ]
            #   seq 1: [tok_D, tok_E, tok_F, tok_G, tok_H]
            #   position_ids = [[0, 1, 2, 0, 0],   ← padding gets 0 (arbitrary, masked out by is_causal)
            #                   [0, 1, 2, 3, 4]]
            #   cos_cached[0, 0, position_ids, :] →
            #     seq 0: [[0.1, 0.2, 0.3],   ← tok_A gets rotation for position 0
            #             [0.4, 0.5, 0.6],   ← tok_B gets rotation for position 1
            #             [0.7, 0.8, 0.9],   ← tok_C gets rotation for position 2
            #             [0.1, 0.2, 0.3],   ← PAD gets rotation for position 0 (irrelevant)
            #             [0.1, 0.2, 0.3]]   ← PAD gets rotation for position 0 (irrelevant)
            #     seq 1: [[0.1, 0.2, 0.3],   ← tok_D gets rotation for position 0
            #             [0.4, 0.5, 0.6],   ← tok_E gets rotation for position 1
            #             [0.7, 0.8, 0.9],   ← tok_F gets rotation for position 2
            #             [1.0, 1.1, 1.2],   ← tok_G gets rotation for position 3
            #             [1.3, 1.4, 1.5]]   ← tok_H gets rotation for position 4
            #   result: (gpu_batch_size, seq_len, head_size/2) = (2, 5, 3)
            # this is equivalent to torch.stack([cos_cached[0, 0, position_ids[i], :] for i in range(gpu_batch_size)])
            # i.e., per-sequence 1D lookups done in one operation via 2D advanced indexing.
            # .unsqueeze(1) adds the n_heads broadcast dim → (gpu_batch_size, 1, seq_len, head_size/2)
            cos = self.cos_cached[0, 0, position_ids, :].unsqueeze(1).to(device=device, dtype=dtype)
            sin = self.sin_cached[0, 0, position_ids, :].unsqueeze(1).to(device=device, dtype=dtype)
        else:
            # (1, 1, seq_len, head_size/2), with first 2 to broadcast against gpu_batch_size, n_heads
            cos = self.cos_cached[..., :seq_len, :].to(device=device, dtype=dtype)
            sin = self.sin_cached[..., :seq_len, :].to(device=device, dtype=dtype)
        return cos, sin

def apply_rotation(
        q: Tensor,
        k: Tensor,
        cos: Tensor,
        sin: Tensor,
        ) -> Tuple[Tensor, Tensor]:
    gpu_batch_size, n_heads, seq_len, head_size = q.shape
    _, n_kv_heads_k, _, _ = k.shape

    # gpu_batch_size, n_heads, seq_len, head_size // 2, 2
    q_pairs = q.contiguous().reshape(gpu_batch_size, n_heads, seq_len, head_size // 2, 2)
    k_pairs = k.contiguous().reshape(gpu_batch_size, n_kv_heads_k, seq_len, head_size // 2, 2)

    # gpu_batch_size, n_heads, seq_len, head_size/2 each
    q_x1, q_x2 = q_pairs[..., 0], q_pairs[..., 1]
    k_x1, k_x2 = k_pairs[..., 0], k_pairs[..., 1]

    # gpu_batch_size, n_heads, seq_len, head_size
    q_rot = torch.stack([q_x1 * cos - q_x2 * sin, q_x1 * sin + q_x2 * cos], dim=-1)
    k_rot = torch.stack([k_x1 * cos - k_x2 * sin, k_x1 * sin + k_x2 * cos], dim=-1)

    q = q_rot.reshape(gpu_batch_size, n_heads, seq_len, head_size)
    k = k_rot.reshape(gpu_batch_size, n_kv_heads_k, seq_len, head_size)
    
    return q, k

