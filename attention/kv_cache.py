from typing import List, Tuple, Union

import torch
from torch import Tensor

from config.model import GPTConfig

################################################
#                   KV Cache                   #
################################################
class KVCache:
    # Pre-allocated KV cache for autoregressive generation with wave batching.
    # Each sequence in the batch tracks its own write position (self.pos is a per-sequence tensor),
    # allowing batched decode of variable-length prompts without padding in the cache.
    # During decode, new K/V entries are scatter-written at each sequence's position,
    # and the returned K/V slice goes up to max(pos) across the batch — sequences with
    # shorter histories have padding K/V at some positions, masked out via attn_mask in SDPA.
    # SDPA is used instead of FlexAttention for the decode path because our num_blocks_per_axis variable
    # is used for both Q and KV, assumeing Q_LEN == KV_LEN.
    # and during decode Q_LEN=1.
    # There is also no benefit from block-level optimization with a single query token.
    def __init__(
            self,
            gpt_config: 'GPTConfig',
            batch_size: int,
            max_seq_len: int,
            device: Union[str, torch.device],
            dtype: torch.dtype,
            ) -> None:
        self.pos = torch.zeros(batch_size, dtype=torch.long, device=device)
        self.k: List[Tensor] = []
        self.v: List[Tensor] = []
        _full_set = set(gpt_config.full_attention_layers)
        for i in range(gpt_config.n_layers):
            if i in _full_set:
                n_kv = gpt_config.global_n_kv_heads
                h_size = gpt_config.global_attn_dim // gpt_config.n_heads
            else:
                n_kv = gpt_config.n_kv_heads
                h_size = gpt_config.attn_dim // gpt_config.n_heads
            self.k.append(torch.zeros(batch_size, n_kv, max_seq_len, h_size, device=device, dtype=dtype))
            self.v.append(torch.zeros(batch_size, n_kv, max_seq_len, h_size, device=device, dtype=dtype))

    def update(self, layer_idx: int, new_k: Tensor, new_v: Tensor) -> Tuple[Tensor, Tensor]:
        # write new K/V at each sequence's position and return the valid cache slice.
        # for prefill (new_k seq_len > 1): all sequences start at 0, regular slice write.
        # for decode (new_k seq_len == 1): sequences may be at different positions, use scatter.
        seq_len = new_k.size(2)
        if seq_len > 1:
            # prefill: all sequences write starting at 0
            self.k[layer_idx][:, :, :seq_len] = new_k
            self.v[layer_idx][:, :, :seq_len] = new_v
        else:
            # decode: scatter at per-sequence positions
            # pos: (batch_size,) → (batch_size, 1, 1, 1) for scatter along dim=2
            pos_idx = self.pos[:, None, None, None].expand_as(new_k) # new_k (gpu_batch_size, n_kv_heads, 1, head_size)
            # scatter along second dim, write on pos_idx, write new_k (or new_v)
            self.k[layer_idx].scatter_(2, pos_idx, new_k)
            self.v[layer_idx].scatter_(2, pos_idx, new_v)
        # return cache up to max position across the batch (+ seq_len for prefill, +1 for decode)
        end = int(self.pos.max().item()) + seq_len
        return self.k[layer_idx][:, :, :end], self.v[layer_idx][:, :, :end]

