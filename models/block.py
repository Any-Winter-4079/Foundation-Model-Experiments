from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask

from attention.kv_cache import KVCache
from attention.self_attention import SelfAttention
from config.model import GPTConfig
from config.training import TrainingConfig
from models.feed_forward.dense import MLP

################################################
#                    Block                     #
################################################
class Block(nn.Module):
    def __init__(self, gpt_config: GPTConfig, training_config: TrainingConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.use_bf16_autocast = training_config.use_bf16_autocast
        self.cast_1d_weights_params_and_scales_to_weight_dtype_if_no_autocast = training_config.cast_1d_weights_params_and_scales_to_weight_dtype_if_no_autocast

        self.use_weighted_residual_path = gpt_config.use_weighted_residual_path
        self.use_weighted_main_path = gpt_config.use_weighted_main_path

        if self.use_weighted_residual_path:
            self.attn_residual_scale = nn.Parameter(torch.full((1,), gpt_config.weighted_residual_path_init))
            self.mlp_residual_scale = nn.Parameter(torch.full((1,), gpt_config.weighted_residual_path_init))

        if self.use_weighted_main_path:
            self.attn_main_scale = nn.Parameter(torch.full((1,), gpt_config.weighted_main_path_init))
            self.mlp_main_scale = nn.Parameter(torch.full((1,), gpt_config.weighted_main_path_init))

        if (not self.use_weighted_residual_path) and (not self.use_weighted_main_path):
            self.residual_scale = nn.Parameter(torch.full((1,), (2 * gpt_config.n_layers) ** -0.5), requires_grad=False)

        if gpt_config.norm_type.lower() == "rms":
            self.ln_1 = nn.RMSNorm(gpt_config.d_model)
        else:
            self.ln_1 = nn.LayerNorm(gpt_config.d_model)
        self.attn = SelfAttention(gpt_config, training_config, layer_idx=layer_idx)
        if gpt_config.norm_type.lower() == "rms":
            self.ln_2 = nn.RMSNorm(gpt_config.d_model)
        else:
            self.ln_2 = nn.LayerNorm(gpt_config.d_model)
        self.mlp = MLP(gpt_config, training_config)

    def forward(
            self,
            x: Tensor,
            flex_attn_block_mask: Optional[BlockMask] = None,
            sdpa_attn_mask: Optional[Tensor] = None,
            kv_cache: Optional['KVCache'] = None,
            layer_idx_for_kv_cache: Optional[int] = None,
            position_ids: Optional[Tensor] = None,
            ) -> Tensor:
        kv_kwargs = dict(kv_cache=kv_cache, layer_idx_for_kv_cache=layer_idx_for_kv_cache, position_ids=position_ids)
        # weighted path
        if self.use_weighted_residual_path or self.use_weighted_main_path:
            # cast if no autocast and cast is requested (1D Norm can be fp32, x can be bf16)
            x_ln1 = x
            if not self.use_bf16_autocast and self.cast_1d_weights_params_and_scales_to_weight_dtype_if_no_autocast and x_ln1.dtype != self.ln_1.weight.dtype:
                x_ln1 = x_ln1.to(dtype=self.ln_1.weight.dtype)

            # Attention
            if self.use_weighted_residual_path and not self.use_weighted_main_path:
                # x = alpha_1 * x + Attn(norm(x))
                x = self.attn_residual_scale * x + self.attn(self.ln_1(x_ln1), flex_attn_block_mask=flex_attn_block_mask, sdpa_attn_mask=sdpa_attn_mask, **kv_kwargs)
            elif not self.use_weighted_residual_path and self.use_weighted_main_path:
                # x = x + beta_1 * Attn(norm(x))
                x = x + self.attn_main_scale * self.attn(self.ln_1(x_ln1), flex_attn_block_mask=flex_attn_block_mask, sdpa_attn_mask=sdpa_attn_mask, **kv_kwargs)
            else:
                # x = alpha_1 * x + beta_1 * Attn(norm(x))
                x = self.attn_residual_scale * x + self.attn_main_scale * self.attn(self.ln_1(x_ln1), flex_attn_block_mask=flex_attn_block_mask, sdpa_attn_mask=sdpa_attn_mask, **kv_kwargs)

            # cast if no autocast and cast is requested (1D Norm can be fp32, x can be bf16)
            x_ln2 = x
            if not self.use_bf16_autocast and self.cast_1d_weights_params_and_scales_to_weight_dtype_if_no_autocast and x_ln2.dtype != self.ln_2.weight.dtype:
                x_ln2 = x_ln2.to(dtype=self.ln_2.weight.dtype)

            # MLP
            if self.use_weighted_residual_path and not self.use_weighted_main_path:
                # x = alpha_2 * x + MLP(norm(x))
                x = self.mlp_residual_scale * x + self.mlp(self.ln_2(x_ln2))
            elif not self.use_weighted_residual_path and self.use_weighted_main_path:
                # x = x + beta_2 * MLP(norm(x))
                x = x + self.mlp_main_scale * self.mlp(self.ln_2(x_ln2))
            else:
                # x = alpha_2 * x + beta_2 * MLP(norm(x))
                x = self.mlp_residual_scale * x + self.mlp_main_scale * self.mlp(self.ln_2(x_ln2))

        # standard path
        else:
            # cast if no autocast and cast is requested (1D Norm can be fp32, x can be bf16)
            x_ln1 = x
            if not self.use_bf16_autocast and self.cast_1d_weights_params_and_scales_to_weight_dtype_if_no_autocast and x_ln1.dtype != self.ln_1.weight.dtype:
                x_ln1 = x_ln1.to(dtype=self.ln_1.weight.dtype)
            x = x + self.residual_scale * self.attn(self.ln_1(x_ln1), flex_attn_block_mask=flex_attn_block_mask, sdpa_attn_mask=sdpa_attn_mask, **kv_kwargs)
            # cast if no autocast and cast is requested (1D Norm can be fp32, x can be bf16)
            x_ln2 = x
            if not self.use_bf16_autocast and self.cast_1d_weights_params_and_scales_to_weight_dtype_if_no_autocast and x_ln2.dtype != self.ln_2.weight.dtype:
                x_ln2 = x_ln2.to(dtype=self.ln_2.weight.dtype)
            x = x + self.residual_scale * self.mlp(self.ln_2(x_ln2))
        return x
