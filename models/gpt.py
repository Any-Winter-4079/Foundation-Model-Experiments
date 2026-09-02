import inspect
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn.attention.flex_attention import BlockMask, and_masks, create_block_mask

from attention.kv_cache import KVCache
from config.model import GPTConfig
from config.training import TrainingConfig
from models.block import Block
from optimizers.muon import Muon

################################################
#                     GPT                      #
################################################
class GPT(nn.Module):
    def __init__(self, gpt_config: GPTConfig, training_config: TrainingConfig) -> None:
        super().__init__()
        self.max_seq_len = training_config.max_seq_len
        self.pad_token_id = gpt_config.pad_token_id
        self.pos_encoding_type = gpt_config.pos_encoding_type.lower()
        self.optimizer_type = training_config.optimizer_type.lower()

        self.muon_lr_scale = training_config.muon_lr_scale
        self.muon_backend = training_config.muon_backend
        self.muon_backend_steps = training_config.muon_backend_steps
        self.muon_momentum = training_config.muon_momentum
        self.muon_use_nesterov = training_config.muon_use_nesterov

        # <-- Uncomment for gradient norm clipping -->
        # self.use_grad_norm_clipping = gpt_config.use_grad_norm_clipping
        # self.gradient_clipping_norm = gpt_config.gradient_clipping_norm
        # <-- Uncomment for gradient norm clipping -->

        self.use_qk_norm = gpt_config.use_qk_norm
        self.use_qk_debug_log = gpt_config.use_qk_debug_log

        # <-- Uncomment for logit soft-capping -->
        # self.use_lm_head_logit_softcapping = gpt_config.use_lm_head_logit_softcapping
        # self.lm_head_logit_softcap = gpt_config.lm_head_logit_softcap
        # self._logits_absmax_stats = None
        # <-- Uncomment for logit soft-capping -->

        self.is_causal = gpt_config.is_causal

        self.use_flex_attention = gpt_config.use_flex_attention
        self.flex_block_size = training_config.flex_block_size

        self.use_doc_masking = gpt_config.use_doc_masking

        self.use_attn_logit_softcapping = gpt_config.use_attn_logit_softcapping
        self.attn_logit_softcap = gpt_config.attn_logit_softcap

        self.use_bf16_autocast = training_config.use_bf16_autocast
        self.cast_1d_weights_params_and_scales_to_weight_dtype_if_no_autocast = training_config.cast_1d_weights_params_and_scales_to_weight_dtype_if_no_autocast
        self.data_uses_padding = training_config.data_uses_padding
        self.keep_fp32_loss = training_config.keep_fp32_loss

        swa_min_window_size = min(training_config.swa_schedule_values)
        if (self.use_doc_masking or self.use_attn_logit_softcapping or swa_min_window_size < training_config.max_seq_len):
            assert self.use_flex_attention, "Document masking / SWA / attention soft-capping require use_flex_attention=True"
        self.initial_sliding_window_size = training_config.swa_initial_window_size
        # register as a 0-dim tensor buffer so torch.compile treats it as a dynamic tensor input
        self.register_buffer(
            "sliding_window_size",
            torch.tensor(self.initial_sliding_window_size, dtype=torch.int32),
            # no need to save in model checkpoints, it's set by step count
            persistent=False
        )

        # full (global) attention layer routing
        if gpt_config.full_attention_layers:
            for idx in gpt_config.full_attention_layers:
                assert 0 <= idx < gpt_config.n_layers, f"full_attention_layers index {idx} out of range [0, {gpt_config.n_layers})"
        _full_set = set(gpt_config.full_attention_layers)
        self.has_full_attention_layers = bool(_full_set)
        self.layer_is_full_attention = [i in _full_set for i in range(gpt_config.n_layers)]

        if self.pos_encoding_type == "rope" or self.pos_encoding_type == "nope":
            self.transformer = nn.ModuleDict(dict(
                wte = nn.Embedding(gpt_config.vocab_size, gpt_config.d_model),
                h = nn.ModuleList([Block(gpt_config, training_config, layer_idx=i) for i in range(gpt_config.n_layers)]),
                ln_f = nn.RMSNorm(gpt_config.d_model) if gpt_config.norm_type.lower() == "rms" else nn.LayerNorm(gpt_config.d_model),
            ))
        else:
            self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(gpt_config.vocab_size, gpt_config.d_model),
            wpe = nn.Embedding(training_config.max_seq_len, gpt_config.d_model),
            h = nn.ModuleList([Block(gpt_config, training_config, layer_idx=i) for i in range(gpt_config.n_layers)]),
            ln_f = nn.RMSNorm(gpt_config.d_model) if gpt_config.norm_type.lower() == "rms" else nn.LayerNorm(gpt_config.d_model),
        ))
        self.lm_head = nn.Linear(gpt_config.d_model, gpt_config.vocab_size, bias=False)
        
        self.use_tied_embeddings = gpt_config.use_tied_embeddings
        if self.use_tied_embeddings:
            self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def configure_optimizers(
            self,
            lr: float,
            adamw_betas: Tuple[float, float],
            adamw_eps: float,
            adamw_weight_decay: float,
            device_type: str,
            master_process: bool,
            log_buffer: List[str],
            ) -> Dict[str, torch.optim.Optimizer]:
        # collect trainable params
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        if master_process:
            # log param dtypes (bf16/fp32)
            bf16_params = [pn for pn, p in param_dict.items() if p.dtype == torch.bfloat16]
            fp32_params = [pn for pn, p in param_dict.items() if p.dtype == torch.float32]
            message = (
                f"optimizer parameters:\n"
                f"\tbfloat16 params: {len(bf16_params)} tensors{' (e.g., ' + bf16_params[0] + ')' if bf16_params else ''}\n"
                f"\tfloat32 params: {len(fp32_params)} tensors{' (e.g., ' + fp32_params[0] + ')' if fp32_params else ''}"
            )
            print(message)
            log_buffer.append(message)

        # muon params: 2d weights, excluding embeddings and lm_head
        muon_param_names = set()
        if self.optimizer_type == "muon":
            for pn, p in param_dict.items():
                if p.dim() == 2 and ("wte" not in pn) and ("lm_head" not in pn):
                    muon_param_names.add(pn)

        # adamw gets everything else, split by decay rule
        decay_params = [p for n, p in param_dict.items() if (n not in muon_param_names) and (p.dim() >= 2)]
        nodecay_params = [p for n, p in param_dict.items() if (n not in muon_param_names) and (p.dim() < 2)]
        # create param groups
        optim_groups = [
            {'params': decay_params, 'weight_decay': adamw_weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            # log parameter counts
            messages = [
                f"num decayed parameter tensors (AdamW): {len(decay_params)}, with {num_decay_params:,} parameters",
                f"num non-decayed parameter tensors (AdamW): {len(nodecay_params)}, with {num_nodecay_params:,} parameters",
            ]
            if self.optimizer_type == "muon":
                muon_params_count = sum(param_dict[n].numel() for n in muon_param_names)
                messages.append(f"num Muon parameter tensors: {len(muon_param_names)}, with {muon_params_count:,} parameters")
            for message in messages:
                print(message)
                log_buffer.append(message)

        # enable fused adamw on cuda when available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and "cuda" in device_type
        if master_process:
            message = f"using fused AdamW: {use_fused}"
            print(message)
            log_buffer.append(message)

        # initialize adamw
        adamw = torch.optim.AdamW(optim_groups, lr=lr, betas=adamw_betas, eps=adamw_eps, fused=use_fused)

        # if not using Muon, still return a dict for consistency with the training loop
        if self.optimizer_type != "muon":
            return {"adamw": adamw}

        # initialize muon over its selected params
        muon_params = [param_dict[n] for n in muon_param_names]
        muon = Muon(
            muon_params,
            lr=lr * self.muon_lr_scale,
            momentum=self.muon_momentum,
            nesterov=self.muon_use_nesterov,
            backend=self.muon_backend,
            backend_steps=self.muon_backend_steps,
        )
        return {"adamw": adamw, "muon": muon}
    
    def _build_flex_attn_block_mask(
            self,
            attn_mask: Optional[Tensor],
            document_ids: Optional[Tensor],
            gpu_batch_size: int,
            seq_len: int,
            device: Union[str, torch.device],
            ignore_doc_mask: bool = False,
            apply_swa: bool = True,
            ) -> Optional[BlockMask]:
        # Attention builds, per sequence, a matrix of shape (seq_len, seq_len) when it does Q @ K^T
        # Splitting into 128x128 blocks (if block_size=128), each a 128×128 submatrix of 128 rows
        # (queries) × 128 columns (keys), we get:
        #                                       ┌───────────┬───────────┬───────────┬───────────┐
        #   queries 0-127                       │   (0,0)   │   (0,1)   │   (0,2)   │   (0,3)   │
        #                                       ├───────────┼───────────┼───────────┼───────────┤
        #   queries 128-255                     │   (1,0)   │   (1,1)   │   (1,2)   │   (1,3)   │
        #                                       ├───────────┼───────────┼───────────┼───────────┤
        #   ...                                 │   (2,0)   │   (2,1)   │   (2,2)   │   (2,3)   │
        #                                       ├───────────┼───────────┼───────────┼───────────┤
        # queries (seq_len-128)-(seq_len-1)     │   (3,0)   │   (3,1)   │   (3,2)   │   (3,3)   │
        #                                       └───────────┴───────────┴───────────┴───────────┘
        #                                        keys 0-127  keys 128-255    ...     keys (seq_len-128)-(seq_len-1)
        
        # In other words, (seq_len, seq_len) → (num_blocks_per_axis, num_blocks_per_axis)
        # The goal is to skip applying a mask to the full matrix, and prematurely
        # determine which blocks of 128x128 can compute full attention, 
        # which need partial masking and which can be dropped entirely
        
        # - full: every (q,k) pair in the tile is allowed → we can skip pointwise masking on this block
        # - partial: some (q,k) pairs are allowed → we must run pointwise masking on this block
        # - none: no (q,k) pairs are allowed → we can drop the block entirely

        # Now, we have various causes of masking:
        # is_causal: q shouldn't look into the future (we allow q >= k, prevent k > q)
        # use_doc_masking: q and k must share the same doc id (to avoid looking at keys from other docs)
        # attn_mask: both tokens must be valid (which, in our case, means both not being padding)
 
        # The speedup comes when we do not call create_block_mask at all and instead hand FlexAttention 
        # a BlockMask that already says, per query block, which KV blocks are:
        # - full-attn (no pointwise mask needed → mask_mod is skipped)
        # - partial (pointwise mask_mod still runs, but only on those tiles)

        block_size = self.flex_block_size
        assert seq_len % block_size == 0, f"seq_len {seq_len} must be a multiple of flex_block_size {block_size}"

        window_size = self.sliding_window_size

        # per token static (i.e., reused across training steps) steps mask
        def flex_static_mask_mod(batch_index, head_index, query_index, key_index):
            # causal mask
            if self.is_causal:
                keep_flag = query_index >= key_index
            else:
                keep_flag = torch.ones((), dtype=torch.bool, device=device)
            # SWA mask (skipped for full attention layers)
            if apply_swa:
                if self.is_causal:
                    keep_flag = keep_flag & ((query_index - key_index) <= window_size)
                else:
                    keep_flag = keep_flag & (torch.abs(query_index - key_index) <= window_size)
            return keep_flag
        mask_mod = flex_static_mask_mod

        # per token dynamic (i.e., can change across training steps) masks
        if attn_mask is not None:
            # custom (padding) mask
            def valid_mask_mod(batch_index, head_index, query_index, key_index):
                return attn_mask[batch_index, query_index] & attn_mask[batch_index, key_index]
            mask_mod = and_masks(mask_mod, valid_mask_mod)
        if self.use_doc_masking and not ignore_doc_mask and (document_ids is not None):
            # doc mask
            def same_doc_mod(batch_index, head_index, query_index, key_index):
                return document_ids[batch_index, query_index] == document_ids[batch_index, key_index]
            mask_mod = and_masks(mask_mod, same_doc_mod)

        # tiny bit faster path: supports gpu_batch_size >= 1, but currently needs is_causal, plus used 
        # only for training to avoid hitting re-compilation
        # Builds full/partial blocks up front, uses from_kv_blocks to skip applying mask_mod to certain blocks
        # -- Start of (modified) source: https://github.com/KellerJordan/modded-nanogpt/blob/master/records/071225_BosAlign/c1fd8a38-bb9f-45c4-8af0-d37f70c993f3.txt --
        if self.is_causal and self.training:
            # there are (seq_len // block_size)^2 total blocks, and seq_len // block_size per 
            # axis (i.e., per row or column), e.g., 1024 // 128 = 8 blocks per axis
            num_blocks_per_axis = seq_len // block_size
            # e.g., 0, 1,..., 7
            block_indices = torch.arange(num_blocks_per_axis, dtype=torch.int32, device=device)

            # Each block covers token indices:
            # for queries: q ∈ [i*block_size, (i+1)*block_size-1] with i=0..num_blocks_per_axis-1
            # for keys:    k ∈ [j*block_size, (j+1)*block_size-1] with j=0..num_blocks_per_axis-1
            # and we will compute 2 variables for each of causal, doc, and attn_mask masks:
            # - which blocks are fully dense or valid, i.e., all (q, k) pairs are valid
            # - which blocks are sparse, i.e., some (q, k) pairs are valid, yet not all

            # Starting with causality:
            # block_indices is e.g. (8,), and inserting a length-1 dimension at the end,
            # via block_indices[:, None], we make it (8,1), which makes the operation:
            # block_indices[:, None] >= block_indices broadcast (8,) into (1,8)
            # to result in (8,1) op (1,8) → an (8,8) or (num_blocks_per_axis, num_blocks_per_axis) tensor;
            # to simplify, assuming seq_len 8, and block_size 2 (num_blocks_per_axis=4), we have:
            # [[True,  False, False, False],
            #  [True,  True,  False, False],
            #  [True,  True,  True,  False],
            #  [True,  True,  True,  True]]
            # with the tensor having True values for blocks at least partially (causally) valid
            causal_partially_valid_blocks_2d = block_indices[:, None] >= block_indices
            # on the other hand, if we are more strict and all True values are blocks fully (causally) valid
            # (which excludes the diagonal since some elements have q < k), we have:
            # [[False, False, False, False],
            #  [True,  False, False, False],
            #  [True,  True,  False, False],
            #  [True,  True,  True,  False]]
            causal_fully_valid_blocks_2d = block_indices[:, None] > block_indices
            # broadcast the 2D pattern across batch: (gpu_batch_size, num_blocks_per_axis, num_blocks_per_axis)
            causal_partially_valid_blocks = causal_partially_valid_blocks_2d.expand(gpu_batch_size, -1, -1)
            causal_fully_valid_blocks     = causal_fully_valid_blocks_2d.expand(gpu_batch_size, -1, -1)

            # Moving on to doc ids:
            # if at least one (q, k) pair in the block share the same doc, it is partially (doc-wise) valid
            # if all (q, k) pairs in the block share the same doc, it is fully (doc-wise) valid
            if self.use_doc_masking and not ignore_doc_mask and (document_ids is not None):
                # take each sequence in the batch independently
                # e.g., for one sequence in the batch:
                # document_ids[b] = [0,0, 0,1, 1,1, 2,2]
                # blocks [0,0] | [0,1] | [1,1] | [2,2]
                # document_ids has shape (gpu_batch_size, seq_len); here, (B, seq_len)
                # so for one sequence we would take seq_len, e.g.,
                # [0, 0, 0, 1, 1, 1, 2, 2]
                docs = document_ids.to(device=device)
                # and then the lowest and highest doc id in each block_size, e.g., if block size was 2:
                # [0, 0, 0, 1, 1, 1, 2, 2] → view → [[0, 0], [0, 1], [1, 1], [2, 2]]
                # the lowest doc id per block would be [0, 0, 1, 2], with size (num_blocks_per_axis,)
                docs_view = docs.view(gpu_batch_size, num_blocks_per_axis, block_size)
                docs_low  = docs_view[:, :,  0].contiguous()
                # and the highest doc id per block would be [0, 1, 1, 2], with size (num_blocks_per_axis,)
                docs_high = docs_view[:, :, -1].contiguous()
                # (num_blocks_per_axis, 1) op (num_blocks_per_axis,) → 
                # (num_blocks_per_axis, 1) op (num_blocks_per_axis,) left padded to (1, num_blocks_per_axis) → 
                # (num_blocks_per_axis, num_blocks_per_axis), i.e., for one sequence:
                # -------------------------------------------------------------------------
                # (docs_low[:, None] <= docs_high):
                # [[True,  True,  True,  True],  because 0 <= 0,  0 <= 1,  0 <= 1,   0 <= 2
                # [ True,  True,  True,  True],  because 0 <= 0,  0 <= 1,  0 <= 1,   0 <= 2
                # [ False, True,  True,  True],  because 1 !<= 0, 1 <= 1,  1 <= 1,   1 <= 2
                # [ False, False, False, True]], because 2 !<= 0, 2 !<= 1, 2 !<= 1,  2 <= 2
                # -------------------------------------------------------------------------
                # (docs_high[:, None] >= docs_low):
                # [[True, True, False, False],   because 0 >= 0,  0 >= 0,  0 !>= 1,  0 !>= 2
                #  [True, True, True,  False],   because 1 >= 0,  1 >= 0,  1 >= 1,   1 !>= 2
                #  [True, True, True,  False],   because 1 >= 0,  1 >= 0,  1 >= 1,   1 !>= 2
                #  [True, True, True,  True]]    because 2 >= 0,  2 >= 0,  2 >= 1,   2 >= 2
                # -------------------------------------------------------------------------
                # (docs_low[:, None] <= docs_high) & (docs_high[:, None] >= docs_low):
                # [[True,  True,  False, False], because block 0's doc ids at least partially overlap with 0 and 1
                #  [True,  True,  True,  False], because block 1's doc ids at least partially overlap with 0, 1, 2
                #  [False, True,  True,  False], because block 2's doc ids at least partially overlap with 1 and 2
                #  [False, False, False, True]]  because block 3's doc ids at least partially overlap with 3
                doc_partially_valid_blocks = (
                    (docs_low[:, :, None] <= docs_high[:, None, :]) &
                    (docs_high[:, :, None] >= docs_low[:, None, :])
                )
                # while for full (q,k) doc validity, we get:
                # -------------------------------------------------------------------------
                # (docs_low[:, None] == docs_high) & (docs_high[:, None] == docs_low)
                # [[True,  False, False, False],  because all block 0's (q,k) share the same doc id in block 0
                #  [False, False, False, False],  because all block 1's (q,k) share the same doc id in *no block*
                #  [False, False, True,  False],  because all block 2's (q,k) share the same doc id in block 2
                #  [False, False, False, True]]   because all block 3's (q,k) share the same doc id in block 3
                # recalling that:
                # block 0 [0, 0]
                # block 1 [0, 1]
                # block 2 [1, 1]
                # block 3 [2, 2]
                doc_fully_valid_blocks = (
                    (docs_low[:, :, None] == docs_high[:, None, :]) & 
                    (docs_high[:, :, None] == docs_low[:, None, :])
                )
            else:
                doc_partially_valid_blocks = torch.ones(
                    (gpu_batch_size, num_blocks_per_axis, num_blocks_per_axis), dtype=torch.bool, device=device
                )
                doc_fully_valid_blocks = doc_partially_valid_blocks

            # Finally, for padding -or other custom attn_mask-:
            if attn_mask is not None:
                # e.g., for one sequence in the batch: [True, True, True, True, True, False, False, False]
                valid = attn_mask.to(device=device, dtype=torch.bool)
                # viewing the valid tokens as [[True, True], [True, True], [True, False], [False, False]] per sequence
                valid_blocks = valid.view(gpu_batch_size, num_blocks_per_axis, block_size)
                # q any indicates whether there is any query valid in the block:
                # [True, True, True, False] per sequence
                q_any = valid_blocks.any(dim=2)
                # k any indicates whether there is any key valid in the block:
                # [True, True, True, False] per sequence
                k_any = valid_blocks.any(dim=2)
                # in this case, q_any == k_any
                # q all indicates whether all queries are valid in the block:
                # [True, True, False, False] per sequence
                q_all = valid_blocks.all(dim=2)
                # k all indicates whether all keys are valid in the block:
                # [True, True, False, False] per sequence
                k_all = valid_blocks.all(dim=2)
                # in this case, q_all == k_all
                # -------------------------------------------------------------------------
                # q_any[:, :, None] & k_any[:, None, :]:
                # [[True,  True,   True,  False],
                #  [True,  True,   True,  False],
                #  [True,  True,   True,  False],
                #  [False, False,  False, False]] per sequence
                attn_mask_blockmask_any = q_any[:, :, None] & k_any[:, None, :]
                # -------------------------------------------------------------------------
                # q_all[:, :, None] & k_all[:, None, :]:
                # [[True,  True,  False,  False],
                #  [True,  True,  False,  False],
                #  [False, False, False,  False],
                #  [False, False, False,  False]] per sequence
                attn_mask_blockmask_all = q_all[:, :, None] & k_all[:, None, :]
                # -------------------------------------------------------------------------
            else:
                attn_mask_blockmask_any = torch.ones(
                    (gpu_batch_size, num_blocks_per_axis, num_blocks_per_axis), dtype=torch.bool, device=device
                )
                attn_mask_blockmask_all = attn_mask_blockmask_any

            # combining the constraints:
            # blockmask_any → tiles that may have some valid (q,k) pairs and must run mask_mod pointwise, e.g.,
            # [[True,  False, False, False],
            #  [True,  True,  False, False],
            #  [False, True,  True,  False],
            #  [False, False, False, False]] per sequence
            blockmask_any = causal_partially_valid_blocks & doc_partially_valid_blocks & attn_mask_blockmask_any
            # blockmask_all → tiles that are guaranteed fully dense and can skip mask_mod, e.g.,
            # [[False, False, False, False],
            #  [False, False, False, False],
            #  [False, False, False, False],
            #  [False, False, False, False]] per sequence
            blockmask_all = causal_fully_valid_blocks & doc_fully_valid_blocks & attn_mask_blockmask_all

            def map_to_ordered(blockmask_full_or_partial):
                # map (num_blocks_per_axis, num_blocks_per_axis) into its ordered representation
                # and return blockmask and sort-indices, both as (B?, H?, num_query_blocks)
                num_valid_kv_blocks_per_q_block = blockmask_full_or_partial.sum(dim=-1, dtype=torch.int32)
                # descending=False argsort → False before True
                # stable=True argsort      → keep order of appearance, e.g., [True, True] → [0, 1] (and not [1, 0])
                # row 0 of blockmask_any & ~blockmask_all is [True,  False, False, False] →
                # argsort(False-first) gives indices [1, 2, 3, 0] → flipped is [0, 3, 2, 1] (True-first)
                # row 1 of blockmask_any & ~blockmask_all is [True,  True,  False, False]
                # argsort(False-first) gives indices [2, 3, 0, 1] → flipped is [1, 0, 3, 2] (True-first)
                # row 2 of blockmask_any & ~blockmask_all is [False, True,  True,  False]
                # argsort(False-first) gives indices [0, 3, 1, 2] → flipped is [2, 1, 3, 0] (True-first)
                # row 3 of blockmask_any & ~blockmask_all is [False, False, False, False]
                # argsort(False-first) gives indices [0, 1, 2, 3] → flipped is [3, 2, 1, 0] (True-first)
                fully_valid_kv_indices = blockmask_full_or_partial.argsort(dim=-1, descending=False, 
                                                                        stable=True).flip(-1).to(torch.int32)
                # BlockMask expects (B?, H?, num_query_blocks), so we insert two leading dims; 
                # and flip() may produce a non-contiguous view, so we call contiguous(), resulting in, e.g.,
                # -------------------------------------------------------------------------
                # (given) blockmask_any & ~blockmask_all:
                # -------------------------------------------------------------------------
                # [[True,  False, False, False],
                #  [True,  True,  False, False],
                #  [False, True,  True,  False],
                #  [False, False, False, False]] per sequence
                # -------------------------------------------------------------------------
                # num_only_partially_valid_kv_blocks:
                # -------------------------------------------------------------------------
                # [[[1, 2, 2, 0]]] because row 0: 1 valid block; rows 1-2: 2; row 3: 0 (per sequence)
                # -------------------------------------------------------------------------
                # only_partial_kv_indices and while other orders are possible,
                # this keeps elements closest to the diagonal,
                # e.g., if SWA and sliding_window_max_size_blocks=1,
                # index 0 is kept for the first row (because it is the first column and we keep only 1 block)
                # index 1 is kept for the second row (because it is the first column and we keep only 1 block)
                # index 2 is kept for the third row (because it is the first column and we keep only 1 block)
                # index 3 is kept for the fourth row (because it is the first column and we keep only 1 block)
                # NOTE:
                # Despite not being used in this fast path (the path enforces is_causal), it must be noted:
                # This order fails if not is_causal because it keeps last valid block(s), which if is_causal 
                # is correct because future blocks are False, keeping the last *starting from the diagonal*, 
                # but would take the last column(s) always if not is_causal)
                # -------------------------------------------------------------------------
                # [[[0,    3,     2,     1],
                #   [1,    0,     3,     2],
                #   [2,    1,     3,     0],
                #   [3,    2,     1,     0]]] per sequence
                return (num_valid_kv_blocks_per_q_block[:, None, :].contiguous(), 
                        fully_valid_kv_indices[:, None, :].contiguous())

            num_only_partially_valid_kv_blocks, only_partial_kv_indices = map_to_ordered(blockmask_any & ~blockmask_all)
            num_fully_valid_kv_blocks, fully_valid_kv_indices = map_to_ordered(blockmask_all)

            # SWA window clamping (skipped for full attention layers)
            if apply_swa:
                # when using block_size, window_size is not exact (e.g., 2 full blocks + 1 partial block attended to
                # -in general, for block_size = 128 and window_size = 384 (window_size_blocks = 3)
                # excluding doc and other attn masks, that means 128 + 128 + 64 attended to tokens (64 tokens short))-
                # use tensor math: (window_size / block_size).ceil()
                window_size_blocks = (window_size.float() / block_size).ceil().to(torch.int32)

                # is_causal → the diagonal block is partial
                # therefore, always exclude one full block per row to make room for the diagonal block (in other words,
                # start counting window_size from the diagonal to the left)
                clamped_num_fully_valid_kv_blocks = torch.clamp_max(num_fully_valid_kv_blocks, window_size_blocks - 1)
                clamped_num_only_partially_valid_kv_blocks = torch.clamp_max(num_only_partially_valid_kv_blocks,
                                                                            window_size_blocks - clamped_num_fully_valid_kv_blocks)
            else:
                clamped_num_fully_valid_kv_blocks = num_fully_valid_kv_blocks
                clamped_num_only_partially_valid_kv_blocks = num_only_partially_valid_kv_blocks
            
            # https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html
            return BlockMask.from_kv_blocks(
                # -------------------------------------------------------------------------
                # which KV tiles to visit for each query tile
                # -------------------------------------------------------------------------
                # kv_num_blocks (Tensor) – Number of kv_blocks in each Q_BLOCK_SIZE row tile
                clamped_num_only_partially_valid_kv_blocks,
                # kv_indices (Tensor) – Indices of key-value blocks in each Q_BLOCK_SIZE row tile
                only_partial_kv_indices,
                # -------------------------------------------------------------------------
                # subset of the above that are known to be fully dense 
                # (every (q,k) pair in the tile is valid)
                # -------------------------------------------------------------------------
                # full_kv_num_blocks (Optional[Tensor]) – Number of full kv_blocks 
                # in each Q_BLOCK_SIZE row tile
                clamped_num_fully_valid_kv_blocks,
                # full_kv_indices (Optional[Tensor]) – Indices of full key-value blocks 
                # in each Q_BLOCK_SIZE row tile
                fully_valid_kv_indices,
                # -------------------------------------------------------------------------
                # per-element boolean function (b,h,q_idx,k_idx) -> bool that's applied 
                # inside partial tiles to zero out disallowed (q,k) pairs
                # (required if is_causal due to partial diagonal tiles)
                # -------------------------------------------------------------------------
                # BLOCK_SIZE (Union[int, tuple[int, int]]) – Size of KV_BLOCK_SIZE x Q_BLOCK_SIZE tiles
                BLOCK_SIZE=block_size,
                # mask_mod (Optional[Callable]) – Function to modify the mask
                mask_mod=mask_mod,
            )
        # -- End of (modified) source: https://github.com/KellerJordan/modded-nanogpt/blob/master/records/071225_BosAlign/c1fd8a38-bb9f-45c4-8af0-d37f70c993f3.txt --

        # old path (for is_causal=False, validation, sampling, and hellaswag)
        return create_block_mask(
            mask_mod,
            B=gpu_batch_size,
            H=None,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            BLOCK_SIZE=block_size,
            _compile=self.training
        )

    def _normalize_attn_mask(
            self,
            attn_mask: Optional[Tensor],
            gpu_batch_size: int,
            seq_len: int,
            device: Union[str, torch.device],
            for_flex: bool,
            ) -> Optional[Tensor]:
        if attn_mask is None:
            return None
        if attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.bool()
        if attn_mask.dim() != 2:
            raise ValueError(f"attn_mask must be 2D [gpu_batch_size, seq_len], got shape {attn_mask.shape}")
        if attn_mask.size(0) != gpu_batch_size or attn_mask.size(1) != seq_len:
            raise ValueError(f"attn_mask shape mismatch: expected [{gpu_batch_size}, {seq_len}], got {list(attn_mask.shape)}")

        attn_mask = attn_mask.to(device=device, non_blocking=True)

        if for_flex:
            return attn_mask

        # [gpu_batch_size, 1, seq_len, 1]
        q_valid = attn_mask[:, None, :, None]
        # [gpu_batch_size, 1, 1, seq_len]
        k_valid = attn_mask[:, None, None, :]
        # [gpu_batch_size, 1, seq_len, seq_len]
        allow_mask_4d = q_valid & k_valid
        if self.is_causal:
            allow_causal = torch.ones((1, 1, seq_len, seq_len), dtype=torch.bool, device=device).tril()
            allow_mask_4d = allow_mask_4d & allow_causal
        return allow_mask_4d

    def forward(
            self,
            indices: Tensor,
            targets: Optional[Tensor] = None,
            attn_mask: Optional[Tensor] = None,
            ignore_doc_mask: bool = False,
            document_ids: Optional[Tensor] = None,
            kv_cache: Optional['KVCache'] = None,
            position_ids: Optional[Tensor] = None,
            ) -> Tuple[Tensor, Optional[Tensor]]:
        # ignore_doc_mask to avoid using it in:
        # val, sampling, hellaswag
        # even if used in train
        gpu_batch_size, seq_len = indices.size()
        te = self.transformer.wte(indices)

        # if RoPE or NoPE, rotate later in attention or do nothing, respectively
        if self.pos_encoding_type == "rope" or self.pos_encoding_type == "nope":
            x = te
        else:
            # position_ids provides per-sequence positions for KV cache generation;
            # without it, positions are 0..seq_len-1 (training/eval)
            if position_ids is not None:
                pe = self.transformer.wpe(position_ids)
            else:
                pe = self.transformer.wpe(torch.arange(0, seq_len, dtype=torch.long, device=indices.device))
            x = te + pe

        if kv_cache is not None:
            # attn_mask marks valid (non-padding) cache positions per sequence.
            # For prefill with is_causal, right-padding is naturally excluded (padding is in the future).
            sdpa_attn_mask_4d = None
            if attn_mask is not None and seq_len == 1:
                # decode: single query token attends to all valid KV positions
                # attn_mask is (gpu_batch_size, max_cache_len), True = valid, False = padding
                kv_len = int(kv_cache.pos.max().item()) + 1 # +1 because update writes before returning
                kv_valid = attn_mask[:, :kv_len]
                sdpa_attn_mask_4d = kv_valid[:, None, None, :] # (gpu_batch_size, 1, 1, kv_len)

            for i, block in enumerate(self.transformer.h):
                x = block(x, sdpa_attn_mask=sdpa_attn_mask_4d, kv_cache=kv_cache, layer_idx_for_kv_cache=i, position_ids=position_ids)
        else:
            # training/eval path
            attn_mask = self._normalize_attn_mask(
                attn_mask=attn_mask,
                gpu_batch_size=gpu_batch_size,
                seq_len=seq_len,
                device=indices.device,
                for_flex=self.use_flex_attention,
            )

            if self.use_flex_attention:
                swa_flex_attn_block_mask = self._build_flex_attn_block_mask(
                    attn_mask=attn_mask,
                    document_ids=document_ids,
                    gpu_batch_size=gpu_batch_size,
                    seq_len=seq_len,
                    device=indices.device,
                    ignore_doc_mask=ignore_doc_mask,
                    apply_swa=True,
                )
                full_flex_attn_block_mask = self._build_flex_attn_block_mask(
                    attn_mask=attn_mask,
                    document_ids=document_ids,
                    gpu_batch_size=gpu_batch_size,
                    seq_len=seq_len,
                    device=indices.device,
                    ignore_doc_mask=ignore_doc_mask,
                    apply_swa=False,
                ) if self.has_full_attention_layers else None
                sdpa_attn_mask = None
            else:
                swa_flex_attn_block_mask = None
                full_flex_attn_block_mask = None
                sdpa_attn_mask = attn_mask

            for block, is_full in zip(self.transformer.h, self.layer_is_full_attention):
                flex_mask = full_flex_attn_block_mask if is_full else swa_flex_attn_block_mask
                x = block(x, flex_attn_block_mask=flex_mask, sdpa_attn_mask=sdpa_attn_mask)
        # cast if no autocast and cast is requested (1D Norm can be fp32, x can be bf16)
        x_ln_f = x
        if not self.use_bf16_autocast and self.cast_1d_weights_params_and_scales_to_weight_dtype_if_no_autocast and x_ln_f.dtype != self.transformer.ln_f.weight.dtype:
            x_ln_f = x_ln_f.to(dtype=self.transformer.ln_f.weight.dtype)
        x = self.transformer.ln_f(x_ln_f)

        # [gpu_batch_size, seq_len, vocab_size]
        # cast if no autocast and cast is requested (x can be fp32, lm_head weights can be bf16)
        if not self.use_bf16_autocast and self.cast_1d_weights_params_and_scales_to_weight_dtype_if_no_autocast and x.dtype != self.lm_head.weight.dtype:
            x = x.to(dtype=self.lm_head.weight.dtype)
        logits = self.lm_head(x)

        if targets is not None:
            logits_for_loss = logits.float() if self.keep_fp32_loss and logits.dtype != torch.float32 else logits
            if self.data_uses_padding:
                loss_mask = (targets != self.pad_token_id)
                loss = F.cross_entropy(logits_for_loss.view(-1, logits.size(-1)), targets.view(-1), reduction='none')
                n_non_masked_tokens = loss_mask.sum()
                sum_non_masked_loss_tokens = (loss * loss_mask.view(-1)).sum()
                loss = sum_non_masked_loss_tokens / n_non_masked_tokens
            else:
                loss = F.cross_entropy(logits_for_loss.view(-1, logits.size(-1)), targets.view(-1), reduction='mean')
        else:
            loss = None

        return logits, loss
