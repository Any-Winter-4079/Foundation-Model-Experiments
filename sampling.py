import math
from typing import Any, List, Optional, Sequence, Union

import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn import functional as F

from attention.kv_cache import KVCache

################################################
#                   Sampling                   #
################################################
def get_sample_token_count(step: int, base: int = 5, step_interval: int = 1000, max_tokens: int = 50) -> int:
    return min(base + (step // step_interval) * base, max_tokens)

def sample_next_token(
        last_logits: Tensor,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
        master_process: bool,
        log_buffer: List[str],
        ) -> Tensor:
    # sample the next token per sequence from last_logits (gpu_batch_size, vocab_size)
    # returns next_token_ids (gpu_batch_size, 1)
    if temperature == 0.0:
        # then if temperature is 0, we cannot divide by 0, so take the max logit from vocab_size
        next_token_ids = torch.argmax(last_logits, dim=-1, keepdim=True)
    else:
        # else, divide by the temperature
        temp_adjusted_logits = last_logits / temperature

        # apply top_k filtering
        if top_k is not None and top_k > 0:
            # topk returns the k largest elements of the given input tensor along a given dimension
            # resulting in gpu_batch_size, vocab_size
            sorted_values, _ = torch.topk(temp_adjusted_logits, min(top_k, temp_adjusted_logits.size(-1)), dim=-1, largest=True, sorted=True)
            # get the k-th largest logit of each sequence (i.e., last in sorted_values)
            # and add a new dimension of size 1 at the end to give size (gpu_batch_size, 1)
            sequences_k_th_logit = sorted_values[:, -1].unsqueeze(1)
            # mask out everything less than the top_k logit
            temp_adjusted_logits[temp_adjusted_logits < sequences_k_th_logit] = -float('Inf')

        # apply top_p (nucleus) filtering
        if top_p is not None and top_p < 1.0:
            # apply softmax to get probabilities for the continuation to the last non-pad token
            # resulting in (still) gpu_batch_size, vocab_size
            probs = F.softmax(temp_adjusted_logits, dim=-1)
            # sort probabilities in descending order
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)

            # obtain the cumulative sums of the probabilities sorted in descending order
            # e.g., cumulative probs of [0.3, 0.6, 0.8, 1.0] for [0.3, 0.3, 0.2, 0.2]
            # with size gpu_batch_size, vocab_size
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            if master_process:
                message = "first 2 sequences cumulative probs:"
                print(message)
                log_buffer.append(message)
                for i in range(min(2, cumulative_probs.size(0))):
                    values = cumulative_probs[i, :10]
                    message = f"\tseq {i}: {values.tolist()}"
                    print(message)
                    log_buffer.append(message)
            # obtain the indices of logits to remove,
            # initially excluding the logit that makes the cumulative sum match top_p,
            # as we then shift to the right one position
            # cumulative_probs [0.3, 0.6, 0.8, 1.0], top_p 0.75 would thus give (for a single sequence):
            # [False, False, True, True] which shifted to the right is [False, False, False, True]
            # while cumulative_probs [0.3, 0.6, 0.8, 1.0], top_p 0.8:
            # [False, False, True, True] which shifted to the right is [False, False, False, True],
            # being in both instances the smallest possible set that has at least top_p cumulative probability
            sorted_indices_to_remove = cumulative_probs >= top_p
            # shift the indices to the right one position
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            # replacing the last index value, now in the first position, to False
            sorted_indices_to_remove[..., 0] = False
            mask = sorted_indices_to_remove.to(torch.bool)

            scatter_mask = torch.zeros_like(temp_adjusted_logits, dtype=torch.bool)
            scatter_mask = scatter_mask.scatter(dim=-1, index=sorted_indices, src=mask)
            if master_process:
                for i in range(min(2, scatter_mask.size(0))):
                    masked_count = scatter_mask[i].sum().item()
                    total_count = scatter_mask.size(1)
                    #print(f"sequence {i} mask: {masked_count}/{total_count} logits masked")
                    if masked_count == total_count:
                        message = f"all logits masked for sequence {i}!"
                        print(message)
                        log_buffer.append(message)
            temp_adjusted_logits = temp_adjusted_logits.masked_fill(scatter_mask, -float('inf'))

        probs = F.softmax(temp_adjusted_logits, dim=-1)
        next_token_ids = torch.multinomial(probs, num_samples=1)

    return next_token_ids

def sample(
        sample_sequences: Sequence[str],
        max_new_tokens: int = 5,
        temperature: float = 0.7,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        *,
        gpt_model: Any,
        raw_gpt_model: Any,
        gpt_config: Any,
        training_config: Any,
        tokenizer: Any,
        device: Union[str, torch.device],
        ctx: Any,
        ddp_rank: int,
        ddp_world_size: int,
        master_process: bool,
        log_buffer: List[str],
        ) -> None:
    gpt_model.eval()
    eager_gpt_model = raw_gpt_model._orig_mod if hasattr(raw_gpt_model, "_orig_mod") else raw_gpt_model
    with torch.inference_mode():
        # convert all prompts to ids,
        # e.g., 'AGI is' to [4760, 40, 318]
        # 'AGI is not' to [4760, 40, 318, 407]
        # and truncate them leaving max_new_tokens for generation
        max_allowed_input_len = raw_gpt_model.max_seq_len - max_new_tokens
        initial_input_ids_list = [
            tokenizer.encode(sequence)[:max_allowed_input_len]
            for sequence in sample_sequences
        ]

        if gpt_config.use_kv_cache:
            # Wave-batched KV cache generation:
            # sequences are split into waves of gpu_batch_size_sample.
            # each wave: right-pad prompts to equal length → batched prefill (is_causal handles padding) →
            # set kv_cache.pos to actual prompt lengths → batched decode one token at a time.
            # the generated token is inserted into the sequence at its real position, and the mask
            # for the next decode step is derived from the sequence (not padding = valid).
            # position_ids provides per-sequence positions for correct RoPE / absolute PE.
            local_messages = []
            seq_idx_global = 0

            for wave_start in range(0, len(initial_input_ids_list), training_config.gpu_batch_size_sample):
                wave_ids_list = initial_input_ids_list[wave_start:wave_start + training_config.gpu_batch_size_sample]
                wave_batch_size = len(wave_ids_list)
                wave_prompt_lengths = [len(ids) for ids in wave_ids_list]
                max_prompt_len = max(wave_prompt_lengths)
                # round up to a multiple of 8 for tensor core efficiency on prefill matmuls
                max_total_len = math.ceil((max_prompt_len + max_new_tokens) / 8) * 8

                # right-pad all prompts in this wave to max_total_len
                # (right-padding is used so real tokens start at position 0, which works with
                # all position encoding types: absolute, RoPE, and NoPE)
                padded_sequences = torch.full(
                    (wave_batch_size, max_total_len),
                    gpt_config.pad_token_id,
                    dtype=torch.long,
                    device=device
                )
                for i, ids in enumerate(wave_ids_list):
                    padded_sequences[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)

                # create KV cache for this wave
                kv_cache = KVCache(
                    gpt_config=gpt_config,
                    batch_size=wave_batch_size,
                    max_seq_len=max_total_len,
                    device=device,
                    dtype=torch.bfloat16,
                )

                # position_ids for prefill: 0, 1, 2, ... for all sequences
                # (right-padded, so real tokens naturally start at position 0;
                # padding tokens get arbitrary positions but is_causal masks them out)
                prefill_position_ids = torch.arange(max_prompt_len, dtype=torch.long, device=device).unsqueeze(0).expand(wave_batch_size, -1)

                # prefill: batched forward pass over all prompts in the wave.
                # with right-padding and is_causal, real tokens never attend to padding
                # (padding is to the right, causally masked out), so no explicit attn_mask is needed.
                prefill_tokens = padded_sequences[:, :max_prompt_len]
                with ctx:
                    logits, _ = eager_gpt_model(prefill_tokens, kv_cache=kv_cache, position_ids=prefill_position_ids)

                # after prefill, set cache positions to actual prompt lengths (not padded length),
                # so the first decode token scatter-writes at the correct position per sequence,
                # overwriting padding K/V
                kv_cache.pos = torch.tensor(wave_prompt_lengths, dtype=torch.long, device=device)

                # take logits at each sequence's last real token (not last padded position)
                last_real_indices = kv_cache.pos - 1
                last_logits = logits[torch.arange(wave_batch_size, device=device), last_real_indices, :]
                next_token_ids = sample_next_token(
                    last_logits, temperature, top_k, top_p, master_process, log_buffer
                )

                # insert generated token into each sequence at its real position
                padded_sequences[torch.arange(wave_batch_size, device=device), kv_cache.pos] = next_token_ids[:, 0]
                # NOTE: do NOT increment kv_cache.pos here — the decode forward pass needs to
                # scatter-write this token's K/V at kv_cache.pos (the position we just inserted at).
                # pos is incremented AFTER the forward pass writes to the cache.

                # decode: generate one token at a time, all sequences in lockstep
                for _ in range(max_new_tokens - 1):
                    # position_ids for decode: the position of the token being processed
                    # (kv_cache.pos points to where this token was inserted, and where its K/V will be scatter-written)
                    decode_position_ids = kv_cache.pos.unsqueeze(1)  # (wave_batch_size, 1)

                    # attention mask: derived from the sequence — not padding = valid
                    # (gpu_batch_size, max_total_len) → True where real tokens exist
                    attn_mask = padded_sequences != gpt_config.pad_token_id

                    with ctx:
                        logits, _ = eager_gpt_model(next_token_ids, kv_cache=kv_cache, attn_mask=attn_mask, position_ids=decode_position_ids)

                    # K/V was scatter-written at kv_cache.pos by update(); now advance
                    kv_cache.pos += 1

                    next_token_ids = sample_next_token(
                        logits[:, 0, :], temperature, top_k, top_p, master_process, log_buffer
                    )

                    # insert generated token at each sequence's next position
                    padded_sequences[torch.arange(wave_batch_size, device=device), kv_cache.pos] = next_token_ids[:, 0]

                # decode and collect messages for this wave
                # kv_cache.pos points to the last token whose K/V was written; +1 for total length
                for i in range(wave_batch_size):
                    seq_len_i = int(kv_cache.pos[i].item()) + 1
                    decoded = tokenizer.decode([
                        token for token in padded_sequences[i, :seq_len_i].tolist()
                        if token < eager_gpt_model.pad_token_id])
                    message = f"[rank {ddp_rank}] seq {seq_idx_global} >>> {decoded}"
                    print(message)
                    local_messages.append(message)
                    seq_idx_global += 1

        else:
            # no KV cache path: original batched generation

            # find the max input length (in ids) in this batch
            max_input_len = max(len(ids) for ids in initial_input_ids_list)

            # get a length cap to pre-allocate the tensor while not needing to go to max_seq_len
            alloc_len = max_input_len + max_new_tokens

            # rounding up to a nice multiple for better tensor cores / GPU efficiency
            round_multiple = (raw_gpt_model.flex_block_size if raw_gpt_model.use_flex_attention else 8)
            alloc_len = math.ceil(alloc_len / round_multiple) * round_multiple
            alloc_len = min(alloc_len, raw_gpt_model.max_seq_len)

            # pre-allocate tensor of size len(sample_sequences), alloc_len and fill with padding
            # to significanly boost performance (versus a new tensor size every generation)!
            generated_sequences = torch.full(
                (len(sample_sequences), alloc_len),
                raw_gpt_model.pad_token_id,
                dtype=torch.long,
                device=device
            )

            # then replace the first padding tokens of each pre-allocated sequence with their original tokens
            for i, seq_ids in enumerate(initial_input_ids_list):
                generated_sequences[i, :len(seq_ids)] = torch.tensor(seq_ids, dtype=torch.long, device=device)

            # track actual (non-padding) sequence lengths
            actual_sequence_lengths = torch.tensor([len(seq) for seq in initial_input_ids_list], dtype=torch.long, device=device)

            for _ in range(max_new_tokens):
                # then, for each new token to generate, update the mask by effectively comparing if 0, ..., alloc_len < the non-padded length for each sequence
                # using unsqueeze(0) to add a new dimension of size 1 at the beginning to give size (1, alloc_len)
                # and unsqueeze(1) to add a new dimension of size 1 at the end to give size (len(sample_sequences), 1)
                # attn_mask = torch.arange(alloc_len, device=device).unsqueeze(0) < actual_sequence_lengths.unsqueeze(1)

                with ctx:
                    # and predict, resulting in (len(sample_sequences), seq_len, vocab_size)
                    # NOTE: for decoding and right padding, tokens cannot attend to padding anyway
                    # so we can skip passing an attn_mask
                    logits, _ = eager_gpt_model(generated_sequences, attn_mask=None, ignore_doc_mask=True, document_ids=None)

                # of which we take the vocab_size values for each sequence's continuation to the last non-pad token,
                # resulting in len(sample_sequences), vocab_size
                last_logits = logits[torch.arange(len(sample_sequences), device=device), actual_sequence_lengths - 1, :]

                next_token_ids = sample_next_token(
                    last_logits, temperature, top_k, top_p, master_process, log_buffer
                )

                # replace the 'actual_sequence_length' position of each sequence with the selected token
                generated_sequences[torch.arange(len(sample_sequences), device=device), actual_sequence_lengths] = next_token_ids[:, 0]
                # increase the actual, non-padded sequence endings
                actual_sequence_lengths += 1

            # and decode, ignoring above 50256
            local_messages = []
            for i in range(len(sample_sequences)):
                decoded = tokenizer.decode([
                    token for token in generated_sequences[i, :actual_sequence_lengths[i]].tolist()
                    if token < raw_gpt_model.pad_token_id])
                message = f"[rank {ddp_rank}] seq {i} >>> {decoded}"
                print(message)
                local_messages.append(message)

        all_messages = [None for _ in range(ddp_world_size)]
        dist.all_gather_object(all_messages, local_messages)

        if master_process:
            for rank_messages in all_messages:
                log_buffer.extend(rank_messages)
        dist.barrier()
    gpt_model.train()
