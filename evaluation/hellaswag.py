import math
from typing import Any, List, Union

import torch
import torch.distributed as dist
from torch.nn import functional as F

################################################
#                   HellaSwag                  #
################################################
# ---------------------------------------------------------
# Normalizing token probabilities to account for sequence length and comparing continuations A-D has the issue of an 
# easy continuation achieving high scores because once the answer deviates, the continuation is easy to guess, e.g.,

# (extreme example)
# She is cooking dinner,
# A) but she would rather be dancing in LA with the bride's friends tonight -> hard to guess dancing, LA, bride's
# B) one two three four five six -> easy to guess continuation after one two
# Here, it would not matter B makes less sense than A, because once 'one two' are told to be the continuation
# to the model, it has no way to be surprised by 'three four five six'
# ---------------------------------------------------------

# ---------------------------------------------------------
# From Language Models are Few-Shot Learners
# GPT-3 achieves 78.1% accuracy in the one-shot setting and 79.3% accuracy in the few-shot setting, outperforming the
# 75.4% accuracy of a fine-tuned 1.5B parameter language model [ZHR+19] but still a fair amount lower than the overall
# SOTA of 85.6% achieved by the fine-tuned multi-task model ALUM.

# Zero-shot
# Small | Med  | Large | XL   | 2.7B  | 6.7B  | 13B  | 175B
# 33.7  | 43.6 |  51.0 | 54.7 |  62.8 |  67.4 | 70.9 | 78.9

# One-shot (we will evaluate one-shot, but with Answer: A, Answer: B, Answer: C, Answer: D token probabilities)
# Small | Med  | Large | XL   | 2.7B  | 6.7B  | 13B  | 175B
# 33.0  | 42.9 | 50.5  | 53.5 | 61.9  | 66.5  | 70.0 | 78.1

# ...
# ---------------------------------------------------------

# ---------------------------------------------------------
# The A, B, C, D answers are uniformly distributed:

# HellaSwag train split answer label distribution:
#   A: 9986 (25.02%)
#   B: 10031 (25.14%)
#   C: 9867 (24.73%)
#   D: 10021 (25.11%)

# HellaSwag validation split answer label distribution:
#   A: 2515 (25.04%)
#   B: 2485 (24.75%)
#   C: 2584 (25.73%)
#   D: 2458 (24.48%)

# which make even a model biased towards some token, e.g., A,
# as good as a random-guesser model
# ---------------------------------------------------------

# ---------------------------------------------------------
# The fact they are equi-probable allows us to evaluate performance by doing:

# batch sequence 1 passed to the model

# Sentence to pick the continuation of: A little girl in a room standing in front of some chairs is hitting a dora pinata. She hits it a few times and then its someone else's turn. an older girl
# Continuation options:
# A) with a pink apron including a brown jacket tries to stop her but she hits it again.
# B) comes into the room and gives her and the girl laughs while they continue to hit the pinata.
# C) is also there in the room playing and before long some other little girl is swinging her away and she hasn't seen anyone.
# D) hits it so hard that it doesn't break but the whole pinata falls down and the adults have to put it back up.
# Correct continuation: D)

# Sentence to pick the continuation of: She is cooking dinner,
# Continuation options:
# A) but she would rather be dancing in LA with the bride's friends tonight
# B) one two three four five six
# Correct continuation: A)

# batch sequence 2 passed to the model

# ... (one-shot example)

# Sentence to pick the continuation of: She is cooking dinner,
# Continuation options:
# A) but she would rather be dancing in LA with the bride's friends tonight
# B) one two three four five six
# Correct continuation: B)

# to evaluate the likelihood the model assigns to each continuation
# ---------------------------------------------------------

# ---------------------------------------------------------
# For efficiency, we could score the log-probability of:
# " a" (for A)
# " b" (for B)
# " c" (for C)
# " d" (for D)

# but we'd rely on these tokens existing on the tokenizer (could break if we change our tokenizer)
# plus why not add " a", " A", " first",  etc. -if existent-,, eventually deciding to pass 4 batch sequences
# with the continuations included, one per batch sequence, even if more
# wasteful compared to passing a single sequence per HellaSwag sentence
# ---------------------------------------------------------

# ---------------------------------------------------------
# For length-normalization (to avoid penalizing long sequences):
# In prob space, the product of probabilities needs to be raised to 1/len(tokens) -i.e., the geometric mean-,
# while in log-prob space the sum of log-probs needs to be divided by len(tokens),
# e.g., given seq_A with 1 token, seq_B with 3:

# P(seq_A)           = 0.5 (favored)                  | P(seq_B)           = 0.5 * 0.5 * 0.5 = 0.125
# norm(P(seq_A))     = (0.5) ^ 1/1 = 0.5 (fair)       | norm(P(seq_B))     = (0.5 * 0.5 * 0.5) ^ (1/3) = 0.5

# log-P(seq_A)       = −0.69314 (favored)             | log-P(seq_B)       = -2.07944
# norm(log-P(seq_A)) = −0.69314 / 1 = −0.69314 (fair) | norm(log-P(seq_B)) = (−0.69314 + −0.69314 + −0.69314) / 3 = −0.69314
# ---------------------------------------------------------

# ---------------------------------------------------------
# If using hard-targets (only one token is correct, all others are incorrect):

# average loss = (1/sequence length) * sum(-log(probability of correct token at each position))
# likelihood = product of probabilities
# normalized likelihood = product of probabilities ^ (1/sequence length)
# average log-likelihood = log ((product of probabilities) ^ (1/sequence length)) = (1/sequence length) * sum(log-probabilities)
# average negative log-likelihood (nll) = - (average log-likelihood)
# perplexity (ppl) = e ^ (average negative log-likelihood)
# log-ppl = average loss = average nll
# ppl = e ^ (log-ppl) = e ^ (average loss) = e ^ (nll)

# F.cross_entropy() calculates - F.log_softmax(), and then either it reduces to mean or we do so manually
# ---------------------------------------------------------

def evaluate_hellaswag_one_shot(
        hellaswag_examples_per_batch: int = 16,
        *,
        gpt_model: Any,
        raw_gpt_model: Any,
        tokenizer: Any,
        hellaswag_train_dataset: Any,
        local_hellaswag_val_dataset: Any,
        device: Union[str, torch.device],
        ctx: Any,
        ddp_rank: int,
        ddp_world_size: int,
        master_process: bool,
        log_buffer: List[str],
        ) -> float:
    hellaswag_examples_per_batch = max(1, hellaswag_examples_per_batch)
    max_gpu_sequences_per_batch = hellaswag_examples_per_batch * 4

    gpt_model.eval()
    total_correct_predictions = 0
    total_dataset_examples = len(local_hellaswag_val_dataset)
    processed_examples_counter = 0

    one_shot_example = hellaswag_train_dataset[0]
    one_shot_context = one_shot_example['ctx']
    one_shot_endings = one_shot_example['endings']
    one_shot_correct_label_char = chr(ord('A') + int(one_shot_example['label']))

    max_model_seq_len = raw_gpt_model.max_seq_len 

    with torch.inference_mode():
        num_example_batches = (total_dataset_examples + hellaswag_examples_per_batch - 1) // hellaswag_examples_per_batch

        for i in range(num_example_batches):
            batch_slice_start = i * hellaswag_examples_per_batch
            batch_slice_stop = min((i + 1) * hellaswag_examples_per_batch, total_dataset_examples)
            current_examples = local_hellaswag_val_dataset.select(range(batch_slice_start, batch_slice_stop))
            # if i == 0:
            #     for rank in range(ddp_world_size):
            #         dist.barrier()
            #         if ddp_rank == rank:
            #             print(f"[rank {ddp_rank}] first HellaSwag batch:")
            #             for j, ex in enumerate(current_examples):
            #                 print(f"[rank {ddp_rank}] example {j}:")
            #                 print(f"ctx: {ex['ctx']}")
            #                 print(f"endings:")
            #                 for k, ending in enumerate(ex['endings']):
            #                     print(f"\t{chr(ord('A') + k)}) {ending}")
            #                 print(f"label: {ex['label']}")
            #                 print("-" * 60)
            #         dist.barrier()

            all_seq_ids = []
            orig_example_indices = []
            prefix_lengths = []

            for ex_idx_in_batch, example in enumerate(current_examples):
                context = example['ctx']
                continuations = [example['endings'][j] for j in range(4)]
                
                common_prefix_text_base = (
                    f"Sentence to pick the continuation of: {one_shot_context}\n"
                    "Continuation options:\n"
                    f"A) {one_shot_endings[0]}\n"
                    f"B) {one_shot_endings[1]}\n"
                    f"C) {one_shot_endings[2]}\n"
                    f"D) {one_shot_endings[3]}\n"
                    f"Correct continuation: {one_shot_correct_label_char})\n\n"
                    f"Sentence to pick the continuation of: {context}\n"
                    "Continuation options:\n"
                    f"A) {continuations[0]}\n"
                    f"B) {continuations[1]}\n"
                    f"C) {continuations[2]}\n"
                    f"D) {continuations[3]}\n"
                    "Correct continuation: "
                )
                prefix_ids = tokenizer.encode(common_prefix_text_base)
                
                for j in range(4):
                    final_token_text = f"{chr(ord('A') + j)})"
                    final_token_ids = tokenizer.encode(final_token_text)
                    combined_seq_ids = prefix_ids + final_token_ids
                    prefix_lengths.append(len(prefix_ids))
                    
                    if len(combined_seq_ids) > max_model_seq_len:
                        raise ValueError(
                            f"sequence for HellaSwag example {batch_slice_start + ex_idx_in_batch} is too long ({len(combined_seq_ids)} > {max_model_seq_len})"
                        )
                    
                    all_seq_ids.append(combined_seq_ids)
                    orig_example_indices.append(ex_idx_in_batch)

            # allocate to the actual max length in this pass (rounded for kernels, capped to model max)
            max_len_in_batch = max(len(s) for s in all_seq_ids) if all_seq_ids else 1
            round_multiple = (raw_gpt_model.flex_block_size if raw_gpt_model.use_flex_attention else 8)
            alloc_len = min(max_model_seq_len, math.ceil(max_len_in_batch / round_multiple) * round_multiple)

            pre_allocated_input_ids = torch.full(
                (max_gpu_sequences_per_batch, alloc_len),
                raw_gpt_model.pad_token_id,
                dtype=torch.long,
                device=device
            )
            pre_allocated_attn_mask = torch.zeros(
                (max_gpu_sequences_per_batch, alloc_len),
                dtype=torch.bool,
                device=device
            )

            for idx, seq_ids in enumerate(all_seq_ids):
                pre_allocated_input_ids[idx, :len(seq_ids)] = torch.tensor(seq_ids, dtype=torch.long, device=device)
                pre_allocated_attn_mask[idx, :len(seq_ids)] = True

            with ctx:
                # NOTE: for decoding and right padding, tokens cannot attend to padding anyway
                # so we can skip passing an attn_mask
                eager_gpt_model = raw_gpt_model._orig_mod if hasattr(raw_gpt_model, "_orig_mod") else raw_gpt_model
                logits, _ = eager_gpt_model(pre_allocated_input_ids, attn_mask=None, ignore_doc_mask=True)
            
            # log softmax: softmax followed by log (to sum log-probs vs. multiply low-value probs)
            log_probs = F.log_softmax(logits, dim=-1)

            example_scores = [[] for _ in range(len(current_examples))]

            for seq_idx in range(len(all_seq_ids)):
                original_ex_idx = orig_example_indices[seq_idx]
                
                true_len = pre_allocated_attn_mask[seq_idx].sum().item()
                prefix_len = prefix_lengths[seq_idx]
                
                seq_log_prob = 0.0
                num_tokens = 0

                for token_idx in range(prefix_len, true_len):
                    target_token_id = pre_allocated_input_ids[seq_idx, token_idx].item()
                    seq_log_prob += log_probs[seq_idx, token_idx - 1, target_token_id].item()
                    num_tokens += 1

                normalized_log_prob = seq_log_prob / num_tokens if num_tokens > 0 else 0.0

                example_scores[original_ex_idx].append(normalized_log_prob)

            for ex_idx, scores in enumerate(example_scores):
                correct_label_idx = int(current_examples[ex_idx]['label'])
                predicted_label_idx = max(enumerate(scores), key=lambda x: x[1])[0]

                if predicted_label_idx == correct_label_idx:
                    total_correct_predictions += 1
                
                processed_examples_counter += 1

                if processed_examples_counter % 200 == 0:
                    for rank in range(ddp_world_size):
                        dist.barrier()
                        if ddp_rank == rank:
                            print(f"[rank {ddp_rank}] processed {processed_examples_counter} examples: acc: {total_correct_predictions / processed_examples_counter:.4f}")
                    dist.barrier()

    local_accuracy = total_correct_predictions / total_dataset_examples
    local_message = f"[rank {ddp_rank}] HellaSwag local acc: {total_correct_predictions} / {total_dataset_examples} = {local_accuracy:.4f}"
    print(local_message)
    all_messages = [None for _ in range(ddp_world_size)]
    dist.all_gather_object(all_messages, local_message)
    if master_process:
        log_buffer.extend(all_messages)
    dist.barrier()

    # convert local totals to tensors for sum reduction across ranks
    correct_tensor = torch.tensor(total_correct_predictions, dtype=torch.long, device=device)
    count_tensor = torch.tensor(total_dataset_examples, dtype=torch.long, device=device)
    dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)

    accuracy = correct_tensor.item() / count_tensor.item()
    gpt_model.train()
    return accuracy

def evaluate_hellaswag_standard(
        hellaswag_examples_per_batch: int = 16,
        *,
        gpt_model: Any,
        raw_gpt_model: Any,
        tokenizer: Any,
        hellaswag_train_dataset: Any,
        local_hellaswag_val_dataset: Any,
        device: Union[str, torch.device],
        ctx: Any,
        ddp_rank: int,
        ddp_world_size: int,
        master_process: bool,
        log_buffer: List[str],
        ) -> float:
    hellaswag_examples_per_batch = max(1, hellaswag_examples_per_batch)
    max_gpu_sequences_per_batch = hellaswag_examples_per_batch * 4

    gpt_model.eval()
    total_correct_predictions = 0
    total_dataset_examples = len(local_hellaswag_val_dataset)
    processed_examples_counter = 0

    one_shot_example = hellaswag_train_dataset[0]
    one_shot_context = one_shot_example['ctx']
    one_shot_endings = one_shot_example['endings']
    one_shot_correct_label_char = chr(ord('A') + int(one_shot_example['label']))

    max_model_seq_len = raw_gpt_model.max_seq_len 
    
    with torch.inference_mode():
        num_example_batches = (total_dataset_examples + hellaswag_examples_per_batch - 1) // hellaswag_examples_per_batch

        for i in range(num_example_batches):
            batch_slice_start = i * hellaswag_examples_per_batch
            batch_slice_stop = min((i + 1) * hellaswag_examples_per_batch, total_dataset_examples)
            current_examples = local_hellaswag_val_dataset.select(range(batch_slice_start, batch_slice_stop))

            all_seq_ids = []
            orig_example_indices = []
            prefix_lengths = []

            for ex_idx_in_batch, example in enumerate(current_examples):
                context = example['ctx']
                continuations = [example['endings'][j] for j in range(4)]
                
                common_context_text = (
                    f"Sentence to pick the continuation of: {one_shot_context}\n"
                    "Continuation options:\n"
                    f"A) {one_shot_endings[0]}\n"
                    f"B) {one_shot_endings[1]}\n"
                    f"C) {one_shot_endings[2]}\n"
                    f"D) {one_shot_endings[3]}\n"
                    f"Correct continuation: {one_shot_correct_label_char})\n\n"
                    f"Sentence to pick the continuation of: {context}\n"
                )
                
                context_ids = tokenizer.encode(common_context_text)
                
                for j in range(4):
                    continuation_text = f" {continuations[j]}"
                    continuation_ids = tokenizer.encode(continuation_text)
                    
                    combined_seq_ids = context_ids + continuation_ids
                    
                    if len(combined_seq_ids) > max_model_seq_len:
                        raise ValueError(
                            f"sequence for HellaSwag example {batch_slice_start + ex_idx_in_batch} is too long ({len(combined_seq_ids)} > {max_model_seq_len})"
                        )
                    
                    all_seq_ids.append(combined_seq_ids)
                    orig_example_indices.append(ex_idx_in_batch)
                    prefix_lengths.append(len(context_ids))

            # allocate to the actual max length in this pass (rounded for kernels, capped to model max)
            max_len_in_batch = max(len(s) for s in all_seq_ids) if all_seq_ids else 1
            round_multiple = (raw_gpt_model.flex_block_size if raw_gpt_model.use_flex_attention else 8)
            alloc_len = min(max_model_seq_len, math.ceil(max_len_in_batch / round_multiple) * round_multiple)

            pre_allocated_input_ids = torch.full(
                (max_gpu_sequences_per_batch, alloc_len),
                raw_gpt_model.pad_token_id,
                dtype=torch.long,
                device=device
            )
            pre_allocated_attn_mask = torch.zeros(
                (max_gpu_sequences_per_batch, alloc_len),
                dtype=torch.bool,
                device=device
            )

            for idx, seq_ids in enumerate(all_seq_ids):
                pre_allocated_input_ids[idx, :len(seq_ids)] = torch.tensor(seq_ids, dtype=torch.long, device=device)
                pre_allocated_attn_mask[idx, :len(seq_ids)] = True

            with ctx:
                # NOTE: for decoding and right padding, tokens cannot attend to padding anyway
                # so we can skip passing an attn_mask
                eager_gpt_model = raw_gpt_model._orig_mod if hasattr(raw_gpt_model, "_orig_mod") else raw_gpt_model
                logits, _ = eager_gpt_model(pre_allocated_input_ids, attn_mask=None, ignore_doc_mask=True)
            
            log_probs = F.log_softmax(logits, dim=-1)

            example_scores = [[] for _ in range(len(current_examples))]

            for seq_idx in range(len(all_seq_ids)):
                original_ex_idx = orig_example_indices[seq_idx]
                prefix_len = prefix_lengths[seq_idx]
                true_len = pre_allocated_attn_mask[seq_idx].sum().item()
                
                seq_log_prob = 0.0
                num_continuation_tokens = 0
                for token_idx in range(prefix_len, true_len):
                    target_token_id = pre_allocated_input_ids[seq_idx, token_idx].item()
                    seq_log_prob += log_probs[seq_idx, token_idx - 1, target_token_id].item()
                    num_continuation_tokens += 1
                
                normalized_log_prob = seq_log_prob / num_continuation_tokens

                example_scores[original_ex_idx].append(normalized_log_prob)

            for ex_idx, scores in enumerate(example_scores):
                correct_label_idx = int(current_examples[ex_idx]['label'])
                predicted_label_idx = max(enumerate(scores), key=lambda x: x[1])[0]

                if predicted_label_idx == correct_label_idx:
                    total_correct_predictions += 1
                
                processed_examples_counter += 1

                if processed_examples_counter % 200 == 0:
                    for rank in range(ddp_world_size):
                        dist.barrier()
                        if ddp_rank == rank:
                            print(f"[rank {ddp_rank}] processed {processed_examples_counter} examples: acc: {total_correct_predictions / processed_examples_counter:.4f}")
                    dist.barrier()

    local_accuracy = total_correct_predictions / total_dataset_examples
    local_message = f"[rank {ddp_rank}] HellaSwag local acc: {total_correct_predictions} / {total_dataset_examples} = {local_accuracy:.4f}"
    print(local_message)
    all_messages = [None for _ in range(ddp_world_size)]
    dist.all_gather_object(all_messages, local_message)
    if master_process:
        log_buffer.extend(all_messages)
    dist.barrier()

    correct_tensor = torch.tensor(total_correct_predictions, dtype=torch.long, device=device)
    count_tensor = torch.tensor(total_dataset_examples, dtype=torch.long, device=device)
    dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)

    accuracy = correct_tensor.item() / count_tensor.item()
    gpt_model.train()
    return accuracy
