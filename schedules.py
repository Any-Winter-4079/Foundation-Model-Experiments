import math
from typing import List, Optional

################################################
#                Learning Rate                 #
################################################
def get_token_based_adamw_lr(
        train_tokens_processed: int,
        lr_warmup_tokens: int,
        lr_warmup_and_cosine_tokens: int,
        total_decay_tokens: int,
        adamw_max_lr: float,
        adamw_hard_min_lr: float,
        ) -> float:
    if train_tokens_processed < lr_warmup_tokens:
        return (train_tokens_processed + 1) * adamw_max_lr / lr_warmup_tokens
    elif train_tokens_processed >= lr_warmup_and_cosine_tokens:
        return adamw_hard_min_lr
    else:
        # coeff starts at 1.0 and goes to 0.0
        coeff = 0.5 * (1.0 + math.cos(math.pi * (train_tokens_processed - lr_warmup_tokens) / total_decay_tokens))
        return adamw_hard_min_lr + coeff * (adamw_max_lr - adamw_hard_min_lr)

################################################
#                Schedule Helpers              #
################################################
def get_next_update_tokens(schedule_keys: List[int], idx: int, max_tokens: int) -> int:
    next_idx = (idx + 1) if schedule_keys[0] == 0 else idx
    return schedule_keys[next_idx] if next_idx < len(schedule_keys) else max_tokens + 1

################################################
#             Schedule Constraints             #
################################################
def enforce_flex_block_constraints(value: int, flex_block_size: int, enforce_even_blocks: bool) -> int:
    num_blocks = max(1, value // flex_block_size)
    if enforce_even_blocks:
        num_blocks = math.ceil(num_blocks / 2) * 2
    return num_blocks * flex_block_size

def enforce_min_max_constraints(raw_values: List[int], min_value: Optional[int], max_value: Optional[int]) -> List[int]:
    values = []
    for value in raw_values:
        if min_value is not None and value < min_value:
            value = min_value
        if max_value is not None and value > max_value:
            value = max_value
        values.append(value if not values or value >= values[-1] else values[-1])
    return values

################################################
#            Pure Schedule Functions           #
################################################
def factor_schedule(
        start: int,
        factor: float,
        end: Optional[int] = None,
        count: Optional[int] = None,
        ) -> List[int]:
    if count == 1:
        return [start]
    if count is not None:
        return [int(start * (factor ** i)) for i in range(count)]
    schedule = []
    x = start
    while x <= end:
        schedule.append(x)
        x = int(x * factor)
    return schedule

def add_schedule(
        start: int,
        increment: int,
        end: Optional[int] = None,
        count: Optional[int] = None,
        ) -> List[int]:
    if count == 1:
        return [start]
    if count is not None:
        return [start + i * increment for i in range(count)]
    schedule = []
    x = start
    while x <= end:
        schedule.append(x)
        x += increment
    return schedule

def custom_schedule(values: List[int]) -> List[int]:
    # return the values. We make it a function, to match the other schedule functions
    return values

