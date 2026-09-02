import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Tuple

import tiktoken
import torch.nn as nn
from tiktoken.core import Encoding

from schedules import (
    add_schedule,
    custom_schedule,
    enforce_flex_block_constraints,
    enforce_min_max_constraints,
    get_token_based_adamw_lr,
)

@dataclass
class TrainingConfig:
    # steps and tokens
    total_tokens_per_step_train: int = 2**18 # for max batch size, 2**19 == 524_288 or ~0.5M tokens from Language Models are Few-Shot Learners
    # 32 * 2048 → 65_536 tokens
    # up to 4 gpus →
    #   65_536 * 1 → 65_536 tokens  → grad_accum_mini_steps = 4 (for 2**18 == 262_144 step tokens)
    #   65_536 * 2 → 131_072 tokens → grad_accum_mini_steps = 2 (for 2**18 == 262_144 step tokens)
    #   65_536 * 4 → 262_144 tokens → grad_accum_mini_steps = 1 (for 2**18 == 262_144 step tokens)
    # for 8 gpus →
    #   total_tokens_per_step needs to increase, or we need to reduce gpu batch size or seq len
    gpu_batch_size_train: int = 8
    gpu_batch_size_val: int = 8
    seq_len_train: int = 8192
    seq_len_val: int = 8192
    max_tokens: int = 5 * 10**9

    flex_block_size: int = 128
    enforce_even_blocks: bool = False # this *can* increase the window past its max to fit even block count
    # custom_schedule expects: "values", with a list of train_tokens_processed to increase the batch size at (e.g., after 10M, 25M, etc. tokens processed)
    # add_schedule expects:    "start", "increment", "end"|"count" to set the train_tokens_processed to increase the batch size at
    # factor_schedule expects: "start", "factor", "end"|"count", to set the train_tokens_processed to increase the batch size at
    batch_size_keys_schedule: Dict[str, Any] = field(default_factory=lambda: {
        "fn": custom_schedule,
        "kwargs": {
            "values": [0, 78_643_200], # in train_tokens_processed
            # "start": 2_457_600,
            # "factor": 2, # in train_tokens_processed
            # "count": 4,
        },
    })
    # custom_schedule expects: "values", with a list of batch sizes to increase to (e.g., 128, 256, etc. window tokens)
    # add_schedule expects:    "start", "increment", "end"|"count" to set the schedule of batch sizes to change to
    # factor_schedule expects: "start", "factor", "end"|"count", to set the schedule of batch sizes to change to
    batch_size_values_schedule: Dict[str, Any] = field(default_factory=lambda: {
        "fn": custom_schedule,
        "kwargs": {
            "values": [4, 8], # in batch size
        },
    })

    # token-based SWA window size schedule
    # custom_schedule expects: "values", with a list of train_tokens_processed to increase the window size at (e.g., after 10M, 25M, etc. tokens processed)
    # add_schedule expects:    "start", "increment", "end"|"count" to set the train_tokens_processed to increase the window size at
    # factor_schedule expects: "start", "factor", "end"|"count", to set the train_tokens_processed to increase the window size at
    swa_keys_schedule: Dict[str, Any] = field(default_factory=lambda: {
        "fn": add_schedule,
        "kwargs": {
            "start": 32_768_000, # in train_tokens_processed
            "increment": 200 * 262_144, # in train_tokens_processed
            "count": 11,
        },
    })
    # custom_schedule expects: "values", with a list of window sizes to increase to (e.g., 128, 256, etc. window tokens)
    # add_schedule expects:    "start", "increment", "end"|"count" to set the schedule of window sizes to change to
    # factor_schedule expects: "start", "factor", "end"|"count", to set the schedule of window sizes to change to
    swa_values_schedule: Dict[str, Any] = field(default_factory=lambda: {
        "fn": add_schedule,
        "kwargs": {
            "start": 128, # in window tokens
            "increment": 128, # in window tokens
            "count": 11,
        },
    })
    # filled by resolve(ddp_world_size)
    batch_size_schedule_keys: List[int] = field(init=False)
    batch_size_schedule_values: List[int] = field(init=False)
    swa_schedule_keys: List[int] = field(init=False)
    swa_schedule_values: List[int] = field(init=False)
    swa_initial_window_size: int = field(init=False)
    swa_final_window_size: int = field(init=False)
    # derived from the above
    max_seq_len: int = max(seq_len_train, seq_len_val)
    max_train_steps: int = max_tokens // total_tokens_per_step_train

    # tokenizer
    tokenizer: Encoding = tiktoken.get_encoding("gpt2")

    # optimizers
    adamw_betas: Tuple[float, float] = (0.9, 0.95)
    adamw_eps: float = 1e-8
    adamw_max_lr: float = 5e-3
    lr_warmup_tokens: int = 0
    lr_warmup_and_cosine_tokens: int = 600 * 10**6
    adamw_weight_decay: float = 0.01
    adamw_hard_min_lr: float = 7e-4
    optimizer_type: str = "muon" # "muon" (and adamw) or any other name for "adamw"
    muon_lr_scale: float = 0.15 # Muon usually likes a smaller LR than AdamW (e.g., 0.1x)
    muon_backend: str = "polarexpress" # "svd" | "newtonschulz5" | "polarexpress"
    muon_backend_steps: int = 5 # ~5-10; the higher, the slightly “tighter” orthogonalization (and slower)
    muon_momentum: float = 0.95
    muon_use_nesterov: bool = True
    # derived from the above
    total_decay_tokens: int = lr_warmup_and_cosine_tokens - lr_warmup_tokens
    # filled by resolve
    lr_schedule: List[Dict[str, Any]] = field(default_factory=list)

    # loading/checkpointing
    checkpoint_interval: int = 1000
    max_checkpoints_to_keep: int = 0
    resume_from_checkpoint: bool = False
    resume_checkpoint_path: str = "model_step_0000125_val_6.8745_train_6.9019.safetensors"
    resume_state_dict_path: str = "training_state_step_0000125.pt"
    resume_timestamp: str = "20250801_085237"
    hf_user: str = os.environ.get("hf_user")
    hf_token: str = os.environ.get("hf_token")
    # derived from the above
    timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S") if not resume_from_checkpoint else resume_timestamp
    checkpoint_dir: str = f"./checkpoints/{timestamp}"
    hub_repo_id: str = f"{hf_user}/nanogpt_{timestamp}"

    # logging (derived from the above)
    config_and_log_dir: str = f"./configs_and_logs/{timestamp}"
    log_filename: str = os.path.join(config_and_log_dir, f"log.txt")
    config_filename: str = os.path.join(config_and_log_dir, f"config.txt")

    # dataloader
    data_path: str = "./data/edu_fineweb10B"

    # data
    data_uses_padding: bool = False

    # validation
    val_target: float = 3.28
    val_tokens: int = 5 * 2**21 # 5 * 2**21 == 10_485_760
    shuffle_val_tokens: bool = False # shuffle or first 10_485_760 tokens of the FineWeb validation shard for the NanoGPT Speedrun
    val_interval: int = 25
    train_val_margin: float = -9.0 # to save compute, start running validation when training loss + train_val_margin <= val_target

    # sampling
    sample_interval: int = 25
    run_sampling: bool = False
    gpu_batch_size_sample: int = 2 # wave batch size for KV cache generation (sequences are split into waves of this size)
    sample_sequences: List[str] = field(default_factory=lambda:[
        "The universe has always been",
        "Who am I? I am a language model",
        "Artificial General Intelligence is",
        "Artificial General Intelligence is not",
        "2+2 is",
        "The quick brown fox jumps",
        "Earth is",
        "Could you tell me what time it is?",
    ])

    # hellaswag
    hellaswag_interval: int = 50
    run_benchmarks: bool = False

    # seeding
    base_seed: int = 1337

    # kernel warmup
    kernel_warmup_train_steps: int = 2

    # torch compile
    # NOTE: workaround for RuntimeError: Error: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run is 
    # to disable cudagraphs in max-autotune when grad_accum_mini_steps > 1
    torch_compile_mode_no_grad_accum: str = "max-autotune" # "max-autotune" | "reduce-overhead" | "max-autotune-no-cudagraphs" | "default"
    torch_compile_mode_grad_accum: str = "max-autotune-no-cudagraphs" # "max-autotune" | "reduce-overhead" | "max-autotune-no-cudagraphs" | "default"
    torch_compile_fullgraph: bool = True
    torch_compile_dynamic: bool = False

    # torch._inductor.config.cpp_wrapper = True

    # precision
    # upon initialization, weights are fp32; we can cast them to bf16 with .bfloat16()
    # torch.autocast(..., dtype=torch.bfloat16) does not change weight datatypes; it
    # (momentarily) casts (for operations eligible to run in bf16) the operands to bf16, e.g.
    # - the input/previous activation
    # - the weight matrix
    # therefore the output being bf16
    # However, with autocast alone, fp32 weights are still read from memory using 4 bytes, and only some operations are eligible
    # set_float32_matmul_precision applies if operands are fp32
    # - highest -> perform matmul in fp32 (24 mantissa bits with 23 bits explicitly stored)
    # - high -> perform matmul either in tf32 or treat fp32 as the sum of 2 bf16 numbers
    # - medium -> perform matmul in bf16 (8 mantissa bits with 7 bits explicitly stored)
    # Source: https://docs.pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    keep_fp32_loss: bool = True
    use_all_bf16_and_null_ctx: bool = False # all (1d/2d) weights, nn.Parameter, scales to bf16, and no autocast (nullcontext)
    use_bf16_weights_params_or_scales: bool = False # some (1d/2d) weights, nn.Parameter, or scales to bf16 (can be overwritten/ignored if use_all_bf16_and_null_ctx=True -which casts all-)
    bf16_weights_params_and_scales: Dict[str, Any] = field(default_factory=lambda: {
        "nn.Linear": nn.Linear,
        "nn.Embedding": nn.Embedding,
        #"nn.Parameter": nn.Parameter
    }) # which (1d/2d) weights, (learnable) nn.Parameter, or (nn.Parameter with requires_grad=False as) scale to cast to bf16 if use_bf16_weights_params_or_scales=True (can be overwritten to {"all": object} if use_all_bf16_and_null_ctx=True -which casts all-)
    float32_matmul_precision: str = "high" # "highest" | "high" | "medium" (ignored if no fp32 operands or if they are eligible by autocast to run in bf16)
    use_bf16_autocast: bool = True # autocast eligible inputs/weights (momentarily) to bf16 (can be overwritten/ignored if use_all_bf16_and_null_ctx=True -which uses null context-)
    keep_1d_weights_params_and_scales_in_fp32: bool = False # keep 1d weights, nn.Parameter, and scales in fp32 overwriting bf16_weights_params_and_scales={...} or use_all_bf16_and_null_ctx=True
    cast_1d_weights_params_and_scales_to_weight_dtype_if_no_autocast: bool = False # match 1d weights, learnable nn.Parameter, and scales data type (e.g., fp32) to that of other operands (e.g., bf16 for 2d weights), applicable when use_bf16_autocast=False

    # filled by resolve(ddp_world_size)
    # steps and tokens
    total_tokens_per_mini_step_train: int = field(init=False)
    grad_accum_mini_steps: int = field(init=False)
    total_tokens_per_step_val: int = field(init=False)
    val_steps: int = field(init=False)
    torch_compile_mode: str = field(init=False)

    def resolve(self, ddp_world_size: int) -> None:
        # bf16
        if self.use_all_bf16_and_null_ctx:
            self.use_bf16_autocast = False
            self.bf16_weights_params_and_scales = {"all": object}

        # Muon Polar Express
        if self.optimizer_type == "muon" and self.muon_backend == "polarexpress":
            assert self.muon_backend_steps == 5, (
                "Using Polar Express currently requires muon_backend_steps=5 steps due to "
                f"pre-computed coefficients (got {self.muon_backend_steps} steps)"
            )

        # steps and tokens
        self.total_tokens_per_mini_step_train = ddp_world_size * self.gpu_batch_size_train * self.seq_len_train
        self.grad_accum_mini_steps = self.total_tokens_per_step_train // self.total_tokens_per_mini_step_train
        assert self.total_tokens_per_step_train % self.total_tokens_per_mini_step_train == 0, \
            "total_tokens_per_step must be a multiple of tokens per mini-step"
        self.total_tokens_per_step_val = ddp_world_size * self.gpu_batch_size_val * self.seq_len_val
        self.val_steps = math.ceil(self.val_tokens / self.total_tokens_per_step_val)
        self.total_decay_tokens = self.lr_warmup_and_cosine_tokens - self.lr_warmup_tokens
        self.torch_compile_mode = (
            self.torch_compile_mode_no_grad_accum
            if self.grad_accum_mini_steps == 1
            else self.torch_compile_mode_grad_accum
        )

        # batch size schedule
        keys_schedule = self.batch_size_keys_schedule
        values_schedule = self.batch_size_values_schedule
        keys_schedule["max"] = self.max_tokens
        raw_keys = keys_schedule["fn"](**keys_schedule["kwargs"])
        self.batch_size_schedule_keys = enforce_min_max_constraints(
            raw_values=raw_keys,
            min_value=keys_schedule.get("min"),
            max_value=keys_schedule.get("max"),
        )
        raw_values = values_schedule["fn"](**values_schedule["kwargs"])
        self.batch_size_schedule_values = enforce_min_max_constraints(
            raw_values=raw_values,
            min_value=values_schedule.get("min"),
            max_value=values_schedule.get("max"),
        )

        # SWA window schedule
        keys_schedule = self.swa_keys_schedule
        values_schedule = self.swa_values_schedule
        keys_schedule["max"] = self.max_tokens
        raw_swa_keys = keys_schedule["fn"](**keys_schedule["kwargs"])
        self.swa_schedule_keys = enforce_min_max_constraints(
            raw_values=raw_swa_keys,
            min_value=keys_schedule.get("min"),
            max_value=keys_schedule.get("max"),
        )
        raw_swa_values = values_schedule["fn"](**values_schedule["kwargs"])
        swa_values = enforce_min_max_constraints(
            raw_values=raw_swa_values,
            min_value=values_schedule.get("min"),
            max_value=values_schedule.get("max"),
        )
        self.swa_schedule_values = [
            enforce_flex_block_constraints(value, self.flex_block_size, self.enforce_even_blocks)
            for value in swa_values
        ]
        self.swa_initial_window_size = self.swa_schedule_values[0]
        self.swa_final_window_size = self.swa_schedule_values[-1]

        # lr schedule (token ranges map to functions; functions take tokens since range start)
        # get_token_based_adamw_lr expects: lr_warmup_tokens, lr_warmup_and_cosine_tokens, total_decay_tokens,
        # adamw_max_lr, adamw_hard_min_lr
        self.lr_schedule = [
            {
                "start": 0,
                "end": self.max_tokens,
                "fn": get_token_based_adamw_lr,
                "kwargs": {
                    "lr_warmup_tokens": self.lr_warmup_tokens,
                    "lr_warmup_and_cosine_tokens": self.lr_warmup_and_cosine_tokens,
                    "total_decay_tokens": self.total_decay_tokens,
                    "adamw_max_lr": self.adamw_max_lr,
                    "adamw_hard_min_lr": self.adamw_hard_min_lr,
                },
            },
        ]

