import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import contextlib
import copy
import math
import sys
import time
from dataclasses import fields
from typing import Any, Dict

import torch
import torch.distributed as dist
import torch.nn as nn
import triton
from datasets import load_dataset
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from attention.self_attention import qk_scale_debug_string
from checkpointing import load_checkpoint, save_checkpoint
from config.model import GPTConfig
from config.training import TrainingConfig
from data_loader import DataLoader
from evaluation.hellaswag import evaluate_hellaswag_standard
from models.gpt import GPT
from sampling import get_sample_token_count, sample
from schedules import get_next_update_tokens

# pip install tiktoken huggingface_hub safetensors

# torchrun --standalone --nproc_per_node=4 train.py
# NOTE:
# 1) torchrun sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
# 2) model wrappers:
# gpt_model = DDP(compiled GPT), where DDP handles cross-gpu gradient sync/all-reduce
# raw_gpt_model = gpt_model.module, which removes only the DDP wrapper (still compiled)
# raw_gpt_model._orig_mod removes the compile wrapper too and runs eager GPT

################################################
# 4 x H100 SXM | 56 vCPU 503 GB RAM | $10.86/h #
################################################

################################################
#      Initialization & Non-Model Config       #
################################################
ddp_rank = int(os.environ['RANK'])
ddp_local_rank = int(os.environ['LOCAL_RANK'])
ddp_world_size = int(os.environ['WORLD_SIZE'])
device_type = "cuda"
device = f'cuda:{ddp_local_rank}'
torch.cuda.set_device(device)
init_process_group(backend='nccl', device_id=ddp_local_rank)
master_process = ddp_rank == 0

training_config = TrainingConfig()
training_config.resolve(ddp_world_size)

# steps and tokens
gpu_batch_size_train = training_config.gpu_batch_size_train
gpu_batch_size_val = training_config.gpu_batch_size_val
seq_len_train = training_config.seq_len_train
seq_len_val = training_config.seq_len_val
max_tokens = training_config.max_tokens
batch_size_schedule_keys = training_config.batch_size_schedule_keys
batch_size_schedule_values = training_config.batch_size_schedule_values
# derived from the above
max_seq_len = training_config.max_seq_len
max_train_steps = training_config.max_train_steps
# derived from the above and world size
total_tokens_per_mini_step_train = training_config.total_tokens_per_mini_step_train
grad_accum_mini_steps = training_config.grad_accum_mini_steps
total_tokens_per_step_val = training_config.total_tokens_per_step_val
val_steps = training_config.val_steps

# tokenizer
tokenizer = training_config.tokenizer

# optimizers
adamw_betas = training_config.adamw_betas
adamw_eps = training_config.adamw_eps
adamw_max_lr = training_config.adamw_max_lr
lr_schedule = training_config.lr_schedule
lr_warmup_tokens = training_config.lr_warmup_tokens
lr_warmup_and_cosine_tokens = training_config.lr_warmup_and_cosine_tokens
adamw_weight_decay = training_config.adamw_weight_decay
adamw_hard_min_lr = training_config.adamw_hard_min_lr
# derived from the above
total_decay_tokens = training_config.total_decay_tokens

# checkpointing
max_checkpoints_to_keep = training_config.max_checkpoints_to_keep
resume_from_checkpoint = training_config.resume_from_checkpoint
resume_checkpoint_path = training_config.resume_checkpoint_path
resume_state_dict_path = training_config.resume_state_dict_path
hf_token = training_config.hf_token
# derived from the above
timestamp = training_config.timestamp
checkpoint_dir = training_config.checkpoint_dir
hub_repo_id = training_config.hub_repo_id

# logging (derived from the above)
config_and_log_dir = training_config.config_and_log_dir
log_filename = training_config.log_filename
config_filename = training_config.config_filename

# dataloader
data_path = training_config.data_path

# validation
val_target = training_config.val_target
val_tokens = training_config.val_tokens
shuffle_val_tokens = training_config.shuffle_val_tokens
val_interval = training_config.val_interval
train_val_margin = training_config.train_val_margin

# sampling
sample_interval = training_config.sample_interval
sample_sequences = training_config.sample_sequences

# hellaswag
hellaswag_interval = training_config.hellaswag_interval

# seeding
base_seed = training_config.base_seed

# kernel warmup
kernel_warmup_train_steps = training_config.kernel_warmup_train_steps

# buffer to 'write to disk' only at checkpointing steps, to avoid e.g. stopping training at step 233,
# restoring a model from step 200 -last val step that improves val loss-  and having training logs up
# to 233 then continuing from 201 (e.g., 232, 233, 201, 202, ...).
# This ensures training losses are stored only after a checkpointing step happens.
log_buffer = []

if master_process:
    log_buffer.append(f"python: {sys.version}")
    log_buffer.append(f"torch: {torch.__version__}")
    log_buffer.append(f"torch.cuda: {torch.version.cuda}")
    log_buffer.append(f"triton: {triton.__version__}")
    log_buffer.append("=" * 100)
    with open(sys.argv[0], "r") as f:
        log_buffer.append(f.read())
    log_buffer.append("=" * 100)

if master_process:
    message = f"per-gpu gradient accumulation mini-steps: {grad_accum_mini_steps}"
    print(message)
    log_buffer.append(message)

if master_process:
    message = f"max train steps: {max_train_steps:,}"
    print(message)
    log_buffer.append(message)

torch.set_float32_matmul_precision(training_config.float32_matmul_precision)

if master_process:
    print(f"{val_tokens:,} val tokens to be consumed in {val_steps:,} steps ({total_tokens_per_step_val:,} tokens per val step)")

# to save compute, start running validation when training loss + train_val_margin <= val_target
allow_val = False

if master_process:
    os.makedirs(config_and_log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

start_step = 0
train_tokens_processed = 0
total_train_t = 0.0
total_val_t = 0.0
total_sample_t = 0.0
total_hellaswag_t = 0.0
total_t = 0.0
best_val_loss = float('inf')

epoch = 0
current_shard_idx = 0
grad_accum_mini_steps_per_shard_counter = 0

seed = base_seed + ddp_rank
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

################################################
#                     BF16                     #
################################################
def convert_to_bf16(
        gpt_model: GPT,
        bf16_weights_params_and_scales: Dict[str, Any],
        keep_1d_weights_params_and_scales_in_fp32: bool
        ) -> None:
    # cast all (1d/2d) weights, (learnable) nn.Parameter, (frozen nn.Parameter as) scale
    cast_all = ("all" in bf16_weights_params_and_scales) and (bf16_weights_params_and_scales["all"] is object)
    if cast_all:
        for p in gpt_model.parameters():
            if p.is_floating_point() and p.dtype != torch.bfloat16:
                p.data = p.data.to(torch.bfloat16)
    # potentially cast some of (1d/2d) weights, (learnable) nn.Parameter, (frozen nn.Parameter as) scale
    else:
        target_classes = tuple(bf16_weights_params_and_scales.values())

        # module classes (e.g., nn.Linear, nn.Embedding)
        module_classes = tuple(cls for cls in target_classes if isinstance(cls, type) and issubclass(cls, nn.Module))
        if module_classes:
            for m in gpt_model.modules():
                if isinstance(m, module_classes):
                    m.bfloat16()

        # nn.Parameter
        if any(cls is nn.Parameter for cls in target_classes):
            for p in gpt_model.parameters():
                if p.is_floating_point() and p.dtype != torch.bfloat16:
                    p.data = p.data.to(torch.bfloat16)

    # revert excluded weights, params, scales (as nn.Parameters with requires_grad=False)
    if keep_1d_weights_params_and_scales_in_fp32:
        for p in gpt_model.parameters():
            if p.dim() == 1 and p.dtype == torch.bfloat16:
                p.data = p.data.float()

################################################
#           Model Building / Loading           #
################################################
gpt_config = GPTConfig()
if resume_from_checkpoint:
    # get start_step, train dataloader config (epoch, grad_accum_mini_steps_per_shard_counter, etc.),
    # optimizer state and model
    (start_step, torch_rng_state_cpu, torch_rng_state_cuda, train_tokens_processed,
        total_train_t, total_val_t, total_sample_t, total_hellaswag_t, total_t, best_val_loss, epoch,
        current_shard_idx, grad_accum_mini_steps_per_shard_counter,
        optimizer_state_dicts, gpt_model) = load_checkpoint(
            training_config=training_config,
            hub_repo_id=hub_repo_id,
            resume_checkpoint_path=resume_checkpoint_path,
            resume_state_dict_path=resume_state_dict_path,
            hf_token=hf_token,
        )
    if master_process:
        message = f"loaded checkpoint from step {start_step - 1}"
        print(message)
        log_buffer.append(message)
    dist.barrier()

    # move the downloaded gpt_model to device, compile it, set up DDP, and set raw_gpt_model to create the optimizer
    # based on the downloaded model
    gpt_model.to(device)
    # bf16
    if training_config.use_all_bf16_and_null_ctx or training_config.use_bf16_weights_params_or_scales:
        convert_to_bf16(
            gpt_model,
            training_config.bf16_weights_params_and_scales,
            training_config.keep_1d_weights_params_and_scales_in_fp32
        )
    gpt_model = torch.compile(
        gpt_model,
        mode=training_config.torch_compile_mode,
        fullgraph=training_config.torch_compile_fullgraph,
        dynamic=training_config.torch_compile_dynamic
    )
    gpt_model = DDP(gpt_model, device_ids=[ddp_local_rank])
    raw_gpt_model = gpt_model.module
    optimizers = raw_gpt_model.configure_optimizers(
        adamw_max_lr, adamw_betas, adamw_eps, adamw_weight_decay, device_type,
        master_process, log_buffer,
    )
    # use the downloaded optimizer_state_dicts for the optimizers
    for k, optimizer in optimizers.items():
        if k in optimizer_state_dicts:
            optimizer.load_state_dict(optimizer_state_dicts[k])

    # restore the random number generator state
    torch.set_rng_state(torch_rng_state_cpu)
    torch.cuda.set_rng_state(torch_rng_state_cuda)
    # wait for all ranks to sync
    dist.barrier()
else:
    gpt_model = GPT(gpt_config, training_config)
    gpt_model.to(device)
    # bf16
    if training_config.use_all_bf16_and_null_ctx or training_config.use_bf16_weights_params_or_scales:
        convert_to_bf16(
            gpt_model,
            training_config.bf16_weights_params_and_scales,
            training_config.keep_1d_weights_params_and_scales_in_fp32
        )
    gpt_model = torch.compile(
        gpt_model,
        mode=training_config.torch_compile_mode,
        fullgraph=training_config.torch_compile_fullgraph,
        dynamic=training_config.torch_compile_dynamic
    )
    gpt_model = DDP(gpt_model, device_ids=[ddp_local_rank])
    raw_gpt_model = gpt_model.module
    optimizers = raw_gpt_model.configure_optimizers(
        adamw_max_lr, adamw_betas, adamw_eps, adamw_weight_decay, device_type,
        master_process, log_buffer,
    )

################################################
#               Context Manager                #
################################################
if training_config.use_all_bf16_and_null_ctx or not training_config.use_bf16_autocast:
    # no-op context manager
    ctx = contextlib.nullcontext()
else:
    # standard Mixed Precision
    ctx = torch.autocast(device_type=device_type, dtype=torch.bfloat16)

################################################
#             DataLoader Building              #
################################################
# if resume_from_checkpoint: epoch, current_shard_idx, and grad_accum_mini_steps_per_shard_counter are overriden
# above and thus set to non-zero values in the DataLoader()
train_data_loader = DataLoader(
    gpu_batch_size_train, seq_len_train, ddp_world_size, ddp_rank, data_path, "train",
    epoch=epoch, current_shard_idx=current_shard_idx,
    grad_accum_mini_steps_per_shard_counter=grad_accum_mini_steps_per_shard_counter,
    pad_token_id=gpt_config.pad_token_id, eos_token_id=gpt_config.eos_token_id,
    return_document_ids=raw_gpt_model.use_doc_masking,
    master_process=master_process, log_buffer=log_buffer,
)
val_data_loader = DataLoader(
    gpu_batch_size_val, seq_len_val, ddp_world_size, ddp_rank, data_path, "val",
    epoch=0, current_shard_idx=0, grad_accum_mini_steps_per_shard_counter=0,
    pad_token_id=gpt_config.pad_token_id, eos_token_id=gpt_config.eos_token_id,
    return_document_ids=raw_gpt_model.use_doc_masking,
    shuffle_val_tokens=shuffle_val_tokens,
    master_process=master_process, log_buffer=log_buffer,
)

################################################
# HellaSwag Loading and Splitting across gpus  #
################################################
# load HellaSwag and split it across all GPUs similar to how the train DataLoader assigns different chunks to each GPU
hellaswag_train_dataset = load_dataset("hellaswag", split="train")
hellaswag_val_dataset = load_dataset("hellaswag", split="validation")
total_hellaswag_val_examples = len(hellaswag_val_dataset)
hellaswag_val_examples_per_rank = (total_hellaswag_val_examples + ddp_world_size - 1) // ddp_world_size
hellaswag_val_start_idx = ddp_rank * hellaswag_val_examples_per_rank
hellaswag_val_end_idx = min(hellaswag_val_start_idx + hellaswag_val_examples_per_rank, total_hellaswag_val_examples)
local_hellaswag_val_dataset = hellaswag_val_dataset.select(range(hellaswag_val_start_idx, hellaswag_val_end_idx))
local_message = (
    f"[rank {ddp_rank}] gets HellaSwag sentences "
    f"{hellaswag_val_start_idx:,} to {hellaswag_val_end_idx - 1:,}"
)
print(local_message)
all_messages = [None for _ in range(ddp_world_size)]
dist.all_gather_object(all_messages, local_message)
if master_process:
    log_buffer.extend(all_messages)
dist.barrier()

################################################
#             Config Saving to file            #
################################################
def save_config_info() -> None:
    with open(config_filename, "w") as f:
        f.write(f"timestamp: {timestamp}\n")
        f.write(f"ddp world size: {ddp_world_size}\n\n")

        f.write("\n###################")
        f.write("\n# Training Config #")
        f.write("\n###################\n")
        for config_field in fields(training_config):
            if config_field.name in ['hf_user', 'hf_token']:
                 continue
            elif config_field.name == 'tokenizer':
                 f.write(f"tokenizer: gpt2 (tiktoken)\n")
            elif config_field.name == 'bf16_weights_params_and_scales':
                 keys = list(training_config.bf16_weights_params_and_scales.keys())
                 f.write(f"bf16_weights_params_and_scales: {keys}\n")
            elif config_field.init is False:
                 continue
            else:
                value = getattr(training_config, config_field.name)
                f.write(f"{config_field.name}: {value}\n")

        f.write("\n###########################")
        f.write("\n# Derived Training Config #")
        f.write("\n###########################\n")
        f.write(f"total_tokens_per_mini_step_train: {total_tokens_per_mini_step_train}\n")
        f.write(f"grad_accum_mini_steps: {grad_accum_mini_steps}\n")
        f.write(f"total_tokens_per_step_val: {total_tokens_per_step_val}\n")
        f.write(f"val_steps: {val_steps}\n")
        f.write(f"max_train_steps: {max_train_steps}\n")
        f.write(f"total_decay_tokens: {total_decay_tokens}\n")
        f.write(f"torch_compile_mode: {training_config.torch_compile_mode}\n")

        f.write("\n################")
        f.write("\n# Model Config #")
        f.write("\n################\n")
        for k, v in gpt_config.__dict__.items():
             f.write(f"{k}: {v}\n")

    print(f"training and model configs saved to {config_filename}")

if master_process:
    save_config_info()

################################################
#              Parameter Logging               #
################################################
if master_process:
    total_params = sum(p.numel() for p in gpt_model.parameters() if p.requires_grad)
    message = f"{total_params:,} parameters"
    print(message)
    log_buffer.append(message)

    # active parameters per token (124M speedrun rule): each embedding table counts as d_model active params
    emb_params = 0
    emb_tables = 0
    # token embedding
    wte = raw_gpt_model.transformer.wte.weight
    emb_params += wte.numel()
    emb_tables += 1
    # positional embedding (if absolute)
    if hasattr(raw_gpt_model.transformer, "wpe"):
        emb_params += raw_gpt_model.transformer.wpe.weight.numel()
        emb_tables += 1
    # lm_head counts only if untied
    if not gpt_config.use_tied_embeddings:
        emb_params += raw_gpt_model.lm_head.weight.numel()
        emb_tables += 1

    active_params = total_params - emb_params + emb_tables * gpt_config.d_model
    message = f"active parameters per token: {active_params:,}"
    print(message)
    log_buffer.append(message)

################################################
#                Kernel Warmup                 #
################################################
def _kernel_warmup(num_train_steps: int = 2) -> None:
    # snapshot everything so we don't "cheat"
    model_state = copy.deepcopy(raw_gpt_model.state_dict())
    optimizer_states = {k: copy.deepcopy(opt.state_dict()) for k, opt in optimizers.items()}
    rng_state_cpu = torch.get_rng_state()
    rng_state_cuda = torch.cuda.get_rng_state()

    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()

    # calculate all unique batch sizes to compile before training starts
    warmup_batch_sizes = sorted(set(batch_size_schedule_values), reverse=True)
    print(f"kernel_warmup: gpu {ddp_rank} will compile kernels for batch sizes: {warmup_batch_sizes}")

    # train-shape warmup (compile both DDP graphs)
    gpt_model.train()
    with torch.enable_grad():
        for batch_size in warmup_batch_sizes:
            print(f"kernel_warmup: gpu {ddp_rank} compiling kernels for batch size: {batch_size}")
            for _ in range(num_train_steps):
                for optimizer in optimizers.values():
                    optimizer.zero_grad(set_to_none=True)

                for mini_step in range(grad_accum_mini_steps):
                    # mimic the real training loop’s DDP behavior
                    gpt_model.require_backward_grad_sync = (mini_step == grad_accum_mini_steps - 1)

                    # make shapes match training
                    x_train = torch.randint(
                        0, raw_gpt_model.pad_token_id,
                        (batch_size, seq_len_train), device=device
                    )
                    y_train = torch.randint(
                        0, raw_gpt_model.pad_token_id,
                        (batch_size, seq_len_train), device=device
                    )

                    # synthesize doc_ids so the doc-masking + SWA FlexAttention path compiles
                    if raw_gpt_model.use_doc_masking:
                        # set random-ish EOS boundaries
                        step = max(16, seq_len_train // 8)
                        idxs = torch.arange(seq_len_train, device=device)[None, :]
                        rand_offsets = torch.randint(0, step, (batch_size, 1), device=device)
                        is_eos = ((idxs + rand_offsets) % step == 0)
                        doc_ids_train = torch.cumsum(is_eos.to(torch.int32), dim=1) - is_eos.to(torch.int32)
                    else:
                        doc_ids_train = None

                    with ctx:
                        _, warm_loss = gpt_model(x_train, y_train, document_ids=doc_ids_train)

                    (warm_loss / grad_accum_mini_steps).backward()

                for optimizer in optimizers.values():
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()

    # val-shape warmup (compile eval forward)
    gpt_model.eval()
    with torch.inference_mode():
        x_val = torch.randint(
            0, raw_gpt_model.pad_token_id,
            (gpu_batch_size_val, seq_len_val), device=device
        )
        y_val = torch.randint(
            0, raw_gpt_model.pad_token_id,
            (gpu_batch_size_val, seq_len_val), device=device
        )
        if raw_gpt_model.use_doc_masking:
            step = max(16, seq_len_val // 8)
            idxs = torch.arange(seq_len_val, device=device)[None, :]
            rand_offsets = torch.randint(0, step, (gpu_batch_size_val, 1), device=device)
            is_eos = ((idxs + rand_offsets) % step == 0)
            doc_ids_val = torch.cumsum(is_eos.to(torch.int32), dim=1) - is_eos.to(torch.int32)
        else:
            doc_ids_val = None

        _ = gpt_model(x_val, y_val, document_ids=doc_ids_val)

    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()

    # sampling/hellaswag warmup is skipped because both run via _orig_mod eager path

    # restore state
    raw_gpt_model.load_state_dict(model_state)
    for k, optimizer in optimizers.items():
        optimizer.load_state_dict(optimizer_states[k])
    torch.set_rng_state(rng_state_cpu)
    torch.cuda.set_rng_state(rng_state_cuda)

    torch.cuda.synchronize()
    dist.barrier()

dist.barrier()
torch.cuda.synchronize()
kernel_warmup_start_t = time.perf_counter()

_kernel_warmup(num_train_steps=kernel_warmup_train_steps)

torch.cuda.synchronize()
dist.barrier()
kernel_warmup_time_s = time.perf_counter() - kernel_warmup_start_t
kernel_warmup_time_t = torch.tensor([kernel_warmup_time_s], dtype=torch.float64, device=device)
dist.all_reduce(kernel_warmup_time_t, op=dist.ReduceOp.MAX)
kernel_warmup_time_s = float(kernel_warmup_time_t.item())
if master_process:
    message = f"kernel warmup compile time: {kernel_warmup_time_s:.2f}s ({kernel_warmup_time_s / 60.0:.2f} min)"
    print(message)
    log_buffer.append(message)

##################################################################
#  Training, Validation, Sampling, HellaSwag, Checkpointing loop #
##################################################################
try:
    swa_schedule_keys = training_config.swa_schedule_keys
    swa_schedule_values = training_config.swa_schedule_values
    swa_schedule_idx = 0
    current_window_size = swa_schedule_values[swa_schedule_idx]
    next_swa_update_tokens = get_next_update_tokens(
        swa_schedule_keys,
        swa_schedule_idx,
        training_config.max_tokens
    )

    batch_schedule_idx = 0
    current_batch_size = batch_size_schedule_values[batch_schedule_idx]
    next_batch_update_tokens = get_next_update_tokens(
        batch_size_schedule_keys,
        batch_schedule_idx,
        training_config.max_tokens
    )

    lr_schedule_idx = 0
    next_lr_update_tokens = lr_schedule[lr_schedule_idx]["end"]

    for step in range(start_step, max_train_steps):
        
        torch.cuda.synchronize()
        start_train_t = time.time()
        gpt_model.train()
        for optimizer in optimizers.values():
            optimizer.zero_grad()
        train_loss = 0.0
        # step_checkpointed = False

        # get batch size
        if train_tokens_processed >= next_batch_update_tokens:
            if batch_schedule_idx + 1 < len(batch_size_schedule_keys):
                batch_schedule_idx += 1
                current_batch_size = batch_size_schedule_values[batch_schedule_idx]
                next_batch_update_tokens = get_next_update_tokens(
                    batch_size_schedule_keys,
                    batch_schedule_idx,
                    training_config.max_tokens
                )

        # per-gpu grad accumulation mini-steps
        for mini_step in range(grad_accum_mini_steps):
            x_train, y_train, doc_ids_train = train_data_loader.next_batch()
            x_train = x_train[:current_batch_size].pin_memory().to(device, non_blocking=True)
            y_train = y_train[:current_batch_size].pin_memory().to(device, non_blocking=True)
            if doc_ids_train is not None:
                doc_ids_train = doc_ids_train[:current_batch_size].pin_memory().to(device, non_blocking=True)
            gpt_model.require_backward_grad_sync = (mini_step == grad_accum_mini_steps - 1)
            with ctx:
                _, step_train_loss = gpt_model(x_train, y_train, document_ids=doc_ids_train)
            step_train_loss /= grad_accum_mini_steps
            train_loss += step_train_loss.detach()
            step_train_loss.backward()

        dist.all_reduce(train_loss, op=dist.ReduceOp.AVG)
        
        # # if clipping is enabled and threshold > 0, use it; otherwise use +inf (no-op clip).
        # grad_clip_enabled = raw_gpt_model.use_grad_norm_clipping
        # gradient_clipping_norm = raw_gpt_model.gradient_clipping_norm if (grad_clip_enabled and raw_gpt_model.gradient_clipping_norm > 0.0) else float('inf')
        # # instead of clamping each weight’s gradient (individually), we compute the global L2
        # # norm of all gradients across the parameter set and if ||G|| > max_norm: we scale 
        # # all gradients down proportionally
        # # -------------
        # # the steps are:
        # # forward pass -> loss
        # # loss.backward() -> autograd computes true grads in p.grad
        # # (if using AMP GradScaler) scaler.unscale_(optimizer)
        # # clip grads (global norm) -> in-place scale of p.grad
        # # optimizer.step()
        # # optimizer.zero_grad() (next step)
        # # -------------
        # # Hence, after deriving everything (dJ/dw_i for each w_i), we have
        # # G which can be seen as a (total_params, 1) grad column vector 
        # # that (if gradient clipping is applied) is rescaled to 
        # # have its L2 norm <= grad_clip_norm_threshold
        # grad_norm_pre_clipping = torch.nn.utils.clip_grad_norm_(gpt_model.parameters(), gradient_clipping_norm)
        # # recompute global norm after clipping (cheap)
        # if grad_clip_enabled:
        #     # grad_sq = [p.grad.detach().float().pow(2).sum() for p in gpt_model.parameters() if p.grad is not None]
        #     # grad_norm_post_clipping = float(torch.stack(grad_sq).sum().sqrt())
        #     grad_norm_post_clipping = grad_norm_pre_clipping if grad_norm_pre_clipping < raw_gpt_model.gradient_clipping_norm else raw_gpt_model.gradient_clipping_norm
        # else:
        #     grad_norm_post_clipping = grad_norm_pre_clipping
        # grad_norm_text = f"{grad_norm_pre_clipping:.4f}" + (f" pre- / {grad_norm_post_clipping:.4f} post-clipping" if grad_clip_enabled else "")
        # <-- Uncomment for gradient norm clipping -->
        # grad_norm_pre_clipping = torch.nn.utils.clip_grad_norm_(gpt_model.parameters(),raw_gpt_model.gradient_clipping_norm)
        # grad_norm_text = (
        #         f"{float(grad_norm_pre_clipping):.4f} pre-"
        # )
        # <-- Uncomment for gradient norm clipping -->

        # update lr
        if train_tokens_processed >= next_lr_update_tokens:
            if lr_schedule_idx + 1 < len(lr_schedule):
                lr_schedule_idx += 1
                next_lr_update_tokens = lr_schedule[lr_schedule_idx]["end"]
        lr_spec = lr_schedule[lr_schedule_idx]
        lr_tokens = train_tokens_processed - lr_spec["start"]
        adamw_lr = lr_spec["fn"](lr_tokens, **lr_spec["kwargs"])

        # update SWA window size (token schedule)
        if train_tokens_processed >= next_swa_update_tokens:
            if swa_schedule_idx + 1 < len(swa_schedule_keys):
                swa_schedule_idx += 1
                current_window_size = swa_schedule_values[swa_schedule_idx]
                next_swa_update_tokens = get_next_update_tokens(
                    swa_schedule_keys,
                    swa_schedule_idx,
                    training_config.max_tokens
                )
        # update the buffer in-place to avoid graph breaks
        raw_gpt_model.sliding_window_size.fill_(current_window_size)

        # AdamW gets the base learning rate
        for param_group in optimizers["adamw"].param_groups:
            param_group['lr'] = adamw_lr
        # Muon if present gets scaled learning rate
        if "muon" in optimizers:
            for param_group in optimizers["muon"].param_groups:
                param_group['lr'] = adamw_lr * raw_gpt_model.muon_lr_scale
        # Step present optimizers
        for optimizer in optimizers.values():
            optimizer.step()

        # the logits can vary across ranks
        # <-- Uncomment for logit soft-capping -->
        # logits_absmax_pre_capping = torch.tensor(float(raw_gpt_model._logits_absmax_stats["logits_absmax_pre_capping"]), device=device)
        # logits_absmax_post_capping = torch.tensor(float(raw_gpt_model._logits_absmax_stats["logits_absmax_post_capping"]), device=device)
        # # MAX or MEAN
        # dist.all_reduce(logits_absmax_pre_capping, op=dist.ReduceOp.MAX)
        # dist.all_reduce(logits_absmax_post_capping, op=dist.ReduceOp.MAX)
        # <-- Uncomment for logit soft-capping -->

        tl = float(train_loss)

        torch.cuda.synchronize()
        end_train_t = time.time()
        train_step_t = end_train_t - start_train_t
        total_train_t += train_step_t
        total_t += train_step_t
        
        # obtain the actual tokens trained on this step (which can vary due to varying batch size)
        # on all gpus (so each updates the schedule), *not just on master*
        actual_tokens_this_step = current_batch_size * seq_len_train * ddp_world_size * grad_accum_mini_steps
        train_tokens_processed += actual_tokens_this_step

        if master_process:
            # <-- Uncomment for logit soft-capping -->
            # if raw_gpt_model.use_lm_head_logit_softcapping:
            #     logits_suffix = f" | logits absmax pre/post {logits_absmax_pre_capping.item():.4f}/{logits_absmax_post_capping.item():.4f}"
            # else:
            #     logits_suffix = f" | logits absmax {logits_absmax_pre_capping.item():.4f}"
            # <-- Uncomment for logit soft-capping -->
            # the scale is constant across ranks because it is head-dependant, and all
            # gpus share the heads (even though with different data)
            sw_size_suffix = f"sw size: {raw_gpt_model.sliding_window_size}"
            qk_suffix = ""
            if raw_gpt_model.use_qk_norm and raw_gpt_model.use_qk_debug_log:
                qk_suffix = qk_scale_debug_string(raw_gpt_model)

            train_log_content = (
                f"step: {step:,} | "
                f"train loss: {tl:.8f} | "
                f"train ppl: {math.exp(tl):,.2f} | "
                f"train step time: {1000*(train_step_t):,.2f} ms | "
                # f"grad norm: {grad_norm_text} | "
                f"adamw lr: {adamw_lr:.8f} | "
                f"tok/s: {actual_tokens_this_step / train_step_t:,.2f} | "
                f"total toks: {train_tokens_processed:,} | "
                f"total train time: {total_train_t/60:,.2f} min | "
                f"{sw_size_suffix} | "
                f"{qk_suffix}"
                f"batch size: {current_batch_size:,}"
                # f"{logits_suffix}"
            )
            print(train_log_content)
            # with open(log_filename, "a") as f:
            #     f.write(train_log_content + "\n")
            log_buffer.append(train_log_content)

        if training_config.run_sampling and ((step % sample_interval == 0 and step > 0) or step == max_train_steps - 1):
            torch.cuda.synchronize()
            start_sample_t = time.time()
            max_new_tokens = get_sample_token_count(step)
            sample(
                sample_sequences,
                max_new_tokens=max_new_tokens,
                gpt_model=gpt_model,
                raw_gpt_model=raw_gpt_model,
                gpt_config=gpt_config,
                training_config=training_config,
                tokenizer=tokenizer,
                device=device,
                ctx=ctx,
                ddp_rank=ddp_rank,
                ddp_world_size=ddp_world_size,
                master_process=master_process,
                log_buffer=log_buffer,
            )
            torch.cuda.synchronize()
            end_sample_t = time.time()
            sample_step_t = end_sample_t - start_sample_t
            total_sample_t += sample_step_t
            total_t += sample_step_t
            if master_process:
                message = f"step: {step:,} | sampling time: {(sample_step_t):,.2f} s"
                print(message)
                log_buffer.append(message)

        if training_config.run_benchmarks and ((step % hellaswag_interval == 0 and step > 0) or step == max_train_steps - 1):
            torch.cuda.synchronize()
            start_hellaswag_t = time.time()
            accuracy = evaluate_hellaswag_standard(
                gpt_model=gpt_model,
                raw_gpt_model=raw_gpt_model,
                tokenizer=tokenizer,
                hellaswag_train_dataset=hellaswag_train_dataset,
                local_hellaswag_val_dataset=local_hellaswag_val_dataset,
                device=device,
                ctx=ctx,
                ddp_rank=ddp_rank,
                ddp_world_size=ddp_world_size,
                master_process=master_process,
                log_buffer=log_buffer,
            )
            torch.cuda.synchronize()
            end_hellaswag_t = time.time()
            hellaswag_step_t = end_hellaswag_t - start_hellaswag_t
            total_hellaswag_t += hellaswag_step_t
            total_t += hellaswag_step_t
            if master_process:
                message = f"step: {step:,} | HellaSwag acc: {accuracy:.4f} | HellaSwag time: {(hellaswag_step_t):,.2f} s"
                print(message)
                log_buffer.append(message)
        
        # val gating
        if (tl + train_val_margin <= val_target) and not allow_val:
            allow_val = True
            if master_process:
                message = (f"val enabled at step {step} - {tl} train loss")
                print(message)
                log_buffer.append(message)
        
        if (allow_val and (step % val_interval == 0 and step > 0)) or step == max_train_steps - 1:
            torch.cuda.synchronize()
            start_val_t = time.time()
            gpt_model.eval()
            if master_process:
                message = f"resetting val loader at step {step}"
                print(message)
                log_buffer.append(message)
            val_data_loader.reset()
            with torch.inference_mode():
                val_loss = 0.0
                for _ in range(val_steps):
                    x_val, y_val, doc_ids_val = val_data_loader.next_batch()
                    x_val = x_val.pin_memory().to(device, non_blocking=True)
                    y_val = y_val.pin_memory().to(device, non_blocking=True)
                    if doc_ids_val is not None:
                        doc_ids_val = doc_ids_val.pin_memory().to(device, non_blocking=True)
                    _, step_val_loss = gpt_model(x_val, y_val, document_ids=doc_ids_val)
                    # cast to float32 so accumulation and comparison stay in float32,
                    # to avoid the target to be bf16-casted (e.g., to 3.28125)
                    val_loss += step_val_loss.float() / val_steps
                dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if master_process:
                    # pass (exact values as) arguments to avoid modifying the counters before it saves
                    # as save_checkpoint must defer writing to disk and the training loop continues
                    # before variables such as step or train_tokens_processed are stored
                    if max_checkpoints_to_keep > 0:
                        save_checkpoint(step=step,
                            torch_rng_state_cpu=torch.get_rng_state(),
                            torch_rng_state_cuda=torch.cuda.get_rng_state(),
                            train_loss=train_loss.item(),
                            val_loss=val_loss.item(),
                            train_tokens_processed=train_tokens_processed,
                            total_train_t=total_train_t,
                            total_val_t=total_val_t,
                            total_sample_t=total_sample_t,
                            total_hellaswag_t=total_hellaswag_t,
                            total_t=total_t,
                            best_val_loss=best_val_loss,
                            epoch=train_data_loader.epoch,
                            current_shard_idx=train_data_loader.current_shard_idx,
                            grad_accum_mini_steps_per_shard_counter=train_data_loader.grad_accum_mini_steps_per_shard_counter,
                            optimizers=optimizers,
                            gpt_model=gpt_model,
                            checkpoint_dir=checkpoint_dir,
                            max_checkpoints_to_keep=max_checkpoints_to_keep,
                            log_buffer=log_buffer)
                    message = f"new best val loss: {best_val_loss:.8f}"
                    print(message)
                    log_buffer.append(message)
                    # only after we improve val loss and a checkpoint is saved, we push the logs
                    with open(log_filename, "a") as f:
                        for line in log_buffer:
                            f.write(line + "\n")
                        log_buffer.clear()
                dist.barrier()
                # step_checkpointed = True

            torch.cuda.synchronize()
            end_val_t = time.time()
            val_step_t = end_val_t - start_val_t
            total_val_t += val_step_t
            total_t += val_step_t

            if master_process:
                val_log_content = f"step: {step:,} | val loss: {val_loss.item():.8f} | val ppl: {math.exp(val_loss.item()):,.2f} | val time: {1000*(val_step_t):,.2f} ms"
                print(val_log_content)
                # with open(log_filename, "a") as f:
                #     f.write(val_log_content + "\n")
                log_buffer.append(val_log_content)

            if val_loss <= val_target:
                if master_process:
                    print(f"val loss {val_loss.item():.8f} reached target {val_target}")
                break

        # if not step_checkpointed and step % checkpoint_interval == 0 and step > 0 and master_process:
        #     start_checkpointing_t = time.time()
        #     save_checkpoint(train_loss=train_loss)
        #     torch.cuda.synchronize()
        #     end_checkpointing_t = time.time()
        #     checkpointing_step_t = end_checkpointing_t - start_checkpointing_t
        #     total_t += checkpointing_step_t

except Exception as e:
    if master_process:
        print(f"[rank {ddp_rank}] unhandled exception: {e}")
        import traceback
        traceback.print_exc()
    dist.barrier()
    raise
finally:
    dist.barrier()
    destroy_process_group()
