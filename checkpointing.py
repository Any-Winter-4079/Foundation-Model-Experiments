import os
from typing import Any, Dict, List, Tuple

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_model
from torch import Tensor

from config.model import GPTConfig
from config.training import TrainingConfig
from models.gpt import GPT

################################################
#                Checkpointing                 #
################################################
def keep_latest_checkpoints(
        checkpoint_dir: str,
        max_checkpoints_to_keep: int,
        log_buffer: List[str],
        ) -> None:
    all_files = os.listdir(checkpoint_dir)

    # group files by type
    model_files = sorted(
        [f for f in all_files if f.endswith(".safetensors")],
        key=lambda f: os.path.getmtime(os.path.join(checkpoint_dir, f))
    )
    state_files = sorted(
        [f for f in all_files if f.endswith(".pt")],
        key=lambda f: os.path.getmtime(os.path.join(checkpoint_dir, f))
    )

    # for every file that is past the last max_checkpoints_to_keep, remove it,
    # resulting in empty list if max_checkpoints_to_keep >= len(files),
    # and thus preserving all files in that case

    # prune excess model files
    for old_file in model_files[:-max_checkpoints_to_keep]:
        path = os.path.join(checkpoint_dir, old_file)
        os.remove(path)
        message = f"removed model: {path}"
        print(message)
        log_buffer.append(message)

    # prune excess state files
    for old_file in state_files[:-max_checkpoints_to_keep]:
        path = os.path.join(checkpoint_dir, old_file)
        os.remove(path)
        message = f"removed training state: {path}"
        print(message)
        log_buffer.append(message)

def save_checkpoint(
        step: int,
        torch_rng_state_cpu: Tensor,
        torch_rng_state_cuda: Tensor,
        val_loss: float,
        train_loss: float,
        train_tokens_processed: int,
        total_train_t: float,
        total_val_t: float,
        total_sample_t: float,
        total_hellaswag_t: float,
        total_t: float,
        best_val_loss: float,
        epoch: int,
        current_shard_idx: int,
        grad_accum_mini_steps_per_shard_counter: int,
        optimizers: Dict[str, torch.optim.Optimizer],
        gpt_model: Any,
        checkpoint_dir: str,
        max_checkpoints_to_keep: int,
        log_buffer: List[str],
        ) -> None:

    # create the model path by adding its timestamp, and train and val losses, if available
    parts = [f"model", f"step_{step:07d}"]
    parts.append(f"val_{val_loss:.4f}")
    parts.append(f"train_{train_loss:.4f}")
    checkpoint_name = "_".join(parts) + ".safetensors"
    safetensors_path = os.path.join(checkpoint_dir, checkpoint_name)

    # save the model locally
    save_model(gpt_model.module, safetensors_path)
    message = f"saved model weights locally in: {safetensors_path}"
    print(message)
    log_buffer.append(message)

    # gather the training state
    training_state = {
        'step': step,
        'torch_rng_state_cpu': torch_rng_state_cpu,
        'torch_rng_state_cuda': torch_rng_state_cuda,
        'train_tokens_processed': train_tokens_processed,
        'total_train_t': total_train_t,
        'total_val_t': total_val_t,
        'total_sample_t': total_sample_t,
        'total_hellaswag_t': total_hellaswag_t,
        'total_t': total_t,
        'best_val_loss': best_val_loss,
        'epoch': epoch,
        'current_shard_idx': current_shard_idx,
        'grad_accum_mini_steps_per_shard_counter': grad_accum_mini_steps_per_shard_counter,
        'optimizer_state_dicts': {k: optimizer.state_dict() for k, optimizer in optimizers.items()}
    }

    # create the training state path
    state_path = os.path.join(checkpoint_dir, f"training_state_step_{step:07d}.pt")

    # save the training state locally
    torch.save(training_state, state_path)
    message = f"saved training state locally in: {state_path}"
    print(message)
    log_buffer.append(message)

    # retain last max_checkpoints_to_keep
    keep_latest_checkpoints(checkpoint_dir, max_checkpoints_to_keep, log_buffer)

# TODO: fix checkpointing to handle the added logic of SWA ramp, ...
def load_checkpoint(
        training_config: TrainingConfig,
        hub_repo_id: str,
        resume_checkpoint_path: str,
        resume_state_dict_path: str,
        hf_token: str,
        ) -> Tuple[
    int,                                # start_step
    Tensor,                             # torch_rng_state_cpu
    Tensor,                             # torch_rng_state_cuda
    int,                                # train_tokens_processed
    float,                              # total_train_t
    float,                              # total_val_t
    float,                              # total_sample_t
    float,                              # total_hellaswag_t
    float,                              # total_t
    float,                              # best_val_loss
    int,                                # epoch
    int,                                # current_shard_idx
    int,                                # grad_accum_mini_steps_per_shard_counter
    Dict[str, Any],                     # optimizer_state_dicts
    "GPT",                              # gpt_model
]:
    model_path = hf_hub_download(
        repo_id=hub_repo_id,
        filename=resume_checkpoint_path,
        token=hf_token,
        repo_type="model"
    )
    training_state_path = hf_hub_download(
        repo_id=hub_repo_id,
        filename=resume_state_dict_path,
        token=hf_token,
        repo_type="model"
    )

    # load weights
    raw_state_dict = load_file(model_path, device='cpu')
    # print(list(raw_state_dict.keys())[:5])

    # strip `_orig_mod.` prefix from keys
    model_state_dict = {
        k.replace("_orig_mod.", ""): v for k, v in raw_state_dict.items()
    }
    # print(list(model_state_dict.keys())[:5])

    gpt_config = GPTConfig()
    gpt_model = GPT(gpt_config, training_config)
    gpt_model.load_state_dict(model_state_dict, strict=False)

    # tie weights, creating wte.weight
    if gpt_model.use_tied_embeddings:
        gpt_model.transformer.wte.weight = gpt_model.lm_head.weight

    # load training state
    training_state = torch.load(training_state_path, map_location="cpu")

    return (
        # continue on the next step after val
        training_state['step'] + 1,
        training_state['torch_rng_state_cpu'],
        training_state['torch_rng_state_cuda'],
        training_state['train_tokens_processed'],
        training_state['total_train_t'],
        training_state['total_val_t'],
        training_state['total_sample_t'],
        training_state['total_hellaswag_t'],
        training_state['total_t'],
        training_state['best_val_loss'],
        training_state['epoch'],
        training_state['current_shard_idx'],
        training_state['grad_accum_mini_steps_per_shard_counter'],
        training_state['optimizer_state_dicts'],
        gpt_model,
    )
