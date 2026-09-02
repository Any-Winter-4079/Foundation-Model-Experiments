import os
import threading
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

################################################
#                  DataLoader                  #
################################################
class DataLoader:
    def __init__(self,
                 gpu_batch_size: int,
                 seq_len: int,
                 ddp_world_size: int,
                 ddp_rank: int,
                 data_folder: str,
                 split: str = "train",
                 epoch: int = 0,
                 current_shard_idx: int = 0,
                 grad_accum_mini_steps_per_shard_counter: int = 0,
                 pad_token_id: int = 50257, 
                 eos_token_id: int = 50256,
                 return_document_ids: bool = False, 
                 shuffle_val_tokens: bool = True,
                 master_process: bool = False,
                 log_buffer: Optional[List[str]] = None,
                 ) -> None:
        # batch size each gpu can fit in
        self.gpu_batch_size = gpu_batch_size
        # seq len for each gpu
        self.seq_len = seq_len
        # total gpu count
        self.ddp_world_size = ddp_world_size
        # 'id' of gpu
        self.ddp_global_rank = ddp_rank
        self.split = split
        self.epoch = epoch
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.return_document_ids = return_document_ids
        self.shuffle_val_tokens = shuffle_val_tokens
        self.master_process = master_process
        self.log_buffer = log_buffer if log_buffer is not None else []

        shards = os.listdir(data_folder)
        shards = [shard for shard in shards if split in shard]
        sorted_shards = sorted(shards)
        sorted_shards = [os.path.join(data_folder, shard) for shard in sorted_shards]
        assert len(sorted_shards) > 0, f"no shards found for split {split}"
        if self.master_process:
            message = f"found {len(sorted_shards)} shards for split {split}"
            print(message)
            self.log_buffer.append(message)
        self.sorted_shards = sorted_shards
        self.current_shard_idx = current_shard_idx

        # load shard tokens (CPU long tensor; materialize now to avoid memmap stalls)
        self.tokens = self.load_shard_tokens(sorted_shards[self.current_shard_idx])

        # there are ddp_world_size (e.g., 2) gpus,
        # each to get gpu_batch_size (e.g., 2) sequences,
        # each of seq_len (e.g., 1024) tokens
        # then for the current shard (loading the next shard upon token exhaustion):
        # we discard tokens that wouldn't fit into the final grad accumulation mini-step,
        # e.g., tokens in positions 12289-16000 if there were 16000 tokens total in 
        # the shard, keeping 0-12288 (12289 tokens, with 12288 fitting nicely into 3 
        # grad accum mini steps of 2 * 2 * 1024 = 4096 tokens each plus a final token
        # for y, since it is always one token ahead of x)
        self.grad_accum_tokens_per_mini_step = ddp_world_size * gpu_batch_size * seq_len
        # needing grad_accum_mini_steps_per_shard (e.g., 3) to exhaust tokens
        self.grad_accum_mini_steps_per_shard = len(self.tokens) // self.grad_accum_tokens_per_mini_step
        # e.g., take up to 3 * 4096 = 12288 and a final token since y is one ahead
        self.tokens = self.tokens[:self.grad_accum_tokens_per_mini_step * self.grad_accum_mini_steps_per_shard + 1]

        # precompute all possible x sequence start indices (stride = seq_len)
        # e.g., [0, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192, 9216, 10240, 11264]
        # to get tokens 0-1023, 1024-2047, 2048-3071, 3072-4095, 4096-5119, 5120-6143,
        # 6144-7167, 7168-8191, 8192-9215, 9216-10239, 10240-11263, 11264-12287
        self.x_seq_starts = list(range(0, len(self.tokens) - seq_len, seq_len))

        if self.master_process:
            self.show_shard_info()

        # shuffle x chunk starts, e.g.,
        # [0, 2048, 11264, 10240, 4096, 9216, 1024, 5120, 6144, 8192, 3072, 7168]
        self._shuffle_shard()

        # after this counter reaches self.grad_accum_mini_steps_per_shard, the next shard should replace self.tokens
        self.grad_accum_mini_steps_per_shard_counter = grad_accum_mini_steps_per_shard_counter

        # prefetch machinery
        self._prefetch_thread = None
        self._next_tokens = None
        self._start_shard_prefetch()
    
    def show_shard_info(self) -> None:
        messages = [
            f"total {self.split} shard {self.current_shard_idx} tokens: {len(self.tokens):,}",
            f"total gpus: {self.ddp_world_size}",
            f"gpu batch size: {self.gpu_batch_size:,} sequences",
            f"sequence length: {self.seq_len:,} tokens",
            f"tokens fed to gpu per grad accum mini-step: {(self.seq_len * self.gpu_batch_size):,} ({self.ddp_world_size:,} gpus, {self.grad_accum_tokens_per_mini_step:,} total tokens)",
            f"per-gpu grad accumulation mini-steps for {self.split} shard {self.current_shard_idx} (each mini-step processing {self.grad_accum_tokens_per_mini_step:,} tokens): {(len(self.tokens) // self.grad_accum_tokens_per_mini_step):,}"
        ]
        for message in messages:
            print(message)
            self.log_buffer.append(message)
    
    def _shuffle_shard(self) -> None:
        if self.split == "train" or (self.split == "val" and self.shuffle_val_tokens):
            g = torch.Generator()
            g.manual_seed(self.current_shard_idx + self.epoch)
            # e.g., [0, 2048, 11264, 10240, 4096, 9216, 1024, 5120, 6144, 8192, 3072, 7168]
            shuffled_indices = torch.randperm(len(self.x_seq_starts), generator=g).tolist()
            self.x_seq_starts = [self.x_seq_starts[i] for i in shuffled_indices]
        # else: keep original order for validation (global prefix)

        # e.g., 12 * 0 / 2 = 0 for gpu0, 12 * 1 / 2 = 6 for gpu 1
        gpu_x_seq_start_idx = len(self.x_seq_starts) * self.ddp_global_rank // self.ddp_world_size
        # e.g., 0 + 12 / 2 = 6 for gpu0, 6 + 12 / 2 = 12 for gpu 1
        gpu_x_seq_end_idx = gpu_x_seq_start_idx + len(self.x_seq_starts) // self.ddp_world_size
        # e.g., [0, 2048, 11264, 10240, 4096, 9216] for gpu 0
        # [1024, 5120, 6144, 8192, 3072, 7168] for gpu 1
        self.gpu_x_seq_starts = self.x_seq_starts[gpu_x_seq_start_idx:gpu_x_seq_end_idx]


    def _start_shard_prefetch(self) -> None:
        # prefetch the next shard if not already running
        if self._prefetch_thread is not None and self._prefetch_thread.is_alive():
            return
        next_idx = (self.current_shard_idx + 1) % len(self.sorted_shards)

        def _worker():
            # load + convert once on CPU (int32 -> long), avoids memmap stalls
            np_arr = np.load(self.sorted_shards[next_idx])
            tokens = torch.tensor(np_arr.astype(np.int32), dtype=torch.long)
            self._next_tokens = tokens

        self._prefetch_thread = threading.Thread(target=_worker, daemon=True)
        self._prefetch_thread.start()

    def load_shard_tokens(self, shard_path: str) -> Tensor:
        # materialize to CPU long tensor (stable, no memmap)
        return torch.tensor(np.load(shard_path).astype(np.int32), dtype=torch.long)
    
    def reset(self) -> None:
        self.current_shard_idx = 0
        self.grad_accum_mini_steps_per_shard_counter = 0
        self.tokens = self.load_shard_tokens(self.sorted_shards[self.current_shard_idx])
        # recompute/all the same work as in __init__ for the current shard
        self.grad_accum_mini_steps_per_shard = len(self.tokens) // self.grad_accum_tokens_per_mini_step
        self.tokens = self.tokens[: self.grad_accum_tokens_per_mini_step * self.grad_accum_mini_steps_per_shard + 1]
        self.x_seq_starts = list(range(0, len(self.tokens) - self.seq_len, self.seq_len))
        self._shuffle_shard()
        # restart prefetch after reset
        self._next_tokens = None
        self._start_shard_prefetch()
    
    def next_batch(self) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        if self.grad_accum_mini_steps_per_shard_counter == self.grad_accum_mini_steps_per_shard:
            self.current_shard_idx += 1
            self.grad_accum_mini_steps_per_shard_counter = 0
            if self.current_shard_idx >= len(self.sorted_shards):
                self.current_shard_idx = 0
                self.epoch += 1
                if self.master_process:
                    message = f"--- starting epoch {self.epoch} ---"
                    print(message); self.log_buffer.append(message)
                    self.show_shard_info()

            # use prefetched tokens if ready, else fall back to blocking load
            if self._next_tokens is not None:
                self.tokens = self._next_tokens
                self._next_tokens = None
            else:
                self.tokens = self.load_shard_tokens(self.sorted_shards[self.current_shard_idx])

            # recompute shard-dependent fields after swapping shards
            self.grad_accum_mini_steps_per_shard = len(self.tokens) // self.grad_accum_tokens_per_mini_step
            self.tokens = self.tokens[: self.grad_accum_tokens_per_mini_step * self.grad_accum_mini_steps_per_shard + 1]
            self.x_seq_starts = list(range(0, len(self.tokens) - self.seq_len, self.seq_len))

            self._shuffle_shard()
            # immediately prefetch the following shard
            self._start_shard_prefetch()
        
        if (self.split == "val") and (not self.shuffle_val_tokens):
            # global, ordered prefix partitioning across ranks
            global_batch_start = self.grad_accum_mini_steps_per_shard_counter * self.ddp_world_size * self.gpu_batch_size
            start = global_batch_start + self.ddp_global_rank * self.gpu_batch_size
            batch_starts = self.x_seq_starts[start:start + self.gpu_batch_size]
        else:
            # e.g., 0 for the 1st grad accum mini-step, gpu_batch_size for the 2nd grad accum mini-step, etc.
            i = self.grad_accum_mini_steps_per_shard_counter * self.gpu_batch_size
            # e.g., for gpu 0: [0, 2048] for the 1st grad accum mini-step, [11264, 10240] for the 2nd grad accum mini-step, etc.
            #       for gpu 1: [1024, 5120] for the 1st grad accum mini-step, [6144, 8192] for the 2nd grad accum mini-step, etc.
            batch_starts = self.gpu_x_seq_starts[i:i + self.gpu_batch_size]
            # e.g., for gpu 0: [0-1023, 2048-3071] for the 1st grad accum mini-step
            #       for gpu 1: [1024-2047, 5120-6143] for the 1st grad accum mini-step

        x = torch.stack([self.tokens[start:start + self.seq_len] for start in batch_starts])
        y = torch.stack([self.tokens[start + 1:start + self.seq_len + 1] for start in batch_starts])

        # optional doc ids (per-token doc index within the window)
        doc_ids = None
        if self.return_document_ids:
            is_eos = (x == self.eos_token_id)
            doc_ids = torch.cumsum(is_eos.to(torch.int32), dim=1)
            doc_ids = doc_ids - is_eos.to(torch.int32)

        # a full mini-step is to be done after this
        self.grad_accum_mini_steps_per_shard_counter += 1
        return x, y, doc_ids
