# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os


os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import json
import logging

import hydra
import torch
import torch.distributed as dist
from accelerate import init_empty_weights
from omegaconf import OmegaConf
from tensordict import TensorDict
from torch import nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import CPUOffload, FullStateDictConfig, MixedPrecision, ShardingStrategy, StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel

from verl.utils.data_processor import build_tokenizer
from verl.utils.dataset import sft_dataset
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.distributed import initialize_global_process_group
from verl.utils.fsdp_utils import get_fsdp_wrap_policy
from verl.utils.torch_functional import get_cosine_schedule_with_warmup
from verl.utils.tracking import Tracking


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_SFT_LOGGING_LEVEL", "WARN"))


class FSDPSFTTrainer:
    def __init__(self, config, device_mesh: DeviceMesh):
        self.config = config
        self.device_mesh = device_mesh
        self.tokenizer = build_tokenizer(self.config.model.model_path)
        self._normalize_config_bsz()
        self._build_dataloader()
        self._build_model_optimizer()
        dist.barrier()
        if self.device_mesh.get_rank() == 0:
            print(json.dumps(OmegaConf.to_container(self.config), indent=2, ensure_ascii=False))

    def _normalize_config_bsz(self):
        dp_size = self.device_mesh.size()
        assert self.config.data.total_batch_size % dp_size == 0
        assert self.config.data.micro_batch_size % dp_size == 0
        self.config.data.total_batch_size //= dp_size
        self.config.data.micro_batch_size //= dp_size
        if self.device_mesh.get_rank() == 0:
            print(f"Reduce total batch size to {self.config.data.total_batch_size}.")
            print(f"Reduce micro batch size to {self.config.data.micro_batch_size}.")

    def _build_dataloader(self):
        if self.config.data.train_dataset == "gsm8k":
            self.train_dataset = sft_dataset.GSM8KDataset(
                self.config.model.model_path, self.config.data.max_seq_len, self.config.data.truncation, "train"
            )
        elif self.config.data.train_dataset == "openo1":
            self.train_dataset = sft_dataset.OpenO1Dataset(
                self.config.model.model_path, self.config.data.max_seq_len, self.config.data.truncation, "train"
            )
        else:
            raise NotImplementedError(f"{self.config.data.train_dataset} was not found.")

        if self.config.data.val_dataset is None:
            self.val_dataset = None
        elif self.config.data.val_dataset == "gsm8k":
            self.val_dataset = sft_dataset.GSM8KDataset(
                self.config.model.model_path, self.config.data.max_seq_len, self.config.data.truncation, "test"
            )
        else:
            raise NotImplementedError(f"{self.config.data.val_dataset} was not found.")

        # build dataloader
        rank = self.device_mesh.get_rank()
        world_size = self.device_mesh.size()
        self.train_sampler = DistributedSampler(
            self.train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
        )
        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.total_batch_size,
            sampler=self.train_sampler,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
        )
        if self.val_dataset is not None:
            self.val_sampler = DistributedSampler(
                self.val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=True
            )
            self.val_dataloader = DataLoader(
                dataset=self.val_dataset,
                batch_size=self.config.data.micro_batch_size,
                sampler=self.val_sampler,
                num_workers=8,
                pin_memory=True,
                drop_last=True,
            )

    def _build_model_optimizer(self):
        log_gpu_memory_usage("Before model allocation", logger=logger)
        model_config = AutoConfig.from_pretrained(self.config.model.model_path, trust_remote_code=True)

        fsdp_kwargs = {}
        if self.config.model.fsdp_config.sync_module_states and not model_config.tie_word_embeddings:
            fsdp_kwargs["sync_module_states"] = True
            if self.device_mesh.get_rank() == 0:
                fsdp_kwargs["param_init_fn"] = None
                empty_init = False
            else:
                fsdp_kwargs["param_init_fn"] = lambda module: module.to_empty(device="cuda", recurse=False)
                empty_init = True
        else:
            empty_init = False

        if empty_init:
            with init_empty_weights():
                self.model = AutoModelForCausalLM.from_config(
                    model_config,
                    torch_dtype=torch.float32,
                    attn_implementation="flash_attention_2",
                    trust_remote_code=True,
                )
        else:
            self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
                self.config.model.model_path,
                config=model_config,
                torch_dtype=torch.float32,
                attn_implementation="flash_attention_2",
                trust_remote_code=True,
            )

        if self.config.model.enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        log_gpu_memory_usage("After model allocation", logger=logger)
        fsdp_kwargs["mixed_precision"] = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        )
        fsdp_kwargs["auto_wrap_policy"] = get_fsdp_wrap_policy(
            self.model, wrap_policy=self.config.model.fsdp_config.wrap_policy
        )
        if self.config.model.fsdp_config.cpu_offload:
            fsdp_kwargs["cpu_offload"] = CPUOffload(offload_params=self.config.model.fsdp_config.offload_params)

        self.fsdp_model = FSDP(
            module=self.model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            device_id=torch.cuda.current_device(),
            use_orig_params=False,
            device_mesh=self.device_mesh,
            **fsdp_kwargs,
        )
        log_gpu_memory_usage("After FSDP wrapping", logger=logger)

        self.optimizer = AdamW(
            self.fsdp_model.parameters(),
            lr=self.config.optim.lr,
            betas=self.config.optim.betas,
            weight_decay=self.config.optim.weight_decay,
        )
        log_gpu_memory_usage("After initialize optimizer", logger=logger)

        steps_per_epoch = len(self.train_dataloader)
        total_steps = steps_per_epoch * self.config.trainer.total_epochs
        num_warmup_steps = int(total_steps * self.config.optim.warmup_steps_ratio)
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps
        )

    def _compute_loss(self, batch: TensorDict):
        loss_mask = batch.pop("loss_mask").cuda()
        labels = batch["input_ids"].clone().cuda()
        logits: torch.Tensor = self.fsdp_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            position_ids=batch["position_ids"],
            use_cache=False,
        ).logits
        # Upcast to fp32 to avoid overflow
        logits = logits.float()  # TODO: use fused ce
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_mask = loss_mask[..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        # Flatten the tokens
        shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        loss = loss_fct(shift_logits, shift_labels)
        loss = loss * loss_mask

        valid_tokens = loss_mask.sum()
        if self.config.data.dp_loss_balancing:
            dist.all_reduce(valid_tokens)  # becomes total valid tokens in all ranks
            valid_tokens = valid_tokens / dist.get_world_size()

        loss = loss.sum() / valid_tokens
        return loss

    def training_step(self, batch: TensorDict):
        self.fsdp_model.train()

        log_gpu_memory_usage("Before optimizer zero_grad", logger=logger)
        self.optimizer.zero_grad()
        log_gpu_memory_usage("After optimizer zero_grad", logger=logger)

        micro_batches = batch.split(self.config.data.micro_batch_size)
        n_micro_batches = len(micro_batches)
        step_loss = 0
        for micro_batch in micro_batches:
            loss = self._compute_loss(batch=micro_batch) / n_micro_batches
            loss.backward()
            step_loss += loss.item()

        self.fsdp_model.clip_grad_norm_(max_norm=self.config.optim.clip_grad)

        log_gpu_memory_usage("Before optimizer step", logger=logger)
        self.optimizer.step()
        self.lr_scheduler.step()
        log_gpu_memory_usage("After optimizer step", logger=logger)

        step_loss = torch.tensor(step_loss).cuda()
        lr = self.lr_scheduler.get_last_lr()[0]
        dist.all_reduce(step_loss, op=dist.ReduceOp.AVG)
        return {"train/loss": step_loss.detach().item(), "train/lr": lr}

    def validation_step(self, batch: TensorDict):
        self.fsdp_model.eval()
        with torch.no_grad():
            loss = self._compute_loss(batch)
            dist.all_reduce(loss, op=dist.ReduceOp.AVG)

        return loss

    def save_checkpoint(self, step):
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.fsdp_model, StateDictType.FULL_STATE_DICT, cfg):
            state_dict = self.fsdp_model.state_dict()

        for name, param in state_dict.items():
            state_dict[name] = param.to(torch.bfloat16)

        path = os.path.join(self.config.trainer.save_checkpoint_path, f"global_step_{step}")
        if self.device_mesh.get_rank() == 0:
            self.model.save_pretrained(path, state_dict=state_dict)
            self.tokenizer.save_pretrained(path)

        dist.barrier()

    def fit(self):
        rank = self.device_mesh.get_rank()
        if rank == 0:
            tracking = Tracking(
                project_name=self.config.trainer.project_name,
                experiment_name=self.config.trainer.experiment_name,
            )

        # TODO: support self.config.trainer.load_checkpoint_path
        global_step = 0
        for epoch in range(self.config.trainer.total_epochs):
            self.train_sampler.set_epoch(epoch=epoch)
            for data in tqdm(
                self.train_dataloader,
                desc=f"Train {epoch + 1}/{self.config.trainer.total_epochs}",
                disable=(rank != 0),
            ):
                if global_step == 0:
                    for key, value in data.items():
                        print(f"[rank {rank}]: {key}'s shape: {value.shape}, device: {value.device}, {value}")

                data = TensorDict(data, batch_size=self.config.data.total_batch_size).cuda()
                metric = self.training_step(data)
                if rank == 0:
                    tqdm.write(f"Train loss: {metric['train/loss']:.4f}, lr: {metric['train/lr']:.2e}")
                    tracking.log(data=metric, step=global_step)

                global_step += 1

            if self.val_dataset is not None:
                self.val_sampler.set_epoch(epoch=epoch)
                val_losses = []
                for data in tqdm(
                    self.val_dataloader,
                    desc=f"Eval {epoch + 1}/{self.config.trainer.total_epochs}",
                    disable=(rank != 0),
                ):
                    data = TensorDict(data, batch_size=self.config.data.micro_batch_size).cuda()
                    val_loss = self.validation_step(data)
                    val_losses.append(val_loss)

                if rank == 0:
                    val_loss = torch.mean(torch.stack(val_losses)).item()
                    print(f"Eval loss: {val_loss:.4f}")
                    tracking.log(data={"val/loss": val_loss}, step=global_step)

            dist.barrier()
            # save checkpoint
            self.save_checkpoint(step=global_step)


@hydra.main(config_path="config", config_name="sft_trainer", version_base=None)
def main(config):
    _, _, world_size = initialize_global_process_group()
    device_mesh = init_device_mesh(device_type="cuda", mesh_shape=(world_size,), mesh_dim_names=("dp",))
    trainer = FSDPSFTTrainer(config=config, device_mesh=device_mesh)
    trainer.fit()


if __name__ == "__main__":
    main()
