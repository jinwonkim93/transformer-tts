import logging
import os

import torch
from accelerate import Accelerator, skip_first_batches
from accelerate.utils import (
    AutocastKwargs,
    InitProcessGroupKwargs,
    TorchDynamoPlugin,
    set_seed,
)
from accelerate.utils.memory import release_memory
from torch.utils.data.dataloader import DataLoader


class Trainer:

    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        dataset_trn,
        dataset_val,
        collate_fn,
        config,
    ):

        self.model = model
        self.config = config

        self.dataset_trn = dataset_trn
        self.dataset_val = dataset_val

        if config.experiment.dtype == "float16":
            mixed_precision = "fp16"
            torch_dtype = torch.float16
        elif config.experiment.dtype == "bfloat16":
            mixed_precision = "bf16"
            torch_dtype = torch.bfloat16
        else:
            mixed_precision = "no"
            torch_dtype = torch.float32

        # A. Preparation
        kwargs_handlers = [InitProcessGroupKwargs(timeout=timedelta(minutes=120))]
        self.autocast_kwargs = AutocastKwargs(enabled=(mixed_precision != "fp16"))
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.experiment.gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            log_with=config.experiment.report_to,
            project_dir=config.experiment.output_dir,
            kwargs_handlers=kwargs_handlers,
        )

        self.accelerator.init_trackers(
            project_name=config.wandb.wandb_project,
            config={
                "learning_rate": config.experiment.learning_rate,
                "num_train_epochs": config.experiment.num_train_epochs,
                "gradient_accumulation_steps": config.experiment.gradient_accumulation_steps,
                "per_device_train_batch_size": config.experiment.per_device_train_batch_size,
                "global_batch_size": config.experiment.per_device_train_batch_size
                * self.accelerator.num_processes,
                "mixed_precision": mixed_precision,
                "lr_scheduler_type": config.experiment.lr_scheduler_type,
                "warmup_steps": config.experiment.warmup_steps,
                "weight_decay": config.experiment.weight_decay,
                "adam_beta1": config.experiment.adam_beta1,
                "adam_beta2": config.experiment.adam_beta2,
            },
            init_kwargs=(
                {"wandb": {"name": config.wandb.wandb_run_name}}
                if config.wandb.wandb_run_name
                else {}
            ),
        )

        # Prepare Dataset
        num_workers = 16

        self.loader_trn = DataLoader(
            self.dataset_trn,
            shuffle=False,
            pin_memory=True,
            batch_size=config.experiment.per_device_train_batch_size,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )

        self.loader_trn = Accelerator.prepare(self.loader_trn)

        self.loader_val = DataLoader(
            self.dataset_val,
            shuffle=False,
            pin_memory=True,
            batch_size=config.experiment.per_device_train_batch_size,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )

        self.loader_val = Accelerator.prepare(self.loader_val)

        # Prepare everything with accelerate
        self.model, self.optimizer, self.lr_scheduler = accelerator.prepare(
            self.model, self.optimizer, self.lr_scheduler
        )

    def train_one_epoch(self, optimizer=None, scheduler=None, epoch=0):
        raise NotImplementedError

    def train(self, max_epoch=100):
        raise NotImplementedError

    def eval(self, valid=True, epoch=0):
        raise NotImplementedError
