"""
Custom HuggingFace Trainer with GradientMaskOptimizer integration.
Saves training metrics and mask stats to JSONL for paper analysis.
"""

from __future__ import annotations

import json, os, time
import torch
from transformers import Trainer, TrainingArguments
from transformers.trainer import has_length, is_sagemaker_mp_enabled
from typing import Dict, Optional
import warnings

from .optimizer import GradientMaskOptimizer


class LAMoGMTTrainer(Trainer):

    def __init__(
        self,
        mask_method: str = "none",
        mask_global_ratio: float = 0.3,
        mask_alpha: float = 1.0,
        mask_beta1: float = 0.9,
        mask_min_ratio: float = 0.05,
        mask_max_ratio: float = 0.95,
        mask_warmup_steps: int = 0,
        dgmm_config: dict = None,
        *args,
        **kwargs,
    ):
        self.mask_method = mask_method
        self.mask_global_ratio = mask_global_ratio
        self.mask_alpha = mask_alpha
        self.mask_beta1 = mask_beta1
        self.mask_min_ratio = mask_min_ratio
        self.mask_max_ratio = mask_max_ratio
        self.mask_warmup_steps = mask_warmup_steps
        self.dgmm_config = dgmm_config or {}
        self._mask_opt: Optional[GradientMaskOptimizer] = None
        self._log_file = None
        self._start_time = time.time()
        self._peak_memory = 0

        super().__init__(*args, **kwargs)

    def create_optimizer(self):
        base_optimizer = super().create_optimizer()
        if self.mask_method == "none":
            return base_optimizer

        named_params = list(self.model.named_parameters())
        named_params = [(n, p) for n, p in named_params if p.requires_grad]
        self._mask_opt = GradientMaskOptimizer(
            base_optimizer=base_optimizer,
            named_params=named_params,
            method=self.mask_method,
            global_ratio=self.mask_global_ratio,
            alpha=self.mask_alpha,
            beta1=self.mask_beta1,
            min_mask_ratio=self.mask_min_ratio,
            max_mask_ratio=self.mask_max_ratio,
            warmup_steps=self.mask_warmup_steps,
            dgmm_config=self.dgmm_config,
        )
        # CRITICAL: replace self.optimizer so Trainer uses our wrapper
        self.optimizer = self._mask_opt
        mem = self._mask_opt.estimate_memory()
        if mem["importance_ema_gb"] > 0:
            print(f"[LA-Mo-GMT] EMA buffers: {mem['importance_ema_gb']:.2f} GB")
        return self._mask_opt

    def _init_log_file(self):
        if self._log_file is None and self.args.output_dir:
            os.makedirs(self.args.output_dir, exist_ok=True)
            path = os.path.join(self.args.output_dir, "training_log.jsonl")
            self._log_file = open(path, "a")

    def log(self, logs: Dict[str, float]) -> None:
        self._init_log_file()

        if self._mask_opt is not None:
            stats = self._mask_opt.get_last_stats()
            for k, v in stats.items():
                if isinstance(v, float):
                    logs[f"mask/{k}"] = v

        # Track peak GPU memory
        if torch.cuda.is_available():
            mem = torch.cuda.max_memory_allocated() / 1e9
            if mem > self._peak_memory:
                self._peak_memory = mem
            logs["gpu_memory_gb"] = mem

        # Save to JSONL
        if self._log_file is not None:
            # Filter out non-serializable values
            clean = {}
            for k, v in logs.items():
                if isinstance(v, (int, float, str, bool)):
                    clean[k] = v
                elif v is None:
                    clean[k] = None
            clean["_timestamp"] = time.time()
            self._log_file.write(json.dumps(clean) + "\n")
            self._log_file.flush()

        super().log(logs)

    def get_summary(self) -> Dict:
        return {
            "method": self.mask_method,
            "global_ratio": self.mask_global_ratio,
            "alpha": self.mask_alpha,
            "total_steps": self.state.global_step,
            "total_time_hours": (time.time() - self._start_time) / 3600,
            "peak_gpu_memory_gb": self._peak_memory,
            "best_loss": self.state.best_metric,
        }

    def __del__(self):
        if self._log_file is not None:
            self._log_file.close()
