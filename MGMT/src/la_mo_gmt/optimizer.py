"""
GradientMaskOptimizer: wraps any PyTorch optimizer to apply gradient masking
before each step(). Supports GMT, Mo-GMT, LA-GMT, LA-Mo-GMT, and RMT.
"""

from __future__ import annotations

import torch
from typing import Dict, List, Optional, Tuple

from .masking import apply_gradient_mask


class GradientMaskOptimizer(torch.optim.Optimizer):
    """
    Optimizer wrapper that applies gradient masking before each step.

    Inherits from torch.optim.Optimizer so HuggingFace Trainer accepts it.
    Delegates ALL attribute access to base_optimizer except our own logic.
    """

    def __init__(
        self,
        base_optimizer: torch.optim.Optimizer,
        named_params: List[Tuple[str, torch.nn.Parameter]],
        method: str = "gmt",
        global_ratio: float = 0.3,
        alpha: float = 1.0,
        beta1: float = 0.9,
        min_mask_ratio: float = 0.05,
        max_mask_ratio: float = 0.95,
        warmup_steps: int = 0,
        dgmm_config: dict = None,
    ):
        self.base_optimizer = base_optimizer
        self.named_params = named_params
        self.method = method
        self.global_ratio = global_ratio
        self.alpha = alpha
        self.beta1 = beta1
        self.min_mask_ratio = min_mask_ratio
        self.max_mask_ratio = max_mask_ratio
        self.warmup_steps = warmup_steps
        self.dgmm_config = dgmm_config or {}

        # Per-parameter FP32 EMA buffers for momentum-based methods
        # Keyed by id(param), only created when needed
        self.importance_m: Dict[int, torch.Tensor] = {}
        self._step_count = 0

    def step(self, closure=None) -> Optional[float]:
        """
        Apply gradient masking, then delegate to base optimizer's step().
        """
        self._step_count += 1

        # Apply gradient masking
        stats = apply_gradient_mask(
            named_params=self.named_params,
            importance_m=self.importance_m,
            method=self.method,
            global_ratio=self.global_ratio,
            alpha=self.alpha,
            beta1=self.beta1,
            min_mask_ratio=self.min_mask_ratio,
            max_mask_ratio=self.max_mask_ratio,
            update_importance=True,
            step=self._step_count,
            warmup_steps=self.warmup_steps,
            dgmm_config=self.dgmm_config,
        )
        self._last_stats = stats

        return self.base_optimizer.step(closure)

    def zero_grad(self, set_to_none: bool = False):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    @property
    def param_groups(self):
        return self.base_optimizer.param_groups

    @param_groups.setter
    def param_groups(self, value):
        self.base_optimizer.param_groups = value

    @property
    def state(self):
        return self.base_optimizer.state

    @state.setter
    def state(self, value):
        self.base_optimizer.state = value

    @property
    def defaults(self):
        return self.base_optimizer.defaults

    @defaults.setter
    def defaults(self, value):
        self.base_optimizer.defaults = value

    def state_dict(self):
        sd = self.base_optimizer.state_dict()
        sd["_mask_step_count"] = self._step_count
        if self.importance_m:
            name_to_pid = {}
            for name, p in self.named_params:
                pid = id(p)
                if pid in self.importance_m:
                    name_to_pid[name] = self.importance_m[pid].clone()
            sd["_mask_importance_m"] = name_to_pid
        return sd

    def load_state_dict(self, state_dict):
        if "_mask_step_count" in state_dict:
            self._step_count = state_dict.pop("_mask_step_count")
        if "_mask_importance_m" in state_dict:
            saved_ema = state_dict.pop("_mask_importance_m")
            # Determine target device from named_params
            target_device = None
            for _, p in self.named_params:
                if p.device.type == 'cuda':
                    target_device = p.device
                    break
            for name, p in self.named_params:
                if name in saved_ema:
                    ema = saved_ema[name]
                    if target_device is not None and ema.device != target_device:
                        ema = ema.to(target_device)
                    self.importance_m[id(p)] = ema
        self.base_optimizer.load_state_dict(state_dict)

    def __getattr__(self, name):
        """Delegate any unknown attribute to base_optimizer."""
        return getattr(self.base_optimizer, name)

    @torch.no_grad()
    def estimate_memory(self) -> Dict[str, float]:
        """Return estimated extra memory usage in GB."""
        total_bytes = 0
        for pid, buf in self.importance_m.items():
            total_bytes += buf.numel() * buf.element_size()
        return {
            "importance_ema_gb": total_bytes / 1e9,
            "num_buffers": len(self.importance_m),
        }

    def get_last_stats(self) -> Dict[str, float]:
        return getattr(self, "_last_stats", {})
