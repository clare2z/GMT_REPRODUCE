"""
Gradient masking strategies for sparse LLM fine-tuning.

Methods:
  - gmt:        GMT (AAAI-25) — mask by |grad|
  - mo_gmt:     Mo-GMT — mask by |EMA of grad| (momentum-guided)
  - la_gmt:     LA-GMT — per-layer adaptive ratios from |grad|
  - la_mo_gmt:  LA-Mo-GMT — per-layer adaptive ratios from |EMA of grad|
  - rmt:        Random Mask Tuning (baseline)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


def get_layer_groups(
    model: nn.Module,
    named_parameters: List[Tuple[str, nn.Parameter]],
) -> Dict[int, List[nn.Parameter]]:
    """
    Group parameters by transformer layer index.

    For a HuggingFace model like LlamaForCausalLM:
      - model.layers.0.self_attn.q_proj.weight  -> layer 0
      - model.layers.5.mlp.down_proj.weight     -> layer 5
      - model.embed_tokens.weight               -> layer -1 (embedding)
      - lm_head.weight                           -> layer -2 (head)

    Returns dict mapping layer_idx -> list of parameters.
    """
    layer_groups: Dict[int, List[nn.Parameter]] = defaultdict(list)

    for name, param in named_parameters:
        if not param.requires_grad:
            continue

        # Try to extract transformer layer index
        layer_idx = _extract_layer_idx(name)
        layer_groups[layer_idx].append(param)

    return dict(layer_groups)


def _extract_layer_idx(name: str) -> int:
    """Extract transformer layer index from parameter name."""
    # Common patterns in HF models:
    #   model.layers.0.xxx   (Llama, Mistral, etc.)
    #   transformer.h.0.xxx  (GPT-2 style)
    #   decoder.layers.0.xxx (T5 decoder)

    parts = name.split(".")
    for i, p in enumerate(parts):
        if p in ("layers", "h", "layer", "blocks"):
            if i + 1 < len(parts) and parts[i + 1].isdigit():
                return int(parts[i + 1])
        if p.isdigit() and i > 0 and parts[i - 1] in ("layers", "h", "layer", "blocks"):
            return int(p)

    # Fallback: put all non-layer params in group -1
    return -1



def _compute_global_threshold(active_params, importance_src, global_ratio, max_samples=5000000):
    """Compute a single global threshold using sampled k-th value.

    importance_src: callable, takes a param, returns the importance tensor (|grad| or EMA).
    max_samples: maximum number of elements to sample for threshold estimation.
    Returns a scalar threshold; |importance| >= threshold are kept.
    """

    # Collect sampled |importance| values, proportional to param size
    total_elements = sum(p.numel() for _, p in active_params)
    scale = min(1.0, max_samples / max(total_elements, 1))

    samples = []
    for name, p in active_params:
        imp = importance_src(name, p)
        n = imp.numel()
        if scale < 1.0:
            n_sample = max(100, int(n * scale))
            idx = torch.randint(0, n, (n_sample,), device=imp.device)
            samples.append(imp.abs().flatten()[idx].float())
        else:
            samples.append(imp.abs().flatten().float())

    all_vals = torch.cat(samples)
    k = max(1, int(all_vals.numel() * global_ratio))  # number to MASK
    threshold = torch.kthvalue(all_vals, k).values.item()
    return threshold

def apply_gradient_mask(
    named_params: List[Tuple[str, nn.Parameter]],
    importance_m: Dict[int, torch.Tensor],
    method: str = "gmt",
    global_ratio: float = 0.3,
    alpha: float = 1.0,
    beta1: float = 0.9,
    min_mask_ratio: float = 0.05,
    max_mask_ratio: float = 0.95,
    update_importance: bool = True,
    step: int = 0,
    warmup_steps: int = 0,
) -> Dict[str, float]:
    """
    Apply gradient masking to model parameters in-place.

    Args:
        named_params: List of (name, parameter) tuples.
        importance_m: Dict mapping id(param) -> FP32 EMA tensor (for momentum methods).
            FP32 is essential — BF16 underflows small gradients, destroying importance ranking.
        method: One of 'none', 'gmt', 'mo_gmt', 'la_gmt', 'la_mo_gmt',
            'gmt_global', 'mo_gmt_global', 'rmt'.
        global_ratio: Fraction of gradients to MASK (0.0 = no masking, 0.99 = 1% updated).
        alpha: Concentration for layer-adaptive ratio allocation (>0).
        beta1: EMA decay for momentum importance estimation.
        min_mask_ratio, max_mask_ratio: Clip bounds for per-layer mask ratios.
        update_importance: Whether to update the EMA buffers.
        step: Current training step (for random seed in RMT).
        warmup_steps: Number of initial steps where EMA is accumulated but no mask is applied.

    Returns:
        Stats dict with per-layer mask ratios for logging.
    """
    if method == "none" or global_ratio <= 0.0:
        return {"global_ratio": 0.0}

    # Filter to params with gradients and requires_grad
    active_params = [(n, p) for n, p in named_params if p.grad is not None and p.requires_grad]
    if not active_params:
        return {"global_ratio": 0.0}

    # ── Update importance EMA buffers ──
    if method in ("mo_gmt", "la_mo_gmt", "mo_gmt_global", "mhsgm") and update_importance:
        with torch.no_grad():
            for _, p in active_params:
                pid = id(p)
                if pid not in importance_m:
                    importance_m[pid] = torch.zeros_like(p.grad, dtype=torch.bfloat16)
                importance_m[pid].mul_(beta1).add_(p.grad, alpha=1.0 - beta1)

    # ── Warmup: accumulate EMA only, skip masking ──
    if warmup_steps > 0 and step < warmup_steps:
        return {"global_ratio": 0.0, "warmup": True, "step": step}

    # ── Layer-adaptive ratio allocation ──
    layer_ratios: Dict[int, float] = {}
    if method in ("la_gmt", "la_mo_gmt"):
        layer_ratios = _compute_layer_ratios(
            active_params, importance_m, method, global_ratio, alpha,
            min_mask_ratio, max_mask_ratio,
        )

    # ── Apply per-parameter masking ──
    first_grad = active_params[0][1].grad
    total_masked = torch.tensor(0, device=first_grad.device, dtype=torch.int64)

    # Pre-compute layer statistics for MHSGM (per-layer mean + global mean)
    layer_stats = {}
    global_mean = 0.0
    if method == "mhsgm":
        layer_ex = defaultdict(float)
        layer_n = defaultdict(int)
        global_sum = 0.0
        global_n = 0
        for name, p in active_params:
            lidx = _extract_layer_idx(name)
            pid = id(p)
            imp = importance_m.get(pid, p.grad.abs()).abs()
            s = imp.sum().item()
            layer_ex[lidx] += s
            layer_n[lidx] += imp.numel()
            global_sum += s
            global_n += imp.numel()
        layer_stats = {l: layer_ex[l] / max(layer_n[l], 1) for l in layer_ex}
        global_mean = global_sum / max(global_n, 1)

    # Global threshold methods: compute one threshold for all parameters
    global_threshold = None
    if method in ("gmt_global", "mo_gmt_global", "mhsgm"):
        def _imp_src(name, p):
            if method == "gmt_global":
                return p.grad
            if method == "mhsgm":
                lidx = _extract_layer_idx(name)
                pid = id(p)
                imp = importance_m.get(pid, p.grad.abs()).abs()
                mu_l = layer_stats[lidx]
                R = imp / (mu_l + 1e-8)
                G = imp / (global_mean + 1e-8)
                I = (mu_l / (global_mean + 1e-8)) ** max(alpha, 0.0)
                return (alpha * R + (1.0 - alpha) * G) * I
            return importance_m.get(id(p), p.grad)

        global_threshold = _compute_global_threshold(
            active_params, _imp_src, global_ratio,
        )
    total_elements = 0

    with torch.no_grad():
        for name, p in active_params:
            grad = p.grad
            numel = grad.numel()
            total_elements += numel

            layer_idx = _extract_layer_idx(name)
            ratio = layer_ratios.get(layer_idx, global_ratio)

            if method == "rmt":
                # Random mask: randomly zero out `ratio` fraction of gradients
                g = torch.rand(numel, device=grad.device)
                threshold = torch.quantile(g, ratio)
                mask = (g >= threshold).view_as(grad)
            else:
                # Get importance scores
                if method == "mhsgm":
                    pid = id(p)
                    imp = importance_m.get(pid, p.grad.abs()).abs()
                    lidx = _extract_layer_idx(name)
                    mu_l = layer_stats[lidx]
                    R = imp / (mu_l + 1e-8)
                    G = imp / (global_mean + 1e-8)
                    I = (mu_l / (global_mean + 1e-8)) ** max(alpha, 0.0)
                    scores = ((alpha * R + (1.0 - alpha) * G) * I).float().view(-1)
                elif method in ("mo_gmt", "la_mo_gmt", "mo_gmt_global"):
                    pid = id(p)
                    scores = importance_m[pid].abs().float().view(-1)
                else:
                    # gmt, la_gmt, gmt_global: use raw gradient magnitude
                    scores = grad.abs().view(-1)

                # Compute threshold
                if method in ("gmt_global", "mo_gmt_global", "mhsgm"):
                    # Global threshold: one value for all parameters
                    mask = (scores >= global_threshold).view_as(grad)
                else:
                    # Per-parameter top-K fraction
                    k = max(1, int(numel * (1.0 - ratio)))
                    threshold = torch.sort(scores).values[numel - k]
                    mask = (scores >= threshold).view_as(grad)

            # Count masked elements (keep on GPU, sync once at end)
            total_masked += (mask == 0).sum()

            # Apply mask (zero out masked gradients)
            grad.mul_(mask)

    stats = {
        "global_ratio": global_ratio,
        "effective_ratio": total_masked.item() / max(total_elements, 1),
    }
    # Add per-layer ratios if available
    for lidx, r in layer_ratios.items():
        stats[f"layer_{lidx}_ratio"] = r


    return stats


def _compute_layer_ratios(
    active_params: List[Tuple[str, nn.Parameter]],
    importance_m: Dict[int, torch.Tensor],
    method: str,
    global_ratio: float,
    alpha: float,
    min_ratio: float,
    max_ratio: float,
) -> Dict[int, float]:
    """
    Compute per-layer mask ratios based on layer importance.

    Important layers (higher mean importance) -> lower mask ratio -> more updates.
    Less important layers -> higher mask ratio -> fewer updates.

    Uses iterative water-filling: allocate budget proportional to importance^alpha,
    clip to [min_ratio, max_ratio] per-layer bounds, redistribute excess.
    Ensures that sum (1 - r_l) * |P_l| = (1 - global_ratio) * |P_all|.
    """
    # Group by layer, compute importance
    layer_params: Dict[int, List[nn.Parameter]] = defaultdict(list)
    for name, p in active_params:
        layer_params[_extract_layer_idx(name)].append(p)

    layer_importance: Dict[int, float] = {}
    for lidx, params in layer_params.items():
        device = params[0].grad.device
        total_sum = torch.tensor(0.0, device=device, dtype=torch.float64)
        total_n = 0
        for p in params:
            pid = id(p)
            if method == "la_mo_gmt" and pid in importance_m:
                total_sum += importance_m[pid].abs().sum()
            else:
                total_sum += p.grad.abs().sum()
            total_n += p.numel()
        layer_importance[lidx] = total_sum.item() / max(total_n, 1)

    # Normalize: weight proportional to importance^alpha
    imps = torch.tensor(list(layer_importance.values()))
    if imps.sum() == 0 or alpha == 0:
        return {}  # Fallback to uniform

    weights = imps ** alpha
    weights = weights / weights.sum()

    layer_indices = list(layer_importance.keys())

    # Global floor: all layers at least mask global_ratio * 2/3
    global_floor = global_ratio * 2.0 / 3.0
    per_layer_min: Dict[int, float] = {}
    for lidx in layer_indices:
        if lidx == -1:
            per_layer_min[lidx] = 0.0  # embedding: no floor
        else:
            per_layer_min[lidx] = global_floor  # uniform floor for all transformer layers

    # Per-layer param counts and budget bounds
    layer_n: Dict[int, int] = {}
    budget_min: Dict[int, float] = {}
    budget_max: Dict[int, float] = {}
    for lidx in layer_indices:
        n = sum(p.numel() for p in layer_params[lidx])
        layer_n[lidx] = n
        budget_min[lidx] = max(1.0, n * (1.0 - max_ratio))
        budget_max[lidx] = n * (1.0 - per_layer_min[lidx])

    total_params = sum(p.numel() for _, p in active_params)
    total_update_budget = total_params * (1.0 - global_ratio)

    # Initial allocation proportional to importance weights
    budget: Dict[int, float] = {}
    for i, lidx in enumerate(layer_indices):
        budget[lidx] = weights[i].item() * total_update_budget

    # Iterative water-filling: clip and redistribute excess
    n_layers = len(layer_indices)
    frozen = set()
    for _ in range(20):
        excess = 0.0
        unclipped_weight_sum = 0.0

        for i, lidx in enumerate(layer_indices):
            if lidx in frozen:
                continue
            lo = budget_min[lidx]
            hi = budget_max[lidx]
            if budget[lidx] <= lo:
                excess += lo - budget[lidx]
                budget[lidx] = lo
                frozen.add(lidx)
            elif budget[lidx] >= hi:
                excess += budget[lidx] - hi
                budget[lidx] = hi
                frozen.add(lidx)
            else:
                unclipped_weight_sum += weights[i].item()

        if len(frozen) == n_layers or abs(excess) < 1.0:
            break

        if unclipped_weight_sum > 0:
            for i, lidx in enumerate(layer_indices):
                if lidx not in frozen:
                    budget[lidx] += excess * weights[i].item() / unclipped_weight_sum

    # Convert budgets to ratios (per-layer min/max clamping)
    layer_ratios: Dict[int, float] = {}
    for lidx in layer_indices:
        ratio = 1.0 - budget[lidx] / layer_n[lidx]
        ratio = max(per_layer_min[lidx], min(max_ratio, ratio))
        layer_ratios[lidx] = ratio

    return layer_ratios


















