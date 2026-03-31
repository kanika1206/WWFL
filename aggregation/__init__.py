"""
Aggregation schemes from Section 5.2 of WW-FL paper.
All run in plaintext (simulating MPC in CrypTen simulation mode).
"""
import torch
import torch.nn.functional as F
from typing import List, Optional
import copy


def fedavg(models: List[dict], weights: Optional[List[float]] = None) -> dict:
    """Weighted average of state dicts. Default: uniform."""
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    agg = {}
    for k in models[0]:
        agg[k] = sum(w * m[k].float() for w, m in zip(weights, models))
    return agg


def trimmed_mean(models: List[dict], alpha: int) -> dict:
    """
    Trimmed Mean [YCRB18]: for each coordinate, exclude top-alpha and
    bottom-alpha values across clients, then average the rest.
    Paper: alpha=2 for WW-FL (10 clusters), alpha=20 for FL (100 clients).
    """
    n = len(models)
    assert 2 * alpha < n, f"alpha={alpha} too large for {n} models"
    agg = {}
    for k in models[0]:
        stacked = torch.stack([m[k].float() for m in models], dim=0)  # [n, ...]
        flat    = stacked.view(n, -1)          # [n, params]
        sorted_flat, _ = torch.sort(flat, dim=0)
        trimmed = sorted_flat[alpha: n - alpha]  # [n-2alpha, params]
        agg[k]  = trimmed.mean(dim=0).view(models[0][k].shape).to(models[0][k].dtype)
    return agg


def trimmed_mean_variant(models: List[dict], alpha: int, beta: int) -> dict:
    """
    TM Variant (Algorithm 4 in paper):
    Sample `beta` random coordinates, run TM-List on those, find TopK outlier
    cluster IDs, exclude them globally, then FedAvg the rest.
    alpha = trim threshold, beta = sample size.
    Paper values from Tab 10/11: alpha=2, beta=10/100/1000.
    """
    n      = len(models)
    # Flatten all models
    flat   = []
    shapes = {}
    keys   = list(models[0].keys())
    for m in models:
        parts = []
        for k in keys:
            t = m[k].float().view(-1)
            shapes[k] = m[k].shape
            parts.append(t)
        flat.append(torch.cat(parts))

    total_params = flat[0].numel()
    beta = min(beta, total_params)

    # Sample random coordinate indices
    sample_idx = torch.randperm(total_params)[:beta]

    # Count how often each model appears as outlier in the sample
    outlier_counts = torch.zeros(n)
    for i in sample_idx:
        col = torch.tensor([flat[j][i].item() for j in range(n)])
        sorted_vals, sorted_idx = torch.sort(col)
        # bottom alpha and top alpha are outliers
        for idx in sorted_idx[:alpha].tolist() + sorted_idx[n - alpha:].tolist():
            outlier_counts[int(idx)] += 1

    # Exclude top-2alpha most frequent outliers
    _, top_outliers = torch.topk(outlier_counts, k=min(2 * alpha, n - 1))
    exclude_set     = set(top_outliers.tolist())
    benign_models   = [m for i, m in enumerate(models) if i not in exclude_set]

    if len(benign_models) == 0:
        benign_models = models  # fallback

    return fedavg(benign_models)


def fltrust(models: List[dict], root_model: dict) -> dict:
    """
    FLTrust [CFLG21]: weight each client by cosine similarity to root model
    (trained on clean root dataset). Negative similarities → 0.
    """
    def flatten(sd):
        return torch.cat([v.float().view(-1) for v in sd.values()])

    root_flat = flatten(root_model)
    weights   = []
    for m in models:
        m_flat = flatten(m)
        sim    = F.cosine_similarity(m_flat.unsqueeze(0), root_flat.unsqueeze(0)).item()
        weights.append(max(0.0, sim))

    total = sum(weights)
    if total == 0:
        weights = [1.0 / len(models)] * len(models)
    else:
        weights = [w / total for w in weights]

    return fedavg(models, weights)
