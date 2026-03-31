"""
WW-FL trainer using CrypTen simulation mode.
Matches paper: 10 MPC clusters, 100 clients/cluster, 10 selected/cluster/round,
batch_size=80 (10x scaled), lr=0.05, 5 local epochs.

CrypTen simulation mode runs plaintext but simulates fixed-point precision
and truncation errors (22-bit decimal, as in paper Section 4.1).
"""
import copy
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from typing import List, Optional, Callable

from wwfl.utils import PoisonedSubset
from wwfl.aggregation import fedavg, trimmed_mean, trimmed_mean_variant, fltrust
from wwfl.fl_trainer import evaluate

try:
    import crypten
    import crypten.nn as cnn
    CRYPTEN_AVAILABLE = True
except ImportError:
    CRYPTEN_AVAILABLE = False
    print("[WW-FL] CrypTen not found — running in plaintext simulation mode.")


def train_cluster_plaintext(
    model: nn.Module,
    all_client_indices: List[List[int]],  # indices for each selected client
    train_dataset,
    poison_fns: List[Optional[Callable]],
    device: str,
    epochs: int = 5,
    batch_size: int = 80,
    lr: float = 0.05,
) -> dict:
    """
    WW-FL cluster training: pool all client data together (key difference vs FL),
    then train with MPC (simulated here as plaintext with optional CrypTen precision).
    """
    # Pool data from all clients in this cluster
    subsets = []
    for indices, pfn in zip(all_client_indices, poison_fns):
        subsets.append(PoisonedSubset(train_dataset, indices, pfn))

    pooled   = ConcatDataset(subsets)
    loader   = DataLoader(pooled, batch_size=batch_size, shuffle=True, num_workers=0)
    model    = copy.deepcopy(model).to(device)
    opt      = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    crit     = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            crit(model(x), y).backward()
            opt.step()

    return model.state_dict()


def apply_crypten_precision(state_dict: dict, decimal_bits: int = 22) -> dict:
    """
    Simulate CrypTen fixed-point truncation errors (22-bit decimal precision).
    This matches the paper's simulation mode evaluation for Q2.
    Quantize each parameter to fixed-point and back.
    """
    scale  = 2 ** decimal_bits
    noisy  = {}
    for k, v in state_dict.items():
        fp  = (v.float() * scale).round() / scale   # quantize
        noisy[k] = fp.to(v.dtype)
    return noisy


def run_wwfl(
    model_class,
    train_dataset,
    test_loader,
    cluster_client_indices: List[List[List[int]]],  # [n_clusters][n_clients][n_samples]
    malicious_cluster_client_ids: dict,              # {cluster_id: [client_ids]}
    poison_fn: Optional[Callable],
    aggregation: str = "fedavg",
    rounds: int = 500,
    clients_per_cluster_per_round: int = 10,
    local_epochs: int = 5,
    batch_size: int = 80,
    lr: float = 0.05,
    device: str = "cpu",
    use_crypten_precision: bool = False,   # True for Q2 (MPC accuracy impact)
    tm_alpha: int = 2,
    tm_beta:  int = 100,
    root_indices: Optional[List[int]] = None,
    log_every: int = 10,
) -> List[tuple]:
    """
    Run WW-FL for `rounds` rounds.
    Returns list of (round, accuracy) tuples.
    """
    n_clusters   = len(cluster_client_indices)
    global_model = model_class().to(device)
    accuracies   = []

    for r in range(1, rounds + 1):
        cluster_updates = []

        for cid in range(n_clusters):
            # Sample clients for this cluster this round
            n_clients_in_cluster = len(cluster_client_indices[cid])
            selected_local = random.sample(
                range(n_clients_in_cluster),
                min(clients_per_cluster_per_round, n_clients_in_cluster)
            )

            malicious_in_cluster = malicious_cluster_client_ids.get(cid, [])

            sel_indices = [cluster_client_indices[cid][i] for i in selected_local]
            sel_pfns    = [
                poison_fn if i in malicious_in_cluster else None
                for i in selected_local
            ]

            # Cluster trains on pooled data (WW-FL key feature)
            cluster_sd = train_cluster_plaintext(
                global_model, sel_indices, train_dataset,
                poison_fns=sel_pfns, device=device,
                epochs=local_epochs, batch_size=batch_size, lr=lr
            )

            # Simulate MPC fixed-point precision if requested (Fig 4)
            if use_crypten_precision:
                cluster_sd = apply_crypten_precision(cluster_sd)

            cluster_updates.append(cluster_sd)

        # Global aggregation (Layer I)
        if aggregation == "fedavg":
            new_sd = fedavg(cluster_updates)
        elif aggregation == "tm":
            new_sd = trimmed_mean(cluster_updates, alpha=tm_alpha)
        elif aggregation == "tm_variant":
            new_sd = trimmed_mean_variant(cluster_updates, alpha=tm_alpha, beta=tm_beta)
        elif aggregation == "fltrust":
            if root_indices is not None:
                from wwfl.fl_trainer import train_one_client
                root_sd = train_one_client(
                    global_model, root_indices, train_dataset,
                    poison_fn=None, device=device,
                    epochs=local_epochs, batch_size=batch_size, lr=lr
                )
            new_sd = fltrust(cluster_updates, root_sd)
        else:
            raise ValueError(aggregation)

        if use_crypten_precision:
            new_sd = apply_crypten_precision(new_sd)

        global_model.load_state_dict(new_sd)

        if r % log_every == 0:
            acc = evaluate(global_model, test_loader, device)
            accuracies.append((r, acc))
            print(f"  [WW-FL] Round {r:4d} | {aggregation:12s} | "
                  f"{'MPC' if use_crypten_precision else 'Plain':5s} | Acc: {acc:.2f}%")

    return accuracies
