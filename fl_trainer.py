"""
Plain Federated Learning baseline (FedAvg).
Matches paper config: 1000 clients, 100 selected/round, 200 samples/client,
5 local epochs, batch_size=8, lr=0.005.
"""
import copy
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Optional, Callable

from wwfl.utils import PoisonedSubset
from wwfl.aggregation import fedavg, trimmed_mean, trimmed_mean_variant, fltrust


def train_one_client(
    model: nn.Module,
    indices: List[int],
    dataset,
    poison_fn: Optional[Callable],
    device: str,
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 0.005,
) -> dict:
    """Train model on one client's data, return updated state dict."""
    client_ds = PoisonedSubset(dataset, indices, poison_fn)
    loader    = DataLoader(client_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    model     = copy.deepcopy(model).to(device)
    opt       = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    crit      = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            crit(model(x), y).backward()
            opt.step()
    return model.state_dict()


def evaluate(model: nn.Module, test_loader: DataLoader, device: str) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total   += y.size(0)
    return 100.0 * correct / total


def run_fl(
    model_class,
    train_dataset,
    test_loader,
    client_indices: List[List[int]],   # [n_clients][n_samples]
    malicious_ids: List[int],          # which client IDs are malicious
    poison_fn: Optional[Callable],
    aggregation: str = "fedavg",       # fedavg | tm | tm_variant | fltrust
    rounds: int = 500,
    clients_per_round: int = 100,
    local_epochs: int = 5,
    batch_size: int = 8,
    lr: float = 0.005,
    device: str = "cpu",
    tm_alpha: int = 20,
    tm_beta:  int = 100,
    root_indices: Optional[List[int]] = None,  # for FLTrust
    log_every: int = 10,
) -> List[float]:
    """
    Run plain FL for `rounds` rounds. Returns list of validation accuracies.
    """
    global_model = model_class().to(device)
    n_clients    = len(client_indices)
    accuracies   = []

    # FLTrust root model (trained on clean root data)
    root_model_sd = None
    if aggregation == "fltrust" and root_indices is not None:
        from wwfl.utils import PoisonedSubset
        root_model_sd = train_one_client(
            global_model, root_indices, train_dataset,
            poison_fn=None, device=device,
            epochs=local_epochs, batch_size=batch_size, lr=lr
        )

    for r in range(1, rounds + 1):
        selected = random.sample(range(n_clients), clients_per_round)
        updates  = []

        for cid in selected:
            pfn = poison_fn if cid in malicious_ids else None
            sd  = train_one_client(
                global_model, client_indices[cid], train_dataset,
                poison_fn=pfn, device=device,
                epochs=local_epochs, batch_size=batch_size, lr=lr
            )
            updates.append(sd)

        # Aggregate
        if aggregation == "fedavg":
            new_sd = fedavg(updates)
        elif aggregation == "tm":
            new_sd = trimmed_mean(updates, alpha=tm_alpha)
        elif aggregation == "tm_variant":
            new_sd = trimmed_mean_variant(updates, alpha=tm_alpha, beta=tm_beta)
        elif aggregation == "fltrust":
            if root_indices is not None:
                root_model_sd = train_one_client(
                    global_model, root_indices, train_dataset,
                    poison_fn=None, device=device,
                    epochs=local_epochs, batch_size=batch_size, lr=lr
                )
            new_sd = fltrust(updates, root_model_sd)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

        global_model.load_state_dict(new_sd)

        if r % log_every == 0:
            acc = evaluate(global_model, test_loader, device)
            accuracies.append((r, acc))
            print(f"  [FL] Round {r:4d} | {aggregation:12s} | Acc: {acc:.2f}%")

    return accuracies
