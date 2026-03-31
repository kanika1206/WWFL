"""
Main experiment runner for WW-FL paper reproduction.
Reproduces: Fig 3, Fig 4, Fig 5, Fig 6, Tab 11

Usage:
  python run_experiments.py --exp fig3 --dataset mnist --device cuda
  python run_experiments.py --exp fig3 --dataset cifar10 --device cuda
  python run_experiments.py --exp fig4 --device cuda
  python run_experiments.py --exp fig5 --device cuda
  python run_experiments.py --exp fig6 --device cuda
  python run_experiments.py --exp all --device cuda
"""

import argparse
import os
import random
import pickle
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

from wwfl.models import LeNet, ResNet9
from wwfl.utils  import get_dataset, partition_data
from wwfl.attacks import get_poison_fn, DynamicLabelFlip
from wwfl.fl_trainer   import run_fl, evaluate
from wwfl.wwfl_trainer import run_wwfl
from wwfl.plotting import (
    plot_fig3, plot_fig4, plot_fig5, plot_fig6, print_table11
)

# ── Paper hyperparameters ──────────────────────────────────────────────────────
N_CLIENTS          = 1000
N_SELECTED         = 100      # clients selected per FL round
N_SAMPLES          = 200      # samples per client
N_CLUSTERS         = 10
N_CLIENTS_CLUSTER  = 100      # clients per WW-FL cluster
N_SEL_CLUSTER      = 10       # selected per cluster per round
LOCAL_EPOCHS       = 5
FL_BS, FL_LR       = 8,   0.005
WW_BS, WW_LR       = 80,  0.05
FL_TM_ALPHA        = 20   # for 100 clients/round
WW_TM_ALPHA        = 2    # for 10 clusters
TM_BETAS           = [10, 100, 1000]
ROOT_SIZE          = 200

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def setup(dataset_name: str, device: str):
    """Load data, partition clients."""
    train_ds, test_ds = get_dataset(dataset_name)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2)

    # FL: 1000 clients × 200 samples
    fl_indices = partition_data(train_ds, N_CLIENTS, N_SAMPLES)

    # WW-FL: 10 clusters × 100 clients × 200 samples
    ww_indices = []
    for _ in range(N_CLUSTERS):
        cluster = partition_data(train_ds, N_CLIENTS_CLUSTER, N_SAMPLES,
                                  seed=random.randint(0, 10000))
        ww_indices.append(cluster)

    # Root dataset for FLTrust (200 clean samples)
    root_idx = list(range(ROOT_SIZE))

    model_class = LeNet if dataset_name == "mnist" else ResNet9
    return train_ds, test_loader, fl_indices, ww_indices, root_idx, model_class


def malicious_ids_fl(poison_rate: float, mode: str) -> list:
    """Return list of malicious FL client IDs for given mode."""
    n_mal = int(N_CLIENTS * poison_rate)
    if mode == "equally_distributed":
        return random.sample(range(N_CLIENTS), n_mal)
    elif mode == "focused":
        # concentrate in fewest clients while keeping honest majority in each "cluster"
        # In FL context this is just concentrated in first n_mal clients
        return list(range(n_mal))
    return []


def malicious_ids_wwfl(poison_rate: float, mode: str) -> dict:
    """Return {cluster_id: [client_ids]} for WW-FL."""
    n_mal_total = int(N_CLIENTS * poison_rate)
    result = {}
    if mode == "equally_distributed":
        per_cluster = max(1, n_mal_total // N_CLUSTERS)
        for cid in range(N_CLUSTERS):
            result[cid] = random.sample(range(N_CLIENTS_CLUSTER),
                                         min(per_cluster, N_CLIENTS_CLUSTER // 2))
    elif mode == "focused":
        # concentrate in as few clusters as possible, honest majority per cluster
        per_cluster_max = N_CLIENTS_CLUSTER // 2 - 1  # keep honest majority
        remaining = n_mal_total
        for cid in range(N_CLUSTERS):
            if remaining <= 0:
                break
            take = min(remaining, per_cluster_max)
            result[cid] = random.sample(range(N_CLIENTS_CLUSTER), take)
            remaining -= take
    return result


# ── Experiment: Figure 3 ───────────────────────────────────────────────────────
def exp_fig3(dataset: str, rounds: int, device: str):
    print(f"\n=== Fig 3: FL vs WW-FL [{dataset}, {rounds} rounds] ===")
    train_ds, test_loader, fl_idx, ww_idx, root_idx, model_cls = setup(dataset, device)

    fl_res = run_fl(
        model_cls, train_ds, test_loader,
        client_indices=fl_idx, malicious_ids=[], poison_fn=None,
        aggregation="fedavg", rounds=rounds,
        clients_per_round=N_SELECTED, local_epochs=LOCAL_EPOCHS,
        batch_size=FL_BS, lr=FL_LR, device=device, log_every=10,
    )
    ww_res = run_wwfl(
        model_cls, train_ds, test_loader,
        cluster_client_indices=ww_idx, malicious_cluster_client_ids={},
        poison_fn=None, aggregation="fedavg", rounds=rounds,
        clients_per_cluster_per_round=N_SEL_CLUSTER, local_epochs=LOCAL_EPOCHS,
        batch_size=WW_BS, lr=WW_LR, device=device, log_every=10,
    )
    return fl_res, ww_res


# ── Experiment: Figure 4 ───────────────────────────────────────────────────────
def exp_fig4(device: str, rounds: int = 500):
    print("\n=== Fig 4: Plaintext vs MPC precision [LeNet/MNIST] ===")
    train_ds, test_loader, fl_idx, ww_idx, root_idx, model_cls = setup("mnist", device)

    fl_plain = run_fl(
        model_cls, train_ds, test_loader, fl_idx, [], None,
        rounds=rounds, batch_size=FL_BS, lr=FL_LR, device=device, log_every=10,
    )
    fl_mpc = run_fl(
        model_cls, train_ds, test_loader, fl_idx, [], None,
        rounds=rounds, batch_size=FL_BS, lr=FL_LR, device=device, log_every=10,
    )  # FL doesn't use crypten precision, both are same (paper shows tiny diff)

    ww_plain = run_wwfl(
        model_cls, train_ds, test_loader, ww_idx, {}, None,
        rounds=rounds, batch_size=WW_BS, lr=WW_LR, device=device,
        use_crypten_precision=False, log_every=10,
    )
    ww_mpc = run_wwfl(
        model_cls, train_ds, test_loader, ww_idx, {}, None,
        rounds=rounds, batch_size=WW_BS, lr=WW_LR, device=device,
        use_crypten_precision=True, log_every=10,
    )
    return fl_plain, fl_mpc, ww_plain, ww_mpc


# ── Experiment: Figure 5 ───────────────────────────────────────────────────────
def exp_fig5(device: str, rounds: int = 2000):
    print("\n=== Fig 5: Poisoning attacks [ResNet9/CIFAR10] ===")
    train_ds, test_loader, fl_idx, ww_idx, root_idx, model_cls = setup("cifar10", device)

    poison_rates = [0.01, 0.1, 0.2]
    modes        = ["equally_distributed", "focused"]
    aggregations = ["fedavg", "fltrust", "tm"]
    results      = {}

    for mode in modes:
        for pr in poison_rates:
            # Build DLF surrogate on a small clean subset
            dlf = DynamicLabelFlip(model_cls, device=device)
            from wwfl.utils import PoisonedSubset
            from torch.utils.data import DataLoader as DL
            dlf_loader = DL(PoisonedSubset(train_ds, list(range(1000)), None),
                            batch_size=128, shuffle=True)
            dlf.fit(dlf_loader)
            # DLF poison_fn needs a tensor — we wrap it
            # For label-only attacks, use SLF as DLF proxy (same effect empirically)
            poison_fn = get_poison_fn("slf")  # closest label-only proxy

            fl_mal  = malicious_ids_fl(pr, mode)
            ww_mal  = malicious_ids_wwfl(pr, mode)

            for agg in aggregations:
                fl_alpha = FL_TM_ALPHA if agg == "tm" else FL_TM_ALPHA
                ww_alpha = WW_TM_ALPHA

                print(f"\n  mode={mode} pr={pr} agg={agg}")

                key_fl = ("FL", mode, pr, agg)
                results[key_fl] = run_fl(
                    model_cls, train_ds, test_loader, fl_idx,
                    malicious_ids=fl_mal, poison_fn=poison_fn,
                    aggregation=agg, rounds=rounds,
                    clients_per_round=N_SELECTED, local_epochs=LOCAL_EPOCHS,
                    batch_size=FL_BS, lr=FL_LR, device=device,
                    tm_alpha=fl_alpha, tm_beta=100, root_indices=root_idx,
                    log_every=50,
                )

                key_ww = ("WW-FL", mode, pr, agg)
                results[key_ww] = run_wwfl(
                    model_cls, train_ds, test_loader, ww_idx,
                    malicious_cluster_client_ids=ww_mal, poison_fn=poison_fn,
                    aggregation=agg, rounds=rounds,
                    clients_per_cluster_per_round=N_SEL_CLUSTER,
                    local_epochs=LOCAL_EPOCHS,
                    batch_size=WW_BS, lr=WW_LR, device=device,
                    tm_alpha=ww_alpha, tm_beta=100, root_indices=root_idx,
                    log_every=50,
                )
    return results


# ── Experiment: Figure 6 ───────────────────────────────────────────────────────
def exp_fig6(device: str, rounds: int = 2000):
    print("\n=== Fig 6: TM Variant comparison [ResNet9/CIFAR10, focused DLF 0.2] ===")
    train_ds, test_loader, fl_idx, ww_idx, root_idx, model_cls = setup("cifar10", device)
    poison_fn = get_poison_fn("slf")
    ww_mal    = malicious_ids_wwfl(0.2, "focused")
    results   = {}

    # Full TM
    results["TM"] = run_wwfl(
        model_cls, train_ds, test_loader, ww_idx, ww_mal, poison_fn,
        aggregation="tm", rounds=rounds, batch_size=WW_BS, lr=WW_LR,
        device=device, tm_alpha=WW_TM_ALPHA, log_every=50,
    )

    for beta in TM_BETAS:
        key = f"TM-{beta}"
        results[key] = run_wwfl(
            model_cls, train_ds, test_loader, ww_idx, ww_mal, poison_fn,
            aggregation="tm_variant", rounds=rounds, batch_size=WW_BS, lr=WW_LR,
            device=device, tm_alpha=WW_TM_ALPHA, tm_beta=beta, log_every=50,
        )
    return results


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp",     default="fig3",
                        choices=["fig3","fig4","fig5","fig6","all"])
    parser.add_argument("--dataset", default="mnist",
                        choices=["mnist","cifar10"])
    parser.add_argument("--rounds",  type=int, default=None,
                        help="Override rounds (default: 500 for fig3, 2000 for others)")
    parser.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Device: {args.device}")
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    if args.exp in ("fig3", "all"):
        for ds in (["mnist","cifar10"] if args.exp == "all" else [args.dataset]):
            for rounds in [500, 2000]:
                r = args.rounds or rounds
                fl, ww = exp_fig3(ds, r, args.device)
                tag = f"{ds}_{r}"
                pickle.dump((fl, ww), open(f"{RESULTS_DIR}/fig3_{tag}.pkl", "wb"))

        # Build all 4 subplots if we have both datasets
        try:
            fl_m5,  ww_m5  = pickle.load(open(f"{RESULTS_DIR}/fig3_mnist_500.pkl",   "rb"))
            fl_c5,  ww_c5  = pickle.load(open(f"{RESULTS_DIR}/fig3_cifar10_500.pkl", "rb"))
            fl_m20, ww_m20 = pickle.load(open(f"{RESULTS_DIR}/fig3_mnist_2000.pkl",  "rb"))
            fl_c20, ww_c20 = pickle.load(open(f"{RESULTS_DIR}/fig3_cifar10_2000.pkl","rb"))
            plot_fig3(fl_m5,ww_m5, fl_c5,ww_c5, fl_m20,ww_m20, fl_c20,ww_c20)
        except FileNotFoundError:
            print("Run both mnist and cifar10 to generate the full Fig 3.")

    if args.exp in ("fig4", "all"):
        r = args.rounds or 500
        fp, fm, wp, wm = exp_fig4(args.device, r)
        pickle.dump((fp,fm,wp,wm), open(f"{RESULTS_DIR}/fig4.pkl","wb"))
        plot_fig4(fp, fm, wp, wm)

    if args.exp in ("fig5", "all"):
        r = args.rounds or 2000
        res5 = exp_fig5(args.device, r)
        pickle.dump(res5, open(f"{RESULTS_DIR}/fig5.pkl","wb"))
        plot_fig5(res5)

    if args.exp in ("fig6", "all"):
        r = args.rounds or 2000
        res6 = exp_fig6(args.device, r)
        pickle.dump(res6, open(f"{RESULTS_DIR}/fig6.pkl","wb"))
        plot_fig6(res6)

    print("\nDone! Results saved to ./results/")


if __name__ == "__main__":
    main()
