"""
Plotting utilities to reproduce Figures 3, 4, 5, 6 and Table 11 from WW-FL paper.
"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})


def _unzip(results):
    if not results:
        return [], []
    rounds, accs = zip(*results)
    return list(rounds), list(accs)


def plot_fig3(fl_mnist_500, wwfl_mnist_500,
              fl_cifar_500, wwfl_cifar_500,
              fl_mnist_2000, wwfl_mnist_2000,
              fl_cifar_2000, wwfl_cifar_2000,
              save_path="results/fig3_accuracy.png"):
    """Figure 3: FL vs WW-FL validation accuracy (4 subplots)."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    datasets = [
        (fl_mnist_500,  wwfl_mnist_500,  "LeNet trained on MNIST for 500 epochs",  axes[0, 0]),
        (fl_cifar_500,  wwfl_cifar_500,  "ResNet9 on CIFAR10 for 500 epochs",      axes[0, 1]),
        (fl_mnist_2000, wwfl_mnist_2000, "LeNet trained on MNIST for 2000 epochs", axes[1, 0]),
        (fl_cifar_2000, wwfl_cifar_2000, "ResNet9 on CIFAR10 for 2000 epochs",     axes[1, 1]),
    ]

    for fl_res, ww_res, title, ax in datasets:
        fl_r, fl_a   = _unzip(fl_res)
        ww_r, ww_a   = _unzip(ww_res)
        ax.plot(fl_r, fl_a, "b--", label="FL",    linewidth=1.5)
        ax.plot(ww_r, ww_a, "g-",  label="WW-FL", linewidth=1.5)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 100)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path)
    print(f"Saved: {save_path}")
    return fig


def plot_fig4(fl_plain, fl_crypten, wwfl_plain, wwfl_crypten,
              save_path="results/fig4_mpc_accuracy.png"):
    """Figure 4: Plaintext vs CrypTen simulation accuracy for LeNet/MNIST."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))

    for results, label, style in [
        (fl_plain,    "FL (Plaintext)",  "b--"),
        (fl_crypten,  "FL (CrypTen)",    "b:"),
        (wwfl_plain,  "WW-FL (Plaintext)", "g-"),
        (wwfl_crypten,"WW-FL (CrypTen)",   "g-."),
    ]:
        r, a = _unzip(results)
        ax.plot(r, a, style, label=label, linewidth=1.5)

    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 100)
    ax.set_title("FL and WW-FL: Plaintext vs MPC (CrypTen simulation) — LeNet/MNIST")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path)
    print(f"Saved: {save_path}")
    return fig


def plot_fig5(results_dict: dict, save_path="results/fig5_poisoning.png"):
    """
    Figure 5: Accuracy under DLF attack for FL and WW-FL.
    results_dict keys: (scheme, mode, poison_rate, method)
    scheme: 'FL' | 'WW-FL'
    mode:   'equally_distributed' | 'focused'
    poison_rate: 0.01 | 0.1 | 0.2
    method: 'fedavg' | 'fltrust' | 'tm'
    Value: list of (round, accuracy)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    poison_rates = [0.01, 0.1, 0.2]
    modes        = ["equally_distributed", "focused"]
    methods      = ["fedavg", "fltrust", "tm"]
    colors       = {"fedavg": "blue", "fltrust": "orange", "tm": "green"}
    styles       = {"FL": "--", "WW-FL": "-"}

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for row, mode in enumerate(modes):
        for col, pr in enumerate(poison_rates):
            ax = axes[row, col]
            for method in methods:
                for scheme in ["FL", "WW-FL"]:
                    key = (scheme, mode, pr, method)
                    if key not in results_dict:
                        continue
                    r, a = _unzip(results_dict[key])
                    label = f"{method.upper()} ({scheme})"
                    ax.plot(r, a, linestyle=styles[scheme],
                            color=colors[method], label=label, linewidth=1.2)
            ax.set_title(f"DLF - {mode.replace('_', ' ').title()}\nPoison: {pr}")
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Accuracy")
            ax.set_ylim(0, 100)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

    fig.suptitle("Figure 5: DLF Attack — ResNet9/CIFAR10", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path)
    print(f"Saved: {save_path}")
    return fig


def plot_fig6(tm_results: dict, save_path="results/fig6_tm_variant.png"):
    """
    Figure 6: TM vs TM variant (sample sizes 10, 100, 1000) under focused DLF.
    tm_results keys: 'TM' | 'TM-10' | 'TM-100' | 'TM-1000'
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))

    styles = {"TM": "b-", "TM-10": "g--", "TM-100": "r-.", "TM-1000": "m:"}
    for label, style in styles.items():
        if label not in tm_results:
            continue
        r, a = _unzip(tm_results[label])
        ax.plot(r, a, style, label=label, linewidth=1.5)

    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 100)
    ax.set_title("Figure 6: TM vs TM Variant — Focused DLF @ 0.2 poison rate, ResNet9/CIFAR10")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path)
    print(f"Saved: {save_path}")
    return fig


def print_table11(timing_results: dict):
    """
    Table 11: Communication and run-time for aggregation schemes.
    timing_results: dict with keys like 'FedAvg', 'FLTrust', 'TM', 'TM-10', etc.
    Each value: {'comm_mb': float, 'time_s': float}
    """
    print("\n=== Table 11: Aggregation Overhead ===")
    print(f"{'Scheme':<12} {'Comm (MB)':>12} {'Time (s)':>12}")
    print("-" * 38)
    for scheme, vals in timing_results.items():
        print(f"{scheme:<12} {vals.get('comm_mb', 0):>12.2f} {vals.get('time_s', 0):>12.3f}")
    print()
