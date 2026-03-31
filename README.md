# WW-FL Paper Reproduction

Reproduces **Fig 3, 4, 5, 6** and **Table 11** from:
> WW-FL: Secure and Private Large-Scale Federated Learning (IACR TCHES 2026)

---

## Setup

```bash
# Python 3.8–3.10 required for CrypTen
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install crypten matplotlib numpy tqdm
```

---

## Run experiments

### Figure 3 — FL vs WW-FL accuracy (4 subplots)
```bash
# Run MNIST (faster, ~1-2h on GPU)
python run_experiments.py --exp fig3 --dataset mnist --rounds 500 --device cuda
python run_experiments.py --exp fig3 --dataset mnist --rounds 2000 --device cuda

# Run CIFAR10 (slower, ~3-5h on GPU)
python run_experiments.py --exp fig3 --dataset cifar10 --rounds 500 --device cuda
python run_experiments.py --exp fig3 --dataset cifar10 --rounds 2000 --device cuda
```
Results saved as `results/fig3_accuracy.png`

### Figure 4 — Plaintext vs MPC precision
```bash
python run_experiments.py --exp fig4 --rounds 500 --device cuda
```
Results saved as `results/fig4_mpc_accuracy.png`

### Figure 5 — Poisoning attacks
```bash
python run_experiments.py --exp fig5 --rounds 2000 --device cuda
```
Results saved as `results/fig5_poisoning.png`
⚠️ This is the slowest experiment (~8-12h). Consider `--rounds 500` for a quick preview.

### Figure 6 — TM variant comparison
```bash
python run_experiments.py --exp fig6 --rounds 2000 --device cuda
```
Results saved as `results/fig6_tm_variant.png`

---

## Paper config summary (from Table 7)

| Parameter | FL | WW-FL |
|---|---|---|
| Clients | 1000 | 1000 (10 clusters × 100) |
| Selected/round | 100 | 10 per cluster |
| Samples/client | 200 | 200 |
| Batch size | 8 | 80 |
| Learning rate | 0.005 | 0.05 |
| Local epochs | 5 | 5 |

## Notes
- CrypTen must be on Python 3.8–3.10 (not 3.11+)
- For Fig 4 (MPC precision), we simulate fixed-point quantization (22-bit) in `wwfl_trainer.py`
- Results are checkpointed as `.pkl` files so you can re-plot without re-running
