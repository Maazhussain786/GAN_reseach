# GAN Research (Code-Only)

This repository contains code for local, reproducible experiments on GAN-augmented orange ripeness classification.

## Included

- local_repro/experiment_runner.py
- local_repro/plot_results.py
- local_repro/README.md
- scripts/ (legacy helper scripts)
- setup_msvc.ps1
- test_cuda_simple.py

## Excluded from Git

- Datasets and archives
- Images and generated figures
- Checkpoints and model weights
- CSV outputs and notebook files

## Quick Start

1. Create/activate Python environment.
2. Install dependencies:

```powershell
pip install torch torchvision pandas scikit-learn tqdm pillow matplotlib
```

3. Run experiments:

```powershell
python local_repro/experiment_runner.py --model-name densenet121 --scenario field --epochs 1 --freeze-backbone --real-train-subsample 0.4 --field-eval-distortion-prob 0.62 --ratios 0 0.25 0.5 1.0 2.0 --seeds 42 1337 2025 7 99
```

4. Generate figures:

```powershell
python local_repro/plot_results.py
```
