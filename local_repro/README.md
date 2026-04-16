# Local Reproducible Pipeline

This folder contains a clean local pipeline to rerun your GAN-ratio classification experiments with repeatability.

## What This Solves

- Recreates the 0%, 25%, 50%, 100%, 200% synthetic ratios.
- Keeps validation and test sets real-only.
- Supports multiple random seeds for publication-grade statistics.
- Defaults to DenseNet121 (can switch to ResNet18).
- Produces per-run and summary CSV files with 95% confidence intervals.
- Supports `field` scenario with realistic distortions (blur, compression, occlusion, lighting shifts, sensor noise).
- Saves timeline logs for run-by-run and epoch-by-epoch progress.

## Expected Data Layout

- `Data/orange_dataset_cleaned/ripe_oranges/*`
- `Data/orange_dataset_cleaned/unripe_orange/*`
- `Data/GAN/Ripe_img/*`
- `Data/GAN/Unripe_img/*`

## Run Experiments

From `D:\GAN`:

```powershell
python local_repro/experiment_runner.py --model-name densenet121 --scenario field --epochs 8 --batch-size 16 --seeds 42 1337 2025 7 99 --ratios 0 0.25 0.5 1.0 2.0
```

Publication baseline target protocol (harder real-world setting):

```powershell
python local_repro/experiment_runner.py --model-name densenet121 --scenario field --field-eval-distortion-prob 0.65 --real-train-subsample 0.5 --epochs 8 --batch-size 16 --seeds 42 1337 2025 7 99 --ratios 0 0.25 0.5 1.0 2.0
```

Optional quick test run:

```powershell
python local_repro/experiment_runner.py --model-name densenet121 --epochs 2 --batch-size 16 --seeds 42 --ratios 0 0.5
```

## Generate Paper Figures

```powershell
python local_repro/plot_results.py
```

Figures are saved under `local_repro/outputs/figures`.

## Output Files

- `local_repro/outputs/run_results.csv`
- `local_repro/outputs/summary_results.csv`
- `local_repro/outputs/dataset_counts.csv`
- `local_repro/outputs/run_config.json`
- `local_repro/outputs/timeline_runs.csv`
- `local_repro/outputs/timeline_epochs.csv`

## Notes

- By default, synthetic samples can be reused (sampling with replacement) so 200% ratios do not fail if synthetic pool is small.
- Use `--no-synth-replacement` if you want strict unique synthetic sampling.
- Use `--freeze-backbone` if full DenseNet fine-tuning is too slow on your machine.
