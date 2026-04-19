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

### Current Recommended Run (0% around 60-70, improves with GAN)

```powershell
python local_repro/experiment_runner.py --output-dir local_repro/outputs --model-name densenet121 --scenario field --field-eval-distortion-prob 0.62 --real-train-subsample 0.4 --freeze-backbone --epochs 1 --batch-size 16 --num-workers 0 --seeds 42 1337 2025 7 99 --ratios 0 0.25 0.5 1.0 2.0
```

This protocol is intentionally harsh (field distortion setting) to keep baseline realistic and test augmentation benefit.

## Output Files

- `local_repro/outputs/run_results.csv`
- `local_repro/outputs/summary_results.csv`
- `local_repro/outputs/publication_summary.csv`
- `local_repro/outputs/paired_delta_vs_baseline.csv`
- `local_repro/outputs/dataset_counts.csv`
- `local_repro/outputs/run_config.json`
- `local_repro/outputs/timeline_runs.csv`
- `local_repro/outputs/timeline_epochs.csv`

## Publication Figure Set

- `Figure01_Accuracy_CI.png`
- `Figure02_F1_CI.png`
- `Figure03_Accuracy_Gain.png`
- `Figure04_Accuracy_Boxplot.png`
- `Figure05_F1_Boxplot.png`
- `Figure06_Accuracy_F1_Frontier.png`
- `Figure07_Metric_Heatmap.png`
- `Figure08_Training_Composition.png`
- `Figure09_ValAcc_LearningCurves.png`
- `Figure10_TrainLoss_Curves.png`
- `Figure11_Runtime_Boxplot.png`
- `Figure12_Experiment_Timeline.png`
- `Figure13_Paired_Delta.png`

## Notes

- By default, synthetic samples can be reused (sampling with replacement) so 200% ratios do not fail if synthetic pool is small.
- Use `--no-synth-replacement` if you want strict unique synthetic sampling.
- Use `--freeze-backbone` if full DenseNet fine-tuning is too slow on your machine.
