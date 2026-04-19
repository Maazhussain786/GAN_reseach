from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_fig(figures_dir: Path, name: str) -> None:
    plt.tight_layout()
    plt.savefig(figures_dir / name, dpi=300)
    plt.close()


def build_publication_tables(outputs: Path, runs: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    summary = summary.copy().sort_values("ratio_pct")
    baseline = float(summary.loc[summary["ratio_pct"] == 0, "acc_mean"].iloc[0])

    summary["acc_gain_pp"] = (summary["acc_mean"] - baseline) * 100.0
    summary["acc_rank"] = summary["acc_mean"].rank(ascending=False, method="min").astype(int)
    summary["f1_rank"] = summary["f1_mean"].rank(ascending=False, method="min").astype(int)

    publication_cols = [
        "ratio_pct",
        "n_runs",
        "acc_mean",
        "acc_std",
        "acc_ci95",
        "acc_gain_pp",
        "f1_mean",
        "f1_std",
        "f1_ci95",
        "acc_rank",
        "f1_rank",
    ]
    publication_table = summary[publication_cols]
    publication_table.to_csv(outputs / "publication_summary.csv", index=False)

    pivot = runs.pivot_table(index="seed", columns="ratio_pct", values="test_acc", aggfunc="mean")
    if 0 not in pivot.columns:
        raise RuntimeError("Baseline ratio 0 is missing from run_results.csv")

    baseline_by_seed = pivot[0]
    deltas = []
    for ratio in sorted([c for c in pivot.columns if c != 0]):
        delta = (pivot[ratio] - baseline_by_seed).dropna()
        n = len(delta)
        if n == 0:
            continue
        mean_delta = float(delta.mean())
        std_delta = float(delta.std(ddof=1)) if n > 1 else 0.0
        ci95 = 1.96 * (std_delta / np.sqrt(n)) if n > 1 else 0.0
        improved_runs = int((delta > 0).sum())
        deltas.append(
            {
                "ratio_pct": int(ratio),
                "n_pairs": int(n),
                "mean_delta_acc": mean_delta,
                "std_delta_acc": std_delta,
                "ci95_delta_acc": ci95,
                "improved_runs": improved_runs,
                "improved_fraction": improved_runs / n,
            }
        )

    delta_df = pd.DataFrame(deltas).sort_values("ratio_pct")
    delta_df.to_csv(outputs / "paired_delta_vs_baseline.csv", index=False)
    return delta_df


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    outputs = root / "local_repro" / "outputs"
    figures = outputs / "figures"
    ensure_dir(figures)

    run_path = outputs / "run_results.csv"
    summary_path = outputs / "summary_results.csv"
    counts_path = outputs / "dataset_counts.csv"
    timeline_run_path = outputs / "timeline_runs.csv"
    timeline_epoch_path = outputs / "timeline_epochs.csv"

    if not run_path.exists() or not summary_path.exists():
        raise FileNotFoundError("run_results.csv and summary_results.csv are required in local_repro/outputs")

    runs = pd.read_csv(run_path)
    summary = pd.read_csv(summary_path).sort_values("ratio_pct")
    counts = pd.read_csv(counts_path) if counts_path.exists() else None
    timeline_runs = pd.read_csv(timeline_run_path) if timeline_run_path.exists() else None
    timeline_epochs = pd.read_csv(timeline_epoch_path) if timeline_epoch_path.exists() else None

    delta_df = build_publication_tables(outputs, runs, summary)

    x = summary["ratio_pct"].to_numpy()
    acc_mean = summary["acc_mean"].to_numpy()
    acc_ci = summary["acc_ci95"].fillna(0.0).to_numpy()
    f1_mean = summary["f1_mean"].to_numpy()
    f1_ci = summary["f1_ci95"].fillna(0.0).to_numpy()

    baseline_acc = float(summary.loc[summary["ratio_pct"] == 0, "acc_mean"].iloc[0])
    gain = (acc_mean - baseline_acc) * 100.0

    # Figure 01: Accuracy with CI
    plt.figure(figsize=(7, 4.5))
    plt.errorbar(x, acc_mean, yerr=acc_ci, marker="o", capsize=4, linewidth=2)
    plt.xlabel("Synthetic Ratio (%)")
    plt.ylabel("Accuracy")
    plt.title("Figure 01: Accuracy vs Synthetic Ratio (Mean +/- 95% CI)")
    plt.grid(True, alpha=0.3)
    save_fig(figures, "Figure01_Accuracy_CI.png")

    # Figure 02: F1 with CI
    plt.figure(figsize=(7, 4.5))
    plt.errorbar(x, f1_mean, yerr=f1_ci, marker="o", capsize=4, linewidth=2, color="#c23b22")
    plt.xlabel("Synthetic Ratio (%)")
    plt.ylabel("F1-score")
    plt.title("Figure 02: F1-score vs Synthetic Ratio (Mean +/- 95% CI)")
    plt.grid(True, alpha=0.3)
    save_fig(figures, "Figure02_F1_CI.png")

    # Figure 03: Gain over baseline
    plt.figure(figsize=(7, 4.5))
    plt.bar([str(int(v)) for v in x], gain, color="#4c78a8")
    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("Synthetic Ratio (%)")
    plt.ylabel("Accuracy Gain vs 0% (pp)")
    plt.title("Figure 03: Accuracy Gain over Real-only Baseline")
    plt.grid(axis="y", alpha=0.3)
    save_fig(figures, "Figure03_Accuracy_Gain.png")

    # Figure 04: Accuracy distribution
    order = sorted(runs["ratio_pct"].unique())
    plt.figure(figsize=(7, 4.5))
    box_data = [runs.loc[runs["ratio_pct"] == r, "test_acc"].values for r in order]
    plt.boxplot(box_data, tick_labels=[str(r) for r in order], showmeans=True)
    plt.xlabel("Synthetic Ratio (%)")
    plt.ylabel("Test Accuracy")
    plt.title("Figure 04: Run-to-run Accuracy Distribution")
    plt.grid(axis="y", alpha=0.3)
    save_fig(figures, "Figure04_Accuracy_Boxplot.png")

    # Figure 05: F1 distribution
    plt.figure(figsize=(7, 4.5))
    box_data_f1 = [runs.loc[runs["ratio_pct"] == r, "test_f1"].values for r in order]
    plt.boxplot(box_data_f1, tick_labels=[str(r) for r in order], showmeans=True)
    plt.xlabel("Synthetic Ratio (%)")
    plt.ylabel("Test F1-score")
    plt.title("Figure 05: Run-to-run F1 Distribution")
    plt.grid(axis="y", alpha=0.3)
    save_fig(figures, "Figure05_F1_Boxplot.png")

    # Figure 06: Accuracy/F1 frontier with CI bars
    plt.figure(figsize=(7, 4.5))
    plt.errorbar(acc_mean, f1_mean, xerr=acc_ci, yerr=f1_ci, fmt="o", capsize=4)
    for _, row in summary.iterrows():
        plt.annotate(f"{int(row['ratio_pct'])}%", (row["acc_mean"], row["f1_mean"]))
    plt.xlabel("Accuracy (mean)")
    plt.ylabel("F1-score (mean)")
    plt.title("Figure 06: Accuracy-F1 Trade-off")
    plt.grid(True, alpha=0.3)
    save_fig(figures, "Figure06_Accuracy_F1_Frontier.png")

    # Figure 07: Metric heatmap
    heat = np.vstack([acc_mean, f1_mean])
    plt.figure(figsize=(8, 3.2))
    im = plt.imshow(heat, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
    plt.yticks([0, 1], ["Accuracy", "F1"])
    plt.xticks(np.arange(len(x)), [str(int(v)) for v in x])
    plt.xlabel("Synthetic Ratio (%)")
    plt.title("Figure 07: Metric Heatmap by Ratio")
    plt.colorbar(im, fraction=0.03, pad=0.03)
    save_fig(figures, "Figure07_Metric_Heatmap.png")

    # Figure 08: Dataset composition
    if counts is not None and len(counts) > 0:
        comp = (
            counts.groupby("ratio_pct", as_index=False)
            .agg(train_real_total=("train_real_total", "mean"), train_synth_total=("train_synth_total", "mean"))
            .sort_values("ratio_pct")
        )
        plt.figure(figsize=(7, 4.5))
        xs = np.arange(len(comp))
        plt.bar(xs, comp["train_real_total"], label="Real (train)")
        plt.bar(xs, comp["train_synth_total"], bottom=comp["train_real_total"], label="Synthetic (train)")
        plt.xticks(xs, [str(int(v)) for v in comp["ratio_pct"]])
        plt.xlabel("Synthetic Ratio (%)")
        plt.ylabel("Number of training samples")
        plt.title("Figure 08: Training Set Composition")
        plt.legend()
        plt.grid(axis="y", alpha=0.3)
        save_fig(figures, "Figure08_Training_Composition.png")

    # Figure 09: Epoch validation accuracy
    if timeline_epochs is not None and len(timeline_epochs) > 0:
        ep_acc = (
            timeline_epochs.groupby(["ratio_pct", "epoch"], as_index=False)
            .agg(val_acc_mean=("val_acc", "mean"), val_acc_std=("val_acc", "std"))
            .sort_values(["ratio_pct", "epoch"])
        )
        plt.figure(figsize=(8, 5))
        for ratio in sorted(ep_acc["ratio_pct"].unique()):
            sub = ep_acc.loc[ep_acc["ratio_pct"] == ratio]
            plt.plot(sub["epoch"], sub["val_acc_mean"], marker="o", label=f"{int(ratio)}%")
        plt.xlabel("Epoch")
        plt.ylabel("Validation Accuracy (mean over seeds)")
        plt.title("Figure 09: Validation Accuracy Learning Curves")
        plt.grid(True, alpha=0.3)
        plt.legend(title="Ratio")
        save_fig(figures, "Figure09_ValAcc_LearningCurves.png")

        # Figure 10: Epoch training loss
        ep_loss = (
            timeline_epochs.groupby(["ratio_pct", "epoch"], as_index=False)
            .agg(loss_mean=("train_loss", "mean"))
            .sort_values(["ratio_pct", "epoch"])
        )
        plt.figure(figsize=(8, 5))
        for ratio in sorted(ep_loss["ratio_pct"].unique()):
            sub = ep_loss.loc[ep_loss["ratio_pct"] == ratio]
            plt.plot(sub["epoch"], sub["loss_mean"], marker="o", label=f"{int(ratio)}%")
        plt.xlabel("Epoch")
        plt.ylabel("Training Loss (mean over seeds)")
        plt.title("Figure 10: Training Loss Curves")
        plt.grid(True, alpha=0.3)
        plt.legend(title="Ratio")
        save_fig(figures, "Figure10_TrainLoss_Curves.png")

    # Figure 11: Runtime by ratio
    plt.figure(figsize=(7, 4.5))
    run_sec_data = [runs.loc[runs["ratio_pct"] == r, "run_seconds"].values for r in order]
    plt.boxplot(run_sec_data, tick_labels=[str(r) for r in order], showmeans=True)
    plt.xlabel("Synthetic Ratio (%)")
    plt.ylabel("Runtime per run (seconds)")
    plt.title("Figure 11: Runtime Distribution by Ratio")
    plt.grid(axis="y", alpha=0.3)
    save_fig(figures, "Figure11_Runtime_Boxplot.png")

    # Figure 12: Cumulative runtime timeline
    if timeline_runs is not None and len(timeline_runs) > 0:
        timeline_runs = timeline_runs.sort_values("run_index").copy()
        timeline_runs["cum_hours"] = timeline_runs["run_seconds"].cumsum() / 3600.0
        plt.figure(figsize=(8, 4.8))
        plt.plot(timeline_runs["run_index"], timeline_runs["cum_hours"], marker="o")
        plt.xlabel("Run Index")
        plt.ylabel("Cumulative Runtime (hours)")
        plt.title("Figure 12: Experiment Timeline")
        plt.grid(True, alpha=0.3)
        save_fig(figures, "Figure12_Experiment_Timeline.png")

    # Figure 13: Paired delta vs baseline
    if len(delta_df) > 0:
        plt.figure(figsize=(7, 4.5))
        plt.bar(delta_df["ratio_pct"].astype(str), delta_df["mean_delta_acc"] * 100.0, color="#2ca02c")
        plt.axhline(0, color="black", linewidth=1)
        plt.xlabel("Synthetic Ratio (%)")
        plt.ylabel("Mean paired delta vs 0% (pp)")
        plt.title("Figure 13: Paired Accuracy Delta vs Baseline")
        plt.grid(axis="y", alpha=0.3)
        save_fig(figures, "Figure13_Paired_Delta.png")

    print("Saved figures to:", figures)
    print("Saved tables:")
    print(outputs / "publication_summary.csv")
    print(outputs / "paired_delta_vs_baseline.csv")


if __name__ == "__main__":
    main()
