from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    outputs = root / "local_repro" / "outputs"
    figures = outputs / "figures"
    ensure_dir(figures)

    run_path = outputs / "run_results.csv"
    summary_path = outputs / "summary_results.csv"
    timeline_path = outputs / "timeline_runs.csv"

    if not run_path.exists() or not summary_path.exists():
        raise FileNotFoundError("run_results.csv and summary_results.csv are required in local_repro/outputs")

    runs = pd.read_csv(run_path)
    summary = pd.read_csv(summary_path).sort_values("ratio")
    timeline = pd.read_csv(timeline_path) if timeline_path.exists() else None

    x = summary["ratio_pct"].values
    acc_mean = summary["acc_mean"].values
    acc_ci = summary["acc_ci95"].fillna(0.0).values
    f1_mean = summary["f1_mean"].values
    f1_ci = summary["f1_ci95"].fillna(0.0).values

    baseline_acc = float(summary.loc[summary["ratio_pct"] == 0, "acc_mean"].iloc[0])
    gain = (acc_mean - baseline_acc) * 100.0

    # Figure 1: Accuracy with 95% CI
    plt.figure(figsize=(7, 4.5))
    plt.errorbar(x, acc_mean, yerr=acc_ci, marker="o", capsize=4)
    plt.xlabel("Synthetic Ratio (%)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Synthetic Ratio (Mean ± 95% CI)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures / "Figure1_Accuracy_CI.png", dpi=300)
    plt.close()

    # Figure 2: F1 with 95% CI
    plt.figure(figsize=(7, 4.5))
    plt.errorbar(x, f1_mean, yerr=f1_ci, marker="o", capsize=4, color="#d62728")
    plt.xlabel("Synthetic Ratio (%)")
    plt.ylabel("F1-score")
    plt.title("F1-score vs Synthetic Ratio (Mean ± 95% CI)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures / "Figure2_F1_CI.png", dpi=300)
    plt.close()

    # Figure 3: Accuracy gain over baseline
    plt.figure(figsize=(7, 4.5))
    plt.bar([str(int(v)) for v in x], gain)
    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("Synthetic Ratio (%)")
    plt.ylabel("Accuracy Gain vs 0% (pp)")
    plt.title("Accuracy Gain over Real-only Baseline")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures / "Figure3_Accuracy_Gain.png", dpi=300)
    plt.close()

    # Figure 4: Accuracy distribution by ratio
    plt.figure(figsize=(7, 4.5))
    order = sorted(runs["ratio_pct"].unique())
    data = [runs.loc[runs["ratio_pct"] == r, "test_acc"].values for r in order]
    plt.boxplot(data, tick_labels=[str(r) for r in order], showmeans=True)
    plt.xlabel("Synthetic Ratio (%)")
    plt.ylabel("Test Accuracy")
    plt.title("Run-to-run Accuracy Distribution")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures / "Figure4_Accuracy_Boxplot.png", dpi=300)
    plt.close()

    # Figure 5: Accuracy-F1 frontier
    plt.figure(figsize=(7, 4.5))
    plt.scatter(acc_mean, f1_mean, s=60)
    for _, row in summary.iterrows():
        plt.annotate(f"{int(row['ratio_pct'])}%", (row["acc_mean"], row["f1_mean"]))
    plt.xlabel("Accuracy (mean)")
    plt.ylabel("F1-score (mean)")
    plt.title("Accuracy-F1 Trade-off by Synthetic Ratio")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures / "Figure5_Tradeoff.png", dpi=300)
    plt.close()

    # Figure 6: Experiment timeline
    if timeline is not None and len(timeline) > 0:
        timeline = timeline.sort_values("run_index").copy()
        timeline["cum_hours"] = timeline["run_seconds"].cumsum() / 3600.0
        plt.figure(figsize=(7, 4.5))
        plt.plot(timeline["run_index"], timeline["cum_hours"], marker="o")
        for _, row in timeline.iterrows():
            plt.annotate(f"{int(row['ratio_pct'])}%/s{int(row['seed'])}", (row["run_index"], row["cum_hours"]))
        plt.xlabel("Run Index")
        plt.ylabel("Cumulative Runtime (hours)")
        plt.title("Experiment Timeline")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures / "Figure6_Experiment_Timeline.png", dpi=300)
        plt.close()

    print("Saved figures to:", figures)


if __name__ == "__main__":
    main()
