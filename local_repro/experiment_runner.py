import argparse
import json
import math
import random
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm.auto import tqdm


CLASS_MAP = {
    "ripe": {
        "real_dir": "ripe_oranges",
        "synthetic_dir": "Ripe_img",
        "label": 0,
    },
    "unripe": {
        "real_dir": "unripe_orange",
        "synthetic_dir": "Unripe_img",
        "label": 1,
    },
}


@dataclass
class RunConfig:
    project_root: Path
    real_root: Path
    synthetic_root: Path
    output_dir: Path
    ratios: Sequence[float]
    seeds: Sequence[int]
    train_frac: float
    val_frac: float
    test_frac: float
    real_train_subsample: float
    epochs: int
    batch_size: int
    lr: float
    num_workers: int
    model_name: str
    freeze_backbone: bool
    image_size: int
    allow_synth_replacement: bool
    scenario: str
    field_eval_distortion_prob: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PathDataset(Dataset):
    def __init__(self, items: Sequence[Tuple[str, int]], tfm: transforms.Compose):
        self.items = list(items)
        self.tfm = tfm

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        return self.tfm(img), label


class AddGaussianNoise:
    def __init__(self, std: float = 0.03):
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(tensor) * self.std
        return torch.clamp(tensor + noise, 0.0, 1.0)


class RandomJPEGCompression:
    def __init__(self, quality_min: int = 35, quality_max: int = 85):
        self.quality_min = quality_min
        self.quality_max = quality_max

    def __call__(self, img: Image.Image) -> Image.Image:
        quality = random.randint(self.quality_min, self.quality_max)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        compressed = Image.open(buf).convert("RGB")
        return compressed


class RandomRectangleOcclusion:
    def __init__(self, max_frac: float = 0.18):
        self.max_frac = max_frac

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        _, h, w = tensor.shape
        occ_h = max(1, int(random.uniform(0.06, self.max_frac) * h))
        occ_w = max(1, int(random.uniform(0.06, self.max_frac) * w))
        y0 = random.randint(0, max(0, h - occ_h))
        x0 = random.randint(0, max(0, w - occ_w))
        tensor[:, y0 : y0 + occ_h, x0 : x0 + occ_w] = 0.0
        return tensor


def list_images(folder: Path) -> List[str]:
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
    files: List[str] = []
    for p in patterns:
        files.extend([str(x) for x in folder.glob(p)])
    return sorted(files)


def split_real_images(cfg: RunConfig, seed: int) -> Dict[str, Dict[str, List[str]]]:
    if not math.isclose(cfg.train_frac + cfg.val_frac + cfg.test_frac, 1.0, rel_tol=1e-6):
        raise ValueError("train_frac + val_frac + test_frac must equal 1.0")

    splits: Dict[str, Dict[str, List[str]]] = {}
    for cls_name, cls_info in CLASS_MAP.items():
        real_dir = cfg.real_root / cls_info["real_dir"]
        imgs = list_images(real_dir)
        if len(imgs) < 10:
            raise RuntimeError(f"Not enough real images for class {cls_name} at {real_dir}")

        train, temp = train_test_split(
            imgs,
            test_size=(1.0 - cfg.train_frac),
            random_state=seed,
            shuffle=True,
        )

        # Split temp into val/test using relative fraction.
        val_ratio_within_temp = cfg.val_frac / (cfg.val_frac + cfg.test_frac)
        val, test = train_test_split(
            temp,
            test_size=(1.0 - val_ratio_within_temp),
            random_state=seed,
            shuffle=True,
        )

        splits[cls_name] = {"train": train, "val": val, "test": test}

    return splits


def sample_synthetic(
    candidates: Sequence[str],
    needed: int,
    replacement: bool,
    rng: random.Random,
) -> List[str]:
    if needed <= 0:
        return []
    if len(candidates) == 0:
        raise RuntimeError("Synthetic candidates are empty but synthetic samples were requested")

    if replacement:
        return [rng.choice(candidates) for _ in range(needed)]

    if needed > len(candidates):
        raise RuntimeError(
            f"Requested {needed} synthetic samples without replacement but only {len(candidates)} available"
        )

    return rng.sample(list(candidates), needed)


def build_items_for_run(
    cfg: RunConfig,
    splits: Dict[str, Dict[str, List[str]]],
    ratio: float,
    seed: int,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]], List[Tuple[str, int]], Dict[str, int]]:
    rng = random.Random(seed)

    train_items: List[Tuple[str, int]] = []
    val_items: List[Tuple[str, int]] = []
    test_items: List[Tuple[str, int]] = []

    counts = {
        "train_real_total": 0,
        "train_synth_total": 0,
        "val_real_total": 0,
        "test_real_total": 0,
    }

    for cls_name, cls_info in CLASS_MAP.items():
        label = cls_info["label"]
        train_real = list(splits[cls_name]["train"])
        if cfg.real_train_subsample < 1.0:
            keep_n = max(1, int(len(train_real) * cfg.real_train_subsample))
            train_real = rng.sample(train_real, keep_n)

        val_real = splits[cls_name]["val"]
        test_real = splits[cls_name]["test"]

        synth_dir = cfg.synthetic_root / cls_info["synthetic_dir"]
        synth_candidates = list_images(synth_dir)
        synth_n = int(round(len(train_real) * ratio))
        synth_imgs = sample_synthetic(
            synth_candidates,
            synth_n,
            replacement=cfg.allow_synth_replacement,
            rng=rng,
        )

        train_items.extend([(p, label) for p in train_real])
        train_items.extend([(p, label) for p in synth_imgs])
        val_items.extend([(p, label) for p in val_real])
        test_items.extend([(p, label) for p in test_real])

        counts[f"train_real_{cls_name}"] = len(train_real)
        counts[f"train_synth_{cls_name}"] = len(synth_imgs)
        counts[f"val_real_{cls_name}"] = len(val_real)
        counts[f"test_real_{cls_name}"] = len(test_real)

        counts["train_real_total"] += len(train_real)
        counts["train_synth_total"] += len(synth_imgs)
        counts["val_real_total"] += len(val_real)
        counts["test_real_total"] += len(test_real)

    rng.shuffle(train_items)
    rng.shuffle(val_items)
    rng.shuffle(test_items)
    return train_items, val_items, test_items, counts


def build_model(model_name: str, freeze_backbone: bool, device: torch.device) -> nn.Module:
    model_name = model_name.lower().strip()

    if model_name == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        if freeze_backbone:
            for p in model.features.parameters():
                p.requires_grad = False
        model.classifier = nn.Linear(model.classifier.in_features, 2)
    elif model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        if freeze_backbone:
            for p in model.parameters():
                p.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, 2)
    else:
        raise ValueError("Unsupported model_name. Use densenet121 or resnet18")

    return model.to(device)


def build_transforms(cfg: RunConfig) -> Tuple[transforms.Compose, transforms.Compose]:
    if cfg.scenario == "clean":
        train_tfm = transforms.Compose(
            [
                transforms.RandomResizedCrop(cfg.image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor(),
            ]
        )

        eval_tfm = transforms.Compose(
            [
                transforms.Resize(int(cfg.image_size * 1.15)),
                transforms.CenterCrop(cfg.image_size),
                transforms.ToTensor(),
            ]
        )
        return train_tfm, eval_tfm

    if cfg.scenario == "field":
        # Simulates orchard artifacts: motion blur, compression, lighting shifts, occlusions, sensor noise.
        train_tfm = transforms.Compose(
            [
                transforms.RandomResizedCrop(cfg.image_size, scale=(0.6, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomPerspective(distortion_scale=0.20, p=0.35),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.6, 1.8))], p=0.35),
                transforms.RandomApply([RandomJPEGCompression(quality_min=30, quality_max=80)], p=0.35),
                transforms.ColorJitter(brightness=0.35, contrast=0.30, saturation=0.20, hue=0.03),
                transforms.ToTensor(),
                transforms.RandomApply([RandomRectangleOcclusion(max_frac=0.22)], p=0.30),
                transforms.RandomApply([AddGaussianNoise(std=0.04)], p=0.35),
                transforms.RandomErasing(p=0.18, scale=(0.02, 0.12)),
            ]
        )

        eval_tfm = transforms.Compose(
            [
                transforms.Resize(int(cfg.image_size * 1.20)),
                transforms.CenterCrop(cfg.image_size),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.5, 1.4))], p=cfg.field_eval_distortion_prob),
                transforms.RandomApply([RandomJPEGCompression(quality_min=35, quality_max=85)], p=cfg.field_eval_distortion_prob),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.28, contrast=0.25, saturation=0.18, hue=0.02)],
                    p=cfg.field_eval_distortion_prob,
                ),
                transforms.ToTensor(),
                transforms.RandomApply([RandomRectangleOcclusion(max_frac=0.18)], p=max(0.30, cfg.field_eval_distortion_prob * 0.65)),
                transforms.RandomApply([AddGaussianNoise(std=0.03)], p=cfg.field_eval_distortion_prob),
            ]
        )
        return train_tfm, eval_tfm

    raise ValueError("scenario must be clean or field")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    epochs: int,
) -> float:
    model.train()
    losses: List[float] = []
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False)

    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return float(np.mean(losses)) if losses else float("nan")


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    preds: List[int] = []
    gts: List[int] = []

    eval_start = time.perf_counter()
    with torch.no_grad():
        pbar = tqdm(loader, desc="Evaluating", leave=False)
        for x, y in pbar:
            x = x.to(device)
            y_hat = torch.argmax(model(x), dim=1).cpu().numpy().tolist()
            preds.extend(y_hat)
            gts.extend(y.numpy().tolist())

    acc = accuracy_score(gts, preds)
    f1 = f1_score(gts, preds, average="binary")
    elapsed = time.perf_counter() - eval_start
    print(f"Evaluation finished in {elapsed:.1f}s | acc={acc:.4f} f1={f1:.4f}")
    return float(acc), float(f1)


def run_experiments(cfg: RunConfig) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_tfm, eval_tfm = build_transforms(cfg)

    run_rows: List[Dict] = []
    count_rows: List[Dict] = []
    timeline_rows: List[Dict] = []
    epoch_rows: List[Dict] = []

    total_runs = len(cfg.seeds) * len(cfg.ratios)
    run_counter = 0
    run_durations: List[float] = []

    for seed in cfg.seeds:
        print(f"\n=== Seed {seed} ===")
        set_seed(seed)
        splits = split_real_images(cfg, seed)

        for ratio in cfg.ratios:
            run_counter += 1
            run_started = datetime.now()
            run_start_perf = time.perf_counter()
            print(f"\n--- Ratio {ratio:.2f} ({int(ratio * 100)}%) | run {run_counter}/{total_runs} ---")
            train_items, val_items, test_items, counts = build_items_for_run(cfg, splits, ratio, seed)

            train_loader = DataLoader(
                PathDataset(train_items, train_tfm),
                batch_size=cfg.batch_size,
                shuffle=True,
                num_workers=cfg.num_workers,
                pin_memory=(device.type == "cuda"),
            )
            val_loader = DataLoader(
                PathDataset(val_items, eval_tfm),
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                pin_memory=(device.type == "cuda"),
            )
            test_loader = DataLoader(
                PathDataset(test_items, eval_tfm),
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                pin_memory=(device.type == "cuda"),
            )

            model = build_model(cfg.model_name, cfg.freeze_backbone, device)
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(params, lr=cfg.lr)
            loss_fn = nn.CrossEntropyLoss()

            best_val_acc = -1.0
            best_state = None

            for epoch in range(1, cfg.epochs + 1):
                epoch_start = time.perf_counter()
                train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, epoch, cfg.epochs)
                val_acc, val_f1 = evaluate(model, val_loader, device)
                epoch_elapsed = time.perf_counter() - epoch_start
                print(
                    f"Epoch {epoch:02d}/{cfg.epochs} | "
                    f"loss={train_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} "
                    f"epoch_time={epoch_elapsed:.1f}s"
                )

                epoch_rows.append(
                    {
                        "seed": seed,
                        "ratio": ratio,
                        "ratio_pct": int(ratio * 100),
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val_acc": val_acc,
                        "val_f1": val_f1,
                        "epoch_seconds": epoch_elapsed,
                    }
                )

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

            if best_state is not None:
                model.load_state_dict(best_state)

            test_acc, test_f1 = evaluate(model, test_loader, device)
            print(f"Final Test | acc={test_acc:.4f} f1={test_f1:.4f}")

            run_elapsed = time.perf_counter() - run_start_perf
            run_durations.append(run_elapsed)
            avg_run_sec = float(np.mean(run_durations))
            remain_runs = total_runs - run_counter
            eta_sec = avg_run_sec * remain_runs
            print(
                f"Run timing | elapsed={run_elapsed:.1f}s avg={avg_run_sec:.1f}s ETA_remaining={eta_sec/60.0:.1f} min"
            )

            row = {
                "seed": seed,
                "ratio": ratio,
                "ratio_pct": int(ratio * 100),
                "model": cfg.model_name,
                "freeze_backbone": cfg.freeze_backbone,
                "epochs": cfg.epochs,
                "batch_size": cfg.batch_size,
                "lr": cfg.lr,
                "scenario": cfg.scenario,
                "val_best_acc": best_val_acc,
                "test_acc": test_acc,
                "test_f1": test_f1,
                "run_seconds": run_elapsed,
            }
            row.update(counts)
            run_rows.append(row)
            count_rows.append(
                {
                    "seed": seed,
                    "ratio": ratio,
                    "ratio_pct": int(ratio * 100),
                    **counts,
                }
            )
            timeline_rows.append(
                {
                    "seed": seed,
                    "ratio": ratio,
                    "ratio_pct": int(ratio * 100),
                    "run_index": run_counter,
                    "run_started_at": run_started.isoformat(timespec="seconds"),
                    "run_finished_at": datetime.now().isoformat(timespec="seconds"),
                    "run_seconds": run_elapsed,
                    "eta_seconds_after_run": eta_sec,
                }
            )

    run_df = pd.DataFrame(run_rows).sort_values(["ratio", "seed"]).reset_index(drop=True)
    counts_df = pd.DataFrame(count_rows).sort_values(["ratio", "seed"]).reset_index(drop=True)
    timeline_df = pd.DataFrame(timeline_rows).sort_values(["run_index"]).reset_index(drop=True)
    epoch_df = pd.DataFrame(epoch_rows).sort_values(["ratio", "seed", "epoch"]).reset_index(drop=True)

    summary = (
        run_df.groupby(["ratio", "ratio_pct"], as_index=False)
        .agg(
            acc_mean=("test_acc", "mean"),
            acc_std=("test_acc", "std"),
            f1_mean=("test_f1", "mean"),
            f1_std=("test_f1", "std"),
            n_runs=("test_acc", "count"),
        )
        .sort_values("ratio")
        .reset_index(drop=True)
    )
    summary["acc_ci95"] = 1.96 * (summary["acc_std"].fillna(0.0) / np.sqrt(summary["n_runs"]))
    summary["f1_ci95"] = 1.96 * (summary["f1_std"].fillna(0.0) / np.sqrt(summary["n_runs"]))

    run_path = cfg.output_dir / "run_results.csv"
    summary_path = cfg.output_dir / "summary_results.csv"
    counts_path = cfg.output_dir / "dataset_counts.csv"
    cfg_path = cfg.output_dir / "run_config.json"
    timeline_path = cfg.output_dir / "timeline_runs.csv"
    epoch_path = cfg.output_dir / "timeline_epochs.csv"

    run_df.to_csv(run_path, index=False)
    summary.to_csv(summary_path, index=False)
    counts_df.to_csv(counts_path, index=False)
    timeline_df.to_csv(timeline_path, index=False)
    epoch_df.to_csv(epoch_path, index=False)

    with cfg_path.open("w", encoding="utf-8") as f:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()}, f, indent=2)

    print("\nSaved:")
    print(run_path)
    print(summary_path)
    print(counts_path)
    print(cfg_path)
    print(timeline_path)
    print(epoch_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local reproducible GAN-ratio experiments")
    parser.add_argument("--project-root", type=str, default=".")
    parser.add_argument("--real-root", type=str, default="Data/orange_dataset_cleaned")
    parser.add_argument("--synthetic-root", type=str, default="Data/GAN")
    parser.add_argument("--output-dir", type=str, default="local_repro/outputs")
    parser.add_argument("--ratios", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0, 2.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1337, 2025, 7, 99])
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--real-train-subsample", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--model-name", type=str, default="densenet121", choices=["densenet121", "resnet18"])
    parser.add_argument("--freeze-backbone", action="store_true", default=False)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--no-synth-replacement", action="store_true")
    parser.add_argument("--scenario", type=str, default="field", choices=["clean", "field"])
    parser.add_argument("--field-eval-distortion-prob", type=float, default=0.60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()

    cfg = RunConfig(
        project_root=project_root,
        real_root=(project_root / args.real_root).resolve(),
        synthetic_root=(project_root / args.synthetic_root).resolve(),
        output_dir=(project_root / args.output_dir).resolve(),
        ratios=args.ratios,
        seeds=args.seeds,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        real_train_subsample=args.real_train_subsample,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        model_name=args.model_name,
        freeze_backbone=args.freeze_backbone,
        image_size=args.image_size,
        allow_synth_replacement=(not args.no_synth_replacement),
        scenario=args.scenario,
        field_eval_distortion_prob=args.field_eval_distortion_prob,
    )

    print("Experiment config:")
    print(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()}, indent=2))

    run_experiments(cfg)


if __name__ == "__main__":
    main()
