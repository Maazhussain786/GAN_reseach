from pathlib import Path
import random
from PIL import Image, ImageOps, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper_latex" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REAL_RIPE = ROOT / "Data" / "orange_dataset_cleaned" / "ripe_oranges"
REAL_UNRIPE = ROOT / "Data" / "orange_dataset_cleaned" / "unripe_orange"
SYN_RIPE = ROOT / "Data" / "GAN" / "Ripe_img"
SYN_UNRIPE = ROOT / "Data" / "GAN" / "Unripe_img"

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(folder: Path):
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in VALID_EXT])


def sample_images(paths, n, seed):
    if len(paths) == 0:
        return []
    rng = random.Random(seed)
    if len(paths) <= n:
        return paths
    return rng.sample(paths, n)


def make_grid(image_paths, out_path: Path, title: str, cell_size=192, cols=4, margin=24, header_h=72):
    if not image_paths:
        print(f"[WARN] No images for {title}; skipping {out_path.name}")
        return

    rows = (len(image_paths) + cols - 1) // cols
    width = cols * cell_size + (cols + 1) * margin
    height = rows * cell_size + (rows + 1) * margin + header_h

    canvas = Image.new("RGB", (width, height), color=(248, 248, 248))
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 20), title, fill=(20, 20, 20))

    for i, p in enumerate(image_paths):
        r = i // cols
        c = i % cols
        x0 = margin + c * (cell_size + margin)
        y0 = header_h + margin + r * (cell_size + margin)

        img = Image.open(p).convert("RGB")
        img = ImageOps.fit(img, (cell_size, cell_size), method=Image.Resampling.LANCZOS)
        canvas.paste(img, (x0, y0))
        draw.rectangle([x0, y0, x0 + cell_size, y0 + cell_size], outline=(180, 180, 180), width=1)

    canvas.save(out_path, format="PNG")
    print(f"[OK] Saved {out_path}")


def make_side_by_side(left_path: Path, right_path: Path, out_path: Path, left_title: str, right_title: str):
    if not left_path.exists() or not right_path.exists():
        print(f"[WARN] Missing input for {out_path.name}; skipping")
        return

    left = Image.open(left_path).convert("RGB")
    right = Image.open(right_path).convert("RGB")

    # Match heights.
    h = min(left.height, right.height)
    left = left.resize((int(left.width * h / left.height), h), Image.Resampling.LANCZOS)
    right = right.resize((int(right.width * h / right.height), h), Image.Resampling.LANCZOS)

    margin = 32
    header_h = 72
    width = left.width + right.width + margin * 3
    height = h + margin * 2 + header_h

    canvas = Image.new("RGB", (width, height), color=(250, 250, 250))
    draw = ImageDraw.Draw(canvas)

    draw.text((margin, 20), left_title, fill=(20, 20, 20))
    draw.text((margin * 2 + left.width, 20), right_title, fill=(20, 20, 20))

    canvas.paste(left, (margin, header_h))
    canvas.paste(right, (margin * 2 + left.width, header_h))

    canvas.save(out_path, format="PNG")
    print(f"[OK] Saved {out_path}")


def main():
    n = 16

    real_ripe = sample_images(list_images(REAL_RIPE), n, seed=11)
    real_unripe = sample_images(list_images(REAL_UNRIPE), n, seed=13)
    synth_ripe = sample_images(list_images(SYN_RIPE), n, seed=17)
    synth_unripe = sample_images(list_images(SYN_UNRIPE), n, seed=19)

    make_grid(real_ripe, OUT_DIR / "real_ripe_grid.png", "Real Ripe Samples")
    make_grid(real_unripe, OUT_DIR / "real_unripe_grid.png", "Real Unripe Samples")
    make_grid(synth_ripe, OUT_DIR / "synth_ripe_grid.png", "Synthetic Ripe Samples")
    make_grid(synth_unripe, OUT_DIR / "synth_unripe_grid.png", "Synthetic Unripe Samples")

    make_side_by_side(
        OUT_DIR / "real_ripe_grid.png",
        OUT_DIR / "synth_ripe_grid.png",
        OUT_DIR / "ripe_real_vs_synth.png",
        "Real Ripe",
        "Synthetic Ripe",
    )
    make_side_by_side(
        OUT_DIR / "real_unripe_grid.png",
        OUT_DIR / "synth_unripe_grid.png",
        OUT_DIR / "unripe_real_vs_synth.png",
        "Real Unripe",
        "Synthetic Unripe",
    )

    print("\nDone. Qualitative panels are in:")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
