# scripts/generate_images.py
#
# Generate synthetic images using the trained WGAN-GP models
# (trained on CROPPED oranges) so that each class (ripe, unripe)
# reaches TARGET_PER_CLASS images.

import time
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.utils import save_image

# Import train indices to detect how many real images exist
from split_dataset import train_ripe_idx, train_unripe_idx

# ============================================================
# Config
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nz = 100
ngf = 64
TARGET_PER_CLASS = 1000
BATCH_SIZE = 64

# Read real training counts
n_train_ripe = len(train_ripe_idx)
n_train_unripe = len(train_unripe_idx)

extra_ripe = max(0, TARGET_PER_CLASS - n_train_ripe)
extra_unripe = max(0, TARGET_PER_CLASS - n_train_unripe)

print("=" * 60)
print(f"Real ripe train images:   {n_train_ripe}")
print(f"Real unripe train images: {n_train_unripe}")
print(f"Target per class:         {TARGET_PER_CLASS}")
print("-" * 60)
print(f"Need to generate {extra_ripe} ripe images")
print(f"Need to generate {extra_unripe} unripe images")
print("-" * 60)
print(f"Using Device: {device}")
if device.type == "cuda":
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
print("=" * 60)


# ============================================================
# Generator (MUST match training architecture)
# ============================================================

class Generator(nn.Module):
    """Same architecture as in train_gans_wgangp_both.py"""
    def __init__(self, nz, ngf, nc=3):
        super().__init__()
        self.main = nn.Sequential(
            # input Z: (nz) -> (ngf*8) x 4 x 4
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            # (ngf*8) x 4 x 4 -> (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # (ngf*4) x 8 x 8 -> (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # (ngf*2) x 16 x 16 -> (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # (ngf) x 32 x 32 -> 3 x 64 x 64
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.main(z)


def load_generator(checkpoint_path: str) -> Generator:
    """
    Load generator from WGAN-GP checkpoint.

    Checkpoint format (from train_gans_wgangp_both.py):
    {
        'epoch': int,
        'netG_state_dict': state_dict,
        ...
    }
    or older format: just state_dict.
    """
    print(f"Loading checkpoint: {checkpoint_path}")

    netG = Generator(nz, ngf).to(device)

    # Handle PyTorch 2.6+ (weights_only=True by default) and older versions
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,  # important for your earlier error
        )
    except TypeError:
        # Older PyTorch without weights_only arg
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "netG_state_dict" in checkpoint:
        netG.load_state_dict(checkpoint["netG_state_dict"])
        epoch_loaded = checkpoint.get("epoch", "UNKNOWN")
        print(f"✓ Loaded generator state from epoch {epoch_loaded}")
    else:
        # Old format: checkpoint is the state_dict
        netG.load_state_dict(checkpoint)
        print("✓ Loaded generator weights (state_dict only)")

    netG.eval()
    return netG


# ============================================================
# Time formatting helper
# ============================================================

def format_time(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:02d}h:{m:02d}m:{s:02d}s"
    return f"{m:02d}m:{s:02d}s"


# ============================================================
# Image Generation with ETA
# ============================================================

def generate_images(netG, out_dir, prefix, num_images):
    """
    Generate synthetic images using trained generator.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if num_images <= 0:
        print(f"✓ No images required for {prefix}. Skipping.\n")
        return

    print(f"\n{'='*60}")
    print(f"Generating {num_images} images for '{prefix}'")
    print(f"Saving to: {out_dir}")
    print(f"{'='*60}")

    start_time = time.time()
    count = 0
    img_idx = 1

    total_batches = (num_images + BATCH_SIZE - 1) // BATCH_SIZE
    batch_counter = 0

    with torch.no_grad():
        while count < num_images:
            batch_counter += 1
            batch = min(BATCH_SIZE, num_images - count)

            noise = torch.randn(batch, nz, 1, 1, device=device)
            fake = netG(noise).cpu()
            fake = fake * 0.5 + 0.5  # [-1,1] → [0,1]

            for i in range(batch):
                save_path = out_dir / f"{prefix}_{img_idx:04d}.png"
                save_image(fake[i], save_path)
                img_idx += 1
                count += 1

            elapsed = time.time() - start_time
            batches_left = total_batches - batch_counter
            eta_time = (elapsed / batch_counter) * batches_left if batch_counter > 0 else 0
            progress = (count / num_images) * 100

            print(
                f"[{prefix}] Progress: {count}/{num_images} ({progress:.1f}%) | "
                f"Batch {batch_counter}/{total_batches} | "
                f"Elapsed: {format_time(elapsed)} | "
                f"ETA: {format_time(eta_time)}"
            )

    total_time = time.time() - start_time
    print(f"✓ Generated {num_images} images in {format_time(total_time)}")
    print(f"{'='*60}\n")


# ============================================================
# Quality Check Function (Optional)
# ============================================================

def quality_check_samples(netG, prefix, num_samples=16):
    """
    Generate a grid of sample images for visual quality inspection.
    """
    print(f"Generating quality check samples for {prefix}...")

    with torch.no_grad():
        noise = torch.randn(num_samples, nz, 1, 1, device=device)
        samples = netG(noise).cpu()

    samples = samples * 0.5 + 0.5  # [-1,1] → [0,1]

    grid_path = Path(f"./quality_check_{prefix}.png")
    save_image(samples, grid_path, nrow=4, padding=2)
    print(f"✓ Quality check saved to: {grid_path}\n")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("WGAN-GP Synthetic Image Generation (CROPPED)")
    print("="*60 + "\n")

    # --------------------------------------------------------
    # Preferred checkpoints (latest epochs on CROPPED data)
    # --------------------------------------------------------
    ripe_ckpt   = "./gan_ripe_wgangp_cropped/checkpoint_ripe_epoch0300.pth"
    unripe_ckpt = "./gan_unripe_wgangp_cropped/checkpoint_unripe_epoch0300.pth"

    # --------------------------------------------------------
    # Fallback: if exact files are missing, pick latest matching
    # --------------------------------------------------------
    if not Path(ripe_ckpt).exists():
        print(f"⚠️ Ripe checkpoint not found at {ripe_ckpt}")
        ripe_dir = Path("./gan_ripe_wgangp_cropped")
        checkpoints = sorted(ripe_dir.glob("checkpoint_ripe_epoch*.pth"))
        if checkpoints:
            ripe_ckpt = str(checkpoints[-1])
            print(f"✓ Using latest ripe checkpoint: {ripe_ckpt}")
        else:
            print("❌ No ripe checkpoints found!")
            exit(1)

    if not Path(unripe_ckpt).exists():
        print(f"⚠️ Unripe checkpoint not found at {unripe_ckpt}")
        unripe_dir = Path("./gan_unripe_wgangp_cropped")
        checkpoints = sorted(unripe_dir.glob("checkpoint_unripe_epoch*.pth"))
        if checkpoints:
            unripe_ckpt = str(checkpoints[-1])
            print(f"✓ Using latest unripe checkpoint: {unripe_ckpt}")
        else:
            print("❌ No unripe checkpoints found!")
            exit(1)

    print()

    # Load GAN models
    netG_ripe = load_generator(ripe_ckpt)
    netG_unripe = load_generator(unripe_ckpt)

    # Quality check grids
    print("\n" + "-"*60)
    print("Quality Check: Generating sample grids")
    print("-"*60)
    quality_check_samples(netG_ripe, "ripe", num_samples=16)
    quality_check_samples(netG_unripe, "unripe", num_samples=16)

    # Generate synthetic images
    print("\n" + "-"*60)
    print("Generating Synthetic Training Data")
    print("-"*60)

    generate_images(netG_ripe,   "./generated_augmented/ripe",   "ripe",   extra_ripe)
    generate_images(netG_unripe, "./generated_augmented/unripe", "unripe", extra_unripe)

    print("\n" + "="*60)
    print("🎉 All synthetic image generation complete!")
    print("="*60)
    print(f"Ripe images saved to:   ./generated_augmented/ripe/")
    print(f"Unripe images saved to: ./generated_augmented/unripe/")
    print(f"\nTotal synthetic images generated: {extra_ripe + extra_unripe}")
    print("="*60 + "\n")
