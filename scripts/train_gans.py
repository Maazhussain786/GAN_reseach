# train_unripe_wgangp.py
#
# Train a WGAN-GP ONLY for UNRIPE oranges

import time
from pathlib import Path
import numpy as np
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torchvision import utils, transforms

# Import ONLY unripe loader & indices
from split_dataset import unripe_train_loader, train_unripe_idx

# ============================================================
# 0. Device & Seeds
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)
np.random.seed(42)
if device.type == "cuda":
    torch.cuda.manual_seed_all(42)

# ============================================================
# 1. Hyperparameters
# ============================================================

image_size = 64
nz = 100
ngf = 64
ndf = 64

num_epochs = 500
batch_size = 32

lr_g = 0.0001
lr_d = 0.0004
beta1 = 0.0
beta2 = 0.9

n_critic = 5
lambda_gp = 10
augment_prob = 0.3

# ============================================================
# 2. Data Augmentation
# ============================================================

augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
])

def augment_batch(images, prob=augment_prob):
    if np.random.rand() < prob:
        # images are in [-1,1], bring to [0,1]
        images = (images + 1) / 2
        images_list = []
        for img in images:
            img_pil = transforms.ToPILImage()(img.cpu())
            img_aug = augmentation(img_pil)
            img_tensor = transforms.ToTensor()(img_aug)
            images_list.append(img_tensor)
        images = torch.stack(images_list).to(device)
        images = images * 2 - 1
    return images

# ============================================================
# 3. Models
# ============================================================

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, nz, ngf, nc=3):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.main(z)


class Discriminator(nn.Module):
    def __init__(self, ndf, nc=3):
        super().__init__()
        self.main = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(nc, ndf, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(ndf * 8, 1, 4, 1, 0)),
        )

    def forward(self, x):
        return self.main(x).view(-1)

# ============================================================
# 4. Gradient Penalty
# ============================================================

def compute_gradient_penalty(D, real_samples, fake_samples, device):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)

    d_interpolates = D(interpolates)

    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# ============================================================
# 5. Progress Bar Helpers
# ============================================================

def print_progress_bar(current, total, prefix='', bar_length=40):
    percent = current / total
    filled = int(bar_length * percent)
    bar = '█' * filled + '░' * (bar_length - filled)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent*100:.1f}%')
    sys.stdout.flush()

def format_time(seconds):
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:02d}h:{m:02d}m:{s:02d}s"
    return f"{m:02d}m:{s:02d}s"

# ============================================================
# 6. Training Function (UNRIPE ONLY)
# ============================================================

def train_wgan_gp_unripe(dataloader, output_dir):
    label_name = "unripe"

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    netG = Generator(nz, ngf).to(device)
    netG.apply(weights_init)

    netD = Discriminator(ndf).to(device)
    netD.apply(weights_init)

    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    optimizerG = optim.Adam(netG.parameters(), lr=lr_g, betas=(beta1, beta2))
    optimizerD = optim.Adam(netD.parameters(), lr=lr_d, betas=(beta1, beta2))

    num_batches = len(dataloader)
    total_steps = num_epochs * num_batches

    print(f"\n{'='*70}")
    print(f"Training WGAN-GP for UNRIPE ORANGES")
    print(f"{'='*70}")
    print(f"Training images: {len(dataloader.dataset)}")
    print(f"Epochs: {num_epochs} | Steps: {total_steps}")
    print(f"LR_G: {lr_g} | LR_D: {lr_d} | n_critic: {n_critic}")
    print(f"{'='*70}\n")

    global_step = 0
    start_time = time.time()

    d_losses, g_losses = [], []

    for epoch in range(num_epochs):
        epoch_d_loss, epoch_g_loss = [], []

        for i, (real_images, _) in enumerate(dataloader):
            real_images = real_images.to(device)
            b_size = real_images.size(0)

            real_images = augment_batch(real_images, augment_prob)

            # Train Discriminator (Critic)
            for _ in range(n_critic):
                netD.zero_grad()

                d_real = netD(real_images).mean()

                noise = torch.randn(b_size, nz, 1, 1, device=device)
                fake_images = netG(noise).detach()
                d_fake = netD(fake_images).mean()

                gp = compute_gradient_penalty(netD, real_images, fake_images, device)

                d_loss = -d_real + d_fake + lambda_gp * gp

                d_loss.backward()
                optimizerD.step()

            # Train Generator
            netG.zero_grad()

            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake_images = netG(noise)
            g_fake = netD(fake_images).mean()

            g_loss = -g_fake

            g_loss.backward()
            optimizerG.step()

            epoch_d_loss.append(d_loss.item())
            epoch_g_loss.append(g_loss.item())

            global_step += 1

            elapsed = time.time() - start_time
            steps_left = total_steps - global_step
            time_per_step = elapsed / global_step if global_step > 0 else 0
            eta_seconds = steps_left * time_per_step

            wasserstein_d = d_real.item() - d_fake.item()

            progress_text = (
                f"[{label_name}] Epoch {epoch+1:3d}/{num_epochs} | "
                f"Step {i+1:2d}/{num_batches} | "
                f"D: {d_loss.item():6.2f} | G: {g_loss.item():6.2f} | "
                f"W: {wasserstein_d:6.2f} | "
                f"Elapsed: {format_time(elapsed)} | ETA: {format_time(eta_seconds)}"
            )

            print_progress_bar(
                epoch * num_batches + i + 1,
                total_steps,
                prefix=progress_text,
                bar_length=30,
            )

        print()  # newline after bar

        avg_d = np.mean(epoch_d_loss)
        avg_g = np.mean(epoch_g_loss)
        d_losses.append(avg_d)
        g_losses.append(avg_g)

        # Save samples every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            fake = fake * 0.5 + 0.5
            utils.save_image(
                fake,
                output_dir / f"{label_name}_fake_epoch{epoch+1:04d}.png",
                nrow=8,
            )
            print(f"    → Saved sample: {label_name}_fake_epoch{epoch+1:04d}.png")

        # Save checkpoints every 50 epochs
        if (epoch + 1) % 50 == 0 or epoch == num_epochs - 1:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "netG_state_dict": netG.state_dict(),
                    "netD_state_dict": netD.state_dict(),
                    "optimizerG_state_dict": optimizerG.state_dict(),
                    "optimizerD_state_dict": optimizerD.state_dict(),
                    "d_losses": d_losses,
                    "g_losses": g_losses,
                },
                output_dir / f"checkpoint_{label_name}_epoch{epoch+1:04d}.pth",
            )
            print(f"    → Checkpoint saved: epoch {epoch+1}")

    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"✓ Training complete for UNRIPE oranges! Time: {format_time(total_time)}")
    print(f"{'='*70}\n")

    return netG, netD

# ============================================================
# 7. Main
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("WGAN-GP Training - UNRIPE Orange Dataset")
    print("="*70)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU:  {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")
    print("="*70)
    print(f"Unripe train images: {len(train_unripe_idx)}")
    print("="*70)

    gan_unripe_dir = "./gan_unripe_wgangp_only"

    netG_unripe, netD_unripe = train_wgan_gp_unripe(unripe_train_loader, gan_unripe_dir)

    print("\n" + "="*70)
    print("🎉 UNRIPE GAN TRAINING COMPLETE!")
    print("="*70 + "\n")
