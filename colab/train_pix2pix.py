"""
DocRestore AI - Google Colab Pix2Pix Training Script
DocRestore AI: Document Shadow Removal Using Pix2Pix GAN

This script trains the Pix2Pix GAN (UNetGenerator + PatchGANDiscriminator)
on paired document shadow datasets:
- Input: Shadowed document RGB [B, 3, 256, 256] in [-1, 1]
- Target: Clean document RGB [B, 3, 256, 256] in [-1, 1]
- Objective: Adversarial BCE/LSGAN Loss + 100 * L1 Reconstruction Loss
- Checkpointing: Saves best_generator.pth and training_loss_curves.png
"""

import argparse
import os
from pathlib import Path
import random
import time
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Architecture matches local model.py exactly
from model import PatchGANDiscriminator, UNetGenerator


class PairedDocumentDataset(Dataset):
    """
    Dataset for synchronized paired document images (Shadowed and Clean Target).
    Applies identical spatial transformations (e.g. Random Horizontal Flip)
    to both images simultaneously to maintain strict pixel correspondence.
    """

    def __init__(self, shadow_dir: str, clean_dir: str, image_size: int = 256, augment: bool = True):
        self.shadow_dir = Path(shadow_dir)
        self.clean_dir = Path(clean_dir)
        self.image_size = image_size
        self.augment = augment

        valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        self.shadow_paths = sorted(
            [p for p in self.shadow_dir.iterdir() if p.suffix.lower() in valid_exts]
        )

        self.pairs = []
        for s_path in self.shadow_paths:
            # Look for exact or matching stem
            clean_path = self.clean_dir / s_path.name
            if clean_path.exists():
                self.pairs.append((s_path, clean_path))
            else:
                alt_stem = s_path.stem.replace("_shadow", "_clean").replace("-shadow", "-clean")
                for ext in valid_exts:
                    cand = self.clean_dir / f"{alt_stem}{ext}"
                    if cand.exists():
                        self.pairs.append((s_path, cand))
                        break

        if len(self.pairs) == 0:
            raise RuntimeError(f"No paired images found in {shadow_dir} and {clean_dir}")

        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        s_path, c_path = self.pairs[idx]
        shadow_img = Image.open(s_path).convert("RGB")
        clean_img = Image.open(c_path).convert("RGB")

        # Synchronized resize
        shadow_img = shadow_img.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
        clean_img = clean_img.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)

        # Synchronized horizontal flip augmentation
        if self.augment and random.random() > 0.5:
            shadow_img = shadow_img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            clean_img = clean_img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        # To tensor [0, 1] then normalize to [-1, 1]
        t_shadow = transforms.ToTensor()(shadow_img)
        t_clean = transforms.ToTensor()(clean_img)

        t_shadow = self.normalize(t_shadow)
        t_clean = self.normalize(t_clean)

        return t_shadow, t_clean


def train_pix2pix(
    shadow_train_dir: str,
    clean_train_dir: str,
    epochs: int = 80,
    batch_size: int = 16,
    lr: float = 2e-4,
    lambda_l1: float = 100.0,
    save_dir: str = "checkpoints",
):
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Starting Pix2Pix Document Shadow Removal Training on device: {device}")

    # Dataset & DataLoader
    dataset = PairedDocumentDataset(shadow_train_dir, clean_train_dir, image_size=256, augment=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    print(f"Dataset loaded: {len(dataset)} paired document images ({len(dataloader)} batches/epoch).")

    # Models
    generator = UNetGenerator().to(device)
    discriminator = PatchGANDiscriminator().to(device)

    # Loss functions & Optimizers
    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_l1 = nn.L1Loss()

    optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    history = {"epoch": [], "loss_d": [], "loss_g": [], "loss_l1": []}
    best_l1 = float("inf")

    start_total_time = time.time()

    for epoch in range(1, epochs + 1):
        generator.train()
        discriminator.train()

        running_loss_d = 0.0
        running_loss_g = 0.0
        running_loss_l1 = 0.0

        for batch_idx, (real_shadow, real_clean) in enumerate(dataloader):
            real_shadow = real_shadow.to(device)
            real_clean = real_clean.to(device)
            bs = real_shadow.size(0)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_d.zero_grad()

            # Real loss: D(shadow, real_clean) -> 1
            pred_real = discriminator(real_shadow, real_clean)
            loss_d_real = criterion_gan(pred_real, torch.ones_like(pred_real))

            # Fake loss: D(shadow, G(shadow).detach()) -> 0
            fake_clean = generator(real_shadow)
            pred_fake = discriminator(real_shadow, fake_clean.detach())
            loss_d_fake = criterion_gan(pred_fake, torch.zeros_like(pred_fake))

            # Total Discriminator loss
            loss_d = (loss_d_real + loss_d_fake) * 0.5
            loss_d.backward()
            optimizer_d.step()

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_g.zero_grad()

            # Adversarial loss: D(shadow, fake_clean) -> 1
            pred_fake_g = discriminator(real_shadow, fake_clean)
            loss_g_gan = criterion_gan(pred_fake_g, torch.ones_like(pred_fake_g))

            # Reconstruction L1 loss
            loss_g_l1 = criterion_l1(fake_clean, real_clean)

            # Combined Generator objective: GAN loss + lambda * L1
            loss_g = loss_g_gan + (lambda_l1 * loss_g_l1)
            loss_g.backward()
            optimizer_g.step()

            running_loss_d += loss_d.item()
            running_loss_g += loss_g.item()
            running_loss_l1 += loss_g_l1.item()

        epoch_loss_d = running_loss_d / len(dataloader)
        epoch_loss_g = running_loss_g / len(dataloader)
        epoch_loss_l1 = running_loss_l1 / len(dataloader)

        history["epoch"].append(epoch)
        history["loss_d"].append(epoch_loss_d)
        history["loss_g"].append(epoch_loss_g)
        history["loss_l1"].append(epoch_loss_l1)

        print(
            f"Epoch [{epoch:03d}/{epochs:03d}] | "
            f"Loss D: {epoch_loss_d:.4f} | "
            f"Loss G: {epoch_loss_g:.4f} | "
            f"L1: {epoch_loss_l1:.4f}"
        )

        # Save best generator checkpoint based on lowest reconstruction L1 loss
        if epoch_loss_l1 < best_l1:
            best_l1 = epoch_loss_l1
            best_path = save_path / "best_generator.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "generator_state_dict": generator.state_dict(),
                    "l1_loss": epoch_loss_l1,
                    "generator_loss": epoch_loss_g,
                },
                best_path,
            )
            # Also save raw state_dict for maximum convenience
            torch.save(generator.state_dict(), save_path / "generator.pth")

    total_time = time.time() - start_total_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes.")
    print(f"Best checkpoint saved at: {save_path / 'generator.pth'}")

    # Plot loss curves
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["epoch"], history["loss_g"], label="Generator Loss", color="blue")
    plt.plot(history["epoch"], history["loss_d"], label="Discriminator Loss", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("GAN Adversarial Loss Curves")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history["epoch"], history["loss_l1"], label="L1 Reconstruction Loss", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("L1 Pixel Loss")
    plt.title("L1 Reconstruction Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    curve_path = save_path / "loss_curves.png"
    plt.savefig(curve_path, dpi=200)
    plt.close()
    print(f"Training loss curves saved to: {curve_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DocRestore Pix2Pix GAN on Colab")
    parser.add_argument("--shadow_dir", type=str, required=True, help="Shadowed training images")
    parser.add_argument("--clean_dir", type=str, required=True, help="Clean ground truth images")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--save_dir", type=str, default="checkpoints")

    args = parser.parse_args()
    train_pix2pix(
        shadow_train_dir=args.shadow_dir,
        clean_train_dir=args.clean_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_dir=args.save_dir,
    )
