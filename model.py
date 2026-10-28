"""
DocRestore AI - Model Architecture
DocRestore AI: Document Shadow Removal Using Pix2Pix GAN

Implements:
1. UNetGenerator: Symmetrical 8-stage U-Net Generator with skip connections
   matching the Google Colab trained checkpoint contract.
2. PatchGANDiscriminator: 70x70 receptive field discriminator for conditional adversarial training.
3. Checkpoint loading with strict parameter validation and DataParallel support.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn

from config import (
    FALLBACK_CHECKPOINT_PATH,
    GENERATOR_CHECKPOINT_PATH,
    IN_CHANNELS,
    NUM_DISCRIMINATOR_FILTERS,
    NUM_GENERATOR_FILTERS,
    OUT_CHANNELS,
    get_device,
)


class Block(nn.Module):
    """
    Modular building block for Pix2Pix U-Net:
    - Down: Conv2d -> BatchNorm2d -> LeakyReLU(0.2)
    - Up: ConvTranspose2d -> BatchNorm2d -> [Dropout(0.5)] -> ReLU
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        down: bool = True,
        act: str = "relu",
        use_dropout: bool = False,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect")
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True) if act == "relu" else nn.LeakyReLU(0.2, inplace=True),
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x


class UNetGenerator(nn.Module):
    """
    Standard Pix2Pix U-Net Generator for document shadow restoration.
    Preserves text stroke sharpness and layout via symmetrical skip connections.
    """

    def __init__(
        self,
        in_channels: int = IN_CHANNELS,
        out_channels: int = OUT_CHANNELS,
        features: int = NUM_GENERATOR_FILTERS,
        num_filters: Optional[int] = None,
        bottleneck_bias: bool = False,
    ):
        super().__init__()
        if num_filters is not None:
            features = num_filters
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.down2 = Block(features, features * 2, down=True, act="leaky", use_dropout=False)
        self.down3 = Block(features * 2, features * 4, down=True, act="leaky", use_dropout=False)
        self.down4 = Block(features * 4, features * 8, down=True, act="leaky", use_dropout=False)
        self.down5 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False)
        self.down6 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False)
        self.down7 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1, bias=bottleneck_bias),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(features * 8, features * 8, 4, 2, 1, bias=bottleneck_bias),
            nn.BatchNorm2d(features * 8),
            nn.ReLU(inplace=True),
        )

        self.up1 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True)
        self.up2 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True)
        self.up3 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True)
        self.up4 = Block(features * 8 * 2, features * 4, down=False, act="relu", use_dropout=False)
        self.up5 = Block(features * 4 * 2, features * 2, down=False, act="relu", use_dropout=False)
        self.up6 = Block(features * 2 * 2, features, down=False, act="relu", use_dropout=False)

        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, out_channels, 4, 2, 1),
            nn.Tanh(),  # Normalized output range [-1, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)

        b = self.bottleneck(d7)

        up1 = self.up1(torch.cat([b, d7], dim=1))
        up2 = self.up2(torch.cat([up1, d6], dim=1))
        up3 = self.up3(torch.cat([up2, d5], dim=1))
        up4 = self.up4(torch.cat([up3, d4], dim=1))
        up5 = self.up5(torch.cat([up4, d3], dim=1))
        up6 = self.up6(torch.cat([up5, d2], dim=1))

        return self.final_up(torch.cat([up6, d1], dim=1))


class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN 70x70 Discriminator for Pix2Pix.
    Classifies whether each 70x70 image patch is real or generated.
    Takes conditional input: degraded image (3ch) + candidate image (3ch) = 6ch.
    """

    def __init__(
        self,
        in_channels: int = IN_CHANNELS * 2,
        num_filters: int = NUM_DISCRIMINATOR_FILTERS,
    ):
        super().__init__()
        nf = num_filters

        def discriminator_block(in_c: int, out_c: int, stride: int = 2, normalize: bool = True):
            layers = [
                nn.Conv2d(in_c, out_c, kernel_size=4, stride=stride, padding=1, bias=not normalize)
            ]
            if normalize:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            # Layer 1: 6 -> 64 (no norm)
            *discriminator_block(in_channels, nf, stride=2, normalize=False),
            # Layer 2: 64 -> 128
            *discriminator_block(nf, nf * 2, stride=2, normalize=True),
            # Layer 3: 128 -> 256
            *discriminator_block(nf * 2, nf * 4, stride=2, normalize=True),
            # Layer 4: 256 -> 512, stride 1
            *discriminator_block(nf * 4, nf * 8, stride=1, normalize=True),
            # Layer 5: Output patch classification map
            nn.Conv2d(nf * 8, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, condition: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        # Concatenate condition and candidate along channel axis
        x = torch.cat([condition, candidate], dim=1)
        return self.model(x)


def is_checkpoint_available(
    checkpoint_path: Optional[Union[str, Path]] = None,
) -> Tuple[bool, Path]:
    """
    Checks if a trained generator checkpoint file is present on disk.
    If a specific path is requested, tests that exact path.
    Otherwise checks `models/generator.pth` and secondary fallback `checkpoints/best_generator.pth`.
    """
    if checkpoint_path is not None:
        target_path = Path(checkpoint_path)
        return target_path.is_file(), target_path

    if GENERATOR_CHECKPOINT_PATH.is_file():
        return True, GENERATOR_CHECKPOINT_PATH

    if FALLBACK_CHECKPOINT_PATH.is_file():
        return True, FALLBACK_CHECKPOINT_PATH

    return False, GENERATOR_CHECKPOINT_PATH


def load_generator_checkpoint(
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None,
) -> UNetGenerator:
    """
    Loads trained Generator weights into UNetGenerator.

    Supports:
    - Raw Generator state_dict
    - Training checkpoint dictionary containing `generator_state_dict` or `state_dict`
    - Checkpoints saved with `nn.DataParallel` ('module.' prefix removal)

    Raises:
    - FileNotFoundError if checkpoint is not found.
    - RuntimeError if keys or layer dimensions do not match.
    """
    if device is None:
        device = get_device()

    available, resolved_path = is_checkpoint_available(checkpoint_path)
    if not available:
        raise FileNotFoundError(
            f"Generator checkpoint not found at: '{resolved_path}'. "
            f"Please place your trained weights file at '{GENERATOR_CHECKPOINT_PATH}'."
        )

    checkpoint = torch.load(resolved_path, map_location=device, weights_only=False)

    # Extract state_dict if wrapped in a training checkpoint dictionary
    if isinstance(checkpoint, dict):
        if "generator_state_dict" in checkpoint:
            state_dict = checkpoint["generator_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Clean potential DataParallel 'module.' prefix
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            cleaned_state_dict[k[7:]] = v
        else:
            cleaned_state_dict[k] = v

    # Auto-detect whether bottleneck conv layers were trained with bias
    has_bottleneck_bias = "bottleneck.0.bias" in cleaned_state_dict

    model = UNetGenerator(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        features=NUM_GENERATOR_FILTERS,
        bottleneck_bias=has_bottleneck_bias,
    )

    # Load with strict parameter matching
    model.load_state_dict(cleaned_state_dict, strict=True)
    model.to(device)
    model.eval()

    return model
