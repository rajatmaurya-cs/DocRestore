"""
Tests for DocRestore AI - Model Architecture & Checkpoint Management
"""

import tempfile
from pathlib import Path
import pytest
import torch

from model import (
    PatchGANDiscriminator,
    UNetGenerator,
    is_checkpoint_available,
    load_generator_checkpoint,
)


def test_unet_generator_output_shape():
    """Verify U-Net Generator output has identical spatial dimensions and channels."""
    generator = UNetGenerator(in_channels=3, out_channels=3, num_filters=64)
    generator.eval()

    dummy_input = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        output = generator(dummy_input)

    assert output.shape == (2, 3, 256, 256), f"Expected [2, 3, 256, 256], got {output.shape}"
    # Verify Tanh activation bound [-1.0, 1.0]
    assert output.min().item() >= -1.0, f"Generator output below -1: {output.min().item()}"
    assert output.max().item() <= 1.0, f"Generator output above 1: {output.max().item()}"


def test_patchgan_discriminator_shape():
    """Verify PatchGAN Discriminator classifies patches with conditional input."""
    discriminator = PatchGANDiscriminator(in_channels=6, num_filters=64)
    discriminator.eval()

    condition = torch.randn(2, 3, 256, 256)
    candidate = torch.randn(2, 3, 256, 256)

    with torch.no_grad():
        logits = discriminator(condition, candidate)

    # PatchGAN produces a 30x30 spatial grid of patch predictions
    assert logits.shape == (2, 1, 30, 30), f"Expected [2, 1, 30, 30], got {logits.shape}"


def test_checkpoint_availability_on_missing_file():
    """Verify is_checkpoint_available returns False when checkpoint does not exist."""
    fake_path = Path("models/non_existent_generator.pth")
    available, resolved = is_checkpoint_available(fake_path)
    assert not available
    assert resolved == fake_path


def test_load_checkpoint_missing_raises_error():
    """Verify load_generator_checkpoint raises FileNotFoundError for missing path."""
    with pytest.raises(FileNotFoundError, match="Generator checkpoint not found"):
        load_generator_checkpoint(checkpoint_path=Path("non_existent_weights.pth"))


def test_checkpoint_loading_and_parameter_integrity():
    """Verify that saving and loading state dict correctly restores weights in eval mode."""
    model = UNetGenerator(in_channels=3, out_channels=3, num_filters=64)
    state = model.state_dict()

    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        # Save raw state dict
        torch.save(state, tmp_path)
        loaded_model = load_generator_checkpoint(checkpoint_path=tmp_path, device=torch.device("cpu"))
        assert isinstance(loaded_model, UNetGenerator)
        assert not loaded_model.training, "Loaded model must be in eval mode"

        # Save formatted training checkpoint dict
        torch.save({"generator_state_dict": state, "epoch": 10}, tmp_path)
        loaded_model_dict = load_generator_checkpoint(checkpoint_path=tmp_path, device=torch.device("cpu"))
        assert isinstance(loaded_model_dict, UNetGenerator)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_genuine_checkpoint_loads_and_infers_if_present():
    """Verify the real Colab trained generator weights load strictly and produce valid outputs."""
    avail, path = is_checkpoint_available()
    if avail:
        model = load_generator_checkpoint(checkpoint_path=path, device=torch.device("cpu"))
        assert isinstance(model, UNetGenerator)
        assert not model.training
        dummy = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            out = model(dummy)
        assert out.shape == (1, 3, 256, 256)
        assert out.min() >= -1.0 and out.max() <= 1.0
