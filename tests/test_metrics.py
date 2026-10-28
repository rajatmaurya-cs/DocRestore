"""
Tests for DocRestore AI - Quantitative Metrics (PSNR & SSIM)
"""

import numpy as np
from PIL import Image
import pytest

from evaluation.metrics import (
    calculate_document_metrics,
    calculate_psnr,
    calculate_ssim,
    ensure_numpy_rgb,
)


def test_ensure_numpy_rgb():
    """Verify conversion from various image types to uint8 RGB."""
    pil_gray = Image.new("L", (50, 50), 128)
    arr = ensure_numpy_rgb(pil_gray)
    assert arr.shape == (50, 50, 3)
    assert arr.dtype == np.uint8

    arr_float = np.ones((50, 50, 3), dtype=np.float32)
    arr_converted = ensure_numpy_rgb(arr_float)
    assert arr_converted.dtype == np.uint8
    assert arr_converted.max() == 255


def test_psnr_identical_images():
    """PSNR of identical images should be extremely high (approaching infinity)."""
    img = Image.new("RGB", (100, 100), (255, 255, 255))
    score = calculate_psnr(img, img)
    # Skimage returns infinity or a very large number for identical inputs
    assert score > 80.0 or np.isinf(score)


def test_ssim_identical_images():
    """SSIM of identical images must equal exactly 1.0."""
    img = Image.new("RGB", (100, 100), (200, 200, 200))
    score = calculate_ssim(img, img)
    assert pytest.approx(score, rel=1e-4) == 1.0


def test_metrics_on_degraded_image():
    """Verify degraded/shadowed document yields lower PSNR and SSIM."""
    clean = np.ones((100, 100, 3), dtype=np.uint8) * 240
    # Simulate a cast shadow by reducing illumination in a rectangular region
    shadowed = clean.copy()
    shadowed[20:80, 20:80] = 60

    metrics = calculate_document_metrics(clean, shadowed)

    assert "psnr_db" in metrics
    assert "ssim" in metrics
    assert "mae_pixel" in metrics
    assert metrics["psnr_db"] < 40.0
    assert 0.0 <= metrics["ssim"] < 1.0
    assert metrics["mae_pixel"] > 0.0


def test_metrics_dimension_mismatch_handling():
    """Verify metrics calculation gracefully handles different resolutions via resizing."""
    clean = Image.new("RGB", (200, 200), (250, 250, 250))
    restored = Image.new("RGB", (100, 100), (250, 250, 250))

    metrics = calculate_document_metrics(clean, restored)
    assert metrics["ssim"] > 0.95
