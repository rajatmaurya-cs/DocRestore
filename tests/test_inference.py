"""
Tests for DocRestore AI - Image Preprocessing, Padding, and Inference Integrity
"""

from io import BytesIO
import numpy as np
from PIL import Image
import pytest
import torch

from inference import (
    open_document_image,
    postprocess_tensor,
    preprocess_document,
    restore_document_image,
)
from model import UNetGenerator


def test_open_document_image_formats():
    """Verify open_document_image handles RGB, RGBA, and in-memory bytes."""
    # RGBA image
    rgba_pil = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
    opened_rgb = open_document_image(rgba_pil)
    assert opened_rgb.mode == "RGB"
    assert opened_rgb.size == (100, 100)

    # BytesIO image
    buf = BytesIO()
    rgba_pil.save(buf, format="PNG")
    buf.seek(0)
    opened_bytes = open_document_image(buf)
    assert opened_bytes.mode == "RGB"
    assert opened_bytes.size == (100, 100)


def test_preprocess_document_letterbox_shapes_and_values():
    """Verify preprocess_document produces correctly normalized 256x256 tensor."""
    # Non-square document (e.g. 800x400)
    doc_img = Image.new("RGB", (800, 400), (200, 200, 200))
    tensor, meta = preprocess_document(doc_img, target_size=(256, 256), mode="letterbox")

    assert tensor.shape == (1, 3, 256, 256)
    assert tensor.dtype == torch.float32
    # Normalized into [-1.0, 1.0]
    assert tensor.min().item() >= -1.0 - 1e-4
    assert tensor.max().item() <= 1.0 + 1e-4

    assert meta.orig_width == 800
    assert meta.orig_height == 400
    assert meta.new_width == 256
    assert meta.new_height == 128  # Aspect ratio 2:1 preserved
    assert meta.pad_top == (256 - 128) // 2


def test_postprocess_preserves_original_dimensions():
    """Verify postprocessing crops padding and restores exact initial dimensions."""
    orig_w, orig_h = 1200, 800
    dummy_doc = Image.new("RGB", (orig_w, orig_h), (255, 255, 255))
    _, meta = preprocess_document(dummy_doc, target_size=(256, 256), mode="letterbox")

    dummy_output = torch.zeros(1, 3, 256, 256)  # Mid-gray in [-1, 1]
    restored = postprocess_tensor(dummy_output, meta)

    assert restored.size == (orig_w, orig_h), f"Expected ({orig_w}, {orig_h}), got {restored.size}"
    assert restored.mode == "RGB"


def test_restore_document_raises_on_missing_checkpoint():
    """Verify inference raises FileNotFoundError when no trained checkpoint exists."""
    dummy_doc = Image.new("RGB", (256, 256), (255, 255, 255))
    with pytest.raises(FileNotFoundError, match="trained model checkpoint not found"):
        restore_document_image(
            dummy_doc,
            generator=None,
            checkpoint_path="non_existent/generator.pth",
        )


def test_restore_document_with_provided_generator():
    """Verify end-to-end execution flow when a generator instance is provided."""
    model = UNetGenerator(in_channels=3, out_channels=3, num_filters=64)
    model.eval()

    sample_doc = Image.new("RGB", (320, 240), (240, 240, 240))
    res = restore_document_image(sample_doc, generator=model, device=torch.device("cpu"))

    assert "restored_image" in res
    assert "inference_time_seconds" in res
    assert res["restored_image"].size == (320, 240)
    assert res["inference_time_seconds"] > 0
