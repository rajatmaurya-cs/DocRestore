"""
DocRestore AI - Quantitative Image Metrics
DocRestore AI: Document Shadow Removal Using Pix2Pix GAN

Provides standard computer vision document quality metrics:
- Peak Signal-to-Noise Ratio (PSNR)
- Structural Similarity Index Measure (SSIM)
- Mean Absolute Error (MAE)
"""

from typing import Dict, Union
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr_func
from skimage.metrics import structural_similarity as ssim_func


def ensure_numpy_rgb(image: Union[np.ndarray, Image.Image]) -> np.ndarray:
    """Ensures input is a uint8 RGB NumPy array with shape (H, W, 3)."""
    if isinstance(image, Image.Image):
        arr = np.array(image.convert("RGB"), dtype=np.uint8)
    elif isinstance(image, np.ndarray):
        if image.ndim == 2:  # Grayscale
            arr = np.stack([image] * 3, axis=-1)
        elif image.ndim == 3 and image.shape[2] == 4:  # RGBA
            arr = image[:, :, :3]
        else:
            arr = image

        if arr.dtype != np.uint8:
            if arr.max() <= 1.0:
                arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.clip(0, 255).astype(np.uint8)
    else:
        raise TypeError(f"Expected PIL.Image or np.ndarray, got {type(image)}")

    return arr


def calculate_psnr(
    clean_target: Union[np.ndarray, Image.Image],
    restored_prediction: Union[np.ndarray, Image.Image],
) -> float:
    """
    Computes Peak Signal-to-Noise Ratio (PSNR) in decibels (dB) between
    ground-truth clean document and restored output.
    Higher is better. Typically > 25-30 dB indicates strong restoration.
    """
    target = ensure_numpy_rgb(clean_target)
    prediction = ensure_numpy_rgb(restored_prediction)

    # Harmonize shapes if slightly misaligned
    if target.shape != prediction.shape:
        pred_pil = Image.fromarray(prediction).resize(
            (target.shape[1], target.shape[0]),
            Image.Resampling.LANCZOS,
        )
        prediction = np.array(pred_pil, dtype=np.uint8)

    # If identical signals, return a standard high value (100 dB) to avoid divide-by-zero warnings
    mse = np.mean((target.astype(np.float64) - prediction.astype(np.float64)) ** 2)
    if mse == 0:
        return 100.0

    val = psnr_func(target, prediction, data_range=255)
    return float(val)


def calculate_ssim(
    clean_target: Union[np.ndarray, Image.Image],
    restored_prediction: Union[np.ndarray, Image.Image],
) -> float:
    """
    Computes Structural Similarity Index Measure (SSIM) in range [-1, 1].
    Evaluates luminance, contrast, and structural stroke preservation.
    Closer to 1.0 indicates superior textual/structural fidelity.
    """
    target = ensure_numpy_rgb(clean_target)
    prediction = ensure_numpy_rgb(restored_prediction)

    if target.shape != prediction.shape:
        pred_pil = Image.fromarray(prediction).resize(
            (target.shape[1], target.shape[0]),
            Image.Resampling.LANCZOS,
        )
        prediction = np.array(pred_pil, dtype=np.uint8)

    # channel_axis=2 for RGB images
    val = ssim_func(target, prediction, channel_axis=2, data_range=255)
    return float(val)


def calculate_document_metrics(
    clean_target: Union[np.ndarray, Image.Image],
    restored_prediction: Union[np.ndarray, Image.Image],
) -> Dict[str, float]:
    """Computes all core document restoration metrics (PSNR, SSIM, MAE)."""
    target = ensure_numpy_rgb(clean_target)
    pred = ensure_numpy_rgb(restored_prediction)

    if target.shape != pred.shape:
        pred_pil = Image.fromarray(pred).resize(
            (target.shape[1], target.shape[0]),
            Image.Resampling.LANCZOS,
        )
        pred = np.array(pred_pil, dtype=np.uint8)

    mse = np.mean((target.astype(np.float64) - pred.astype(np.float64)) ** 2)
    if mse == 0:
        psnr = 100.0
    else:
        psnr = float(psnr_func(target, pred, data_range=255))
    ssim = float(ssim_func(target, pred, channel_axis=2, data_range=255))
    mae = float(np.mean(np.abs(target.astype(np.float32) - pred.astype(np.float32))))

    return {
        "psnr_db": psnr,
        "ssim": ssim,
        "mae_pixel": mae,
    }
