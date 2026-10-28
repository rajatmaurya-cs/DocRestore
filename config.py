"""
DocRestore AI - Configuration & Hyperparameters
DocRestore AI: Document Shadow Removal Using Pix2Pix GAN
"""

from pathlib import Path
import torch

# Base project directory
BASE_DIR = Path(__file__).resolve().parent

# Core subdirectories
MODELS_DIR = BASE_DIR / "models"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
OUTPUTS_DIR = BASE_DIR / "outputs"
EVALUATION_DIR = BASE_DIR / "evaluation"

# Primary Generator checkpoint path
GENERATOR_CHECKPOINT_PATH = MODELS_DIR / "generator.pth"
# Secondary fallback search path
FALLBACK_CHECKPOINT_PATH = CHECKPOINTS_DIR / "best_generator.pth"

# Model Contract Hyperparameters
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)
IN_CHANNELS = 3
OUT_CHANNELS = 3
NUM_GENERATOR_FILTERS = 64
NUM_DISCRIMINATOR_FILTERS = 64

# Pix2Pix Training Hyperparameters (for Colab / training pipeline)
LAMBDA_L1 = 100.0
LEARNING_RATE = 2e-4
BETA1 = 0.5
BETA2 = 0.999
DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 80

# Normalization constants (Pix2Pix standard: [-1, 1])
NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD = (0.5, 0.5, 0.5)


def get_device() -> torch.device:
    """
    Selects the best available computational hardware:
    1. Apple Silicon MPS (Metal Performance Shaders) for Mac M-series
    2. NVIDIA CUDA for Colab or Linux/Windows GPU setups
    3. CPU fallback
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
