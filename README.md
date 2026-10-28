# DocRestore AI: Document Shadow Removal Using Pix2Pix GAN

**DocRestore AI** is a deep learning computer vision application designed to restore document photographs and scans degraded by harsh cast shadows, uneven illumination, paper folds, and poor ambient lighting. 

Built using a genuine **Pix2Pix Conditional Generative Adversarial Network (cGAN)** in PyTorch, the system normalizes page illumination and background contrast while faithfully preserving printed typography, handwritten notes, ink strokes, and layout structure.

---

## 1. Deep Learning Architecture: Pix2Pix cGAN

Standard image restoration filters (such as global thresholding or morphological contrast stretching) either wash out faint handwriting or fail to distinguish illumination shadows from dark document content. DocRestore AI solves this problem by framing document restoration as a paired **image-to-image translation** task.

```
       [Shadowed Document X]
                 │
                 ▼
        ┌─────────────────┐
        │  Generator (G)  │ ──► [Restored Document G(X)]
        │    (U-Net 256)  │
        └─────────────────┘
                 │
                 ▼
        ┌─────────────────┐
        │Discriminator (D)│ ◄── [Real Clean Target Y]
        │   (PatchGAN)    │
        └─────────────────┘
                 │
                 ▼
     [Real/Fake Patch Matrix]
```

### A. Generator — Symmetrical U-Net 256
- **Encoder:** 8 progressive downsampling convolutional blocks (`Conv2d` $\to$ `BatchNorm` $\to$ `LeakyReLU(0.2)`), downsampling the $256 \times 256$ image into an informative $1 \times 1$ latent bottleneck ($512$ channels).
- **Decoder:** 8 progressive upsampling transposed convolutional blocks (`ConvTranspose2d` $\to$ `BatchNorm` $\to$ `Dropout` $\to$ `ReLU`).
- **Skip Connections:** Direct symmetrical channel concatenations connect each encoder activation map to its corresponding decoder layer. This is vital for document restoration: skip connections bypass the bottleneck, allowing fine-grained text strokes, glyph boundaries, and subtle handwriting lines to pass through without becoming blurry.
- **Output Layer:** $1 \times 1$ transposed convolution with $\tanh$ activation producing 3-channel RGB normalized in $[-1, 1]$.

### B. Discriminator — 70×70 PatchGAN
- Rather than classifying the entire document with a single scalar real/fake score, the PatchGAN discriminator penalizes structural errors at the level of local **$70 \times 70$ image patches**.
- **Conditional Input:** The discriminator receives $6$ channels: the degraded document $X$ concatenated with either the generated clean candidate $G(X)$ or the ground-truth clean document $Y$.
- This focuses learning on high-frequency realism (crisp text contrast, uniform page texture) while leaving low-frequency color uniformity to the $L_1$ loss.

### C. Combined Loss Objective
The Generator is trained against a dual objective:
$$\mathcal{L}_{G} = \mathcal{L}_{GAN}(G, D) + \lambda \mathcal{L}_{L1}(G)$$
1. **Adversarial Loss ($\mathcal{L}_{GAN}$):** Forces the Generator to create natural document illumination patterns that fool the discriminator into believing the page was captured under ideal lighting.
2. **Reconstruction Loss ($\mathcal{L}_{L1}$):** Measures pixel-wise absolute error between the generated document and the clean target:
   $$\mathcal{L}_{L1} = \mathbb{E}_{X, Y}\left[ \| Y - G(X) \|_1 \right]$$
   We set $\lambda = 100$. $L_1$ loss is strictly preferred over $L_2$ (mean squared error) because $L_2$ penalizes large outliers heavily and produces blurry images, whereas $L_1$ encourages crisp, sharp edges.

---

## 2. Training vs. Inference: Workflow Separation

| Dimension | Training Pipeline (Google Colab GPU) | Local Inference (DocRestore-AI App) |
| :--- | :--- | :--- |
| **Compute Engine** | NVIDIA T4 / A100 GPU | Apple Silicon MPS (Metal) / CPU Fallback |
| **Active Networks** | Both Generator ($G$) & Discriminator ($D$) | **Generator ($G$) only** |
| **Gradient Updates** | Backpropagation via Adam ($\beta_1=0.5, \beta_2=0.999$) | Strictly disabled (`torch.no_grad()`, `eval()`) |
| **Input Constraints** | Fixed $256 \times 256$ paired image crops | Arbitrary image dimensions (with letterbox unpadding) |
| **Data Requirements** | Paired dataset with clean ground truth | Single user-uploaded shadowed image |

---

## 3. Google Colab Training Guide

Because training a GAN on thousands of high-resolution document pairs is compute-intensive, training is executed in Google Colab using a free T4 GPU.

### Step 1: Open Google Colab
1. Upload the provided notebook: `colab/DocRestore_Colab_Training.ipynb` to [Google Colab](https://colab.research.google.com).
2. Set runtime type: **Runtime $\to$ Change runtime type $\to$ Hardware accelerator: T4 GPU**.

### Step 2: Datasets for Document Shadow Removal
Recommended benchmark datasets:
- **Jung et al. Document Shadow Dataset:** Paired real-world document photographs under natural cast shadows with scanned ground-truth counterparts.
- **DocShadow Dataset / SDSR:** Comprehensive paired dataset featuring document folds, desk shadows, and hand occlusion.

Download or mount your paired dataset into Colab:
```
data/
├── train/
│   ├── shadow/    # Degraded document images
│   └── clean/     # Ground-truth clean scans
```

### Step 3: Run Training
Execute the training cells or run:
```bash
python colab/train_pix2pix.py --shadow_dir data/train/shadow --clean_dir data/train/clean --epochs 80 --batch_size 16
```
Training takes approximately **20 to 30 minutes** on a T4 GPU for 80 epochs.

### Step 4: Export Checkpoint to Local Application
Once training completes, download `best_generator.pth` and copy it to your local workspace:
```bash
cp ~/Downloads/best_generator.pth models/generator.pth
```
*(Status: **Integrated**. Your Google Colab trained checkpoint `best_generator.pth` [217MB] is now loaded into `models/generator.pth` and `checkpoints/best_generator.pth` along with `checkpoints/history.csv` and `checkpoints/metadata.json`.)*

---

## 4. Local Installation & Application Setup

### Prerequisites
- macOS (Apple Silicon M1/M2/M3 or Intel), Linux, or Windows
- Python 3.10+ (Python 3.12 verified)

### Step 1: Virtual Environment Setup
From the project root directory:
```bash
# Create isolated virtual environment (if not already created)
python3 -m venv .venv

# Activate virtual environment
# On macOS / Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Launch Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser to `http://localhost:8501`.

---

## 5. Model Evaluation & Quantitative Metrics

To benchmark a trained Generator on an independent test set:
```bash
python evaluation/evaluate.py \
  --shadow_dir path/to/test/shadow \
  --clean_dir path/to/test/clean \
  --checkpoint models/generator.pth \
  --output_dir outputs/evaluation
```

### Evaluation Metrics
- **PSNR (Peak Signal-to-Noise Ratio):** Measures logarithmic ratio between maximum signal power ($255$) and mean squared reconstruction noise. Higher is better (values $> 28$ dB indicate clean reconstruction).
- **SSIM (Structural Similarity Index Measure):** Measures human perceptual similarity in luminance, contrast, and structure in range $[0, 1]$. Values $> 0.90$ indicate strong text shape and stroke fidelity.

The script exports individual sample metrics to `test_evaluation_metrics.csv` and visual 3-panel comparisons `[Shadow Input | Restored Output | Ground Truth Clean]` to `outputs/evaluation/comparisons/`.

---

## 6. Running the Local Test Suite

Run unit and integration tests using `pytest`:
```bash
pytest -v
```

Tests verify:
1. U-Net Generator input/output shapes ($[B, 3, 256, 256]$) and $\tanh$ bounds.
2. PatchGAN Discriminator receptive field and patch classification shapes ($[B, 1, 30, 30]$).
3. Preprocessing letterbox padding and dimension-preserving unpadding.
4. Metric calculations (PSNR, SSIM, MAE).
5. Missing-checkpoint detection and error-handling behavior.

---

## 7. College Viva & Technical Interview Q&A

### Q1: Why Pix2Pix GAN instead of standard autoencoders or thresholding?
> **Answer:** Standard autoencoders trained with Mean Squared Error ($L_2$) average out all plausible pixel values, producing blurry, hazy text. Global thresholding (like Otsu) destroys color information, gradients, and faint handwriting. Pix2Pix pairs a U-Net Generator with a PatchGAN Discriminator: the $L_1$ loss enforces pixel-accurate color recovery while the adversarial loss enforces crisp, high-frequency text boundaries and realistic paper texture.

### Q2: Why are skip connections indispensable for document images?
> **Answer:** In a standard bottleneck autoencoder, information is compressed down to a tiny spatial dimension ($1 \times 1$), which discards high-frequency spatial coordinates. Symmetrical skip connections pass fine-grained spatial information directly from encoder feature maps to decoder feature maps, ensuring that individual character strokes, accents, and punctuation marks are preserved intact.

### Q3: Why is PatchGAN used rather than a standard DCGAN Discriminator?
> **Answer:** A standard GAN discriminator outputs a single scalar indicating whether the entire image is real or fake. This can lead to mode collapse and ignores local lighting artifacts. PatchGAN classifies each $70 \times 70$ patch independently. Because documents have spatially distributed text and localized shadows, PatchGAN forces the Generator to remove shadows across every local region of the page.

### Q4: What happens if a model is trained for only 5 epochs on a very small dataset?
> **Answer:** 
> 1. **Under-convergence:** 5 epochs are insufficient for the U-Net weights to separate complex illumination gradients from text ink, resulting in residual shadow halos.
> 2. **Color Shift:** Early GAN epochs often produce yellowish or grayish paper casts before the $L_1$ loss stabilizes page background reflectance.
> 3. **Text Distortion:** The adversarial discriminator may introduce high-frequency checkerboard artifacts if upsampling transposed convolutions have not had enough iterations to harmonize. At least 50–80 epochs on paired images are required for clean convergence.
