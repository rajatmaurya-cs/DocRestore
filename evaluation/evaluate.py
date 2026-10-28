"""
DocRestore AI - Quantitative Model Evaluation Script
DocRestore AI: Document Shadow Removal Using Pix2Pix GAN

Evaluates a trained Generator checkpoint against paired test document images:
- Computes PSNR and SSIM comparing Restored vs Clean Target.
- Compares Degraded Baseline (Shadowed vs Clean) vs Restored Improvement.
- Exports results to CSV and generates side-by-side visual comparisons.
"""

import argparse
from pathlib import Path
from typing import List, Tuple
import pandas as pd
from PIL import Image

from config import GENERATOR_CHECKPOINT_PATH, OUTPUTS_DIR, get_device
from evaluation.metrics import calculate_document_metrics
from inference import restore_document_image
from model import is_checkpoint_available, load_generator_checkpoint


def find_paired_test_images(
    shadow_dir: Path,
    clean_dir: Path,
) -> List[Tuple[Path, Path]]:
    """
    Finds paired shadowed and clean document images:
    Matches by exact filename or common base stem.
    """
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    shadow_files = sorted(
        [p for p in shadow_dir.iterdir() if p.suffix.lower() in valid_extensions]
    )

    pairs = []
    for s_path in shadow_files:
        stem = s_path.stem
        # Try direct match
        direct_clean = clean_dir / s_path.name
        if direct_clean.is_file():
            pairs.append((s_path, direct_clean))
            continue

        # Try common suffix replacement (e.g. sample1_shadow.png -> sample1_clean.png)
        alt_name = (
            stem.replace("_shadow", "_clean")
            .replace("-shadow", "-clean")
            .replace("_degraded", "_gt")
        )
        matched = False
        for ext in valid_extensions:
            cand = clean_dir / f"{alt_name}{ext}"
            if cand.is_file():
                pairs.append((s_path, cand))
                matched = True
                break

        if not matched:
            print(f"[Warning] No matching clean target found for '{s_path.name}'. Skipping.")

    return pairs


def create_comparison_strip(
    shadow_img: Image.Image,
    restored_img: Image.Image,
    clean_img: Image.Image,
) -> Image.Image:
    """Concatenates Shadowed, Restored, and Clean Ground Truth side-by-side."""
    h = min(shadow_img.height, restored_img.height, clean_img.height, 512)
    s = shadow_img.resize((int(shadow_img.width * (h / shadow_img.height)), h))
    r = restored_img.resize((int(restored_img.width * (h / restored_img.height)), h))
    c = clean_img.resize((int(clean_img.width * (h / clean_img.height)), h))

    total_w = s.width + r.width + c.width
    strip = Image.new("RGB", (total_w, h))
    strip.paste(s, (0, 0))
    strip.paste(r, (s.width, 0))
    strip.paste(c, (s.width + r.width, 0))
    return strip


def run_evaluation(
    shadow_dir: Path,
    clean_dir: Path,
    checkpoint_path: Path = GENERATOR_CHECKPOINT_PATH,
    output_dir: Path = OUTPUTS_DIR / "evaluation",
) -> None:
    """Runs test evaluation and saves CSV metrics and visual panels."""
    available, resolved_path = is_checkpoint_available(checkpoint_path)
    if not available:
        print(
            f"\n[Error] Checkpoint not found at '{resolved_path}'.\n"
            "Evaluation requires a genuine trained generator checkpoint.\n"
            "Please train the model on Google Colab and place 'generator.pth' into 'models/'."
        )
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir = output_dir / "comparisons"
    visuals_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"[Info] Loading trained Generator from: {resolved_path} onto {device}...")
    generator = load_generator_checkpoint(checkpoint_path=resolved_path, device=device)

    pairs = find_paired_test_images(shadow_dir, clean_dir)
    if not pairs:
        print(f"[Error] No valid paired images found between '{shadow_dir}' and '{clean_dir}'.")
        return

    print(f"[Info] Evaluating {len(pairs)} test document image pairs...")

    records = []
    for idx, (s_path, c_path) in enumerate(pairs, start=1):
        clean_img = Image.open(c_path).convert("RGB")

        # Run restoration
        res = restore_document_image(
            image_input=s_path,
            generator=generator,
            device=device,
            mode="letterbox",
        )
        restored_img = res["restored_image"]
        shadow_img = res["original_image"]

        # Calculate metrics for degraded input vs clean ground truth (baseline)
        baseline_metrics = calculate_document_metrics(clean_img, shadow_img)
        # Calculate metrics for restored output vs clean ground truth
        restored_metrics = calculate_document_metrics(clean_img, restored_img)

        delta_psnr = restored_metrics["psnr_db"] - baseline_metrics["psnr_db"]
        delta_ssim = restored_metrics["ssim"] - baseline_metrics["ssim"]

        records.append(
            {
                "image_name": s_path.name,
                "baseline_psnr_db": round(baseline_metrics["psnr_db"], 2),
                "restored_psnr_db": round(restored_metrics["psnr_db"], 2),
                "psnr_improvement_db": round(delta_psnr, 2),
                "baseline_ssim": round(baseline_metrics["ssim"], 4),
                "restored_ssim": round(restored_metrics["ssim"], 4),
                "ssim_improvement": round(delta_ssim, 4),
                "mae_pixel": round(restored_metrics["mae_pixel"], 2),
                "latency_sec": round(res["inference_time_seconds"], 4),
            }
        )

        # Save visual comparison strip
        strip = create_comparison_strip(shadow_img, restored_img, clean_img)
        strip.save(visuals_dir / f"compare_{s_path.stem}.png")

        print(
            f"[{idx}/{len(pairs)}] {s_path.name} | "
            f"PSNR: {restored_metrics['psnr_db']:.2f} dB (+{delta_psnr:.2f}) | "
            f"SSIM: {restored_metrics['ssim']:.4f} (+{delta_ssim:.4f})"
        )

    # Compile DataFrame and summary stats
    df = pd.DataFrame(records)
    csv_path = output_dir / "test_evaluation_metrics.csv"
    df.to_csv(csv_path, index=False)

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY RESULTS")
    print("=" * 60)
    print(f"Total Test Pairs Evaluated: {len(df)}")
    print(f"Average Restored PSNR:     {df['restored_psnr_db'].mean():.2f} dB")
    print(f"Average PSNR Improvement:  +{df['psnr_improvement_db'].mean():.2f} dB")
    print(f"Average Restored SSIM:     {df['restored_ssim'].mean():.4f}")
    print(f"Average SSIM Improvement:  +{df['ssim_improvement'].mean():.4f}")
    print(f"Results exported to:       {csv_path}")
    print(f"Visual comparisons in:     {visuals_dir}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DocRestore AI Generator on Paired Test Set")
    parser.add_argument("--shadow_dir", type=Path, required=True, help="Directory of shadow document images")
    parser.add_argument("--clean_dir", type=Path, required=True, help="Directory of clean ground truth images")
    parser.add_argument("--checkpoint", type=Path, default=GENERATOR_CHECKPOINT_PATH, help="Path to generator.pth")
    parser.add_argument("--output_dir", type=Path, default=OUTPUTS_DIR / "evaluation", help="Output directory")

    args = parser.parse_args()
    run_evaluation(
        shadow_dir=args.shadow_dir,
        clean_dir=args.clean_dir,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
    )
