#!/usr/bin/env python3
"""
Multi-Scale Patch Extraction
==============================

Extract H&E patches at multiple resolutions (2μm, 8μm, 32μm, 128μm)
for multi-scale training.

CRITICAL: Output to fast ext4 filesystem, NOT /mnt/ NTFS!

Usage:
    python extract_multiscale_patches.py --patient P1 --scales 2um 8um 32um
    python extract_multiscale_patches.py --all_patients

Author: Max Van Belkum + Claude Opus 4.5
Date: December 2024
"""

import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional
import warnings

import numpy as np
from PIL import Image
from tqdm import tqdm


# CRITICAL: Use fast ext4 paths!
OUTPUT_ROOT = Path("/home/user/work/encoder-loss-ablation-2um/data/multiscale")
LABEL_ROOT = Path("/home/user/work/encoder-loss-ablation-2um/data")

# WSI paths (may be on slower NTFS but read-only is acceptable)
WSI_PATHS = {
    'P1': Path("/home/user/work/enact_data/GSM8594567_P1CRC_tissue_image.btf"),
    'P2': Path("/home/user/work/enact_data/GSM8594568_P2CRC_tissue_image.btf"),
    'P5': Path("/home/user/work/enact_data/GSM8594569_P5CRC_tissue_image.btf"),
}

# Alternative paths
WSI_PATHS_ALT = {
    'P1': Path("/mnt/x/img2st_rotation_demo/downloads/binned_outputs/square_002um/spatial/tissue_fullres_image.btf"),
    'P2': Path("/mnt/x/img2st_rotation_demo/downloads/crc_hd/P2/square_002um/spatial/tissue_fullres_image.btf"),
    'P5': Path("/mnt/x/img2st_rotation_demo/downloads/crc_hd/P5/square_002um/spatial/tissue_fullres_image.btf"),
}

# Scale factors (physical size relative to 2μm)
SCALE_FACTORS = {
    '2um': 1,
    '8um': 4,
    '32um': 16,
    '128um': 64
}


def get_patch_coords(patient: str) -> List[Tuple[int, int]]:
    """Get patch coordinates from existing label files."""
    label_dir = LABEL_ROOT / f"{patient}_precomputed_labels_v2"
    if not label_dir.exists():
        label_dir = LABEL_ROOT / f"{patient}_precomputed_labels"

    if not label_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")

    coords = []
    for f in label_dir.glob("patch_*.npy"):
        parts = f.stem.split('_')
        if len(parts) >= 3:
            try:
                x = int(parts[1])
                y = int(parts[2])
                coords.append((x, y))
            except ValueError:
                continue

    return sorted(coords)


def load_wsi(patient: str):
    """Load whole slide image."""
    import tifffile

    # Try primary path first
    wsi_path = WSI_PATHS.get(patient)
    if wsi_path and wsi_path.exists():
        print(f"Loading WSI from: {wsi_path}")
        return tifffile.imread(wsi_path)

    # Try alternative path
    wsi_path = WSI_PATHS_ALT.get(patient)
    if wsi_path and wsi_path.exists():
        print(f"Loading WSI from alternative: {wsi_path}")
        return tifffile.imread(wsi_path)

    raise FileNotFoundError(f"WSI not found for patient {patient}")


def extract_patch(
    wsi: np.ndarray,
    x: int,
    y: int,
    scale: str,
    output_dir: Path,
    patch_size: int = 224,
    base_scale_px: int = 224  # Size of 2μm patch in WSI pixels
) -> bool:
    """
    Extract a single patch at specified scale.

    At 2μm, a 224x224 patch covers 224*2 = 448μm physical size.
    At 8μm, we need 4x more physical area but same 224x224 output.
    """
    try:
        factor = SCALE_FACTORS[scale]

        # Calculate center of patch in WSI coordinates
        center_x = x * base_scale_px + base_scale_px // 2
        center_y = y * base_scale_px + base_scale_px // 2

        # Calculate region size (larger for coarser scales)
        region_size = base_scale_px * factor

        # Calculate bounds
        start_x = max(0, center_x - region_size // 2)
        start_y = max(0, center_y - region_size // 2)
        end_x = min(wsi.shape[1], start_x + region_size)
        end_y = min(wsi.shape[0], start_y + region_size)

        # Handle edge cases
        if end_x - start_x < region_size * 0.5 or end_y - start_y < region_size * 0.5:
            return False

        # Extract region
        region = wsi[start_y:end_y, start_x:end_x]
        if region.size == 0:
            return False

        # Resize to standard patch size
        img = Image.fromarray(region)
        img = img.resize((patch_size, patch_size), Image.LANCZOS)

        # Save as PNG (lossless, fast)
        output_path = output_dir / f"patch_{x:03d}_{y:03d}.png"
        img.save(output_path, optimize=False)

        return True

    except Exception as e:
        warnings.warn(f"Failed to extract patch ({x}, {y}) at {scale}: {e}")
        return False


def extract_patient_patches(
    patient: str,
    scales: List[str],
    output_root: Path = OUTPUT_ROOT,
    num_workers: int = 8,
    max_patches: Optional[int] = None
):
    """Extract all patches for a patient at multiple scales."""
    print(f"\n{'='*60}")
    print(f"Extracting patches for patient {patient}")
    print(f"Scales: {scales}")
    print(f"{'='*60}\n")

    # Get patch coordinates
    coords = get_patch_coords(patient)
    if max_patches:
        coords = coords[:max_patches]
    print(f"Found {len(coords)} patches")

    # Load WSI
    wsi = load_wsi(patient)
    print(f"WSI shape: {wsi.shape}")

    # Create output directories
    for scale in scales:
        scale_dir = output_root / patient / scale
        scale_dir.mkdir(parents=True, exist_ok=True)

    # Check what's already extracted
    for scale in scales:
        scale_dir = output_root / patient / scale
        existing = len(list(scale_dir.glob("patch_*.png")))
        print(f"  {scale}: {existing} existing patches")

    # Extract patches (parallel for each scale)
    for scale in scales:
        scale_dir = output_root / patient / scale

        # Skip if already mostly done
        existing = set(f.stem for f in scale_dir.glob("patch_*.png"))
        coords_to_extract = [
            (x, y) for x, y in coords
            if f"patch_{x:03d}_{y:03d}" not in existing
        ]

        if not coords_to_extract:
            print(f"  {scale}: Already complete, skipping")
            continue

        print(f"\n  Extracting {len(coords_to_extract)} patches at {scale}...")

        success_count = 0
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(extract_patch, wsi, x, y, scale, scale_dir): (x, y)
                for x, y in coords_to_extract
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc=scale):
                if future.result():
                    success_count += 1

        print(f"  {scale}: Extracted {success_count}/{len(coords_to_extract)} patches")

    print(f"\nDone with patient {patient}")


def main():
    parser = argparse.ArgumentParser(description="Multi-Scale Patch Extraction")

    parser.add_argument("--patient", type=str, choices=["P1", "P2", "P5"],
                        help="Patient to extract")
    parser.add_argument("--all_patients", action="store_true",
                        help="Extract all patients")
    parser.add_argument("--scales", nargs="+", default=["2um", "8um", "32um"],
                        help="Scales to extract")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_ROOT),
                        help="Output directory (should be ext4!)")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of parallel workers")
    parser.add_argument("--max_patches", type=int, default=None,
                        help="Max patches per patient (for debugging)")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Warn if output is on NTFS
    if str(output_dir).startswith("/mnt/"):
        warnings.warn("Output directory is on NTFS (/mnt/). This will be SLOW!")
        print("Recommend using /home/user/work/ instead")

    patients = ["P1", "P2", "P5"] if args.all_patients else [args.patient]

    if not patients[0]:
        print("Specify --patient or --all_patients")
        return

    for patient in patients:
        try:
            extract_patient_patches(
                patient=patient,
                scales=args.scales,
                output_root=output_dir,
                num_workers=args.num_workers,
                max_patches=args.max_patches
            )
        except Exception as e:
            print(f"Error processing {patient}: {e}")
            continue

    print("\n" + "="*60)
    print("Multi-scale extraction complete!")
    print(f"Output: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
