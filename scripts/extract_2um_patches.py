#!/usr/bin/env python3
"""
Extract 2μm Patches from WSI
=============================

Direct extraction using tifffile for reliable coordinate mapping.

Usage:
    python extract_2um_patches.py --patient P1
    python extract_2um_patches.py --all

Author: Max Van Belkum + Claude Opus 4.5
Date: December 2024
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm
import tifffile


# Fast ext4 paths
DATA_ROOT = Path("/home/user/work/encoder-loss-ablation-2um/data")
OUTPUT_ROOT = DATA_ROOT / "multiscale"

# WSI paths
WSI_PATHS = {
    'P1': Path("/home/user/work/enact_data/GSM8594567_P1CRC_tissue_image.btf"),
    'P2': Path("/home/user/work/enact_data/GSM8594568_P2CRC_tissue_image.btf"),
    'P5': Path("/home/user/work/enact_data/GSM8594569_P5CRC_tissue_image.btf"),
}


def get_patch_coords(patient: str) -> List[Tuple[int, int, str]]:
    """Get patch coordinates from label files."""
    label_dir = DATA_ROOT / f"{patient}_precomputed_labels_v2"
    if not label_dir.exists():
        label_dir = DATA_ROOT / f"{patient}_precomputed_labels"

    patches = []
    for f in label_dir.glob("patch_*.npy"):
        parts = f.stem.split('_')
        if len(parts) >= 3:
            try:
                x, y = int(parts[1]), int(parts[2])
                patches.append((x, y, f.stem))
            except ValueError:
                continue

    return sorted(patches)


def extract_2um_patches(
    patient: str,
    output_dir: Path = OUTPUT_ROOT,
    patch_size: int = 256,
    max_patches: int = None
) -> dict:
    """
    Extract 2μm patches for a patient.

    Args:
        patient: Patient ID (P1, P2, P5)
        output_dir: Output directory
        patch_size: Output patch size (256)
        max_patches: Limit for testing

    Returns:
        Extraction statistics
    """
    print(f"\n{'='*60}")
    print(f"Extracting 2μm patches for {patient}")
    print(f"{'='*60}")

    # Get patch coordinates
    patches = get_patch_coords(patient)
    if max_patches:
        patches = patches[:max_patches]
    print(f"Found {len(patches)} patches")

    # Load WSI
    wsi_path = WSI_PATHS.get(patient)
    if not wsi_path or not wsi_path.exists():
        print(f"ERROR: WSI not found: {wsi_path}")
        return {'error': 'WSI not found'}

    print(f"Loading WSI: {wsi_path}")
    wsi = tifffile.imread(wsi_path)
    print(f"WSI shape: {wsi.shape}")

    # Calculate grid mapping
    all_x = [p[0] for p in patches]
    all_y = [p[1] for p in patches]

    max_x = max(all_x)
    max_y = max(all_y)

    # Pixels per patch step (map grid indices to WSI pixels)
    px_per_step_x = wsi.shape[1] / (max_x + 1)
    px_per_step_y = wsi.shape[0] / (max_y + 1)
    print(f"Pixels per grid step: {px_per_step_x:.1f} x {px_per_step_y:.1f}")

    # Create output directory
    scale_dir = output_dir / patient / "scale_2um"
    scale_dir.mkdir(parents=True, exist_ok=True)

    # Check existing patches
    existing = set(f.stem for f in scale_dir.glob("patch_*.png"))
    patches_to_extract = [(x, y, name) for x, y, name in patches
                          if name not in existing]

    print(f"Already extracted: {len(existing)}")
    print(f"To extract: {len(patches_to_extract)}")

    if not patches_to_extract:
        print("All patches already extracted")
        return {'patient': patient, 'extracted': 0, 'total': len(patches)}

    # Extract patches
    extracted = 0
    failed = 0

    for x, y, name in tqdm(patches_to_extract, desc=f"Extracting {patient}"):
        try:
            # Calculate region in WSI coordinates
            region_size_x = int(px_per_step_x)
            region_size_y = int(px_per_step_y)

            start_x = int(x * px_per_step_x)
            start_y = int(y * px_per_step_y)
            end_x = min(start_x + region_size_x, wsi.shape[1])
            end_y = min(start_y + region_size_y, wsi.shape[0])

            # Extract region
            region = wsi[start_y:end_y, start_x:end_x]

            if region.size == 0 or region.shape[0] < 10 or region.shape[1] < 10:
                failed += 1
                continue

            # Resize to output size
            img = Image.fromarray(region)
            img = img.resize((patch_size, patch_size), Image.LANCZOS)

            # Save
            output_path = scale_dir / f"{name}.png"
            img.save(output_path, optimize=False)
            extracted += 1

        except Exception as e:
            print(f"Error on {name}: {e}")
            failed += 1

    # Save metadata
    meta = {
        'patient': patient,
        'total_patches': len(patches),
        'extracted': extracted,
        'failed': failed,
        'existing': len(existing),
        'wsi_shape': list(wsi.shape),
        'px_per_step': [px_per_step_x, px_per_step_y]
    }

    with open(scale_dir / "extraction_meta.json", 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\nExtraction complete:")
    print(f"  Extracted: {extracted}")
    print(f"  Failed: {failed}")
    print(f"  Total: {len(existing) + extracted}")

    return meta


def main():
    parser = argparse.ArgumentParser(description="Extract 2μm patches")
    parser.add_argument("--patient", type=str, choices=["P1", "P2", "P5"],
                        help="Patient to extract")
    parser.add_argument("--all", action="store_true", help="Extract all patients")
    parser.add_argument("--max_patches", type=int, default=None,
                        help="Limit patches for testing")

    args = parser.parse_args()

    patients = ["P1", "P2", "P5"] if args.all else [args.patient] if args.patient else []

    if not patients:
        print("Specify --patient or --all")
        return

    for patient in patients:
        extract_2um_patches(
            patient=patient,
            max_patches=args.max_patches
        )

    print("\n" + "="*60)
    print("2μm extraction complete!")
    print("="*60)


if __name__ == "__main__":
    main()
