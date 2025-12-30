#!/usr/bin/env python3
"""
Extract 2μm Patches from WSI - Version 2
==========================================

Uses tissue_positions.parquet for correct coordinate mapping.

The coordinate system:
- WSI is at fullres (native resolution)
- pxl_scaled = pxl_row_in_fullres * scale_factor
- patch_row = pxl_scaled // 256

Scale factors (tissue_hires_scalef):
- P1: 0.084381066
- P2: 0.2750
- P5: 0.2735

Usage:
    python extract_2um_patches_v2.py --patient P1
    python extract_2um_patches_v2.py --all

Author: Max Van Belkum + Claude Opus 4.5
Date: December 2024
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
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

# Tissue positions paths
POSITIONS_PATHS = {
    'P1': Path("/mnt/x/img2st_rotation_demo/downloads/binned_outputs/square_002um/spatial/tissue_positions.parquet"),
    'P2': Path("/mnt/x/img2st_rotation_demo/downloads/crc_hd/P2/square_002um/spatial/tissue_positions.parquet"),
    'P5': Path("/mnt/x/img2st_rotation_demo/downloads/crc_hd/P5/square_002um/spatial/tissue_positions.parquet"),
}

# Scale factors - use coordinate_scale from metadata (NOT tissue_hires_scalef!)
# These are loaded from metadata.json; defaults here as fallback
SCALE_FACTORS = {
    'P1': 0.275,   # coordinate_scale from precomputed_labels/metadata.json
    'P2': 0.2750,
    'P5': 0.2735,
}

PATCH_SIZE_PIXELS = 256  # Patch size in scaled coordinates


def get_patch_info(patient: str) -> Tuple[List[Dict], pd.DataFrame]:
    """Get patch coordinates and tissue positions."""
    # Load patches
    patches_file = DATA_ROOT / f"{patient}_precomputed_labels_v2" / "patches.json"
    if not patches_file.exists():
        patches_file = DATA_ROOT / f"{patient}_precomputed_labels" / "patches.json"

    with open(patches_file) as f:
        patches = json.load(f)

    # Load tissue positions
    positions_path = POSITIONS_PATHS.get(patient)
    if positions_path and positions_path.exists():
        positions = pd.read_parquet(positions_path)
    else:
        # Try alternative paths
        alt_paths = [
            DATA_ROOT / patient / "tissue_positions.parquet",
            Path(f"/home/user/work/enact_data/{patient}/tissue_positions.parquet"),
        ]
        positions = None
        for alt in alt_paths:
            if alt.exists():
                positions = pd.read_parquet(alt)
                break

    return patches, positions


def extract_2um_patches(
    patient: str,
    output_dir: Path = OUTPUT_ROOT,
    patch_size: int = 256,
    max_patches: int = None
) -> dict:
    """
    Extract 2μm patches for a patient using correct coordinate mapping.
    """
    print(f"\n{'='*60}")
    print(f"Extracting 2μm patches for {patient}")
    print(f"{'='*60}")

    # Get patches and positions
    patches, positions = get_patch_info(patient)
    if max_patches:
        patches = patches[:max_patches]
    print(f"Found {len(patches)} patches")

    scale = SCALE_FACTORS.get(patient, 0.25)
    print(f"Scale factor: {scale}")

    # Load WSI
    wsi_path = WSI_PATHS.get(patient)
    if not wsi_path or not wsi_path.exists():
        print(f"ERROR: WSI not found: {wsi_path}")
        return {'error': 'WSI not found'}

    print(f"Loading WSI: {wsi_path}")
    wsi = tifffile.imread(wsi_path)
    print(f"WSI shape: {wsi.shape} (H x W x C)")
    wsi_h, wsi_w = wsi.shape[:2]

    # Create output directory
    scale_dir = output_dir / patient / "scale_2um"
    scale_dir.mkdir(parents=True, exist_ok=True)

    # Check existing
    existing = set(f.stem for f in scale_dir.glob("patch_*.png"))
    patches_to_extract = [p for p in patches if p['patch_id'] not in existing]

    print(f"Already extracted: {len(existing)}")
    print(f"To extract: {len(patches_to_extract)}")

    if not patches_to_extract:
        print("All patches already extracted")
        return {'patient': patient, 'extracted': 0, 'total': len(patches)}

    # The key coordinate mapping:
    # - Patch is at (patch_row, patch_col) in the scaled coordinate system
    # - To get fullres coords: fullres = scaled / scale_factor
    # - Patch covers scaled pixels [patch_row*256, (patch_row+1)*256)
    # - So fullres coords: [patch_row*256/scale, (patch_row+1)*256/scale)

    extracted = 0
    failed = 0

    for patch in tqdm(patches_to_extract, desc=f"Extracting {patient}"):
        try:
            patch_id = patch['patch_id']
            patch_row = patch['patch_row']
            patch_col = patch['patch_col']

            # Convert scaled coords to fullres WSI coords
            # Each patch covers 256 scaled pixels = 256/scale fullres pixels
            fullres_patch_size = int(PATCH_SIZE_PIXELS / scale)

            start_y = int(patch_row * PATCH_SIZE_PIXELS / scale)
            start_x = int(patch_col * PATCH_SIZE_PIXELS / scale)
            end_y = min(start_y + fullres_patch_size, wsi_h)
            end_x = min(start_x + fullres_patch_size, wsi_w)

            # Bounds check
            if start_x < 0 or start_y < 0:
                failed += 1
                continue
            if end_x - start_x < fullres_patch_size * 0.5:
                failed += 1
                continue
            if end_y - start_y < fullres_patch_size * 0.5:
                failed += 1
                continue

            # Extract region
            region = wsi[start_y:end_y, start_x:end_x]

            if region.size == 0:
                failed += 1
                continue

            # Resize to output size
            img = Image.fromarray(region)
            img = img.resize((patch_size, patch_size), Image.LANCZOS)

            # Save
            output_path = scale_dir / f"{patch_id}.png"
            img.save(output_path, optimize=False)
            extracted += 1

        except Exception as e:
            print(f"Error on {patch.get('patch_id', 'unknown')}: {e}")
            failed += 1

    # Save metadata
    meta = {
        'patient': patient,
        'total_patches': len(patches),
        'extracted': extracted,
        'failed': failed,
        'existing': len(existing),
        'scale_factor': scale,
        'wsi_shape': list(wsi.shape),
    }

    with open(scale_dir / "extraction_meta.json", 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\nExtraction complete:")
    print(f"  Extracted: {extracted}")
    print(f"  Failed: {failed}")
    print(f"  Total: {len(existing) + extracted}")

    return meta


def main():
    parser = argparse.ArgumentParser(description="Extract 2μm patches (v2)")
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
