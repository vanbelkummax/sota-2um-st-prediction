"""
Multi-Scale Spatial Transcriptomics Data Loading
=================================================

Efficient data loading for multi-resolution training.
Uses Linux ext4 paths for speed (NOT /mnt/ NTFS).

Data locations (fast ext4):
- /home/user/work/encoder-loss-ablation-2um/data/P*_precomputed_labels_v2/
- /home/user/work/encoder-loss-ablation-2um/data/multiscale/

Author: Max Van Belkum + Claude Opus 4.5
Date: December 2024
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from PIL import Image
import json
from typing import Dict, List, Optional, Tuple, Union
import torchvision.transforms as T
from concurrent.futures import ThreadPoolExecutor
import warnings


# CRITICAL: Use Linux ext4 paths for speed
DATA_ROOT = Path("/home/user/work/encoder-loss-ablation-2um/data")
IMAGE_ROOT = Path("/home/user/work/encoder-loss-ablation-2um/data/multiscale")
CONSOLIDATED_ROOT = DATA_ROOT / "consolidated"

# Fallback to original image paths if multiscale not extracted
ORIGINAL_IMAGE_PATHS = {
    'P1': Path("/mnt/x/img2st_rotation_demo/downloads/binned_outputs/square_002um/spatial"),
    'P2': Path("/mnt/x/img2st_rotation_demo/downloads/crc_hd/P2/square_002um/spatial"),
    'P5': Path("/mnt/x/img2st_rotation_demo/downloads/crc_hd/P5/square_002um/spatial"),
}


class MultiScaleSTDataset(Dataset):
    """
    Multi-scale spatial transcriptomics dataset.

    Loads:
    - H&E patches at multiple resolutions (2μm, 8μm, 32μm, 128μm)
    - Gene expression labels at corresponding resolutions
    - Tissue masks

    All data loaded from fast ext4 filesystem.
    """

    def __init__(
        self,
        patients: List[str],
        scales: List[str] = ['2um', '8um'],
        data_root: Path = DATA_ROOT,
        image_root: Path = IMAGE_ROOT,
        n_genes: int = 50,
        patch_size: int = 224,
        label_size: int = 128,
        transform: Optional[T.Compose] = None,
        use_cache: bool = True,
        max_patches_per_patient: Optional[int] = None
    ):
        """
        Initialize multi-scale dataset.

        Args:
            patients: List of patient IDs ['P1', 'P2', 'P5']
            scales: List of scales to load ['2um', '8um', '32um']
            data_root: Root directory for labels (ext4)
            image_root: Root directory for images (ext4)
            n_genes: Number of genes
            patch_size: Image patch size (224)
            label_size: Label array size (128)
            transform: Image transforms
            use_cache: Cache labels in memory
            max_patches_per_patient: Limit patches for debugging
        """
        self.patients = patients
        self.scales = scales
        self.data_root = Path(data_root)
        self.image_root = Path(image_root)
        self.n_genes = n_genes
        self.patch_size = patch_size
        self.label_size = label_size
        self.use_cache = use_cache

        # Default transforms
        if transform is None:
            self.transform = T.Compose([
                T.Resize((patch_size, patch_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

        # Build patch index
        self.patches = []
        self._label_cache = {} if use_cache else None

        for patient in patients:
            patient_patches = self._index_patient(patient, max_patches_per_patient)
            self.patches.extend(patient_patches)

        print(f"MultiScaleSTDataset: {len(self.patches)} patches from {patients}")
        print(f"  Scales: {scales}")

    def _index_patient(
        self,
        patient: str,
        max_patches: Optional[int] = None
    ) -> List[Dict]:
        """Index all patches for a patient."""
        patches = []

        # Find label files
        label_dir = self.data_root / f"{patient}_precomputed_labels_v2"
        if not label_dir.exists():
            label_dir = self.data_root / f"{patient}_precomputed_labels"

        if not label_dir.exists():
            warnings.warn(f"Label directory not found: {label_dir}")
            return patches

        # Get all label files
        label_files = sorted(label_dir.glob("patch_*.npy"))

        for label_file in label_files:
            # Parse patch coordinates from filename
            # Format: patch_XXX_YYY.npy
            name = label_file.stem
            parts = name.split('_')
            if len(parts) >= 3:
                try:
                    x = int(parts[1])
                    y = int(parts[2])
                except ValueError:
                    continue

                patch_info = {
                    'patient': patient,
                    'x': x,
                    'y': y,
                    'label_file': label_file,
                    'images': {}
                }

                # Find corresponding images at each scale
                for scale in self.scales:
                    img_path = self._find_image(patient, x, y, scale)
                    if img_path is not None:
                        patch_info['images'][scale] = img_path

                # Only add if we have at least one scale's image
                # Prefer 2μm if available, otherwise accept any scale
                if patch_info['images']:
                    patches.append(patch_info)

            if max_patches and len(patches) >= max_patches:
                break

        return patches

    def _find_image(
        self,
        patient: str,
        x: int,
        y: int,
        scale: str
    ) -> Optional[Path]:
        """Find image file for given patch and scale."""
        # Try both naming conventions: 'scale_8um' and '8um'
        dir_names = [f"scale_{scale}", scale]

        for dir_name in dir_names:
            multiscale_dir = self.image_root / patient / dir_name
            if multiscale_dir.exists():
                img_path = multiscale_dir / f"patch_{x:03d}_{y:03d}.png"
                if img_path.exists():
                    return img_path
                # Try jpg
                img_path = multiscale_dir / f"patch_{x:03d}_{y:03d}.jpg"
                if img_path.exists():
                    return img_path

        # Fall back to original 2μm patches from precomputed_labels_v2
        if scale == '2um':
            # Check for pre-extracted 2μm patches
            label_dir = self.data_root / f"{patient}_precomputed_labels_v2"
            img_dir = label_dir / "images"
            if img_dir.exists():
                img_path = img_dir / f"patch_{x:03d}_{y:03d}.png"
                if img_path.exists():
                    return img_path

        return None

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a multi-scale sample.

        Returns:
            Dict with:
                - 'images': Dict[scale, tensor (3, 224, 224)]
                - 'labels_2um': tensor (n_genes, 128, 128)
                - 'labels_8um': tensor (n_genes, 32, 32) if available
                - 'mask': tensor (1, 128, 128)
                - 'patient': str
                - 'coords': (x, y)
        """
        patch_info = self.patches[idx]

        result = {
            'images': {},
            'patient': patch_info['patient'],
            'coords': (patch_info['x'], patch_info['y'])
        }

        # Load images at each scale
        for scale, img_path in patch_info['images'].items():
            try:
                img = Image.open(img_path).convert('RGB')
                img = self.transform(img)
                result['images'][scale] = img
            except Exception as e:
                warnings.warn(f"Failed to load image {img_path}: {e}")
                # Return zero tensor as fallback
                result['images'][scale] = torch.zeros(3, self.patch_size, self.patch_size)

        # Load labels
        label_file = patch_info['label_file']
        cache_key = str(label_file)

        if self.use_cache and cache_key in self._label_cache:
            labels_2um = self._label_cache[cache_key]
        else:
            try:
                labels_2um = np.load(label_file)
                if self.use_cache:
                    self._label_cache[cache_key] = labels_2um
            except Exception as e:
                warnings.warn(f"Failed to load labels {label_file}: {e}")
                labels_2um = np.zeros((self.label_size, self.label_size, self.n_genes))

        # Convert to (G, H, W) format
        if labels_2um.shape[-1] == self.n_genes:
            labels_2um = labels_2um.transpose(2, 0, 1)  # (H, W, G) -> (G, H, W)

        result['labels_2um'] = torch.from_numpy(labels_2um.astype(np.float32))

        # Compute aggregated labels for auxiliary resolutions
        labels_tensor = result['labels_2um']  # (G, H, W)

        if '8um' in self.scales:
            # 8μm = 4x4 average of 2μm
            labels_8um = torch.nn.functional.avg_pool2d(
                labels_tensor.unsqueeze(0), kernel_size=4
            ).squeeze(0)
            result['labels_8um'] = labels_8um

        if '32um' in self.scales:
            # 32μm = 16x16 average of 2μm
            labels_32um = torch.nn.functional.avg_pool2d(
                labels_tensor.unsqueeze(0), kernel_size=16
            ).squeeze(0)
            result['labels_32um'] = labels_32um

        # Create mask (non-zero expression regions)
        mask = (labels_tensor.sum(dim=0, keepdim=True) > 0).float()  # (1, H, W)
        result['mask'] = mask

        return result


class SingleScaleSTDataset(Dataset):
    """
    Single-scale dataset for baseline experiments.
    Optimized for speed with memory-mapped labels.
    """

    def __init__(
        self,
        patients: List[str],
        data_root: Path = DATA_ROOT,
        n_genes: int = 50,
        transform: Optional[T.Compose] = None,
        use_consolidated: bool = True
    ):
        self.patients = patients
        self.data_root = Path(data_root)
        self.n_genes = n_genes

        if transform is None:
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

        # Try consolidated format first (faster)
        self.consolidated = None
        if use_consolidated:
            consolidated_file = CONSOLIDATED_ROOT / "all_labels.npy"
            if consolidated_file.exists():
                self.consolidated = np.load(consolidated_file, mmap_mode='r')

        # Index patches
        self.patches = []
        for patient in patients:
            self._index_patient(patient)

    def _index_patient(self, patient: str):
        """Index patches for a patient."""
        label_dir = self.data_root / f"{patient}_precomputed_labels_v2"
        if not label_dir.exists():
            return

        for label_file in sorted(label_dir.glob("patch_*.npy")):
            self.patches.append({
                'patient': patient,
                'label_file': label_file
            })

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        patch = self.patches[idx]

        # Load labels
        labels = np.load(patch['label_file'])
        if labels.shape[-1] == self.n_genes:
            labels = labels.transpose(2, 0, 1)

        labels = torch.from_numpy(labels.astype(np.float32))
        mask = (labels.sum(dim=0, keepdim=True) > 0).float()

        return {
            'labels': labels,
            'mask': mask,
            'patient': patch['patient']
        }


def extract_multiscale_patches(
    patient: str,
    wsi_path: Path,
    output_dir: Path,
    scales: List[str] = ['2um', '8um', '32um'],
    patch_coords: Optional[List[Tuple[int, int]]] = None,
    patch_size: int = 224,
    num_workers: int = 8
):
    """
    Extract multi-scale patches from a whole slide image.

    For each 2μm patch, extracts corresponding patches at coarser resolutions
    centered on the same location.

    Args:
        patient: Patient ID
        wsi_path: Path to WSI file
        output_dir: Output directory (should be on ext4!)
        scales: Scales to extract
        patch_coords: Optional list of (x, y) coordinates
        patch_size: Patch size in pixels
        num_workers: Parallel workers
    """
    import tifffile
    from concurrent.futures import ThreadPoolExecutor

    output_dir = Path(output_dir)

    # Create scale directories
    for scale in scales:
        (output_dir / patient / scale).mkdir(parents=True, exist_ok=True)

    # Load WSI
    print(f"Loading WSI: {wsi_path}")
    wsi = tifffile.imread(wsi_path)
    print(f"  Shape: {wsi.shape}")

    # Scale factors (relative to 2μm)
    scale_factors = {
        '2um': 1,
        '8um': 4,
        '32um': 16,
        '128um': 64
    }

    def extract_patch(coords: Tuple[int, int]):
        x, y = coords

        for scale in scales:
            factor = scale_factors.get(scale, 1)

            # Calculate region in WSI coordinates
            # At 2μm, patch covers 224*2 = 448μm
            # At 8μm, same physical region needs 224 pixels at 4x lower resolution
            center_x = x * patch_size + patch_size // 2
            center_y = y * patch_size + patch_size // 2

            # Expand region for coarser scales
            region_size = patch_size * factor
            start_x = max(0, center_x - region_size // 2)
            start_y = max(0, center_y - region_size // 2)
            end_x = min(wsi.shape[1], start_x + region_size)
            end_y = min(wsi.shape[0], start_y + region_size)

            # Extract and resize
            region = wsi[start_y:end_y, start_x:end_x]
            if region.size == 0:
                continue

            img = Image.fromarray(region)
            img = img.resize((patch_size, patch_size), Image.LANCZOS)

            # Save
            out_path = output_dir / patient / scale / f"patch_{x:03d}_{y:03d}.png"
            img.save(out_path)

    # Get coordinates from existing labels if not provided
    if patch_coords is None:
        label_dir = DATA_ROOT / f"{patient}_precomputed_labels_v2"
        patch_coords = []
        for f in label_dir.glob("patch_*.npy"):
            parts = f.stem.split('_')
            if len(parts) >= 3:
                try:
                    patch_coords.append((int(parts[1]), int(parts[2])))
                except ValueError:
                    continue

    print(f"Extracting {len(patch_coords)} patches at scales {scales}")

    # Parallel extraction
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        list(executor.map(extract_patch, patch_coords))

    print(f"Done! Patches saved to {output_dir / patient}")


def create_dataloaders(
    train_patients: List[str],
    test_patient: str,
    scales: List[str] = ['2um', '8um'],
    batch_size: int = 8,
    num_workers: int = 8
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test dataloaders.

    Args:
        train_patients: List of training patient IDs
        test_patient: Test patient ID
        scales: Scales to load
        batch_size: Batch size
        num_workers: DataLoader workers

    Returns:
        train_loader, test_loader
    """
    train_dataset = MultiScaleSTDataset(
        patients=train_patients,
        scales=scales,
        use_cache=True
    )

    test_dataset = MultiScaleSTDataset(
        patients=[test_patient],
        scales=scales,
        use_cache=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader
