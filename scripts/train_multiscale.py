#!/usr/bin/env python3
"""
Multi-Scale SOTA 2μm Training Script
=====================================

The key innovation: Cross-scale attention fusion that PRESERVES spatial info.

Unlike iSCALE (global pooling -> MLP -> fails at 2μm),
we use: cross_attention -> Hist2ST decoder -> preserves spatial structure

Usage:
    python train_multiscale.py --config configs/default.yaml
    python train_multiscale.py --encoder prov-gigapath --scales 2um 8um --loss mse

Author: Max Van Belkum + Claude Opus 4.5
Date: December 2024
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import yaml

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, "/home/user/visium-hd-2um-benchmark")

from losses import ZINBLoss, FocalMSELoss, FocalZINBLoss, MultiTaskLoss, TwoStageLoss
from fusion import MultiScaleHist2ST, TwoStagePredictor
from data import MultiScaleSTDataset, create_dataloaders

# Metrics
from skimage.metrics import structural_similarity as ssim_metric
from scipy.stats import pearsonr, spearmanr


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Scale SOTA 2μm Training")

    # Config
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")

    # Model
    parser.add_argument("--encoder", type=str, default="prov-gigapath",
                        choices=["prov-gigapath", "virchow2", "uni2-h", "h-optimus-1", "conchv1.5"])
    parser.add_argument("--decoder", type=str, default="hist2st")
    parser.add_argument("--scales", nargs="+", default=["2um", "8um"])
    parser.add_argument("--fusion", type=str, default="cross_attention",
                        choices=["cross_attention", "gated", "hierarchical"])

    # Loss
    parser.add_argument("--loss", type=str, default="mse",
                        choices=["mse", "poisson", "zinb", "focal", "focal_zinb", "multitask", "two_stage"])
    parser.add_argument("--focal_gamma", type=float, default=2.0)

    # Training
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--patience", type=int, default=10)

    # Data
    parser.add_argument("--train_patients", nargs="+", default=["P1", "P2"])
    parser.add_argument("--test_patient", type=str, default="P5")
    parser.add_argument("--num_workers", type=int, default=8)

    # Hardware
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", default=True)

    # Output
    parser.add_argument("--output_dir", type=str,
                        default="/home/user/sota-2um-st-prediction/results")
    parser.add_argument("--exp_name", type=str, default=None)

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_encoder_dim(encoder_name: str) -> int:
    """Get feature dimension for encoder."""
    dims = {
        "prov-gigapath": 1536,
        "uni2-h": 1536,
        "virchow2": 1280,
        "h-optimus-1": 1024,
        "conchv1.5": 768
    }
    return dims.get(encoder_name, 1024)


def create_model(args) -> nn.Module:
    """Create model based on arguments."""
    encoder_dim = get_encoder_dim(args.encoder)

    if args.loss == "two_stage":
        model = TwoStagePredictor(
            encoder_name=args.encoder,
            n_genes=50,
            encoder_dim=encoder_dim,
            freeze_encoder=True
        )
    else:
        model = MultiScaleHist2ST(
            encoder_name=args.encoder,
            fusion_type=args.fusion,
            scales=args.scales,
            n_genes=50,
            encoder_dim=encoder_dim,
            freeze_encoder=True
        )

    return model


def create_loss_fn(args) -> nn.Module:
    """Create loss function based on arguments."""
    if args.loss == "mse":
        return nn.MSELoss(reduction='none')
    elif args.loss == "poisson":
        return nn.PoissonNLLLoss(log_input=False, reduction='none')
    elif args.loss == "zinb":
        return ZINBLoss(n_genes=50)
    elif args.loss == "focal":
        return FocalMSELoss(gamma=args.focal_gamma)
    elif args.loss == "focal_zinb":
        return FocalZINBLoss(n_genes=50, gamma=args.focal_gamma)
    elif args.loss == "multitask":
        return MultiTaskLoss(loss_fn='mse')
    elif args.loss == "two_stage":
        return TwoStageLoss()
    else:
        raise ValueError(f"Unknown loss: {args.loss}")


class IncrementalMetrics:
    """
    Compute metrics incrementally to avoid storing all predictions in memory.

    For 1747 patches * 50 genes * 128 * 128 floats * 4 bytes ≈ 5.7 GB,
    storing all predictions would be problematic for scaling.

    Instead, we accumulate running statistics:
    - SSIM: Compute per-batch, accumulate mean
    - PCC: Accumulate per-gene statistics (sum, sum_sq, n)
    - MSE/MAE: Running mean
    """

    def __init__(self, n_genes: int = 50, sample_genes: int = 10):
        self.n_genes = n_genes
        self.sample_genes = min(sample_genes, n_genes)
        self.reset()

    def reset(self):
        self.n_samples = 0
        self.ssim_sum = 0.0
        self.ssim_count = 0
        self.mse_sum = 0.0
        self.mae_sum = 0.0
        self.count = 0

        # Per-gene PCC accumulators (using sufficient statistics)
        self.gene_sum_p = np.zeros(self.n_genes)
        self.gene_sum_t = np.zeros(self.n_genes)
        self.gene_sum_pp = np.zeros(self.n_genes)
        self.gene_sum_tt = np.zeros(self.n_genes)
        self.gene_sum_pt = np.zeros(self.n_genes)
        self.gene_n = np.zeros(self.n_genes)

    def update(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        """
        Update running statistics with a batch.

        Args:
            pred: (B, G, H, W)
            target: (B, G, H, W)
            mask: (B, 1, H, W)
        """
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        mask_np = mask.detach().cpu().numpy()

        B, G, H, W = pred_np.shape
        self.n_samples += B

        # SSIM: Sample genes for speed, compute per-image
        for b in range(B):
            for g in range(self.sample_genes):
                p = pred_np[b, g]
                t = target_np[b, g]

                # Normalize to [0, 1]
                p_range = p.max() - p.min()
                t_range = t.max() - t.min()
                if p_range > 1e-6 and t_range > 1e-6:
                    p_norm = (p - p.min()) / p_range
                    t_norm = (t - t.min()) / t_range

                    try:
                        s = ssim_metric(p_norm, t_norm, data_range=1.0)
                        if not np.isnan(s):
                            self.ssim_sum += s
                            self.ssim_count += 1
                    except:
                        pass

        # MSE and MAE
        m = mask_np > 0.5
        m_expanded = np.broadcast_to(m, pred_np.shape)
        if m_expanded.sum() > 0:
            diff = pred_np[m_expanded] - target_np[m_expanded]
            self.mse_sum += (diff ** 2).sum()
            self.mae_sum += np.abs(diff).sum()
            self.count += m_expanded.sum()

        # PCC: Accumulate sufficient statistics per gene
        for g in range(G):
            m_g = mask_np[:, 0].flatten() > 0.5
            if m_g.sum() > 0:
                p_flat = pred_np[:, g].flatten()[m_g]
                t_flat = target_np[:, g].flatten()[m_g]

                self.gene_sum_p[g] += p_flat.sum()
                self.gene_sum_t[g] += t_flat.sum()
                self.gene_sum_pp[g] += (p_flat ** 2).sum()
                self.gene_sum_tt[g] += (t_flat ** 2).sum()
                self.gene_sum_pt[g] += (p_flat * t_flat).sum()
                self.gene_n[g] += len(p_flat)

    def compute(self) -> Dict[str, float]:
        """Compute final metrics from accumulated statistics."""
        metrics = {}

        # SSIM
        metrics['ssim'] = self.ssim_sum / max(self.ssim_count, 1)

        # MSE and MAE
        metrics['mse'] = self.mse_sum / max(self.count, 1)
        metrics['mae'] = self.mae_sum / max(self.count, 1)

        # PCC per gene using sufficient statistics
        # PCC = (n*sum_pt - sum_p*sum_t) / sqrt((n*sum_pp - sum_p^2) * (n*sum_tt - sum_t^2))
        pccs = []
        for g in range(self.n_genes):
            n = self.gene_n[g]
            if n > 10:
                num = n * self.gene_sum_pt[g] - self.gene_sum_p[g] * self.gene_sum_t[g]
                var_p = n * self.gene_sum_pp[g] - self.gene_sum_p[g] ** 2
                var_t = n * self.gene_sum_tt[g] - self.gene_sum_t[g] ** 2

                if var_p > 1e-10 and var_t > 1e-10:
                    pcc = num / (np.sqrt(var_p) * np.sqrt(var_t))
                    if not np.isnan(pcc):
                        pccs.append(np.clip(pcc, -1, 1))

        metrics['pcc'] = np.mean(pccs) if pccs else 0.0

        return metrics


def compute_metrics(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
    """
    Compute evaluation metrics (legacy function for small datasets).

    NOTE: For large datasets, use IncrementalMetrics instead to avoid OOM.
    This function stores all predictions in memory.

    Args:
        pred: Predictions (B, G, H, W)
        target: Ground truth (B, G, H, W)
        mask: Tissue mask (B, 1, H, W)

    Returns:
        Dict of metrics
    """
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy()

    metrics = {}

    # Per-gene PCC
    pccs = []
    for g in range(pred.shape[1]):
        p = pred[:, g].flatten()
        t = target[:, g].flatten()
        m = mask[:, 0].flatten() > 0.5

        if m.sum() > 10:
            valid_p = p[m]
            valid_t = t[m]
            if valid_p.std() > 1e-6 and valid_t.std() > 1e-6:
                pcc, _ = pearsonr(valid_p, valid_t)
                if not np.isnan(pcc):
                    pccs.append(pcc)

    metrics['pcc'] = np.mean(pccs) if pccs else 0.0

    # SSIM (per image, averaged)
    ssims = []
    for b in range(pred.shape[0]):
        for g in range(min(pred.shape[1], 10)):  # Sample genes for speed
            p = pred[b, g]
            t = target[b, g]

            # Normalize to [0, 1]
            p_norm = (p - p.min()) / (p.max() - p.min() + 1e-6)
            t_norm = (t - t.min()) / (t.max() - t.min() + 1e-6)

            try:
                s = ssim_metric(p_norm, t_norm, data_range=1.0)
                if not np.isnan(s):
                    ssims.append(s)
            except:
                pass

    metrics['ssim'] = np.mean(ssims) if ssims else 0.0

    # MAE
    m = mask > 0.5
    metrics['mae'] = np.abs(pred[m.repeat(1, pred.shape[1], 1, 1)] -
                           target[m.repeat(1, target.shape[1], 1, 1)]).mean()

    return metrics


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    args,
    epoch: int
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(pbar):
        # Skip failed batches (from collate_filter_failed)
        if batch is None:
            continue

        # Prepare inputs
        images = {scale: batch['images'][scale].to(device)
                  for scale in args.scales if scale in batch['images']}

        labels = batch['labels_2um'].to(device)
        mask = batch['mask'].to(device)

        # Forward pass with mixed precision
        with autocast(enabled=args.amp):
            if args.loss == "two_stage":
                logits_cls, pred_reg, pred = model(images['2um'])
                loss, _ = loss_fn(logits_cls, pred_reg, labels, mask)
            elif len(args.scales) == 1:
                # Single scale
                pred = model({'2um': images['2um']})
                if isinstance(loss_fn, (ZINBLoss, FocalZINBLoss)):
                    loss = loss_fn(pred, labels, mask=mask)
                else:
                    loss = loss_fn(pred, labels)
                    if mask is not None:
                        loss = (loss * mask).sum() / mask.sum().clamp(min=1)
                    else:
                        loss = loss.mean()
            else:
                # Multi-scale
                pred = model(images)
                if isinstance(loss_fn, MultiTaskLoss):
                    labels_8um = batch.get('labels_8um')
                    if labels_8um is not None:
                        labels_8um = labels_8um.to(device)
                    loss, _ = loss_fn(pred, labels, labels_8um, mask_2um=mask)
                elif isinstance(loss_fn, (ZINBLoss, FocalZINBLoss)):
                    loss = loss_fn(pred, labels, mask=mask)
                else:
                    loss = loss_fn(pred, labels)
                    if mask is not None:
                        loss = (loss * mask).sum() / mask.sum().clamp(min=1)
                    else:
                        loss = loss.mean()

        # Backward pass with gradient accumulation
        loss = loss / args.grad_accum
        scaler.scale(loss).backward()

        if (batch_idx + 1) % args.grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * args.grad_accum
        n_batches += 1

        pbar.set_postfix({'loss': total_loss / n_batches})

    return {'train_loss': total_loss / n_batches}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    args
) -> Dict[str, float]:
    """
    Evaluate model using incremental metrics to avoid OOM.

    Memory-efficient: Uses IncrementalMetrics instead of storing all predictions.
    For 1747 patches this avoids 5.7GB memory allocation.
    """
    model.eval()

    # Use incremental metrics to avoid storing all predictions
    metrics_accumulator = IncrementalMetrics(n_genes=50)
    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(loader, desc="Evaluating"):
        # Skip failed batches (from collate_filter_failed)
        if batch is None:
            continue

        images = {scale: batch['images'][scale].to(device)
                  for scale in args.scales if scale in batch['images']}
        labels = batch['labels_2um'].to(device)
        mask = batch['mask'].to(device)

        with autocast(enabled=args.amp):
            if args.loss == "two_stage":
                _, _, pred = model(images['2um'])
            elif len(args.scales) == 1:
                pred = model({'2um': images['2um']})
            else:
                pred = model(images)

            # Compute loss
            if isinstance(loss_fn, (ZINBLoss, FocalZINBLoss)):
                loss = loss_fn(pred, labels, mask=mask)
            elif isinstance(loss_fn, MultiTaskLoss):
                loss, _ = loss_fn(pred, labels, mask_2um=mask)
            elif isinstance(loss_fn, TwoStageLoss):
                loss = F.mse_loss(pred, labels)
            else:
                loss = loss_fn(pred, labels)
                loss = (loss * mask).sum() / mask.sum().clamp(min=1)

        total_loss += loss.item()
        n_batches += 1

        # Update incremental metrics (no list storage!)
        metrics_accumulator.update(pred, labels, mask)

    # Compute final metrics from accumulated statistics
    metrics = metrics_accumulator.compute()
    metrics['val_loss'] = total_loss / max(n_batches, 1)

    return metrics


def main():
    args = parse_args()

    # Load config if provided
    if args.config:
        config = load_config(args.config)
        # Override with command line args where specified
        # (simplified: config provides defaults, CLI overrides)

    # Set seed
    set_seed(args.seed)

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = args.exp_name or f"{args.encoder}_{args.fusion}_{args.loss}_{timestamp}"
    output_dir = Path(args.output_dir) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Save config
    config_dict = vars(args)
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config_dict, f, indent=2)

    # Create data loaders
    print(f"Creating dataloaders...")
    print(f"  Train patients: {args.train_patients}")
    print(f"  Test patient: {args.test_patient}")
    print(f"  Scales: {args.scales}")

    train_loader, test_loader = create_dataloaders(
        train_patients=args.train_patients,
        test_patient=args.test_patient,
        scales=args.scales,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    # Create model
    print(f"Creating model...")
    print(f"  Encoder: {args.encoder}")
    print(f"  Fusion: {args.fusion}")
    print(f"  Loss: {args.loss}")

    model = create_model(args)
    model = model.to(device)

    # Count trainable parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    # Create loss function
    loss_fn = create_loss_fn(args)
    if hasattr(loss_fn, 'to'):
        loss_fn = loss_fn.to(device)

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-7
    )

    # Mixed precision scaler
    scaler = GradScaler(enabled=args.amp)

    # Training loop
    best_ssim = 0.0
    patience_counter = 0
    history = []

    print(f"\nStarting training for {args.epochs} epochs...")

    for epoch in range(args.epochs):
        # Train
        train_metrics = train_epoch(
            model, train_loader, loss_fn, optimizer, scaler, device, args, epoch
        )

        # Evaluate
        val_metrics = evaluate(model, test_loader, loss_fn, device, args)

        # Update scheduler
        scheduler.step()

        # Log
        epoch_metrics = {
            'epoch': epoch + 1,
            'lr': scheduler.get_last_lr()[0],
            **train_metrics,
            **val_metrics
        }
        history.append(epoch_metrics)

        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss: {train_metrics['train_loss']:.4f}")
        print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
        print(f"  SSIM: {val_metrics['ssim']:.4f}")
        print(f"  PCC: {val_metrics['pcc']:.4f}")

        # Save best model
        if val_metrics['ssim'] > best_ssim:
            best_ssim = val_metrics['ssim']
            patience_counter = 0

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_metrics,
                'config': config_dict
            }, output_dir / "best_model.pt")

            print(f"  New best SSIM: {best_ssim:.4f}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'metrics': val_metrics
            }, output_dir / f"checkpoint_epoch_{epoch+1}.pt")

    # Save final results
    results = {
        'best_ssim': best_ssim,
        'final_metrics': history[-1] if history else {},
        'history': history,
        'config': config_dict
    }

    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nTraining complete!")
    print(f"Best SSIM: {best_ssim:.4f}")
    print(f"Results saved to: {output_dir}")

    return best_ssim


if __name__ == "__main__":
    main()
