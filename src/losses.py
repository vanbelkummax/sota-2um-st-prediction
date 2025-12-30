"""
Advanced Loss Functions for Sparse Spatial Transcriptomics
===========================================================

Handles 95-98% sparsity at 2μm resolution through:
1. ZINB (Zero-Inflated Negative Binomial) - models structural zeros
2. Focal Loss - downweights easy examples (zeros)
3. Multi-task Loss - joint multi-resolution training

Author: Max Van Belkum + Claude Opus 4.5
Date: December 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ZINBLoss(nn.Module):
    """
    Zero-Inflated Negative Binomial Loss for sparse count data.

    Models gene expression as mixture:
    - With probability π: structural zero (gene not expressed in this cell type)
    - With probability (1-π): NB(μ, θ) distribution

    Args:
        n_genes: Number of genes (for per-gene dispersion)
        eps: Small constant for numerical stability
        learn_theta: Whether to learn per-gene dispersion
        learn_pi: Whether to predict zero-inflation probability
    """

    def __init__(
        self,
        n_genes: int = 50,
        eps: float = 1e-10,
        learn_theta: bool = True,
        learn_pi: bool = True,
        init_theta: float = 1.0
    ):
        super().__init__()
        self.eps = eps
        self.n_genes = n_genes
        self.learn_theta = learn_theta
        self.learn_pi = learn_pi

        # Per-gene dispersion parameter (learnable)
        if learn_theta:
            # Initialize in log-space for stability
            self.log_theta = nn.Parameter(torch.full((n_genes,), float(torch.log(torch.tensor(init_theta)))))
        else:
            self.register_buffer('log_theta', torch.full((n_genes,), float(torch.log(torch.tensor(init_theta)))))

    @property
    def theta(self) -> torch.Tensor:
        """Dispersion parameter (always positive via exp)"""
        return torch.exp(self.log_theta).clamp(min=self.eps, max=1e6)

    def forward(
        self,
        mu: torch.Tensor,
        target: torch.Tensor,
        pi: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute ZINB negative log-likelihood.

        Args:
            mu: Predicted mean (B, G, H, W) - must be positive
            target: Observed counts (B, G, H, W)
            pi: Zero-inflation probability (B, G, H, W) or None
            mask: Valid tissue mask (B, 1, H, W)

        Returns:
            Scalar loss value
        """
        # Ensure mu is positive
        mu = F.softplus(mu) + self.eps

        # Get theta with correct shape for broadcasting
        theta = self.theta.view(1, -1, 1, 1)  # (1, G, 1, 1)

        # If pi not provided, use fixed small value
        if pi is None:
            pi = torch.zeros_like(mu) + 0.1
        else:
            pi = torch.sigmoid(pi)  # Ensure [0, 1]

        # Probability of zero from NB component
        # P(Y=0|NB) = (theta / (theta + mu))^theta
        nb_zero_prob = torch.pow(theta / (theta + mu + self.eps), theta)

        # Total probability of zero (mixture)
        prob_zero = pi + (1 - pi) * nb_zero_prob

        # Log probability for non-zero observations (NB component)
        # log P(Y=y|NB) = log Gamma(y+theta) - log Gamma(theta) - log Gamma(y+1)
        #                 + theta*log(theta/(theta+mu)) + y*log(mu/(theta+mu))
        log_nb_nonzero = (
            torch.lgamma(target + theta)
            - torch.lgamma(theta)
            - torch.lgamma(target + 1)
            + theta * torch.log(theta / (theta + mu + self.eps) + self.eps)
            + target * torch.log(mu / (theta + mu + self.eps) + self.eps)
        )

        # Probability for non-zero (must come from NB, not zero-inflation)
        prob_nonzero = (1 - pi) * torch.exp(log_nb_nonzero)

        # Negative log-likelihood
        nll = -torch.where(
            target < 0.5,  # Effectively target == 0
            torch.log(prob_zero + self.eps),
            torch.log(prob_nonzero + self.eps)
        )

        # Apply mask if provided
        if mask is not None:
            mask = mask.expand_as(nll)
            nll = nll * mask
            n_valid = mask.sum().clamp(min=1)
            return nll.sum() / n_valid

        return nll.mean()


class FocalMSELoss(nn.Module):
    """
    Focal Loss variant for regression with sparse targets.

    Downweights easy examples (zeros) to focus on harder non-zero predictions.

    focal_weight = (1 - exp(-mse))^gamma

    Args:
        gamma: Focusing parameter (higher = more focus on hard examples)
        alpha: Balance factor for zeros vs non-zeros
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute focal MSE loss.

        Args:
            pred: Predictions (B, G, H, W)
            target: Ground truth (B, G, H, W)
            mask: Valid tissue mask (B, 1, H, W)

        Returns:
            Scalar loss value
        """
        # Standard MSE
        mse = (pred - target) ** 2

        # Focal weight: hard examples get higher weight
        # Use negative exponential to map MSE to [0, 1] range
        focal_weight = torch.pow(1 - torch.exp(-mse), self.gamma)

        # Additional alpha weighting for non-zeros
        is_nonzero = (target.abs() > 0.01).float()
        alpha_weight = is_nonzero * self.alpha + (1 - is_nonzero) * (1 - self.alpha)

        # Combined weighted loss
        weighted_loss = focal_weight * alpha_weight * mse

        # Apply mask if provided
        if mask is not None:
            mask = mask.expand_as(weighted_loss)
            weighted_loss = weighted_loss * mask
            n_valid = mask.sum().clamp(min=1)
            return weighted_loss.sum() / n_valid

        return weighted_loss.mean()


class FocalZINBLoss(nn.Module):
    """
    Combination of ZINB and Focal weighting.

    Uses ZINB for proper count modeling, adds focal weighting to
    focus on harder predictions.
    """

    def __init__(
        self,
        n_genes: int = 50,
        gamma: float = 2.0,
        eps: float = 1e-10
    ):
        super().__init__()
        self.zinb = ZINBLoss(n_genes=n_genes, eps=eps)
        self.gamma = gamma

    def forward(
        self,
        mu: torch.Tensor,
        target: torch.Tensor,
        pi: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Focal-weighted ZINB loss."""
        # Get per-element ZINB NLL
        mu_safe = F.softplus(mu) + self.zinb.eps
        theta = self.zinb.theta.view(1, -1, 1, 1)

        if pi is None:
            pi = torch.zeros_like(mu) + 0.1
        else:
            pi = torch.sigmoid(pi)

        # Compute NLL per element
        nb_zero_prob = torch.pow(theta / (theta + mu_safe + self.zinb.eps), theta)
        prob_zero = pi + (1 - pi) * nb_zero_prob

        log_nb_nonzero = (
            torch.lgamma(target + theta)
            - torch.lgamma(theta)
            - torch.lgamma(target + 1)
            + theta * torch.log(theta / (theta + mu_safe + self.zinb.eps) + self.zinb.eps)
            + target * torch.log(mu_safe / (theta + mu_safe + self.zinb.eps) + self.zinb.eps)
        )
        prob_nonzero = (1 - pi) * torch.exp(log_nb_nonzero)

        nll = -torch.where(
            target < 0.5,
            torch.log(prob_zero + self.zinb.eps),
            torch.log(prob_nonzero + self.zinb.eps)
        )

        # Focal weighting based on prediction difficulty
        pred_error = (mu_safe - target).abs()
        focal_weight = torch.pow(1 - torch.exp(-pred_error), self.gamma)

        weighted_nll = focal_weight * nll

        if mask is not None:
            mask = mask.expand_as(weighted_nll)
            weighted_nll = weighted_nll * mask
            n_valid = mask.sum().clamp(min=1)
            return weighted_nll.sum() / n_valid

        return weighted_nll.mean()


class MultiTaskLoss(nn.Module):
    """
    Multi-resolution training loss.

    Combines losses at multiple resolutions with learned or fixed weights.

    L_total = w_2um * L_2um + w_8um * L_8um + w_32um * L_32um

    8μm and 32μm targets are less noisy (spatially averaged), providing
    smoother gradients for learning.
    """

    def __init__(
        self,
        loss_fn: str = 'mse',
        weights: Tuple[float, float, float] = (1.0, 0.5, 0.25),
        learn_weights: bool = False,
        n_genes: int = 50
    ):
        super().__init__()

        # Base loss function
        if loss_fn == 'mse':
            self.loss_2um = nn.MSELoss(reduction='none')
            self.loss_8um = nn.MSELoss(reduction='none')
            self.loss_32um = nn.MSELoss(reduction='none')
        elif loss_fn == 'zinb':
            self.loss_2um = ZINBLoss(n_genes=n_genes)
            self.loss_8um = ZINBLoss(n_genes=n_genes)
            self.loss_32um = ZINBLoss(n_genes=n_genes)
        elif loss_fn == 'focal':
            self.loss_2um = FocalMSELoss()
            self.loss_8um = FocalMSELoss()
            self.loss_32um = FocalMSELoss()
        else:
            raise ValueError(f"Unknown loss function: {loss_fn}")

        self.loss_fn = loss_fn

        # Resolution weights
        if learn_weights:
            # Learnable in log-space
            self.log_weights = nn.Parameter(torch.log(torch.tensor(weights)))
        else:
            self.register_buffer('log_weights', torch.log(torch.tensor(weights)))

    @property
    def weights(self) -> torch.Tensor:
        """Normalized weights via softmax"""
        return F.softmax(self.log_weights, dim=0)

    def forward(
        self,
        pred_2um: torch.Tensor,
        target_2um: torch.Tensor,
        target_8um: Optional[torch.Tensor] = None,
        target_32um: Optional[torch.Tensor] = None,
        mask_2um: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute multi-resolution loss.

        Args:
            pred_2um: 2μm predictions (B, G, 128, 128)
            target_2um: 2μm ground truth (B, G, 128, 128)
            target_8um: 8μm ground truth (B, G, 32, 32) - optional
            target_32um: 32μm ground truth (B, G, 8, 8) - optional
            mask_2um: Tissue mask at 2μm (B, 1, 128, 128)

        Returns:
            total_loss: Weighted sum of losses
            loss_dict: Individual loss components
        """
        w = self.weights
        loss_dict = {}

        # 2μm loss (primary)
        if self.loss_fn == 'mse':
            l2 = self.loss_2um(pred_2um, target_2um)
            if mask_2um is not None:
                l2 = (l2 * mask_2um).sum() / mask_2um.sum().clamp(min=1)
            else:
                l2 = l2.mean()
        else:
            l2 = self.loss_2um(pred_2um, target_2um, mask=mask_2um)
        loss_dict['loss_2um'] = l2.item()

        total_loss = w[0] * l2

        # 8μm loss (auxiliary)
        if target_8um is not None:
            pred_8um = F.avg_pool2d(pred_2um, kernel_size=4)
            if self.loss_fn == 'mse':
                l8 = self.loss_8um(pred_8um, target_8um).mean()
            else:
                l8 = self.loss_8um(pred_8um, target_8um)
            loss_dict['loss_8um'] = l8.item()
            total_loss = total_loss + w[1] * l8

        # 32μm loss (auxiliary)
        if target_32um is not None:
            pred_32um = F.avg_pool2d(pred_2um, kernel_size=16)
            if self.loss_fn == 'mse':
                l32 = self.loss_32um(pred_32um, target_32um).mean()
            else:
                l32 = self.loss_32um(pred_32um, target_32um)
            loss_dict['loss_32um'] = l32.item()
            total_loss = total_loss + w[2] * l32

        loss_dict['total'] = total_loss.item()
        loss_dict['weights'] = w.detach().cpu().tolist()

        return total_loss, loss_dict


class TwoStageLoss(nn.Module):
    """
    Two-stage prediction loss for sparse data.

    Stage 1: Binary classification - is gene expressed?
    Stage 2: Regression - how much expression? (only on expressed bins)
    """

    def __init__(self, cls_weight: float = 1.0, reg_weight: float = 1.0):
        super().__init__()
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.mse = nn.MSELoss(reduction='none')

    def forward(
        self,
        logits_cls: torch.Tensor,
        pred_reg: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute two-stage loss.

        Args:
            logits_cls: Classification logits (B, G, H, W)
            pred_reg: Regression predictions (B, G, H, W)
            target: Ground truth counts (B, G, H, W)
            mask: Tissue mask (B, 1, H, W)

        Returns:
            total_loss, loss_dict
        """
        # Stage 1: Binary classification
        is_expressed = (target > 0.01).float()
        loss_cls = self.bce(logits_cls, is_expressed)

        # Stage 2: Regression only on expressed bins
        expressed_mask = is_expressed > 0.5
        loss_reg = self.mse(pred_reg, target)
        loss_reg = loss_reg * expressed_mask.float()

        # Apply tissue mask
        if mask is not None:
            mask = mask.expand_as(loss_cls)
            loss_cls = (loss_cls * mask).sum() / mask.sum().clamp(min=1)
            n_expressed = (expressed_mask * mask).sum().clamp(min=1)
            loss_reg = (loss_reg * mask).sum() / n_expressed
        else:
            loss_cls = loss_cls.mean()
            n_expressed = expressed_mask.sum().clamp(min=1)
            loss_reg = loss_reg.sum() / n_expressed

        total_loss = self.cls_weight * loss_cls + self.reg_weight * loss_reg

        return total_loss, {
            'loss_cls': loss_cls.item(),
            'loss_reg': loss_reg.item(),
            'total': total_loss.item(),
            'n_expressed': expressed_mask.sum().item()
        }
