"""
Multi-Scale Feature Fusion Modules
==================================

Key insight: iSCALE's global pooling destroys spatial info.
Our approach: Cross-scale attention that PRESERVES spatial dimensions.

2μm features QUERY 8μm/32μm context without losing (H, W) structure.

CRITICAL FOV ALIGNMENT NOTE:
----------------------------
When extracting 224x224 patches at different scales (2μm, 8μm, etc.),
the physical Field of View (FOV) differs:
- 2μm patch at 224px covers ~448μm physical area
- 8μm patch at 224px covers ~1792μm physical area (4x larger)

The 2μm patch is the CENTER CROP of the 8μm region.
For element-wise fusion (GatedScaleFusion), we must extract the
center region of coarse features that corresponds to the 2μm FOV.

CrossScaleAttention can learn this mapping via attention weights,
but GatedScaleFusion requires explicit center-cropping.

Author: Max Van Belkum
Date: December 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import math


class CrossScaleAttention(nn.Module):
    """
    Cross-scale attention for multi-resolution feature fusion.

    2μm features serve as QUERIES, 8μm/32μm features provide CONTEXT.
    Crucially, spatial dimensions (H, W) are PRESERVED throughout.

    This is the key difference from iSCALE:
    - iSCALE: concat -> global_pool -> MLP (loses spatial info)
    - Ours: cross_attention -> spatial_decoder (preserves spatial info)

    Args:
        dim: Feature dimension
        num_heads: Number of attention heads
        num_scales: Number of context scales (default 2: 8μm, 32μm)
        dropout: Attention dropout
    """

    def __init__(
        self,
        dim: int = 1024,
        num_heads: int = 8,
        num_scales: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Query projection for 2μm features
        self.q_proj = nn.Linear(dim, dim)

        # Key/Value projections for each context scale
        self.k_projs = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_scales)])
        self.v_projs = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_scales)])

        # Output projection
        self.out_proj = nn.Linear(dim, dim)

        # Learnable scale importance weights
        self.scale_weights = nn.Parameter(torch.ones(num_scales) / num_scales)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        query: torch.Tensor,
        context: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Cross-scale attention.

        Args:
            query: 2μm features (B, C, H, W)
            context: List of [8μm features, 32μm features, ...] each (B, C, H, W)

        Returns:
            Fused features (B, C, H, W) - SPATIAL PRESERVED
        """
        B, C, H, W = query.shape

        # Flatten spatial dims for attention: (B, H*W, C)
        q = query.flatten(2).transpose(1, 2)  # (B, H*W, C)

        # Project query
        q = self.q_proj(q)  # (B, H*W, C)
        q = q.view(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)  # (B, heads, H*W, head_dim)

        # Aggregate attention from each context scale
        scale_weights = F.softmax(self.scale_weights, dim=0)
        attn_output = torch.zeros_like(q)

        for i, ctx in enumerate(context):
            # Flatten context
            ctx_flat = ctx.flatten(2).transpose(1, 2)  # (B, H*W, C)

            # Project key/value
            k = self.k_projs[i](ctx_flat)
            v = self.v_projs[i](ctx_flat)

            # Reshape for multi-head attention
            k = k.view(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)

            # Scaled dot-product attention
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.dropout(attn_weights)

            # Weighted attention output
            scale_out = torch.matmul(attn_weights, v)  # (B, heads, H*W, head_dim)
            attn_output = attn_output + scale_weights[i] * scale_out

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, H * W, C)  # (B, H*W, C)
        attn_output = self.out_proj(attn_output)

        # Residual connection and reshape to spatial
        output = q.transpose(1, 2).contiguous().view(B, H * W, C)  # Reshape q back
        output = self.norm(output + attn_output)
        output = output.transpose(1, 2).view(B, C, H, W)  # (B, C, H, W)

        return output


class GatedScaleFusion(nn.Module):
    """
    Gated fusion of multi-scale features WITH FOV ALIGNMENT.

    CRITICAL: Different scales have different FOVs!
    - 2μm features at (0,0) represent top-left of the 2μm patch
    - 8μm features at (0,0) represent top-left of the 8μm patch (4x larger area)

    The 2μm patch is the CENTER of the 8μm patch in physical space.
    Before element-wise fusion, we must center-crop coarse features.

    Scale factors (relative to 2μm):
    - 2μm: 1x (reference)
    - 8μm: 4x larger FOV
    - 32μm: 16x larger FOV
    - 128μm: 64x larger FOV

    gate_i = sigmoid(W_i * global_pool(aligned_feat_i))
    output = sum(gate_i * aligned_feat_i)
    """

    # Scale factors for FOV alignment (relative to 2μm)
    SCALE_FACTORS = {
        '2um': 1,
        '8um': 4,
        '32um': 16,
        '128um': 64
    }

    def __init__(self, dim: int = 1024, num_scales: int = 3,
                 scale_names: List[str] = None):
        super().__init__()
        self.num_scales = num_scales

        # Default scale names if not provided
        if scale_names is None:
            scale_names = ['2um', '8um', '32um'][:num_scales]
        self.scale_names = scale_names

        # Per-scale gating networks
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Linear(dim, dim),
                nn.Sigmoid()
            )
            for _ in range(num_scales)
        ])

    def _center_crop_and_resize(
        self,
        feat: torch.Tensor,
        scale_factor: int,
        target_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Extract center region of coarse features corresponding to fine FOV.

        Args:
            feat: Coarse features (B, C, H, W)
            scale_factor: How much larger the FOV is (e.g., 4 for 8μm vs 2μm)
            target_size: Target spatial size (H, W)

        Returns:
            Aligned features (B, C, target_H, target_W)
        """
        if scale_factor == 1:
            return feat

        B, C, H, W = feat.shape

        # The center region that corresponds to the fine-scale FOV
        # is 1/scale_factor of the total area, centered
        crop_h = H // scale_factor
        crop_w = W // scale_factor

        # Ensure minimum size of 1
        crop_h = max(1, crop_h)
        crop_w = max(1, crop_w)

        # Calculate center crop coordinates
        start_h = (H - crop_h) // 2
        start_w = (W - crop_w) // 2

        # Extract center crop
        cropped = feat[:, :, start_h:start_h+crop_h, start_w:start_w+crop_w]

        # Resize to match target (interpolate to broadcast context)
        if cropped.shape[-2:] != target_size:
            cropped = F.interpolate(
                cropped,
                size=target_size,
                mode='bilinear',
                align_corners=False
            )

        return cropped

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Gated fusion with FOV alignment.

        Args:
            features: List of [2μm, 8μm, 32μm, ...] features, each (B, C, H, W)
                      First element is assumed to be 2μm (finest scale)

        Returns:
            Fused features (B, C, H, W) at 2μm resolution
        """
        assert len(features) == self.num_scales

        # Target size is the finest scale (2μm)
        target = features[0]
        target_size = target.shape[-2:]

        output = torch.zeros_like(target)

        for i, feat in enumerate(features):
            scale_name = self.scale_names[i]
            scale_factor = self.SCALE_FACTORS.get(scale_name, 1)

            # Align coarse features to 2μm FOV via center-crop
            aligned_feat = self._center_crop_and_resize(
                feat, scale_factor, target_size
            )

            # Compute gate from aligned features
            gate = self.gates[i](aligned_feat)  # (B, C)
            gate = gate.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)

            output = output + gate * aligned_feat

        return output


class HierarchicalFusion(nn.Module):
    """
    Hierarchical bottom-up fusion of multi-scale features.

    Coarse scales are progressively refined by finer scales:
    128μm -> 32μm -> 8μm -> 2μm

    Each step uses cross-attention where finer scale queries coarser.
    """

    def __init__(self, dim: int = 1024, num_heads: int = 8):
        super().__init__()

        # Cross-attention at each level
        self.attn_32_from_128 = CrossScaleAttention(dim, num_heads, num_scales=1)
        self.attn_8_from_32 = CrossScaleAttention(dim, num_heads, num_scales=1)
        self.attn_2_from_8 = CrossScaleAttention(dim, num_heads, num_scales=1)

        # Feature projection/normalization at each level
        self.proj_128 = nn.Conv2d(dim, dim, 1)
        self.proj_32 = nn.Conv2d(dim, dim, 1)
        self.proj_8 = nn.Conv2d(dim, dim, 1)
        self.proj_2 = nn.Conv2d(dim, dim, 1)

    def forward(
        self,
        feat_2: torch.Tensor,
        feat_8: torch.Tensor,
        feat_32: Optional[torch.Tensor] = None,
        feat_128: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Hierarchical fusion from coarse to fine.

        Args:
            feat_2: 2μm features (B, C, H, W)
            feat_8: 8μm features (B, C, H, W)
            feat_32: 32μm features (optional)
            feat_128: 128μm features (optional)

        Returns:
            Refined 2μm features (B, C, H, W)
        """
        # Start with finest scale
        feat_2 = self.proj_2(feat_2)
        feat_8 = self.proj_8(feat_8)

        if feat_128 is not None and feat_32 is not None:
            feat_128 = self.proj_128(feat_128)
            feat_32 = self.proj_32(feat_32)

            # 32μm queries 128μm context
            feat_32 = self.attn_32_from_128(feat_32, [feat_128])

        if feat_32 is not None:
            # 8μm queries 32μm context
            feat_8 = self.attn_8_from_32(feat_8, [feat_32])

        # 2μm queries 8μm context
        feat_2 = self.attn_2_from_8(feat_2, [feat_8])

        return feat_2


class MultiScaleHist2ST(nn.Module):
    """
    Multi-Scale Hist2ST: The complete architecture.

    Combines:
    1. Frozen pathology encoder (Prov-GigaPath, Virchow2, etc.)
    2. Cross-scale attention fusion
    3. Hist2ST spatial decoder

    This is the key innovation: multi-scale features flow through
    a SPATIAL decoder, not an MLP.
    """

    def __init__(
        self,
        encoder_name: str = 'prov-gigapath',
        decoder_name: str = 'hist2st',
        fusion_type: str = 'cross_attention',
        scales: List[str] = ['2um', '8um'],
        n_genes: int = 50,
        encoder_dim: int = 1536,
        hidden_dim: int = 1024,
        freeze_encoder: bool = True
    ):
        super().__init__()

        self.scales = scales
        self.n_genes = n_genes
        self.freeze_encoder = freeze_encoder

        # Import encoder wrapper (from existing codebase)
        import sys
        sys.path.insert(0, '/home/user/visium-hd-2um-benchmark')
        from model.encoder_wrapper import get_spatial_encoder

        # Load frozen encoder
        self.encoder = get_spatial_encoder(encoder_name)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()

        # Scale-specific projectors (encoder_dim -> hidden_dim)
        self.scale_projectors = nn.ModuleDict({
            scale: nn.Sequential(
                nn.Conv2d(encoder_dim, hidden_dim, 1),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU()
            )
            for scale in scales
        })

        # Fusion module
        if fusion_type == 'cross_attention':
            self.fusion = CrossScaleAttention(
                dim=hidden_dim,
                num_heads=8,
                num_scales=len(scales) - 1  # Context scales (exclude 2μm)
            )
        elif fusion_type == 'gated':
            self.fusion = GatedScaleFusion(dim=hidden_dim, num_scales=len(scales))
        elif fusion_type == 'hierarchical':
            self.fusion = HierarchicalFusion(dim=hidden_dim)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

        self.fusion_type = fusion_type

        # Import Hist2ST decoder
        from model.enhanced_decoder import Hist2STDecoder
        self.decoder = Hist2STDecoder(
            in_channels=hidden_dim,
            out_channels=n_genes,
            hidden_dim=512
        )

    def forward(
        self,
        images: dict  # {'2um': tensor, '8um': tensor, ...}
    ) -> torch.Tensor:
        """
        Multi-scale forward pass.

        Args:
            images: Dict mapping scale name to image tensor (B, 3, 224, 224)

        Returns:
            Gene predictions (B, n_genes, 128, 128)
        """
        # Encode each scale (frozen)
        features = {}
        with torch.set_grad_enabled(not self.freeze_encoder):
            for scale in self.scales:
                if scale in images:
                    feat = self.encoder(images[scale])  # (B, encoder_dim, 14, 14)
                    feat = self.scale_projectors[scale](feat)  # (B, hidden_dim, 14, 14)
                    features[scale] = feat

        # Fusion
        if self.fusion_type == 'cross_attention':
            # 2μm queries, other scales provide context
            query = features['2um']
            context = [features[s] for s in self.scales if s != '2um']
            fused = self.fusion(query, context)
        elif self.fusion_type == 'gated':
            fused = self.fusion([features[s] for s in self.scales])
        elif self.fusion_type == 'hierarchical':
            fused = self.fusion(
                features.get('2um'),
                features.get('8um'),
                features.get('32um'),
                features.get('128um')
            )

        # Decode (spatial decoder preserves structure)
        pred = self.decoder(fused)  # (B, n_genes, 128, 128)

        return pred


class TwoStagePredictor(nn.Module):
    """
    Two-stage prediction for sparse data.

    Stage 1: Binary classification - is gene expressed?
    Stage 2: Regression - how much? (gated by classification)

    Final output = sigmoid(cls) * softplus(reg)
    """

    def __init__(
        self,
        encoder_name: str = 'prov-gigapath',
        n_genes: int = 50,
        encoder_dim: int = 1536,
        hidden_dim: int = 1024,
        freeze_encoder: bool = True
    ):
        super().__init__()

        import sys
        sys.path.insert(0, '/home/user/visium-hd-2um-benchmark')
        from model.encoder_wrapper import get_spatial_encoder
        from model.enhanced_decoder import Hist2STDecoder

        self.encoder = get_spatial_encoder(encoder_name)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()

        self.projector = nn.Sequential(
            nn.Conv2d(encoder_dim, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU()
        )

        # Two separate decoders
        self.classification_head = Hist2STDecoder(hidden_dim, n_genes, hidden_dim=256)
        self.regression_head = Hist2STDecoder(hidden_dim, n_genes, hidden_dim=256)

        self.freeze_encoder = freeze_encoder

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Two-stage forward.

        Returns:
            logits_cls: Classification logits (B, G, H, W)
            pred_reg: Regression output (B, G, H, W)
            pred_final: Gated prediction (B, G, H, W)
        """
        with torch.set_grad_enabled(not self.freeze_encoder):
            feat = self.encoder(x)

        feat = self.projector(feat)

        logits_cls = self.classification_head(feat)
        pred_reg = F.softplus(self.regression_head(feat))

        # Gated output
        pred_final = torch.sigmoid(logits_cls) * pred_reg

        return logits_cls, pred_reg, pred_final
