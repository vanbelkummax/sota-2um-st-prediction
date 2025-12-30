"""
Foundation Model Encoder Registry
==================================

Unified interface for pathology foundation models:
- Prov-GigaPath (1536-dim) - Best performer in encoder ablation
- Virchow2 (1280-dim)
- UNI2-h (1536-dim)
- H-optimus-1 (1024-dim)
- CONCH v1.5 (768-dim)

All encoders return features of shape (B, H, W, D) or (B, D) depending on config.

Author: Max Van Belkum + Claude Opus 4.5
Date: December 2024
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Union
import warnings


# Encoder configurations
ENCODER_CONFIGS = {
    'prov-gigapath': {
        'dim': 1536,
        'model_name': 'hf_hub:prov-gigapath/prov-gigapath',
        'architecture': 'vit',
        'patch_size': 14,
        'requires_cls_token': True,
    },
    'virchow2': {
        'dim': 1280,
        'model_name': 'paige-ai/Virchow2',
        'architecture': 'vit',
        'patch_size': 14,
        'requires_cls_token': True,
    },
    'uni2-h': {
        'dim': 1536,
        'model_name': 'MahmoodLab/UNI',
        'architecture': 'vit',
        'patch_size': 16,
        'requires_cls_token': True,
    },
    'h-optimus-1': {
        'dim': 1024,
        'model_name': 'bioptimus/H-optimus-1',
        'architecture': 'vit',
        'patch_size': 14,
        'requires_cls_token': True,
    },
    'conchv1.5': {
        'dim': 768,
        'model_name': 'MahmoodLab/CONCH',
        'architecture': 'vit',
        'patch_size': 16,
        'requires_cls_token': True,
    },
    # Fallback ResNet encoder for testing
    'resnet50': {
        'dim': 2048,
        'model_name': 'resnet50',
        'architecture': 'cnn',
        'requires_cls_token': False,
    },
}


class EncoderWrapper(nn.Module):
    """
    Wrapper that provides unified interface for different encoder architectures.

    Returns:
        If return_spatial=True: (B, D, H', W') - spatial feature map
        If return_spatial=False: (B, D) - global feature vector
    """

    def __init__(
        self,
        encoder_name: str = 'prov-gigapath',
        freeze: bool = True,
        return_spatial: bool = True,
        pretrained: bool = True,
    ):
        super().__init__()

        self.encoder_name = encoder_name
        self.config = ENCODER_CONFIGS.get(encoder_name)
        if self.config is None:
            raise ValueError(f"Unknown encoder: {encoder_name}. "
                           f"Available: {list(ENCODER_CONFIGS.keys())}")

        self.dim = self.config['dim']
        self.return_spatial = return_spatial
        self.freeze = freeze

        # Load encoder
        self.encoder = self._load_encoder(pretrained)

        if freeze:
            self._freeze_encoder()

    def _load_encoder(self, pretrained: bool) -> nn.Module:
        """Load the actual encoder model."""
        arch = self.config['architecture']

        if arch == 'cnn':
            return self._load_cnn_encoder(pretrained)
        elif arch == 'vit':
            return self._load_vit_encoder(pretrained)
        else:
            raise ValueError(f"Unknown architecture: {arch}")

    def _load_cnn_encoder(self, pretrained: bool) -> nn.Module:
        """Load CNN-based encoder (ResNet)."""
        import torchvision.models as models

        if self.encoder_name == 'resnet50':
            weights = 'IMAGENET1K_V2' if pretrained else None
            model = models.resnet50(weights=weights)
            # Remove classification head
            model = nn.Sequential(*list(model.children())[:-2])
            return model

        raise ValueError(f"Unknown CNN encoder: {self.encoder_name}")

    def _load_vit_encoder(self, pretrained: bool) -> nn.Module:
        """Load ViT-based encoder."""
        import timm

        model_name = self.config['model_name']

        try:
            if 'hf_hub:' in model_name:
                # HuggingFace hub model
                model = timm.create_model(
                    model_name,
                    pretrained=pretrained,
                    num_classes=0,  # Remove classification head
                )
            else:
                # Try loading from timm or local
                try:
                    model = timm.create_model(
                        model_name,
                        pretrained=pretrained,
                        num_classes=0,
                    )
                except Exception:
                    # Fallback to basic ViT
                    warnings.warn(f"Could not load {model_name}, using base vit_base_patch16_224")
                    model = timm.create_model(
                        'vit_base_patch16_224',
                        pretrained=True,
                        num_classes=0,
                    )
                    # Adjust dim to match expected
                    self.dim = 768

            return model

        except Exception as e:
            warnings.warn(f"Failed to load encoder {model_name}: {e}")
            # Fallback to basic ViT
            model = timm.create_model(
                'vit_base_patch16_224',
                pretrained=True,
                num_classes=0,
            )
            self.dim = 768
            return model

    def _freeze_encoder(self):
        """Freeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input images (B, 3, H, W), typically (B, 3, 224, 224)

        Returns:
            If return_spatial: (B, D, H', W') spatial features
            Else: (B, D) global features
        """
        if self.freeze:
            with torch.no_grad():
                features = self._extract_features(x)
        else:
            features = self._extract_features(x)

        return features

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from encoder."""
        arch = self.config['architecture']

        if arch == 'cnn':
            # CNN outputs (B, D, H', W')
            features = self.encoder(x)
            if not self.return_spatial:
                features = features.mean(dim=(2, 3))  # Global average pool
            return features

        elif arch == 'vit':
            # ViT forward - get patch embeddings
            if hasattr(self.encoder, 'forward_features'):
                features = self.encoder.forward_features(x)
            else:
                features = self.encoder(x)

            # Handle different output formats
            if isinstance(features, tuple):
                features = features[0]

            # features shape: (B, N, D) where N = num_patches + 1 (CLS)
            if len(features.shape) == 3:
                if self.return_spatial:
                    # Reshape to spatial: (B, N-1, D) -> (B, D, H', W')
                    B, N, D = features.shape
                    # Remove CLS token
                    if self.config['requires_cls_token'] and N > 1:
                        features = features[:, 1:, :]  # (B, N-1, D)
                        N = N - 1

                    # Calculate spatial dimensions
                    H = W = int(N ** 0.5)
                    if H * W == N:
                        features = features.reshape(B, H, W, D)
                        features = features.permute(0, 3, 1, 2)  # (B, D, H, W)
                    else:
                        # Non-square, use global pool
                        features = features.mean(dim=1)  # (B, D)
                else:
                    # Use CLS token or mean pool
                    if self.config['requires_cls_token']:
                        features = features[:, 0, :]  # CLS token
                    else:
                        features = features.mean(dim=1)  # Mean pool

            return features

        raise ValueError(f"Unknown architecture: {arch}")


def get_encoder(
    name: str = 'prov-gigapath',
    freeze: bool = True,
    return_spatial: bool = True,
    pretrained: bool = True,
) -> EncoderWrapper:
    """
    Get an encoder by name.

    Args:
        name: Encoder name (prov-gigapath, virchow2, uni2-h, h-optimus-1, conchv1.5)
        freeze: Whether to freeze encoder weights
        return_spatial: If True, return spatial features (B, D, H, W)
        pretrained: Whether to load pretrained weights

    Returns:
        EncoderWrapper instance
    """
    return EncoderWrapper(
        encoder_name=name,
        freeze=freeze,
        return_spatial=return_spatial,
        pretrained=pretrained,
    )


def get_encoder_dim(name: str) -> int:
    """Get the output dimension for an encoder."""
    config = ENCODER_CONFIGS.get(name)
    if config is None:
        raise ValueError(f"Unknown encoder: {name}")
    return config['dim']


def list_encoders() -> Dict[str, Dict]:
    """List all available encoders and their configs."""
    return ENCODER_CONFIGS.copy()


# Quick test
if __name__ == '__main__':
    print("Testing encoder registry...")

    # Test ResNet (always available)
    encoder = get_encoder('resnet50', freeze=True, return_spatial=True)
    x = torch.randn(2, 3, 224, 224)
    y = encoder(x)
    print(f"ResNet50 spatial output: {y.shape}")  # Should be (2, 2048, 7, 7)

    encoder = get_encoder('resnet50', freeze=True, return_spatial=False)
    y = encoder(x)
    print(f"ResNet50 global output: {y.shape}")  # Should be (2, 2048)

    print("\nAvailable encoders:")
    for name, cfg in list_encoders().items():
        print(f"  {name}: dim={cfg['dim']}, arch={cfg['architecture']}")
