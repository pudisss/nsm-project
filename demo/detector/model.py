"""Full multi-domain detector model.

This module assembles the complete detector by combining:
- Three domain-specific branches (RGB, Frequency, Noise)
- Fusion module for combining features
- Classification head for final prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List

from .branches import RGBBranch, FrequencyBranch, NoiseBranch
from .fusion import create_fusion_module


class ClassificationHead(nn.Module):
    """Classification head with multiple FC layers."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128],
        num_classes: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        
        layers = []
        in_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class FeatureNormBlock(nn.Module):
    """Optional feature-level normalization for branch features.

    Args:
        feature_dims: Per-domain feature dimensions.
        enabled: If ``False``, each domain uses ``nn.Identity``.

    Inputs:
        features: Mapping of domain name to tensor shaped ``[B, D_domain]``.

    Outputs:
        Mapping with the same keys and tensor shapes as the input.
    """

    def __init__(self, feature_dims: Dict[str, int], enabled: bool = True):
        super().__init__()
        self.enabled = enabled
        self.norms = nn.ModuleDict({
            domain: nn.LayerNorm(dim) if enabled else nn.Identity()
            for domain, dim in feature_dims.items()
        })

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Normalize each domain feature without changing shape."""
        return {
            domain: self.norms[domain](feature)
            for domain, feature in features.items()
        }


class DomainMaskBlock(nn.Module):
    """Zero selected domain features while preserving tensor shapes."""

    expected_domains = {'rgb', 'freq', 'noise'}

    def __init__(self, active_domains: Dict[str, bool]):
        super().__init__()
        if set(active_domains) != self.expected_domains:
            raise ValueError(
                "active_domains must contain rgb, freq, and noise flags."
            )
        self.active_domains = {
            domain: bool(enabled)
            for domain, enabled in active_domains.items()
        }

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Replace disabled domain features with zeros of the same shape."""
        return {
            domain: (
                feature if self.active_domains[domain]
                else torch.zeros_like(feature)
            )
            for domain, feature in features.items()
        }


class L2NormBlock(nn.Module):
    """Optional per-domain L2 normalization for projected DIL features.

    Args:
        enabled: If ``True``, each feature vector is scaled to unit L2 norm.

    Inputs:
        features: Mapping of domain name to tensor shaped ``[B, D_shared]``.

    Outputs:
        Mapping with the same keys and tensor shapes as the input.
    """

    def __init__(self, enabled: bool = True):
        super().__init__()
        self.enabled = enabled

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """L2-normalize each projected domain feature without changing shape."""
        if not self.enabled:
            return features
        return {
            domain: F.normalize(feature, p=2, dim=1)
            for domain, feature in features.items()
        }


class DomainAdapter(nn.Module):
    """Optional residual adapter for architecture-level OOD handling.

    The adapter is metadata-free and preserves feature dimensionality:
    ``[B, D] -> [B, D]``.
    """

    def __init__(self, feature_dim: int, bottleneck_dim: int, enabled: bool = False):
        super().__init__()
        self.enabled = enabled
        if enabled:
            self.adapter = nn.Sequential(
                nn.Linear(feature_dim, bottleneck_dim),
                nn.GELU(),
                nn.Linear(bottleneck_dim, feature_dim),
            )
        else:
            self.adapter = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a residual adapter when enabled, otherwise identity."""
        assert x.dim() == 2, "DomainAdapter expects [B, D]."
        if not self.enabled:
            return self.adapter(x)
        return x + self.adapter(x)


class DomainInconsistencyModule(nn.Module):
    """Project domain features and compute pairwise DIL similarities.

    When enabled, each input domain feature receives the same three
    inconsistency features: RGB-frequency, RGB-noise, and frequency-noise.
    L2 normalization is intentionally handled by ``L2NormBlock`` outside this
    module so Norm B can be independently ablated.

    Inputs:
        features: Mapping with ``rgb``, ``freq``, and ``noise`` tensors
            shaped ``[B, D_rgb]``, ``[B, D_freq]``, and ``[B, D_noise]``.
        comparison_features: Projected features with a shared dimension.
            With Norm B enabled, these are unit-length vectors and their dot
            products are cosine similarities. With Norm B disabled, the same
            operation produces unnormalized dot-product similarities.

    Outputs:
        Tuple of ``(fusion_features, similarity_features)``. If enabled,
        fusion features are shaped ``[B, D_domain + 3]`` and similarity
        features are shaped ``[B, 3]``. If disabled, fusion features keep
        their original shapes and similarity features is ``None``.
    """

    similarity_dim = 3

    valid_similarity_modes = {'real', 'zero'}

    def __init__(
        self,
        feature_dims: Dict[str, int],
        enabled: bool = True,
        similarity_mode: str = 'real',
    ):
        super().__init__()
        self.enabled = enabled
        if similarity_mode not in self.valid_similarity_modes:
            raise ValueError(
                "similarity_mode must be one of "
                f"{sorted(self.valid_similarity_modes)}, got {similarity_mode!r}."
            )
        self.similarity_mode = similarity_mode
        self.projection_dim = min(feature_dims.values())
        if enabled:
            self.projections = nn.ModuleDict({
                domain: nn.Linear(dim, self.projection_dim)
                for domain, dim in feature_dims.items()
            })
        else:
            self.projections = nn.ModuleDict()

    @property
    def output_extra_dim(self) -> int:
        """Number of DIL features appended to each domain."""
        return self.similarity_dim if self.enabled else 0

    def project(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Project all domains into the shared DIL comparison space."""
        if not self.enabled:
            return {}
        return {
            domain: self.projections[domain](features[domain])
            for domain in ('rgb', 'freq', 'noise')
        }

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        comparison_features: Optional[Dict[str, torch.Tensor]] = None,
        active_domains: Optional[Dict[str, bool]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        """Append pairwise DIL similarity features when enabled."""
        if not self.enabled:
            return features, None

        if comparison_features is None:
            raise ValueError(
                "DIL comparison features are required. Call dil.project(), "
                "then apply Norm B before calling dil.forward()."
            )

        expected_domains = {'rgb', 'freq', 'noise'}
        if set(comparison_features) != expected_domains:
            raise ValueError(
                "DIL comparison features must contain rgb, freq, and noise."
            )
        comparison_dims = {
            feature.shape[1] for feature in comparison_features.values()
        }
        if len(comparison_dims) != 1:
            raise ValueError(
                "DIL comparison features must share the same dimension."
            )

        rgb_comparison = comparison_features['rgb']
        freq_comparison = comparison_features['freq']
        noise_comparison = comparison_features['noise']

        sim_rf = (rgb_comparison * freq_comparison).sum(dim=1, keepdim=True)
        sim_rn = (rgb_comparison * noise_comparison).sum(dim=1, keepdim=True)
        sim_fn = (freq_comparison * noise_comparison).sum(dim=1, keepdim=True)
        sim_features = torch.cat([sim_rf, sim_rn, sim_fn], dim=1)
        assert sim_features.shape[1] == self.similarity_dim
        if active_domains is not None:
            pair_is_active = [
                active_domains['rgb'] and active_domains['freq'],
                active_domains['rgb'] and active_domains['noise'],
                active_domains['freq'] and active_domains['noise'],
            ]
            sim_mask = sim_features.new_tensor(pair_is_active).view(1, -1)
            sim_features = sim_features * sim_mask
        if self.similarity_mode == 'zero':
            sim_features = torch.zeros_like(sim_features)

        fusion_features = {
            domain: torch.cat([feature, sim_features], dim=1)
            for domain, feature in features.items()
        }
        return fusion_features, sim_features


class MultiDomainDetector(nn.Module):
    """Multi-domain AI image detector.
    
    Combines RGB, frequency, and noise domain features to detect
    AI-generated images across different generative architectures.
    """
    
    def __init__(
        self,
        # RGB branch config (Swin Transformer)
        rgb_model: str = "microsoft/swin-tiny-patch4-window7-224",
        rgb_pretrained: bool = True,
        rgb_feature_dim: int = 512,
        # Frequency branch config (ResNet)
        freq_feature_dim: int = 256,
        freq_resnet: str = "resnet18",
        freq_pretrained: bool = True,
        # Noise branch config (ResNet)
        noise_feature_dim: int = 256,
        noise_resnet: str = "resnet18",
        noise_pretrained: bool = True,
        # Fusion config
        fusion_type: str = "attention",
        fusion_hidden_dim: int = 512,
        fusion_num_heads: int = 8,
        fusion_dropout: float = 0.3,
        use_dil: bool = True,
        dil_similarity_mode: str = 'real',
        use_rgb: bool = True,
        use_freq: bool = True,
        use_noise: bool = True,
        use_norm_a: bool = True,
        use_norm_b: bool = True,
        use_ood_adapters: bool = False,
        ood_adapter_bottleneck_dim: int = 128,
        # Classifier config
        classifier_hidden_dims: List[int] = [256, 128],
        classifier_dropout: float = 0.5,
        num_classes: int = 2,
    ):
        super().__init__()
        
        # Create branches
        self.rgb_branch = RGBBranch(
            model_name=rgb_model,
            pretrained=rgb_pretrained,
            feature_dim=rgb_feature_dim,
        )
        
        self.freq_branch = FrequencyBranch(
            input_channels=1,
            feature_dim=freq_feature_dim,
            resnet_type=freq_resnet,
            pretrained=freq_pretrained,
        )
        
        self.noise_branch = NoiseBranch(
            input_channels=3,
            feature_dim=noise_feature_dim,
            resnet_type=noise_resnet,
            pretrained=noise_pretrained,
            use_srm_layer=False,  # Pre-computed residuals from dataset
        )

        self.domain_order = ['rgb', 'freq', 'noise']
        self.branch_feature_dims = {
            'rgb': rgb_feature_dim,
            'freq': freq_feature_dim,
            'noise': noise_feature_dim,
        }
        self.active_domains = {
            'rgb': bool(use_rgb),
            'freq': bool(use_freq),
            'noise': bool(use_noise),
        }

        adapter_bottleneck_dims = {
            domain: max(1, min(dim, ood_adapter_bottleneck_dim))
            for domain, dim in self.branch_feature_dims.items()
        }
        self.ood_adapters = nn.ModuleDict({
            domain: DomainAdapter(
                feature_dim=dim,
                bottleneck_dim=adapter_bottleneck_dims[domain],
                enabled=use_ood_adapters,
            )
            for domain, dim in self.branch_feature_dims.items()
        })

        self.feature_norm = FeatureNormBlock(
            self.branch_feature_dims,
            enabled=use_norm_a,
        )
        self.domain_mask = DomainMaskBlock(self.active_domains)
        self.dil = DomainInconsistencyModule(
            self.branch_feature_dims,
            enabled=use_dil,
            similarity_mode=dil_similarity_mode,
        )
        self.l2_norm = L2NormBlock(enabled=use_norm_b)
        
        # Create fusion module
        branch_dims = [
            self.branch_feature_dims[domain] + self.dil.output_extra_dim
            for domain in self.domain_order
        ]
        self.fusion = create_fusion_module(
            fusion_type=fusion_type,
            input_dims=branch_dims,
            hidden_dim=fusion_hidden_dim,
            num_heads=fusion_num_heads,
            dropout=fusion_dropout,
        )
        # Create classifier
        self.classifier = ClassificationHead(
            input_dim=self.fusion.output_dim,
            hidden_dims=classifier_hidden_dims,
            num_classes=num_classes,
            dropout=classifier_dropout,
        )
        
        # Store config for later
        self.config = {
            'rgb_model': rgb_model,
            'rgb_pretrained': rgb_pretrained,
            'rgb_feature_dim': rgb_feature_dim,
            'freq_feature_dim': freq_feature_dim,
            'freq_resnet': freq_resnet,
            'freq_pretrained': freq_pretrained,
            'noise_feature_dim': noise_feature_dim,
            'noise_resnet': noise_resnet,
            'noise_pretrained': noise_pretrained,
            'fusion_type': fusion_type,
            'fusion_hidden_dim': fusion_hidden_dim,
            'fusion_num_heads': fusion_num_heads,
            'fusion_dropout': fusion_dropout,
            'use_dil': use_dil,
            'dil_similarity_mode': dil_similarity_mode,
            'use_rgb': use_rgb,
            'use_freq': use_freq,
            'use_noise': use_noise,
            'use_norm_a': use_norm_a,
            'use_norm_b': use_norm_b,
            'normalization_schema': 'explicit_dil_l2',
            'use_ood_adapters': use_ood_adapters,
            'ood_adapter_bottleneck_dim': ood_adapter_bottleneck_dim,
            'classifier_hidden_dims': classifier_hidden_dims,
            'classifier_dropout': classifier_dropout,
            'num_classes': num_classes,
        }
    
    def forward(
        self,
        rgb: torch.Tensor,
        freq: torch.Tensor,
        noise: torch.Tensor,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            rgb: RGB tensor [B, 3, H, W]
            freq: FFT spectrum tensor [B, 1, H, W]
            noise: Noise residual tensor [B, 3, H, W]
            return_features: If True, return intermediate features
        
        Returns:
            Dict with 'logits' and optionally 'features'
        """
        # Extract branch features: rgb [B, 512], freq [B, 256], noise [B, 256]
        features = {
            'rgb': self.rgb_branch(rgb),
            'freq': self.freq_branch(freq),
            'noise': self.noise_branch(noise),
        }
        for domain, feature in features.items():
            expected_dim = self.branch_feature_dims[domain]
            assert feature.dim() == 2 and feature.shape[1] == expected_dim, (
                f"{domain} branch expected [B, {expected_dim}], "
                f"got {tuple(feature.shape)}."
            )

        # Optional metadata-free OOD adapters preserve [B, D_domain].
        features = {
            domain: self.ood_adapters[domain](feature)
            for domain, feature in features.items()
        }

        # Disabled domains keep their feature slots but carry no information.
        features = self.domain_mask(features)

        # Optional feature norm preserves [B, D_domain].
        features = self.feature_norm(features)

        # DIL projection maps all domains to the same comparison dimension.
        comparison_features = self.dil.project(features)

        # Norm B optionally scales projected DIL vectors to unit L2 norm.
        comparison_features = self.l2_norm(comparison_features)

        # DIL appends cosine features when Norm B is enabled, or raw
        # dot-product features when Norm B is disabled.
        fusion_features, sim_features = self.dil(
            features,
            comparison_features,
            active_domains=self.active_domains,
        )
        
        # Fuse ordered domain features into [B, fusion.output_dim].
        fused = self.fusion([fusion_features[domain] for domain in self.domain_order])
        
        # Classify
        logits = self.classifier(fused)
        
        output = {'logits': logits}
        
        if return_features:
            output['rgb_features'] = features['rgb']
            output['freq_features'] = features['freq']
            output['noise_features'] = features['noise']
            if sim_features is not None:
                output['similarity_features'] = sim_features
            output['fused_features'] = fused
        
        return output
    
    def predict(
        self,
        rgb: torch.Tensor,
        freq: torch.Tensor,
        noise: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get predictions and probabilities.
        
        Returns:
            Tuple of (predictions, probabilities)
        """
        output = self.forward(rgb, freq, noise)
        logits = output['logits']
        
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        return preds, probs
    
    def get_branch_features(
        self,
        rgb: torch.Tensor,
        freq: torch.Tensor,
        noise: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Get features from each branch for analysis."""
        return {
            'rgb': self.rgb_branch(rgb),
            'freq': self.freq_branch(freq),
            'noise': self.noise_branch(noise),
        }


class SingleBranchDetector(nn.Module):
    """Single-branch detector for ablation studies.
    
    Uses only one domain for detection to measure individual
    branch contribution.
    """
    
    def __init__(
        self,
        branch_type: str = "rgb",
        # Branch configs
        rgb_model: str = "microsoft/swin-tiny-patch4-window7-224",
        rgb_pretrained: bool = True,
        feature_dim: int = 512,
        resnet_type: str = "resnet18",
        # Classifier config
        classifier_hidden_dims: List[int] = [256, 128],
        classifier_dropout: float = 0.5,
        num_classes: int = 2,
    ):
        super().__init__()
        
        self.branch_type = branch_type
        
        if branch_type == "rgb":
            self.branch = RGBBranch(
                model_name=rgb_model,
                pretrained=rgb_pretrained,
                feature_dim=feature_dim,
            )
        elif branch_type == "freq":
            self.branch = FrequencyBranch(
                input_channels=1,
                feature_dim=feature_dim,
                resnet_type=resnet_type,
                pretrained=True,
            )
        elif branch_type == "noise":
            self.branch = NoiseBranch(
                input_channels=3,
                feature_dim=feature_dim,
                resnet_type=resnet_type,
                pretrained=True,
                use_srm_layer=False,
            )
        else:
            raise ValueError(f"Unknown branch type: {branch_type}")
        
        self.classifier = ClassificationHead(
            input_dim=self.branch.feature_dim,
            hidden_dims=classifier_hidden_dims,
            num_classes=num_classes,
            dropout=classifier_dropout,
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Input tensor appropriate for branch type
        
        Returns:
            Dict with 'logits'
        """
        features = self.branch(x)
        logits = self.classifier(features)
        return {'logits': logits}


def create_detector(
    model_config: Optional[Dict] = None,
    **kwargs,
) -> MultiDomainDetector:
    """Factory function to create detector from config.
    
    Args:
        model_config: Model configuration dict
        **kwargs: Override specific parameters
    
    Returns:
        MultiDomainDetector instance
    """
    # Default config
    config = {
        'rgb_model': 'microsoft/swin-tiny-patch4-window7-224',
        'rgb_pretrained': True,
        'rgb_feature_dim': 512,
        'freq_feature_dim': 256,
        'freq_resnet': 'resnet18',
        'freq_pretrained': True,
        'noise_feature_dim': 256,
        'noise_resnet': 'resnet18',
        'noise_pretrained': True,
        'fusion_type': 'attention',
        'fusion_hidden_dim': 512,
        'fusion_num_heads': 8,
        'fusion_dropout': 0.3,
        'use_dil': True,
        'dil_similarity_mode': 'real',
        'use_rgb': True,
        'use_freq': True,
        'use_noise': True,
        'use_norm_a': True,
        'use_norm_b': True,
        'use_ood_adapters': False,
        'ood_adapter_bottleneck_dim': 128,
        'classifier_hidden_dims': [256, 128],
        'classifier_dropout': 0.5,
        'num_classes': 2,
    }
    
    # Update from model_config if provided
    if model_config is not None:
        config.update(model_config)
    
    # Override with kwargs
    config.update(kwargs)
    
    return MultiDomainDetector(**config)


def load_detector(
    checkpoint_path: str,
    device: str = "cuda",
    **override_config,
) -> MultiDomainDetector:
    """Load detector from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        **override_config: Override config parameters
    
    Returns:
        Loaded MultiDomainDetector
    """
    # Checkpoint tensors stay on CPU while the architecture is reconstructed.
    # This avoids holding both checkpoint and model copies in GPU memory.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # Get config from checkpoint
    config = dict(checkpoint.get('config', {}))
    normalization_schema = config.pop('normalization_schema', None)
    if normalization_schema is None:
        # Legacy modular checkpoints always applied L2 normalization inside
        # DIL, while use_norm_b referred to the removed post-fusion LayerNorm.
        config['use_norm_b'] = True
    config.update(override_config)

    # The checkpoint contains every learned backbone parameter. Reconstructing
    # with random initialization prevents network downloads during a live demo;
    # strict state-dict loading below replaces all initialized parameters.
    config['rgb_pretrained'] = False
    config['freq_pretrained'] = False
    config['noise_pretrained'] = False

    state_dict = checkpoint['model_state_dict']
    if any(key.startswith('fusion_norm.') for key in state_dict):
        raise ValueError(
            "This checkpoint used the removed post-fusion Norm B behavior. "
            "It cannot be loaded as the explicit DIL L2 Norm B architecture."
        )
    
    # Create model
    model = create_detector(**config)
    
    # Load weights
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model
