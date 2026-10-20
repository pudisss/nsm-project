"""Data loading and processing for AI image detection."""

from .dataset import MultiDomainDataset, MultiDomainDataModule, collate_fn
from .transforms import (
    RGBTransform,
    FrequencyTransform,
    NoiseTransform,
    MultiDomainTransform,
    AugmentationTransform,
)
from .splits import (
    FamilyAwareSplitter,
    ModelAwareSplitter,
    BalancedSampler,
    load_and_combine_metadata,
    ARCHITECTURE_TYPES,
    add_architecture_family,
)

__all__ = [
    'MultiDomainDataset',
    'MultiDomainDataModule',
    'collate_fn',
    'RGBTransform',
    'FrequencyTransform',
    'NoiseTransform',
    'MultiDomainTransform',
    'AugmentationTransform',
    'ModelAwareSplitter',
    'FamilyAwareSplitter',
    'BalancedSampler',
    'load_and_combine_metadata',
    'ARCHITECTURE_TYPES',
    'add_architecture_family',
]
