"""Self-contained inference-time detector package for the local demo."""

from .model import MultiDomainDetector, load_detector
from .transforms import FrequencyTransform, NoiseTransform, RGBTransform

__all__ = [
    "FrequencyTransform",
    "MultiDomainDetector",
    "NoiseTransform",
    "RGBTransform",
    "load_detector",
]
