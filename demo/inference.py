"""Reusable, offline inference service for the local Gradio demo."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Optional

import numpy as np
import torch
from PIL import Image, ImageOps

from detector.model import load_detector
from detector.threshold import resolve_decision_threshold
from detector.transforms import FrequencyTransform, NoiseTransform, RGBTransform


MAX_IMAGE_PIXELS = 50_000_000


@dataclass(frozen=True)
class PredictionResult:
    """One image prediction and the domain views shown in the demo."""

    predicted_label: int
    prediction: str
    confidence: float
    probability_real: float
    probability_ai: float
    threshold: float
    elapsed_ms: float
    frequency_image: Image.Image
    noise_image: Image.Image


def choose_device(requested: str = "auto") -> str:
    """Resolve auto/CUDA selection with a safe CPU fallback."""
    normalized = requested.lower().strip()
    if normalized == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if normalized == "cuda" and not torch.cuda.is_available():
        return "cpu"
    if normalized not in {"cpu", "cuda"}:
        raise ValueError("Device must be 'auto', 'cuda', or 'cpu'.")
    return normalized


def normalize_image(image: Image.Image) -> Image.Image:
    """Apply phone-camera orientation and normalize input to RGB."""
    if image is None:
        raise ValueError("Please upload or select an image first.")
    if not isinstance(image, Image.Image):
        raise TypeError("The uploaded value is not a valid image.")
    if image.width <= 0 or image.height <= 0:
        raise ValueError("The image has invalid dimensions.")
    if image.width * image.height > MAX_IMAGE_PIXELS:
        raise ValueError("The image is too large. Use an image under 50 megapixels.")
    return ImageOps.exif_transpose(image).convert("RGB")


def tensors_to_views(
    frequency: torch.Tensor,
    noise: torch.Tensor,
) -> tuple[Image.Image, Image.Image]:
    """Convert the exact model inputs into displayable domain views."""
    frequency_array = frequency.squeeze(0).detach().cpu().numpy()
    frequency_array = np.clip(frequency_array, 0.0, 1.0)
    frequency_image = Image.fromarray(
        np.rint(frequency_array * 255.0).astype(np.uint8)
    )

    noise_array = noise.detach().cpu().numpy()
    if noise_array.shape[0] != 3:
        raise ValueError("Expected three SRM residual channels.")
    noise_array = np.transpose(noise_array, (1, 2, 0))
    noise_array = np.clip((noise_array + 1.0) * 0.5, 0.0, 1.0)
    noise_image = Image.fromarray(
        np.rint(noise_array * 255.0).astype(np.uint8)
    )
    return frequency_image, noise_image


class DemoPredictor:
    """Load the detector once and serve repeated PIL-image predictions."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "auto",
        image_size: int = 256,
        threshold: Optional[float] = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        self.device = choose_device(device)
        self.image_size = int(image_size)
        if self.image_size <= 0:
            raise ValueError("Image size must be positive.")

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        self.threshold = resolve_decision_threshold(checkpoint, threshold)
        del checkpoint

        self.model = load_detector(str(self.checkpoint_path), device=self.device)
        self.model.eval()
        self.rgb_transform = RGBTransform(size=self.image_size)
        self.frequency_transform = FrequencyTransform(size=self.image_size)
        self.noise_transform = NoiseTransform(size=self.image_size)

    def predict(self, image: Image.Image) -> PredictionResult:
        """Classify one PIL image and return the model's domain views."""
        normalized = normalize_image(image)
        started = perf_counter()

        rgb = self.rgb_transform(normalized)
        frequency = self.frequency_transform(normalized)
        noise = self.noise_transform(normalized)
        frequency_image, noise_image = tensors_to_views(frequency, noise)

        with torch.inference_mode():
            output = self.model(
                rgb.unsqueeze(0).to(self.device),
                frequency.unsqueeze(0).to(self.device),
                noise.unsqueeze(0).to(self.device),
            )
            probabilities = torch.softmax(output["logits"], dim=1)[0].cpu()

        probability_real = float(probabilities[0].item())
        probability_ai = float(probabilities[1].item())
        predicted_label = int(probability_ai >= self.threshold)
        prediction = "Likely AI-generated" if predicted_label else "Likely real"
        confidence = probability_ai if predicted_label else probability_real

        return PredictionResult(
            predicted_label=predicted_label,
            prediction=prediction,
            confidence=confidence,
            probability_real=probability_real,
            probability_ai=probability_ai,
            threshold=self.threshold,
            elapsed_ms=(perf_counter() - started) * 1000.0,
            frequency_image=frequency_image,
            noise_image=noise_image,
        )
