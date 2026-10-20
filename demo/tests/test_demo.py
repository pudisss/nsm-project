"""Focused tests for the standalone demo without loading the large model."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from detector.threshold import resolve_decision_threshold
from inference import (
    MAX_IMAGE_PIXELS,
    PredictionResult,
    choose_device,
    normalize_image,
    tensors_to_views,
)
from app import (
    MODEL_SPECS,
    SHOWCASE_PATH,
    analyze_image,
    benchmark_markdown,
    load_json,
    resolve_model_selection,
)


def test_threshold_policy_matches_training_behavior() -> None:
    assert resolve_decision_threshold({"decision_threshold": 0.4}) == 0.4
    assert resolve_decision_threshold(
        {"threshold_tuned": True, "best_threshold": 0.63, "decision_threshold": 0.4}
    ) == 0.63
    assert resolve_decision_threshold({}, override=0.7) == 0.7
    with pytest.raises(ValueError):
        resolve_decision_threshold({"decision_threshold": 1.2})


def test_normalize_image_converts_to_rgb() -> None:
    image = Image.new("L", (16, 12), 128)
    normalized = normalize_image(image)
    assert normalized.mode == "RGB"
    assert normalized.size == (16, 12)


def test_normalize_image_rejects_missing_or_oversized_input(monkeypatch) -> None:
    with pytest.raises(ValueError, match="upload"):
        normalize_image(None)  # type: ignore[arg-type]
    monkeypatch.setattr("inference.MAX_IMAGE_PIXELS", 10)
    with pytest.raises(ValueError, match="too large"):
        normalize_image(Image.new("RGB", (4, 4)))
    assert MAX_IMAGE_PIXELS == 50_000_000


def test_domain_views_have_expected_modes_sizes_and_ranges() -> None:
    frequency = torch.linspace(0, 1, 64).reshape(1, 8, 8)
    noise = torch.linspace(-1, 1, 192).reshape(3, 8, 8)
    frequency_image, noise_image = tensors_to_views(frequency, noise)
    assert frequency_image.mode == "L"
    assert noise_image.mode == "RGB"
    assert frequency_image.size == noise_image.size == (8, 8)
    assert np.asarray(frequency_image).min() == 0
    assert np.asarray(frequency_image).max() == 255
    assert np.asarray(noise_image).min() == 0
    assert np.asarray(noise_image).max() == 255


def test_choose_device_validation() -> None:
    assert choose_device("cpu") == "cpu"
    assert choose_device("auto") in {"cpu", "cuda"}
    with pytest.raises(ValueError):
        choose_device("mps")


def test_prediction_result_contract() -> None:
    result = PredictionResult(
        predicted_label=1,
        prediction="Likely AI-generated",
        confidence=0.8,
        probability_real=0.2,
        probability_ai=0.8,
        threshold=0.5,
        elapsed_ms=12.0,
        frequency_image=Image.new("L", (8, 8)),
        noise_image=Image.new("RGB", (8, 8)),
    )
    assert result.probability_real + result.probability_ai == pytest.approx(1.0)


def test_ui_handler_maps_prediction_to_all_outputs() -> None:
    expected = PredictionResult(
        predicted_label=0,
        prediction="Likely real",
        confidence=0.75,
        probability_real=0.75,
        probability_ai=0.25,
        threshold=0.5,
        elapsed_ms=20.0,
        frequency_image=Image.new("L", (8, 8)),
        noise_image=Image.new("RGB", (8, 8)),
    )

    class FakePredictor:
        device = "cpu"

        def predict(self, image: Image.Image) -> PredictionResult:
            return expected

    result_html, probabilities, original, frequency, noise = analyze_image(
        Image.new("RGB", (8, 8)), FakePredictor()
    )
    assert "Likely real" in result_html
    assert probabilities == {"Real": 0.75, "AI-generated": 0.25}
    assert "data:image/png;base64," in original
    assert "Frequency spectrum" in frequency
    assert "Noise residuals" in noise


def test_registered_and_custom_model_selection() -> None:
    concatenation = resolve_model_selection("concatenation", None)
    assert concatenation == MODEL_SPECS["concatenation"]
    assert concatenation.checkpoint.name == "concatenation_best.pt"
    assert concatenation.evaluation is not None

    custom_path = Path("checkpoints/custom.pt")
    custom = resolve_model_selection("gated", custom_path)
    assert custom.checkpoint == custom_path
    assert custom.evaluation is None
    assert "Custom checkpoint" in custom.display_name
    assert "No benchmark metadata" in benchmark_markdown(
        custom.display_name, custom.evaluation
    )


def test_curated_showcase_is_balanced_and_complete() -> None:
    metadata = load_json(SHOWCASE_PATH)
    items = metadata["items"]
    assert Counter(item["known_label"] for item in items) == {
        "real": 10,
        "AI-generated": 10,
    }

    for item in items:
        assert (ROOT / item["file"]).is_file()
        scores = item["ai_probability"]
        assert set(scores) == {"gated", "concatenation"}
        if item["known_label"] == "real":
            assert all(score < 0.5 for score in scores.values())
        else:
            assert all(score >= 0.5 for score in scores.values())
