import torch

from src.training.metrics import MetricsCalculator


def test_dataset_metrics_are_computed_from_dataset_predictions():
    calculator = MetricsCalculator(threshold=0.5)
    labels = torch.tensor([0, 1, 0, 1])
    probabilities = torch.tensor([
        [0.9, 0.1],
        [0.1, 0.9],
        [0.2, 0.8],
        [0.8, 0.2],
    ])

    calculator.update(
        labels,
        probabilities,
        architectures=["real", "fake", "real", "fake"],
        datasets=["CIFAKE", "CIFAKE", "SynthBuster", "SynthBuster"],
    )
    results = calculator.compute_per_dataset()

    assert results["CIFAKE"]["accuracy"] == 1.0
    assert results["SynthBuster"]["accuracy"] == 0.0
    assert results["CIFAKE"]["count"] == 2
    assert results["SynthBuster"]["count"] == 2
