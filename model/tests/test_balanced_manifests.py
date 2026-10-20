import importlib.util
from pathlib import Path

import pandas as pd
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "generate_manifests.py"
)
SPEC = importlib.util.spec_from_file_location("generate_manifests", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_training_balance_is_exact_per_architecture():
    rows = [
        {"label": 0, "architecture": "natural", "id": index}
        for index in range(12)
    ]
    for architecture in ("progan", "stylegan"):
        rows.extend(
            {"label": 1, "architecture": architecture, "id": index}
            for index in range(8)
        )

    result = MODULE.balance_classes(
        pd.DataFrame(rows),
        samples_per_class=12,
        seed=42,
    )

    assert (result["label"] == 0).sum() == 12
    fake_counts = (
        result[result["label"] == 1]["architecture"].value_counts().to_dict()
    )
    assert fake_counts == {"progan": 6, "stylegan": 6}


def test_training_balance_rejects_architecture_shortage():
    data = pd.DataFrame([
        *(
            {"label": 0, "architecture": "natural", "id": index}
            for index in range(8)
        ),
        *(
            {"label": 1, "architecture": "progan", "id": index}
            for index in range(7)
        ),
        {"label": 1, "architecture": "stylegan", "id": 0},
    ])

    with pytest.raises(ValueError, match="stylegan"):
        MODULE.balance_classes(data, samples_per_class=8, seed=42)


def test_primary_test_is_balanced_by_dataset_and_class():
    rows = []
    for dataset in ("CIFAKE", "SynthBuster"):
        for label in (0, 1):
            rows.extend(
                {
                    "dataset": dataset,
                    "label": label,
                    "architecture": f"{dataset}_{label}",
                    "id": index,
                }
                for index in range(5)
            )

    result = MODULE.balance_test_by_dataset_class(
        pd.DataFrame(rows),
        samples_per_dataset_class=3,
        seed=42,
    )

    counts = result.groupby(["dataset", "label"]).size().to_dict()
    assert counts == {
        ("CIFAKE", 0): 3,
        ("CIFAKE", 1): 3,
        ("SynthBuster", 0): 3,
        ("SynthBuster", 1): 3,
    }


def test_cross_architecture_balance_is_equal_per_generator_and_class():
    rows = [
        {"label": 0, "architecture": "natural", "id": index}
        for index in range(20)
    ]
    for architecture, count in (("progan", 7), ("dalle2", 5)):
        rows.extend(
            {"label": 1, "architecture": architecture, "id": index}
            for index in range(count)
        )

    result = MODULE.balance_equal_per_generator(
        pd.DataFrame(rows),
        seed=42,
    )

    assert result["label"].value_counts().to_dict() == {0: 10, 1: 10}
    assert (
        result[result["label"] == 1]["architecture"].value_counts().to_dict()
        == {"progan": 5, "dalle2": 5}
    )
