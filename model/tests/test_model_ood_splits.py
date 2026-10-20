from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.data.splits import ModelAwareSplitter


TRAIN_MODELS = {
    "gan": ["progan", "biggan", "cyclegan", "stargan"],
    "diffusion": [
        "dalle2", "glide", "midjourney", "sd_13", "sd_14", "sd_2",
    ],
}
TEST_MODELS = {
    "gan": ["stylegan", "stylegan2", "gaugan", "whichfaceisreal"],
    "diffusion": ["dalle3", "firefly", "sd_xl"],
}


def make_metadata() -> pd.DataFrame:
    rows = []
    for dataset, real_architecture in (
        ("CNNDetection", "natural"),
        ("SynthBuster", "raise_real"),
    ):
        rows.extend(
            {
                "filepath": f"{dataset}/real_{index}.png",
                "dataset": dataset,
                "architecture": real_architecture,
                "label": 0,
            }
            for index in range(10)
        )
    for split_models in (TRAIN_MODELS, TEST_MODELS):
        for models in split_models.values():
            for architecture in models:
                dataset = (
                    "CNNDetection"
                    if architecture in TRAIN_MODELS["gan"] + TEST_MODELS["gan"]
                    else "SynthBuster"
                )
                rows.extend(
                    {
                        "filepath": f"{dataset}/{architecture}_{index}.png",
                        "dataset": dataset,
                        "architecture": architecture,
                        "label": 1,
                    }
                    for index in range(2)
                )
    return pd.DataFrame(rows)


def make_splitter() -> ModelAwareSplitter:
    return ModelAwareSplitter(
        train_models=TRAIN_MODELS,
        test_models=TEST_MODELS,
        exclude_families=["cnn", "implicit_generative", "face_manipulation"],
        real_strategy="dataset_stratified",
        real_test_fraction=0.2,
        seed=42,
    )


def test_cross_architecture_split_is_generator_disjoint():
    train_df, test_df = make_splitter().split_dataframe(make_metadata())

    train_fake = train_df[train_df["label"] == 1]
    test_fake = test_df[test_df["label"] == 1]
    assert set(train_fake["architecture"]) == {
        model for models in TRAIN_MODELS.values() for model in models
    }
    assert set(test_fake["architecture"]) == {
        model for models in TEST_MODELS.values() for model in models
    }
    assert set(train_fake["architecture_family"]) == {"gan", "diffusion"}
    assert set(test_fake["architecture_family"]) == {"gan", "diffusion"}
    assert set(train_fake["architecture"]).isdisjoint(test_fake["architecture"])


def test_real_images_are_split_80_20_within_each_dataset():
    train_df, test_df = make_splitter().split_dataframe(make_metadata())

    train_real = train_df[train_df["label"] == 0]
    test_real = test_df[test_df["label"] == 0]
    assert train_real.groupby("dataset").size().to_dict() == {
        "CNNDetection": 8,
        "SynthBuster": 8,
    }
    assert test_real.groupby("dataset").size().to_dict() == {
        "CNNDetection": 2,
        "SynthBuster": 2,
    }


def test_unassigned_fake_generator_is_rejected():
    data = pd.concat([
        make_metadata(),
        pd.DataFrame([{
            "filepath": "SynthBuster/wukong.png",
            "dataset": "SynthBuster",
            "architecture": "wukong",
            "label": 1,
        }]),
    ], ignore_index=True)

    with pytest.raises(ValueError, match="not assigned"):
        make_splitter().split_dataframe(data)


def test_missing_configured_generator_is_rejected():
    data = make_metadata()
    data = data[data["architecture"] != "sd_xl"]

    with pytest.raises(ValueError, match="absent from metadata"):
        make_splitter().split_dataframe(data)


def test_primary_experiment_configs_use_cross_architecture_split():
    config_dir = Path(__file__).resolve().parents[1] / "configs" / "experiments"
    config_names = [
        "baseline.yaml",
        "concatenate.yaml",
        "cross_attention.yaml",
        "gated_fusion.yaml",
        "no_dil.yaml",
        "no_norm_a.yaml",
        "no_norm_b.yaml",
    ]
    for config_name in config_names:
        with (config_dir / config_name).open() as config_file:
            data = yaml.safe_load(config_file)["data"]

        assert [dataset["name"] for dataset in data["datasets"]] == [
            "cnndetection",
            "synthbuster",
        ]
        assert data["split"]["level"] == "model"
        assert data["split"]["real_strategy"] == "dataset_stratified"
        assert data["split"]["real_test_fraction"] == 0.53385
        assert data["train_models"] == TRAIN_MODELS
        assert data["test_models"] == TEST_MODELS
        assert data["train_manifest"].endswith(
            "cross_architecture_train_manifest.csv"
        )
        assert data["test_manifest"].endswith(
            "cross_architecture_test_manifest.csv"
        )


def test_fixed_manifests_have_expected_protocol_counts():
    data_dir = Path(__file__).resolve().parents[1] / "src" / "data"
    train = pd.read_csv(data_dir / "cross_architecture_train_manifest.csv")
    test = pd.read_csv(data_dir / "cross_architecture_test_manifest.csv")
    full_test = pd.read_csv(
        data_dir / "cross_architecture_full_test_manifest.csv"
    )

    assert len(train) == 20_000
    assert train["label"].value_counts().to_dict() == {0: 10_000, 1: 10_000}
    assert len(test) == 14_000
    assert test["label"].value_counts().to_dict() == {0: 7_000, 1: 7_000}
    assert len(full_test) == 50_000
    assert full_test["label"].value_counts().to_dict() == {
        0: 27_317,
        1: 22_683,
    }
