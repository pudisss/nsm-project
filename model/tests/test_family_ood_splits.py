import unittest
from pathlib import Path

import pandas as pd
import yaml

from src.data.dataset import MultiDomainDataModule
from src.data.splits import (
    FamilyAwareSplitter,
    add_architecture_family,
)


class FamilyOODSplitTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame([
            {
                "filepath": "cnndetection/real.png",
                "dataset": "CNNDetection",
                "architecture": "natural",
                "label": 0,
            },
            {
                "filepath": "cnndetection/fake.png",
                "dataset": "CNNDetection",
                "architecture": "progan",
                "label": 1,
            },
            {
                "filepath": "cifake/real.png",
                "dataset": "CIFAKE",
                "architecture": "cifar_real",
                "label": 0,
            },
            {
                "filepath": "cifake/fake.png",
                "dataset": "CIFAKE",
                "architecture": "stable_diffusion_v14",
                "label": 1,
            },
        ])

    def test_gan_to_diffusion_family_matched_split(self):
        splitter = FamilyAwareSplitter(
            train_families=["gan"],
            test_families=["diffusion"],
            real_strategy="family_matched",
        )

        train_df, test_df = splitter.split_dataframe(self.df)

        self.assertEqual(set(train_df["dataset"]), {"CNNDetection"})
        self.assertEqual(set(test_df["dataset"]), {"CIFAKE"})
        self.assertEqual(
            set(train_df.loc[train_df["label"] == 1, "architecture_family"]),
            {"gan"},
        )
        self.assertEqual(
            set(test_df.loc[test_df["label"] == 1, "architecture_family"]),
            {"diffusion"},
        )

    def test_reverse_diffusion_to_gan_split(self):
        splitter = FamilyAwareSplitter(
            train_families=["diffusion"],
            test_families=["gan"],
            real_strategy="family_matched",
        )

        train_df, test_df = splitter.split_dataframe(self.df)

        self.assertEqual(set(train_df["dataset"]), {"CIFAKE"})
        self.assertEqual(set(test_df["dataset"]), {"CNNDetection"})

    def test_reverse_config_uses_diffusion_train_and_gan_test_manifests(self):
        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "experiments"
            / "diffusion_to_gan.yaml"
        )
        with config_path.open() as config_file:
            config = yaml.safe_load(config_file)

        data_config = config["data"]
        split_config = data_config["split"]
        self.assertEqual(split_config["train_families"], ["diffusion"])
        self.assertEqual(split_config["test_families"], ["gan"])
        self.assertIn("diffusion_to_gan", data_config["train_manifest"])
        self.assertIn("diffusion_to_gan", data_config["test_manifest"])
        self.assertIn("diffusion_to_gan", data_config["full_test_manifest"])
        self.assertEqual(
            config["training"]["checkpoint_dir"],
            "./checkpoints/diffusion_to_gan_balanced",
        )

    def test_primary_ablation_configs_use_cross_architecture_manifests(self):
        config_dir = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "experiments"
        )
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
            with self.subTest(config=config_name):
                with (config_dir / config_name).open() as config_file:
                    data_config = yaml.safe_load(config_file)["data"]

                self.assertEqual(
                    data_config["train_manifest"],
                    "src/data/cross_architecture_train_manifest.csv",
                )
                self.assertEqual(
                    data_config["test_manifest"],
                    "src/data/cross_architecture_test_manifest.csv",
                )
                self.assertEqual(
                    data_config["full_test_manifest"],
                    "src/data/cross_architecture_full_test_manifest.csv",
                )

    def test_reverse_config_rejects_forward_manifests(self):
        data_module = MultiDomainDataModule(
            root_dir=".",
            train_manifest="gan_train.csv",
            test_manifest="diffusion_test.csv",
            split_level="family",
            train_families=["diffusion"],
            test_families=["gan"],
        )
        data_module.train_df = add_architecture_family(pd.DataFrame([
            {"label": 0, "architecture": "natural"},
            {"label": 1, "architecture": "progan"},
        ]))
        data_module.test_df = add_architecture_family(pd.DataFrame([
            {"label": 0, "architecture": "cifar_real"},
            {"label": 1, "architecture": "stable_diffusion_v14"},
        ]))

        with self.assertRaisesRegex(ValueError, "wrong fake family direction"):
            data_module._validate_manifest_integrity()

    def test_non_gan_cnndetection_family_can_be_explicitly_excluded(self):
        extra = pd.DataFrame([{
            "filepath": "cnndetection/crn.png",
            "dataset": "CNNDetection",
            "architecture": "crn",
            "label": 1,
        }])
        data = pd.concat([self.df, extra], ignore_index=True)
        splitter = FamilyAwareSplitter(
            train_families=["gan"],
            test_families=["diffusion"],
            exclude_families=["cnn"],
            real_strategy="family_matched",
        )

        train_df, test_df = splitter.split_dataframe(data)

        self.assertNotIn("crn", set(train_df["architecture"]))
        self.assertNotIn("crn", set(test_df["architecture"]))

    def test_metadata_family_mismatch_is_rejected(self):
        bad = self.df.copy()
        bad["architecture_family"] = ["real", "diffusion", "real", "diffusion"]

        with self.assertRaisesRegex(ValueError, "disagrees"):
            add_architecture_family(bad)

    def test_unknown_architecture_is_rejected(self):
        bad = self.df.copy()
        bad.loc[1, "architecture"] = "unmapped_generator"

        with self.assertRaisesRegex(ValueError, "Unknown architecture family"):
            add_architecture_family(bad)

    def test_all_official_synthbuster_architectures_are_diffusion(self):
        architectures = [
            "firefly",
            "dalle2",
            "dalle3",
            "glide",
            "midjourney",
            "sd_13",
            "sd_14",
            "sd_2",
            "sd_xl",
        ]
        data = pd.DataFrame({
            "architecture": architectures,
            "label": [1] * len(architectures),
        })

        result = add_architecture_family(data)

        self.assertEqual(set(result["architecture_family"]), {"diffusion"})


if __name__ == "__main__":
    unittest.main()
