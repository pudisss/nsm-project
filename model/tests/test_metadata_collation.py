import unittest

import torch

from src.data.dataset import collate_fn
from src.training.trainer import extract_metadata_values


def _sample(index: int) -> dict:
    return {
        "rgb": torch.zeros(3, 4, 4),
        "freq": torch.zeros(1, 4, 4),
        "noise": torch.zeros(3, 4, 4),
        "label": torch.tensor(index % 2),
        "metadata": {
            "filepath": f"image-{index}.png",
            "architecture": f"arch-{index}",
            "dataset": f"dataset-{index}",
        },
    }


class MetadataCollationTest(unittest.TestCase):
    def test_custom_collate_preserves_metadata_as_sample_dicts(self):
        batch = collate_fn([_sample(0), _sample(1)])

        self.assertEqual(
            extract_metadata_values(batch["metadata"], "architecture"),
            ["arch-0", "arch-1"],
        )
        self.assertEqual(
            extract_metadata_values(batch["metadata"], "dataset"),
            ["dataset-0", "dataset-1"],
        )

    def test_metadata_extraction_accepts_default_collated_dict(self):
        metadata = {
            "filepath": ["image-0.png", "image-1.png"],
            "architecture": ["arch-0", "arch-1"],
            "dataset": ["dataset-0", "dataset-1"],
        }

        self.assertEqual(
            extract_metadata_values(metadata, "architecture"),
            ["arch-0", "arch-1"],
        )

    def test_metadata_extraction_fills_missing_field(self):
        metadata = [{"architecture": "arch-0"}, {"architecture": "arch-1"}]

        self.assertEqual(
            extract_metadata_values(metadata, "dataset"),
            ["unknown", "unknown"],
        )


if __name__ == "__main__":
    unittest.main()
