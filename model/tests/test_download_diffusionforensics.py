from pathlib import Path

from scripts.download_diffusionforensics import _scan_file, sample_records


def test_official_imagenet_domain_does_not_hide_adm_architecture():
    root = Path("dataset")
    record = _scan_file(
        root / "train" / "imagenet" / "adm" / "image.png",
        root,
    )

    assert record is not None
    assert record["split"] == "train"
    assert record["label"] == 1
    assert record["architecture"] == "adm"


def test_official_real_directory_is_labeled_real():
    root = Path("dataset")
    record = _scan_file(
        root / "test" / "lsun_bedroom" / "real" / "image.png",
        root,
    )

    assert record is not None
    assert record["split"] == "test"
    assert record["label"] == 0
    assert record["architecture"] == "natural_images"


def test_sampling_is_balanced_and_reproducible():
    records = []
    for split, count in (("train", 10), ("test", 6)):
        for label in (0, 1):
            records.extend(
                {"split": split, "label": label, "id": index}
                for index in range(count)
            )

    first = sample_records(records, max_train=8, max_test=4, seed=42)
    second = sample_records(records, max_train=8, max_test=4, seed=42)

    assert first == second
    assert len(first) == 12
