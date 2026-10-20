import zipfile
from pathlib import Path

import pytest

from scripts.download_synthbuster import (
    _scan_file,
    build_metadata_csv,
    collect_images,
    extract_archives,
    organize_images,
    sample_records,
)


def test_official_archive_folder_names_use_canonical_architectures():
    root = Path("dataset")
    expected = {
        "stable-diffusion-1-3": "sd_13",
        "stable-diffusion-1-4": "sd_14",
        "midjourney-v5": "midjourney",
        "dalle3": "dalle3",
        "glide": "glide",
    }

    for folder, architecture in expected.items():
        record = _scan_file(
            root / "synthbuster" / folder / "sample.png",
            root,
        )

        assert record is not None
        assert record["label"] == 1
        assert record["architecture"] == architecture


def test_extract_archives_combines_fake_and_real_sources(tmp_path):
    fake_archive = tmp_path / "synthbuster.zip"
    real_archive = tmp_path / "real_RAISE_1k.zip"
    with zipfile.ZipFile(fake_archive, "w") as archive:
        archive.writestr(
            "synthbuster/stable-diffusion-1-4/fake.png",
            b"fake",
        )
    with zipfile.ZipFile(real_archive, "w") as archive:
        archive.writestr("real_RAISE_1k/real.png", b"real")

    extract_dir = tmp_path / "extracted"
    source_root = extract_archives(
        [fake_archive, real_archive],
        extract_dir,
    )

    assert source_root == extract_dir
    assert (
        extract_dir / "synthbuster" / "stable-diffusion-1-4" / "fake.png"
    ).read_bytes() == b"fake"
    assert (extract_dir / "real_RAISE_1k" / "real.png").read_bytes() == b"real"

    records = collect_images(source_root, workers=1)
    sampled = sample_records(records, max_test=2, seed=42)
    metadata = organize_images(sampled, tmp_path / "processed", workers=1)
    csv_path = build_metadata_csv(metadata, tmp_path / "processed")
    assert csv_path.exists()
    assert {row["label"] for row in metadata} == {0, 1}

    extracted_file = extract_dir / "real_RAISE_1k" / "real.png"
    first_mtime = extracted_file.stat().st_mtime_ns
    extract_archives([fake_archive, real_archive], extract_dir)
    assert extracted_file.stat().st_mtime_ns == first_mtime


def test_extract_archives_rejects_path_traversal(tmp_path):
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../outside.txt", b"unsafe")

    with pytest.raises(ValueError, match="Unsafe archive member"):
        extract_archives([archive_path], tmp_path / "extracted")

    assert not (tmp_path / "outside.txt").exists()


def test_sampling_remains_balanced_and_reproducible():
    records = [
        {"label": label, "id": index}
        for label in (0, 1)
        for index in range(10)
    ]

    first = sample_records(records, max_test=8, seed=42)
    second = sample_records(records, max_test=8, seed=42)

    assert first == second
    assert sum(record["label"] == 0 for record in first) == 4
    assert sum(record["label"] == 1 for record in first) == 4
