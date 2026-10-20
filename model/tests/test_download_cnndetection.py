import importlib.util
import os
import stat
import tempfile
import unittest
import zipfile
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "download_cnndetection.py"
)
SPEC = importlib.util.spec_from_file_location("download_cnndetection", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class CNNDetectionDownloadTests(unittest.TestCase):
    def test_official_folder_names_use_canonical_architectures(self):
        cases = {
            "whichfaceisreal": "whichfaceisreal",
            "seeingdark": "sitd",
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for folder, expected in cases.items():
                image = root / folder / "1_fake" / "sample.png"
                image.parent.mkdir(parents=True)
                image.write_bytes(b"image")

                record = MODULE._scan_file(image, root)

                self.assertIsNotNone(record)
                self.assertEqual(record["architecture"], expected)

    def test_non_gan_cnndetection_families_are_not_mislabeled(self):
        expected = {
            "progan": "gan",
            "crn": "cnn",
            "san": "cnn",
            "sitd": "cnn",
            "imle": "implicit_generative",
            "deepfake": "face_manipulation",
        }
        for architecture, family in expected.items():
            self.assertEqual(
                MODULE.FAKE_FAMILY_MAP[architecture],
                family,
            )

    def test_ambiguous_or_missing_label_folder_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = root / "stylegan" / "sample.png"
            missing.parent.mkdir(parents=True)
            missing.write_bytes(b"image")
            ambiguous = (
                root / "stylegan" / "0_real" / "1_fake" / "sample.png"
            )
            ambiguous.parent.mkdir(parents=True)
            ambiguous.write_bytes(b"image")

            self.assertIsNone(MODULE._scan_file(missing, root))
            self.assertIsNone(MODULE._scan_file(ambiguous, root))

    def test_fake_architecture_comes_from_dataset_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "stylegan" / "bedroom" / "1_fake" / "sample.png"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"image")

            record = MODULE._scan_file(image, root)

            self.assertIsNotNone(record)
            self.assertEqual(record["architecture"], "stylegan")
            self.assertEqual(record["label"], 1)

    def test_fake_sampling_is_balanced_across_architectures(self):
        records = []
        for architecture in ("progan", "stylegan", "biggan"):
            for index in range(10):
                records.append({
                    "_source_path": Path(architecture) / f"{index}.png",
                    "split": "test",
                    "label": 1,
                    "architecture": architecture,
                    "suffix": ".png",
                })

        sampled = MODULE.sample_records(records, max_train=0, max_test=18)
        counts = {}
        for record in sampled:
            arch = record["architecture"]
            counts[arch] = counts.get(arch, 0) + 1

        self.assertEqual(counts, {
            "biggan": 3,
            "progan": 3,
            "stylegan": 3,
        })

    def test_extracts_balanced_sample_from_nested_train_zip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive_path = root / "progan_train.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                for label in ("0_real", "1_fake"):
                    for index in range(5):
                        archive.writestr(
                            f"car/{label}/{index}.png", b"image"
                        )

            output = root / "extracted"
            MODULE._extract_sampled_train_zip(
                archive_path, output, max_train=6, seed=42
            )

            images = list(
                (output / "progan_train" / "_train").rglob("*.png")
            )
            real = [path for path in images if "0_real" in path.parts]
            fake = [path for path in images if "1_fake" in path.parts]
            self.assertEqual(len(real), 3)
            self.assertEqual(len(fake), 3)
            self.assertTrue(
                (output / "progan_train" / "_train" / "selection.csv").exists()
            )

    def test_category_named_train_is_not_treated_as_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            category_image = (
                root / "progan" / "train" / "1_fake" / "sample.png"
            )
            category_image.parent.mkdir(parents=True)
            category_image.write_bytes(b"image")

            record = MODULE._scan_file(category_image, root)

            self.assertIsNotNone(record)
            self.assertEqual(record["split"], "test")

    def test_explicit_train_wrapper_is_treated_as_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_image = (
                root / "progan_train" / "_train" / "car"
                / "1_fake" / "sample.png"
            )
            train_image.parent.mkdir(parents=True)
            train_image.write_bytes(b"image")

            record = MODULE._scan_file(train_image, root)

            self.assertIsNotNone(record)
            self.assertEqual(record["split"], "train")

    def test_extracts_split_7z_archive(self):
        import multivolumefile
        import py7zr

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            payload = source / "sample.txt"
            payload.write_bytes(b"split-volume-test" * 100)

            archive_base = root / "sample.7z"
            with multivolumefile.MultiVolume(
                archive_base, mode="wb", volume=256, ext_digits=3
            ) as volume:
                with py7zr.SevenZipFile(volume, mode="w") as archive:
                    archive.write(payload, arcname="sample.txt")

            first_part = root / "sample.7z.001"
            output = root / "output"
            output.mkdir()
            MODULE._extract_7z_with_progress(first_part, output)

            self.assertEqual(
                (output / "sample.txt").read_bytes(),
                payload.read_bytes(),
            )

    def test_single_worker_scan_and_hardlink_transfer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            image = source / "train" / "progan" / "1_fake" / "sample.jpg"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"image-data")

            records = MODULE.collect_images(source, workers=1)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["architecture"], "progan")
            self.assertEqual(records[0]["label"], 1)

            metadata = MODULE.organize_images(
                records, root / "output", workers=1, transfer_mode="hardlink"
            )
            destination = root / "output" / metadata[0]["filepath"]

            self.assertTrue(destination.exists())
            self.assertTrue(os.path.samefile(image, destination))

    def test_replace_output_preserves_hf_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "cnndetection"
            cache_file = dataset_root / "_hf_cache" / "archive.zip"
            train_file = dataset_root / "train" / "sample.png"
            test_file = dataset_root / "test" / "sample.png"
            metadata = dataset_root / "cnndetection_metadata.csv"
            for path in (cache_file, train_file, test_file):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"data")
            metadata.write_text("filepath,label\n", encoding="utf-8")
            train_file.chmod(stat.S_IREAD)

            MODULE.prepare_output_directory(root, replace_output=True)

            self.assertTrue(cache_file.exists())
            self.assertFalse((dataset_root / "train").exists())
            self.assertFalse((dataset_root / "test").exists())
            self.assertFalse(metadata.exists())


if __name__ == "__main__":
    unittest.main()
