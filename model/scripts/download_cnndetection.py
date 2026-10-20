#!/usr/bin/env python3
"""Download and preprocess CNNDetection dataset.

Dataset: Wang et al. 2020 "CNN-generated images are surprisingly easy to spot... for now"
Source:  HuggingFace — sywang/CNNDetection (official repo by paper author Sheng-Yu Wang)
  https://huggingface.co/datasets/sywang/CNNDetection

  Fallback: download original from Google Drive links in the GitHub README:
    https://github.com/peterwang512/CNNDetection
  then point --source-dir at the extracted folder.

Output directory structure
--------------------------
<output_dir>/cnndetection/
    train/
        0_real/   natural_0_real_<id>.<ext>
        1_fake/   <arch>_1_fake_<id>.<ext>
    test/
        0_real/   natural_0_real_<id>.<ext>
        1_fake/   <arch>_1_fake_<id>.<ext>
    cnndetection_metadata.csv   ← columns: filepath,split,label,dataset,
                                   architecture,architecture_family

Architecture labels used
------------------------
Real : natural
Fake : progan, stylegan, stylegan2, biggan, cyclegan, stargan, gaugan,
       deepfake, seeingdark, san, crn, imle, sitd
"""

import argparse
import csv
import os
import random
import shutil
import stat
import sys
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ── Configure ──────────────────────────────────────────────────────────────
HF_DATASET_ID = "sywang/CNNDetection"  # official author repo (Sheng-Yu Wang / PeterWang512)

ARCH_MAP = {
    "progan": "progan",      "progan_train": "progan",
    "stylegan": "stylegan",  "stylegan2": "stylegan2",
    "biggan": "biggan",      "cyclegan": "cyclegan",
    "stargan": "stargan",    "gaugan": "gaugan",
    "deepfake": "deepfake",  "seeingdark": "sitd",
    "san": "san",            "crn": "crn",
    "imle": "imle",          "sitd": "sitd",
    "whichfaceisreal": "whichfaceisreal",
}
FAKE_FAMILY_MAP = {
    "progan": "gan",
    "stylegan": "gan",
    "stylegan2": "gan",
    "biggan": "gan",
    "cyclegan": "gan",
    "stargan": "gan",
    "gaugan": "gan",
    "whichfaceisreal": "gan",
    "crn": "cnn",
    "san": "cnn",
    "sitd": "cnn",
    "imle": "implicit_generative",
    "deepfake": "face_manipulation",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_WORKERS = 4
DEFAULT_DOWNLOAD_WORKERS = 1
DEFAULT_MAX_TRAIN = 10_000  # max training images kept (balanced real/fake)
DEFAULT_MAX_TEST  = 2_000   # max test images kept (balanced real/fake)
DEFAULT_SEED = 42
DOWNLOAD_PROFILES = {
    "progan": [
        "progan_testset.zip",
        "progan_val.zip",
        *(f"progan_train.7z.{part:03d}" for part in range(1, 8)),
    ],
    "full": [
        "CNN_synth_testset.zip",
        "progan_testset.zip",
        "progan_val.zip",
        *(f"progan_train.7z.{part:03d}" for part in range(1, 8)),
    ],
}
# ───────────────────────────────────────────────────────────────────────────


def _count_images(directory: Path) -> int:
    """Count image files recursively in a directory."""
    return sum(1 for p in directory.rglob("*")
               if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def _extract_zip_with_progress(zp: Path, extract_dir: Path) -> None:
    """Extract a zip file one file at a time with a tqdm progress bar."""
    with zipfile.ZipFile(zp, "r") as zf:
        members = zf.namelist()
        for member in tqdm(members, desc=zp.name, unit="file", leave=True):
            zf.extract(member, extract_dir)


def _extract_sampled_train_zip(
    zip_path: Path,
    extract_dir: Path,
    max_train: int,
    seed: int,
) -> None:
    """Extract and record a deterministic balanced ProGAN training sample."""
    if max_train <= 0:
        return

    output_dir = extract_dir / "progan_train" / "_train"
    marker = output_dir / f".done_{max_train}_seed_{seed}"
    if marker.exists():
        print(
            f"  Already extracted {max_train:,} sampled training images "
            f"with seed {seed}."
        )
        return
    if output_dir.exists() and any(output_dir.iterdir()):
        sys.exit(
            "ERROR: an untracked or differently seeded ProGAN training sample "
            f"already exists at {output_dir}. Delete only that directory and "
            "rerun so the research sample can be reproduced."
        )

    per_class = max_train // 2
    rng = random.Random(seed)
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [
            name for name in zf.namelist()
            if Path(name).suffix.lower() in IMAGE_EXTENSIONS
        ]
        real = [name for name in members if "0_real" in Path(name).parts]
        fake = [name for name in members if "1_fake" in Path(name).parts]
        selected = (
            rng.sample(real, min(per_class, len(real)))
            + rng.sample(fake, min(per_class, len(fake)))
        )

        for member in tqdm(
            selected, desc=zip_path.name, unit="file", leave=True
        ):
            zf.extract(member, output_dir)

    marker.parent.mkdir(parents=True, exist_ok=True)
    with (output_dir / "selection.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["archive_member", "label", "seed"])
        for member in selected:
            label = 0 if "0_real" in Path(member).parts else 1
            writer.writerow([member, label, seed])
    marker.touch()


def _extract_7z_with_progress(first_part: Path, extract_dir: Path) -> None:
    """Extract a multi-part 7z archive without repeatedly scanning the HDD."""
    try:
        import multivolumefile
        import py7zr
    except ImportError:
        sys.exit(
            "ERROR: run  pip install py7zr multivolumefile  "
            "to extract split .7z archives"
        )

    archive_base = Path(str(first_part).removesuffix(".001"))
    with tqdm(total=1, desc=first_part.name, unit="archive", leave=True) as pbar:
        with multivolumefile.MultiVolume(
            archive_base, mode="rb", ext_digits=3
        ) as volume:
            with py7zr.SevenZipFile(volume, mode="r") as zf:
                zf.extractall(path=extract_dir)
        pbar.update(1)


def _extract_archives(
    cache_dir: Path, max_train: int, max_test: int, seed: int
) -> Path:
    """Extract archives into an 'extracted' subfolder.

    Stops extracting zip files as soon as max_train + max_test images have
    been extracted — no need to extract every archive in full.
    The large multi-part 7z train archive is only touched if the zips alone
    don't provide enough images.
    """
    extract_dir = cache_dir / "extracted"
    extract_dir.mkdir(exist_ok=True)

    needed = max_train + max_test
    current_count = _count_images(extract_dir)

    # ── zip files: extract one at a time and stop early ────────────────────
    for zp in sorted(cache_dir.glob("*.zip")):
        if current_count >= needed:
            print(f"  {current_count:,} images extracted — "
                  f"stopping early (need only {needed:,}).")
            break

        marker = extract_dir / f".done_{zp.stem}"
        if marker.exists():
            print(f"  Already extracted: {zp.name}")
            continue
        print(f"  Extracting {zp.name} …")
        _extract_zip_with_progress(zp, extract_dir)
        marker.touch()
        current_count = _count_images(extract_dir)

    # ── check again after all zips ─────────────────────────────────────────
    if max_train <= 0 and current_count >= needed:
        print(f"  {current_count:,} images extracted — "
              f"skipping large .7z train archive (need only {needed:,}).")
        return extract_dir

    # ── multi-part 7z train archive (can be 90+ GB) ────────────────────────
    for first_part in sorted(cache_dir.glob("*.7z.001")):
        stem   = first_part.name.replace(".7z.001", "")
        marker = extract_dir / f".done_{stem}_7z"
        if marker.exists():
            print(f"  Already extracted: {first_part.name} (multi-part)")
            continue
        print(f"  Extracting {first_part.name} (multi-part 7z, this may take a while) …")
        _extract_7z_with_progress(first_part, extract_dir)
        marker.touch()

    nested_train_zip = extract_dir / "progan_train.zip"
    if nested_train_zip.exists():
        print(
            f"  Extracting a balanced sample from {nested_train_zip.name} ..."
        )
        _extract_sampled_train_zip(
            nested_train_zip, extract_dir, max_train, seed
        )

    return extract_dir


def download_from_hf(output_dir: Path, hf_token: str | None,
                     max_train: int = DEFAULT_MAX_TRAIN,
                     max_test: int = DEFAULT_MAX_TEST,
                     download_workers: int = DEFAULT_DOWNLOAD_WORKERS,
                     download_profile: str = "progan",
                     seed: int = DEFAULT_SEED) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("ERROR: run  pip install huggingface_hub")

    cache_dir = output_dir / "cnndetection" / "_hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Skip re-download if archives are already present
    required_files = DOWNLOAD_PROFILES[download_profile]
    if any(not (cache_dir / name).exists() for name in required_files):
        print(f"Downloading {HF_DATASET_ID} from HuggingFace …")
        snapshot_download(
            repo_id=HF_DATASET_ID,
            repo_type="dataset",
            local_dir=str(cache_dir),
            token=hf_token,
            max_workers=download_workers,
            allow_patterns=DOWNLOAD_PROFILES[download_profile],
            ignore_patterns=["*.git*", "*.md", "*.txt", "*.json"],
        )
    else:
        print(f"Archives already present in {cache_dir}, skipping download.")

    print("Extracting archives …")
    return _extract_archives(cache_dir, max_train, max_test, seed)


# ── Scan ───────────────────────────────────────────────────────────────────

def _scan_file(img_path: Path, source_root: Path) -> dict | None:
    """Parse a single image path into a metadata record (runs in thread pool)."""
    if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
        return None

    parts = [p.lower() for p in img_path.relative_to(source_root).parts]

    # Explicit internal wrappers avoid confusing the ImageNet category named
    # "train" (locomotive) with the dataset split.
    if "_train" in parts:
        split = "train"
    elif "_test" in parts or "_val" in parts:
        split = "test"
    elif parts and parts[0] in ("train", "test", "val"):
        split = "train" if parts[0] == "train" else "test"
    else:
        split = "test"

    label_folders = [
        part for part in parts if part in ("0_real", "1_fake")
    ]
    if len(label_folders) != 1:
        return None
    label_folder = label_folders[0]
    if label_folder == "0_real":
        label = 0
    else:
        label = 1

    # Architecture is identified by a dataset folder, never by 0_real/1_fake.
    arch = "unknown"
    for part in parts:
        if part in ARCH_MAP:
            arch = ARCH_MAP[part]
            break

    if label == 0:
        arch = "natural"
    elif arch == "unknown":
        return None

    return {
        "_source_path": img_path,
        "split":        split,
        "label":        label,
        "architecture": arch,
        "suffix":       img_path.suffix.lower(),
    }


def collect_images(source_root: Path, workers: int) -> list[dict]:
    """Scan source_root and return image records.

    One worker performs a linear walk, which avoids concurrent seek-heavy stat
    calls and is normally faster on a mechanical disk.
    """
    all_paths = (p for p in source_root.rglob("*") if p.is_file())

    if workers == 1:
        records = []
        for path in tqdm(all_paths, desc="Scanning files", unit="file"):
            result = _scan_file(path, source_root)
            if result is not None:
                records.append(result)
        return records

    all_paths = list(all_paths)
    records = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_scan_file, p, source_root): p for p in all_paths}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Scanning files"):
            result = fut.result()
            if result is not None:
                records.append(result)

    return records


def validate_records(records: list[dict]) -> None:
    """Reject ambiguous or internally inconsistent source records."""
    seen_sources: set[Path] = set()
    valid_fake_archs = set(ARCH_MAP.values())

    for rec in records:
        source = rec["_source_path"]
        if source in seen_sources:
            raise ValueError(f"Duplicate source image: {source}")
        seen_sources.add(source)

        if rec["split"] not in {"train", "test"}:
            raise ValueError(f"Invalid split for {source}: {rec['split']}")
        if rec["label"] == 0 and rec["architecture"] != "natural":
            raise ValueError(
                f"Real image has fake architecture at {source}: "
                f"{rec['architecture']}"
            )
        if rec["label"] == 1 and rec["architecture"] not in valid_fake_archs:
            raise ValueError(
                f"Fake image has unknown architecture at {source}: "
                f"{rec['architecture']}"
            )


def sample_records(
    records: list[dict],
    max_train: int,
    max_test: int,
    seed: int = DEFAULT_SEED,
) -> list[dict]:
    """Randomly sample records, keeping at most max_train/2 per class for train
    and max_test/2 per class for test. Fake samples are balanced across
    generator architectures."""
    buckets: dict[tuple, list] = defaultdict(list)
    rng = random.Random(seed)
    for rec in records:
        buckets[(rec["split"], rec["label"])].append(rec)
    sampled: list[dict] = []
    for (split, label), recs in buckets.items():
        limit = (max_train if split == "train" else max_test) // 2
        if label == 0:
            sampled.extend(rng.sample(recs, min(len(recs), limit)))
            continue

        by_arch: dict[str, list[dict]] = defaultdict(list)
        for rec in recs:
            by_arch[rec["architecture"]].append(rec)

        architectures = sorted(by_arch)
        base, remainder = divmod(limit, len(architectures))
        for index, arch in enumerate(architectures):
            arch_limit = base + (1 if index < remainder else 0)
            arch_records = by_arch[arch]
            sampled.extend(
                rng.sample(arch_records, min(len(arch_records), arch_limit))
            )
    # Read selected source files in path order to reduce HDD seeks.
    return sorted(sampled, key=lambda rec: str(rec["_source_path"]).lower())


# ── Organise ───────────────────────────────────────────────────────────────

def _assign_output_paths(records: list[dict]) -> list[dict]:
    """Pre-assign canonical output path to every record (single-threaded, O(n))."""
    label_dir = {0: "0_real", 1: "1_fake"}
    type_str  = {0: "real",   1: "fake"}
    counters: dict[tuple, int] = defaultdict(int)

    for rec in records:
        key = (rec["split"], rec["label"], rec["architecture"])
        idx = counters[key]
        counters[key] += 1

        arch  = rec["architecture"]
        split = rec["split"]
        lbl   = rec["label"]

        filename = f"{arch}_{lbl}_{type_str[lbl]}_{idx}{rec['suffix']}"
        rec["_rel_path"] = Path("cnndetection") / split / label_dir[lbl] / filename

    return records


def _transfer_one(rec: dict, out_root: Path, transfer_mode: str) -> dict:
    """Transfer one image, using a zero-copy hard link when requested."""
    dest = out_root / rec["_rel_path"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and transfer_mode == "hardlink":
        if not os.path.samefile(rec["_source_path"], dest):
            raise RuntimeError(
                f"Stale output file does not match selected source: {dest}. "
                "Delete the generated train/test folders and metadata CSV, "
                "then rerun."
            )
    elif not dest.exists():
        if transfer_mode == "hardlink":
            try:
                os.link(rec["_source_path"], dest)
            except OSError:
                shutil.copyfile(rec["_source_path"], dest)
        else:
            shutil.copyfile(rec["_source_path"], dest)
    return {
        "filepath":     str(rec["_rel_path"]).replace("\\", "/"),
        "split":        rec["split"],
        "label":        rec["label"],
        "dataset":      "CNNDetection",
        "architecture": rec["architecture"],
        "architecture_family": (
            "real"
            if rec["label"] == 0
            else FAKE_FAMILY_MAP[rec["architecture"]]
        ),
    }


def organize_images(records: list[dict], out_root: Path, workers: int,
                    transfer_mode: str = "copy") -> list[dict]:
    """Transfer images and return metadata rows."""
    records = _assign_output_paths(records)

    if workers == 1:
        return [
            _transfer_one(rec, out_root, transfer_mode)
            for rec in tqdm(records, desc="Transferring images")
        ]

    metadata = [None] * len(records)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_transfer_one, rec, out_root, transfer_mode): i
                   for i, rec in enumerate(records)}
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="Transferring images"):
            idx = futures[fut]
            metadata[idx] = fut.result()

    return [m for m in metadata if m is not None]


# ── Output ─────────────────────────────────────────────────────────────────

def prepare_output_directory(out_root: Path, replace_output: bool) -> None:
    """Remove generated CNNDetection outputs while preserving the source cache."""
    if not replace_output:
        return

    dataset_root = (out_root / "cnndetection").resolve()
    dataset_root.mkdir(parents=True, exist_ok=True)
    targets = [
        dataset_root / "train",
        dataset_root / "test",
        dataset_root / "cnndetection_metadata.csv",
    ]

    def remove_readonly(func, path, _exc_info):
        os.chmod(path, stat.S_IWRITE)
        func(path)

    for target in targets:
        resolved = target.resolve()
        if dataset_root != resolved and dataset_root not in resolved.parents:
            raise ValueError(f"Refusing to remove path outside {dataset_root}: {resolved}")
        if resolved.is_dir():
            shutil.rmtree(resolved, onerror=remove_readonly)
        elif resolved.exists():
            os.chmod(resolved, stat.S_IWRITE)
            resolved.unlink()


def build_metadata_csv(metadata: list[dict], out_root: Path) -> Path:
    df = pd.DataFrame(metadata)
    out_csv = out_root / "cnndetection" / "cnndetection_metadata.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return out_csv


def print_summary(df: pd.DataFrame) -> None:
    print("\n-- CNNDetection summary --------------------------------------")
    print(df.groupby(
        ["split", "architecture_family", "architecture", "label"]
    )
            .size().reset_index(name="count").to_string(index=False))
    print("\nArchitecture-family totals:")
    print(df.groupby(["split", "architecture_family", "label"])
            .size().reset_index(name="count").to_string(index=False))
    print(f"\nTotal images : {len(df):,}")


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    global HF_DATASET_ID

    parser = argparse.ArgumentParser(
        description="Download & preprocess CNNDetection dataset"
    )
    parser.add_argument("--output-dir", "-o", default="/workspace/data/processed",
                        help="Root output directory (default: /workspace/data/processed)")
    parser.add_argument("--hf-token", default=None,
                        help="HuggingFace API token (for gated datasets)")
    parser.add_argument("--source-dir", default=None,
                        help="Path to already-downloaded folder (skips HF download)")
    parser.add_argument("--hf-dataset-id", default=HF_DATASET_ID,
                        help=f"HuggingFace dataset ID (default: {HF_DATASET_ID})")
    parser.add_argument("--workers", "-j", type=int, default=DEFAULT_WORKERS,
                        help=f"Scan/transfer worker threads (default: {DEFAULT_WORKERS}; "
                             "use 1 for a mechanical HDD)")
    parser.add_argument("--download-workers", type=int,
                        default=DEFAULT_DOWNLOAD_WORKERS,
                        help=f"Concurrent HuggingFace downloads "
                             f"(default: {DEFAULT_DOWNLOAD_WORKERS})")
    parser.add_argument("--download-profile", choices=tuple(DOWNLOAD_PROFILES),
                        default="progan",
                        help="'progan' downloads train/validation/test ProGAN "
                             "archives; 'full' also downloads the 20 GB "
                             "cross-generator test archive")
    parser.add_argument("--transfer-mode", choices=("copy", "hardlink"),
                        default="copy",
                        help="Use 'hardlink' to avoid duplicating image data when "
                             "cache and output are on the same disk")
    parser.add_argument(
        "--replace-output",
        action="store_true",
        help=(
            "Replace generated cnndetection train/test folders and metadata "
            "while preserving cnndetection/_hf_cache"
        ),
    )
    parser.add_argument("--max-train", type=int, default=DEFAULT_MAX_TRAIN,
                        help=f"Max training images to keep, balanced real/fake "
                             f"(default: {DEFAULT_MAX_TRAIN})")
    parser.add_argument("--max-test", type=int, default=DEFAULT_MAX_TEST,
                        help=f"Max test images to keep, balanced real/fake "
                             f"(default: {DEFAULT_MAX_TEST})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Sampling seed (default: {DEFAULT_SEED})")
    args = parser.parse_args()

    if args.workers < 1 or args.download_workers < 1:
        parser.error("--workers and --download-workers must be at least 1")

    HF_DATASET_ID = args.hf_dataset_id
    out_root = Path(args.output_dir)
    print(f"Max train : {args.max_train}  |  Max test : {args.max_test}")

    if args.source_dir:
        source_root = Path(args.source_dir)
        print(f"Using local source : {source_root}")
    else:
        source_root = download_from_hf(
            out_root,
            args.hf_token,
            args.max_train,
            args.max_test,
            args.download_workers,
            args.download_profile,
            args.seed,
        )
        print(f"Downloaded to      : {source_root}")

    print(f"Workers            : {args.workers}")
    print(f"Transfer mode      : {args.transfer_mode}")
    print(f"Sampling seed      : {args.seed}")

    records = collect_images(source_root, args.workers)
    print(f"Images found       : {len(records):,}")
    if not records:
        sys.exit("ERROR: no images found — check the path or dataset structure.")

    validate_records(records)
    records = sample_records(
        records, args.max_train, args.max_test, args.seed)
    print(f"Images sampled     : {len(records):,}  "
          f"(max {args.max_train} train / max {args.max_test} test)")

    prepare_output_directory(out_root, args.replace_output)
    metadata = organize_images(
        records, out_root, args.workers, args.transfer_mode)

    csv_path = build_metadata_csv(metadata, out_root)
    print(f"Metadata saved to  : {csv_path}")
    print_summary(pd.read_csv(csv_path))


if __name__ == "__main__":
    main()
