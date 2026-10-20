#!/usr/bin/env python3
"""Extract and preprocess a locally downloaded SynthBuster dataset.

Dataset: Bammey 2023 "SynthBuster: Towards Detection of Diffusion Model Deep Fakes"
Official source: https://zenodo.org/records/10066460
Paper: https://arxiv.org/abs/2308.01441

This script performs no network downloads. Pass local ZIP archives with
--archive, or pass an already extracted directory with --source-dir.

Dataset overview
----------------
  Real  : ~1000 photos from RAISE-1k dataset
  Fake  : ~9000 photorealistic images from 9 generators
          Adobe Firefly, DALL-E 2/3, GLIDE, Midjourney v5,
          Stable Diffusion 1.3/1.4/2/XL
  All images are considered "test" (evaluation-only dataset).

Output directory structure
--------------------------
<output_dir>/synthbuster/
    raise_real/
        test/0_real/   raise_real_0_real_<id>.png
    <arch>/             e.g. dalle2/, midjourney/, firefly/ …
        test/1_fake/   <arch>_1_fake_<id>.png
    synthbuster_metadata.csv   ← columns: filepath,split,label,dataset,
                                  architecture,architecture_family

Architecture labels used
------------------------
Real : raise_real
Fake : firefly, dalle2, dalle3, glide, midjourney, sd_13, sd_14, sd_2, sd_xl
"""

import argparse
import random
import shutil
import sys
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ── Configure ──────────────────────────────────────────────────────────────
DEFAULT_WORKERS = 32

# Map archive folder names to canonical architecture names
ARCH_MAP = {
    # real
    "real": "raise_real", "raise": "raise_real", "raise_1k": "raise_real",
    "raise1k": "raise_real", "raise-1k": "raise_real",
    "real_raise_1k": "raise_real", "real-raise-1k": "raise_real",
    # fake
    "adobe-firefly": "firefly", "firefly": "firefly",
    "dalle-2": "dalle2", "dalle2": "dalle2", "dall-e-2": "dalle2",
    "dalle_2": "dalle2", "dall_e_2": "dalle2",
    "dalle-3": "dalle3", "dalle3": "dalle3", "dall-e-3": "dalle3",
    "dalle_3": "dalle3", "dall_e_3": "dalle3",
    "midjourney": "midjourney", "midjourney-5.1": "midjourney",
    "midjourney-v5": "midjourney", "midjourney_v5": "midjourney",
    "midjourney-5": "midjourney",
    "glide": "glide",
    "stable-diffusion-1.3": "sd_13", "stable-diffusion-1-3": "sd_13",
    "stable_diffusion_1_3": "sd_13",
    "sd_1.3": "sd_13", "sd_13": "sd_13",
    "stable-diffusion-1.4": "sd_14", "stable-diffusion-1-4": "sd_14",
    "sd_1.4": "sd_14", "stable_diffusion_1_4": "sd_14",
    "stable-diffusion-1": "sd_14",   "sd_14": "sd_14",
    "stable-diffusion-2": "sd_2", "stable_diffusion_2": "sd_2",
    "sd_2.1": "sd_2",
    "stable-diffusion-2.1": "sd_2",  "sd_2": "sd_2",
    "stable-diffusion-xl": "sd_xl",  "sdxl": "sd_xl",
    "stable_diffusion_xl": "sd_xl",
    "sd_xl": "sd_xl", "stable-diffusion-xl-base-1.0": "sd_xl",
    # generic fallbacks
    "stable-diffusion": "sd_14",
}

# All SynthBuster images are test-only (no training split)
DEFAULT_SPLIT     = "test"
DEFAULT_MAX_TEST  = 2_000   # max test images kept (balanced real/fake)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
# ───────────────────────────────────────────────────────────────────────────


def _safe_extract_zip(archive_path: Path, extract_dir: Path) -> None:
    """Extract a local ZIP while rejecting members outside extract_dir."""
    destination = extract_dir.resolve()
    with zipfile.ZipFile(archive_path, "r") as archive:
        members = archive.infolist()
        for member in tqdm(
            members, desc=f"Extracting {archive_path.name}", unit="file"
        ):
            member_path = (extract_dir / member.filename).resolve()
            if destination != member_path and destination not in member_path.parents:
                raise ValueError(
                    f"Unsafe archive member in {archive_path}: {member.filename}"
                )
            archive.extract(member, extract_dir)


def extract_archives(archive_paths: list[Path], extract_dir: Path) -> Path:
    """Extract local archives once and return their common extraction root."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    marker_dir = extract_dir / ".synthbuster_extract"
    marker_dir.mkdir(exist_ok=True)

    for archive_path in archive_paths:
        archive_path = archive_path.resolve()
        if not archive_path.is_file():
            raise FileNotFoundError(f"Archive not found: {archive_path}")
        if not zipfile.is_zipfile(archive_path):
            raise ValueError(f"Not a valid ZIP archive: {archive_path}")

        stat = archive_path.stat()
        marker = marker_dir / (
            f"{archive_path.name}.{stat.st_size}.{stat.st_mtime_ns}.done"
        )
        if marker.exists():
            print(f"Already extracted: {archive_path}")
            continue

        _safe_extract_zip(archive_path, extract_dir)
        marker.touch()

    return extract_dir


# ── Scan local directory ───────────────────────────────────────────────────

def _scan_file(img_path: Path, source_root: Path) -> dict | None:
    """Parse one image path into a record (runs in thread pool)."""
    if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
        return None

    parts = [p.lower() for p in img_path.relative_to(source_root).parts]

    # Detect label from folder names
    if any(p in ("0_real", "real") for p in parts):
        label = 0
    elif any(p in ("1_fake", "fake") for p in parts):
        label = 1
    else:
        label = None

    # Detect architecture
    arch = None
    for part in reversed(parts):            # deepest folder wins
        if part in ARCH_MAP:
            arch = ARCH_MAP[part]
            break

    # If no explicit label dir, infer from architecture
    if arch is None:
        return None
    if label is None:
        label = 0 if arch == "raise_real" else 1

    return {
        "_source_path": img_path,
        "_pil_image":   None,
        "split":        DEFAULT_SPLIT,
        "label":        label,
        "architecture": arch,
        "suffix":       img_path.suffix.lower(),
    }


def collect_images(source_root: Path, workers: int) -> list[dict]:
    all_paths = [p for p in source_root.rglob("*") if p.is_file()]
    records = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_scan_file, p, source_root): p for p in all_paths}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Scanning files"):
            r = fut.result()
            if r is not None:
                records.append(r)
    records.sort(key=lambda record: str(record["_source_path"]))
    return records


def sample_records(records: list[dict], max_test: int, seed: int) -> list[dict]:
    """Randomly sample to at most max_test/2 per class (balanced real/fake).
    SynthBuster is test-only so no train split is considered."""
    buckets: dict[int, list] = defaultdict(list)
    for rec in records:
        buckets[rec["label"]].append(rec)
    missing = {0, 1} - set(buckets)
    if missing:
        names = {0: "real", 1: "fake"}
        missing_names = ", ".join(names[label] for label in sorted(missing))
        raise ValueError(
            f"Source directory is missing {missing_names} images. "
            "The official synthbuster.zip contains only fake images; "
            "extract RAISE-1k into the same source directory."
        )
    rng = random.Random(seed)
    sampled: list[dict] = []
    for label, recs in buckets.items():
        limit = max_test // 2
        sampled.extend(rng.sample(recs, min(len(recs), limit)))
    return sampled


# ── Assign output paths (single-threaded) ─────────────────────────────────

def assign_output_paths(records: list[dict]) -> list[dict]:
    label_dir = {0: "0_real", 1: "1_fake"}
    type_str  = {0: "real",   1: "fake"}
    counters: dict[tuple, int] = defaultdict(int)

    for rec in records:
        arch  = rec["architecture"]
        lbl   = rec["label"]
        split = rec["split"]
        key   = (arch, lbl)
        idx   = counters[key]
        counters[key] += 1
        ext = rec.get("suffix", ".png")
        rec["_rel_path"] = (
            Path("synthbuster") / arch / split / label_dir[lbl]
            / f"{arch}_{lbl}_{type_str[lbl]}_{idx}{ext}"
        )
    return records


# ── Copy worker ───────────────────────────────────────────────────────────

def _copy_one(rec: dict, out_root: Path) -> dict:
    dest = out_root / rec["_rel_path"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        shutil.copy2(rec["_source_path"], dest)
    return {
        "filepath":     str(rec["_rel_path"]).replace("\\", "/"),
        "split":        rec["split"],
        "label":        rec["label"],
        "dataset":      "SynthBuster",
        "architecture": rec["architecture"],
        "architecture_family": "real" if rec["label"] == 0 else "diffusion",
    }


def organize_images(records: list[dict], out_root: Path, workers: int) -> list[dict]:
    records = assign_output_paths(records)
    metadata = [None] * len(records)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_copy_one, rec, out_root): i
                   for i, rec in enumerate(records)}
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="Copying images"):
            metadata[futures[fut]] = fut.result()
    return [m for m in metadata if m is not None]


# ── Output ─────────────────────────────────────────────────────────────────

def build_metadata_csv(metadata: list[dict], out_root: Path) -> Path:
    df = pd.DataFrame(metadata)
    out_csv = out_root / "synthbuster" / "synthbuster_metadata.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return out_csv


def print_summary(df: pd.DataFrame) -> None:
    print("\n-- SynthBuster summary --------------------------------------")
    print(df.groupby(["split", "architecture", "label"])
            .size().reset_index(name="count").to_string(index=False))
    print(f"\nTotal images : {len(df):,}")


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract and preprocess local SynthBuster archives"
    )
    parser.add_argument("--output-dir", "-o", default="/workspace/data/processed",
                        help="Root output directory (default: /workspace/data/processed)")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--archive",
        action="append",
        type=Path,
        help=(
            "Local ZIP archive to extract; repeat for synthbuster.zip and "
            "real_RAISE_1k.zip"
        ),
    )
    source.add_argument(
        "--source-dir",
        type=Path,
        help="Path to a directory where both archives are already extracted",
    )
    parser.add_argument(
        "--extract-dir",
        type=Path,
        help=(
            "Extraction directory (default: an 'extracted' directory beside "
            "the first archive)"
        ),
    )
    parser.add_argument("--workers", "-j", type=int, default=DEFAULT_WORKERS,
                        help=f"Parallel worker threads (default: {DEFAULT_WORKERS})")
    parser.add_argument("--max-test", type=int, default=DEFAULT_MAX_TEST,
                        help=f"Max test images to keep, balanced real/fake "
                             f"(default: {DEFAULT_MAX_TEST})")
    parser.add_argument("--seed", type=int, default=42,
                        help="Sampling seed (default: 42)")
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    print(f"Max test  : {args.max_test}")
    print(f"Workers   : {args.workers}")

    if args.archive:
        extract_dir = (
            args.extract_dir
            if args.extract_dir
            else args.archive[0].parent / "extracted"
        )
        try:
            source_root = extract_archives(args.archive, extract_dir)
        except (FileNotFoundError, ValueError, zipfile.BadZipFile) as error:
            sys.exit(f"ERROR: {error}")
    else:
        source_root = args.source_dir

    if not source_root.is_dir():
        sys.exit(f"ERROR: source directory not found: {source_root}")

    print(f"Local source : {source_root}")
    records = collect_images(source_root, args.workers)
    print(f"Images found : {len(records):,}")
    if not records:
        sys.exit("ERROR: no recognized SynthBuster images found.")
    try:
        records = sample_records(records, args.max_test, args.seed)
    except ValueError as error:
        sys.exit(f"ERROR: {error}")
    print(f"Images sampled : {len(records):,}  (<={args.max_test} test)")
    metadata = organize_images(records, out_root, args.workers)

    csv_path = build_metadata_csv(metadata, out_root)
    print(f"\nMetadata saved to : {csv_path}")
    print_summary(pd.read_csv(csv_path))


if __name__ == "__main__":
    main()
