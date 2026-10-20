#!/usr/bin/env python3
"""Generate train and test manifest CSV files from per-dataset metadata.

This script implements the MANIFEST MODE of the training pipeline.

Two pipeline modes
------------------
  Auto mode     Training reads each dataset's metadata CSV directly and
                applies ModelAwareSplitter on-the-fly (controlled by
                dataset_configs + train_models + test_models in config.yaml).

  Manifest mode This script is run ONCE to combine all metadata CSVs into
                a fixed  train_manifest.csv  and  test_manifest.csv.
                Training then uses those frozen files for full reproducibility.

Usage
-----
  # Generate with defaults (reads configs/default.yaml for split assignments)
  python scripts/generate_manifests.py --data-dir /workspace/data/processed

  # Override output location
  python scripts/generate_manifests.py \\
      --data-dir /workspace/data/processed \\
      --output-dir src/data

  # Limit samples per class (for quick experiments)
  python scripts/generate_manifests.py \\
      --data-dir /workspace/data/processed \\
      --samples-per-class 5000

  # Add a dataset that isn't in the default config
  python scripts/generate_manifests.py \\
      --data-dir /workspace/data/processed \\
      --extra-metadata cifake/cifake_metadata.csv \\
      --extra-metadata synthbuster/synthbuster_metadata.csv

Expected metadata CSV columns (produced by download_*.py scripts)
------------------------------------------------------------------
  filepath, split, label, dataset, architecture
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

# Project root (this script lives in scripts/, project root is one level up)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.splits import (
    FamilyAwareSplitter,
    ModelAwareSplitter,
    add_architecture_family,
)

DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs" / "experiments" / "baseline.yaml"
)
DEFAULT_OUTPUT    = PROJECT_ROOT / "src" / "data"
DEFAULT_WORKERS   = 32
DEFAULT_VAL_SPLIT = 0.1
SEED              = 42

# ── Default strict generator assignment ───────────────────────────────────
DEFAULT_TRAIN_MODELS = {
    "gan": ["progan", "biggan", "cyclegan", "stargan"],
    "diffusion": [
        "dalle2", "glide", "midjourney", "sd_13", "sd_14", "sd_2",
    ],
}
DEFAULT_TEST_MODELS = {
    "gan": ["stylegan", "stylegan2", "gaugan", "whichfaceisreal"],
    "diffusion": ["dalle3", "firefly", "sd_xl"],
}
# ───────────────────────────────────────────────────────────────────────────


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ── Load individual metadata CSVs in parallel ─────────────────────────────

def _load_one_csv(args: tuple) -> pd.DataFrame | None:
    """Load and validate a single metadata CSV (runs in thread pool)."""
    csv_path, data_root = args
    full_path = data_root / csv_path
    if not full_path.exists():
        print(f"  [WARN] Not found: {full_path}")
        return None
    df = pd.read_csv(full_path)
    # Drop unnamed index columns
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]
    required = {"filepath", "label", "architecture"}
    missing = required - set(df.columns)
    if missing:
        print(f"  [WARN] {csv_path} missing columns: {missing} - skipping")
        return None
    if "dataset" not in df.columns:
        df["dataset"] = full_path.parent.name
    if "split" not in df.columns:
        df["split"] = "unknown"
    if "representation" in df.columns:
        representation = (
            df["representation"].fillna("source")
            .astype(str).str.lower().str.strip()
        )
        excluded = representation != "source"
        if excluded.any():
            print(
                f"  [WARN] Excluding {int(excluded.sum()):,} non-source "
                f"representation rows from {csv_path}"
            )
            df = df.loc[~excluded].copy()
        if df.empty:
            print(
                f"  [WARN] {csv_path} contains no source-image rows - skipping"
            )
            return None
    # Normalise
    df["architecture"] = df["architecture"].astype(str).str.lower().str.strip()
    df["label"]        = df["label"].astype(int)
    df = add_architecture_family(df)
    df["full_csv_path"] = str(full_path)
    print(f"  Loaded {len(df):>7,} rows <- {full_path.relative_to(data_root)}")
    return df


def load_all_metadata(metadata_paths: list[str], data_root: Path, workers: int) -> pd.DataFrame:
    print(f"Loading {len(metadata_paths)} metadata file(s) with {workers} workers...")
    args_list = [(p, data_root) for p in metadata_paths]
    dfs = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_load_one_csv, a): a for a in args_list}
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="Reading CSVs"):
            df = fut.result()
            if df is not None:
                dfs.append(df)
    if not dfs:
        sys.exit("ERROR: No valid metadata files loaded.")
    combined = pd.concat(dfs, ignore_index=True)
    combined.drop(columns=["full_csv_path"], inplace=True, errors="ignore")
    return combined


# ── Split assignment ───────────────────────────────────────────────────────

# ── Optional balancing ─────────────────────────────────────────────────────

def balance_classes(df: pd.DataFrame, samples_per_class: int, seed: int) -> pd.DataFrame:
    """Select exact class counts, balancing fake samples across architectures."""
    real_df = df[df["label"] == 0]
    fake_df = df[df["label"] == 1]

    if len(real_df) < samples_per_class:
        raise ValueError(
            f"Need {samples_per_class:,} real training samples, "
            f"but only {len(real_df):,} are available"
        )
    real_df = real_df.sample(n=samples_per_class, random_state=seed)

    archs = sorted(fake_df["architecture"].unique())
    if not archs:
        raise ValueError("No fake training architectures are available")
    per_arch = samples_per_class // len(archs)
    remainder = samples_per_class % len(archs)
    parts = []
    for i, arch in enumerate(archs):
        arch_df = fake_df[fake_df["architecture"] == arch]
        n = per_arch + (1 if i < remainder else 0)
        if len(arch_df) < n:
            raise ValueError(
                f"Need {n:,} training samples for {arch!r}, "
                f"but only {len(arch_df):,} are available"
            )
        parts.append(arch_df.sample(n=n, random_state=seed + i))
    fake_df = pd.concat(parts, ignore_index=True)

    combined = pd.concat([real_df, fake_df], ignore_index=True)
    return combined.sample(frac=1, random_state=seed).reset_index(drop=True)


def balance_test_by_dataset_class(
    df: pd.DataFrame,
    samples_per_dataset_class: int,
    seed: int,
) -> pd.DataFrame:
    """Select equal real/fake counts from every test dataset."""
    parts = []
    for dataset_index, dataset in enumerate(sorted(df["dataset"].unique())):
        dataset_df = df[df["dataset"] == dataset]
        for label in (0, 1):
            group = dataset_df[dataset_df["label"] == label]
            if len(group) < samples_per_dataset_class:
                raise ValueError(
                    f"Need {samples_per_dataset_class:,} label={label} rows "
                    f"for test dataset {dataset!r}, but only {len(group):,} "
                    "are available"
                )
            parts.append(
                group.sample(
                    n=samples_per_dataset_class,
                    random_state=seed + dataset_index * 2 + label,
                )
            )
    return (
        pd.concat(parts, ignore_index=True)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )


def balance_equal_per_generator(
    df: pd.DataFrame,
    seed: int,
    samples_per_generator: int | None = None,
) -> pd.DataFrame:
    """Balance fake generators equally, then match the real-image count."""
    fake_df = df[df["label"] == 1]
    real_df = df[df["label"] == 0]
    architectures = sorted(fake_df["architecture"].unique())
    if not architectures:
        raise ValueError("No fake generators are available for balancing")

    available = fake_df["architecture"].value_counts()
    quota = samples_per_generator or int(available.min())
    if quota <= 0:
        raise ValueError("samples_per_generator must be positive")

    fake_parts = []
    for index, architecture in enumerate(architectures):
        group = fake_df[fake_df["architecture"] == architecture]
        if len(group) < quota:
            raise ValueError(
                f"Need {quota:,} samples for {architecture!r}, "
                f"but only {len(group):,} are available"
            )
        fake_parts.append(
            group.sample(n=quota, random_state=seed + index)
        )
    balanced_fake = pd.concat(fake_parts, ignore_index=True)

    if len(real_df) < len(balanced_fake):
        raise ValueError(
            f"Need {len(balanced_fake):,} real images to match fake images, "
            f"but only {len(real_df):,} are available"
        )
    balanced_real = real_df.sample(
        n=len(balanced_fake),
        random_state=seed + len(architectures),
    )
    return (
        pd.concat([balanced_real, balanced_fake], ignore_index=True)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )


# ── Validate integrity ────────────────────────────────────────────────────

def validate_manifests(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    split_level: str,
) -> None:
    if split_level == "family":
        train_families = set(
            train_df.loc[
                train_df["label"] == 1, "architecture_family"
            ].unique()
        )
        test_families = set(
            test_df.loc[
                test_df["label"] == 1, "architecture_family"
            ].unique()
        )
        overlap = train_families & test_families
        if overlap:
            raise ValueError(
                f"Fake family leakage between train and test: {sorted(overlap)}"
            )
        print(
            "  [OK] Family-level OOD isolation: "
            f"train={sorted(train_families)}, "
            f"test={sorted(test_families)}"
        )
        return

    train_fake_archs = set(train_df[train_df["label"] == 1]["architecture"].unique())
    test_fake_archs  = set(test_df[test_df["label"] == 1]["architecture"].unique())
    overlap = train_fake_archs & test_fake_archs
    if overlap:
        raise ValueError(
            "Fake generator leakage between train and test: "
            f"{sorted(overlap)}"
        )
    for name, frame in (("train", train_df), ("test", test_df)):
        families = set(
            frame.loc[frame["label"] == 1, "architecture_family"].unique()
        )
        if families != {"gan", "diffusion"}:
            raise ValueError(
                f"{name} fake data must contain GAN and diffusion families; "
                f"found {sorted(families)}"
            )
    print("  [OK] Generator-disjoint cross-architecture split.")


# ── Save manifests in parallel ────────────────────────────────────────────

def _write_csv(args: tuple) -> None:
    df, path = args
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_manifests(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
    workers: int,
    experiment_name: str = "",
) -> tuple[Path, Path]:
    prefix = f"{experiment_name}_" if experiment_name else ""
    train_path = output_dir / f"{prefix}train_manifest.csv"
    test_path  = output_dir / f"{prefix}test_manifest.csv"

    with ThreadPoolExecutor(max_workers=min(2, workers)) as ex:
        futs = [
            ex.submit(_write_csv, (train_df, train_path)),
            ex.submit(_write_csv, (test_df,  test_path)),
        ]
        for fut in futs:
            fut.result()

    return train_path, test_path


# ── Summary ───────────────────────────────────────────────────────────────

def print_summary(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    for name, df in [("TRAIN", train_df), ("TEST", test_df)]:
        print(f"\n-- {name} manifest --------------------------------------")
        print(df.groupby(["dataset", "architecture", "label"])
                .size().reset_index(name="count").to_string(index=False))
        real = (df["label"] == 0).sum()
        fake = (df["label"] == 1).sum()
        print(f"  Total: {len(df):,}  (real: {real:,}, fake: {fake:,})")


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate train/test manifest CSVs from per-dataset metadata"
    )
    parser.add_argument("--data-dir", "-d", required=True,
                        help="Root directory containing per-dataset folders "
                             "(e.g. /workspace/data/processed)")
    parser.add_argument("--output-dir", "-o", default=str(DEFAULT_OUTPUT),
                        help=f"Where to write manifest CSVs (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--config", "-c", default=str(DEFAULT_CONFIG),
                        help=f"Path to config YAML (default: {DEFAULT_CONFIG})")
    parser.add_argument("--extra-metadata", action="append", default=[],
                        metavar="REL_PATH",
                        help="Additional metadata CSV relative to --data-dir "
                             "(can be repeated, e.g. --extra-metadata cifake/cifake_metadata.csv)")
    parser.add_argument("--samples-per-class", type=int, default=None,
                        help="Require this many real and fake TRAIN samples; "
                             "fake samples are balanced across architectures")
    parser.add_argument(
        "--test-samples-per-dataset-class",
        type=int,
        default=None,
        help=(
            "Require this many real and fake samples from every test dataset "
            "for the primary test manifest"
        ),
    )
    parser.add_argument(
        "--write-full-test",
        action="store_true",
        help="Also write the complete unbalanced test manifest",
    )
    parser.add_argument("--val-split", type=float, default=DEFAULT_VAL_SPLIT,
                        help=f"Fraction of train set to reserve for validation "
                             f"(default: {DEFAULT_VAL_SPLIT}; informational only - "
                             "MultiDomainDataModule handles this at runtime)")
    parser.add_argument("--seed", type=int, default=SEED,
                        help=f"Random seed (default: {SEED})")
    parser.add_argument("--workers", "-j", type=int, default=DEFAULT_WORKERS,
                        help=f"Parallel worker threads (default: {DEFAULT_WORKERS})")
    parser.add_argument("--train-family", action="append", default=None,
                        choices=("gan", "diffusion"),
                        help="Fake family used for training; repeat if needed")
    parser.add_argument("--test-family", action="append", default=None,
                        choices=("gan", "diffusion"),
                        help="Fake family reserved for OOD testing")
    parser.add_argument("--exclude-family", action="append", default=None,
                        help="Fake family excluded from this experiment")
    parser.add_argument("--real-strategy",
                        choices=("family_matched", "dataset_stratified", "random"),
                        default=None,
                        help="How real images are assigned between domains")
    parser.add_argument("--experiment-name", default=None,
                        help="Prefix for generated manifest filenames")
    args = parser.parse_args()

    data_root  = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    print(f"Data root  : {data_root}")
    print(f"Output dir : {output_dir}")
    print(f"Workers    : {args.workers}")

    # Collect metadata file paths from config
    metadata_paths: list[str] = []
    cfg: dict = {}
    if Path(args.config).exists():
        cfg = load_config(Path(args.config))
        for ds in cfg.get("data", {}).get("datasets", []):
            if "metadata" in ds:
                metadata_paths.append(ds["metadata"])
    else:
        print(f"[WARN] Config not found at {args.config} - using defaults only")
        # Defaults use datasets with named GAN and diffusion generators.
        metadata_paths = [
            "cnndetection/cnndetection_metadata.csv",
            "synthbuster/synthbuster_metadata.csv",
        ]

    # Add any extra paths
    for extra in args.extra_metadata:
        if extra not in metadata_paths:
            metadata_paths.append(extra)

    # Load and combine
    combined_df = load_all_metadata(metadata_paths, data_root, args.workers)
    print(f"\nTotal rows loaded : {len(combined_df):,}")

    # Assign train / test splits
    print("\nAssigning train/test splits...")
    split_cfg = cfg.get("data", {}).get("split", {})
    split_level = split_cfg.get("level", "model")
    if args.train_family or args.test_family:
        split_level = "family"

    if split_level == "family":
        train_families = (
            args.train_family
            or split_cfg.get("train_families", ["gan"])
        )
        test_families = (
            args.test_family
            or split_cfg.get("test_families", ["diffusion"])
        )
        real_strategy = (
            args.real_strategy
            or split_cfg.get("real_strategy", "family_matched")
        )
        exclude_families = (
            args.exclude_family
            if args.exclude_family is not None
            else split_cfg.get("exclude_families", [])
        )
        splitter = FamilyAwareSplitter(
            train_families=train_families,
            test_families=test_families,
            exclude_families=exclude_families,
            real_strategy=real_strategy,
            real_test_fraction=split_cfg.get("real_test_fraction", 0.2),
            seed=args.seed,
        )
        train_df, test_df = splitter.split_dataframe(combined_df)
        experiment_name = (
            args.experiment_name
            or split_cfg.get("experiment_name")
            or f"{'_'.join(train_families)}_to_{'_'.join(test_families)}"
        )
        print(
            f"Family OOD experiment: {train_families} -> {test_families} "
            f"(real_strategy={real_strategy})"
        )
    else:
        train_models_cfg = (
            cfg.get("data", {}).get("train_models")
            or DEFAULT_TRAIN_MODELS
        )
        test_models_cfg = (
            cfg.get("data", {}).get("test_models")
            or DEFAULT_TEST_MODELS
        )
        exclude_families = (
            args.exclude_family
            if args.exclude_family is not None
            else split_cfg.get("exclude_families", [])
        )
        real_strategy = (
            args.real_strategy
            or split_cfg.get("real_strategy", "dataset_stratified")
        )
        splitter = ModelAwareSplitter(
            train_models=train_models_cfg,
            test_models=test_models_cfg,
            exclude_families=exclude_families,
            real_strategy=real_strategy,
            real_test_fraction=split_cfg.get("real_test_fraction", 0.2),
            seed=args.seed,
        )
        train_df, test_df = splitter.split_dataframe(combined_df)
        experiment_name = (
            args.experiment_name
            or split_cfg.get("experiment_name")
            or "cross_architecture"
        )
        print(
            "Cross-architecture generator OOD experiment "
            f"(real_strategy={real_strategy})"
        )

    full_test_df = test_df.copy()

    if split_level == "model":
        print("Balancing train and test equally per fake generator...")
        train_df = balance_equal_per_generator(train_df, args.seed)
        test_df = balance_equal_per_generator(test_df, args.seed)
    elif args.samples_per_class:
        print(f"Balancing train to {args.samples_per_class:,} samples/class...")
        train_df = balance_classes(train_df, args.samples_per_class, args.seed)

    if split_level == "family" and args.test_samples_per_dataset_class:
        count = args.test_samples_per_dataset_class
        print(
            f"Balancing test to {count:,} samples per dataset/class..."
        )
        test_df = balance_test_by_dataset_class(test_df, count, args.seed)

    # Validate
    print("Validating manifest integrity...")
    validate_manifests(train_df, test_df, split_level)
    if args.write_full_test:
        validate_manifests(train_df, full_test_df, split_level)

    # Save
    print("Saving manifests...")
    train_path, test_path = save_manifests(
        train_df,
        test_df,
        output_dir,
        args.workers,
        experiment_name,
    )

    print(f"\nTrain manifest : {train_path}  ({len(train_df):,} rows)")
    print(f"Test  manifest : {test_path}  ({len(test_df):,} rows)")
    if args.write_full_test:
        prefix = f"{experiment_name}_" if experiment_name else ""
        full_test_path = output_dir / f"{prefix}full_test_manifest.csv"
        _write_csv((full_test_df, full_test_path))
        print(
            f"Full test manifest : {full_test_path}  "
            f"({len(full_test_df):,} rows)"
        )

    print_summary(train_df, test_df)

    print(
        f"\nDone. To use these manifests in training:\n"
        f"  python scripts/train.py "
        f"--train-manifest {train_path} --test-manifest {test_path} "
        f"--root-dir {data_root}"
    )


if __name__ == "__main__":
    main()
