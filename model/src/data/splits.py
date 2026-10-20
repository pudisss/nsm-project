"""Generator- and family-aware splitting for OOD generalization testing.

Family-level OOD means all fake images from one generator family (for example
GAN) are used for training while another complete family (for example
diffusion) is reserved for testing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import random


# Architecture type mapping
ARCHITECTURE_TYPES = {
    # GAN models (from CNNDetection)
    'progan': 'gan',
    'stylegan': 'gan',
    'stylegan2': 'gan',
    'biggan': 'gan',
    'cyclegan': 'gan',
    'stargan': 'gan',
    'gaugan': 'gan',
    'deepfake': 'face_manipulation',
    'seeingdark': 'cnn',
    'san': 'cnn',
    'crn': 'cnn',
    'imle': 'implicit_generative',
    'sitd': 'cnn',
    'whichfaceisreal': 'gan',
    'natural': 'real',

    # Diffusion models (from DiffusionForensics / DIRE)
    'ddpm': 'diffusion',
    'iddpm': 'diffusion',
    'adm': 'diffusion',
    'pndm': 'diffusion',
    'ldm': 'diffusion',
    'sdv1': 'diffusion',
    'sdv1_new1': 'diffusion',
    'sdv1_new2': 'diffusion',
    'sdv2': 'diffusion',
    'dalle2': 'diffusion',
    'midjourney': 'diffusion',
    'if': 'diffusion',
    'vqdiffusion': 'diffusion',
    'natural_images': 'real',

    # CIFAKE (Bird & Lotfi 2023 — CIFAR-10 real + Stable Diffusion v1.4 fake)
    'cifar_real': 'real',
    'stable_diffusion_v14': 'diffusion',

    # SynthBuster (Bammey 2023 — photorealistic images from multiple generators)
    'raise_real': 'real',
    'firefly': 'diffusion',
    'dalle3': 'diffusion',
    'glide': 'diffusion',
    'sd_13': 'diffusion',
    'sd_14': 'diffusion',
    'sd_2': 'diffusion',
    'sd_xl': 'diffusion',
    'wukong': 'diffusion',
}


def get_architecture_type(arch: str) -> str:
    """Get the architecture type (gan, diffusion, real) for a model."""
    arch_lower = arch.lower().strip()
    return ARCHITECTURE_TYPES.get(arch_lower, 'unknown')


def add_architecture_family(
    df: pd.DataFrame,
    architecture_col: str = 'architecture',
) -> pd.DataFrame:
    """Add and strictly validate the canonical architecture family column."""
    if architecture_col not in df.columns:
        raise ValueError(f"Missing required column: {architecture_col}")

    result = df.copy()
    inferred = result[architecture_col].astype(str).apply(get_architecture_type)

    if 'architecture_family' in result.columns:
        existing = (
            result['architecture_family'].astype(str).str.lower().str.strip()
        )
        mismatch = (existing != inferred) & (inferred != 'unknown')
        if mismatch.any():
            examples = result.loc[
                mismatch, [architecture_col, 'architecture_family']
            ].head(10)
            raise ValueError(
                "architecture_family disagrees with architecture mapping:\n"
                f"{examples.to_string(index=False)}"
            )

    result['architecture_family'] = inferred
    unknown = sorted(
        result.loc[
            result['architecture_family'] == 'unknown', architecture_col
        ].astype(str).unique()
    )
    if unknown:
        raise ValueError(f"Unknown architecture family for: {unknown}")
    return result


class FamilyAwareSplitter:
    """Split fake images by complete generator family."""

    def __init__(
        self,
        train_families: List[str],
        test_families: List[str],
        exclude_families: Optional[List[str]] = None,
        real_strategy: str = 'family_matched',
        real_test_fraction: float = 0.2,
        seed: int = 42,
    ):
        self.train_families = {
            family.lower().strip() for family in train_families
        }
        self.test_families = {
            family.lower().strip() for family in test_families
        }
        self.exclude_families = {
            family.lower().strip() for family in (exclude_families or [])
        }
        self.real_strategy = real_strategy
        self.real_test_fraction = real_test_fraction
        self.seed = seed

        overlap = self.train_families & self.test_families
        if overlap:
            raise ValueError(
                f"Families appear in both train and test: {sorted(overlap)}"
            )
        if not self.train_families or not self.test_families:
            raise ValueError("Both train_families and test_families are required")
        configured_overlap = (
            (self.train_families | self.test_families)
            & self.exclude_families
        )
        if configured_overlap:
            raise ValueError(
                "Families cannot be both selected and excluded: "
                f"{sorted(configured_overlap)}"
            )
        if real_strategy not in {'family_matched', 'random'}:
            raise ValueError(
                "real_strategy must be 'family_matched' or 'random'"
            )
        if not 0 < real_test_fraction < 1:
            raise ValueError("real_test_fraction must be between 0 and 1")

    def split_dataframe(
        self,
        df: pd.DataFrame,
        family_col: str = 'architecture_family',
        label_col: str = 'label',
        dataset_col: str = 'dataset',
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Return train/test frames with zero fake-family overlap."""
        data = add_architecture_family(df)
        excluded_mask = (
            (data[label_col] == 1)
            & data[family_col].isin(self.exclude_families)
        )
        if excluded_mask.any():
            print(
                f"Excluding {int(excluded_mask.sum()):,} fake rows from "
                f"families {sorted(self.exclude_families)}"
            )
            data = data.loc[~excluded_mask].copy()

        fake = data[data[label_col] == 1]
        observed_fake = set(fake[family_col].unique())
        configured = self.train_families | self.test_families
        unknown = observed_fake - configured
        if unknown:
            raise ValueError(
                "Fake families are not assigned to train or test: "
                f"{sorted(unknown)}"
            )

        assignments = pd.Series(index=data.index, dtype='object')
        fake_train = (data[label_col] == 1) & data[family_col].isin(
            self.train_families
        )
        fake_test = (data[label_col] == 1) & data[family_col].isin(
            self.test_families
        )
        assignments.loc[fake_train] = 'train'
        assignments.loc[fake_test] = 'test'

        real_mask = data[label_col] == 0
        if self.real_strategy == 'random':
            real_indices = data.index[real_mask].to_numpy()
            rng = np.random.default_rng(self.seed)
            rng.shuffle(real_indices)
            split_at = int(len(real_indices) * (1 - self.real_test_fraction))
            assignments.loc[real_indices[:split_at]] = 'train'
            assignments.loc[real_indices[split_at:]] = 'test'
        else:
            if dataset_col not in data.columns:
                raise ValueError(
                    "family_matched real strategy requires a dataset column"
                )
            for dataset, group in data.groupby(dataset_col):
                dataset_fake_families = set(
                    group.loc[group[label_col] == 1, family_col].unique()
                )
                train_match = bool(
                    dataset_fake_families & self.train_families
                )
                test_match = bool(
                    dataset_fake_families & self.test_families
                )
                if train_match == test_match:
                    raise ValueError(
                        f"Dataset {dataset!r} cannot be assigned unambiguously: "
                        f"fake families={sorted(dataset_fake_families)}"
                    )
                destination = 'train' if train_match else 'test'
                assignments.loc[group.index[group[label_col] == 0]] = destination

        if assignments.isna().any():
            raise ValueError(
                f"{int(assignments.isna().sum())} rows were not assigned"
            )

        train_df = data.loc[assignments == 'train'].reset_index(drop=True)
        test_df = data.loc[assignments == 'test'].reset_index(drop=True)
        self.validate_split(train_df, test_df, family_col, label_col)
        return train_df, test_df

    def validate_split(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        family_col: str = 'architecture_family',
        label_col: str = 'label',
    ) -> None:
        """Fail if fake-family OOD isolation is violated."""
        train_fake = set(
            train_df.loc[train_df[label_col] == 1, family_col].unique()
        )
        test_fake = set(
            test_df.loc[test_df[label_col] == 1, family_col].unique()
        )
        if train_fake != self.train_families:
            raise ValueError(
                f"Expected train fake families {sorted(self.train_families)}, "
                f"found {sorted(train_fake)}"
            )
        if test_fake != self.test_families:
            raise ValueError(
                f"Expected test fake families {sorted(self.test_families)}, "
                f"found {sorted(test_fake)}"
            )
        overlap = train_fake & test_fake
        if overlap:
            raise ValueError(
                f"Fake family leakage between train and test: {sorted(overlap)}"
            )


class ModelAwareSplitter:
    """Split by generator while retaining multiple generator families."""
    
    def __init__(
        self,
        train_models: Dict[str, List[str]],
        test_models: Dict[str, List[str]],
        exclude_families: Optional[List[str]] = None,
        real_strategy: str = 'dataset_stratified',
        real_test_fraction: float = 0.2,
        seed: int = 42,
    ):
        self.train_models = {
            family.lower().strip(): {
                model.lower().strip() for model in models
            }
            for family, models in train_models.items()
        }
        self.test_models = {
            family.lower().strip(): {
                model.lower().strip() for model in models
            }
            for family, models in test_models.items()
        }
        self.exclude_families = {
            family.lower().strip() for family in (exclude_families or [])
        }
        self.real_strategy = real_strategy
        self.real_test_fraction = real_test_fraction
        self.seed = seed
        
        self._train_set = set().union(*self.train_models.values())
        self._test_set = set().union(*self.test_models.values())
        
        overlap = self._train_set & self._test_set
        if overlap:
            raise ValueError(
                f"Models appear in both train and test: {sorted(overlap)}"
            )
        if not self._train_set or not self._test_set:
            raise ValueError("Both train_models and test_models are required")
        if real_strategy not in {'dataset_stratified', 'random'}:
            raise ValueError(
                "real_strategy must be 'dataset_stratified' or 'random'"
            )
        if not 0 < real_test_fraction < 1:
            raise ValueError("real_test_fraction must be between 0 and 1")

        expected_families = {'gan', 'diffusion'}
        for split_name, models in (
            ('train', self.train_models),
            ('test', self.test_models),
        ):
            missing = expected_families - {
                family for family, names in models.items() if names
            }
            if missing:
                raise ValueError(
                    f"{split_name}_models must include non-empty GAN and "
                    f"diffusion lists; missing {sorted(missing)}"
                )
    
    def is_train_model(self, model: str) -> bool:
        """Check if model should be used for training."""
        return model.lower().strip() in self._train_set
    
    def is_test_model(self, model: str) -> bool:
        """Check if model should be used for testing."""
        return model.lower().strip() in self._test_set
    
    def split_dataframe(
        self,
        df: pd.DataFrame,
        architecture_col: str = 'architecture',
        label_col: str = 'label',
        family_col: str = 'architecture_family',
        dataset_col: str = 'dataset',
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Return strict generator-disjoint train and test frames."""
        data = add_architecture_family(df, architecture_col)
        data[architecture_col] = (
            data[architecture_col].astype(str).str.lower().str.strip()
        )

        excluded_mask = (
            (data[label_col] == 1)
            & data[family_col].isin(self.exclude_families)
        )
        if excluded_mask.any():
            print(
                f"Excluding {int(excluded_mask.sum()):,} fake rows from "
                f"families {sorted(self.exclude_families)}"
            )
            data = data.loc[~excluded_mask].copy()

        observed_fake = set(
            data.loc[data[label_col] == 1, architecture_col].unique()
        )
        configured_fake = self._train_set | self._test_set
        unassigned = observed_fake - configured_fake
        if unassigned:
            raise ValueError(
                "Fake generators are not assigned to train or test: "
                f"{sorted(unassigned)}"
            )
        missing = configured_fake - observed_fake
        if missing:
            raise ValueError(
                "Configured fake generators are absent from metadata: "
                f"{sorted(missing)}"
            )

        assignments = pd.Series(index=data.index, dtype='object')
        fake_mask = data[label_col] == 1
        assignments.loc[
            fake_mask & data[architecture_col].isin(self._train_set)
        ] = 'train'
        assignments.loc[
            fake_mask & data[architecture_col].isin(self._test_set)
        ] = 'test'

        real = data.loc[data[label_col] == 0]
        if real.empty:
            raise ValueError("No real images are available for splitting")
        if self.real_strategy == 'dataset_stratified':
            if dataset_col not in data.columns:
                raise ValueError(
                    "dataset_stratified real splitting requires a dataset column"
                )
            groups = real.groupby(dataset_col, sort=True)
        else:
            groups = [('all', real)]

        for group_index, (_, group) in enumerate(groups):
            indices = group.index.to_numpy(copy=True)
            rng = np.random.default_rng(self.seed + group_index)
            rng.shuffle(indices)
            test_count = max(
                1, int(round(len(indices) * self.real_test_fraction))
            )
            if test_count >= len(indices):
                raise ValueError(
                    "Each real-image group must contain at least two rows"
                )
            assignments.loc[indices[:-test_count]] = 'train'
            assignments.loc[indices[-test_count:]] = 'test'

        if assignments.isna().any():
            raise ValueError(
                f"{int(assignments.isna().sum())} rows were not assigned"
            )

        train_df = data.loc[assignments == 'train'].reset_index(drop=True)
        test_df = data.loc[assignments == 'test'].reset_index(drop=True)
        self.validate_split(
            train_df,
            test_df,
            architecture_col=architecture_col,
            label_col=label_col,
            family_col=family_col,
        )
        return train_df, test_df

    def validate_split(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        architecture_col: str = 'architecture',
        label_col: str = 'label',
        family_col: str = 'architecture_family',
    ) -> None:
        train_fake = train_df.loc[
            train_df[label_col] == 1, architecture_col
        ]
        test_fake = test_df.loc[
            test_df[label_col] == 1, architecture_col
        ]
        train_architectures = set(train_fake.unique())
        test_architectures = set(test_fake.unique())
        overlap = train_architectures & test_architectures
        if overlap:
            raise ValueError(
                "Fake generator leakage between train and test: "
                f"{sorted(overlap)}"
            )
        if train_architectures != self._train_set:
            raise ValueError(
                f"Expected train generators {sorted(self._train_set)}, "
                f"found {sorted(train_architectures)}"
            )
        if test_architectures != self._test_set:
            raise ValueError(
                f"Expected test generators {sorted(self._test_set)}, "
                f"found {sorted(test_architectures)}"
            )
        for split_name, frame in (('train', train_df), ('test', test_df)):
            families = set(
                frame.loc[frame[label_col] == 1, family_col].unique()
            )
            if families != {'gan', 'diffusion'}:
                raise ValueError(
                    f"{split_name} fake data must contain GAN and diffusion "
                    f"families; found {sorted(families)}"
                )
    
    def get_split_summary(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        architecture_col: str = 'architecture',
        label_col: str = 'label',
    ) -> Dict:
        """Generate summary statistics for the split."""
        summary = {
            'train': {
                'total': len(train_df),
                'real': len(train_df[train_df[label_col] == 0]),
                'fake': len(train_df[train_df[label_col] == 1]),
                'architectures': train_df[architecture_col].value_counts().to_dict(),
            },
            'test': {
                'total': len(test_df),
                'real': len(test_df[test_df[label_col] == 0]),
                'fake': len(test_df[test_df[label_col] == 1]),
                'architectures': test_df[architecture_col].value_counts().to_dict(),
            },
        }
        
        # Check for architecture overlap
        train_archs = set(train_df[train_df[label_col] == 1][architecture_col].unique())
        test_archs = set(test_df[test_df[label_col] == 1][architecture_col].unique())
        summary['overlap'] = list(train_archs & test_archs)
        
        return summary


class BalancedSampler:
    """Handles balanced sampling across classes and architectures."""
    
    def __init__(
        self,
        samples_per_class: Optional[int] = None,
        balance_architectures: bool = True,
        seed: int = 42,
    ):
        self.samples_per_class = samples_per_class
        self.balance_architectures = balance_architectures
        self.seed = seed
    
    def sample(
        self,
        df: pd.DataFrame,
        label_col: str = 'label',
        architecture_col: str = 'architecture',
    ) -> pd.DataFrame:
        """Sample from dataframe with class balancing.
        
        Args:
            df: Input dataframe
            label_col: Column containing labels
            architecture_col: Column containing architecture names
            
        Returns:
            Sampled dataframe
        """
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        # Group by label
        real_df = df[df[label_col] == 0]
        fake_df = df[df[label_col] == 1]
        
        # Determine target count per class
        if self.samples_per_class is not None:
            target_per_class = self.samples_per_class
        else:
            # Use minimum class count for balance
            target_per_class = min(len(real_df), len(fake_df))
        
        # Sample real images
        if len(real_df) <= target_per_class:
            sampled_real = real_df
        else:
            sampled_real = real_df.sample(n=target_per_class, random_state=self.seed)
        
        # Sample fake images (optionally balanced across architectures)
        if self.balance_architectures:
            sampled_fake = self._sample_balanced_architectures(
                fake_df, target_per_class, architecture_col
            )
        else:
            if len(fake_df) <= target_per_class:
                sampled_fake = fake_df
            else:
                sampled_fake = fake_df.sample(n=target_per_class, random_state=self.seed)
        
        # Combine and shuffle
        result = pd.concat([sampled_real, sampled_fake], ignore_index=True)
        result = result.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        
        return result
    
    def _sample_balanced_architectures(
        self,
        df: pd.DataFrame,
        target_count: int,
        architecture_col: str,
    ) -> pd.DataFrame:
        """Sample with balanced representation across architectures."""
        architectures = df[architecture_col].unique()
        n_archs = len(architectures)
        
        if n_archs == 0:
            return df.iloc[:0]
        
        # Target samples per architecture
        per_arch = target_count // n_archs
        remainder = target_count % n_archs
        
        samples = []
        for i, arch in enumerate(architectures):
            arch_df = df[df[architecture_col] == arch]
            n_samples = per_arch + (1 if i < remainder else 0)
            
            if len(arch_df) <= n_samples:
                samples.append(arch_df)
            else:
                samples.append(arch_df.sample(n=n_samples, random_state=self.seed + i))
        
        return pd.concat(samples, ignore_index=True)


def load_and_combine_metadata(
    root_dir: str,
    dataset_configs: List[Dict],
) -> pd.DataFrame:
    """Load and combine metadata from multiple datasets.
    
    Args:
        root_dir: Root directory for processed data
        dataset_configs: List of dataset configuration dicts with
                        'name', 'metadata', and 'architecture_type' keys
    
    Returns:
        Combined DataFrame with unified schema
    """
    root = Path(root_dir)
    dfs = []
    
    for config in dataset_configs:
        metadata_path = root / config['metadata']
        if not metadata_path.exists():
            print(f"Warning: Metadata not found: {metadata_path}")
            continue
        
        df = pd.read_csv(metadata_path)
        df['dataset_name'] = config['name']
        df = add_architecture_family(df)
        df['architecture_type'] = df['architecture_family']
        
        # Ensure filepath is absolute/relative to root
        if 'filepath' in df.columns:
            df['full_path'] = df['filepath'].apply(
                lambda x: str(root / x) if not x.startswith(str(root)) else x
            )
        
        dfs.append(df)
    
    if not dfs:
        raise ValueError("No datasets loaded!")
    
    combined = pd.concat(dfs, ignore_index=True)
    return combined


def load_from_manifest(
    manifest_path: str,
    root_dir: str,
) -> pd.DataFrame:
    """Load data from a manifest CSV file.
    
    Args:
        manifest_path: Path to manifest CSV file
        root_dir: Root directory for image files
        
    Returns:
        DataFrame with image metadata
    """
    manifest = Path(manifest_path)
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
    
    df = pd.read_csv(manifest_path)
    
    # Drop index column if present
    if df.columns[0] in ['', 'Unnamed: 0']:
        df = df.drop(columns=[df.columns[0]])
    
    # Add and validate architecture family/type
    if 'architecture' in df.columns:
        df = add_architecture_family(df)
        df['architecture_type'] = df['architecture_family']
    
    # Ensure full_path is set
    root = Path(root_dir)
    if 'filepath' in df.columns and 'full_path' not in df.columns:
        df['full_path'] = df['filepath'].apply(
            lambda x: str(root / x) if not str(x).startswith('/') else x
        )
    
    print(f"Loaded {len(df)} samples from manifest: {manifest_path}")
    
    return df
