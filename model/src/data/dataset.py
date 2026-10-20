"""Dataset classes for multi-domain AI image detection.

This module provides PyTorch Dataset classes that load images and
extract features from multiple domains (RGB, frequency, noise).
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
from PIL import Image
from typing import Dict, List, Optional, Tuple, Callable, Any
from collections import Counter
from sklearn.model_selection import train_test_split

from .transforms import (
    RGBTransform,
    FrequencyTransform,
    NoiseTransform,
    MultiDomainTransform,
    AugmentationTransform,
)
from .splits import (
    FamilyAwareSplitter,
    ModelAwareSplitter,
    BalancedSampler,
    load_and_combine_metadata,
)


def split_train_validation(
    train_df: pd.DataFrame,
    val_split: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split training data while preserving fake architecture representation."""
    data = train_df.reset_index(drop=True)
    strata = data.apply(
        lambda row: (
            f"fake::{row['architecture']}"
            if int(row["label"]) == 1
            else "real"
        ),
        axis=1,
    )

    try:
        train, val = train_test_split(
            data,
            test_size=val_split,
            random_state=seed,
            stratify=strata,
        )
    except ValueError:
        try:
            train, val = train_test_split(
                data,
                test_size=val_split,
                random_state=seed,
                stratify=data["label"],
            )
        except ValueError:
            train, val = train_test_split(
                data,
                test_size=val_split,
                random_state=seed,
                shuffle=True,
                stratify=None,
            )
    return train.reset_index(drop=True), val.reset_index(drop=True)


class MultiDomainDataset(Dataset):
    """Dataset that extracts RGB, frequency, and noise domain features.
    
    This dataset loads images and applies transforms to extract features
    from three complementary domains for AI-generated image detection.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        root_dir: str,
        image_size: int = 256,
        rgb_mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        rgb_std: Tuple[float, ...] = (0.229, 0.224, 0.225),
        augment: bool = False,
        augment_transform: Optional[Callable] = None,
        return_metadata: bool = False,
    ):
        """
        Args:
            df: DataFrame with 'filepath', 'label', 'architecture' columns
            root_dir: Root directory for image files
            image_size: Target image size
            rgb_mean: Normalization mean for RGB
            rgb_std: Normalization std for RGB
            augment: Whether to apply augmentation (for training)
            augment_transform: Custom augmentation transform
            return_metadata: Whether to return metadata dict with each sample
        """
        self.df = df.reset_index(drop=True)
        self.root_dir = Path(root_dir)
        self.return_metadata = return_metadata
        
        # Domain transforms
        self.rgb_transform = RGBTransform(image_size, rgb_mean, rgb_std)
        self.freq_transform = FrequencyTransform(image_size)
        self.noise_transform = NoiseTransform(image_size)
        
        # Augmentation
        self.augment = augment
        if augment and augment_transform is None:
            self.augment_transform = AugmentationTransform(image_size)
        else:
            self.augment_transform = augment_transform
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        
        # Load image
        if 'full_path' in row:
            img_path = row['full_path']
        else:
            img_path = self.root_dir / row['filepath']
        
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            # Return a blank image on error (shouldn't happen with clean data)
            print(f"Error loading {img_path}: {e}")
            img = Image.new('RGB', (256, 256), color='gray')
        
        # Apply augmentation
        if self.augment and self.augment_transform is not None:
            img = self.augment_transform(img)
        
        # Extract domain features
        rgb = self.rgb_transform(img)
        freq = self.freq_transform(img)
        noise = self.noise_transform(img)
        
        # Label
        label = torch.tensor(row['label'], dtype=torch.long)
        
        result = {
            'rgb': rgb,
            'freq': freq,
            'noise': noise,
            'label': label,
        }
        
        if self.return_metadata:
            result['metadata'] = {
                'filepath': str(img_path),
                'architecture': row.get('architecture', 'unknown'),
                'dataset': row.get('dataset', 'unknown'),
            }
        
        return result
    
    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for balanced training."""
        counts = Counter(self.df['label'])
        total = sum(counts.values())
        weights = {cls: total / count for cls, count in counts.items()}
        
        # Normalize
        max_weight = max(weights.values())
        weights = {cls: w / max_weight for cls, w in weights.items()}
        
        return torch.tensor([weights.get(i, 1.0) for i in range(2)])
    
    def get_sample_weights(self) -> torch.Tensor:
        """Get per-sample weights for WeightedRandomSampler."""
        class_weights = self.get_class_weights()
        sample_weights = torch.tensor([
            class_weights[label].item() for label in self.df['label']
        ])
        return sample_weights


class MultiDomainDataModule:
    """Data module that handles dataset creation, splitting, and loading.
    
    Provides a high-level interface for creating train/val/test dataloaders
    with proper splitting and balancing. Supports both automatic splitting
    and loading from pre-defined manifest files.
    """
    
    def __init__(
        self,
        root_dir: str,
        # Option 1: Use manifest files (takes priority if provided)
        train_manifest: Optional[str] = None,
        val_manifest: Optional[str] = None,
        test_manifest: Optional[str] = None,
        # Option 2: Automatic splitting from dataset configs
        dataset_configs: Optional[List[Dict]] = None,
        train_models: Optional[Dict[str, List[str]]] = None,
        test_models: Optional[Dict[str, List[str]]] = None,
        split_level: str = "model",
        train_families: Optional[List[str]] = None,
        test_families: Optional[List[str]] = None,
        exclude_families: Optional[List[str]] = None,
        real_strategy: str = "dataset_stratified",
        real_test_fraction: float = 0.2,
        # Common params
        image_size: int = 256,
        batch_size: int = 32,
        num_workers: int = 8,
        pin_memory: bool = True,
        balance_classes: bool = True,
        samples_per_class: Optional[int] = None,
        val_split: float = 0.1,
        seed: int = 42,
        weighted_sampling: bool = True,
        augmentation_config: Optional[Dict[str, Any]] = None,
    ):
        self.root_dir = root_dir
        
        # Manifest paths
        self.train_manifest = train_manifest
        self.val_manifest = val_manifest
        self.test_manifest = test_manifest
        
        # Auto-splitting config
        self.dataset_configs = dataset_configs
        self.train_models = train_models
        self.test_models = test_models
        self.split_level = split_level
        self.train_families = train_families
        self.test_families = test_families
        self.exclude_families = exclude_families
        self.real_strategy = real_strategy
        self.real_test_fraction = real_test_fraction
        
        # Data params
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.balance_classes = balance_classes
        self.samples_per_class = samples_per_class
        self.val_split = val_split
        self.seed = seed
        self.weighted_sampling = weighted_sampling
        self.augmentation_config = augmentation_config or {}
        
        # Will be populated by setup()
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _normalize_architectures(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize architecture names for consistent validation/comparison."""
        if 'architecture' in df.columns:
            df = df.copy()
            df['architecture'] = df['architecture'].astype(str).str.lower().str.strip()
        return df

    def _validate_manifest_integrity(self):
        """Validate manifest split quality for generalization experiments."""
        if 'architecture' not in self.train_df.columns or 'architecture' not in self.test_df.columns:
            raise ValueError("Manifest files must include an 'architecture' column.")

        train_fake_arch = self.train_df[self.train_df['label'] == 1]['architecture']
        test_fake_arch = self.test_df[self.test_df['label'] == 1]['architecture']

        unknown_train = train_fake_arch[train_fake_arch.str.startswith('unknown')]
        unknown_test = test_fake_arch[test_fake_arch.str.startswith('unknown')]
        if len(unknown_train) > 0 or len(unknown_test) > 0:
            raise ValueError(
                "Unknown fake architectures detected in manifests. "
                f"train_unknown={len(unknown_train)}, test_unknown={len(unknown_test)}. "
                "Please clean manifests before training."
            )

        # In evaluation mode, train/test can intentionally point to the same manifest.
        if Path(self.train_manifest).resolve() == Path(self.test_manifest).resolve():
            return

        comparison_col = (
            'architecture_family'
            if self.split_level == 'family'
            else 'architecture'
        )
        if comparison_col not in self.train_df.columns:
            raise ValueError(
                f"Manifest files must include or infer {comparison_col!r}."
            )
        train_values = set(
            self.train_df.loc[
                self.train_df['label'] == 1, comparison_col
            ].unique()
        )
        test_values = set(
            self.test_df.loc[
                self.test_df['label'] == 1, comparison_col
            ].unique()
        )
        overlap = sorted(train_values & test_values)
        if overlap:
            raise ValueError(
                f"Manifest integrity check failed: fake {comparison_col} "
                "overlap between train and test. "
                f"Overlap: {overlap}"
            )

        if self.split_level == 'family':
            expected_train = {
                family.lower().strip()
                for family in (self.train_families or [])
            }
            expected_test = {
                family.lower().strip()
                for family in (self.test_families or [])
            }
            if expected_train and train_values != expected_train:
                raise ValueError(
                    "Training manifest has the wrong fake family direction: "
                    f"expected {sorted(expected_train)}, "
                    f"found {sorted(train_values)}"
                )
            if expected_test and test_values != expected_test:
                raise ValueError(
                    "Test manifest has the wrong fake family direction: "
                    f"expected {sorted(expected_test)}, "
                    f"found {sorted(test_values)}"
                )
        else:
            expected_train = {
                model.lower().strip()
                for models in (self.train_models or {}).values()
                for model in models
            }
            expected_test = {
                model.lower().strip()
                for models in (self.test_models or {}).values()
                for model in models
            }
            if expected_train and train_values != expected_train:
                raise ValueError(
                    "Training manifest has the wrong fake generators: "
                    f"expected {sorted(expected_train)}, "
                    f"found {sorted(train_values)}"
                )
            if expected_test and test_values != expected_test:
                raise ValueError(
                    "Test manifest has the wrong fake generators: "
                    f"expected {sorted(expected_test)}, "
                    f"found {sorted(test_values)}"
                )
            for name, frame in (
                ("training", self.train_df),
                ("test", self.test_df),
            ):
                families = set(
                    frame.loc[
                        frame['label'] == 1, 'architecture_family'
                    ].unique()
                )
                if families != {'gan', 'diffusion'}:
                    raise ValueError(
                        f"{name.capitalize()} manifest must contain GAN and "
                        f"diffusion fake generators; found {sorted(families)}"
                    )

    def _augmentation_value(self, name: str, key: str, default: Any) -> Any:
        """Read augmentation settings from nested or legacy flat config."""
        transform_cfg = self.augmentation_config.get('transforms', {}).get(name, {})
        if key == 'p':
            legacy_key = f"p_{name}"
            if 'enabled' in transform_cfg and not transform_cfg.get('enabled', True):
                return 0.0
            return transform_cfg.get('p', self.augmentation_config.get(legacy_key, default))
        legacy_keys = {
            ('color', 'factor'): 'color_jitter_factor',
            ('blur', 'sigma_range'): 'blur_sigma_range',
            ('jpeg', 'quality_range'): 'jpeg_quality_range',
            ('resize_artifact', 'scale_range'): 'resize_scale_range',
            ('noise', 'std_range'): 'noise_std_range',
        }
        legacy_key = legacy_keys.get((name, key), key)
        return transform_cfg.get(key, self.augmentation_config.get(legacy_key, default))
    
    def setup(self):
        """Load and prepare datasets."""
        # Option 1: Use manifest files if provided
        if self.train_manifest and self.test_manifest:
            print("Loading data from manifest files...")
            from .splits import load_from_manifest
            
            self.train_df = load_from_manifest(self.train_manifest, self.root_dir)
            self.test_df = load_from_manifest(self.test_manifest, self.root_dir)
            self.train_df = self._normalize_architectures(self.train_df)
            self.test_df = self._normalize_architectures(self.test_df)
            self._validate_manifest_integrity()

            # Always create validation split from training manifest (stratified by label)
            print(
                f"Creating {self.val_split*100:.0f}% validation split from training manifest "
                f"(stratified by label, seed={self.seed})..."
            )
            train_source_df = self.train_df.reset_index(drop=True)
            val_size = int(len(train_source_df) * self.val_split)

            if val_size == 0:
                raise ValueError(
                    "Training manifest too small for validation split. "
                    "Increase data size or val_split."
                )

            train_df, val_df = split_train_validation(
                train_source_df,
                self.val_split,
                self.seed,
            )

            self.train_df = train_df.reset_index(drop=True)
            self.val_df = val_df.reset_index(drop=True)
            
            print(f"Manifest loading complete:")
            print(f"  Train: {len(self.train_df)} images "
                  f"(real: {len(self.train_df[self.train_df['label']==0])}, "
                  f"fake: {len(self.train_df[self.train_df['label']==1])})")
            print(f"  Val: {len(self.val_df)} images "
                  f"(real: {len(self.val_df[self.val_df['label']==0])}, "
                  f"fake: {len(self.val_df[self.val_df['label']==1])})")
            print(f"  Test: {len(self.test_df)} images "
                  f"(real: {len(self.test_df[self.test_df['label']==0])}, "
                  f"fake: {len(self.test_df[self.test_df['label']==1])})")
        
        # Option 2: Automatic splitting from dataset configs
        else:
            print(f"Using automatic {self.split_level}-level splitting...")
            if not self.dataset_configs:
                raise ValueError(
                    "Either provide manifest files (train_manifest, test_manifest) "
                    "or dataset configs"
                )
            
            # Load combined metadata
            combined_df = load_and_combine_metadata(self.root_dir, self.dataset_configs)
            
            if self.split_level == "family":
                splitter = FamilyAwareSplitter(
                    train_families=self.train_families or [],
                    test_families=self.test_families or [],
                    exclude_families=self.exclude_families or [],
                    real_strategy=self.real_strategy,
                    real_test_fraction=self.real_test_fraction,
                    seed=self.seed,
                )
            elif self.split_level == "model":
                if not self.train_models or not self.test_models:
                    raise ValueError(
                        "Model-level splitting requires train_models and "
                        "test_models"
                    )
                splitter = ModelAwareSplitter(
                    train_models=self.train_models,
                    test_models=self.test_models,
                    exclude_families=self.exclude_families or [],
                    real_strategy=self.real_strategy,
                    real_test_fraction=self.real_test_fraction,
                    seed=self.seed,
                )
            else:
                raise ValueError(
                    "split_level must be 'family' or 'model'"
                )
            train_df, test_df = splitter.split_dataframe(combined_df)
            
            # Print split summary
            print(f"Train: {len(train_df)} images "
                  f"(real: {(train_df['label'] == 0).sum()}, "
                  f"fake: {(train_df['label'] == 1).sum()})")
            print(f"Test: {len(test_df)} images "
                  f"(real: {(test_df['label'] == 0).sum()}, "
                  f"fake: {(test_df['label'] == 1).sum()})")
            
            # Balance training data
            if self.balance_classes:
                sampler = BalancedSampler(
                    samples_per_class=self.samples_per_class,
                    balance_architectures=True,
                    seed=self.seed,
                )
                train_df = sampler.sample(train_df)
                print(f"After balancing: {len(train_df)} training images")
            
            # Create validation split from training data
            self.train_df, self.val_df = split_train_validation(
                train_df,
                self.val_split,
                self.seed,
            )
            self.test_df = test_df.reset_index(drop=True)
            
            print(f"Final splits - Train: {len(self.train_df)}, "
                  f"Val: {len(self.val_df)}, Test: {len(self.test_df)}")
        
        # Create datasets
        self.train_dataset = MultiDomainDataset(
            df=self.train_df,
            root_dir=self.root_dir,
            image_size=self.image_size,
            augment=True,
            augment_transform=AugmentationTransform(
                size=self.image_size,
                p_flip=self._augmentation_value('flip', 'p', 0.5),
                p_rotate=self._augmentation_value('rotate', 'p', 0.3),
                p_color=self._augmentation_value('color', 'p', 0.3),
                p_blur=self._augmentation_value('blur', 'p', 0.2),
                p_jpeg=self._augmentation_value('jpeg', 'p', 0.3),
                p_resize_artifact=self._augmentation_value('resize_artifact', 'p', 0.4),
                p_noise=self._augmentation_value('noise', 'p', 0.0),
                color_jitter_factor=self._augmentation_value('color', 'factor', 0.2),
                blur_sigma_range=tuple(self._augmentation_value('blur', 'sigma_range', [0.3, 2.0])),
                jpeg_quality_range=tuple(self._augmentation_value('jpeg', 'quality_range', [45, 95])),
                resize_scale_range=tuple(self._augmentation_value('resize_artifact', 'scale_range', [0.45, 0.9])),
                noise_std_range=tuple(self._augmentation_value('noise', 'std_range', [1.0, 6.0])),
            ),
            return_metadata=False,
        )
        
        self.val_dataset = MultiDomainDataset(
            df=self.val_df,
            root_dir=self.root_dir,
            image_size=self.image_size,
            augment=False,
            return_metadata=True,
        )
        
        self.test_dataset = MultiDomainDataset(
            df=self.test_df,
            root_dir=self.root_dir,
            image_size=self.image_size,
            augment=False,
            return_metadata=True,  # For per-model evaluation
        )
    
    def train_dataloader(self, weighted_sampling: Optional[bool] = None) -> DataLoader:
        """Create training dataloader."""
        if weighted_sampling is None:
            weighted_sampling = self.weighted_sampling
        if weighted_sampling and self.balance_classes:
            sample_weights = self.train_dataset.get_sample_weights()
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )
            shuffle = False
        else:
            sampler = None
            shuffle = True
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )
    
    def test_dataloader(self) -> DataLoader:
        """Create test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )
    
    def get_test_by_architecture(self) -> Dict[str, DataLoader]:
        """Create separate test dataloaders for each architecture."""
        dataloaders = {}
        
        for arch in self.test_df['architecture'].unique():
            arch_df = self.test_df[self.test_df['architecture'] == arch]
            
            if len(arch_df) == 0:
                continue
            
            dataset = MultiDomainDataset(
                df=arch_df,
                root_dir=self.root_dir,
                image_size=self.image_size,
                augment=False,
                return_metadata=True,
            )
            
            dataloaders[arch] = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=collate_fn,
            )
        
        return dataloaders


def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """Custom collate function for multi-domain batches."""
    rgb = torch.stack([item['rgb'] for item in batch])
    freq = torch.stack([item['freq'] for item in batch])
    noise = torch.stack([item['noise'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    
    result = {
        'rgb': rgb,
        'freq': freq,
        'noise': noise,
        'label': labels,
    }
    
    # Include metadata if present
    if 'metadata' in batch[0]:
        result['metadata'] = [item['metadata'] for item in batch]
    
    return result
