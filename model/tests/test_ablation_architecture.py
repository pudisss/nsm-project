import unittest

import torch

from src.models.detector import (
    DomainAdapter,
    DomainInconsistencyModule,
    DomainMaskBlock,
    FeatureNormBlock,
    L2NormBlock,
    create_detector,
)
from src.models.fusion import create_fusion_module
from src.models.branches import SwinModel
from src.utils.config import Config


class AblationArchitectureTest(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.feature_dims = {
            'rgb': 512,
            'freq': 256,
            'noise': 256,
        }
        self.features = {
            domain: torch.randn(self.batch_size, dim)
            for domain, dim in self.feature_dims.items()
        }

    def test_dil_enabled_appends_three_similarity_features(self):
        module = DomainInconsistencyModule(self.feature_dims, enabled=True)
        projected = module.project(self.features)
        normalized = L2NormBlock(enabled=True)(projected)
        fusion_features, sim_features = module(self.features, normalized)

        self.assertEqual(sim_features.shape, (self.batch_size, 3))
        self.assertEqual(fusion_features['rgb'].shape, (self.batch_size, 515))
        self.assertEqual(fusion_features['freq'].shape, (self.batch_size, 259))
        self.assertEqual(fusion_features['noise'].shape, (self.batch_size, 259))
        for feature in normalized.values():
            torch.testing.assert_close(
                torch.linalg.vector_norm(feature, dim=1),
                torch.ones(self.batch_size),
            )

    def test_dil_disabled_preserves_domain_feature_shapes(self):
        module = DomainInconsistencyModule(self.feature_dims, enabled=False)
        fusion_features, sim_features = module(self.features, {})

        self.assertIsNone(sim_features)
        for domain, feature in self.features.items():
            self.assertEqual(fusion_features[domain].shape, feature.shape)

    def test_dil_zero_mode_preserves_dil_shapes_with_zero_similarities(self):
        module = DomainInconsistencyModule(
            self.feature_dims,
            enabled=True,
            similarity_mode='zero',
        )
        projected = module.project(self.features)
        normalized = L2NormBlock(enabled=True)(projected)
        fusion_features, sim_features = module(self.features, normalized)

        torch.testing.assert_close(
            sim_features,
            torch.zeros(self.batch_size, 3),
        )
        self.assertEqual(fusion_features['rgb'].shape, (self.batch_size, 515))
        self.assertEqual(fusion_features['freq'].shape, (self.batch_size, 259))
        self.assertEqual(fusion_features['noise'].shape, (self.batch_size, 259))

    def test_domain_mask_zeroes_disabled_domains_only(self):
        block = DomainMaskBlock({
            'rgb': True,
            'freq': False,
            'noise': True,
        })
        masked = block(self.features)

        torch.testing.assert_close(masked['rgb'], self.features['rgb'])
        torch.testing.assert_close(masked['freq'], torch.zeros_like(self.features['freq']))
        torch.testing.assert_close(masked['noise'], self.features['noise'])

    def test_modular_dil_matches_main_branch_cosine_computation(self):
        module = DomainInconsistencyModule(self.feature_dims, enabled=True)
        projected = module.project(self.features)

        expected = torch.cat([
            torch.nn.functional.cosine_similarity(
                projected['rgb'], projected['freq'], dim=1
            ).unsqueeze(1),
            torch.nn.functional.cosine_similarity(
                projected['rgb'], projected['noise'], dim=1
            ).unsqueeze(1),
            torch.nn.functional.cosine_similarity(
                projected['freq'], projected['noise'], dim=1
            ).unsqueeze(1),
        ], dim=1)

        normalized = L2NormBlock(enabled=True)(projected)
        _, actual = module(self.features, normalized)

        torch.testing.assert_close(actual, expected)

    def test_norm_b_disabled_uses_raw_dot_product(self):
        module = DomainInconsistencyModule(self.feature_dims, enabled=True)
        projected = module.project(self.features)
        unnormalized = L2NormBlock(enabled=False)(projected)
        _, similarities = module(self.features, unnormalized)

        expected_rf = (
            projected['rgb'] * projected['freq']
        ).sum(dim=1)
        torch.testing.assert_close(similarities[:, 0], expected_rf)

    def test_dil_domain_mask_zeroes_only_affected_similarity_columns(self):
        module = DomainInconsistencyModule(self.feature_dims, enabled=True)
        projected = module.project(self.features)
        normalized = L2NormBlock(enabled=True)(projected)
        _, unmasked = module(self.features, normalized)
        _, masked = module(
            self.features,
            normalized,
            active_domains={'rgb': False, 'freq': True, 'noise': True},
        )

        torch.testing.assert_close(masked[:, 0], torch.zeros(self.batch_size))
        torch.testing.assert_close(masked[:, 1], torch.zeros(self.batch_size))
        torch.testing.assert_close(masked[:, 2], unmasked[:, 2])

    def test_config_parses_domain_flags(self):
        config = Config.from_yaml("configs/experiments/baseline.yaml")

        self.assertTrue(config.model.use_rgb)
        self.assertTrue(config.model.use_freq)
        self.assertTrue(config.model.use_noise)

    def test_optional_norms_and_adapter_preserve_shapes(self):
        feature_norm = FeatureNormBlock(self.feature_dims, enabled=True)
        normalized = feature_norm(self.features)
        for domain, feature in normalized.items():
            self.assertEqual(feature.shape, self.features[domain].shape)

        adapter = DomainAdapter(512, bottleneck_dim=64, enabled=True)
        self.assertEqual(adapter(self.features['rgb']).shape, self.features['rgb'].shape)

        l2_norm = L2NormBlock(enabled=True)
        projected = {
            domain: torch.randn(self.batch_size, 64)
            for domain in self.feature_dims
        }
        l2_normalized = l2_norm(projected)
        for feature in l2_normalized.values():
            self.assertEqual(feature.shape, (self.batch_size, 64))

    def test_fusion_strategy_output_shapes(self):
        input_dims = [515, 259, 259]
        features = [
            torch.randn(self.batch_size, dim)
            for dim in input_dims
        ]
        for fusion_type in ('concatenate', 'cross_attention', 'gated_fusion'):
            with self.subTest(fusion_type=fusion_type):
                fusion = create_fusion_module(
                    fusion_type=fusion_type,
                    input_dims=input_dims,
                    hidden_dim=128,
                    num_heads=4,
                    dropout=0.0,
                )
                fusion.eval()
                output = fusion(features)
                self.assertEqual(output.shape, (self.batch_size, 128))

    def test_full_detector_no_pretrained_forward_shape(self):
        if SwinModel is None:
            self.skipTest("transformers is not installed")

        model = create_detector(
            rgb_pretrained=False,
            freq_pretrained=False,
            noise_pretrained=False,
            fusion_hidden_dim=64,
            fusion_num_heads=4,
            classifier_hidden_dims=[32],
        )
        model.eval()

        rgb = torch.randn(self.batch_size, 3, 32, 32)
        freq = torch.randn(self.batch_size, 1, 32, 32)
        noise = torch.randn(self.batch_size, 3, 32, 32)

        with torch.no_grad():
            output = model(rgb, freq, noise, return_features=True)

        self.assertEqual(output['logits'].shape, (self.batch_size, 2))
        self.assertEqual(output['similarity_features'].shape, (self.batch_size, 3))
        self.assertEqual(output['fused_features'].shape, (self.batch_size, 64))


if __name__ == '__main__':
    unittest.main()
