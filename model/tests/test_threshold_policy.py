import unittest

import torch

from src.training.metrics import MetricsCalculator, resolve_decision_threshold
from src.training.trainer import Trainer


class ThresholdPolicyTest(unittest.TestCase):
    def _trainer_with_predictions(self, tune_threshold: bool) -> Trainer:
        trainer = Trainer.__new__(Trainer)
        trainer.decision_threshold = 0.5
        trainer.tune_threshold = tune_threshold
        trainer.threshold_metric = "balanced_accuracy"
        trainer.metrics_calculator = MetricsCalculator(threshold=0.5)
        trainer.metrics_calculator.update(
            labels=torch.tensor([0, 1]),
            probs=torch.tensor([
                [0.60, 0.40],
                [0.55, 0.45],
            ]),
        )
        return trainer

    def test_fixed_threshold_policy_uses_exactly_point_five(self):
        trainer = self._trainer_with_predictions(tune_threshold=False)

        threshold, metrics = trainer._select_validation_threshold()

        self.assertEqual(threshold, 0.5)
        self.assertEqual(metrics["threshold"], 0.5)

    def test_threshold_tuning_requires_explicit_enablement(self):
        trainer = self._trainer_with_predictions(tune_threshold=True)

        threshold, metrics = trainer._select_validation_threshold()

        self.assertNotEqual(threshold, 0.5)
        self.assertGreater(metrics["balanced_accuracy"], 0.5)

    def test_checkpoint_uses_fixed_threshold_by_default(self):
        checkpoint = {
            "decision_threshold": 0.5,
            "best_threshold": 0.35,
            "threshold_tuned": False,
        }

        self.assertEqual(resolve_decision_threshold(checkpoint), 0.5)

    def test_checkpoint_uses_tuned_threshold_only_when_marked(self):
        checkpoint = {
            "decision_threshold": 0.5,
            "best_threshold": 0.35,
            "threshold_tuned": True,
        }

        self.assertEqual(resolve_decision_threshold(checkpoint), 0.35)

    def test_explicit_threshold_override_has_priority(self):
        checkpoint = {
            "decision_threshold": 0.5,
            "best_threshold": 0.35,
            "threshold_tuned": True,
        }

        self.assertEqual(
            resolve_decision_threshold(checkpoint, override=0.6),
            0.6,
        )


if __name__ == "__main__":
    unittest.main()
