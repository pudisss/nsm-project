"""Decision-threshold policy copied from the training workflow."""

from typing import Any, Mapping, Optional


def resolve_decision_threshold(
    checkpoint: Mapping[str, Any],
    override: Optional[float] = None,
) -> float:
    """Resolve and validate the fake-class decision threshold."""
    if override is not None:
        threshold = float(override)
    elif checkpoint.get("threshold_tuned", False):
        threshold = float(checkpoint.get("best_threshold", 0.5))
    else:
        threshold = float(checkpoint.get("decision_threshold", 0.5))

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Decision threshold must be between 0 and 1.")
    return threshold
