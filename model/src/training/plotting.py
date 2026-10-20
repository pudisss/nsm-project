"""Plotting helpers for training metrics."""

from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


def save_training_plots(
    history: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> Dict[str, Path]:
    """Save loss and validation-accuracy curves for a completed run."""
    if not history:
        return {}

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = list(range(1, len(history) + 1))
    train_loss = [float(metrics["train_loss"]) for metrics in history]
    val_loss = [float(metrics["val_loss"]) for metrics in history]
    val_accuracy = [float(metrics["accuracy"]) for metrics in history]

    loss_path = output_dir / "loss_curve.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, marker="o", label="Training loss")
    ax.plot(epochs, val_loss, marker="o", label="Validation loss")
    ax.set(
        title="Training and Validation Loss",
        xlabel="Epoch",
        ylabel="Loss",
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(loss_path, dpi=150)
    plt.close(fig)

    accuracy_path = output_dir / "validation_accuracy.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        epochs,
        [accuracy * 100.0 for accuracy in val_accuracy],
        color="tab:green",
        marker="o",
        label="Validation accuracy",
    )
    ax.set(
        title="Validation Accuracy",
        xlabel="Epoch",
        ylabel="Accuracy (%)",
    )
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(accuracy_path, dpi=150)
    plt.close(fig)

    return {
        "loss": loss_path,
        "validation_accuracy": accuracy_path,
    }
