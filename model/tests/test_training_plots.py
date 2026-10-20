from pathlib import Path

from PIL import Image

from src.training.plotting import save_training_plots


def test_save_training_plots_creates_loss_and_accuracy_pngs(tmp_path: Path):
    history = [
        {"train_loss": 0.8, "val_loss": 0.9, "accuracy": 0.60},
        {"train_loss": 0.5, "val_loss": 0.6, "accuracy": 0.75},
        {"train_loss": 0.3, "val_loss": 0.4, "accuracy": 0.85},
    ]

    paths = save_training_plots(history, tmp_path)

    assert paths == {
        "loss": tmp_path / "loss_curve.png",
        "validation_accuracy": tmp_path / "validation_accuracy.png",
    }
    for path in paths.values():
        assert path.is_file()
        with Image.open(path) as image:
            assert image.format == "PNG"
            assert image.width > 0
            assert image.height > 0


def test_save_training_plots_skips_empty_history(tmp_path: Path):
    assert save_training_plots([], tmp_path) == {}
    assert not list(tmp_path.iterdir())
