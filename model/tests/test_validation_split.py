import pandas as pd

from src.data.dataset import split_train_validation


def test_validation_split_preserves_fake_architectures():
    rows = [
        {"label": 0, "architecture": "natural", "id": index}
        for index in range(20)
    ]
    for architecture in ("progan", "stylegan", "biggan"):
        rows.extend(
            {"label": 1, "architecture": architecture, "id": index}
            for index in range(10)
        )

    train, val = split_train_validation(
        pd.DataFrame(rows),
        val_split=0.2,
        seed=42,
    )

    assert len(train) == 40
    assert len(val) == 10
    assert set(val.loc[val["label"] == 1, "architecture"]) == {
        "progan",
        "stylegan",
        "biggan",
    }
