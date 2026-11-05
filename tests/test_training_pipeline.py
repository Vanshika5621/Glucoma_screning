from pathlib import Path

import pandas as pd
import pytest
import torch

from src.glaucoma_ml.training.train import class_weights_from_frame, prepare_run_dir


def test_balanced_class_weights_use_training_class_counts() -> None:
    frame = pd.DataFrame(
        {"label": ["normal"] * 6 + ["glaucoma"] * 2}
    )

    weights = class_weights_from_frame(frame)

    assert torch.allclose(weights, torch.tensor([2 / 3, 2.0]))


def test_class_weighting_requires_both_classes() -> None:
    with pytest.raises(ValueError, match="both classes"):
        class_weights_from_frame(pd.DataFrame({"label": ["normal", "normal"]}))


def test_run_directory_refuses_to_overwrite_existing_files(tmp_path: Path) -> None:
    (tmp_path / "existing.txt").write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        prepare_run_dir(tmp_path)

    assert (tmp_path / "existing.txt").read_text(encoding="utf-8") == "preserve"
