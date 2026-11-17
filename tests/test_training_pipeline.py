from pathlib import Path

import pandas as pd
import pytest
import torch

from src.glaucoma_ml.training.train import class_weights_from_frame, prepare_run_dir
from src.glaucoma_ml.training.cross_validate import split_outer_folds


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


def test_five_fold_split_requires_five_examples_per_class() -> None:
    frame = pd.DataFrame(
        {
            "label": ["normal"] * 8 + ["glaucoma"] * 4,
            "patient_id": [f"n{index}" for index in range(8)]
            + [f"g{index}" for index in range(4)],
        }
    )

    with pytest.raises(ValueError, match="at least 5 images"):
        split_outer_folds(frame, folds=5, seed=42, patient_wise=True)


def test_five_fold_split_is_stratified_and_disjoint() -> None:
    frame = pd.DataFrame(
        {
            "image_path": [f"image-{index}.jpg" for index in range(20)],
            "label": ["normal"] * 10 + ["glaucoma"] * 10,
            "patient_id": [f"n{index}" for index in range(10)]
            + [f"g{index}" for index in range(10)],
        }
    )

    folds = split_outer_folds(frame, folds=5, seed=42, patient_wise=True)

    assert len(folds) == 5
    for training_indices, holdout_indices in folds:
        assert not set(training_indices) & set(holdout_indices)
        assert set(frame.iloc[holdout_indices]["label"]) == {
            "normal",
            "glaucoma",
        }
        assert set(frame.iloc[training_indices]["patient_id"]).isdisjoint(
            set(frame.iloc[holdout_indices]["patient_id"])
        )


def test_patient_wise_five_fold_split_rejects_missing_identifiers() -> None:
    frame = pd.DataFrame(
        {
            "label": ["normal"] * 5 + ["glaucoma"] * 5,
            "patient_id": [None] + [f"n{index}" for index in range(4)]
            + [f"g{index}" for index in range(5)],
        }
    )

    with pytest.raises(ValueError, match="missing values or blanks"):
        split_outer_folds(frame, folds=5, seed=42, patient_wise=True)


def test_five_fold_split_rejects_image_level_mode() -> None:
    frame = pd.DataFrame(
        {
            "label": ["normal"] * 5 + ["glaucoma"] * 5,
            "patient_id": [f"n{index}" for index in range(5)]
            + [f"g{index}" for index in range(5)],
        }
    )

    with pytest.raises(ValueError, match="image-level splitting is disabled"):
        split_outer_folds(frame, folds=5, seed=42, patient_wise=False)
