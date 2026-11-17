import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from src.glaucoma_ml.evaluation.evaluate_checkpoint import (
    ensure_output_available,
    load_evaluation_metadata,
    load_compatible_checkpoint,
)
from src.glaucoma_ml.evaluation.report import (
    create_report,
    save_report,
    select_threshold_for_recall,
)


def test_report_uses_supplied_test_observations_and_saves_real_outputs(tmp_path) -> None:
    report = create_report(
        labels=[0, 0, 1, 1],
        scores=[0.1, 0.8, 0.6, 0.9],
        training_history=[
            {
                "epoch": 1,
                "train_loss": 0.8,
                "validation_loss": 0.7,
                "train_accuracy": 0.6,
                "validation_accuracy": 0.75,
            }
        ],
        architecture="resnet50",
        dataset_name="test-set",
    )

    assert report["accuracy"] == 0.75
    assert report["sensitivity"] == 1.0
    assert report["specificity"] == 0.5
    assert report["roc_auc"] == 0.75
    assert report["pr_auc"] is not None
    assert report["recall_ci_95"]["lower"] <= report["recall"] <= report["recall_ci_95"]["upper"]
    assert report["test_samples"] == 4
    assert report["positive_samples"] == 2
    assert report["normal_baseline_accuracy"] == 0.5

    save_report(report, tmp_path)

    saved_report = json.loads((tmp_path / "metrics.json").read_text())
    assert saved_report["test_samples"] == 4
    assert (tmp_path / "confusion_matrix.png").is_file()
    assert (tmp_path / "training_history.png").is_file()
    assert (tmp_path / "training_accuracy.png").is_file()
    assert (tmp_path / "validation_accuracy.png").is_file()
    assert (tmp_path / "training_loss.png").is_file()
    assert (tmp_path / "validation_loss.png").is_file()
    assert (tmp_path / "roc_curve.png").is_file()


def test_official_validation_report_is_not_mislabeled_as_test_data() -> None:
    report = create_report(
        labels=[0, 0, 1, 1],
        scores=[0.1, 0.3, 0.7, 0.9],
        training_history=[],
        architecture="resnet50",
        dataset_name="training-set",
        evaluation_split="official_validation",
    )

    assert report["evaluation_split"] == "official_validation"
    assert report["evaluation_samples"] == 4
    assert "test_samples" not in report


def test_evaluation_metadata_selects_only_labeled_partition_fundus_files(
    tmp_path: Path,
) -> None:
    normal_image = tmp_path / "normal.jpg"
    glaucoma_image = tmp_path / "glaucoma.jpg"
    normal_image.touch()
    glaucoma_image.touch()
    metadata_path = tmp_path / "metadata.csv"
    pd.DataFrame(
        [
            {
                "image_path": str(normal_image),
                "asset_type": "fundus_photo",
                "official_split": "official_validation",
                "label": "normal",
            },
            {
                "image_path": str(glaucoma_image),
                "asset_type": "fundus_photo",
                "official_split": "official_validation",
                "label": "glaucoma",
            },
            {
                "image_path": str(tmp_path / "mask.bmp"),
                "asset_type": "segmentation_mask",
                "official_split": "official_validation",
                "label": "",
            },
            {
                "image_path": str(tmp_path / "test.jpg"),
                "asset_type": "fundus_photo",
                "official_split": "official_test",
                "label": "normal",
            },
        ]
    ).to_csv(metadata_path, index=False)

    selected = load_evaluation_metadata(metadata_path, "official_validation")

    assert len(selected) == 2
    assert set(selected["label"]) == {"normal", "glaucoma"}
    assert set(selected["official_split"]) == {"official_validation"}
    assert set(selected["asset_type"]) == {"fundus_photo"}


def test_evaluation_metadata_fails_on_missing_image(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.csv"
    pd.DataFrame(
        [
            {
                "image_path": str(tmp_path / "missing.jpg"),
                "asset_type": "fundus_photo",
                "official_split": "official_validation",
                "label": "normal",
            },
            {
                "image_path": str(tmp_path / "also-missing.jpg"),
                "asset_type": "fundus_photo",
                "official_split": "official_validation",
                "label": "glaucoma",
            },
        ]
    ).to_csv(metadata_path, index=False)

    with pytest.raises(ValueError, match="image does not exist"):
        load_evaluation_metadata(metadata_path, "official_validation")


def test_official_test_report_includes_test_sample_count() -> None:
    report = create_report(
        labels=[0, 0, 1, 1],
        scores=[0.1, 0.3, 0.7, 0.9],
        training_history=[],
        architecture="resnet50",
        dataset_name="training-set",
        evaluation_split="official_test",
    )

    assert report["evaluation_split"] == "official_test"
    assert report["test_samples"] == 4


def test_report_uses_the_supplied_decision_threshold() -> None:
    report = create_report(
        labels=[0, 0, 1, 1],
        scores=[0.1, 0.4, 0.3, 0.9],
        training_history=[],
        architecture="resnet50",
        dataset_name="training-set",
        threshold=0.25,
    )

    assert report["threshold"] == 0.25
    assert report["recall"] == 1.0
    assert report["confusion_matrix"] == [[1, 1], [0, 2]]


def test_threshold_selection_meets_recall_and_maximizes_precision() -> None:
    result = select_threshold_for_recall(
        labels=[0, 0, 1, 1],
        scores=[0.2, 0.7, 0.6, 0.8],
        target_recall=0.9,
    )

    assert result == {
        "threshold": 0.6,
        "target_recall": 0.9,
        "validation_recall": 1.0,
        "validation_precision": 2 / 3,
    }


def test_threshold_selection_requires_both_validation_classes() -> None:
    with pytest.raises(ValueError, match="both classes"):
        select_threshold_for_recall(
            labels=[0, 0],
            scores=[0.1, 0.2],
        )


def test_evaluation_does_not_overwrite_existing_output(tmp_path: Path) -> None:
    existing_report = tmp_path / "metrics.json"
    existing_report.write_text("keep this report", encoding="utf-8")

    with pytest.raises(FileExistsError, match="non-empty output directory"):
        ensure_output_available(tmp_path)

    assert existing_report.read_text(encoding="utf-8") == "keep this report"


def test_checkpoint_with_incompatible_class_order_is_rejected(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "incompatible.pt"
    torch.save(
        {
            "architecture": "resnet50",
            "class_order": ["glaucoma", "normal"],
        },
        checkpoint_path,
    )

    with pytest.raises(ValueError, match="class order"):
        load_compatible_checkpoint(checkpoint_path, torch.device("cpu"))


def test_corrupt_checkpoint_is_rejected(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "corrupt.pt"
    checkpoint_path.write_bytes(b"not a torch checkpoint")

    with pytest.raises(ValueError, match="Could not read checkpoint"):
        load_compatible_checkpoint(checkpoint_path, torch.device("cpu"))
