from io import BytesIO
from pathlib import Path

import pandas as pd
from PIL import Image
import pytest

from src.glaucoma_ml.data.manifest import load_manifest, split_manifest


def test_split_manifest_keeps_patient_ids_disjoint() -> None:
    frame = pd.DataFrame(
        [
            {
                "image_path": f"image-{patient}-{image}.png",
                "label": "glaucoma" if patient < 20 else "normal",
                "patient_id": f"patient-{patient}",
            }
            for patient in range(40)
            for image in range(2)
        ]
    )

    train, validation, test = split_manifest(frame)

    train_patients = set(train["patient_id"])
    validation_patients = set(validation["patient_id"])
    test_patients = set(test["patient_id"])
    assert train_patients.isdisjoint(validation_patients)
    assert train_patients.isdisjoint(test_patients)
    assert validation_patients.isdisjoint(test_patients)
    assert set(train["label"]) == {"normal", "glaucoma"}
    assert set(validation["label"]) == {"normal", "glaucoma"}
    assert set(test["label"]) == {"normal", "glaucoma"}


def test_split_manifest_rejects_missing_patient_ids() -> None:
    frame = pd.DataFrame(
        {
            "image_path": ["one.png", "two.png", "three.png", "four.png"],
            "label": ["normal", "glaucoma", "normal", "glaucoma"],
            "patient_id": ["a", "b", None, "d"],
        }
    )

    with pytest.raises(ValueError, match="missing values"):
        split_manifest(frame)


def test_split_manifest_rejects_absent_patient_ids() -> None:
    frame = pd.DataFrame(
        {
            "image_path": ["one.png", "two.png", "three.png", "four.png"],
            "label": ["normal", "glaucoma", "normal", "glaucoma"],
        }
    )

    with pytest.raises(ValueError, match="genuine patient_id"):
        split_manifest(frame)


def test_split_manifest_keeps_both_classes_with_imbalanced_patient_groups() -> None:
    frame = pd.DataFrame(
        [
            {
                "image_path": f"{patient}-{image}.png",
                "label": label,
                "patient_id": patient,
            }
            for label, patient_count in (("normal", 24), ("glaucoma", 6))
            for patient_index in range(patient_count)
            for patient in [f"{label}-{patient_index}"]
            for image in range(2)
        ]
    )

    splits = split_manifest(frame, seed=1)

    for split in splits:
        assert set(split["label"]) == {"normal", "glaucoma"}
    assert sum(len(split) for split in splits) == len(frame)


def test_split_manifest_is_deterministic_for_patient_groups() -> None:
    frame = pd.DataFrame(
        [
            {
                "image_path": f"image-{patient}-{image}.png",
                "label": "glaucoma" if patient < 12 else "normal",
                "patient_id": f"patient-{patient}",
            }
            for patient in range(30)
            for image in range(2)
        ]
    )

    first = split_manifest(frame, seed=23)
    second = split_manifest(frame, seed=23)

    assert [split["image_path"].tolist() for split in first] == [
        split["image_path"].tolist() for split in second
    ]


def test_split_manifest_rejects_blank_patient_ids() -> None:
    frame = pd.DataFrame(
        {
            "image_path": ["one.png", "two.png", "three.png", "four.png"],
            "label": ["normal", "glaucoma", "normal", "glaucoma"],
            "patient_id": ["a", " ", "c", "d"],
        }
    )

    with pytest.raises(ValueError, match="missing values or blanks"):
        split_manifest(frame)


def test_split_manifest_reports_insufficient_patients_per_class() -> None:
    frame = pd.DataFrame(
        [
            {
                "image_path": f"normal-{patient}.png",
                "label": "normal",
                "patient_id": f"n{patient}",
            }
            for patient in range(5)
        ]
        + [
            {"image_path": "glaucoma.png", "label": "glaucoma", "patient_id": "g1"}
        ]
    )

    with pytest.raises(ValueError, match="at least two patients"):
        split_manifest(frame)


def test_split_manifest_preserves_complete_official_partitions() -> None:
    rows = []
    for split_name in (
        "official_train",
        "official_validation",
        "official_test",
    ):
        for label in ("normal", "glaucoma"):
            rows.append(
                {
                    "image_path": f"{split_name}-{label}.jpg",
                    "label": label,
                    "official_split": split_name,
                    "patient_id": f"{split_name}-{label}",
                }
            )
    frame = pd.DataFrame(rows)

    train, validation, test = split_manifest(frame, seed=7)

    assert set(train["official_split"]) == {"official_train"}
    assert set(validation["official_split"]) == {"official_validation"}
    assert set(test["official_split"]) == {"official_test"}


def test_split_manifest_rejects_patient_overlap_in_official_partitions() -> None:
    rows = []
    for split_name in (
        "official_train",
        "official_validation",
        "official_test",
    ):
        for label in ("normal", "glaucoma"):
            patient_id = (
                "reused-patient"
                if split_name == "official_validation" and label == "normal"
                else f"{split_name}-{label}"
            )
            if split_name == "official_train" and label == "normal":
                patient_id = "reused-patient"
            rows.append(
                {
                    "image_path": f"{split_name}-{label}.jpg",
                    "label": label,
                    "official_split": split_name,
                    "patient_id": patient_id,
                }
            )
    frame = pd.DataFrame(rows)

    with pytest.raises(RuntimeError, match="Patient leakage"):
        split_manifest(frame)


def test_split_manifest_rejects_partial_official_partitions() -> None:
    frame = pd.DataFrame(
        [
            {
                "image_path": "one.jpg",
                "label": "normal",
                "official_split": "official_train",
                "patient_id": "one",
            },
            {
                "image_path": "two.jpg",
                "label": "glaucoma",
                "official_split": "official_train",
                "patient_id": "two",
            },
            {
                "image_path": "three.jpg",
                "label": "normal",
                "official_split": "official_validation",
                "patient_id": "three",
            },
            {
                "image_path": "four.jpg",
                "label": "glaucoma",
                "official_split": "official_validation",
                "patient_id": "four",
            },
        ]
    )

    with pytest.raises(ValueError, match="only some official partitions"):
        split_manifest(frame)


def test_load_manifest_rejects_byte_identical_images_across_paths(
    tmp_path: Path,
) -> None:
    image_buffer = BytesIO()
    Image.new("RGB", (16, 16), color=(20, 40, 60)).save(
        image_buffer,
        format="PNG",
    )
    image_bytes = image_buffer.getvalue()
    first_image = tmp_path / "normal.png"
    second_image = tmp_path / "glaucoma-copy.png"
    first_image.write_bytes(image_bytes)
    second_image.write_bytes(image_bytes)
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {"image_path": first_image.name, "label": "normal"},
            {"image_path": second_image.name, "label": "glaucoma"},
        ]
    ).to_csv(manifest_path, index=False)

    with pytest.raises(ValueError, match="byte-identical images"):
        load_manifest(manifest_path)


def test_load_manifest_accepts_distinct_real_image_bytes(tmp_path: Path) -> None:
    rows = []
    for label, color in (("normal", (10, 20, 30)), ("glaucoma", (30, 20, 10))):
        image_path = tmp_path / f"{label}.png"
        Image.new("RGB", (16, 16), color=color).save(image_path, format="PNG")
        rows.append({"image_path": image_path.name, "label": label})
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)

    loaded = load_manifest(manifest_path)

    assert set(loaded["label"]) == {"normal", "glaucoma"}
    assert len(loaded) == 2
