from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

LABEL_TO_INDEX = {"normal": 0, "glaucoma": 1}


def _stratified_group_holdout(
    indices: pd.Index,
    labels: np.ndarray,
    groups: np.ndarray,
    fraction: float,
    seed: int,
    split_name: str,
) -> tuple[pd.Index, pd.Index]:
    label_values = np.asarray(labels)
    group_values = np.asarray(groups)
    unique_groups = np.unique(group_values)
    class_group_counts = [
        len(np.unique(group_values[label_values == class_name]))
        for class_name in LABEL_TO_INDEX
    ]
    max_splits = min(10, len(unique_groups), min(class_group_counts))
    if max_splits < 2:
        raise ValueError(
            f"Patient-wise {split_name} splitting requires at least two "
            "patients represented in each class."
        )

    target_splits = max(2, round(1 / fraction))
    min_candidate_splits = max(2, min(target_splits - 2, max_splits))
    max_candidate_splits = min(max_splits, target_splits + 2)
    candidate_splits = range(min_candidate_splits, max_candidate_splits + 1)
    index_values = np.asarray(indices)
    overall_prevalence = float(np.mean(label_values == "glaucoma"))
    best_candidate: tuple[float, np.ndarray, np.ndarray] | None = None

    for n_splits in candidate_splits:
        for attempt in range(8):
            splitter = StratifiedGroupKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=seed + attempt,
            )
            for train_positions, holdout_positions in splitter.split(
                np.zeros(len(index_values)),
                label_values,
                group_values,
            ):
                train_labels = label_values[train_positions]
                holdout_labels = label_values[holdout_positions]
                if (
                    set(train_labels) != set(LABEL_TO_INDEX)
                    or set(holdout_labels) != set(LABEL_TO_INDEX)
                ):
                    continue

                holdout_groups = group_values[holdout_positions]
                group_fraction = len(np.unique(holdout_groups)) / len(unique_groups)
                sample_fraction = len(holdout_positions) / len(index_values)
                train_prevalence = float(
                    np.mean(train_labels == "glaucoma")
                )
                holdout_prevalence = float(
                    np.mean(holdout_labels == "glaucoma")
                )
                score = (
                    abs(group_fraction - fraction)
                    + abs(sample_fraction - fraction)
                    + abs(train_prevalence - overall_prevalence)
                    + abs(holdout_prevalence - overall_prevalence)
                )
                if best_candidate is None or score < best_candidate[0]:
                    best_candidate = (
                        score,
                        index_values[train_positions],
                        index_values[holdout_positions],
                    )

    if best_candidate is None:
        raise ValueError(
            f"Could not create a patient-wise {split_name} split containing "
            "both classes in both partitions. More patients per class or "
            "different split fractions are required."
        )
    return pd.Index(best_candidate[1]), pd.Index(best_candidate[2])


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(manifest_path)
    if not {"image_path", "label"}.issubset(frame.columns):
        raise ValueError("Manifest CSV must contain image_path and label columns.")
    if frame.empty:
        raise ValueError("Manifest CSV contains no image rows.")

    frame = frame.copy()
    frame["label"] = frame["label"].astype(str).str.strip().str.lower()
    unknown_labels = sorted(set(frame["label"]) - set(LABEL_TO_INDEX))
    if unknown_labels:
        raise ValueError(
            "Labels must be 'normal' or 'glaucoma'; found: "
            + ", ".join(unknown_labels)
        )

    base_dir = manifest_path.resolve().parent
    resolved_paths = []
    for raw_path in frame["image_path"]:
        path = Path(str(raw_path)).expanduser()
        if not path.is_absolute():
            path = base_dir / path
        path = path.resolve()
        if not path.is_file():
            raise ValueError(f"Image file does not exist: {path}")
        resolved_paths.append(str(path))
    frame["image_path"] = resolved_paths

    if frame["image_path"].duplicated().any():
        raise ValueError("Manifest contains duplicate image paths.")
    if frame["label"].nunique() != 2:
        raise ValueError("Manifest must contain both normal and glaucoma samples.")
    return frame.reset_index(drop=True)


def split_manifest(
    frame: pd.DataFrame,
    seed: int = 42,
    test_fraction: float = 0.15,
    validation_fraction: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = frame.reset_index(drop=True)
    if not 0 < test_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("Split fractions must be between zero and one.")
    if test_fraction + validation_fraction >= 1:
        raise ValueError("Test and validation fractions must sum to less than one.")

    indices = frame.index
    labels = frame["label"].to_numpy()
    if set(labels) != set(LABEL_TO_INDEX):
        raise ValueError("Manifest must contain both normal and glaucoma samples.")
    official_split_names = {
        "official_train",
        "official_validation",
        "official_test",
    }
    official_values = (
        set(frame["official_split"].dropna().astype(str).str.strip())
        if "official_split" in frame.columns
        else set()
    )
    use_official_partitions = official_values == official_split_names
    if (
        official_values.intersection(official_split_names)
        and len(official_values) > 1
        and not use_official_partitions
    ):
        raise ValueError(
            "Manifest contains only some official partitions. Include all of "
            "official_train, official_validation, and official_test, or use "
            "one partition for a stratified image-level split."
        )
    if (
        len(official_values) == 1
        and official_values.intersection(official_split_names)
        and official_values != {"official_train"}
    ):
        raise ValueError(
            "A validation or test-only manifest cannot be used to create "
            "training splits."
        )
    if use_official_partitions:
        split_frames = tuple(
            frame.loc[frame["official_split"] == split_name].reset_index(drop=True)
            for split_name in (
                "official_train",
                "official_validation",
                "official_test",
            )
        )
        if any(split.empty for split in split_frames):
            raise ValueError("Official partitions must all contain image rows.")
        if any(
            set(split["label"]) != set(LABEL_TO_INDEX)
            for split in split_frames
        ):
            raise ValueError(
                "Every official partition must contain normal and glaucoma samples."
            )
        all_paths = pd.concat(
            [split["image_path"] for split in split_frames],
            ignore_index=True,
        )
        if all_paths.duplicated().any():
            raise ValueError("An image path occurs in multiple official partitions.")
        if "patient_id" in frame.columns:
            patient_ids = frame["patient_id"].astype("string").str.strip()
            if patient_ids.isna().any() or patient_ids.eq("").any():
                raise ValueError(
                    "patient_id is present but contains missing values or blanks; "
                    "remove the column when patient identifiers are unavailable."
                )
            patient_sets = [
                set(
                    split["patient_id"]
                    .astype("string")
                    .str.strip()
                    .astype(str)
                )
                for split in split_frames
            ]
            if (
                patient_sets[0] & patient_sets[1]
                or patient_sets[0] & patient_sets[2]
                or patient_sets[1] & patient_sets[2]
            ):
                raise RuntimeError(
                    "Patient leakage detected between official partitions."
                )
        return split_frames

    if "patient_id" in frame.columns:
        patient_ids = frame["patient_id"].astype("string").str.strip()
        if patient_ids.isna().any() or patient_ids.eq("").any():
            raise ValueError(
                "patient_id is present but contains missing values or blanks; "
                "repair the manifest or remove the column to use image-level "
                "splitting."
            )
        groups = patient_ids.to_numpy()
        train_validation_indices, test_indices = _stratified_group_holdout(
            indices,
            labels,
            groups,
            test_fraction,
            seed,
            "test",
        )
        remaining_fraction = validation_fraction / (1 - test_fraction)
        train_indices, validation_indices = _stratified_group_holdout(
            train_validation_indices,
            labels[train_validation_indices],
            groups[train_validation_indices],
            remaining_fraction,
            seed + 1,
            "validation",
        )
    else:
        indices = indices.to_numpy()
        train_validation_indices, test_indices = train_test_split(
            indices,
            test_size=test_fraction,
            random_state=seed,
            stratify=labels,
        )
        relative_validation_fraction = validation_fraction / (1 - test_fraction)
        train_indices, validation_indices = train_test_split(
            train_validation_indices,
            test_size=relative_validation_fraction,
            random_state=seed + 1,
            stratify=labels[train_validation_indices],
        )

    splits = tuple(
        frame.loc[split_indices].reset_index(drop=True)
        for split_indices in (train_indices, validation_indices, test_indices)
    )
    for split_name, split_frame in zip(("train", "validation", "test"), splits):
        if set(split_frame["label"]) != set(LABEL_TO_INDEX):
            raise ValueError(
                f"The {split_name} split does not contain both classes. "
                "Add more patients/images or adjust the split fractions."
            )
    if "patient_id" in frame.columns:
        train_patients = set(splits[0]["patient_id"].astype(str))
        validation_patients = set(splits[1]["patient_id"].astype(str))
        test_patients = set(splits[2]["patient_id"].astype(str))
        if (
            train_patients & validation_patients
            or train_patients & test_patients
            or validation_patients & test_patients
        ):
            raise RuntimeError("Patient leakage detected between data splits.")
    return splits
