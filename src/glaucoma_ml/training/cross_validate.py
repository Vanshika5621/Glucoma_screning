import argparse
import json
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import (
    StratifiedGroupKFold,
)
from torch.utils.data import DataLoader

from src.glaucoma_ml.data.dataset import FundusDataset
from src.glaucoma_ml.data.manifest import (
    LABEL_TO_INDEX,
    load_manifest,
    require_patient_ids,
)
from src.glaucoma_ml.evaluation.report import create_report, save_report
from src.glaucoma_ml.models.classifier import Architecture
from src.glaucoma_ml.training.engine import run_epoch, select_device
from src.glaucoma_ml.training.pipeline import TrainingConfig, train_with_validation

PROJECT_ROOT = Path(__file__).resolve().parents[3]
METRICS = ("recall", "precision", "f1_score", "roc_auc", "pr_auc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run stratified cross-validation on a development manifest only; "
            "the separate official test set must not be included."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument(
        "--architecture",
        choices=("resnet50", "efficientnet_b0"),
        default="resnet50",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--scheduler-patience", type=int, default=2)
    parser.add_argument(
        "--imbalance-strategy",
        choices=("class_weighted", "weighted_sampler", "none"),
        default="class_weighted",
    )
    parser.add_argument("--target-recall", type=float, default=0.9)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def split_outer_folds(
    frame: pd.DataFrame,
    folds: int,
    seed: int,
    patient_wise: bool,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if not patient_wise:
        raise ValueError(
            "Cross-validation requires genuine patient IDs; image-level "
            "splitting is disabled."
        )
    frame = require_patient_ids(frame)
    labels = frame["label"].to_numpy()
    class_counts = frame["label"].value_counts()
    if folds < 2:
        raise ValueError("Cross-validation requires at least two folds.")
    if set(class_counts.index) != set(LABEL_TO_INDEX) or (
        class_counts < folds
    ).any():
        raise ValueError(
            f"Each class must contain at least {folds} images for stratified "
            "cross-validation."
        )
    indices = np.arange(len(frame))
    groups = frame["patient_id"].astype(str).to_numpy()
    groups_per_class = frame.groupby("label")["patient_id"].nunique()
    if (groups_per_class < folds).any():
        raise ValueError(
            f"Each class must contain at least {folds} distinct patients "
            "for patient-wise stratified cross-validation."
        )
    splitter = StratifiedGroupKFold(
        n_splits=folds,
        shuffle=True,
        random_state=seed,
    )
    return list(splitter.split(indices, labels, groups))


def split_inner_validation(
    frame: pd.DataFrame,
    seed: int,
    patient_wise: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not patient_wise:
        raise ValueError(
            "Inner validation requires genuine patient IDs; image-level "
            "splitting is disabled."
        )
    frame = require_patient_ids(frame)
    indices = np.arange(len(frame))
    labels = frame["label"].to_numpy()
    groups = frame["patient_id"].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=seed,
    )
    train_indices = validation_indices = None
    for candidate_train, candidate_validation in splitter.split(
        indices,
        labels,
        groups,
    ):
        if (
            set(labels[candidate_train]) == set(LABEL_TO_INDEX)
            and set(labels[candidate_validation]) == set(LABEL_TO_INDEX)
        ):
            train_indices = candidate_train
            validation_indices = candidate_validation
            break
    if train_indices is None or validation_indices is None:
        raise ValueError(
            "Could not create an inner patient-wise validation split with "
            "both classes."
        )

    training = frame.iloc[train_indices].reset_index(drop=True)
    validation = frame.iloc[validation_indices].reset_index(drop=True)
    if set(training["image_path"]) & set(validation["image_path"]):
        raise ValueError("Inner training and validation images overlap.")
    if patient_wise and (
        set(training["patient_id"].astype(str))
        & set(validation["patient_id"].astype(str))
    ):
        raise ValueError("Inner training and validation patients overlap.")
    return training, validation


def main() -> None:
    args = parse_args()
    if args.folds != 5:
        raise ValueError("This pipeline is configured for the required 5 folds.")
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("Epochs and batch size must be positive.")
    manifest_path = args.manifest.expanduser().resolve()
    frame = require_patient_ids(load_manifest(manifest_path))
    if "official_split" in frame.columns:
        frame = frame.loc[frame["official_split"] == "official_train"].reset_index(
            drop=True
        )
    patient_wise = "patient_id" in frame.columns
    folds = split_outer_folds(frame, args.folds, args.seed, patient_wise)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty cross-validation directory: "
            f"{output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_reports: list[dict[str, object]] = []
    for fold_number, (train_positions, holdout_positions) in enumerate(folds, start=1):
        outer_training = frame.iloc[train_positions].reset_index(drop=True)
        holdout = frame.iloc[holdout_positions].reset_index(drop=True)
        if (
            set(outer_training["label"]) != set(LABEL_TO_INDEX)
            or set(holdout["label"]) != set(LABEL_TO_INDEX)
        ):
            raise ValueError(
                f"Fold {fold_number} does not contain both classes in its "
                "training and holdout partitions."
            )
        if set(outer_training["image_path"]) & set(holdout["image_path"]):
            raise ValueError(f"Image leakage detected in fold {fold_number}.")
        if patient_wise and (
            set(outer_training["patient_id"].astype(str))
            & set(holdout["patient_id"].astype(str))
        ):
            raise ValueError(f"Patient leakage detected in fold {fold_number}.")

        training, validation = split_inner_validation(
            outer_training,
            args.seed + fold_number,
            patient_wise,
        )
        fold_dir = output_dir / f"fold-{fold_number}"
        fold_dir.mkdir()
        config = TrainingConfig(
            architecture=args.architecture,
            dataset_name=args.dataset_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            head_learning_rate=args.learning_rate,
            backbone_learning_rate=args.backbone_learning_rate,
            seed=args.seed + fold_number,
            early_stopping_patience=args.early_stopping_patience,
            scheduler_patience=args.scheduler_patience,
            imbalance_strategy=args.imbalance_strategy,
            pretrained=args.pretrained,
            target_recall=args.target_recall,
        )
        result = train_with_validation(
            training,
            validation,
            config,
            fold_dir / "best_model.pt",
        )
        loader = DataLoader(
            FundusDataset(holdout),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )
        _, labels, scores = run_epoch(
            result.model,
            loader,
            torch.nn.CrossEntropyLoss(),
            select_device(),
        )
        report = create_report(
            labels=labels,
            scores=scores,
            training_history=result.history,
            architecture=args.architecture,
            dataset_name=args.dataset_name,
            evaluation_split=f"cross_validation_fold_{fold_number}",
            threshold=result.threshold_selection["threshold"],
            bootstrap_samples=1000,
            seed=args.seed + fold_number,
        )
        report.update(
            {
                "fold": fold_number,
                "patient_wise_split": patient_wise,
                "threshold_selection": result.threshold_selection,
                "split_counts": {
                    "train": len(training),
                    "validation": len(validation),
                    "fold_holdout": len(holdout),
                },
            }
        )
        save_report(report, fold_dir)
        pd.concat(
            [
                training.assign(training_split="train"),
                validation.assign(training_split="validation"),
                holdout.assign(training_split="fold_holdout"),
            ],
            ignore_index=True,
        ).to_csv(fold_dir / "split_manifest.csv", index=False)
        fold_reports.append(report)
        print(
            f"Fold {fold_number}/5: recall={report['recall']:.4f}, "
            f"precision={report['precision']:.4f}, "
            f"F1={report['f1_score']:.4f}, "
            f"ROC-AUC={report['roc_auc']}, PR-AUC={report['pr_auc']}"
        )

    summary: dict[str, object] = {
        "model_name": args.architecture,
        "dataset_name": args.dataset_name,
        "folds": 5,
        "patient_wise_split": patient_wise,
        "evaluation_protocol": (
            "Five stratified folds on development data; each fold uses a nested "
            "validation split for early stopping and threshold selection. "
            "The independent test set is not used."
        ),
        "fold_results": fold_reports,
        "metrics_mean_std": {
            metric: {
                "mean": mean(float(report[metric]) for report in fold_reports),
                "standard_deviation": (
                    stdev(float(report[metric]) for report in fold_reports)
                    if len(fold_reports) > 1
                    else 0.0
                ),
            }
            for metric in METRICS
        },
    }
    (output_dir / "cross_validation_metrics.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(f"Saved 5-fold summary to {output_dir / 'cross_validation_metrics.json'}")


if __name__ == "__main__":
    main()
