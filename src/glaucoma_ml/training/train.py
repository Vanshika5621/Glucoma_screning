import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.glaucoma_ml.data.dataset import FundusDataset
from src.glaucoma_ml.data.manifest import load_manifest, split_manifest
from src.glaucoma_ml.evaluation.report import create_report, save_report
from src.glaucoma_ml.models.classifier import Architecture
from src.glaucoma_ml.training.engine import run_epoch, select_device
from src.glaucoma_ml.training.pipeline import (
    TrainingConfig,
    class_weights_from_frame,
    train_with_validation,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a reproducible fundus classifier from a CSV manifest."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument(
        "--architecture",
        choices=("resnet50", "efficientnet_b0"),
        default="resnet50",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--scheduler-patience", type=int, default=2)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument(
        "--imbalance-strategy",
        choices=("class_weighted", "weighted_sampler", "none"),
        default="class_weighted",
        help="Choose class-weighted loss, weighted sampling, or no correction.",
    )
    parser.add_argument(
        "--class-weighting",
        choices=("balanced", "none"),
        default=None,
        help="Deprecated alias for --imbalance-strategy.",
    )
    parser.add_argument("--target-recall", type=float, default=0.9)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Initialize from torchvision ImageNet weights.",
    )
    return parser.parse_args()


def prepare_run_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite a non-empty run directory: {resolved}"
        )
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _print_split_counts(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
) -> None:
    for split_name, frame in (
        ("train", train_frame),
        ("validation", validation_frame),
        ("test", test_frame),
    ):
        counts = frame["label"].value_counts().reindex(
            ["normal", "glaucoma"],
            fill_value=0,
        )
        print(
            f"{split_name}: {len(frame)} images "
            f"(normal={counts['normal']}, glaucoma={counts['glaucoma']})"
        )


def main() -> None:
    args = parse_args()
    if (
        args.epochs < 1
        or args.batch_size < 1
        or args.learning_rate <= 0
        or args.backbone_learning_rate <= 0
        or args.early_stopping_patience < 1
        or args.scheduler_patience < 0
    ):
        raise ValueError("Epoch, batch-size, learning-rate, or patience is invalid.")
    if (
        not 0 < args.test_fraction < 1
        or not 0 < args.validation_fraction < 1
        or args.test_fraction + args.validation_fraction >= 1
    ):
        raise ValueError(
            "Test and validation fractions must be positive and sum to less than one."
        )
    if not 0 < args.target_recall <= 1:
        raise ValueError("Target recall must be greater than zero and at most one.")

    imbalance_strategy = args.imbalance_strategy
    if args.class_weighting is not None:
        if args.imbalance_strategy != "class_weighted":
            raise ValueError(
                "Do not combine deprecated --class-weighting with "
                "--imbalance-strategy."
            )
        imbalance_strategy = (
            "class_weighted" if args.class_weighting == "balanced" else "none"
        )

    manifest_path = args.manifest.expanduser().resolve()
    manifest = load_manifest(manifest_path)
    official_partitions = (
        "official_split" in manifest.columns
        and set(manifest["official_split"].dropna().astype(str).str.strip())
        == {"official_train", "official_validation", "official_test"}
    )
    train_frame, validation_frame, test_frame = split_manifest(
        manifest,
        seed=args.seed,
        test_fraction=args.test_fraction,
        validation_fraction=args.validation_fraction,
    )
    all_paths = [
        set(frame["image_path"])
        for frame in (train_frame, validation_frame, test_frame)
    ]
    if (
        all_paths[0] & all_paths[1]
        or all_paths[0] & all_paths[2]
        or all_paths[1] & all_paths[2]
    ):
        raise ValueError("An image occurs in more than one data split.")
    if "patient_id" in manifest:
        patient_sets = [
            set(frame["patient_id"].astype(str))
            for frame in (train_frame, validation_frame, test_frame)
        ]
        if (
            patient_sets[0] & patient_sets[1]
            or patient_sets[0] & patient_sets[2]
            or patient_sets[1] & patient_sets[2]
        ):
            raise ValueError("A patient occurs in more than one data split.")

    architecture: Architecture = args.architecture
    default_run_name = (
        f"{architecture}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    run_dir = prepare_run_dir(
        args.output_dir
        if args.output_dir is not None
        else PROJECT_ROOT / "runs" / default_run_name
    )
    _print_split_counts(train_frame, validation_frame, test_frame)
    print(
        "Split method: "
        + ("official partitions" if official_partitions else "stratified")
        + f"; patient-wise: {'patient_id' in manifest}"
    )

    config = TrainingConfig(
        architecture=architecture,
        dataset_name=args.dataset_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        head_learning_rate=args.learning_rate,
        backbone_learning_rate=args.backbone_learning_rate,
        seed=args.seed,
        early_stopping_patience=args.early_stopping_patience,
        scheduler_patience=args.scheduler_patience,
        imbalance_strategy=imbalance_strategy,
        pretrained=args.pretrained,
        target_recall=args.target_recall,
    )
    checkpoint_path = run_dir / "best_model.pt"
    result = train_with_validation(
        train_frame,
        validation_frame,
        config,
        checkpoint_path,
    )
    print(
        "Selected validation-only threshold "
        f"{result.threshold_selection['threshold']:.4f}; "
        f"recall={result.threshold_selection['validation_recall']:.4f}, "
        f"precision={result.threshold_selection['validation_precision']:.4f}"
    )

    device = select_device()
    test_loader = DataLoader(
        FundusDataset(test_frame, training=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    _, test_labels, test_scores = run_epoch(
        result.model,
        test_loader,
        torch.nn.CrossEntropyLoss(),
        device,
    )
    report = create_report(
        labels=test_labels,
        scores=test_scores,
        training_history=result.history,
        architecture=architecture,
        dataset_name=args.dataset_name,
        evaluation_split="official_test" if official_partitions else "test",
        threshold=result.threshold_selection["threshold"],
        seed=args.seed,
    )
    report.update(
        {
            "patient_wise_split": "patient_id" in manifest,
            "official_partitions_preserved": official_partitions,
            "best_checkpoint_epoch": result.checkpoint["epoch"],
            "threshold_selection": result.threshold_selection,
            "split_counts": {
                "train": len(train_frame),
                "validation": len(validation_frame),
                "test": len(test_frame),
            },
            "class_counts": {
                label: int((test_frame["label"] == label).sum())
                for label in ("normal", "glaucoma")
            },
            "class_counts_by_split": {
                split_name: {
                    label: int((frame["label"] == label).sum())
                    for label in ("normal", "glaucoma")
                }
                for split_name, frame in (
                    ("train", train_frame),
                    ("validation", validation_frame),
                    ("test", test_frame),
                )
            },
            "training_configuration": result.checkpoint["training_config"],
            "checkpoint": str(checkpoint_path),
        }
    )
    save_report(report, run_dir)
    pd.concat(
        [
            train_frame.assign(training_split="train"),
            validation_frame.assign(training_split="validation"),
            test_frame.assign(training_split="test"),
        ],
        ignore_index=True,
    ).to_csv(run_dir / "split_manifest.csv", index=False)
    print(f"Final test recall: {report['recall']:.4f}")
    print(f"Final test precision: {report['precision']:.4f}")
    print(f"Final test F1: {report['f1_score']:.4f}")
    print(f"Final test ROC-AUC: {report['roc_auc']}")
    print(f"Final test PR-AUC: {report['pr_auc']}")
    print(f"Saved checkpoint and report to {run_dir}")


if __name__ == "__main__":
    main()
