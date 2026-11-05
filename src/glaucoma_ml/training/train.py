import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import random

import numpy as np
import pandas as pd
import torch
import torchvision
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader

from src.glaucoma_ml.data.dataset import FundusDataset
from src.glaucoma_ml.data.manifest import load_manifest, split_manifest
from src.glaucoma_ml.data.preprocessing import PREPROCESSING_VERSION
from src.glaucoma_ml.evaluation.report import create_report, save_report
from src.glaucoma_ml.models.classifier import Architecture, build_model
from src.glaucoma_ml.training.engine import run_epoch, select_device

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a research fundus classifier from a labeled CSV manifest."
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--scheduler-patience", type=int, default=2)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument(
        "--class-weighting",
        choices=("balanced", "none"),
        default="balanced",
        help="Use inverse-frequency training-split weights for imbalanced classes.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Empty run directory for checkpoint, configuration, metrics, and plots.",
    )
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Initialize from torchvision ImageNet weights (default; may download "
            "weights). Use --no-pretrained to train from scratch."
        ),
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def class_weights_from_frame(frame: pd.DataFrame) -> torch.Tensor:
    counts = frame["label"].value_counts()
    if set(counts.index) != {"normal", "glaucoma"} or (counts == 0).any():
        raise ValueError("Training data must include both classes.")
    total = int(counts.sum())
    return torch.tensor(
        [total / (2 * counts[label]) for label in ("normal", "glaucoma")],
        dtype=torch.float32,
    )


def prepare_run_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite a non-empty run directory: {resolved}"
        )
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def main() -> None:
    args = parse_args()
    if (
        args.epochs < 1
        or args.batch_size < 1
        or args.learning_rate <= 0
        or args.early_stopping_patience < 1
        or args.scheduler_patience < 0
    ):
        raise ValueError("Epoch, batch-size, learning-rate, or patience values are invalid.")
    if (
        not 0 < args.test_fraction < 1
        or not 0 < args.validation_fraction < 1
        or args.test_fraction + args.validation_fraction >= 1
    ):
        raise ValueError(
            "Test and validation fractions must be positive and sum to less than one."
        )

    seed_everything(args.seed)
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
    architecture: Architecture = args.architecture
    default_run_name = (
        f"{architecture}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    run_dir = prepare_run_dir(
        args.output_dir
        if args.output_dir is not None
        else PROJECT_ROOT / "runs" / default_run_name
    )
    checkpoint_path = run_dir / "best_model.pt"
    device = select_device()
    print(f"Training on {device}; patient-wise split: {'patient_id' in manifest}")

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        FundusDataset(train_frame, training=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    train_metrics_loader = DataLoader(
        FundusDataset(train_frame),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    validation_loader = DataLoader(
        FundusDataset(validation_frame),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        FundusDataset(test_frame),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = build_model(architecture, pretrained=args.pretrained).to(device)
    class_weights = (
        class_weights_from_frame(train_frame).to(device)
        if args.class_weighting == "balanced"
        else None
    )
    loss_function = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=args.scheduler_patience,
        min_lr=1e-7,
    )
    run_config = {
        "architecture": architecture,
        "dataset_name": args.dataset_name,
        "manifest": str(manifest_path),
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "batch_size": args.batch_size,
        "optimizer": "AdamW",
        "initial_learning_rate": args.learning_rate,
        "scheduler": {
            "name": "ReduceLROnPlateau",
            "monitor": "validation_loss",
            "factor": 0.5,
            "patience": args.scheduler_patience,
            "minimum_learning_rate": 1e-7,
        },
        "early_stopping": {
            "monitor": "validation_loss",
            "patience": args.early_stopping_patience,
        },
        "loss": "CrossEntropyLoss",
        "class_weighting": args.class_weighting,
        "class_weights": (
            class_weights.cpu().tolist() if class_weights is not None else None
        ),
        "pretrained": args.pretrained,
        "pretrained_weight_source": (
            f"torchvision {architecture} DEFAULT ImageNet weights"
            if args.pretrained
            else "random initialization"
        ),
        "fine_tuning_strategy": "fine-tune all backbone and classifier parameters",
        "image_size": 224,
        "preprocessing_version": PREPROCESSING_VERSION,
        "augmentation": (
            "RandomHorizontalFlip(p=0.5), RandomRotation(8 degrees), "
            "ColorJitter(brightness=0.08, contrast=0.08, saturation=0.05)"
        ),
        "test_fraction": args.test_fraction,
        "validation_fraction": args.validation_fraction,
        "train_metrics": (
            "epoch-end, augmentation-free inference on training images"
        ),
        "patient_wise_split": "patient_id" in manifest,
        "official_partitions_preserved": official_partitions,
        "device": str(device),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "python_version": platform.python_version(),
        "reproducibility_note": (
            "Random generators are seeded; exact bitwise determinism can vary "
            "by hardware, PyTorch build, and accelerator kernels."
        ),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "training_config.json").write_text(
        json.dumps(run_config, indent=2),
        encoding="utf-8",
    )

    history: list[dict[str, float]] = []
    best_validation_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        run_epoch(
            model,
            train_loader,
            loss_function,
            device,
            optimizer,
        )
        train_loss, train_labels, train_scores = run_epoch(
            model,
            train_metrics_loader,
            loss_function,
            device,
        )
        validation_loss, validation_labels, validation_scores = run_epoch(
            model,
            validation_loader,
            loss_function,
            device,
        )
        train_accuracy = float(
            accuracy_score(
                train_labels,
                [int(score >= 0.5) for score in train_scores],
            )
        )
        validation_accuracy = float(
            accuracy_score(
                validation_labels,
                [int(score >= 0.5) for score in validation_scores],
            )
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "train_accuracy": train_accuracy,
                "validation_accuracy": validation_accuracy,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        print(
            f"Epoch {epoch}/{args.epochs}: "
            f"train_loss={train_loss:.4f}, "
            f"validation_loss={validation_loss:.4f}, "
            f"train_accuracy={train_accuracy:.4f}, "
            f"validation_accuracy={validation_accuracy:.4f}, "
            f"learning_rate={optimizer.param_groups[0]['lr']:.2e}"
        )
        scheduler.step(validation_loss)
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "architecture": architecture,
                    "class_order": ["normal", "glaucoma"],
                    "image_size": 224,
                    "preprocessing_version": PREPROCESSING_VERSION,
                    "dataset_name": args.dataset_name,
                    "epoch": epoch,
                    "validation_loss": validation_loss,
                    "training_config": run_config,
                    "model_state_dict": model.state_dict(),
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stopping_patience:
                print(
                    "Early stopping: validation loss did not improve for "
                    f"{args.early_stopping_patience} epochs."
                )
                break

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    _, test_labels, test_scores = run_epoch(
        model,
        test_loader,
        loss_function,
        device,
    )
    report = create_report(
        labels=test_labels,
        scores=test_scores,
        training_history=history,
        architecture=architecture,
        dataset_name=args.dataset_name,
        evaluation_split="official_test" if official_partitions else "test",
    )
    report["patient_wise_split"] = "patient_id" in manifest
    report["official_partitions_preserved"] = official_partitions
    report["preprocessing_version"] = PREPROCESSING_VERSION
    report["best_checkpoint_epoch"] = checkpoint["epoch"]
    report["split_counts"] = {
        "train": len(train_frame),
        "validation": len(validation_frame),
        "test": len(test_frame),
    }
    report["class_counts"] = {
        label: {
            split_name: int((frame["label"] == label).sum())
            for split_name, frame in (
                ("train", train_frame),
                ("validation", validation_frame),
                ("test", test_frame),
            )
        }
        for label in ("normal", "glaucoma")
    }
    report["training_configuration"] = run_config
    save_report(report, run_dir)
    pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False)
    split_manifest = pd.concat(
        [
            train_frame.assign(training_split="train"),
            validation_frame.assign(training_split="validation"),
            test_frame.assign(training_split="test"),
        ],
        ignore_index=True,
    )
    split_manifest.to_csv(run_dir / "split_manifest.csv", index=False)
    print(f"Saved checkpoint to {checkpoint_path}")
    print(f"Saved configuration, metrics, and plots to {run_dir}")


if __name__ == "__main__":
    main()
