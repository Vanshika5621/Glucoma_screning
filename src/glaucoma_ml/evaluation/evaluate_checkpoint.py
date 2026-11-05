import argparse
import json
from pathlib import Path
import pickle

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.glaucoma_ml.data.dataset import FundusDataset
from src.glaucoma_ml.data.manifest import LABEL_TO_INDEX
from src.glaucoma_ml.data.preprocessing import PREPROCESSING_VERSION
from src.glaucoma_ml.evaluation.report import create_report, save_report
from src.glaucoma_ml.models.classifier import Architecture, build_model
from src.glaucoma_ml.training.engine import run_epoch, select_device

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPECTED_CLASS_ORDER = ["normal", "glaucoma"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an existing checkpoint on an official REFUGE partition "
            "without training or tuning."
        )
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/processed/REFUGE/metadata.csv"),
        help="Dataset inspection metadata CSV.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/best_model.pt"),
        help="Previously trained local checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="New report directory (defaults to reports/refuge-<split>).",
    )
    parser.add_argument(
        "--split",
        choices=("official_validation", "official_test"),
        default="official_validation",
        help="Official REFUGE partition to evaluate.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    expanded = path.expanduser()
    return expanded.resolve() if expanded.is_absolute() else (PROJECT_ROOT / expanded).resolve()


def ensure_output_available(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to write into a non-empty output directory: {output_dir}. "
            "Choose a new --output-dir."
        )


def load_evaluation_metadata(metadata_path: Path, split: str) -> pd.DataFrame:
    frame = pd.read_csv(metadata_path)
    required_columns = {"image_path", "asset_type", "official_split", "label"}
    if not required_columns.issubset(frame.columns):
        raise ValueError(
            "Inspection metadata must contain image_path, asset_type, "
            "official_split, and label columns."
        )
    selected = frame.loc[
        (frame["asset_type"] == "fundus_photo")
        & (frame["official_split"] == split)
    ].copy()
    if selected.empty:
        raise ValueError(f"No {split} fundus photographs were found.")

    selected["label"] = selected["label"].astype(str).str.strip().str.lower()
    if selected["label"].isna().any() or set(selected["label"]) != set(LABEL_TO_INDEX):
        raise ValueError(
            f"{split} metadata must include both normal and glaucoma labels."
        )
    if selected["image_path"].duplicated().any():
        raise ValueError(f"{split} metadata contains duplicate image paths.")

    resolved_paths = []
    for raw_path in selected["image_path"]:
        image_path = Path(str(raw_path)).expanduser()
        if not image_path.is_absolute():
            image_path = PROJECT_ROOT / image_path
        image_path = image_path.resolve()
        if not image_path.is_file():
            raise ValueError(f"{split} image does not exist: {image_path}")
        resolved_paths.append(str(image_path))
    selected["image_path"] = resolved_paths
    return selected.reset_index(drop=True)


def load_compatible_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[nn.Module, Architecture, dict[str, object]]:
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except (OSError, pickle.UnpicklingError, RuntimeError, TypeError) as error:
        raise ValueError(f"Could not read checkpoint: {checkpoint_path}") from error
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must contain a metadata dictionary.")

    architecture_value = checkpoint.get("architecture")
    if architecture_value == "resnet50":
        architecture: Architecture = "resnet50"
    elif architecture_value == "efficientnet_b0":
        architecture = "efficientnet_b0"
    else:
        raise ValueError("Checkpoint architecture is missing or unsupported.")
    if checkpoint.get("class_order") != EXPECTED_CLASS_ORDER:
        raise ValueError("Checkpoint class order is missing or incompatible.")
    if checkpoint.get("image_size") != 224:
        raise ValueError("Checkpoint image size is missing or incompatible.")
    if checkpoint.get("preprocessing_version") != PREPROCESSING_VERSION:
        raise ValueError("Checkpoint preprocessing is missing or incompatible.")
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint model state is missing or invalid.")

    model = build_model(architecture, pretrained=False)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise ValueError("Checkpoint weights do not match its architecture.") from error
    model.to(device)
    model.eval()
    return model, architecture, checkpoint


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("Batch size must be a positive integer.")

    metadata_path = resolve_project_path(args.metadata)
    checkpoint_path = resolve_project_path(args.checkpoint)
    output_dir = resolve_project_path(
        args.output_dir
        if args.output_dir is not None
        else Path("reports") / f"refuge-{args.split.replace('_', '-')}"
    )
    report_path = output_dir / "metrics.json"
    ensure_output_available(output_dir)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    evaluation_frame = load_evaluation_metadata(metadata_path, args.split)
    device = select_device()
    model, architecture, checkpoint = load_compatible_checkpoint(
        checkpoint_path,
        device,
    )
    loader = DataLoader(
        FundusDataset(evaluation_frame, training=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    loss, labels, scores = run_epoch(
        model,
        loader,
        nn.CrossEntropyLoss(),
        device,
    )
    report = create_report(
        labels=labels,
        scores=scores,
        training_history=[],
        architecture=architecture,
        dataset_name=str(checkpoint.get("dataset_name", "unknown")),
        evaluation_split=args.split,
    )
    report.update(
        {
            "evaluation_dataset": "REFUGE",
            "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT))
            if checkpoint_path.is_relative_to(PROJECT_ROOT)
            else str(checkpoint_path),
            "checkpoint_training_dataset": checkpoint.get("dataset_name"),
            "preprocessing_version": PREPROCESSING_VERSION,
            "patient_wise_split": False,
            "patient_ids_available": False,
            "threshold_selection": f"fixed 0.5; not tuned on this {args.split} set",
            "evaluation_loss": loss,
        }
    )
    save_report(report, output_dir)
    print(f"Evaluated {len(evaluation_frame)} REFUGE {args.split} images on {device}.")
    print(f"Accuracy: {report['accuracy']:.4f}")
    print(f"Glaucoma sensitivity: {report['sensitivity']:.4f}")
    print(f"Specificity: {report['specificity']:.4f}")
    print(f"ROC-AUC: {report['roc_auc']}")
    print(f"Confusion matrix [normal, glaucoma]: {report['confusion_matrix']}")
    print(f"Saved {args.split}-only report to {report_path}")


if __name__ == "__main__":
    main()
