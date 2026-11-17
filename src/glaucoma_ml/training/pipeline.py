from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import platform
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
from sklearn.metrics import accuracy_score, average_precision_score
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.glaucoma_ml.data.dataset import FundusDataset
from src.glaucoma_ml.data.preprocessing import PREPROCESSING_VERSION
from src.glaucoma_ml.evaluation.report import (
    save_precision_recall_threshold_plot,
    select_threshold_for_recall,
)
from src.glaucoma_ml.models.classifier import Architecture, build_model
from src.glaucoma_ml.training.engine import run_epoch, select_device


@dataclass(frozen=True)
class TrainingConfig:
    architecture: Architecture = "resnet50"
    dataset_name: str = "unspecified"
    epochs: int = 30
    batch_size: int = 16
    head_learning_rate: float = 1e-4
    backbone_learning_rate: float = 1e-5
    seed: int = 42
    early_stopping_patience: int = 5
    scheduler_patience: int = 2
    imbalance_strategy: str = "class_weighted"
    pretrained: bool = True
    target_recall: float = 0.9


@dataclass
class TrainingResult:
    model: nn.Module
    checkpoint: dict[str, object]
    history: list[dict[str, float]]
    validation_labels: list[int]
    validation_scores: list[float]
    threshold_selection: dict[str, float]


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


def make_optimizer(
    model: nn.Module,
    architecture: Architecture,
    head_learning_rate: float,
    backbone_learning_rate: float,
) -> torch.optim.Optimizer:
    classifier = model.fc if architecture == "resnet50" else model.classifier
    classifier_parameters = list(classifier.parameters())
    classifier_ids = {id(parameter) for parameter in classifier_parameters}
    backbone_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in classifier_ids
    ]
    return torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": backbone_learning_rate},
            {"params": classifier_parameters, "lr": head_learning_rate},
        ]
    )


def _make_train_loader(
    frame: pd.DataFrame,
    config: TrainingConfig,
    generator: torch.Generator,
) -> DataLoader:
    sampler = None
    shuffle = True
    if config.imbalance_strategy == "weighted_sampler":
        counts = frame["label"].value_counts()
        sample_weights = torch.tensor(
            [1.0 / counts[label] for label in frame["label"]],
            dtype=torch.double,
        )
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(frame),
            replacement=True,
            generator=generator,
        )
        shuffle = False
    return DataLoader(
        FundusDataset(frame, training=True),
        batch_size=config.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=0,
        generator=generator,
    )


def train_with_validation(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    config: TrainingConfig,
    checkpoint_path: Path,
) -> TrainingResult:
    if config.imbalance_strategy not in {
        "class_weighted",
        "weighted_sampler",
        "none",
    }:
        raise ValueError(f"Unknown imbalance strategy: {config.imbalance_strategy}")
    if config.epochs < 1 or config.batch_size < 1:
        raise ValueError("Epochs and batch size must be positive.")
    if config.early_stopping_patience < 1 or config.scheduler_patience < 0:
        raise ValueError("Early-stopping/scheduler patience values are invalid.")
    if not 0 < config.target_recall <= 1:
        raise ValueError("Target recall must be greater than zero and at most one.")
    if set(train_frame["label"]) != {"normal", "glaucoma"}:
        raise ValueError("Training data must contain both classes.")
    if set(validation_frame["label"]) != {"normal", "glaucoma"}:
        raise ValueError("Validation data must contain both classes.")
    if set(train_frame["image_path"]) & set(validation_frame["image_path"]):
        raise ValueError("The same image cannot appear in training and validation.")

    seed_everything(config.seed)
    device = select_device()
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = _make_train_loader(train_frame, config, generator)
    train_metrics_loader = DataLoader(
        FundusDataset(train_frame),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )
    validation_loader = DataLoader(
        FundusDataset(validation_frame),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )
    model = build_model(config.architecture, pretrained=config.pretrained).to(device)
    class_weights = (
        class_weights_from_frame(train_frame).to(device)
        if config.imbalance_strategy == "class_weighted"
        else None
    )
    loss_function = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = make_optimizer(
        model,
        config.architecture,
        config.head_learning_rate,
        config.backbone_learning_rate,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=config.scheduler_patience,
        min_lr=1e-7,
    )
    training_config = {
        **asdict(config),
        "loss": "CrossEntropyLoss",
        "class_weights": (
            class_weights.cpu().tolist() if class_weights is not None else None
        ),
        "optimizer": "AdamW with separate backbone and classifier learning rates",
        "scheduler": "ReduceLROnPlateau on validation loss",
        "early_stopping_monitor": "validation PR-AUC",
        "checkpoint_monitor": "validation PR-AUC (average precision)",
        "threshold_policy": (
            "Choose highest-precision validation threshold meeting target recall"
        ),
        "image_size": 224,
        "preprocessing_version": PREPROCESSING_VERSION,
        "augmentation": (
            "Training only: horizontal flip, rotation ±15 degrees, "
            "light random resized crop, mild brightness/contrast/saturation jitter"
        ),
        "device": str(device),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "reproducibility_note": (
            "Random generators are seeded; exact bitwise determinism can vary "
            "by hardware, PyTorch build, and accelerator kernels."
        ),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

    history: list[dict[str, float]] = []
    best_validation_pr_auc = -1.0
    epochs_without_improvement = 0
    best_checkpoint: dict[str, object] | None = None

    for epoch in range(1, config.epochs + 1):
        run_epoch(model, train_loader, loss_function, device, optimizer)
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
            accuracy_score(train_labels, np.asarray(train_scores) >= 0.5)
        )
        validation_accuracy = float(
            accuracy_score(validation_labels, np.asarray(validation_scores) >= 0.5)
        )
        validation_pr_auc = float(
            average_precision_score(validation_labels, validation_scores)
        )
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "train_accuracy": train_accuracy,
            "validation_accuracy": validation_accuracy,
            "validation_pr_auc": validation_pr_auc,
            "backbone_learning_rate": float(optimizer.param_groups[0]["lr"]),
            "head_learning_rate": float(optimizer.param_groups[1]["lr"]),
        }
        history.append(record)
        print(
            f"Epoch {epoch}/{config.epochs}: "
            f"train_loss={train_loss:.4f}, validation_loss={validation_loss:.4f}, "
            f"validation_recall@0.5="
            f"{np.mean(np.asarray(validation_scores)[np.asarray(validation_labels) == 1] >= 0.5):.4f}, "
            f"validation_pr_auc={validation_pr_auc:.4f}"
        )
        scheduler.step(validation_loss)

        if validation_pr_auc > best_validation_pr_auc:
            best_validation_pr_auc = validation_pr_auc
            epochs_without_improvement = 0
            best_checkpoint = {
                "architecture": config.architecture,
                "class_order": ["normal", "glaucoma"],
                "image_size": 224,
                "preprocessing_version": PREPROCESSING_VERSION,
                "dataset_name": config.dataset_name,
                "epoch": epoch,
                "validation_pr_auc": validation_pr_auc,
                "training_config": training_config,
                "model_state_dict": {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                },
            }
            torch.save(best_checkpoint, checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                print(
                    "Early stopping: validation PR-AUC did not improve for "
                    f"{config.early_stopping_patience} epochs."
                )
                break

    if best_checkpoint is None:
        raise RuntimeError("Training did not produce a valid best checkpoint.")
    model.load_state_dict(best_checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    _, validation_labels, validation_scores = run_epoch(
        model,
        validation_loader,
        nn.CrossEntropyLoss(),
        device,
    )
    threshold_selection = select_threshold_for_recall(
        validation_labels,
        validation_scores,
        config.target_recall,
    )
    best_checkpoint["threshold"] = threshold_selection["threshold"]
    best_checkpoint["threshold_selection"] = threshold_selection
    best_checkpoint["training_config"] = training_config
    torch.save(best_checkpoint, checkpoint_path)
    (checkpoint_path.parent / "training_config.json").write_text(
        json.dumps(
            {
                **training_config,
                "selected_threshold": threshold_selection,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (checkpoint_path.parent / "training_history.csv").write_text(
        pd.DataFrame(history).to_csv(index=False),
        encoding="utf-8",
    )
    save_precision_recall_threshold_plot(
        validation_labels,
        validation_scores,
        threshold_selection["threshold"],
        checkpoint_path.parent,
    )
    return TrainingResult(
        model=model,
        checkpoint=best_checkpoint,
        history=history,
        validation_labels=validation_labels,
        validation_scores=validation_scores,
        threshold_selection=threshold_selection,
    )
