from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def create_report(
    labels: list[int],
    scores: list[float],
    training_history: list[dict[str, float]],
    architecture: str,
    dataset_name: str,
    evaluation_split: str = "test",
) -> dict[str, object]:
    predicted = [int(score >= 0.5) for score in scores]
    matrix = confusion_matrix(labels, predicted, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    specificity_denominator = true_negative + false_positive
    specificity = (
        float(true_negative / specificity_denominator)
        if specificity_denominator
        else 0.0
    )

    if len(set(labels)) == 2:
        auc = float(roc_auc_score(labels, scores))
        false_positive_rates, true_positive_rates, _ = roc_curve(labels, scores)
        curve = [
            {
                "false_positive_rate": float(false_positive_rate),
                "true_positive_rate": float(true_positive_rate),
            }
            for false_positive_rate, true_positive_rate in zip(
                false_positive_rates,
                true_positive_rates,
            )
        ]
    else:
        auc = None
        curve = []

    report = {
        "model_name": architecture,
        "dataset_name": dataset_name,
        "evaluation_split": evaluation_split,
        "evaluation_samples": len(labels),
        "accuracy": float(accuracy_score(labels, predicted)),
        "precision": float(
            precision_score(labels, predicted, zero_division=0)
        ),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1_score": float(f1_score(labels, predicted, zero_division=0)),
        "sensitivity": float(
            recall_score(labels, predicted, zero_division=0)
        ),
        "specificity": specificity,
        "roc_auc": auc,
        "confusion_matrix": matrix.tolist(),
        "training_history": training_history,
        "roc_curve": curve,
        "class_order": ["normal", "glaucoma"],
        "threshold": 0.5,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }
    if evaluation_split in ("test", "official_test"):
        report["test_samples"] = len(labels)
    return report


def save_report(report: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    matrix = np.asarray(report["confusion_matrix"])
    figure, axis = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=["Normal", "Glaucoma"],
        yticklabels=["Normal", "Glaucoma"],
        ax=axis,
    )
    split_name = str(report.get("evaluation_split", "test")).replace("_", " ")
    axis.set(xlabel="Predicted", ylabel="Actual", title=split_name.title())
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(figure)

    history = report["training_history"]
    if isinstance(history, list) and history:
        epochs = [item["epoch"] for item in history]
        for metric, title in (("loss", "Loss"), ("accuracy", "Accuracy")):
            figure, axis = plt.subplots(figsize=(6, 4))
            axis.plot(
                epochs,
                [item[f"train_{metric}"] for item in history],
                label="Train",
            )
            axis.plot(
                epochs,
                [item[f"validation_{metric}"] for item in history],
                label="Validation",
            )
            axis.set(
                xlabel="Epoch",
                ylabel=title,
                title=f"Training and validation {title.lower()}",
            )
            axis.legend()
            figure.tight_layout()
            figure.savefig(output_dir / f"training_{metric}.png", dpi=150)
            plt.close(figure)
            for split in ("train", "validation"):
                figure, axis = plt.subplots(figsize=(6, 4))
                axis.plot(
                    epochs,
                    [item[f"{split}_{metric}"] for item in history],
                    label=split.title(),
                )
                axis.set(
                    xlabel="Epoch",
                    ylabel=title,
                    title=f"{split.title()} {title.lower()}",
                )
                axis.legend()
                figure.tight_layout()
                figure.savefig(output_dir / f"{split}_{metric}.png", dpi=150)
                plt.close(figure)

        figure, axes = plt.subplots(1, 2, figsize=(11, 4))
        for axis, metric, title in zip(
            axes,
            ("loss", "accuracy"),
            ("Loss", "Accuracy"),
        ):
            axis.plot(
                epochs,
                [item[f"train_{metric}"] for item in history],
                label="Train",
            )
            axis.plot(
                epochs,
                [item[f"validation_{metric}"] for item in history],
                label="Validation",
            )
            axis.set(xlabel="Epoch", ylabel=title, title=title)
            axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / "training_history.png", dpi=150)
        plt.close(figure)

    curve = report["roc_curve"]
    if isinstance(curve, list) and curve:
        figure, axis = plt.subplots(figsize=(5, 4))
        axis.plot(
            [point["false_positive_rate"] for point in curve],
            [point["true_positive_rate"] for point in curve],
            label=f"ROC-AUC {report['roc_auc']:.3f}",
        )
        axis.plot([0, 1], [0, 1], linestyle="--", color="gray")
        axis.set(
            xlabel="False positive rate",
            ylabel="True positive rate",
            title=f"{split_name.title()} ROC",
        )
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / "roc_curve.png", dpi=150)
        plt.close(figure)
