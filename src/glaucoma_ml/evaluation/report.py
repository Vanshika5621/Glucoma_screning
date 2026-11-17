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
    average_precision_score,
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
    threshold: float = 0.5,
    bootstrap_samples: int = 2000,
    seed: int = 42,
) -> dict[str, object]:
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between zero and one.")
    if len(labels) != len(scores) or not labels:
        raise ValueError("Labels and scores must be non-empty and have equal lengths.")

    predicted = [int(score >= threshold) for score in scores]
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
        pr_auc = float(average_precision_score(labels, scores))
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
        pr_auc = None
        curve = []

    label_array = np.asarray(labels, dtype=np.int64)
    score_array = np.asarray(scores, dtype=np.float64)
    recall_ci, roc_auc_ci = bootstrap_confidence_intervals(
        label_array,
        score_array,
        threshold,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    normal_baseline_accuracy = float(np.mean(label_array == 0))
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
        "pr_auc": pr_auc,
        "recall_ci_95": recall_ci,
        "roc_auc_ci_95": roc_auc_ci,
        "normal_baseline_accuracy": normal_baseline_accuracy,
        "positive_samples": int(np.sum(label_array == 1)),
        "confusion_matrix": matrix.tolist(),
        "training_history": training_history,
        "roc_curve": curve,
        "class_order": ["normal", "glaucoma"],
        "threshold": threshold,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }
    if evaluation_split in ("test", "official_test"):
        report["test_samples"] = len(labels)
    return report


def bootstrap_confidence_intervals(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    bootstrap_samples: int = 2000,
    seed: int = 42,
) -> tuple[dict[str, float] | None, dict[str, float] | None]:
    if bootstrap_samples < 1:
        raise ValueError("Bootstrap sample count must be positive.")
    if set(np.unique(labels)) != {0, 1}:
        return None, None

    rng = np.random.default_rng(seed)
    negative_indices = np.flatnonzero(labels == 0)
    positive_indices = np.flatnonzero(labels == 1)
    recall_values = np.empty(bootstrap_samples, dtype=np.float64)
    roc_auc_values = np.empty(bootstrap_samples, dtype=np.float64)

    for iteration in range(bootstrap_samples):
        sample_indices = np.concatenate(
            [
                rng.choice(negative_indices, size=len(negative_indices), replace=True),
                rng.choice(positive_indices, size=len(positive_indices), replace=True),
            ]
        )
        sampled_labels = labels[sample_indices]
        sampled_scores = scores[sample_indices]
        sampled_predictions = sampled_scores >= threshold
        recall_values[iteration] = np.mean(
            sampled_predictions[sampled_labels == 1]
        )
        roc_auc_values[iteration] = roc_auc_score(
            sampled_labels,
            sampled_scores,
        )

    recall_lower, recall_upper = np.percentile(recall_values, [2.5, 97.5])
    auc_lower, auc_upper = np.percentile(roc_auc_values, [2.5, 97.5])
    return (
        {"lower": float(recall_lower), "upper": float(recall_upper)},
        {"lower": float(auc_lower), "upper": float(auc_upper)},
    )


def select_threshold_for_recall(
    labels: list[int],
    scores: list[float],
    target_recall: float = 0.9,
) -> dict[str, float]:
    if not 0 < target_recall <= 1:
        raise ValueError("Target recall must be greater than zero and at most one.")
    if len(labels) != len(scores) or not labels:
        raise ValueError("Labels and scores must be non-empty and have equal lengths.")
    if set(labels) != {0, 1}:
        raise ValueError("Threshold tuning requires both classes in validation data.")
    if not np.isfinite(scores).all() or any(score < 0 or score > 1 for score in scores):
        raise ValueError("Validation scores must be finite probabilities between zero and one.")

    candidates = sorted({0.0, *scores}, reverse=True)
    eligible: list[tuple[float, float, float]] = []
    for threshold in candidates:
        predictions = [int(score >= threshold) for score in scores]
        recall = float(recall_score(labels, predictions, zero_division=0))
        if recall >= target_recall:
            precision = float(
                precision_score(labels, predictions, zero_division=0)
            )
            eligible.append((precision, threshold, recall))
    if not eligible:
        raise ValueError("No validation threshold meets the requested recall.")
    precision, threshold, recall = max(eligible)
    return {
        "threshold": float(threshold),
        "target_recall": float(target_recall),
        "validation_recall": recall,
        "validation_precision": precision,
    }


def save_precision_recall_threshold_plot(
    labels: list[int],
    scores: list[float],
    threshold: float,
    output_dir: Path,
) -> None:
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(labels, scores)
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(thresholds, precision[:-1], label="Precision")
    axis.plot(thresholds, recall[:-1], label="Recall")
    axis.axvline(
        threshold,
        color="gray",
        linestyle="--",
        label=f"Selected threshold {threshold:.3f}",
    )
    axis.set(
        xlabel="Decision threshold",
        ylabel="Precision / recall",
        title="Validation precision and recall across thresholds",
        ylim=(0, 1.02),
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "validation_thresholds.png", dpi=150)
    plt.close(figure)


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
