from collections.abc import Iterable

import torch
from torch import nn


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_epoch(
    model: nn.Module,
    loader: Iterable,
    loss_function: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, list[int], list[float]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    sample_count = 0
    labels_all: list[int] = []
    scores_all: list[float] = []

    context = torch.enable_grad if training else torch.inference_mode
    with context():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            if optimizer:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_function(logits, labels)
            if optimizer:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            sample_count += batch_size
            probabilities = torch.softmax(logits, dim=1)[:, 1]
            labels_all.extend(labels.detach().cpu().tolist())
            scores_all.extend(probabilities.detach().cpu().tolist())

    if not sample_count:
        raise ValueError("Cannot run a training epoch with an empty data loader.")
    return total_loss / sample_count, labels_all, scores_all
