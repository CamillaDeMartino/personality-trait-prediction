import torch
import torch.nn as nn
from tqdm import tqdm

from src.evaluation.metrics import compute_metrics


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()

    criterion = nn.L1Loss()
    total_loss = 0.0
    all_predictions = []
    all_targets = []

    for batch in tqdm(dataloader, desc="Training", leave=False):
        audio = batch["audio"].to(device)
        video = batch["video"].to(device)
        targets = batch["labels"].to(device)

        optimizer.zero_grad()

        predictions = model(audio, video)
        loss = criterion(predictions, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * audio.size(0)

        all_predictions.append(predictions.detach().cpu())
        all_targets.append(targets.detach().cpu())

    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    avg_loss = total_loss / len(dataloader.dataset)
    metrics = compute_metrics(all_predictions, all_targets)
    metrics["loss"] = avg_loss

    return metrics


@torch.no_grad()
def validate_one_epoch(model, dataloader, device):
    model.eval()

    criterion = nn.L1Loss()
    total_loss = 0.0
    all_predictions = []
    all_targets = []

    for batch in tqdm(dataloader, desc="Validation", leave=False):
        audio = batch["audio"].to(device)
        video = batch["video"].to(device)
        targets = batch["labels"].to(device)

        predictions = model(audio, video)
        loss = criterion(predictions, targets)

        total_loss += loss.item() * audio.size(0)

        all_predictions.append(predictions.cpu())
        all_targets.append(targets.cpu())

    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    avg_loss = total_loss / len(dataloader.dataset)
    metrics = compute_metrics(all_predictions, all_targets)
    metrics["loss"] = avg_loss

    return metrics