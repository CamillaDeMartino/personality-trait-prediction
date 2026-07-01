import torch

from src.config import TRAIT_NAMES_PRETTY


def mae(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Mean Absolute Error over all samples and traits.
    """
    return torch.mean(torch.abs(predictions - targets))


def one_minus_mae(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Official-style metric used in First Impressions:
    1 - MAE.
    """
    return 1.0 - mae(predictions, targets)


def trait_mae(predictions: torch.Tensor, targets: torch.Tensor) -> dict:
    """
    MAE for each Big Five trait.
    """
    errors = torch.mean(torch.abs(predictions - targets), dim=0)

    return {
        trait: errors[i].item()
        for i, trait in enumerate(TRAIT_NAMES_PRETTY)
    }


def compute_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> dict:
    """
    Compute all evaluation metrics.
    """
    return {
        "mae": mae(predictions, targets).item(),
        "one_minus_mae": one_minus_mae(predictions, targets).item(),
        "trait_mae": trait_mae(predictions, targets),
    }