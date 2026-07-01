import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device(preferred_device: str = "cuda") -> torch.device:
    """
    Return CUDA device if available, otherwise CPU.
    """
    if preferred_device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable parameters of a PyTorch model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)