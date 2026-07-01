import torch
import torch.nn as nn


class RegressionHead(nn.Module):
    """
    Regression head for Big Five personality traits.

    Input:
        features: [B, input_dim]

    Output:
        predictions: [B, num_traits]
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        num_traits: int = 5,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_traits),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features)