import torch
import torch.nn as nn


class AudioEncoder(nn.Module):
    """
    CNN encoder for MFCC audio features.

    Input:
        audio: [B, 1, 24, 1319]

    Output:
        features: [B, output_dim]
    """

    def __init__(self, output_dim: int = 256):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        x = self.features(audio)
        x = self.projector(x)
        return x