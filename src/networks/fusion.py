import torch
import torch.nn as nn


class ConcatFusion(nn.Module):
    """
    Basic multimodal fusion by concatenation.

    Input:
        audio_features: [B, audio_dim]
        video_features: [B, video_dim]

    Output:
        fused_features: [B, output_dim]
    """

    def __init__(
        self,
        audio_dim: int = 256,
        video_dim: int = 256,
        hidden_dim: int = 512,
        output_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.fusion = nn.Sequential(
            nn.Linear(audio_dim + video_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        audio_features: torch.Tensor,
        video_features: torch.Tensor,
    ) -> torch.Tensor:

        x = torch.cat([audio_features, video_features], dim=1)
        return self.fusion(x)


class GatedFusion(nn.Module):
    """
    Gated multimodal fusion.

    The model learns how much to trust audio vs video features.
    """

    def __init__(
        self,
        audio_dim: int = 256,
        video_dim: int = 256,
        output_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.audio_proj = nn.Linear(audio_dim, output_dim)
        self.video_proj = nn.Linear(video_dim, output_dim)

        self.gate = nn.Sequential(
            nn.Linear(audio_dim + video_dim, output_dim),
            nn.Sigmoid(),
        )

        self.output = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, audio_features, video_features):
        audio = self.audio_proj(audio_features)
        video = self.video_proj(video_features)

        gate = self.gate(
            torch.cat([audio_features, video_features], dim=1)
        )

        fused = gate * audio + (1.0 - gate) * video

        return self.output(fused)