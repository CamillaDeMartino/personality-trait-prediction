import torch
import torch.nn as nn

from src.networks.audio_encoder import AudioEncoder
from src.networks.video_encoder import VideoEncoder
from src.networks.fusion import ConcatFusion, GatedFusion
from src.networks.regression_head import RegressionHead


class PersonalityNet(nn.Module):
    """
    Multimodal Personality Prediction Network.

    Inputs
    ------
    audio : [B, 1, 24, 1319]
    video : [B, 6, 3, 128, 128]

    Output
    ------
    personality scores : [B, 5]
    """

    def __init__(
        self,
        feature_dim: int = 256,
        pretrained_video: bool = True,
        freeze_backbone: bool = True,
        fusion_type: str = "concat",

    ):
        super().__init__()

        self.audio_encoder = AudioEncoder(
            output_dim=feature_dim
        )

        self.video_encoder = VideoEncoder(
            output_dim=feature_dim,
            pretrained=pretrained_video,
            freeze_backbone=freeze_backbone,
        )

        if fusion_type == "concat":
            self.fusion = ConcatFusion(
                audio_dim=feature_dim,
                video_dim=feature_dim,
                output_dim=feature_dim,
            )
        elif fusion_type == "gated":
            self.fusion = GatedFusion(
                audio_dim=feature_dim,
                video_dim=feature_dim,
                output_dim=feature_dim,
            )
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

        self.regression_head = RegressionHead(
            input_dim=feature_dim,
            num_traits=5,
        )

    def forward(
        self,
        audio: torch.Tensor,
        video: torch.Tensor,
    ) -> torch.Tensor:

        audio_features = self.audio_encoder(audio)

        video_features = self.video_encoder(video)

        fused_features = self.fusion(
            audio_features,
            video_features,
        )

        predictions = self.regression_head(
            fused_features
        )

        return predictions