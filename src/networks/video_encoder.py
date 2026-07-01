import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class VideoEncoder(nn.Module):
    """
    Video encoder based on ResNet18.

    Input:
        [B, T, 3, 128, 128]

    Output:
        [B, output_dim]
    """

    def __init__(
        self,
        output_dim: int = 256,
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)

        backbone.fc = nn.Identity()

        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False
            
            # Unfreeze only the last ResNet block for light fine-tuning
            for param in backbone.layer4.parameters():
                param.requires_grad = True

        self.backbone = backbone

        self.projector = nn.Sequential(
            nn.Linear(512, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

    def forward(self, video):

        B, T, C, H, W = video.shape

        video = video.view(B * T, C, H, W)

        features = self.backbone(video)

        features = features.view(B, T, 512)

        features = features.mean(dim=1)

        features = self.projector(features)

        return features