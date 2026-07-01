import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.config import DEVICE, FUSION_TYPE
from src.utils import get_device, count_parameters
from src.networks.personality_net import PersonalityNet


def count_total_parameters(model):
    return sum(p.numel() for p in model.parameters())


def main():
    device = get_device(DEVICE)

    model = PersonalityNet(
        pretrained_video=False,
        freeze_backbone=True,
        fusion_type=FUSION_TYPE,
    ).to(device)

    total_params = count_total_parameters(model)
    trainable_params = count_parameters(model)
    frozen_params = total_params - trainable_params

    print("=" * 60)
    print("PersonalityNet Model Summary")
    print("=" * 60)
    print(f"Device              : {device}")
    print(f"Total parameters    : {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters   : {frozen_params:,}")

    audio = torch.randn(2, 1, 24, 1319).to(device)
    video = torch.randn(2, 6, 3, 128, 128).to(device)

    with torch.no_grad():
        out = model(audio, video)

    print("\nForward pass")
    print("-" * 60)
    print(f"Audio input : {audio.shape}")
    print(f"Video input : {video.shape}")
    print(f"Output      : {out.shape}")
    print(f"Output range: [{out.min().item():.4f}, {out.max().item():.4f}]")


if __name__ == "__main__":
    main()