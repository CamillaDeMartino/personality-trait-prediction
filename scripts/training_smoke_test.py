import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from torch.utils.data import DataLoader
import torch


from src.config import (
    TRAIN_DIR,
    TRAIN_ANNOTATIONS,
    DEVICE,
    SEED,
    FUSION_TYPE,
    LEARNING_RATE,
)

from src.utils import set_seed, get_device

from src.data.first_impressions_dataset import (
    FirstImpressionsDataset,
)

from src.networks.personality_net import PersonalityNet

from src.training.trainer import train_one_epoch


def main():

    set_seed(SEED)

    device = get_device(DEVICE)

    print(f"Device: {device}")

    dataset = FirstImpressionsDataset(
        video_dir=TRAIN_DIR,
        annotation_file=TRAIN_ANNOTATIONS,
        training=True,
    )

    # Solo pochi campioni
    subset = torch.utils.data.Subset(
        dataset,
        list(range(16)),
    )

    loader = DataLoader(
        subset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )

    model = PersonalityNet(
        pretrained_video=False,
        fusion_type=FUSION_TYPE,
    )

    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    metrics = train_one_epoch(
        model=model,
        dataloader=loader,
        optimizer=optimizer,
        device=device,
    )

    print("\nTraining finished.\n")

    for k, v in metrics.items():

        print(k, ":", v)


if __name__ == "__main__":
    main()