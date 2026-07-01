import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader, Subset

from src.config import (
    TRAIN_DIR,
    VALID_DIR,
    TRAIN_ANNOTATIONS,
    VALID_ANNOTATIONS,
    CHECKPOINTS_DIR,
    METRICS_DIR,
    DEVICE,
    SEED,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    NUM_WORKERS,
    FUSION_TYPE,
    FAST_DEV_RUN,
    FAST_TRAIN_SIZE,
    FAST_VALID_SIZE,
    FAST_EPOCHS,
)

from src.utils import set_seed, get_device
from src.data.first_impressions_dataset import FirstImpressionsDataset
from src.networks.personality_net import PersonalityNet
from src.training.trainer import train_one_epoch, validate_one_epoch


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def main():
    set_seed(SEED)
    device = get_device(DEVICE)

    print("=" * 60)
    print("Training PersonalityNet")
    print("=" * 60)
    print(f"Device: {device}")

    train_dataset = FirstImpressionsDataset(
        video_dir=TRAIN_DIR,
        annotation_file=TRAIN_ANNOTATIONS,
        training=True,
    )

    valid_dataset = FirstImpressionsDataset(
        video_dir=VALID_DIR,
        annotation_file=VALID_ANNOTATIONS,
        training=False,
    )
    
    if FAST_DEV_RUN:
        train_dataset = Subset(train_dataset, list(range(FAST_TRAIN_SIZE)))
        valid_dataset = Subset(valid_dataset, list(range(FAST_VALID_SIZE)))
        epochs = FAST_EPOCHS
    else:
        epochs = EPOCHS

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    model = PersonalityNet(
        pretrained_video=False,
        freeze_backbone=True,
        fusion_type=FUSION_TYPE,
    ).to(device)

    # optimizer = torch.optim.Adam(
    #     model.parameters(),
    #     lr=LEARNING_RATE,
    # )

    head_params = []
    backbone_layer4_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "video_encoder.backbone.layer4" in name:
            backbone_layer4_params.append(param)
        else:
            head_params.append(param)

    optimizer = torch.optim.Adam(
        [
            {"params": head_params, "lr": LEARNING_RATE},
            {"params": backbone_layer4_params, "lr": LEARNING_RATE * 0.1},
        ]
    )

    best_val_mae = float("inf")
    history = []

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        print("\n" + "=" * 60)
        print(f"Epoch {epoch}/{epochs}")
        print("=" * 60)

        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
        )

        valid_metrics = validate_one_epoch(
            model=model,
            dataloader=valid_loader,
            device=device,
        )

        epoch_record = {
            "epoch": epoch,
            "train": train_metrics,
            "valid": valid_metrics,
        }

        history.append(epoch_record)

        print("\nTrain metrics")
        print(train_metrics)

        print("\nValidation metrics")
        print(valid_metrics)

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "valid_mae": valid_metrics["mae"],
            },
            CHECKPOINTS_DIR / f"last_{FUSION_TYPE}_diff_lr.pt",
        )

        if valid_metrics["mae"] < best_val_mae:
            best_val_mae = valid_metrics["mae"]

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "valid_mae": best_val_mae,
                },
                CHECKPOINTS_DIR / f"best_{FUSION_TYPE}_diff_lr.pt",
            )

            print(f"\nNew best model saved with val MAE: {best_val_mae:.6f}")

        save_json(history, METRICS_DIR / f"training_history_{FUSION_TYPE}_diff_lr.json")

    print("\nTraining completed.")
    print(f"Best validation MAE: {best_val_mae:.6f}")


if __name__ == "__main__":
    main()