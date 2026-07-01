import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import (
    TEST_DIR,
    TEST_ANNOTATIONS,
    CHECKPOINTS_DIR,
    METRICS_DIR,
    PREDICTIONS_DIR,
    DEVICE,
    BATCH_SIZE,
    NUM_WORKERS,
    SEED,
    FUSION_TYPE,
)

from src.utils import set_seed, get_device
from src.data.first_impressions_dataset import FirstImpressionsDataset
from src.networks.personality_net import PersonalityNet
from src.evaluation.metrics import compute_metrics


def save_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()

    all_predictions = []
    all_targets = []
    all_filenames = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        audio = batch["audio"].to(device)
        video = batch["video"].to(device)
        targets = batch["labels"].to(device)

        predictions = model(audio, video)

        all_predictions.append(predictions.cpu())
        all_targets.append(targets.cpu())
        all_filenames.extend(batch["filename"])

    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    metrics = compute_metrics(all_predictions, all_targets)

    return metrics, all_predictions, all_targets, all_filenames


def main():
    set_seed(SEED)
    device = get_device(DEVICE)

    checkpoint_path = CHECKPOINTS_DIR / f"best_{FUSION_TYPE}_diff_lr.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print("=" * 60)
    print("Evaluating PersonalityNet on test set")
    print("=" * 60)
    print(f"Device     : {device}")
    print(f"Checkpoint : {checkpoint_path}")

    test_dataset = FirstImpressionsDataset(
        video_dir=TEST_DIR,
        annotation_file=TEST_ANNOTATIONS,
        training=False,
    )

    test_loader = DataLoader(
        test_dataset,
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

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Loaded checkpoint from epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Validation MAE at checkpoint: {checkpoint.get('valid_mae', 'unknown')}")

    metrics, predictions, targets, filenames = evaluate(
        model=model,
        dataloader=test_loader,
        device=device,
    )

    print("\nTest metrics")
    print("-" * 60)
    print(metrics)

    save_json(metrics, METRICS_DIR / f"test_metrics_{FUSION_TYPE}_diff_lr.json")

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    prediction_rows = []

    for filename, pred, target in zip(filenames, predictions, targets):
        prediction_rows.append(
            {
                "filename": filename,
                "prediction": pred.tolist(),
                "target": target.tolist(),
            }
        )

    save_json(
        prediction_rows,
        PREDICTIONS_DIR / f"test_predictions_{FUSION_TYPE}_diff_lr.json",
    )

    print(f"\nSaved metrics to: {METRICS_DIR / f'test_metrics_{FUSION_TYPE}_diff_lr.json'}")
    print(f"Saved predictions to: {PREDICTIONS_DIR / f'test_predictions_{FUSION_TYPE}_diff_lr.json'}")


if __name__ == "__main__":
    main()