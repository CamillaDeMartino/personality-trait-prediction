import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.config import (
    TRAIN_DIR,
    TRAIN_ANNOTATIONS,
    SEED,
)
from src.data.first_impressions_dataset import FirstImpressionsDataset
from src.utils import set_seed


def main():

    set_seed(SEED)

    print("=" * 60)
    print("Checking First Impressions V2 Dataloader")
    print("=" * 60)

    dataset = FirstImpressionsDataset(
        video_dir=TRAIN_DIR,
        annotation_file=TRAIN_ANNOTATIONS,
        training=True,
    )

    print(f"Dataset size: {len(dataset)}")

    sample = dataset[0]

    print("\nSample information")
    print("------------------------------")
    print(f"Filename : {sample['filename']}")
    print(f"Audio    : {sample['audio'].shape}")
    print(f"Video    : {sample['video'].shape}")
    print(f"Labels   : {sample['labels']}")

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
    )

    batch = next(iter(loader))

    print("\nBatch information")
    print("------------------------------")
    print(f"Audio batch : {batch['audio'].shape}")
    print(f"Video batch : {batch['video'].shape}")
    print(f"Labels batch: {batch['labels'].shape}")

    fig, axes = plt.subplots(2, 3, figsize=(10, 6))

    for i in range(6):

        frame = sample["video"][i].permute(1, 2, 0).numpy()

        ax = axes[i // 3][i % 3]
        ax.imshow(frame)
        ax.set_title(f"Frame {i+1}")
        ax.axis("off")

    output_dir = PROJECT_ROOT / "outputs" / "figures" / "check_dataloader"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "dataset_check.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"\nSaved dataset preview to: {output_path}")

if __name__ == "__main__":
    main()