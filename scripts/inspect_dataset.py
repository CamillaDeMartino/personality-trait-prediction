import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pickle

import pandas as pd

from src.config import RAW_DATA_DIR, TRAITS


DATASET_DIR = RAW_DATA_DIR / "first_impressions_v2"

SPLITS = {
    "train": {
        "video_dir": DATASET_DIR / "train",
        "annotation_file": DATASET_DIR / "annotations" / "train-annotation" / "annotation_training.pkl",
    },
    "val": {
        "video_dir": DATASET_DIR / "val",
        "annotation_file": DATASET_DIR / "annotations" / "val-annotation" / "annotation_validation.pkl",
    },
    "test": {
        "video_dir": DATASET_DIR / "test",
        "annotation_file": DATASET_DIR / "annotations" / "test-annotation" / "annotation_test.pkl",
    },
}


def load_annotations(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def annotations_to_dataframe(annotations: dict) -> pd.DataFrame:
    rows = []

    filenames = set()
    for trait in TRAITS:
        filenames.update(annotations[trait].keys())

    for filename in sorted(filenames):
        row = {"filename": filename}
        for trait in TRAITS:
            row[trait] = float(annotations[trait][filename])
        rows.append(row)

    return pd.DataFrame(rows)


def inspect_split(split_name: str, video_dir: Path, annotation_file: Path):
    print("\n" + "=" * 80)
    print(f"SPLIT: {split_name}")
    print("=" * 80)

    if not video_dir.exists():
        raise FileNotFoundError(f"Missing video directory: {video_dir}")

    if not annotation_file.exists():
        raise FileNotFoundError(f"Missing annotation file: {annotation_file}")

    video_files = sorted([p.name for p in video_dir.glob("*.mp4")])
    video_set = set(video_files)

    annotations = load_annotations(annotation_file)
    df = annotations_to_dataframe(annotations)
    annotation_set = set(df["filename"].tolist())

    missing_labels = sorted(video_set - annotation_set)
    missing_videos = sorted(annotation_set - video_set)

    print(f"Video directory : {video_dir}")
    print(f"Annotation file : {annotation_file}")
    print(f"Videos          : {len(video_files)}")
    print(f"Annotations     : {len(df)}")
    print(f"Videos w/o label: {len(missing_labels)}")
    print(f"Labels w/o video: {len(missing_videos)}")

    if missing_labels:
        print("\nFirst videos without label:")
        print(missing_labels[:10])

    if missing_videos:
        print("\nFirst labels without video:")
        print(missing_videos[:10])

    print("\nTrait statistics:")
    print(df[TRAITS].describe().T[["mean", "std", "min", "max"]])

    return df


def main():
    print("=" * 80)
    print("Inspecting First Impressions V2 dataset")
    print("=" * 80)
    print(f"Dataset path: {DATASET_DIR}")

    all_dfs = []

    for split_name, paths in SPLITS.items():
        df = inspect_split(
            split_name=split_name,
            video_dir=paths["video_dir"],
            annotation_file=paths["annotation_file"],
        )
        df["split"] = split_name
        all_dfs.append(df)

    full_df = pd.concat(all_dfs, ignore_index=True)

    print("\n" + "=" * 80)
    print("GLOBAL SUMMARY")
    print("=" * 80)
    print(full_df.groupby("split").size())

    print("\nGlobal trait statistics:")
    print(full_df[TRAITS].describe().T[["mean", "std", "min", "max"]])


if __name__ == "__main__":
    main()