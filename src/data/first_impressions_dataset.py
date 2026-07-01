from pathlib import Path
from typing import Dict, List

import pickle
import torch
from torch.utils.data import Dataset

from src.data.preprocessing import preprocess_sample


class FirstImpressionsDataset(Dataset):
    """
    PyTorch Dataset for the ChaLearn First Impressions V2 dataset.
    """

    def __init__(
        self,
        video_dir: str | Path,
        annotation_file: str | Path,
        training: bool = True,
    ):

        self.video_dir = Path(video_dir)
        self.training = training

        with open(annotation_file, "rb") as f:
            self.annotations: Dict[str, Dict[str, float]] = pickle.load(
                f,
                encoding="latin1",
            )

        self.video_files: List[Path] = sorted(
            [
                file
                for file in self.video_dir.iterdir()
                if file.is_file()
            ]
        )

    def __len__(self):

        return len(self.video_files)

    def __getitem__(self, index):

        video_path = self.video_files[index]

        audio, video, labels = preprocess_sample(
            video_path=video_path,
            annotations=self.annotations,
            training=self.training,
        )

        audio = torch.from_numpy(audio).permute(2, 0, 1).float()

        video = torch.from_numpy(video).permute(0, 3, 1, 2).float()

        labels = torch.from_numpy(labels).float()

        return {
            "audio": audio,
            "video": video,
            "labels": labels,
            "filename": video_path.name,
        }