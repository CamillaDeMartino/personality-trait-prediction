from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from src.config import TRAITS
from src.data.audio import preprocess_audio
from src.data.video import preprocess_video


def read_labels(
    file_name: str,
    annotations: Dict[str, Dict[str, float]],
) -> np.ndarray:
    """
    Read Big Five labels for a single video.

    Output shape:
    (5,)
    """
    labels = [float(annotations[trait][file_name]) for trait in TRAITS]
    return np.array(labels, dtype=np.float32)


def preprocess_sample(
    video_path: str | Path,
    annotations: Dict[str, Dict[str, float]],
    training: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess one video sample.

    Returns
    -------
    audio : np.ndarray
        Shape (24, 1319, 1)

    video : np.ndarray
        Shape (6, 128, 128, 3)

    labels : np.ndarray
        Shape (5,)
    """
    video_path = Path(video_path)
    file_name = video_path.name

    audio = preprocess_audio(video_path)
    video = preprocess_video(video_path, training=training)
    labels = read_labels(file_name, annotations)

    return audio, video, labels