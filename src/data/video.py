from pathlib import Path
from typing import List

import cv2
import numpy as np

from src.config import (
    CROP_SIZE,
    N_VIDEO_FRAMES,
    RESIZE_HEIGHT,
    RESIZE_WIDTH,
)


def get_number_of_frames(video_path: str | Path) -> int:
    """
    Return the number of frames in a video using OpenCV.
    """
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if frame_count <= 0:
        raise ValueError(f"Could not determine frame count for: {video_path}")

    return frame_count

def sample_frame_indices(
    total_frames: int,
    num_samples: int = N_VIDEO_FRAMES,
    training: bool = True,
) -> List[int]:
    """
    Sample frame indices from a video.

    During training, frames are sampled randomly.
    During validation/test, frames are sampled uniformly.
    """
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")

    if total_frames <= num_samples:
        return list(range(total_frames)) + [total_frames - 1] * (num_samples - total_frames)

    if training:
        indices = np.random.choice(total_frames, size=num_samples, replace=False)
        return sorted(indices.tolist())

    return np.linspace(0, total_frames - 1, num_samples, dtype=int).tolist()


def read_frames(
    video_path: str | Path,
    frame_indices: List[int],
) -> List[np.ndarray]:
    """
    Read selected frames from a video and convert them from BGR to RGB.
    """
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    frames = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()

        if not success or frame is None:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"No frames could be read from: {video_path}")

    while len(frames) < len(frame_indices):
        frames.append(frames[-1].copy())

    return frames


def resize_frame(
    frame: np.ndarray,
    width: int = RESIZE_WIDTH,
    height: int = RESIZE_HEIGHT,
) -> np.ndarray:
    """
    Resize a frame.
    """
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def crop_frame(
    frame: np.ndarray,
    crop_size: int = CROP_SIZE,
    training: bool = True,
) -> np.ndarray:
    """
    Crop a frame.

    Training: random crop.
    Validation/test: center crop.
    """
    h, w, _ = frame.shape

    if h < crop_size or w < crop_size:
        raise ValueError(
            f"Frame too small for crop_size={crop_size}. "
            f"Frame shape: {frame.shape}"
        )

    if training:
        top = np.random.randint(0, h - crop_size + 1)
        left = np.random.randint(0, w - crop_size + 1)
    else:
        top = (h - crop_size) // 2
        left = (w - crop_size) // 2

    return frame[top:top + crop_size, left:left + crop_size, :]


def preprocess_frame(
    frame: np.ndarray,
    training: bool = True,
) -> np.ndarray:
    """
    Resize, crop and normalize a single frame.

    Output range: [0, 1]
    Output shape: (crop_size, crop_size, 3)
    """
    frame = resize_frame(frame)
    frame = crop_frame(frame, training=training)
    frame = frame.astype(np.float32) / 255.0
    return frame


def preprocess_video(
    video_path: str | Path,
    num_frames: int = N_VIDEO_FRAMES,
    training: bool = True,
) -> np.ndarray:
    """
    Full video preprocessing pipeline.

    Video
        -> selected frames
        -> resize
        -> crop
        -> normalize

    Output shape:
    (num_frames, crop_size, crop_size, 3)
    """
    total_frames = get_number_of_frames(video_path)
    frame_indices = sample_frame_indices(
        total_frames=total_frames,
        num_samples=num_frames,
        training=training,
    )

    frames = read_frames(video_path, frame_indices)
    processed_frames = [
        preprocess_frame(frame, training=training)
        for frame in frames
    ]

    return np.stack(processed_frames).astype(np.float32)