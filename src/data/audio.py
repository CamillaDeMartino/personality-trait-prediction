from pathlib import Path
import ffmpeg
import imageio_ffmpeg
import librosa
import subprocess
import numpy as np

from src.config import SAMPLE_RATE, N_MFCC, MFCC_MAX_FRAMES


def extract_audio_from_video(video_path: str | Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """
    Extract mono audio from a video file using imageio-ffmpeg binary.

    Returns
    -------
    np.ndarray
        Audio waveform as float32 array.
    """
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()

    command = [
        ffmpeg_bin,
        "-i", str(video_path),
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-"
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(
            f"Failed to extract audio from {video_path}\n\nFFmpeg stderr:\n{stderr}"
        )

    audio = np.frombuffer(result.stdout, np.float32)

    if audio.size == 0:
        raise ValueError(f"No audio extracted from {video_path}")

    return audio
    

def compute_mfcc(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    n_mfcc: int = N_MFCC,
) -> np.ndarray:
    """
    Compute MFCC features from raw audio.

    Output shape before padding:
    (n_mfcc, time_frames)
    """
    if audio.ndim != 1:
        raise ValueError(f"Expected mono audio array, got shape {audio.shape}")

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=n_mfcc,
    )

    return mfcc.astype(np.float32)


def standardize_mfcc(mfcc: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Standardize MFCC features to zero mean and unit variance.
    """
    mean = np.mean(mfcc)
    std = np.std(mfcc)

    return ((mfcc - mean) / (std + eps)).astype(np.float32)


def pad_or_truncate_mfcc(
    mfcc: np.ndarray,
    max_frames: int = MFCC_MAX_FRAMES,
) -> np.ndarray:
    """
    Pad or truncate MFCC features along the temporal axis.

    Expected input shape:
    (n_mfcc, time_frames)

    Output shape:
    (n_mfcc, max_frames)
    """
    n_mfcc, current_frames = mfcc.shape

    if current_frames == max_frames:
        return mfcc.astype(np.float32)

    if current_frames > max_frames:
        return mfcc[:, :max_frames].astype(np.float32)

    pad_width = max_frames - current_frames
    padding = np.zeros((n_mfcc, pad_width), dtype=np.float32)

    return np.hstack([padding, mfcc]).astype(np.float32)


def preprocess_audio(
    video_path: str | Path,
    sample_rate: int = SAMPLE_RATE,
    n_mfcc: int = N_MFCC,
    max_frames: int = MFCC_MAX_FRAMES,
) -> np.ndarray:
    """
    Full audio preprocessing pipeline.

    Video
        -> raw mono audio
        -> MFCC
        -> standardization
        -> padding/truncation
        -> channel dimension

    Output shape:
    (n_mfcc, max_frames, 1)
    """
    audio = extract_audio_from_video(video_path, sample_rate=sample_rate)
    mfcc = compute_mfcc(audio, sample_rate=sample_rate, n_mfcc=n_mfcc)
    mfcc = standardize_mfcc(mfcc)
    mfcc = pad_or_truncate_mfcc(mfcc, max_frames=max_frames)

    return mfcc[..., np.newaxis].astype(np.float32)