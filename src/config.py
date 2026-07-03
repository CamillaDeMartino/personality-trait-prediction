from pathlib import Path

# =========================
# Project paths
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

DATASET_NAME = "first_impressions_v2"
FIRST_IMPRESSIONS_DIR = RAW_DATA_DIR / DATASET_NAME

TRAIN_DIR = FIRST_IMPRESSIONS_DIR / "train"
VALID_DIR = FIRST_IMPRESSIONS_DIR / "val"
TEST_DIR = FIRST_IMPRESSIONS_DIR / "test"

MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
EXPORTED_MODELS_DIR = MODELS_DIR / "exported"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
LOGS_DIR = OUTPUTS_DIR / "logs"

REPORTS_DIR = PROJECT_ROOT / "reports"


# =========================
# Dataset files
# =========================
ANNOTATIONS_DIR = FIRST_IMPRESSIONS_DIR / "annotations"

TRAIN_ANNOTATIONS = (
    ANNOTATIONS_DIR / "train-annotation" / "annotation_training.pkl"
)

VALID_ANNOTATIONS = (
    ANNOTATIONS_DIR / "val-annotation" / "annotation_validation.pkl"
)

TEST_ANNOTATIONS = (
    ANNOTATIONS_DIR / "test-annotation" / "annotation_test.pkl"
)

TRAIN_PROCESSED = PROCESSED_DATA_DIR / "train_processed.pkl"
VALID_PROCESSED = PROCESSED_DATA_DIR / "valid_processed.pkl"
TEST_PROCESSED = PROCESSED_DATA_DIR / "test_processed.pkl"


# =========================
# Personality traits
# =========================

TRAITS = [
    "extraversion",
    "neuroticism",
    "agreeableness",
    "conscientiousness",
    "openness",
]

TRAIT_NAMES_PRETTY = [
    "Extraversion",
    "Neuroticism",
    "Agreeableness",
    "Conscientiousness",
    "Openness",
]

N_TRAITS = len(TRAITS)


# =========================
# Audio parameters
# =========================

SAMPLE_RATE = 44100
N_MFCC = 24
MFCC_MAX_FRAMES = 1319
AUDIO_INPUT_SHAPE = (N_MFCC, MFCC_MAX_FRAMES, 1)


# =========================
# Video parameters
# =========================

N_VIDEO_FRAMES = 6
RESIZE_WIDTH = 248
RESIZE_HEIGHT = 140
CROP_SIZE = 128
VIDEO_INPUT_SHAPE = (N_VIDEO_FRAMES, CROP_SIZE, CROP_SIZE, 3)


# =========================
# Training parameters
# =========================

SEED = 42
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 10

FAST_DEV_RUN = False  # set to True for a quick run to check for bugs
FAST_TRAIN_SIZE = 1000
FAST_VALID_SIZE = 400
FAST_EPOCHS = 5

# =========================
# PyTorch parameters
# =========================

DEVICE = "cuda"  # will fallback to CPU in the training script if CUDA is not available
NUM_WORKERS = 4
PIN_MEMORY = True

# ==========================================
# Model
# ==========================================

FUSION_TYPE = "concat"


# =========================
# Utility
# =========================

def create_project_dirs() -> None:
    """
    Create all directories required by the project.
    Safe to call multiple times.
    """
    dirs = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        FIRST_IMPRESSIONS_DIR,
        TRAIN_DIR,
        VALID_DIR,
        TEST_DIR,
        GT_DIR,
        CHECKPOINTS_DIR,
        EXPORTED_MODELS_DIR,
        FIGURES_DIR,
        METRICS_DIR,
        PREDICTIONS_DIR,
        LOGS_DIR,
        REPORTS_DIR,
    ]

    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)