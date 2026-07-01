import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import json
from src.config import PREDICTIONS_DIR, FIGURES_DIR, TRAIT_NAMES_PRETTY, FUSION_TYPE
from scipy.stats import pearsonr

def load_predictions(path: Path):
    with open(path, "r") as f:
        data = json.load(f)

    predictions = np.array([item["prediction"] for item in data], dtype=np.float32)
    targets = np.array([item["target"] for item in data], dtype=np.float32)
    filenames = [item["filename"] for item in data]

    return predictions, targets, filenames


def plot_prediction_vs_target(predictions, targets, trait_name, trait_index, output_dir):
    y_true = targets[:, trait_index]
    y_pred = predictions[:, trait_index]

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.35, s=12)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Ground Truth")
    plt.ylabel("Prediction")
    plt.title(f"Prediction vs Ground Truth - {trait_name}")
    plt.tight_layout()

    output_path = output_dir / f"scatter_{trait_name.lower()}_{FUSION_TYPE}.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    return output_path

def plot_histogram(predictions, targets, trait_name, trait_index, output_dir):
    y_true = targets[:, trait_index]
    y_pred = predictions[:, trait_index]

    plt.figure(figsize=(7, 5))
    plt.hist(y_true, bins=30, alpha=0.5, label="Ground Truth")
    plt.hist(y_pred, bins=30, alpha=0.5, label="Prediction")
    plt.xlim(0, 1)
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.title(f"Distribution - {trait_name}")
    plt.legend()
    plt.tight_layout()

    output_path = output_dir / f"hist_{trait_name.lower()}_{FUSION_TYPE}.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    return output_path


def plot_residuals(predictions, targets, trait_name, trait_index, output_dir):
    y_true = targets[:, trait_index]
    y_pred = predictions[:, trait_index]
    residuals = y_pred - y_true

    plt.figure(figsize=(7, 5))
    plt.scatter(y_true, residuals, alpha=0.35, s=12)
    plt.axhline(0, linestyle="--")
    plt.xlim(0, 1)
    plt.xlabel("Ground Truth")
    plt.ylabel("Prediction - Ground Truth")
    plt.title(f"Residuals - {trait_name}")
    plt.tight_layout()

    output_path = output_dir / f"residuals_{trait_name.lower()}_{FUSION_TYPE}.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    return output_path


def compute_trait_statistics(predictions, targets):
    stats = {}

    for i, trait_name in enumerate(TRAIT_NAMES_PRETTY):
        y_true = targets[:, i]
        y_pred = predictions[:, i]
        residuals = y_pred - y_true

        r, p_value = pearsonr(y_true, y_pred)

        stats[trait_name] = {
            "mae": float(np.mean(np.abs(residuals))),
            "rmse": float(np.sqrt(np.mean(residuals ** 2))),
            "pearson_r": float(r),
            "pearson_p_value": float(p_value),
            "bias_mean_prediction_minus_target": float(np.mean(residuals)),
            "target_mean": float(np.mean(y_true)),
            "prediction_mean": float(np.mean(y_pred)),
            "target_std": float(np.std(y_true)),
            "prediction_std": float(np.std(y_pred)),
        }

    return stats

def main():
    predictions_path = PREDICTIONS_DIR / f"test_predictions_{FUSION_TYPE}.json"
    output_dir = FIGURES_DIR / "prediction_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions, targets, filenames = load_predictions(predictions_path)

    print("=" * 60)
    print("Prediction analysis")
    print("=" * 60)
    print(f"Loaded predictions: {len(filenames)}")
    print(f"Predictions shape : {predictions.shape}")
    print(f"Targets shape     : {targets.shape}")

    saved_files = []

    for i, trait_name in enumerate(TRAIT_NAMES_PRETTY):
        scatter_path = plot_prediction_vs_target(
            predictions=predictions,
            targets=targets,
            trait_name=trait_name,
            trait_index=i,
            output_dir=output_dir,
        )

        hist_path = plot_histogram(
            predictions=predictions,
            targets=targets,
            trait_name=trait_name,
            trait_index=i,
            output_dir=output_dir,
        )

        residual_path = plot_residuals(
            predictions=predictions,
            targets=targets,
            trait_name=trait_name,
            trait_index=i,
            output_dir=output_dir,
        )

        saved_files.extend([scatter_path, hist_path, residual_path])

    stats = compute_trait_statistics(predictions, targets)

    analysis_path = output_dir / f"prediction_statistics_{FUSION_TYPE}.json"
    with open(analysis_path, "w") as f:
        json.dump(stats, f, indent=4)

    print("\nPrediction statistics:")
    print(json.dumps(stats, indent=4))

    saved_files.append(analysis_path)

    print("\nSaved files:")
    for path in saved_files:
        print(path)


if __name__ == "__main__":
    main()

