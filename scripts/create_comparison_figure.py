import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def main():
    baseline_dir = PROJECT_ROOT / "outputs" / "figures" / "prediction_analysis"
    plateau_dir = PROJECT_ROOT / "outputs" / "figures" / "prediction_analysis_plateau"

    output_dir = PROJECT_ROOT / "outputs" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    images = [
        (
            "Prediction vs Ground Truth",
            baseline_dir / "scatter_extraversion_concat.png",
            plateau_dir / "scatter_extraversion_concat_plateau.png",
        ),
        (
            "Distribution",
            baseline_dir / "hist_extraversion_concat.png",
            plateau_dir / "hist_extraversion_concat_plateau.png",
        ),
        (
            "Residuals",
            baseline_dir / "residuals_extraversion_concat.png",
            plateau_dir / "residuals_extraversion_concat_plateau.png",
        ),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(10, 13))

    column_titles = ["Baseline", "Final (+ ReduceLROnPlateau)"]

    for col, title in enumerate(column_titles):
        axes[0, col].set_title(title, fontsize=14, fontweight="bold", pad=12)

    for row, (row_title, baseline_path, plateau_path) in enumerate(images):
        for col, path in enumerate([baseline_path, plateau_path]):
            if not path.exists():
                raise FileNotFoundError(f"Missing image: {path}")

            img = mpimg.imread(path)
            axes[row, col].imshow(img)
            axes[row, col].axis("off")

        axes[row, 0].set_ylabel(
            row_title,
            fontsize=12,
            fontweight="bold",
            rotation=90,
            labelpad=20,
        )

    plt.tight_layout()

    png_path = output_dir / "prediction_comparison_extraversion.png"

    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
