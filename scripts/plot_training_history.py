import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import torch


HISTORY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
    / "training_history_concat_plateau.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
    / "training_analysis"
)


def load_history(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Training history not found: {path}")

    with open(path, "r") as file:
        return json.load(file)


def reconstruct_learning_rates(
    validation_mae: list[float],
    initial_lr: float = 1e-3,
    factor: float = 0.5,
    patience: int = 2,
    min_lr: float = 1e-6,
) -> list[float]:
    """
    Reconstruct the learning-rate evolution using the same
    ReduceLROnPlateau configuration adopted during training.
    """

    dummy_parameter = torch.nn.Parameter(torch.tensor(0.0))

    optimizer = torch.optim.Adam(
        [dummy_parameter],
        lr=initial_lr,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=factor,
        patience=patience,
        min_lr=min_lr,
    )

    learning_rates = []

    for value in validation_mae:
        scheduler.step(value)
        learning_rates.append(optimizer.param_groups[0]["lr"])

    return learning_rates


def plot_mae_curves(
    epochs: list[int],
    train_mae: list[float],
    valid_mae: list[float],
    output_path: Path,
) -> None:

    best_index = min(range(len(valid_mae)), key=valid_mae.__getitem__)
    best_epoch = epochs[best_index]
    best_value = valid_mae[best_index]

    plt.figure(figsize=(9, 5))

    plt.plot(
        epochs,
        train_mae,
        marker="o",
        label="Training MAE",
    )

    plt.plot(
        epochs,
        valid_mae,
        marker="o",
        label="Validation MAE",
    )

    plt.scatter(
        best_epoch,
        best_value,
        s=80,
        zorder=5,
        label=f"Best validation MAE: {best_value:.4f}",
    )

    plt.annotate(
        f"Best epoch: {best_epoch}",
        xy=(best_epoch, best_value),
        xytext=(best_epoch - 5, best_value + 0.003),
        arrowprops={"arrowstyle": "->"},
    )

    plt.xlabel("Epoch")
    plt.ylabel("Mean Absolute Error")
    plt.title("Training and Validation MAE")
    plt.xticks(epochs)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close()


def plot_mae_and_learning_rate(
    epochs: list[int],
    train_mae: list[float],
    valid_mae: list[float],
    learning_rates: list[float],
    output_path: Path,
) -> None:

    figure, mae_axis = plt.subplots(figsize=(10, 5.5))

    mae_axis.plot(
        epochs,
        train_mae,
        marker="o",
        label="Training MAE",
    )

    mae_axis.plot(
        epochs,
        valid_mae,
        marker="o",
        label="Validation MAE",
    )

    best_idx = valid_mae.index(min(valid_mae))
    best_epoch = epochs[best_idx]
    best_value = valid_mae[best_idx]

    mae_axis.scatter(
        best_epoch,
        best_value,
        s=180,
        marker="*",
        color="red",
        zorder=10,
    )

    mae_axis.annotate(
        f"Best MAE\n{best_value:.4f}",
        xy=(best_epoch, best_value),
        xytext=(16.5, best_value + 0.0018),
        arrowprops=dict(arrowstyle="->"),
        fontsize=10,
    )

    mae_axis.set_xlabel("Epoch")
    mae_axis.set_ylabel("Mean Absolute Error")
    mae_axis.set_xticks(epochs)
    mae_axis.grid(alpha=0.3)

    lr_axis = mae_axis.twinx()

    lr_axis.step(
        epochs,
        learning_rates,
        where="post",
        linewidth=2,
        linestyle="--",
        label="Learning Rate",
    )

    # --------------------------------------------------
    # Annotate learning-rate reductions
    # --------------------------------------------------

    lr_axis.annotate(
        "LR ↓ 1e-3 → 5e-4",
        xy=(6, learning_rates[5]),
        xytext=(2.2, 7e-4),
        arrowprops=dict(arrowstyle="->"),
        fontsize=10,
    )

    lr_axis.annotate(
        "LR ↓ 5e-4 → 2.5e-4",
        xy=(16, learning_rates[15]),
        xytext=(12.5, 4.5e-4),
        arrowprops=dict(arrowstyle="->"),
        fontsize=10,
    )
    lr_axis.set_ylabel("Learning Rate")
    lr_axis.set_yscale("log")

    handles_mae, labels_mae = mae_axis.get_legend_handles_labels()
    handles_lr, labels_lr = lr_axis.get_legend_handles_labels()

    mae_axis.legend(
        handles_mae + handles_lr,
        labels_mae + labels_lr,
        loc="upper right",
    )

    plt.title("Training Behaviour and Learning-Rate Schedule")
    figure.tight_layout()

    plt.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(figure)


def main() -> None:

    history = load_history(HISTORY_PATH)

    epochs = [record["epoch"] for record in history]

    train_mae = [
        record["train"]["mae"]
        for record in history
    ]

    valid_mae = [
        record["valid"]["mae"]
        for record in history
    ]

    learning_rates = reconstruct_learning_rates(
        validation_mae=valid_mae,
        initial_lr=1e-3,
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mae_path = OUTPUT_DIR / "training_validation_mae.png"

    combined_path = (
        OUTPUT_DIR
        / "training_validation_mae_learning_rate.png"
    )

    plot_mae_curves(
        epochs=epochs,
        train_mae=train_mae,
        valid_mae=valid_mae,
        output_path=mae_path,
    )

    plot_mae_and_learning_rate(
        epochs=epochs,
        train_mae=train_mae,
        valid_mae=valid_mae,
        learning_rates=learning_rates,
        output_path=combined_path,
    )

    print("=" * 60)
    print("Training figures generated")
    print("=" * 60)
    print(f"MAE curves : {mae_path}")
    print(f"MAE + LR   : {combined_path}")

    print("\nReconstructed learning rates:")
    for epoch, lr in zip(epochs, learning_rates):
        print(f"Epoch {epoch:2d}: {lr:.8f}")


if __name__ == "__main__":
    main()


