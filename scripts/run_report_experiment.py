# -*- coding: utf-8 -*-
"""Run the MNIST report experiment and save metrics/plot artifacts."""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from data import load_mnist
from network import NeuralNetwork
from optimizers import Adam
from training import evaluate


def train_with_history(model, optimizer, x_train, y_train, epochs, batch_size):
    """Train while returning average epoch losses."""
    loss_history = []
    train_size = x_train.shape[0]

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        indices = np.random.permutation(train_size)
        epoch_loss = 0.0
        batch_count = 0

        for start in range(0, train_size, batch_size):
            batch_indices = indices[start : start + batch_size]
            x_batch = x_train[batch_indices]
            y_batch = y_train[batch_indices]

            loss = model.gradient(x_batch, y_batch)
            optimizer.update(model.params, model.grads)

            epoch_loss += loss
            batch_count += 1

        average_loss = epoch_loss / batch_count
        loss_history.append(float(average_loss))
        elapsed = time.perf_counter() - epoch_start
        print(f"epoch {epoch:02d}/{epochs} loss={average_loss:.6f} time={elapsed:.2f}s", flush=True)

    return loss_history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-sizes", default="512,256")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dropout-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="report_assets")
    args = parser.parse_args()

    np.random.seed(args.seed)
    output_dir = ROOT_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    (x_train, y_train), (x_test, y_test) = load_mnist()
    hidden_sizes = [
        int(value.strip())
        for value in args.hidden_sizes.split(",")
        if value.strip()
    ]

    model = NeuralNetwork(
        hidden_sizes=hidden_sizes,
        use_batchnorm=True,
        use_dropout=True,
        dropout_ratio=args.dropout_ratio,
    )
    optimizer = Adam(lr=args.lr)

    start = time.perf_counter()
    loss_history = train_with_history(
        model,
        optimizer,
        x_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    training_seconds = time.perf_counter() - start

    train_accuracy, total_params = evaluate(model, x_train, y_train)
    test_accuracy, _ = evaluate(model, x_test, y_test)

    plot_path = output_dir / "loss_curve.png"
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, args.epochs + 1), loss_history, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Average Cross Entropy Loss")
    plt.title("MNIST Training Loss Curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()

    metrics = {
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "dropout_ratio": args.dropout_ratio,
        "hidden_sizes": hidden_sizes,
        "optimizer": "Adam",
        "batchnorm_momentum": 0.9,
        "weight_initialization": "He",
        "training_seconds": training_seconds,
        "train_accuracy_percent": train_accuracy,
        "test_accuracy_percent": test_accuracy,
        "total_params": int(total_params),
        "loss_history": loss_history,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "matplotlib_version": matplotlib.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "plot_path": str(plot_path.relative_to(ROOT_DIR)),
    }

    metrics_path = output_dir / "experiment_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"training_seconds={training_seconds:.2f}")
    print(f"train_accuracy_percent={train_accuracy:.2f}")
    print(f"test_accuracy_percent={test_accuracy:.2f}")
    print(f"total_params={total_params:,}")
    print(f"loss_start={loss_history[0]:.6f}")
    print(f"loss_end={loss_history[-1]:.6f}")
    print(f"plot_path={plot_path.relative_to(ROOT_DIR)}")
    print(f"metrics_path={metrics_path.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
