# -*- coding: utf-8 -*-
"""Run several MNIST hyperparameter experiments for the report."""

import csv
import json
import platform
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from data import load_mnist
from network import NeuralNetwork
from optimizers import Adam
from training import evaluate


EXPERIMENTS = [
    {
        "name": "baseline",
        "hidden_sizes": [512, 256],
        "dropout_ratio": 0.5,
        "epochs": 20,
        "batch_size": 128,
        "learning_rate": 0.001,
        "purpose": "기준 설정",
    },
    {
        "name": "dropout_0_4",
        "hidden_sizes": [512, 256],
        "dropout_ratio": 0.4,
        "epochs": 20,
        "batch_size": 128,
        "learning_rate": 0.001,
        "purpose": "Dropout 비율 감소",
    },
    {
        "name": "epochs_30",
        "hidden_sizes": [512, 256],
        "dropout_ratio": 0.5,
        "epochs": 30,
        "batch_size": 128,
        "learning_rate": 0.001,
        "purpose": "반복 횟수 증가",
    },
    {
        "name": "batch_64",
        "hidden_sizes": [512, 256],
        "dropout_ratio": 0.5,
        "epochs": 20,
        "batch_size": 64,
        "learning_rate": 0.001,
        "purpose": "미니배치 크기 감소",
    },
    {
        "name": "wide_1024_512",
        "hidden_sizes": [1024, 512],
        "dropout_ratio": 0.5,
        "epochs": 20,
        "batch_size": 128,
        "learning_rate": 0.001,
        "purpose": "은닉층 너비 증가",
    },
    {
        "name": "deep_512_512_256",
        "hidden_sizes": [512, 512, 256],
        "dropout_ratio": 0.5,
        "epochs": 20,
        "batch_size": 128,
        "learning_rate": 0.001,
        "purpose": "은닉층 깊이 증가",
    },
    {
        "name": "tuned_dropout_0_4_epochs_30",
        "hidden_sizes": [512, 256],
        "dropout_ratio": 0.4,
        "epochs": 30,
        "batch_size": 128,
        "learning_rate": 0.001,
        "purpose": "Dropout 감소와 반복 증가 조합",
    },
]


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

        average_loss = float(epoch_loss / batch_count)
        loss_history.append(average_loss)
        elapsed = time.perf_counter() - epoch_start
        print(
            f"  epoch {epoch:02d}/{epochs} loss={average_loss:.6f} time={elapsed:.2f}s",
            flush=True,
        )

    return loss_history


def run_experiment(config, x_train, y_train, x_test, y_test, seed):
    """Run one config and return metrics."""
    np.random.seed(seed)

    model = NeuralNetwork(
        hidden_sizes=config["hidden_sizes"],
        use_batchnorm=True,
        use_dropout=True,
        dropout_ratio=config["dropout_ratio"],
    )
    optimizer = Adam(lr=config["learning_rate"])

    print(
        f"[{config['name']}] hidden={config['hidden_sizes']} "
        f"dropout={config['dropout_ratio']} epochs={config['epochs']} "
        f"batch={config['batch_size']} lr={config['learning_rate']}",
        flush=True,
    )

    start = time.perf_counter()
    loss_history = train_with_history(
        model,
        optimizer,
        x_train,
        y_train,
        epochs=config["epochs"],
        batch_size=config["batch_size"],
    )
    training_seconds = time.perf_counter() - start

    train_accuracy, total_params = evaluate(model, x_train, y_train)
    test_accuracy, _ = evaluate(model, x_test, y_test)

    updates_per_epoch = int(np.ceil(x_train.shape[0] / config["batch_size"]))

    result = {
        **config,
        "seed": seed,
        "train_accuracy_percent": float(train_accuracy),
        "test_accuracy_percent": float(test_accuracy),
        "total_params": int(total_params),
        "training_seconds": float(training_seconds),
        "updates_per_epoch": updates_per_epoch,
        "total_updates": updates_per_epoch * config["epochs"],
        "loss_start": float(loss_history[0]),
        "loss_end": float(loss_history[-1]),
        "loss_history": loss_history,
    }

    print(
        f"[{config['name']}] test={test_accuracy:.2f}% "
        f"train={train_accuracy:.2f}% loss={loss_history[-1]:.6f} "
        f"time={training_seconds:.2f}s params={total_params:,}",
        flush=True,
    )
    return result


def save_csv(results, path):
    fieldnames = [
        "name",
        "purpose",
        "hidden_sizes",
        "dropout_ratio",
        "epochs",
        "batch_size",
        "updates_per_epoch",
        "total_updates",
        "learning_rate",
        "train_accuracy_percent",
        "test_accuracy_percent",
        "total_params",
        "training_seconds",
        "loss_start",
        "loss_end",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = dict(result)
            row["hidden_sizes"] = "->".join(str(size) for size in row["hidden_sizes"])
            writer.writerow({key: row[key] for key in fieldnames})


def save_plots(results, output_dir):
    sorted_results = sorted(
        results,
        key=lambda item: item["test_accuracy_percent"],
        reverse=True,
    )
    names = [item["name"] for item in sorted_results]
    accuracies = [item["test_accuracy_percent"] for item in sorted_results]

    plt.figure(figsize=(11, 5))
    bars = plt.bar(names, accuracies)
    plt.ylabel("Test Accuracy (%)")
    plt.title("MNIST Hyperparameter Sweep - Test Accuracy")
    plt.ylim(min(accuracies) - 0.4, max(accuracies) + 0.2)
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.25)
    for bar, accuracy in zip(bars, accuracies):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{accuracy:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.tight_layout()
    plt.savefig(output_dir / "hparam_accuracy.png", dpi=160)
    plt.close()

    plt.figure(figsize=(11, 6))
    for result in results:
        xs = range(1, result["epochs"] + 1)
        plt.plot(xs, result["loss_history"], label=result["name"])
    plt.xlabel("Epoch")
    plt.ylabel("Average Cross Entropy Loss")
    plt.title("MNIST Hyperparameter Sweep - Loss Curves")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "hparam_loss_curves.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    params = [item["total_params"] / 1_000_000 for item in results]
    times = [item["training_seconds"] for item in results]
    tests = [item["test_accuracy_percent"] for item in results]
    plt.scatter(params, tests, s=[max(40, time_value) for time_value in times])
    for item, x_value, y_value in zip(results, params, tests):
        plt.annotate(item["name"], (x_value, y_value), fontsize=8)
    plt.xlabel("Parameters (millions)")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Model Size vs Test Accuracy")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "hparam_params_vs_accuracy.png", dpi=160)
    plt.close()


def main():
    seed = 42
    output_dir = ROOT_DIR / "report_assets" / "hparam_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)

    (x_train, y_train), (x_test, y_test) = load_mnist()
    results = [
        run_experiment(config, x_train, y_train, x_test, y_test, seed)
        for config in EXPERIMENTS
    ]

    payload = {
        "environment": {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "matplotlib_version": matplotlib.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "results": results,
    }
    (output_dir / "hparam_sweep_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    save_csv(results, output_dir / "hparam_sweep_results.csv")
    save_plots(results, output_dir)

    best = max(results, key=lambda item: item["test_accuracy_percent"])
    print(
        f"best={best['name']} test={best['test_accuracy_percent']:.2f}% "
        f"train={best['train_accuracy_percent']:.2f}% "
        f"loss={best['loss_end']:.6f}",
        flush=True,
    )
    print(f"output_dir={output_dir.relative_to(ROOT_DIR)}", flush=True)


if __name__ == "__main__":
    main()
