# -*- coding: utf-8 -*-
"""Generate diagnostic experiments that make training correlations visible."""

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
from optimizers import Adam, SGD


def accuracy(model, x, y):
    y_pred = model.predict(x)
    return float(np.mean(np.argmax(y_pred, axis=1) == y) * 100)


def evaluate_snapshot(model, x_train, y_train, x_test, y_test, epoch):
    return {
        "epoch": epoch,
        "train_loss": float(model.loss(x_train, y_train, train=False)),
        "test_loss": float(model.loss(x_test, y_test, train=False)),
        "train_accuracy_percent": accuracy(model, x_train, y_train),
        "test_accuracy_percent": accuracy(model, x_test, y_test),
    }


def make_optimizer(name, lr):
    if name == "Adam":
        return Adam(lr=lr)
    if name == "SGD":
        return SGD(lr=lr)
    raise ValueError(f"unsupported optimizer: {name}")


def train_case(config, x_train, y_train, x_test, y_test, seed):
    np.random.seed(seed)
    model = NeuralNetwork(
        hidden_sizes=config["hidden_sizes"],
        use_batchnorm=config["use_batchnorm"],
        use_dropout=config["use_dropout"],
        dropout_ratio=config["dropout_ratio"],
    )
    optimizer = make_optimizer(config["optimizer"], config["learning_rate"])
    train_size = x_train.shape[0]
    snapshots = []

    print(
        f"[{config['name']}] train={train_size} hidden={config['hidden_sizes']} "
        f"optimizer={config['optimizer']} lr={config['learning_rate']} "
        f"epochs={config['epochs']} batch={config['batch_size']} "
        f"bn={config['use_batchnorm']} dropout={config['dropout_ratio'] if config['use_dropout'] else 0}",
        flush=True,
    )
    start = time.perf_counter()

    for epoch in range(1, config["epochs"] + 1):
        indices = np.random.permutation(train_size)
        epoch_loss = 0.0
        batch_count = 0

        for start_idx in range(0, train_size, config["batch_size"]):
            batch_indices = indices[start_idx : start_idx + config["batch_size"]]
            loss = model.gradient(x_train[batch_indices], y_train[batch_indices])
            optimizer.update(model.params, model.grads)
            epoch_loss += loss
            batch_count += 1

        if epoch == 1 or epoch % config["eval_interval"] == 0 or epoch == config["epochs"]:
            snapshot = evaluate_snapshot(model, x_train, y_train, x_test, y_test, epoch)
            snapshot["epoch_train_loss"] = float(epoch_loss / batch_count)
            snapshots.append(snapshot)
            print(
                f"  epoch {epoch:03d}/{config['epochs']} "
                f"train_acc={snapshot['train_accuracy_percent']:.2f}% "
                f"test_acc={snapshot['test_accuracy_percent']:.2f}% "
                f"gap={snapshot['train_accuracy_percent'] - snapshot['test_accuracy_percent']:.2f}%p "
                f"test_loss={snapshot['test_loss']:.4f}",
                flush=True,
            )

    training_seconds = time.perf_counter() - start
    final = snapshots[-1]
    result = {
        **config,
        "seed": seed,
        "train_size": int(train_size),
        "test_size": int(x_test.shape[0]),
        "training_seconds": float(training_seconds),
        "final_train_accuracy_percent": final["train_accuracy_percent"],
        "final_test_accuracy_percent": final["test_accuracy_percent"],
        "final_gap_percent_point": final["train_accuracy_percent"] - final["test_accuracy_percent"],
        "final_train_loss": final["train_loss"],
        "final_test_loss": final["test_loss"],
        "snapshots": snapshots,
    }
    print(
        f"[{config['name']}] final test={result['final_test_accuracy_percent']:.2f}% "
        f"gap={result['final_gap_percent_point']:.2f}%p time={training_seconds:.2f}s",
        flush=True,
    )
    return result


def count_loss_increases(loss_history):
    return sum(
        1
        for previous, current in zip(loss_history, loss_history[1:])
        if current > previous
    )


def save_existing_correlation_plots(output_dir):
    hparam_path = ROOT_DIR / "report_assets" / "hparam_sweep" / "hparam_sweep_results.json"
    payload = json.loads(hparam_path.read_text(encoding="utf-8"))
    results = payload["results"]

    names = [item["name"] for item in results]
    test_acc = np.array([item["test_accuracy_percent"] for item in results])
    updates = np.array([item["total_updates"] for item in results])
    loss_end = np.array([item["loss_end"] for item in results])
    params = np.array([item["total_params"] for item in results])
    gaps = np.array(
        [
            item["train_accuracy_percent"] - item["test_accuracy_percent"]
            for item in results
        ]
    )
    increase_counts = np.array(
        [count_loss_increases(item["loss_history"]) for item in results]
    )

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    plots = [
        (axes[0, 0], updates, test_acc, "Total Updates", "Test Accuracy (%)"),
        (axes[0, 1], loss_end, test_acc, "Final Loss", "Test Accuracy (%)"),
        (axes[1, 0], gaps, test_acc, "Train-Test Accuracy Gap (%p)", "Test Accuracy (%)"),
        (axes[1, 1], params / 1_000_000, test_acc, "Parameters (millions)", "Test Accuracy (%)"),
    ]
    for ax, xs, ys, xlabel, ylabel in plots:
        ax.scatter(xs, ys)
        for name, x_value, y_value in zip(names, xs, ys):
            ax.annotate(name, (x_value, y_value), fontsize=8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    fig.suptitle("Correlation Summary from Full-Data Hyperparameter Sweep")
    fig.tight_layout()
    fig.savefig(output_dir / "correlation_summary.png", dpi=160)
    plt.close(fig)

    plt.figure(figsize=(9, 5))
    plt.bar(names, increase_counts)
    plt.ylabel("Number of Epoch-to-Epoch Loss Increases")
    plt.title("Is Loss Uniformly Decreasing?")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "loss_monotonicity.png", dpi=160)
    plt.close()

    rows = []
    for item, increase_count in zip(results, increase_counts):
        rows.append(
            {
                "name": item["name"],
                "test_accuracy_percent": item["test_accuracy_percent"],
                "loss_end": item["loss_end"],
                "total_updates": item["total_updates"],
                "total_params": item["total_params"],
                "train_test_gap_percent_point": item["train_accuracy_percent"]
                - item["test_accuracy_percent"],
                "loss_increase_count": int(increase_count),
            }
        )
    return rows


def save_diagnostic_csv(results, path):
    fieldnames = [
        "name",
        "group",
        "optimizer",
        "learning_rate",
        "hidden_sizes",
        "train_size",
        "test_size",
        "epochs",
        "batch_size",
        "use_batchnorm",
        "use_dropout",
        "dropout_ratio",
        "final_train_accuracy_percent",
        "final_test_accuracy_percent",
        "final_gap_percent_point",
        "final_train_loss",
        "final_test_loss",
        "training_seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = dict(result)
            row["hidden_sizes"] = "->".join(str(size) for size in row["hidden_sizes"])
            writer.writerow({key: row[key] for key in fieldnames})


def plot_overfit(results, output_dir):
    overfit_results = [item for item in results if item["group"] == "overfit"]

    plt.figure(figsize=(11, 6))
    for result in overfit_results:
        epochs = [snapshot["epoch"] for snapshot in result["snapshots"]]
        train_acc = [snapshot["train_accuracy_percent"] for snapshot in result["snapshots"]]
        test_acc = [snapshot["test_accuracy_percent"] for snapshot in result["snapshots"]]
        plt.plot(epochs, train_acc, label=f"{result['name']} train")
        plt.plot(epochs, test_acc, "--", label=f"{result['name']} test")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Intentional Overfitting: Train vs Test Accuracy")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "overfit_accuracy_gap.png", dpi=160)
    plt.close()

    plt.figure(figsize=(11, 6))
    for result in overfit_results:
        epochs = [snapshot["epoch"] for snapshot in result["snapshots"]]
        gaps = [
            snapshot["train_accuracy_percent"] - snapshot["test_accuracy_percent"]
            for snapshot in result["snapshots"]
        ]
        plt.plot(epochs, gaps, label=result["name"])
    plt.xlabel("Epoch")
    plt.ylabel("Train-Test Accuracy Gap (%p)")
    plt.title("Generalization Gap Grows with Overfitting")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "overfit_generalization_gap.png", dpi=160)
    plt.close()

    plt.figure(figsize=(11, 6))
    for result in overfit_results:
        epochs = [snapshot["epoch"] for snapshot in result["snapshots"]]
        train_loss = [snapshot["train_loss"] for snapshot in result["snapshots"]]
        test_loss = [snapshot["test_loss"] for snapshot in result["snapshots"]]
        plt.plot(epochs, train_loss, label=f"{result['name']} train")
        plt.plot(epochs, test_loss, "--", label=f"{result['name']} test")
    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Intentional Overfitting: Train vs Test Loss")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "overfit_loss_gap.png", dpi=160)
    plt.close()


def plot_lr_stability(results, output_dir):
    lr_results = [item for item in results if item["group"] == "lr_stability"]

    plt.figure(figsize=(11, 6))
    for result in lr_results:
        epochs = [snapshot["epoch"] for snapshot in result["snapshots"]]
        test_loss = [snapshot["test_loss"] for snapshot in result["snapshots"]]
        plt.plot(epochs, test_loss, label=result["name"])
    plt.xlabel("Epoch")
    plt.ylabel("Test Cross Entropy Loss")
    plt.title("Learning Rate and Optimizer Stability")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "learning_rate_stability.png", dpi=160)
    plt.close()


def main():
    seed = 42
    output_dir = ROOT_DIR / "report_assets" / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)

    (x_train_all, y_train_all), (x_test_all, y_test_all) = load_mnist()
    x_test = x_test_all[:5000]
    y_test = y_test_all[:5000]

    configs = [
        {
            "name": "tiny_wide_no_reg",
            "group": "overfit",
            "hidden_sizes": [1024, 512],
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "epochs": 120,
            "batch_size": 64,
            "train_limit": 500,
            "use_batchnorm": False,
            "use_dropout": False,
            "dropout_ratio": 0.0,
            "eval_interval": 2,
        },
        {
            "name": "tiny_wide_regularized",
            "group": "overfit",
            "hidden_sizes": [1024, 512],
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "epochs": 120,
            "batch_size": 64,
            "train_limit": 500,
            "use_batchnorm": True,
            "use_dropout": True,
            "dropout_ratio": 0.5,
            "eval_interval": 2,
        },
        {
            "name": "adam_lr_0_001",
            "group": "lr_stability",
            "hidden_sizes": [512, 256],
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "epochs": 30,
            "batch_size": 128,
            "train_limit": 5000,
            "use_batchnorm": True,
            "use_dropout": True,
            "dropout_ratio": 0.5,
            "eval_interval": 1,
        },
        {
            "name": "sgd_lr_0_1",
            "group": "lr_stability",
            "hidden_sizes": [512, 256],
            "optimizer": "SGD",
            "learning_rate": 0.1,
            "epochs": 30,
            "batch_size": 128,
            "train_limit": 5000,
            "use_batchnorm": True,
            "use_dropout": True,
            "dropout_ratio": 0.5,
            "eval_interval": 1,
        },
        {
            "name": "sgd_lr_1_0",
            "group": "lr_stability",
            "hidden_sizes": [512, 256],
            "optimizer": "SGD",
            "learning_rate": 1.0,
            "epochs": 30,
            "batch_size": 128,
            "train_limit": 5000,
            "use_batchnorm": True,
            "use_dropout": True,
            "dropout_ratio": 0.5,
            "eval_interval": 1,
        },
    ]

    results = []
    for config in configs:
        train_limit = config["train_limit"]
        result = train_case(
            config,
            x_train_all[:train_limit],
            y_train_all[:train_limit],
            x_test,
            y_test,
            seed,
        )
        results.append(result)

    correlation_rows = save_existing_correlation_plots(output_dir)
    save_diagnostic_csv(results, output_dir / "diagnostic_results.csv")
    plot_overfit(results, output_dir)
    plot_lr_stability(results, output_dir)

    payload = {
        "environment": {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "matplotlib_version": matplotlib.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "notes": {
            "seed": seed,
            "test_eval_size": int(x_test.shape[0]),
            "purpose": "diagnostic experiments for visible correlations, not final score selection",
        },
        "correlation_rows": correlation_rows,
        "diagnostic_results": results,
    }
    (output_dir / "diagnostic_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"output_dir={output_dir.relative_to(ROOT_DIR)}", flush=True)


if __name__ == "__main__":
    main()
