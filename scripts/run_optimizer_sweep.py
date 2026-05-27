# -*- coding: utf-8 -*-
"""Compare Adam and SGD on the same MNIST hyperparameter cases."""

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
from training import evaluate


CORE_CASES = [
    {
        "case": "baseline",
        "hidden_sizes": [512, 256],
        "dropout_ratio": 0.5,
        "epochs": 20,
        "batch_size": 128,
        "purpose": "기준 설정",
    },
    {
        "case": "dropout_0_4",
        "hidden_sizes": [512, 256],
        "dropout_ratio": 0.4,
        "epochs": 20,
        "batch_size": 128,
        "purpose": "Dropout 비율 감소",
    },
    {
        "case": "epochs_30",
        "hidden_sizes": [512, 256],
        "dropout_ratio": 0.5,
        "epochs": 30,
        "batch_size": 128,
        "purpose": "반복 횟수 증가",
    },
    {
        "case": "batch_64",
        "hidden_sizes": [512, 256],
        "dropout_ratio": 0.5,
        "epochs": 20,
        "batch_size": 64,
        "purpose": "미니배치 크기 감소",
    },
    {
        "case": "wide_1024_512",
        "hidden_sizes": [1024, 512],
        "dropout_ratio": 0.5,
        "epochs": 20,
        "batch_size": 128,
        "purpose": "은닉층 너비 증가",
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


def make_optimizer(name, lr):
    if name == "Adam":
        return Adam(lr=lr)
    if name == "SGD":
        return SGD(lr=lr)
    raise ValueError(f"unsupported optimizer: {name}")


def run_case(case_config, optimizer_name, learning_rate, x_train, y_train, x_test, y_test, seed):
    """Run one optimizer/case pair and return metrics."""
    np.random.seed(seed)
    model = NeuralNetwork(
        hidden_sizes=case_config["hidden_sizes"],
        use_batchnorm=True,
        use_dropout=True,
        dropout_ratio=case_config["dropout_ratio"],
    )
    optimizer = make_optimizer(optimizer_name, learning_rate)

    print(
        f"[{optimizer_name}:{case_config['case']}] hidden={case_config['hidden_sizes']} "
        f"dropout={case_config['dropout_ratio']} epochs={case_config['epochs']} "
        f"batch={case_config['batch_size']} lr={learning_rate}",
        flush=True,
    )

    start = time.perf_counter()
    loss_history = train_with_history(
        model,
        optimizer,
        x_train,
        y_train,
        epochs=case_config["epochs"],
        batch_size=case_config["batch_size"],
    )
    training_seconds = time.perf_counter() - start

    train_accuracy, total_params = evaluate(model, x_train, y_train)
    test_accuracy, _ = evaluate(model, x_test, y_test)
    updates_per_epoch = int(np.ceil(x_train.shape[0] / case_config["batch_size"]))

    result = {
        **case_config,
        "name": f"{optimizer_name.lower()}_{case_config['case']}",
        "optimizer": optimizer_name,
        "learning_rate": learning_rate,
        "seed": seed,
        "train_accuracy_percent": float(train_accuracy),
        "test_accuracy_percent": float(test_accuracy),
        "total_params": int(total_params),
        "training_seconds": float(training_seconds),
        "updates_per_epoch": updates_per_epoch,
        "total_updates": updates_per_epoch * case_config["epochs"],
        "loss_start": float(loss_history[0]),
        "loss_end": float(loss_history[-1]),
        "loss_history": loss_history,
    }
    print(
        f"[{optimizer_name}:{case_config['case']}] test={test_accuracy:.2f}% "
        f"train={train_accuracy:.2f}% loss={loss_history[-1]:.6f} "
        f"time={training_seconds:.2f}s",
        flush=True,
    )
    return result


def load_adam_results(path):
    """Load matching Adam results from the prior hyperparameter sweep."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_case = {
        result["name"]: {
            **result,
            "case": result["name"],
            "optimizer": "Adam",
        }
        for result in payload["results"]
    }
    return [by_case[case["case"]] for case in CORE_CASES]


def save_csv(results, path):
    fieldnames = [
        "optimizer",
        "case",
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
    cases = [case["case"] for case in CORE_CASES]
    adam = {result["case"]: result for result in results if result["optimizer"] == "Adam"}
    sgd = {result["case"]: result for result in results if result["optimizer"] == "SGD"}

    x = np.arange(len(cases))
    width = 0.36
    adam_acc = [adam[case]["test_accuracy_percent"] for case in cases]
    sgd_acc = [sgd[case]["test_accuracy_percent"] for case in cases]

    plt.figure(figsize=(11, 5))
    plt.bar(x - width / 2, adam_acc, width, label="Adam")
    plt.bar(x + width / 2, sgd_acc, width, label="SGD")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Adam vs SGD - Test Accuracy")
    plt.xticks(x, cases, rotation=25, ha="right")
    plt.ylim(min(min(adam_acc), min(sgd_acc)) - 0.4, max(max(adam_acc), max(sgd_acc)) + 0.2)
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "optimizer_accuracy.png", dpi=160)
    plt.close()

    plt.figure(figsize=(11, 6))
    for result in results:
        if result["case"] in {"baseline", "batch_64", "wide_1024_512"}:
            xs = range(1, result["epochs"] + 1)
            label = f"{result['optimizer']} {result['case']}"
            plt.plot(xs, result["loss_history"], label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Average Cross Entropy Loss")
    plt.title("Adam vs SGD - Representative Loss Curves")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "optimizer_loss_curves.png", dpi=160)
    plt.close()


def main():
    seed = 42
    output_dir = ROOT_DIR / "report_assets" / "optimizer_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)

    adam_results_path = ROOT_DIR / "report_assets" / "hparam_sweep" / "hparam_sweep_results.json"
    adam_results = load_adam_results(adam_results_path)

    (x_train, y_train), (x_test, y_test) = load_mnist()
    sgd_results = [
        run_case(
            case,
            optimizer_name="SGD",
            learning_rate=0.1,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            seed=seed,
        )
        for case in CORE_CASES
    ]
    results = adam_results + sgd_results

    payload = {
        "environment": {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "matplotlib_version": matplotlib.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "notes": {
            "adam_learning_rate": 0.001,
            "sgd_learning_rate": 0.1,
            "seed": seed,
        },
        "results": results,
    }
    (output_dir / "optimizer_sweep_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    save_csv(results, output_dir / "optimizer_sweep_results.csv")
    save_plots(results, output_dir)

    best = max(results, key=lambda item: item["test_accuracy_percent"])
    print(
        f"best={best['optimizer']}:{best['case']} "
        f"test={best['test_accuracy_percent']:.2f}% "
        f"train={best['train_accuracy_percent']:.2f}% "
        f"loss={best['loss_end']:.6f}",
        flush=True,
    )
    print(f"output_dir={output_dir.relative_to(ROOT_DIR)}", flush=True)


if __name__ == "__main__":
    main()
