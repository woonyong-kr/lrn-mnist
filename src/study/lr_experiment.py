import os

import matplotlib.pyplot as plt
import numpy as np


def function_1(x):
    """f(x) = sum(x^2 + x). The minimum is at x = -0.5 for every dimension."""
    return np.sum(x**2 + x)


def gradient_1(x):
    """Analytic gradient of function_1."""
    return 2 * x + 1


def gradient_descent_trace(f, grad_f, init_x, lr, step_num=100, tolerance=1e-3):
    x = init_x.astype(float).copy()
    optimum = np.full_like(x, -0.5)
    history = [x.copy()]
    losses = [f(x)]
    converged_at = None

    for step in range(1, step_num + 1):
        grad = grad_f(x)
        x -= lr * grad

        history.append(x.copy())
        losses.append(f(x))

        if converged_at is None and np.all(np.abs(x - optimum) <= tolerance):
            converged_at = step

    return {
        "lr": lr,
        "history": np.array(history),
        "losses": np.array(losses),
        "converged_at": converged_at,
        "final_x": x.copy(),
    }


def run_lr_experiment(
    init_x=None,
    learning_rates=None,
    step_num=100,
    tolerance=1e-3,
):
    if init_x is None:
        init_x = np.array([1.0, 2.0, -1.0])

    if learning_rates is None:
        learning_rates = [0.01, 0.05, 0.1, 0.3, 0.5, 0.8, 0.9, 1.0, 1.1]

    return [
        gradient_descent_trace(
            function_1,
            gradient_1,
            init_x,
            lr,
            step_num=step_num,
            tolerance=tolerance,
        )
        for lr in learning_rates
    ]


def print_summary(results):
    print("lr별 수렴 결과")
    print("-" * 72)
    print(f"{'lr':>8} | {'수렴 step':>10} | {'final x':>34} | {'final loss':>12}")
    print("-" * 72)

    for result in results:
        step = result["converged_at"]
        step_text = str(step) if step is not None else "미수렴"
        final_x = np.array2string(result["final_x"], precision=4, suppress_small=True)
        final_loss = result["losses"][-1]
        print(f"{result['lr']:>8.2f} | {step_text:>10} | {final_x:>34} | {final_loss:>12.6f}")


def plot_lr_experiment(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(9, 5))
    for result in results:
        history = result["history"]
        plt.plot(history[:, 0], label=f"lr={result['lr']}")
    plt.axhline(-0.5, color="black", linestyle="--", linewidth=1, label="optimum x=-0.5")
    plt.xlabel("iteration")
    plt.ylabel("x[0]")
    plt.title("Learning Rate vs x Convergence")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lr_x_path.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5))
    for result in results:
        plt.plot(result["losses"], label=f"lr={result['lr']}")
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.title("Learning Rate vs Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lr_loss_path.png"), dpi=160)
    plt.close()

    labels = [str(result["lr"]) for result in results]
    steps = [
        result["converged_at"] if result["converged_at"] is not None else np.nan
        for result in results
    ]

    plt.figure(figsize=(9, 5))
    plt.bar(labels, steps)
    plt.xlabel("learning rate")
    plt.ylabel("steps to reach tolerance")
    plt.title("Steps Needed to Reach x=-0.5")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lr_convergence_steps.png"), dpi=160)
    plt.close()


if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    output_dir = os.path.join(current_dir, "figures")

    results = run_lr_experiment(step_num=100, tolerance=1e-3)
    print_summary(results)
    plot_lr_experiment(results, output_dir)

    print()
    print("그래프 저장 위치:", output_dir)
