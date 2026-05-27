import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from data import load_mnist  # ← dataset.mnist 대신

(x_train, t_train), (x_test, t_test) = load_mnist()

# 2차 함수 예시
def function_1(x):
    return np.sum(x**2 + x)

# 편미분 값 계산
def numeric_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]  # 원래 값 보존

        x[idx] = tmp_val + h
        fx1 = f(x)

        x[idx] = tmp_val - h
        fx2 = f(x)

        grad[idx] = (fx1 - fx2) / (2 * h)
        x[idx] = tmp_val
        it.iternext()

    return grad



def gradient_descent(f, init_x, lr=0.01, step_num=100, print_every=5):
    """경사하강으로 최적점을 찾아가는 함수.

    Returns:
        x: 최종 위치
        history: 매 시점 x 기록 (shape: (checkpoint, dim))
        losses: 매 시점 손실 기록
    """
    x = init_x.astype(float)
    history = [x.copy()]
    losses = [f(x)]

    for i in range(step_num):

        grad = numeric_gradient(f, x)
        x -= lr * grad

        if (i + 1) % print_every == 0:
            print(f"{i+1}회 훈련: grad:{grad}, x =", x)
            history.append(x.copy())
            losses.append(f(x))

    return x, np.array(history), np.array(losses)

#경사하강 경로를 그리는 재사용 함수
def plot_gradient_descent_history(history, losses=None):
    import matplotlib.pyplot as plt

    if history.ndim != 2:
        raise ValueError("history는 2D 배열이어야 합니다.")

    steps = np.arange(history.shape[0])

    plt.figure(figsize=(8, 4))
    for d in range(history.shape[1]):
        plt.plot(steps, history[:, d], label=f"x[{d}]")

    plt.xlabel("step checkpoint")
    plt.ylabel("x value")
    plt.title("Gradient Descent Path")
    plt.legend()
    plt.grid(True)

    if losses is not None:
        plt.figure(figsize=(8, 4))
        plt.plot(np.arange(len(losses)), losses)
        plt.xlabel("step checkpoint")
        plt.ylabel("loss")
        plt.title("Loss History")
        plt.grid(True)

    plt.show()


if __name__ == "__main__":
    #init_x = np.array([1.0, 2.0, -1.0])
    init_x = np.array([1.0, 2.0, -1.0])
    #init_x = np.array([-1.0])
    print("배열 x:", init_x)
    x, history, losses = gradient_descent(function_1, init_x, lr=0.8, step_num=100)
    print("result:", x)

    plot_gradient_descent_history(history, losses)
