# -*- coding: utf-8 -*-
"""손실 함수 모음."""

import numpy as np


def cross_entropy_loss(y_pred, y_true):
    """
    Cross Entropy Error (배치 평균).
    y_pred: (batch_size, 10) 확률
    y_true: (batch_size,) 정수 레이블 0~9
    """
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(1, -1)
        y_true = y_true.reshape(1, -1)

    if y_true.size == y_pred.size:
        y_true = np.argmax(y_true, axis=1)

    batch_size = y_pred.shape[0]
    clipped = np.clip(y_pred, 1e-7, 1.0)
    return -np.sum(np.log(clipped[np.arange(batch_size), y_true])) / batch_size


def cross_entropy_gradient(y_pred, y_true):
    """
    Softmax + Cross Entropy를 함께 미분한 출력층 gradient.

    y_pred가 softmax 결과일 때 dL/dlogits는 y_pred - 정답(one-hot)이 됩니다.
    배치 평균 loss를 쓰고 있으므로 batch_size로 나눕니다.
    """
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(1, -1)

    if y_true.ndim != 1:
        y_true = np.argmax(y_true, axis=1)

    batch_size = y_pred.shape[0]
    dout = y_pred.copy()
    dout[np.arange(batch_size), y_true] -= 1
    return dout / batch_size
