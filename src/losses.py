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
