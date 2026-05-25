# -*- coding: utf-8 -*-
"""손실 함수 모음."""

import numpy as np


def cross_entropy_loss(y_pred, y_true):
    """
    Cross Entropy Error (배치 평균).
    y_pred: (batch_size, 10) 확률
    y_true: (batch_size,) 정수 레이블 0~9

    y_pred = np.array([
        [0.1, 0.2, 0.7],
        [0.8, 0.1, 0.1],
    ])
    y_true = np.array([2, 0])
    """
    # 정답 클래스 확률의 log 값을 이용해 batch 평균 cross entropy를 계산하세요.
    # 힌트: np.clip으로 log(0)을 피하고, np.arange(batch_size)로 정답 위치를 고릅니다.
    batch_size = y_true.shape[0]

    #정답으로 예상되는 확률값 배열 추출
    correct_probs = y_pred[np.arange(batch_size), y_true]

    #확률값 배열 정규화
    delta = 1e-7
    correct_probs = np.clip(correct_probs, delta, 1) 

    #공식 갈기기
    E = -np.sum(np.log(correct_probs)) / batch_size
    
    return E
