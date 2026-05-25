# -*- coding: utf-8 -*-
"""파라미터 업데이트 규칙을 모아 둔 optimizer 모듈."""

import numpy as np

class SGD:
    """
    확률적 경사하강법(SGD).

    가장 단순한 optimizer로, 각 파라미터를 gradient 반대 방향으로 lr만큼 이동합니다.
    """

    def __init__(self, lr=0.01):
        """Args: lr: 한 번 업데이트할 때 gradient에 곱할 학습률."""
        self.lr = lr

    def update(self, params, grads):
        """params dict의 모든 파라미터를 제자리(in-place)에서 갱신합니다."""
        for key in params.keys():
            params[key] -= self.lr * grads[key]
        


class Adam:
    """
    Adam Optimizer.

    gradient의 이동평균(m)과 제곱 이동평균(v)을 함께 사용해 파라미터별 학습률을 조절합니다.
    MNIST 과제에서는 SGD보다 빠르게 손실이 내려가는지 비교해 볼 수 있습니다.
    """

    def __init__(self, lr=0.001, momentum = 0.9):
        """Args: lr: Adam 업데이트의 기본 학습률."""
        self.lr = lr
        self.m, self.v = {}, {}
        self.t = 0 #update가 몇 번째인지 나타내는 카운터

        self.momentum = momentum

    def update(self, params, grads):
        """Adam 공식에 따라 params dict의 모든 파라미터를 갱신합니다."""
        # m, v 이동평균과 bias correction을 사용해 params를 업데이트하세요.
        '''
        m: 방향
        v: 크기 보정
        lr: 마지막 업데이트에서 곱함
        
        '''
        #업데이트 횟수 카운트
        self.t += 1 

        #하이퍼파라미터 2개
        beta1 = 0.9
        beta2 = 0.999

        eps = 1e-7

        for key in params.keys():
            '''
            v = av - lr*grads
            W = W + v
            '''

            #각 key에 대해 m,v가 없으면 0으로 초기화
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
            if key not in self.v:
                self.v[key] = np.zeros_like(params[key])

            #m에 미분 이동 평균 저장, m: 어느 방향으로 갈지
            self.m[key] = beta1 *self.m[key] + (1-beta1) * grads[key]
            #v에 미분 제곱 이동 평균 저장, v: 얼마나 조심해서 갈지
            self.v[key] =  beta2 * self.v[key] + (1-beta2) * grads[key] ** 2

            '''bias correction하기
            처음에는 m과 v가 0에서 시작한다.
            m, v는 실제 평균보다 작게 잡히는 경향이 있다. 이를 보정해야 한다.
            '''
            m_hat = self.m[key] / (1 - beta1*self.t)
            v_hat = self.v[key] / (1 - beta2*self.t)

            #가중치 업데이트
            params[key] =  params[key] - self.lr * m_hat / (np.sqrt(v_hat) + eps)
        