import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from data import load_mnist  # ← dataset.mnist 대신

#필요한 클래스/함수 import
from src.activations import Softmax
from src.losses import cross_entropy_loss

class TwoLayerNet:

    """
    input_size: 입력층 뉴런 개수
    hidden_size: 은닉층 뉴런 개수
    output_size: 출력층 뉴런 개수
    """
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):

        #가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * \
                            np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * \
                            np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(hidden_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = Softmax(a1)
        a2 = np.dot(z1, W2)+ b2
        y = Softmax(a2)

        return y
    
    """
    loss(입력 데이터, 정답레이블): 오차

    x:입력 데이터, t:정답 레이블
    y: 추측 데이터
    반환값: 오차(손실 함수의 값)
    """
    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_loss(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y==t) / float(x.shape[0])
        return accuracy