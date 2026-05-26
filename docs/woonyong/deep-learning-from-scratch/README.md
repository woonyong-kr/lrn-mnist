# 밑바닥부터 시작하는 딥러닝 1 학습 노트

이 문서 묶음은 책의 문장을 옮긴 요약이 아니라, 책에서 배우는 흐름을 내 방식으로 다시 정리한 학습 노트임.

목표는 다음 질문에 답할 수 있게 되는 것임.

```text
왜 퍼셉트론에서 신경망으로 넘어가는가?
왜 활성화 함수가 필요한가?
왜 softmax와 cross entropy를 같이 쓰는가?
왜 loss를 미분해서 W, b를 바꾸는가?
왜 수치미분은 느리고 역전파는 빠른가?
왜 Adam, BatchNorm, Dropout 같은 기법이 필요한가?
CNN은 기존 신경망과 무엇이 다른가?
```

## 읽는 순서

1. [01-python-numpy.md](./01-python-numpy.md)  
   Python, NumPy, 행렬 shape, 브로드캐스팅, 그래프 출력

2. [02-perceptron.md](./02-perceptron.md)  
   퍼셉트론, AND/OR/NAND/XOR, 선형 분리, 다층 구조

3. [03-neural-network-forward.md](./03-neural-network-forward.md)  
   신경망 순전파, 활성화 함수, 행렬곱, softmax, MNIST 추론

4. [04-learning-loss-gradient.md](./04-learning-loss-gradient.md)  
   손실함수, cross entropy, 수치미분, gradient descent, mini-batch

5. [05-backpropagation.md](./05-backpropagation.md)  
   계산 그래프, 연쇄법칙, Affine/ReLU/Sigmoid/SoftmaxWithLoss 역전파

6. [06-training-techniques.md](./06-training-techniques.md)  
   SGD, Momentum, AdaGrad, RMSProp, Adam, 초기화, BatchNorm, Dropout, Weight Decay

7. [07-cnn.md](./07-cnn.md)  
   CNN, convolution, padding, stride, pooling, im2col, 이미지 분류 흐름

8. [08-deep-learning-practice.md](./08-deep-learning-practice.md)  
   깊은 신경망, 정확도 개선 전략, 하이퍼파라미터, 실험 기록 방법

9. [09-mnist-code-map.md](./09-mnist-code-map.md)  
   현재 repo 코드와 개념 연결, 파일별 역할, 실행 흐름

## 전체 큰 그림

딥러닝 학습은 결국 아래 사이클임.

```text
1. 현재 W, b로 예측한다.
2. 예측 y와 정답 t를 비교해서 loss L을 만든다.
3. L을 줄이려면 W, b를 어느 방향으로 움직여야 하는지 gradient를 구한다.
4. optimizer가 gradient를 보고 W, b를 업데이트한다.
5. 이 과정을 반복한다.
```

MNIST를 기준으로 쓰면:

```text
이미지 x
 -> Affine
 -> ReLU
 -> Affine
 -> Softmax
 -> Cross Entropy Loss
 -> Backpropagation
 -> Adam update
```

## 가장 중요한 구분

처음에는 아래 4개가 자주 섞임.

```text
model:
  y = f(x; W, b)
  현재 파라미터로 예측을 만드는 함수

loss function:
  L = loss(y, t)
  예측이 얼마나 틀렸는지 숫자 하나로 만드는 함수

gradient:
  dL/dW, dL/db
  W, b를 바꾸면 loss가 얼마나 바뀌는지

optimizer:
  W <- W - update
  gradient를 보고 실제 파라미터를 바꾸는 규칙
```

Adam은 loss function이 아님.

```text
cross entropy가 loss를 만들고
backprop이 gradient를 만들고
Adam이 그 gradient로 W, b를 바꿈
```

## 수학을 볼 때의 기준

수식이 나오면 무조건 이 질문으로 바꾸면 됨.

```text
이 값은 순전파에서 무엇인가?
이 값은 loss에 어떤 영향을 주는가?
이 값이 조금 변하면 다음 값이 얼마나 변하는가?
이 gradient는 어떤 파라미터를 고치기 위한 신호인가?
```

예:

```text
dW2 = z1.T @ da2
```

라고 나오면:

```text
z1: 앞 뉴런이 얼마나 켜졌는가
da2: 뒤 출력이 얼마나 틀렸는가
dW2: 그 둘을 연결하는 W2를 얼마나 고칠 것인가
```

로 읽으면 됨.

## 현재 repo와 같이 보는 방법

먼저 문서를 읽고, 아래 파일을 같이 보면 좋음.

```text
src/learning/calculate.py
  숫자로 직접 역전파를 따라가는 파일

src/learning/mini_batch.py
  작은 독립 학습 예제

src/network.py
  실제 과제용 NeuralNetwork 구성

src/layers.py
  Affine, BatchNorm, Dropout

src/activations.py
  ReLU, Softmax

src/losses.py
  cross entropy loss, gradient

src/optimizers.py
  SGD, Adam

mnist_lab.ipynb
  테스트, 학습, 평가를 한 번에 실행하는 노트북
```

## 추천 학습 방식

각 문서를 읽을 때는 이렇게 하면 좋음.

```text
1. 개념을 읽는다.
2. 작은 숫자 예제를 손으로 따라간다.
3. 코드에서 같은 이름을 찾는다.
4. shape를 적는다.
5. 이 gradient가 어느 파라미터를 고치는지 말로 설명해본다.
```

특히 shape는 계속 적어야 함.

```text
x:       batch_size x input_dim
W:       input_dim x output_dim
b:       output_dim
out:     batch_size x output_dim
dout:    batch_size x output_dim
dW:      input_dim x output_dim
db:      output_dim
dx:      batch_size x input_dim
```

