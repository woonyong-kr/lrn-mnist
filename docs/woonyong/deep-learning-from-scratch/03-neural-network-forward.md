# 03. 신경망 순전파

신경망 순전파는 현재 W, b를 가지고 예측값을 만드는 과정임.

MNIST 숫자 분류를 기준으로 보면:

```text
입력 이미지 x
 -> Affine
 -> Activation
 -> Affine
 -> Softmax
 -> 확률 y
```

순전파는 학습이 아니라 예측임.
학습은 이 예측을 보고 loss를 구하고, gradient로 W, b를 바꾸는 것까지 포함함.

## 1. 퍼셉트론과 신경망의 차이

퍼셉트론은:

```text
y = 1 if x @ W + b > 0
```

처럼 threshold로 딱 자름.

신경망은:

```text
a = x @ W + b
z = activation(a)
```

처럼 activation function을 통과시킴.

이유:

```text
미분 가능한 함수를 써야 gradient 기반 학습이 가능함
비선형 activation을 넣어야 여러 층을 쌓는 의미가 생김
```

## 2. Affine 계층

Affine은 선형 변환 + bias임.

```text
a = x @ W + b
```

예:

```text
x = [1, 2, 3]

W = [
  [0.1, 0.2],
  [0.0, 0.1],
  [0.3, -0.2],
]

b = [0, 0]
```

계산:

```text
a[0] = 1*0.1 + 2*0.0 + 3*0.3 + 0 = 1.0
a[1] = 1*0.2 + 2*0.1 + 3*(-0.2) + 0 = -0.2
```

결과:

```text
a = [1.0, -0.2]
```

## 3. Activation 함수

Affine만 여러 번 쌓으면 사실 큰 affine 하나와 같아짐.

예:

```text
y = (x @ W1) @ W2
  = x @ (W1 @ W2)
```

중간에 비선형 함수가 없으면 층을 많이 쌓아도 표현력이 크게 늘지 않음.

그래서 activation이 필요함.

```text
a1 = x @ W1 + b1
z1 = activation(a1)
a2 = z1 @ W2 + b2
```

## 4. Sigmoid

Sigmoid는 값을 0과 1 사이로 눌러줌.

```text
sigmoid(x) = 1 / (1 + exp(-x))
```

예:

```text
sigmoid(1.0) = 0.7311
sigmoid(-0.2) = 0.4502
```

장점:

```text
출력이 0~1 사이
부드럽게 변함
미분 가능
```

단점:

```text
값이 너무 크거나 작으면 기울기가 거의 0이 됨
깊은 신경망에서 gradient가 사라지기 쉬움
```

## 5. ReLU

ReLU는 현재 가장 기본적으로 많이 쓰는 activation임.

```text
ReLU(x) = max(0, x)
```

예:

```text
ReLU(-2) = 0
ReLU(0) = 0
ReLU(3) = 3
```

장점:

```text
계산이 단순함
양수 구간에서 gradient가 1이라 학습이 잘 흐름
깊은 네트워크에서 sigmoid보다 유리한 경우가 많음
```

단점:

```text
음수 구간에서는 gradient가 0이라 뉴런이 죽을 수 있음
```

현재 본소스는 은닉층에 ReLU를 사용함.

## 6. 출력층 Activation

출력층은 문제 유형에 따라 다름.

```text
회귀:
  identity function
  숫자 그대로 출력

이진 분류:
  sigmoid
  0~1 확률 하나 출력

다중 클래스 분류:
  softmax
  클래스별 확률 여러 개 출력
```

MNIST는 0~9 중 하나를 고르는 문제라서 softmax를 씀.

## 7. Softmax

Softmax는 점수들을 확률로 바꿈.

```text
y_i = exp(a_i) / sum(exp(a_k))
```

예:

```text
a = [2, 1, 5]
```

계산하면:

```text
y = [0.047, 0.017, 0.936]
```

합은 1임.

```text
0.047 + 0.017 + 0.936 = 1.0
```

가장 큰 점수 5를 가진 클래스가 가장 높은 확률을 가짐.

## 8. Softmax에서 exp를 쓰는 이유

exp는 큰 값을 더 크게, 작은 값을 더 작게 벌려줌.

```text
exp(1) = 2.718
exp(2) = 7.389
exp(5) = 148.413
```

그래서 점수 차이가 확률 차이로 더 분명하게 나타남.

그리고 exp는 항상 양수임.

```text
확률은 음수가 되면 안 됨
```

softmax는:

```text
양수로 만들고
전체 합으로 나눠서
합이 1인 확률로 만든다
```

고 보면 됨.

## 9. Softmax overflow 방지

exp는 입력이 크면 숫자가 너무 커짐.

그래서 실제 코드는 최댓값을 빼고 계산함.

```python
shifted = x - np.max(x, axis=1, keepdims=True)
exp_x = np.exp(shifted)
y = exp_x / np.sum(exp_x, axis=1, keepdims=True)
```

왜 결과가 같냐면:

```text
exp(a_i - c) / sum(exp(a_k - c))
= exp(a_i)exp(-c) / sum(exp(a_k)exp(-c))
= exp(a_i) / sum(exp(a_k))
```

공통으로 곱해진 `exp(-c)`가 약분되기 때문임.

## 10. MNIST 추론 흐름

현재 코드의 큰 흐름은:

```text
x: 128 x 784

Affine1:
  x @ W1 + b1
  128 x 512

BatchNorm1:
  128 x 512

ReLU1:
  128 x 512

Dropout1:
  128 x 512

Affine2:
  128 x 256

BatchNorm2:
  128 x 256

ReLU2:
  128 x 256

Dropout2:
  128 x 256

Affine3:
  128 x 10

Softmax:
  128 x 10
```

최종 `128 x 10`은:

```text
128개 이미지 각각에 대해
0~9일 확률 10개
```

를 뜻함.

## 11. 예측값 고르기

softmax 결과가:

```text
[0.01, 0.02, 0.80, 0.05, ...]
```

이면 가장 큰 값의 index가 예측 클래스임.

```python
pred = np.argmax(y, axis=1)
```

MNIST에서:

```text
pred = 2
```

이면 모델이 숫자 2라고 예측했다는 뜻임.

## 12. 순전파와 학습의 차이

순전파:

```text
현재 W, b로 y를 만든다.
```

학습:

```text
y와 정답 t를 비교해서 loss를 만든다.
loss를 줄이는 gradient를 구한다.
W, b를 업데이트한다.
```

즉 순전파는 학습의 일부임.

```text
순전파만 하면 예측
순전파 + loss + 역전파 + 업데이트를 반복하면 학습
```

## 확인 질문

```text
1. affine만 여러 층 쌓으면 왜 표현력이 늘지 않는가?
2. ReLU가 sigmoid보다 깊은 신경망에서 유리한 이유는?
3. MNIST 출력층에서 softmax를 쓰는 이유는?
4. logits와 확률 y는 어떻게 다른가?
5. softmax에서 최댓값을 빼도 결과가 같은 이유는?
```

