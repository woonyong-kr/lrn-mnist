# Affine -> BatchNorm -> ReLU -> Dropout 한 층 흐름 이해

이 문서는 신경망의 은닉층 한 블록이 입력 `x`를 어떻게 다음 값으로 바꾸는지 단계적으로 정리한다.

현재 프로젝트의 은닉층 구조는 대략 다음과 같다.

```text
x
-> Affine
-> BatchNorm
-> ReLU
-> Dropout
-> 다음 층
```

코드 기준 위치:

```text
C:\Dev\Crafton-Jungle\04.AI\wk13_6_mnist\src\network.py
C:\Dev\Crafton-Jungle\04.AI\wk13_6_mnist\src\layers.py
C:\Dev\Crafton-Jungle\04.AI\wk13_6_mnist\src\activations.py
```

---

## 1. 전체 흐름 한 줄 요약

각 층의 역할은 다음과 같다.

```text
Affine    : 입력을 가중치 W와 편향 b로 선형 변환한다.
BatchNorm : 변환된 값을 feature별로 정규화해서 분포를 안정화한다.
ReLU      : 음수는 0으로 막고, 양수만 통과시킨다.
Dropout   : 학습 중 일부 뉴런 출력을 랜덤하게 0으로 꺼서 과적합을 줄인다.
```

즉 한 블록은:

```text
입력 x를 새로운 feature 공간으로 변환하고,
값의 분포를 안정화한 뒤,
비선형성을 넣고,
일부 뉴런을 꺼서 너무 외우지 못하게 만든다.
```

---

## 2. 예시에 사용할 입력

예시를 작게 만들기 위해 입력 샘플 2개, 입력 feature 3개를 사용한다.

```python
x = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
])
```

shape는:

```text
x.shape = (2, 3)
```

의미:

```text
batch_size = 2
input_dim  = 3
```

표로 보면:

```text
          feature0   feature1   feature2
sample0      1          2          3
sample1      4          5          6
```

이제 이 입력을 은닉 뉴런 2개짜리 층으로 보낸다고 하자.

```text
input_dim  = 3
output_dim = 2
```

---

## 3. Affine: xW + b

Affine layer는 완전연결층이다.

```python
out = x @ W + b
```

예를 들어:

```python
W = np.array([
    [ 0.1, -0.2],
    [ 0.3,  0.4],
    [-0.5,  0.2],
])

b = np.array([0.1, -0.1])
```

shape는:

```text
x.shape = (2, 3)
W.shape = (3, 2)
b.shape = (2,)
```

계산 결과 shape:

```text
(2, 3) @ (3, 2) + (2,) = (2, 2)
```

계산을 직접 해보면:

```text
sample0 = [1, 2, 3]

첫 번째 출력:
1*0.1 + 2*0.3 + 3*(-0.5) + 0.1
= 0.1 + 0.6 - 1.5 + 0.1
= -0.7

두 번째 출력:
1*(-0.2) + 2*0.4 + 3*0.2 - 0.1
= -0.2 + 0.8 + 0.6 - 0.1
= 1.1
```

```text
sample1 = [4, 5, 6]

첫 번째 출력:
4*0.1 + 5*0.3 + 6*(-0.5) + 0.1
= 0.4 + 1.5 - 3.0 + 0.1
= -1.0

두 번째 출력:
4*(-0.2) + 5*0.4 + 6*0.2 - 0.1
= -0.8 + 2.0 + 1.2 - 0.1
= 2.3
```

Affine 결과:

```text
z =
[
  [-0.7, 1.1],
  [-1.0, 2.3],
]
```

여기서 `z`는 다음 층으로 넘어가는 중간값이다.

Affine의 의미:

```text
기존 feature들을 가중합해서 새로운 feature를 만든다.
```

MNIST에서는:

```text
784개 픽셀 값
-> 512개 은닉 feature
```

처럼 차원을 바꾼다.

---

## 4. BatchNorm: feature별로 평균 0, 분산 1 근처로 맞추기

Affine 결과가:

```text
z =
[
  [-0.7, 1.1],
  [-1.0, 2.3],
]
```

라고 하자.

BatchNorm은 feature별로 평균과 분산을 구한다.

```text
feature0 값: [-0.7, -1.0]
feature1 값: [ 1.1,  2.3]
```

평균:

```text
mean0 = (-0.7 + -1.0) / 2 = -0.85
mean1 = ( 1.1 +  2.3) / 2 =  1.70

mean = [-0.85, 1.70]
```

평균을 뺀 값:

```text
z - mean =
[
  [ 0.15, -0.60],
  [-0.15,  0.60],
]
```

분산:

```text
var0 = ((0.15)^2 + (-0.15)^2) / 2 = 0.0225
var1 = ((-0.60)^2 + (0.60)^2) / 2 = 0.36

var = [0.0225, 0.36]
```

표준편차:

```text
sqrt(var) = [0.15, 0.60]
```

정규화:

```text
x_hat = (z - mean) / sqrt(var + eps)
```

대략:

```text
x_hat =
[
  [ 1.0, -1.0],
  [-1.0,  1.0],
]
```

처음에는 보통:

```python
gamma = [1, 1]
beta  = [0, 0]
```

이므로:

```text
BatchNorm output = gamma * x_hat + beta
                 = x_hat
```

BatchNorm 결과:

```text
bn_out =
[
  [ 1.0, -1.0],
  [-1.0,  1.0],
]
```

BatchNorm의 의미:

```text
Affine이 만든 값의 분포를 feature별로 안정화한다.
```

중요한 점:

```text
BatchNorm은 sample별로 하는 게 아니라 feature별로 한다.
즉 axis=0 방향으로 평균/분산을 구한다.
```

---

## 5. ReLU: 음수는 막고 양수만 통과

BatchNorm 결과가:

```text
bn_out =
[
  [ 1.0, -1.0],
  [-1.0,  1.0],
]
```

라고 하자.

ReLU는 다음 규칙을 적용한다.

```text
값이 0보다 크면 그대로 둔다.
값이 0 이하이면 0으로 바꾼다.
```

계산:

```text
ReLU( 1.0) = 1.0
ReLU(-1.0) = 0.0
```

결과:

```text
relu_out =
[
  [1.0, 0.0],
  [0.0, 1.0],
]
```

ReLU의 의미:

```text
단순한 선형 변환만 반복하지 않도록 비선형성을 넣는다.
```

만약 Affine만 여러 번 쌓으면:

```text
Affine -> Affine -> Affine
```

결국 하나의 큰 선형 변환과 비슷해진다.

ReLU를 넣으면:

```text
어떤 뉴런은 켜지고,
어떤 뉴런은 꺼지는 구조
```

가 생긴다. 그래서 더 복잡한 패턴을 표현할 수 있다.

---

## 6. Dropout: 학습 중 일부 뉴런 끄기

ReLU 결과가:

```text
relu_out =
[
  [1.0, 0.0],
  [0.0, 1.0],
]
```

라고 하자.

Dropout은 학습 중 `mask`를 만든다.

예를 들어 `drop_ratio = 0.5`이고, 랜덤 mask가 이렇게 나왔다고 하자.

```text
mask =
[
  [ True, False],
  [ True,  True],
]
```

이 mask를 곱한다.

```text
dropout_out = relu_out * mask
```

계산:

```text
[
  [1.0, 0.0],
  [0.0, 1.0],
]
*
[
  [ True, False],
  [ True,  True],
]
=
[
  [1.0, 0.0],
  [0.0, 1.0],
]
```

이번 예시는 원래 0인 곳이 많아서 변화가 작아 보인다.

다른 예시를 보면 더 명확하다.

```text
relu_out =
[
  [2.0, 3.0],
  [4.0, 5.0],
]

mask =
[
  [ True, False],
  [False,  True],
]

dropout_out =
[
  [2.0, 0.0],
  [0.0, 5.0],
]
```

Dropout의 의미:

```text
학습 중 매번 일부 뉴런을 꺼서,
특정 뉴런 조합에만 과하게 의존하지 않게 만든다.
```

현재 프로젝트 구현은 학습/추론을 이렇게 나눈다.

```text
train=True:
    mask를 새로 만들고 x * mask 반환

train=False:
    랜덤으로 끄지 않고 x * (1 - drop_ratio) 반환
```

---

## 7. 전체 예시를 한 번에 보기

입력:

```text
x =
[
  [1, 2, 3],
  [4, 5, 6],
]
```

Affine 후:

```text
z =
[
  [-0.7, 1.1],
  [-1.0, 2.3],
]
```

BatchNorm 후:

```text
bn_out =
[
  [ 1.0, -1.0],
  [-1.0,  1.0],
]
```

ReLU 후:

```text
relu_out =
[
  [1.0, 0.0],
  [0.0, 1.0],
]
```

Dropout 후:

```text
dropout_out =
[
  [1.0, 0.0],
  [0.0, 1.0],
]
```

이 값이 다음 층의 입력이 된다.

```text
다음 Affine의 x = dropout_out
```

---

## 8. shape 흐름

작은 예시의 shape:

```text
x           : (2, 3)
W           : (3, 2)
b           : (2,)

Affine out  : (2, 2)
BatchNorm   : (2, 2)
ReLU        : (2, 2)
Dropout     : (2, 2)
```

MNIST 실제 첫 번째 은닉층:

```text
x           : (batch_size, 784)
W1          : (784, 512)
b1          : (512,)

Affine1 out : (batch_size, 512)
BatchNorm1  : (batch_size, 512)
ReLU1       : (batch_size, 512)
Dropout1    : (batch_size, 512)
```

두 번째 은닉층:

```text
x           : (batch_size, 512)
W2          : (512, 256)
b2          : (256,)

Affine2 out : (batch_size, 256)
BatchNorm2  : (batch_size, 256)
ReLU2       : (batch_size, 256)
Dropout2    : (batch_size, 256)
```

출력층:

```text
x           : (batch_size, 256)
W3          : (256, 10)
b3          : (10,)

Affine3 out : (batch_size, 10)
Softmax     : (batch_size, 10)
```

출력층 뒤에는 보통 `BatchNorm`, `ReLU`, `Dropout`을 붙이지 않는다.

---

## 9. 각 층이 학습하는 것

각 층이 직접 학습하거나 저장하는 값은 다르다.

```text
Affine
- 학습하는 값: W, b
- backward에서 만드는 gradient: dW, db

BatchNorm
- 학습하는 값: gamma, beta
- 학습 중 저장하는 값: running_mean, running_var
- backward에서 만드는 gradient: dgamma, dbeta

ReLU
- 학습하는 값 없음
- forward 때 mask 저장
- backward 때 x <= 0이었던 위치의 gradient를 0으로 막음

Dropout
- 학습하는 값 없음
- forward 때 mask 저장
- backward 때 꺼졌던 뉴런의 gradient를 0으로 막음
```

---

## 10. 직관 요약

이 블록을 사람 말로 풀면:

```text
Affine:
    입력 feature들을 섞어서 새로운 feature를 만든다.

BatchNorm:
    새 feature들의 값 범위가 너무 흔들리지 않게 정리한다.

ReLU:
    양수 신호만 살리고 음수 신호는 끈다.

Dropout:
    학습 중 일부 신호를 랜덤으로 꺼서 특정 뉴런에 과하게 의존하지 않게 한다.
```

따라서:

```text
Affine -> BatchNorm -> ReLU -> Dropout
```

은 단순히 값을 통과시키는 파이프라인이 아니라:

```text
변환 -> 안정화 -> 선택적 활성화 -> 과적합 방지
```

의 흐름이라고 보면 된다.

