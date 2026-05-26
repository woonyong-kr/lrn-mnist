# 01. Python, NumPy, Shape

딥러닝 구현에서 Python 자체보다 더 중요한 건 NumPy 배열의 shape를 읽는 능력임.

신경망 계산은 대부분 아래 두 가지로 이루어짐.

```text
행렬곱
원소별 연산
```

MNIST를 예로 들면 이미지 하나는 28x28 픽셀이고, 펼치면 784개 숫자임.

```text
이미지 1개:      784
이미지 N개:      N x 784
가중치 W1:       784 x 512
첫 은닉층 출력:  N x 512
```

## 1. 스칼라, 벡터, 행렬, 텐서

```text
스칼라: 숫자 하나
  3

벡터: 숫자 여러 개
  [1, 2, 3]

행렬: 숫자를 2차원 표로 둔 것
  [
    [1, 2],
    [3, 4],
  ]

텐서: 3차원 이상까지 포함하는 배열
```

딥러닝에서는 전부 텐서라고 부르기도 함.

## 2. Shape 읽기

shape는 배열의 크기임.

```python
x.shape
```

MNIST 학습 데이터는 보통:

```text
x_train.shape = (60000, 784)
y_train.shape = (60000,)
```

의미:

```text
60000: 이미지 개수
784: 이미지 하나의 픽셀 개수
```

테스트 데이터는:

```text
x_test.shape = (10000, 784)
y_test.shape = (10000,)
```

## 3. 행렬곱

NumPy에서 행렬곱은 `@` 또는 `np.dot`을 씀.

```python
out = x @ W
```

shape 규칙:

```text
(N x A) @ (A x B) = (N x B)
```

중간 차원 A가 같아야 함.

예:

```text
x:  1 x 3
W:  3 x 2
out: 1 x 2
```

숫자로 보면:

```text
x = [1, 2, 3]

W = [
  [0.1, 0.2],
  [0.0, 0.1],
  [0.3, -0.2],
]
```

계산:

```text
out[0] = 1*0.1 + 2*0.0 + 3*0.3 = 1.0
out[1] = 1*0.2 + 2*0.1 + 3*(-0.2) = -0.2
```

결과:

```text
out = [1.0, -0.2]
```

## 4. Bias와 브로드캐스팅

Affine 계층은:

```text
out = x @ W + b
```

여기서:

```text
x @ W: N x output_dim
b:     output_dim
```

NumPy는 b를 자동으로 각 행에 더함.

예:

```text
x @ W =
[
  [1.0, -0.2],
  [0.5,  0.7],
]

b = [0.1, 0.2]
```

결과:

```text
[
  [1.1, 0.0],
  [0.6, 0.9],
]
```

이걸 브로드캐스팅이라고 함.

## 5. 원소별 연산

Sigmoid, ReLU, 제곱, exp, log 같은 함수는 보통 배열 원소마다 적용됨.

```python
np.exp(x)
np.log(x)
x ** 2
```

예:

```text
x = [1, 2, 3]
np.exp(x) = [e^1, e^2, e^3]
```

## 6. axis 이해

배치 계산에서 자주 나오는 것이 `axis`임.

```text
axis=0: 세로 방향, 행들을 따라 계산
axis=1: 가로 방향, 열들을 따라 계산
```

예:

```text
x =
[
  [1, 2, 3],
  [4, 5, 6],
]
```

```text
np.sum(x, axis=0) = [5, 7, 9]
np.sum(x, axis=1) = [6, 15]
```

딥러닝에서 bias gradient는 보통:

```python
db = np.sum(dout, axis=0)
```

임.

이 말은:

```text
배치 안의 여러 샘플에서 나온 오차를 feature 위치별로 합친다.
```

는 뜻임.

## 7. keepdims

softmax에서 자주 나옴.

```python
np.max(x, axis=1, keepdims=True)
```

`keepdims=True`를 쓰면 차원을 유지함.

예:

```text
x.shape = (3, 10)
np.max(x, axis=1).shape = (3,)
np.max(x, axis=1, keepdims=True).shape = (3, 1)
```

softmax에서는 각 행마다 최댓값을 빼야 하므로 `(3, 1)` 형태가 편함.

```python
shifted = x - np.max(x, axis=1, keepdims=True)
```

## 8. 배치 처리

샘플 하나씩 처리하면:

```text
x: 784
```

하지만 보통은 여러 장을 한 번에 처리함.

```text
x_batch: batch_size x 784
```

이유:

```text
행렬 연산이 빠름
gradient가 더 안정적임
GPU/NumPy 계산에 유리함
```

mini-batch 학습은 전체 데이터에서 일부만 뽑아 한 번 업데이트하는 방식임.

```text
전체 데이터 60000개
batch_size 128
한 번 업데이트에 128개 사용
```

## 9. 학습에서 자주 보는 shape

현재 과제 네트워크는:

```text
784 -> 512 -> 256 -> 10
```

batch_size가 128이면:

```text
x:      128 x 784
W1:     784 x 512
b1:     512
a1:     128 x 512

W2:     512 x 256
b2:     256
a2:     128 x 256

W3:     256 x 10
b3:     10
logits: 128 x 10
y:      128 x 10
t:      128
```

## 10. 실수 방지 체크

행렬 연산이 헷갈리면 이 순서로 확인하면 됨.

```text
1. x shape를 적는다.
2. W shape를 적는다.
3. 가운데 차원이 맞는지 본다.
4. 결과 shape를 적는다.
5. b가 결과의 마지막 차원과 맞는지 본다.
```

예:

```text
x @ W + b

x: 128 x 784
W: 784 x 512
b: 512

결과: 128 x 512
```

## 확인 질문

```text
1. (100 x 784) @ (784 x 50)의 결과 shape는?
2. b가 (50,)이면 위 결과에 더할 수 있는가?
3. np.sum(dout, axis=0)은 왜 bias gradient에 쓰이는가?
4. softmax에서 keepdims=True를 쓰는 이유는?
```

