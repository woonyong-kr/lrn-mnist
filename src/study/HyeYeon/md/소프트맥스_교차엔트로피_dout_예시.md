# Softmax + CrossEntropy의 dout 만들기

## 1. 상황

Softmax와 CrossEntropy를 같이 쓰면 출력층 gradient는 다음처럼 단순해진다.

```text
dout = y_pred - one_hot(y_batch)
```

여기서:

```text
y_pred: 모델이 예측한 클래스별 확률
y_batch: 정답 클래스 번호
dout: backward에 넘길 출력층 gradient
```

MNIST에서는 정답이 one-hot 배열이 아니라 정수 라벨로 들어온다.

```python
y_batch = np.array([2, 0, 1])
```

따라서 one-hot을 직접 만들지 않고, 정답 클래스 위치에서만 `1`을 빼는 방식으로 `dout`을 만든다.

## 2. 예시 y_pred

예측 확률이 다음과 같다고 하자.

```python
y_pred = np.array([
    [0.1, 0.2, 0.7],
    [0.8, 0.1, 0.1],
    [0.2, 0.6, 0.2],
])
```

의미:

```text
샘플 0:
클래스 0 확률 0.1
클래스 1 확률 0.2
클래스 2 확률 0.7

샘플 1:
클래스 0 확률 0.8
클래스 1 확률 0.1
클래스 2 확률 0.1

샘플 2:
클래스 0 확률 0.2
클래스 1 확률 0.6
클래스 2 확률 0.2
```

정답은 다음과 같다고 하자.

```python
y_batch = np.array([2, 0, 1])
```

의미:

```text
샘플 0의 정답: 클래스 2
샘플 1의 정답: 클래스 0
샘플 2의 정답: 클래스 1
```

## 3. y_pred를 복사한다

먼저 예측값을 복사한다.

```python
dout = y_pred.copy()
```

복사 직후:

```python
dout = np.array([
    [0.1, 0.2, 0.7],
    [0.8, 0.1, 0.1],
    [0.2, 0.6, 0.2],
])
```

원본 `y_pred`를 직접 바꾸지 않기 위해 `copy()`를 사용한다.

## 4. 행 번호를 만든다

```python
np.arange(len(y_batch))
```

`len(y_batch)`는 3이다.

```python
np.arange(3)
```

결과:

```python
array([0, 1, 2])
```

이 값은 샘플 번호, 즉 행 번호이다.

```text
0번 행: 샘플 0
1번 행: 샘플 1
2번 행: 샘플 2
```

## 5. y_batch는 열 번호이다

```python
y_batch = np.array([2, 0, 1])
```

이 값은 각 샘플의 정답 클래스 번호이다.

즉 열 번호이다.

```text
샘플 0 정답 클래스: 2
샘플 1 정답 클래스: 0
샘플 2 정답 클래스: 1
```

## 6. 정답 위치만 선택한다

다음 코드는:

```python
dout[np.arange(len(y_batch)), y_batch]
```

다음과 같다.

```python
dout[[0, 1, 2], [2, 0, 1]]
```

이것은 다음 위치를 고른다.

```text
dout[0, 2]
dout[1, 0]
dout[2, 1]
```

즉 각 샘플의 정답 클래스 위치만 선택한다.

현재 값은:

```text
dout[0, 2] = 0.7
dout[1, 0] = 0.8
dout[2, 1] = 0.6
```

## 7. 정답 위치에서 1을 뺀다

```python
dout[np.arange(len(y_batch)), y_batch] -= 1
```

계산:

```text
dout[0, 2] = 0.7 - 1 = -0.3
dout[1, 0] = 0.8 - 1 = -0.2
dout[2, 1] = 0.6 - 1 = -0.4
```

나머지 위치는 그대로 둔다.

결과:

```python
dout = np.array([
    [ 0.1,  0.2, -0.3],
    [-0.2,  0.1,  0.1],
    [ 0.2, -0.4,  0.2],
])
```

## 8. one-hot으로 직접 계산하면 같은 결과

정답을 one-hot으로 만들면:

```python
one_hot = np.array([
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
])
```

그러면:

```python
y_pred - one_hot
```

은:

```python
np.array([
    [0.1, 0.2, 0.7],
    [0.8, 0.1, 0.1],
    [0.2, 0.6, 0.2],
])
-
np.array([
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
])
```

결과:

```python
np.array([
    [ 0.1,  0.2, -0.3],
    [-0.2,  0.1,  0.1],
    [ 0.2, -0.4,  0.2],
])
```

앞에서 정답 위치에만 `1`을 뺀 결과와 같다.

## 9. batch 평균을 낸다

CrossEntropy loss는 보통 batch 평균으로 계산한다.

따라서 gradient도 batch 크기로 나눈다.

```python
dout /= len(y_batch)
```

여기서는 batch size가 3이다.

```python
dout = np.array([
    [ 0.0333,  0.0667, -0.1000],
    [-0.0667,  0.0333,  0.0333],
    [ 0.0667, -0.1333,  0.0667],
])
```

이 값을 `model.backward()`에 넘긴다.

```python
model.backward(dout)
```

## 10. 최종 코드

```python
y_pred = model.forward(x_batch, train=True)
loss = cross_entropy_loss(y_pred, y_batch)

dout = y_pred.copy()
dout[np.arange(len(y_batch)), y_batch] -= 1
dout /= len(y_batch)

model.backward(dout)
optimizer.update(model.params, model.grads)
```

## 11. 핵심 정리

```text
y_pred는 예측 확률이다.
y_batch는 정답 클래스 번호이다.
np.arange(len(y_batch))는 행 번호이다.
y_batch는 열 번호이다.
두 배열을 같이 쓰면 각 샘플의 정답 위치만 고를 수 있다.
정답 위치에서 1을 빼면 y_pred - one_hot(y_batch)와 같은 결과가 된다.
```
