# Dropout mask 트러블슈팅: `x[self.mask]`와 `x * self.mask`의 차이

## 문제 상황

Dropout의 `forward`를 구현하면서 처음에는 다음처럼 작성했다.

```python
x = x[self.mask]
```

하지만 테스트에서 다음 오류가 발생했다.

```text
assert (4,) == (3, 4)
```

즉 입력 `x`의 shape는 `(3, 4)`였는데, 출력 `out`의 shape가 `(4,)`로 바뀌었다.

Dropout에서는 출력 shape가 입력 shape와 같아야 한다.

```text
입력 shape:  (3, 4)
출력 shape:  (3, 4)
```

따라서 `x[self.mask]` 방식은 Dropout 구현에 맞지 않는다.

## Dropout에서 원하는 동작

Dropout은 일부 뉴런의 출력을 0으로 만든다.

중요한 점은 배열에서 원소를 제거하는 것이 아니라, 원래 위치는 유지하면서 값만 0으로 바꾼다는 것이다.

예를 들어 입력이 다음과 같다고 하자.

```python
x = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
])
```

Dropout 결과는 이런 형태여야 한다.

```python
out = np.array([
    [0, 2, 0, 4],
    [5, 0, 7, 0],
    [0, 10, 11, 0],
])
```

값 일부가 0이 되었지만 shape는 그대로이다.

```python
x.shape
# (3, 4)

out.shape
# (3, 4)
```

## 내가 처음 쓴 코드가 왜 안 됐나

처음 쓴 코드는 다음과 같다.

```python
x = x[self.mask]
```

이 코드는 mask가 `True`인 위치만 골라낸다.

예를 들어:

```python
x = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
])
```

mask가 다음과 같다고 하자.

```python
self.mask = np.array([
    [False, True, False, True],
    [True, False, True, False],
    [False, True, True, False],
])
```

이때:

```python
x[self.mask]
```

는 `True`인 위치의 값만 골라낸다.

```python
array([2, 4, 5, 7, 10, 11])
```

즉 원래 2차원 배열이었던 `(3, 4)` 구조가 사라지고, 선택된 값만 모인 1차원 배열이 된다.

```python
x.shape
# (3, 4)

x[self.mask].shape
# (6,)
```

이것이 테스트에서 shape가 달라진 이유이다.

## 왜 테스트에서는 `(4,)`가 나왔나

테스트 입력은 다음과 비슷한 형태이다.

```python
x = np.random.randn(3, 4)
```

즉 shape는 `(3, 4)`이다.

만약 mask가 어떤 방식으로 만들어져서 첫 번째 축이나 특정 행만 선택되면 출력 shape가 `(4,)`가 될 수 있다.

예를 들어:

```python
x = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
])
```

행 단위 mask가 다음과 같으면:

```python
mask = np.array([True, False, False])
```

이때:

```python
x[mask]
```

결과는 첫 번째 행만 선택된다.

```python
array([[1, 2, 3, 4]])
```

shape는:

```python
(1, 4)
```

만약 어떤 방식으로 한 행 자체를 꺼내게 되면:

```python
x[0]
```

shape는:

```python
(4,)
```

이처럼 boolean indexing은 값을 선택하는 기능이라, 원래 shape를 보장하지 않는다.

Dropout에서는 이 방식이 위험하다.

## 올바른 방식: `x * self.mask`

Dropout에서는 mask로 값을 선택하는 것이 아니라, mask를 곱해서 일부 값을 0으로 만들어야 한다.

```python
x = x * self.mask
```

예를 들어:

```python
x = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
])
```

mask:

```python
self.mask = np.array([
    [False, True, False, True],
    [True, False, True, False],
    [False, True, True, False],
])
```

곱셈:

```python
x * self.mask
```

NumPy에서 `True`는 `1`, `False`는 `0`처럼 계산된다.

따라서:

```python
array([
    [0, 2, 0, 4],
    [5, 0, 7, 0],
    [0, 10, 11, 0],
])
```

shape는 그대로 유지된다.

```python
(3, 4)
```

## 핵심 차이

```python
x[self.mask]
```

의 의미:

```text
mask가 True인 원소만 골라낸다.
원래 배열 구조가 깨질 수 있다.
Dropout에는 부적합하다.
```

```python
x * self.mask
```

의 의미:

```text
mask가 False인 위치는 0으로 만든다.
mask가 True인 위치는 원래 값을 유지한다.
shape가 유지된다.
Dropout에 적합하다.
```

## Dropout 구현에서 기억할 점

Dropout의 목적은 뉴런을 “삭제”하는 것이 아니라, 해당 step에서 출력값을 “0으로 막는 것”이다.

따라서 배열의 크기와 구조는 유지되어야 한다.

```text
잘못된 방향:
값을 선택해서 배열을 줄인다.

올바른 방향:
원래 배열에 mask를 곱해서 일부 값만 0으로 만든다.
```

정리하면:

```python
# 틀린 접근
x = x[self.mask]

# 맞는 접근
x = x * self.mask
```
