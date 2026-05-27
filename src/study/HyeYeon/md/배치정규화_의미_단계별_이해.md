# 배치 정규화의 의미 단계별 이해

이 문서는 `BatchNorm`이 왜 필요한지, `forward`에서 어떤 계산을 하는지, `gamma`와 `beta`가 왜 있는지 예시 중심으로 정리한다.

코드 기준 위치:

```text
C:\Dev\Crafton-Jungle\04.AI\wk13_6_mnist\src\layers.py
```

---

## 0. 왜 평균 0, 분산 1 근처로 맞추는가

핵심은:

```text
다음 층이 받는 입력의 중심과 크기를 안정적으로 만들기 위해서
```

이다.

신경망에서 한 층은 앞 층의 출력을 입력으로 받는다.

```text
이전 층
-> 현재 층
-> 다음 층
```

그런데 학습 중에는 이전 층의 `W`, `b`가 계속 바뀐다. 그러면 다음 층이 받는 입력 분포도 계속 바뀐다.

예를 들어 어떤 층이 처음에는 이런 값을 받았다고 하자.

```text
[-1, 0, 1]
```

그런데 앞 층의 가중치가 업데이트되면서 다음에는 이런 값이 들어올 수 있다.

```text
[50, 70, 90]
```

또 다른 업데이트 후에는 이렇게 치우칠 수도 있다.

```text
[-100, -80, -60]
```

이러면 다음 층 입장에서는 매번 입력의 상태가 달라진다.

```text
이번에는 입력이 너무 크다.
이번에는 값이 전부 음수 쪽으로 치우쳤다.
이번에는 값의 퍼짐이 너무 작다.
```

특히 ReLU 같은 활성화 함수는 입력 분포의 영향을 크게 받는다.

```text
입력이 대부분 음수
-> ReLU 출력이 거의 0
-> gradient가 잘 흐르지 않음

입력이 너무 큼
-> 다음 층 출력도 커짐
-> 학습이 불안정해질 수 있음
```

그래서 BatchNorm은 Affine 출력 값을 일단 다루기 쉬운 형태로 정리한다.

```text
평균 0 근처
분산 1 근처
```

평균을 0으로 맞춘다는 것은:

```text
값들의 중심을 0 근처로 옮긴다
```

는 뜻이다.

분산을 1로 맞춘다는 것은:

```text
값들의 퍼짐 정도를 적당한 크기로 맞춘다
```

는 뜻이다.

예를 들어 Affine 출력이 다음과 같다고 하자.

```text
[100, 120, 140]
```

이 값은 중심이 120 근처에 있고, 스케일도 크다.

먼저 평균 120을 빼면:

```text
[-20, 0, 20]
```

값의 중심이 0으로 이동한다.

그 다음 표준편차로 나누면 대략:

```text
[-1.22, 0, 1.22]
```

처럼 값의 퍼짐도 적당한 범위가 된다.

즉 BatchNorm은 다음 층에게 이런 효과를 준다.

```text
앞 층이 어떤 스케일의 값을 만들었든,
일단 비슷한 범위로 정리해서 넘겨준다.
```

다만 모든 값을 무조건 평균 0, 분산 1로 고정하면 표현력이 줄어들 수 있다. 어떤 층은 평균이 3 근처인 입력이나, 더 크게 퍼진 입력이 필요할 수도 있기 때문이다.

그래서 BatchNorm에는 `gamma`, `beta`가 있다.

```python
out = gamma * x_hat + beta
```

의미는:

```text
일단 평균 0, 분산 1 근처로 안정화한다.
하지만 필요한 경우 gamma와 beta를 학습해서 다시 크기와 위치를 조정한다.
```

정리하면:

```text
평균을 0으로 맞춘다
-> 값의 중심을 안정화한다.

분산을 1로 맞춘다
-> 값의 크기와 퍼짐을 안정화한다.

gamma와 beta를 둔다
-> 안정화하되, 필요한 분포는 다시 학습할 수 있게 한다.
```

---

## 1. 배치 정규화가 하는 일

배치 정규화는 한마디로:

```text
한 미니배치 안에서 feature별 값의 평균을 0 근처,
분산을 1 근처로 맞춘 뒤,
다시 gamma와 beta로 필요한 만큼 조정하는 층
```

이다.

신경망의 각 층은 앞 층의 출력을 입력으로 받는다.

```text
입력 x
-> Affine
-> BatchNorm
-> ReLU
-> 다음 층
```

그런데 학습 중에는 앞 층의 가중치가 계속 바뀐다. 그러면 다음 층 입장에서는 입력 분포가 계속 흔들린다.

```text
처음에는 입력이 대략 -1 ~ 1 사이였는데,
다음 업데이트 후에는 20 ~ 40 사이가 될 수 있음
```

이러면 다음 층은 매번 다른 스케일의 입력에 적응해야 한다. 배치 정규화는 이 흔들림을 줄이기 위해 중간 출력을 일정한 범위로 맞춘다.

---

## 2. feature마다 정규화한다는 뜻

입력 `x`가 다음과 같다고 하자.

```python
x = np.array([
    [10, 100],
    [20, 120],
    [30, 140],
])
```

shape는:

```text
(batch_size, feature_dim) = (3, 2)
```

해석하면:

```text
샘플 3개
feature 2개
```

표로 보면:

```text
          feature 0   feature 1
sample 0     10         100
sample 1     20         120
sample 2     30         140
```

BatchNorm은 sample별로 정규화하는 게 아니라, **feature별로** 정규화한다.

즉:

```text
feature 0: [10, 20, 30]끼리 평균/분산 계산
feature 1: [100, 120, 140]끼리 평균/분산 계산
```

그래서 NumPy에서는 `axis=0`을 쓴다.

```python
mean = np.mean(x, axis=0)
var = np.var(x, axis=0)
```

결과:

```text
mean = [20, 120]
var  = [66.666..., 266.666...]
```

---

## 3. 평균을 빼는 이유

먼저 feature별 평균을 뺀다.

```python
x_centered = x - mean
```

계산:

```text
x =
[
  [10, 100],
  [20, 120],
  [30, 140],
]

mean = [20, 120]

x - mean =
[
  [-10, -20],
  [  0,   0],
  [ 10,  20],
]
```

이제 각 feature는 평균이 0인 값들이 된다.

```text
feature 0: [-10, 0, 10]   평균 0
feature 1: [-20, 0, 20]   평균 0
```

평균을 뺀다는 것은:

```text
값들의 중심을 0으로 옮긴다
```

라고 보면 된다.

---

## 4. 표준편차로 나누는 이유

평균만 0으로 맞추면 아직 scale은 다를 수 있다.

```text
feature 0: [-10, 0, 10]
feature 1: [-20, 0, 20]
```

feature 1은 feature 0보다 변화 폭이 2배 크다.

그래서 표준편차로 나눠서 scale을 비슷하게 맞춘다.

```python
x_hat = (x - mean) / np.sqrt(var + eps)
```

여기서 `eps`는 0으로 나누는 일을 막기 위한 아주 작은 값이다.

대략 계산하면:

```text
sqrt(var) = [8.16, 16.33]
```

따라서:

```text
feature 0:
[-10, 0, 10] / 8.16
= [-1.22, 0, 1.22]

feature 1:
[-20, 0, 20] / 16.33
= [-1.22, 0, 1.22]
```

최종 `x_hat`은 대략:

```text
[
  [-1.22, -1.22],
  [ 0.00,  0.00],
  [ 1.22,  1.22],
]
```

이제 두 feature 모두:

```text
평균은 0 근처
분산은 1 근처
```

가 된다.

---

## 5. 그러면 모든 값이 비슷해져서 표현력이 사라지지 않나?

이 의문이 중요하다.

정규화만 하면 항상 평균 0, 분산 1인 값으로 고정된다. 그러면 어떤 층에서는 오히려 불편할 수 있다.

예를 들어 다음 층이 사실은 이런 입력을 원할 수도 있다.

```text
평균이 3 근처이고,
값의 폭이 2배 정도 큰 입력
```

그래서 BatchNorm은 정규화 후에 `gamma`, `beta`를 적용한다.

```python
out = gamma * x_hat + beta
```

의미:

```text
gamma: 정규화된 값을 얼마나 크게/작게 늘릴지
beta : 정규화된 값을 어느 방향으로 이동시킬지
```

처음에는 보통:

```python
gamma = np.ones(feature_dim)
beta = np.zeros(feature_dim)
```

로 둔다.

그러면 처음에는:

```text
out = 1 * x_hat + 0
```

즉 정규화 결과를 그대로 쓴다.

하지만 학습이 진행되면 optimizer가 `gamma`, `beta`도 업데이트한다. 그러면 네트워크가 필요한 경우 정규화된 값을 다시 적절한 위치와 크기로 바꿀 수 있다.

---

## 6. gamma와 beta 예시

정규화 결과가 다음과 같다고 하자.

```text
x_hat =
[
  [-1.22, -1.22],
  [ 0.00,  0.00],
  [ 1.22,  1.22],
]
```

그리고:

```python
gamma = np.array([2, 0.5])
beta = np.array([10, -3])
```

이면:

```python
out = gamma * x_hat + beta
```

계산은 feature별로 적용된다.

```text
feature 0:
2 * [-1.22, 0, 1.22] + 10
= [7.56, 10.00, 12.44]

feature 1:
0.5 * [-1.22, 0, 1.22] - 3
= [-3.61, -3.00, -2.39]
```

결과:

```text
out =
[
  [ 7.56, -3.61],
  [10.00, -3.00],
  [12.44, -2.39],
]
```

즉 BatchNorm은 무조건 값을 평균 0, 분산 1로 고정하는 층이 아니다.

정확히는:

```text
일단 안정적인 형태로 정규화한 뒤,
학습 가능한 gamma/beta로 필요한 분포를 다시 만들 수 있게 하는 층
```

이다.

---

## 7. train=True일 때와 train=False일 때

BatchNorm은 학습 중과 추론 중 동작이 다르다.

### 학습 중 train=True

학습 중에는 현재 미니배치의 평균과 분산을 사용한다.

```python
mean = np.mean(x, axis=0)
var = np.var(x, axis=0)
x_hat = (x - mean) / np.sqrt(var + eps)
out = gamma * x_hat + beta
```

그리고 동시에 running 통계를 업데이트한다.

```python
running_mean = momentum * running_mean + (1 - momentum) * mean
running_var = momentum * running_var + (1 - momentum) * var
```

### 추론 중 train=False

추론할 때는 보통 샘플 하나 또는 작은 배치가 들어올 수 있다. 이때 현재 입력만으로 평균/분산을 계산하면 불안정하다.

그래서 학습 중에 모아둔 running 통계를 사용한다.

```python
x_hat = (x - running_mean) / np.sqrt(running_var + eps)
out = gamma * x_hat + beta
```

즉:

```text
train=True  : 현재 batch의 mean/var 사용
train=False : 학습 중 누적한 running_mean/running_var 사용
```

---

## 8. running_mean과 running_var가 필요한 이유

예를 들어 학습 중 어떤 feature의 batch 평균들이 이렇게 나왔다고 하자.

```text
batch 1 mean = 10
batch 2 mean = 12
batch 3 mean = 9
batch 4 mean = 11
```

추론 때는 이 전체 흐름을 대표하는 평균이 필요하다.

단순히 마지막 batch 평균만 쓰면:

```text
running_mean = 11
```

처럼 마지막 batch에 너무 의존한다.

그래서 이동평균을 쓴다.

```text
running = momentum * running + (1 - momentum) * current_batch_stat
```

`momentum = 0.9`라면:

```text
기존 running 통계 90%
현재 batch 통계 10%
```

를 섞는다.

이렇게 하면 batch마다 튀는 값을 부드럽게 누적할 수 있다.

---

## 9. 코드와 연결하기

현재 `BatchNorm.forward`의 핵심 흐름은 다음과 같다.

```python
if train:
    batch_size = x.shape[0]
    mean = (1 / batch_size) * np.sum(x, axis=0)
    var = (1 / batch_size) * np.sum((x - mean) ** 2, axis=0)

    x_hat = (x - mean) / np.sqrt(var + self.eps)

    self.x = x
    self.mean = mean
    self.var = var
    self.x_hat = x_hat

    self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
    self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var

    return self.gamma * x_hat + self.beta
else:
    x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
    return self.gamma * x_hat + self.beta
```

여기서 중요한 축은 `axis=0`이다.

```text
axis=0
= batch 방향으로 묶어서 feature별 평균/분산 계산
```

---

## 10. 직관 요약

BatchNorm이 없으면:

```text
앞 층의 가중치가 바뀜
-> 다음 층 입력 분포가 계속 흔들림
-> 학습이 불안정해질 수 있음
```

BatchNorm이 있으면:

```text
앞 층 출력
-> feature별 평균 0, 분산 1 근처로 정리
-> gamma/beta로 필요한 크기와 위치를 다시 학습
-> 다음 층이 더 안정적인 입력을 받음
```

최종적으로 BatchNorm은:

```text
값을 무조건 작게 만드는 층이 아니라,
중간 데이터의 분포를 안정화하고,
필요한 스케일과 위치는 다시 학습하게 해주는 층
```

이라고 이해하면 된다.
