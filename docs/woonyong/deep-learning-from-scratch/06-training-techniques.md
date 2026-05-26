# 06. 학습 개선 기법

여기서는 학습을 더 잘 되게 만드는 기법들을 정리함.

큰 흐름은 이거임.

```text
optimizer:
  gradient를 어떻게 이용해 W, b를 업데이트할지

initialization:
  처음 W를 어떻게 잡을지

BatchNorm:
  각 층의 입력 분포를 안정화

regularization:
  overfitting을 줄임

hyperparameter search:
  lr, batch_size, hidden size 같은 값을 찾음
```

## 1. Optimizer가 하는 일

역전파가 끝나면 gradient가 생김.

```text
dW1, db1, dW2, db2, ...
```

optimizer는 이 gradient를 보고 파라미터를 업데이트함.

```text
params <- params - update
```

즉 optimizer는 gradient를 만드는 함수가 아님.

```text
backpropagation:
  gradient를 만든다.

optimizer:
  gradient를 가지고 파라미터를 바꾼다.
```

## 2. SGD

가장 단순한 방식:

```text
W = W - lr * dW
```

예:

```text
W = 1.0
dW = 0.3
lr = 0.1

W_new = 1.0 - 0.1*0.3
      = 0.97
```

장점:

```text
단순함
이해하기 좋음
구현이 쉬움
```

단점:

```text
지형이 복잡하면 흔들림
방향마다 gradient 크기가 다르면 비효율적
학습률 선택에 민감함
```

## 3. Momentum

Momentum은 이전 이동 방향을 기억함.

```text
v = momentum * v - lr * grad
W = W + v
```

여기서:

```text
v: velocity, 이동 속도
```

의미:

```text
계속 같은 방향으로 gradient가 나오면 더 빠르게 감
방향이 왔다 갔다 하는 축은 흔들림이 줄어듦
```

숫자 예:

```text
momentum = 0.9
lr = 0.1
v = 0
grad = 0.3

v = 0.9*0 - 0.1*0.3 = -0.03
W = W + v
```

다음에도 grad가 0.3이면:

```text
v = 0.9*(-0.03) - 0.1*0.3
  = -0.027 - 0.03
  = -0.057
```

이전 속도가 누적되어 더 크게 움직임.

## 4. AdaGrad

AdaGrad는 파라미터별로 학습률을 조절함.

```text
h = h + grad^2
W = W - lr * grad / (sqrt(h) + eps)
```

의미:

```text
gradient가 자주 크게 나온 파라미터는 점점 작게 움직임
gradient가 별로 안 나온 파라미터는 상대적으로 크게 움직임
```

숫자 예:

```text
grad = 0.2
h = 0

h = 0 + 0.2^2 = 0.04
update = lr * 0.2 / sqrt(0.04)
       = lr * 0.2 / 0.2
       = lr
```

같은 파라미터에 gradient가 계속 쌓이면 h가 커지고 update가 작아짐.

단점:

```text
h가 계속 누적되므로 나중에는 거의 안 움직일 수 있음
```

## 5. RMSProp

RMSProp은 AdaGrad의 누적 문제를 줄이려고 이동평균을 씀.

```text
v = beta * v + (1 - beta) * grad^2
W = W - lr * grad / (sqrt(v) + eps)
```

AdaGrad와 차이:

```text
AdaGrad:
  과거 gradient^2를 계속 누적

RMSProp:
  최근 gradient^2를 더 중요하게 반영
```

그래서 학습 후반에도 업데이트가 완전히 작아지는 문제를 줄임.

## 6. Adam

Adam은 Momentum과 RMSProp을 합친 느낌임.

Adam은 두 값을 기억함.

```text
m: gradient의 이동평균
v: gradient 제곱의 이동평균
```

공식:

```text
m = beta1 * m + (1 - beta1) * grad
v = beta2 * v + (1 - beta2) * grad^2

m_hat = m / (1 - beta1^t)
v_hat = v / (1 - beta2^t)

W = W - lr * m_hat / (sqrt(v_hat) + eps)
```

기본값:

```text
beta1 = 0.9
beta2 = 0.999
eps = 1e-8
```

## 7. Adam bias correction

Adam은 m과 v를 0에서 시작함.

첫 gradient를 g라고 하면:

```text
m1 = 0.9*0 + 0.1*g = 0.1g
```

실제 gradient는 g인데, m1은 0에서 시작해서 작게 잡힘.

그래서:

```text
m_hat = m / (1 - beta1^t)
```

로 보정함.

t=1일 때:

```text
m_hat = 0.1g / (1 - 0.9)
      = 0.1g / 0.1
      = g
```

v도 같음.

```text
v1 = 0.999*0 + 0.001*g^2 = 0.001g^2
v_hat = 0.001g^2 / (1 - 0.999)
      = g^2
```

이 보정이 없으면 Adam 초반 업데이트가 의도보다 작아질 수 있음.

## 8. Adam 숫자 예시

```text
W = 1.0
grad = 0.2
lr = 0.001
beta1 = 0.9
beta2 = 0.999
t = 1
m = 0
v = 0
```

계산:

```text
m = 0.9*0 + 0.1*0.2 = 0.02
v = 0.999*0 + 0.001*(0.2^2)
  = 0.00004
```

보정:

```text
m_hat = 0.02 / 0.1 = 0.2
v_hat = 0.00004 / 0.001 = 0.04
```

업데이트:

```text
W_new = 1.0 - 0.001 * 0.2 / sqrt(0.04)
      = 1.0 - 0.001 * 0.2 / 0.2
      = 0.999
```

## 9. 현재 코드의 Adam

`src/optimizers.py`:

```python
self.m[key] = beta1 * self.m[key] + (1 - beta1) * grads[key]
self.v[key] = beta2 * self.v[key] + (1 - beta2) * (grads[key] ** 2)
m_hat = self.m[key] / (1 - beta1**self.t)
v_hat = self.v[key] / (1 - beta2**self.t)
params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + eps)
```

이 코드는 파라미터 dict의 모든 항목에 대해 Adam 업데이트를 적용함.

```text
W1, b1, gamma1, beta1, W2, b2, ...
```

전부 같은 규칙으로 업데이트됨.

## 10. 가중치 초기화

처음 W를 어떻게 잡느냐도 중요함.

너무 작으면:

```text
신호가 층을 지나며 작아짐
gradient도 작아짐
학습이 느림
```

너무 크면:

```text
activation이 포화됨
gradient가 폭발하거나 사라짐
```

## 11. Xavier 초기화

Sigmoid/Tanh 계열에서 자주 씀.

```text
W ~ N(0, 1 / fan_in)
```

입력 개수 fan_in에 맞춰 분산을 조절함.

목표:

```text
층을 지나도 값의 분산이 너무 커지거나 작아지지 않게 하기
```

## 12. He 초기화

ReLU 계열에서 자주 씀.

```text
W ~ N(0, 2 / fan_in)
```

ReLU는 음수 절반 정도를 0으로 막기 때문에 Xavier보다 조금 더 큰 분산을 씀.

현재 코드:

```python
np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
```

즉 He 초기화임.

## 13. BatchNorm

BatchNorm은 배치 단위로 값을 정규화함.

```text
mu = mean(x)
var = mean((x - mu)^2)
x_hat = (x - mu) / sqrt(var + eps)
out = gamma * x_hat + beta
```

왜 쓰냐면:

```text
각 층 입력 분포가 너무 흔들리면 학습이 어려움
BatchNorm은 평균과 분산을 맞춰 학습을 안정화함
```

gamma, beta는 다시 표현력을 주기 위한 학습 파라미터임.

```text
gamma:
  정규화된 값을 얼마나 키울지

beta:
  정규화된 값을 얼마나 옮길지
```

## 14. BatchNorm 역전파 직관

BatchNorm의 순전파는 여러 단계임.

```text
x
 -> mean
 -> x - mean
 -> variance
 -> std
 -> normalize
 -> gamma, beta
 -> out
```

역전파는 이걸 거꾸로 따라감.

중요한 gradient:

```text
dbeta = sum(dout)
dgamma = sum(dout * x_hat)
```

왜냐하면:

```text
out = gamma * x_hat + beta
```

이므로:

```text
dout/dgamma = x_hat
dout/dbeta = 1
```

## 15. Dropout

Dropout은 학습 중 일부 뉴런을 랜덤으로 끔.

```text
mask = random > drop_ratio
out = x * mask
```

의미:

```text
특정 뉴런 조합에 과하게 의존하지 않게 함
```

역전파:

```text
dx = dout * mask
```

순전파 때 꺼진 뉴런은 역전파 때도 gradient가 흐르지 않음.

## 16. Weight Decay

Weight decay는 W가 너무 커지지 않게 벌점을 줌.

```text
L_total = L_data + lambda * sum(W^2)
```

미분:

```text
dL_total/dW = dL_data/dW + 2*lambda*W
```

의미:

```text
W가 클수록 줄이는 방향의 gradient가 추가됨
```

overfitting을 줄이는 데 도움됨.

## 17. Hyperparameter

학습되는 값:

```text
W, b, gamma, beta
```

사람이 정하는 값:

```text
learning_rate
batch_size
hidden_size
dropout_ratio
epochs
optimizer 종류
weight_decay 크기
```

이런 사람이 정하는 값을 하이퍼파라미터라고 함.

## 18. 튜닝 기준

학습이 안 내려가면:

```text
learning rate가 너무 작거나 큼
초기화가 나쁨
gradient가 안 흐름
코드 버그
```

train은 좋은데 test가 나쁘면:

```text
overfitting
Dropout, weight decay, data augmentation 고려
모델 크기 조절
```

둘 다 나쁘면:

```text
underfitting
모델 표현력 부족
학습 부족
입력 전처리 문제
```

## 확인 질문

```text
1. Adam은 loss function인가 optimizer인가?
2. Adam에서 m과 v는 각각 무엇을 기억하는가?
3. bias correction은 왜 필요한가?
4. ReLU에서는 왜 He 초기화가 자연스러운가?
5. BatchNorm의 gamma, beta는 왜 필요한가?
6. Dropout은 순전파와 역전파에서 각각 무엇을 하는가?
```

