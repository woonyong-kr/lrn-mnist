# 04. 학습, 손실함수, 수치미분

학습은 단순히 예측하는 것이 아니라, 예측이 틀렸을 때 W와 b를 고치는 과정임.

큰 흐름:

```text
1. 현재 W, b로 예측 y를 만든다.
2. y와 정답 t를 비교해서 loss L을 만든다.
3. L을 줄이려면 W, b를 어느 방향으로 움직여야 하는지 gradient를 구한다.
4. W, b를 조금 업데이트한다.
```

## 1. 데이터에서 학습한다는 것

규칙을 사람이 직접 쓰는 방식:

```text
if 픽셀이 이런 모양이면 3
if 저런 모양이면 8
```

MNIST처럼 숫자 모양이 다양하면 이런 규칙을 직접 쓰기 어려움.

딥러닝은 규칙을 사람이 직접 쓰는 대신:

```text
데이터를 보고 W, b를 찾는다.
```

고 보면 됨.

## 2. 손실함수

손실함수는 모델이 얼마나 틀렸는지 숫자 하나로 만드는 함수임.

```text
L = loss(y, t)
```

여기서:

```text
y: 모델 예측
t: 정답
L: 틀린 정도
```

학습의 목표는:

```text
L을 작게 만드는 W, b를 찾는 것
```

## 3. 왜 정확도만 쓰지 않는가

정확도는 맞으면 1, 틀리면 0임.

문제는 W를 조금 바꿔도 정확도가 잘 변하지 않는다는 것임.

예:

```text
정답은 3

변경 전 확률:
3일 확률 0.20
7일 확률 0.21
예측은 7이라 오답

변경 후 확률:
3일 확률 0.209
7일 확률 0.211
여전히 예측은 7이라 오답
```

정확도는 둘 다 0임.

하지만 실제로는 정답 3의 확률이 올라갔으니 조금 좋아진 상태임.
정확도는 이 작은 개선을 표현하지 못함.

loss는 이런 작은 차이를 연속적인 숫자로 표현함.

그래서 미분해서 학습할 수 있음.

## 4. MSE

평균제곱오차:

```text
L = 1/2 * sum((y_i - t_i)^2)
```

예:

```text
y = [0.1, 0.8, 0.1]
t = [0, 1, 0]
```

계산:

```text
L = 1/2 * ((0.1-0)^2 + (0.8-1)^2 + (0.1-0)^2)
  = 1/2 * (0.01 + 0.04 + 0.01)
  = 0.03
```

MSE는 회귀 문제에서 자연스러움.

```text
예측값과 정답 숫자의 거리 자체가 중요할 때
```

예:

```text
온도 예측
가격 예측
수요량 예측
```

## 5. Cross Entropy

다중 클래스 분류에서는 cross entropy를 많이 씀.

```text
L = -sum(t_i * log(y_i))
```

정답이 0번이면:

```text
t = [1, 0, 0]
```

그래서:

```text
L = -(1*log(y0) + 0*log(y1) + 0*log(y2))
  = -log(y0)
```

즉 정답 클래스 확률만 남음.

정답 확률에 따른 값:

```text
y_true = 0.99 -> loss = 0.010
y_true = 0.90 -> loss = 0.105
y_true = 0.50 -> loss = 0.693
y_true = 0.10 -> loss = 2.303
y_true = 0.01 -> loss = 4.605
```

정답 확률이 낮으면 벌점이 크게 커짐.

## 6. Mini-batch 손실

데이터 하나의 loss만 보고 업데이트하면 너무 흔들릴 수 있음.

그래서 여러 개를 묶어서 평균 loss를 씀.

```text
L_batch = (L1 + L2 + ... + LN) / N
```

코드에서는:

```python
return -np.sum(np.log(correct_probs)) / batch_size
```

처럼 평균을 냄.

batch 평균을 쓰면 gradient도 batch_size로 나뉘는 것이 자연스러움.

## 7. 미분과 Gradient

미분은:

```text
입력을 조금 바꾸면 출력이 얼마나 바뀌는가?
```

학습에서 중요한 질문은:

```text
W를 조금 바꾸면 loss가 얼마나 바뀌는가?
b를 조금 바꾸면 loss가 얼마나 바뀌는가?
```

즉:

```text
dL/dW
dL/db
```

를 구해야 함.

이것이 gradient임.

## 8. 수치미분

수치미분은 직접 조금 바꿔보는 방식임.

```text
dL/dw = [L(w + h) - L(w - h)] / (2h)
```

예:

```text
W1[0,0]만 +h
나머지 W, b는 그대로
순전파해서 loss_plus 계산

W1[0,0]만 -h
나머지 W, b는 그대로
순전파해서 loss_minus 계산

gradient = (loss_plus - loss_minus) / (2h)
```

여기서 중요한 건:

```text
loss 값 0.7을 W에 더하는 게 아님.
h만큼 더하고 빼는 대상은 파라미터 W 하나임.
```

loss는 결과로 나온 숫자고,
수치미분은 그 loss가 파라미터 변화에 얼마나 민감한지 보는 것임.

## 9. 모든 파라미터에 대해 반복

네트워크가:

```text
W1: 784 x 50
b1: 50
W2: 50 x 10
b2: 10
```

이면 파라미터 수는:

```text
784*50 + 50 + 50*10 + 10 = 39760
```

수치미분은 각 파라미터마다 +h, -h 순전파를 해야 함.

```text
39760 * 2 = 79520번 순전파
```

그래서 너무 느림.

하지만 원리를 이해하기에는 좋음.

```text
gradient는 진짜로 loss를 줄이는 방향을 알려준다.
```

를 눈으로 확인할 수 있기 때문임.

## 10. Gradient Descent

gradient가 구해지면 W를 업데이트함.

```text
W = W - lr * dW
b = b - lr * db
```

왜 빼냐면 gradient는 loss가 커지는 방향이기 때문임.

loss를 줄이려면 반대로 가야 함.

예:

```text
현재 W = 1.0
dL/dW = 0.3
lr = 0.1
```

업데이트:

```text
W_new = 1.0 - 0.1 * 0.3
      = 0.97
```

만약 gradient가 음수라면:

```text
현재 W = 1.0
dL/dW = -0.3
lr = 0.1

W_new = 1.0 - 0.1 * (-0.3)
      = 1.03
```

즉 gradient 부호에 따라 W가 올라가거나 내려감.

## 11. 학습률

learning rate는 한 번에 얼마나 움직일지 정함.

너무 크면:

```text
최소점을 지나쳐서 loss가 튈 수 있음
```

너무 작으면:

```text
학습이 너무 느림
```

그래서 학습률은 중요한 하이퍼파라미터임.

## 12. Mini-batch SGD

전체 데이터로 gradient를 구하면 안정적이지만 오래 걸림.

데이터 하나만 쓰면 빠르지만 흔들림이 큼.

mini-batch는 중간 방식임.

```text
전체 데이터 중 일부 batch를 뽑음
그 batch로 loss와 gradient를 계산함
한 번 업데이트함
```

예:

```text
train data: 60000개
batch_size: 128
한 epoch당 약 469번 업데이트
```

## 13. 학습 루프

학습 코드는 보통:

```python
for epoch in range(epochs):
    indices = np.random.permutation(train_size)
    for start in range(0, train_size, batch_size):
        x_batch = x_train[batch_indices]
        y_batch = y_train[batch_indices]

        loss = model.gradient(x_batch, y_batch)
        optimizer.update(model.params, model.grads)
```

흐름:

```text
데이터 섞기
batch 뽑기
순전파
loss 계산
gradient 계산
optimizer 업데이트
```

## 14. Cross Entropy가 MNIST에 자연스러운 이유

MNIST는:

```text
0~9 중 하나를 맞추는 문제
```

모델 출력은:

```text
각 숫자일 확률
```

정답은:

```text
정답 클래스 하나
```

cross entropy는:

```text
정답 클래스 확률이 낮으면 크게 벌줌
정답 클래스 확률이 높으면 작게 벌줌
```

이 구조가 문제와 정확히 맞음.

## 15. 핵심 정리

```text
loss는 현재 모델이 얼마나 틀렸는지 나타내는 숫자다.
학습은 loss를 줄이는 W, b를 찾는 것이다.
gradient는 W, b를 조금 바꾸면 loss가 얼마나 바뀌는지다.
수치미분은 gradient를 직접 흔들어보며 구한다.
gradient descent는 gradient 반대 방향으로 파라미터를 움직인다.
mini-batch는 일부 데이터로 gradient를 추정해 빠르게 학습한다.
```

## 확인 질문

```text
1. loss는 왜 정확도보다 학습에 적합한가?
2. cross entropy에서 one-hot 정답의 0인 위치는 왜 사라지는가?
3. 수치미분에서 h를 더하는 대상은 loss인가, 파라미터인가?
4. gradient가 양수면 W는 어느 방향으로 업데이트되는가?
5. mini-batch를 쓰는 이유는?
```

