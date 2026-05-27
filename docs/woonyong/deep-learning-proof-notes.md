# 딥러닝 연산 증명 노트

이 문서는 `src/learning/calculate.py`에서 숫자로 따라간 내용을 더 깊게 풀어쓴 정리임.

목표는 공식 암기가 아니라, 아래 질문에 답할 수 있게 만드는 것임.

```text
왜 loss를 구하는가?
왜 loss를 미분하는가?
왜 softmax + cross entropy는 y - t가 되는가?
왜 역전파는 수치미분보다 빠른가?
왜 optimizer는 gradient를 받아 W, b를 바꾸는가?
Adam은 손실함수랑 무슨 관계인가?
```

먼저 큰 그림부터 보면:

```text
데이터 x
 -> 현재 W, b로 예측 y를 만듦
 -> 정답 t와 비교해서 loss L을 만듦
 -> L을 줄이려면 W, b를 어느 방향으로 움직여야 하는지 gradient를 구함
 -> optimizer가 gradient를 보고 W, b를 업데이트함
```

여기서 Adam은 loss를 직접 만드는 함수가 아님.

```text
loss function: 틀린 정도를 숫자로 만드는 함수
backpropagation: loss를 각 파라미터로 미분해서 gradient를 만드는 과정
optimizer: gradient를 보고 파라미터를 업데이트하는 규칙
```

즉 Adam은 이런 위치에 있음.

```text
cross entropy loss
 -> backprop으로 dW, db 계산
 -> Adam이 dW, db를 보고 W, b 업데이트
```

Adam은 cross entropy 전용도 아니고, MSE 전용도 아님.
Adam은 그냥 `gradient가 주어졌을 때 어떻게 업데이트할지`를 정하는 방식임.

## 1. 기본 기호

앞으로 계속 나오는 기호는 이렇게 보면 됨.

```text
x: 입력 데이터
W: 가중치
b: 편향
a: affine 결과, activation 전 값
z: activation 결과
y: 예측값 또는 확률
t: 정답 target
L: loss
```

MNIST에서는 보통:

```text
x: 이미지 1개를 펼친 784개 숫자
y: 0~9 각 숫자일 확률 10개
t: 정답 숫자
L: 현재 예측이 얼마나 틀렸는지 나타내는 숫자 1개
```

## 2. 함수와 미분

함수는 입력을 넣으면 출력이 나오는 규칙임.

```text
y = f(x)
```

예를 들어:

```text
f(x) = x^2
```

이면:

```text
x = 3
y = 9
```

미분은 이 질문임.

```text
x를 아주 조금 바꾸면 y는 얼마나 바뀌는가?
```

정의는:

```text
df/dx = lim h->0 (f(x + h) - f(x)) / h
```

예를 들어 `f(x) = x^2`이면:

```text
f(x + h) = (x + h)^2
         = x^2 + 2xh + h^2

f(x + h) - f(x)
= x^2 + 2xh + h^2 - x^2
= 2xh + h^2

(f(x + h) - f(x)) / h
= (2xh + h^2) / h
= 2x + h
```

여기서 h가 0에 가까워지면:

```text
df/dx = 2x
```

그래서 `x = 3`이면:

```text
df/dx = 6
```

의미는:

```text
현재 x=3 근처에서는 x를 0.001 올리면 y가 대략 0.006 오른다.
```

## 3. 편미분

입력이 하나가 아니라 여러 개면:

```text
f(x0, x1) = x0^2 + x1^2
```

이때 편미분은 하나만 움직이고 나머지는 고정하는 것임.

```text
df/dx0: x0만 조금 바꾸고 x1은 고정
df/dx1: x1만 조금 바꾸고 x0은 고정
```

계산하면:

```text
df/dx0 = 2x0
df/dx1 = 2x1
```

예를 들어:

```text
x = [3, 4]
```

이면:

```text
gradient = [6, 8]
```

이 배열이 의미하는 건:

```text
x0 방향으로 움직이면 loss가 얼마나 변하는지
x1 방향으로 움직이면 loss가 얼마나 변하는지
```

를 각각 담은 것임.

신경망에서는 입력 x보다 W, b에 대해 이걸 함.

```text
dL/dW1[0,0]
dL/dW1[0,1]
dL/db1[0]
...
```

## 4. 합성함수와 연쇄법칙

신경망은 함수 하나가 아니라 함수가 이어진 구조임.

```text
x -> h(x) -> g(h(x))
```

즉:

```text
f(x) = g(h(x))
```

이때 x가 f에 영향을 주려면 중간 함수 h를 거쳐야 함.

연쇄법칙은:

```text
df/dx = dg/dh * dh/dx
```

말로 쓰면:

```text
x가 h를 얼마나 바꾸는지
*
h가 f를 얼마나 바꾸는지
=
x가 f를 얼마나 바꾸는지
```

왜 곱하냐면 변화량을 중간값 기준으로 쪼갤 수 있기 때문임.

```text
df/dx
= f 변화량 / x 변화량

중간에 h 변화량을 끼워 넣으면:

= (f 변화량 / h 변화량) * (h 변화량 / x 변화량)
```

즉:

```text
df/dx = df/dh * dh/dx
```

구체적으로:

```text
f(x) = g(h(x))
```

이면:

```text
df/dx
= lim h0->0 [g(h(x+h0)) - g(h(x))] / h0
```

중간에 `h(x+h0) - h(x)`를 끼워 넣으면:

```text
= lim h0->0
  [g(h(x+h0)) - g(h(x))] / [h(x+h0) - h(x)]
  *
  [h(x+h0) - h(x)] / h0
```

앞쪽은:

```text
g'(h(x))
```

뒤쪽은:

```text
h'(x)
```

그래서:

```text
f'(x) = g'(h(x)) * h'(x)
```

이게 역전파의 핵심임.

## 5. 신경망은 큰 합성함수다

단순한 2층 신경망을 쓰면:

```text
a1 = x @ W1 + b1
z1 = sigmoid(a1)
a2 = z1 @ W2 + b2
y = softmax(a2)
L = cross_entropy(y, t)
```

즉 loss L은 사실 이런 함수임.

```text
L = f(x, W1, b1, W2, b2, t)
```

학습에서 바꾸는 건 보통 x가 아니라 W, b임.

그래서 우리가 구하는 건:

```text
dL/dW1
dL/db1
dL/dW2
dL/db2
```

임.

각각 의미는:

```text
W1을 조금 바꾸면 L이 얼마나 바뀌는가?
b1을 조금 바꾸면 L이 얼마나 바뀌는가?
W2를 조금 바꾸면 L이 얼마나 바뀌는가?
b2를 조금 바꾸면 L이 얼마나 바뀌는가?
```

## 6. Affine 계층

Affine은:

```text
out = x @ W + b
```

임.

숫자 하나 기준으로 보면:

```text
out_j = x0*W0j + x1*W1j + ... + b_j
```

예를 들어:

```text
x = [1, 2, 3]

W =
[
  [0.1, 0.2],
  [0.0, 0.1],
  [0.3, -0.2],
]

b = [0, 0]
```

이면:

```text
a0 = 1*0.1 + 2*0.0 + 3*0.3 + 0 = 1.0
a1 = 1*0.2 + 2*0.1 + 3*(-0.2) + 0 = -0.2
```

그래서:

```text
a = [1.0, -0.2]
```

Affine 역전파에서 다음 층에서 `dout`이 왔다고 하자.

```text
dout = dL/dout
```

그러면:

```text
dW = x.T @ dout
db = sum(dout, axis=0)
dx = dout @ W.T
```

왜 `dW = x.T @ dout`인가?

출력 하나를 보면:

```text
out_j = x_i * W_ij + 다른 항들
```

W_ij를 조금 바꾸면 out_j는 x_i만큼 바뀜.

```text
dout_j/dW_ij = x_i
```

그리고 loss는 out_j를 통해 변하니까:

```text
dL/dW_ij = dL/dout_j * dout_j/dW_ij
          = dout_j * x_i
```

그래서:

```text
dW_ij = x_i * dout_j
```

이걸 행렬로 쓰면:

```text
dW = x.T @ dout
```

왜 `db = sum(dout, axis=0)`인가?

b_j는 out_j에 그냥 더해짐.

```text
out_j = ... + b_j
```

그러면:

```text
dout_j/db_j = 1
dL/db_j = dL/dout_j * 1 = dout_j
```

샘플이 하나면:

```text
db = dout
```

mini-batch면 여러 샘플의 dout을 클래스별로 더함.

```text
db = sum(dout, axis=0)
```

왜 `dx = dout @ W.T`인가?

x_i는 여러 출력에 동시에 영향을 줌.

```text
out_0 = x_i*W_i0 + ...
out_1 = x_i*W_i1 + ...
out_2 = x_i*W_i2 + ...
```

그래서 x_i가 loss에 미치는 영향은 모든 출력 방향에서 온 영향의 합임.

```text
dL/dx_i = dL/dout_0 * W_i0
        + dL/dout_1 * W_i1
        + dL/dout_2 * W_i2
        + ...
```

이걸 행렬로 쓰면:

```text
dx = dout @ W.T
```

## 7. Sigmoid와 미분

Sigmoid는:

```text
sigmoid(x) = 1 / (1 + exp(-x))
```

결과를 s라고 두면:

```text
s = sigmoid(x)
```

미분은:

```text
ds/dx = s * (1 - s)
```

직접 유도하면:

```text
s = (1 + exp(-x))^-1
```

연쇄법칙으로:

```text
ds/dx
= -1 * (1 + exp(-x))^-2 * d(1 + exp(-x))/dx
```

안쪽 미분:

```text
d(1 + exp(-x))/dx = -exp(-x)
```

그래서:

```text
ds/dx
= exp(-x) / (1 + exp(-x))^2
```

이걸 sigmoid s로 바꿔보면:

```text
s = 1 / (1 + exp(-x))
1 - s = exp(-x) / (1 + exp(-x))
```

따라서:

```text
s * (1 - s)
= 1 / (1 + exp(-x)) * exp(-x) / (1 + exp(-x))
= exp(-x) / (1 + exp(-x))^2
```

즉:

```text
sigmoid_grad(x) = sigmoid(x) * (1 - sigmoid(x))
```

역전파에서:

```text
a -> sigmoid -> z -> L
```

이면:

```text
dL/da = dL/dz * dz/da
      = dz_gradient * sigmoid_grad(a)
```

그래서 코드에서는:

```python
da = dz * sigmoid_grad(a)
```

처럼 씀.

## 8. ReLU와 미분

ReLU는:

```text
ReLU(x) = max(0, x)
```

즉:

```text
x <= 0이면 0
x > 0이면 x
```

미분은:

```text
x <= 0이면 0
x > 0이면 1
```

왜냐하면:

```text
x > 0 구간에서는 ReLU(x) = x 이므로 기울기 1
x < 0 구간에서는 ReLU(x) = 0 이므로 기울기 0
```

그래서 역전파에서는 forward 때 `x <= 0`이었던 곳을 막음.

```python
dx = dout.copy()
dx[x <= 0] = 0
```

의미는:

```text
순전파 때 꺼져 있던 뉴런은 역전파 gradient도 흐르지 않음.
```

## 9. Softmax

Softmax는 점수를 확률처럼 바꾸는 함수임.

```text
y_i = exp(a_i) / sum(exp(a_k))
```

예를 들어:

```text
a = [2, 1, 5]
```

이면:

```text
y0 = exp(2) / (exp(2) + exp(1) + exp(5))
y1 = exp(1) / (exp(2) + exp(1) + exp(5))
y2 = exp(5) / (exp(2) + exp(1) + exp(5))
```

확률 합은 항상 1임.

```text
y0 + y1 + y2
= [exp(a0) + exp(a1) + exp(a2)] / sum(exp(a))
= 1
```

실제 코드에서는 overflow를 막으려고 최댓값을 빼고 exp를 함.

```text
softmax(a) = softmax(a - max(a))
```

왜 같냐면:

```text
exp(a_i - c) / sum(exp(a_k - c))
= exp(a_i) * exp(-c) / sum(exp(a_k) * exp(-c))
```

분자와 분모에 같은 `exp(-c)`가 있으므로 약분됨.

```text
= exp(a_i) / sum(exp(a_k))
```

그래서 최댓값을 빼도 softmax 결과는 같음.

## 10. Cross Entropy

Cross entropy는 정답 확률이 낮으면 큰 벌을 주는 loss임.

one-hot 정답 기준:

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

정답 확률에 따른 loss는:

```text
y_true = 0.99 -> -log(0.99) = 0.010
y_true = 0.90 -> -log(0.90) = 0.105
y_true = 0.50 -> -log(0.50) = 0.693
y_true = 0.10 -> -log(0.10) = 2.303
y_true = 0.01 -> -log(0.01) = 4.605
```

즉:

```text
정답 확률이 1에 가까우면 loss는 0에 가까움
정답 확률이 0에 가까우면 loss는 무한히 커짐
```

그래서 분류 문제에서 자연스러움.

정답을 맞췄는지 틀렸는지만 보는 게 아니라,
정답이라고 얼마나 확신했는지도 벌점에 반영하기 때문임.

## 11. Softmax + Cross Entropy 미분

이 부분이 가장 중요함.

결론은:

```text
dL/da = y - t
```

여기서:

```text
a: softmax에 들어가기 전 점수, logits
y: softmax(a)
t: one-hot 정답
```

2개 클래스 기준으로 증명하면:

```text
a = [a0, a1]
S = exp(a0) + exp(a1)
y0 = exp(a0) / S
y1 = exp(a1) / S
```

정답이 0번이면:

```text
t = [1, 0]
L = -log(y0)
```

y0를 넣으면:

```text
L = -log(exp(a0) / S)
```

로그 성질:

```text
log(A / B) = log(A) - log(B)
```

그래서:

```text
L = -[log(exp(a0)) - log(S)]
  = -log(exp(a0)) + log(S)
  = -a0 + log(S)
```

즉:

```text
L = -a0 + log(exp(a0) + exp(a1))
```

이제 a0에 대해 미분함.

첫 번째 항:

```text
d(-a0)/da0 = -1
```

직접 보면:

```text
[-(a0 + h) - (-a0)] / h
= (-a0 - h + a0) / h
= -h / h
= -1
```

두 번째 항:

```text
log(exp(a0) + exp(a1))
```

여기서:

```text
S = exp(a0) + exp(a1)
```

라고 두면:

```text
d log(S) / da0
= d log(S) / dS * dS / da0
```

각각:

```text
d log(S) / dS = 1 / S
dS / da0 = exp(a0)
```

그래서:

```text
d log(S) / da0 = exp(a0) / S
```

그런데:

```text
y0 = exp(a0) / S
```

이므로:

```text
dL/da0 = -1 + y0
        = y0 - 1
        = y0 - t0
```

왜 `t0 = 1`이냐면 정답이 0번이기 때문임.

이번에는 a1에 대해 미분하면:

```text
L = -a0 + log(exp(a0) + exp(a1))
```

a1은 `-a0`에는 없음.
그래서 첫 번째 항 변화율은 0임.

두 번째 항만 보면:

```text
d log(S) / da1 = exp(a1) / S = y1
```

정답이 0번이므로:

```text
t1 = 0
```

따라서:

```text
dL/da1 = y1 - 0
        = y1 - t1
```

그래서 전체를 벡터로 쓰면:

```text
dL/da = y - t
```

3개, 10개 클래스도 똑같음.

```text
dL/da_k = y_k - t_k
```

MNIST에서 출력이 10개면:

```text
da = [y0 - t0, y1 - t1, ..., y9 - t9]
```

## 12. MSE

Mean Squared Error는 오차 제곱 평균임.

```text
L = 1/2 * sum((y_i - t_i)^2)
```

앞에 1/2를 붙이는 이유는 미분했을 때 2가 사라져서 계산이 깔끔해지기 때문임.

미분하면:

```text
dL/dy_i = y_i - t_i
```

증명:

```text
L_i = 1/2 * (y_i - t_i)^2
```

u = y_i - t_i라고 두면:

```text
L_i = 1/2 * u^2
dL_i/du = u
du/dy_i = 1
```

그래서:

```text
dL_i/dy_i = u * 1 = y_i - t_i
```

MSE는 회귀 문제에서 자연스러움.

예:

```text
집값 예측
온도 예측
매출 수량 예측
```

같이 정답이 연속적인 숫자일 때 많이 씀.

분류에서도 쓸 수는 있지만, softmax + cross entropy가 보통 더 자연스러움.

이유:

```text
분류는 정답 클래스의 확률을 높이는 문제임
cross entropy는 정답 확률에 직접 벌점을 줌
softmax와 합치면 미분이 y - t로 깔끔해짐
```

## 13. Binary Cross Entropy

이진 분류에서는 출력이 하나일 수 있음.

```text
y = sigmoid(a)
t = 0 또는 1
```

손실은:

```text
L = -[t log(y) + (1 - t) log(1 - y)]
```

정답이 1이면:

```text
L = -log(y)
```

정답이 0이면:

```text
L = -log(1 - y)
```

즉:

```text
정답이 1인데 y가 낮으면 큰 벌점
정답이 0인데 y가 높으면 큰 벌점
```

sigmoid + binary cross entropy도 같이 미분하면 깔끔해짐.

```text
dL/da = y - t
```

softmax + cross entropy와 비슷한 모양임.

## 14. 손실함수 선택 기준

손실함수는 문제 유형과 출력 해석에 맞춰 고름.

```text
다중 클래스 분류:
  softmax + cross entropy

이진 분류:
  sigmoid + binary cross entropy

회귀:
  MSE 또는 MAE

이상치에 덜 민감한 회귀:
  MAE 또는 Huber loss
```

중요한 기준은:

```text
모델 출력 y가 어떤 의미인가?
정답 t가 어떤 형태인가?
틀린 것을 어떻게 벌줄 것인가?
미분했을 때 학습 신호가 잘 나오는가?
```

MNIST는 0~9 중 하나를 고르는 문제임.

그래서:

```text
출력: 클래스 10개의 점수
softmax: 점수를 확률 10개로 바꿈
cross entropy: 정답 확률이 낮으면 큰 벌점
```

이 조합이 자연스러움.

## 15. 수치미분

수치미분은 정의 그대로 계산하는 방식임.

```text
dL/dw = [L(w + h) - L(w - h)] / (2h)
```

여기서 중요한 점:

```text
w 하나만 바꾸고 나머지 W, b는 모두 고정함
```

예:

```text
W1[0,0]만 +h
순전파로 loss_plus 계산

W1[0,0]만 -h
순전파로 loss_minus 계산

gradient = (loss_plus - loss_minus) / (2h)
```

이걸 모든 파라미터에 반복하면 전체 gradient를 구할 수 있음.

하지만 파라미터가 많으면 너무 느림.

예:

```text
W1: 784 x 50 = 39200
b1: 50
W2: 50 x 10 = 500
b2: 10

총 39760개
```

수치미분은 파라미터 하나마다 +h, -h 순전파가 필요하므로:

```text
39760 * 2 = 79520번 순전파
```

가 필요함.

그래서 원리 확인용으로는 좋지만 실제 학습에는 느림.

## 16. 역전파

역전파는 수치미분처럼 하나씩 흔들어보지 않음.

대신 순전파 때 저장한 중간값을 이용해서 뒤에서 앞으로 gradient를 계산함.

```text
순전파:
x -> affine -> activation -> affine -> softmax -> loss

역전파:
loss -> softmax/cross entropy -> affine -> activation -> affine
```

핵심은 연쇄법칙임.

예를 들어:

```text
a1 -> sigmoid -> z1 -> affine -> a2 -> loss
```

뒤에서:

```text
dL/dz1
```

을 알게 되면, sigmoid 이전으로 넘길 때:

```text
dL/da1 = dL/dz1 * dz1/da1
        = dz1_gradient * sigmoid_grad(a1)
```

이렇게 바꿈.

그래서 코드에서는:

```python
da1 = dz1 * sigmoid_grad(a1)
```

처럼 계산함.

역전파는 순전파 1번, 역전파 1번으로 모든 파라미터의 gradient를 구할 수 있음.

## 17. Gradient Descent

gradient는 loss가 커지는 방향을 가리킴.

그래서 loss를 줄이려면 반대 방향으로 움직여야 함.

```text
W = W - learning_rate * dW
b = b - learning_rate * db
```

여기서:

```text
learning_rate: 한 번에 얼마나 움직일지 정하는 값
```

너무 크면:

```text
loss 최소점을 지나쳐서 튈 수 있음
```

너무 작으면:

```text
학습이 너무 느림
```

## 18. SGD

SGD는 가장 단순한 optimizer임.

```text
W = W - lr * dW
```

장점:

```text
단순함
구현 쉬움
기본 원리를 이해하기 좋음
```

단점:

```text
지형이 울퉁불퉁하면 흔들림
방향마다 gradient 크기가 다르면 비효율적
학습률 선택에 민감함
```

## 19. Momentum

Momentum은 이전 이동 방향을 기억함.

```text
v = momentum * v - lr * grad
W = W + v
```

의미는:

```text
이전에도 계속 가던 방향이면 더 밀고 감
방향이 자주 바뀌는 축은 흔들림이 줄어듦
```

물리적으로 보면:

```text
공이 경사면을 굴러 내려가는 느낌
```

SGD는 매번 현재 gradient만 보고 움직이지만,
Momentum은 이전 속도도 같이 봄.

## 20. AdaGrad

AdaGrad는 파라미터마다 학습률을 다르게 조절함.

```text
h = h + grad * grad
W = W - lr * grad / (sqrt(h) + eps)
```

의미:

```text
gradient가 자주 크게 나온 파라미터는 점점 작게 움직임
gradient가 작게 나온 파라미터는 상대적으로 크게 움직임
```

장점:

```text
파라미터별로 자동 조절됨
희소한 feature에 유리함
```

단점:

```text
h가 계속 누적되므로 시간이 지나면 업데이트가 너무 작아질 수 있음
```

## 21. RMSProp

RMSProp은 AdaGrad의 누적 문제를 줄이기 위해 이동평균을 씀.

```text
v = beta * v + (1 - beta) * grad^2
W = W - lr * grad / (sqrt(v) + eps)
```

AdaGrad는 과거 gradient 제곱을 전부 누적하지만,
RMSProp은 최근 gradient 제곱을 더 중요하게 봄.

```text
오래된 gradient 영향은 점점 작아짐
최근 gradient 영향은 크게 반영됨
```

## 22. Adam

Adam은 Momentum과 RMSProp을 합친 느낌임.

Adam은 두 가지를 기억함.

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

각각 의미는:

```text
m:
  gradient 방향의 평균
  Momentum 역할

v:
  gradient 크기 제곱의 평균
  파라미터별 학습률 조절 역할

m_hat, v_hat:
  초반에 0으로 시작해서 작게 잡히는 문제를 보정

eps:
  0으로 나누는 문제 방지
```

## 23. Adam의 bias correction이 필요한 이유

Adam은 처음에:

```text
m = 0
v = 0
```

에서 시작함.

첫 번째 gradient가 g라고 해보자.

```text
m1 = beta1 * 0 + (1 - beta1) * g
```

beta1이 0.9면:

```text
m1 = 0.1g
```

실제 gradient는 g인데, m1은 0에서 시작했기 때문에 너무 작음.

그래서:

```text
m_hat = m / (1 - beta1^t)
```

로 보정함.

t=1이면:

```text
m_hat1 = 0.1g / (1 - 0.9^1)
       = 0.1g / 0.1
       = g
```

v도 마찬가지임.

beta2가 0.999이면:

```text
v1 = 0.001g^2
```

너무 작으니까:

```text
v_hat1 = v1 / (1 - 0.999^1)
       = 0.001g^2 / 0.001
       = g^2
```

으로 보정함.

즉 bias correction은 초반에 0에서 시작해서 평균이 작게 잡히는 문제를 보정하는 것임.

## 24. Adam 숫자 예시

파라미터 하나만 있다고 해보자.

```text
W = 1.0
grad = 0.2
lr = 0.001
beta1 = 0.9
beta2 = 0.999
eps = 1e-8
t = 1
```

처음에는:

```text
m = 0
v = 0
```

업데이트:

```text
m = 0.9*0 + 0.1*0.2 = 0.02
v = 0.999*0 + 0.001*(0.2^2)
  = 0.001*0.04
  = 0.00004
```

보정:

```text
m_hat = 0.02 / (1 - 0.9^1)
      = 0.02 / 0.1
      = 0.2

v_hat = 0.00004 / (1 - 0.999^1)
      = 0.00004 / 0.001
      = 0.04
```

업데이트 크기:

```text
lr * m_hat / (sqrt(v_hat) + eps)
= 0.001 * 0.2 / sqrt(0.04)
= 0.001 * 0.2 / 0.2
= 0.001
```

그래서:

```text
W_new = 1.0 - 0.001 = 0.999
```

여기서 중요한 건:

```text
Adam은 grad의 방향과 크기를 그대로 쓰지 않고,
이동평균과 제곱 이동평균으로 조절해서 업데이트함.
```

## 25. Adam과 손실함수의 관계

Adam에 필요한 것은 손실함수 자체가 아니라 gradient임.

흐름은:

```text
loss function이 L을 만듦
backprop이 dL/dW, dL/db를 만듦
Adam이 dL/dW, dL/db를 받아 W, b를 업데이트함
```

즉:

```text
MSE + Adam 가능
Cross Entropy + Adam 가능
Binary Cross Entropy + Adam 가능
```

Adam이 요구하는 조건은 대략:

```text
loss가 미분 가능하거나, 거의 모든 구간에서 gradient를 정의할 수 있어야 함
gradient가 너무 폭발하거나 완전히 사라지지 않도록 관리해야 함
```

그래서 Adam과 같이 알아야 할 손실함수는 하나가 아니라,
문제 유형에 맞는 손실함수임.

MNIST라면:

```text
softmax + cross entropy
```

가 자연스럽고,
그 결과 gradient는:

```text
dL/dlogits = y - t
```

가 됨.

그 다음 이 gradient가 역전파를 통해:

```text
dW2, db2, dW1, db1
```

이 되고, Adam은 이것들을 받아 업데이트함.

## 26. BatchNorm

BatchNorm은 배치 안에서 activation 값을 정규화함.

```text
mu = mean(x)
var = mean((x - mu)^2)
x_hat = (x - mu) / sqrt(var + eps)
y = gamma * x_hat + beta
```

의미:

```text
값의 평균을 0 근처로 맞추고
분산을 1 근처로 맞춘 뒤
gamma, beta로 다시 조절 가능하게 함
```

왜 쓰냐면:

```text
각 층의 입력 분포가 너무 흔들리면 학습이 어려움
정규화하면 더 안정적으로 학습됨
```

여기서 gamma, beta도 학습되는 파라미터임.

```text
gamma: 스케일
beta: 이동
```

역전파에서는:

```text
dgamma = sum(dout * x_hat)
dbeta = sum(dout)
dx = 정규화 과정 전체를 거꾸로 미분한 값
```

BatchNorm은 수식이 길지만 핵심은 이거임.

```text
x -> mean/var -> normalize -> scale/shift -> loss
```

이 흐름을 연쇄법칙으로 거꾸로 따라간다.

## 27. Dropout

Dropout은 학습 중 일부 뉴런을 랜덤하게 꺼버림.

```text
train:
  mask = random > dropout_ratio
  out = x * mask / keep_ratio

test:
  out = x
```

의미:

```text
특정 뉴런 조합에 너무 의존하지 않게 만듦
```

역전파에서는 forward 때 꺼진 뉴런에는 gradient도 흐르지 않음.

```text
dx = dout * mask / keep_ratio
```

Dropout은 overfitting을 줄이는 regularization 방법임.

## 28. Weight Initialization

가중치 초기화가 중요한 이유:

```text
W가 너무 크면 activation이 포화됨
W가 너무 작으면 신호가 사라짐
층이 깊어질수록 gradient가 죽거나 폭발함
```

Sigmoid/Tanh 계열에서는 Xavier 초기화를 많이 씀.

```text
W ~ N(0, 1 / fan_in)
```

ReLU 계열에서는 He 초기화를 많이 씀.

```text
W ~ N(0, 2 / fan_in)
```

현재 코드에서는:

```python
np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
```

를 쓰므로 He 초기화임.

ReLU를 쓰기 때문에 자연스러운 선택임.

## 29. Overfitting과 Regularization

Overfitting은 학습 데이터는 잘 맞추는데 새로운 데이터는 못 맞추는 상태임.

예:

```text
train accuracy는 높음
test accuracy는 낮음
```

줄이는 방법:

```text
더 많은 데이터
Dropout
Weight decay
BatchNorm
Data augmentation
모델 크기 조절
early stopping
```

Weight decay는 loss에 W 크기에 대한 벌점을 추가함.

```text
L_total = L_data + lambda * sum(W^2)
```

이렇게 하면 W가 너무 커지는 것을 막음.

미분하면:

```text
dL_total/dW = dL_data/dW + 2 * lambda * W
```

즉 W가 커질수록 줄이는 방향의 gradient가 추가됨.

## 30. CNN 핵심

MNIST 같은 이미지는 픽셀 위치 관계가 중요함.

완전연결층은 이미지를 784개 숫자로 펼쳐버리기 때문에 공간 구조를 직접 보존하지 않음.

CNN은 필터를 이미지 위로 움직이며 지역 패턴을 찾음.

```text
Convolution:
  작은 필터가 이미지 일부 영역과 곱해져 feature map을 만듦

Pooling:
  영역 안의 대표값을 뽑아 크기를 줄임
```

Convolution도 결국 affine과 비슷함.

```text
출력 = 입력 일부 영역과 필터의 곱의 합 + bias
```

역전파도 원리는 같음.

```text
필터가 출력에 얼마나 영향을 줬는지 계산해서 dW를 구함
입력 각 위치가 여러 필터 위치에 기여했으면 그 gradient를 합침
```

즉 CNN도 특별한 마법이 아니라:

```text
순전파 계산
loss 계산
연쇄법칙으로 역전파
optimizer로 업데이트
```

흐름은 동일함.

## 31. 현재 코드 흐름

현재 본소스 기준으로 학습 흐름은:

```text
src/training.py
  train()
```

안에서:

```python
loss = model.gradient(x_batch, y_batch)
optimizer.update(model.params, model.grads)
```

가 실행됨.

`model.gradient()`는:

```python
y_pred = self.forward(x, train=True)
loss = cross_entropy_loss(y_pred, y)
dout = cross_entropy_gradient(y_pred, y)
self.backward(dout)
return loss
```

즉:

```text
순전파
 -> loss 계산
 -> softmax + cross entropy 미분값 계산
 -> backward
 -> grads 채움
```

그 다음 optimizer가:

```text
params[key] -= update_amount
```

로 W, b를 바꿈.

SGD면:

```text
update_amount = lr * grad
```

Adam이면:

```text
update_amount = lr * m_hat / (sqrt(v_hat) + eps)
```

임.

## 32. 전체를 한 문장으로 정리

딥러닝 학습은 결국 이거임.

```text
현재 파라미터로 예측한다.
정답과 비교해서 loss를 만든다.
loss를 줄이는 방향을 미분으로 구한다.
그 방향을 optimizer 규칙에 따라 파라미터에 반영한다.
이 과정을 반복한다.
```

수치미분은 이 방향을 하나씩 흔들어보며 구하는 방식이고,
역전파는 연쇄법칙으로 한 번에 구하는 방식임.

Adam은 방향을 구하는 함수가 아니라,
이미 구해진 gradient를 조금 더 안정적으로 반영하는 업데이트 규칙임.

