# MNIST 학습 흐름 정리

이 문서는 MNIST 숫자 분류를 기준으로 순전파, loss, softmax, cross entropy,
수치미분, 역전파가 각각 어디에서 쓰이는지 다시 정리한 내용임.

핵심은 이거임.

```text
입력 x
 -> 가중치 W, b로 점수 계산
 -> softmax로 확률 계산
 -> cross entropy로 loss 계산
 -> loss를 줄이는 방향의 gradient 계산
 -> W, b 업데이트
```

학습은 이 과정을 여러 번 반복하는 것임.

```text
순전파 -> loss 계산 -> 역전파로 gradient 계산 -> W, b 업데이트
순전파 -> loss 계산 -> 역전파로 gradient 계산 -> W, b 업데이트
...
```

## 1. 순전파

순전파는 현재 가중치로 예측을 만드는 과정임.

MNIST에서는 이미지 하나가 28x28이라서 펼치면 784개 숫자가 됨.

```text
x = 1x784
```

신경망이 아래처럼 생겼다고 보면:

```text
x -> W1 -> activation -> W2 -> softmax -> loss
```

계산은 보통 이렇게 흘러감.

```python
a1 = x @ W1 + b1
z1 = sigmoid(a1)  # 또는 ReLU
a2 = z1 @ W2 + b2
y = softmax(a2)
loss = cross_entropy(y, t)
```

여기서 중요한 이름은 이거임.

```text
a1, a2: 가중치 계산으로 나온 점수
z1: activation을 통과한 값
y: softmax 결과, 즉 확률
t: 정답
loss: 얼마나 틀렸는지를 나타내는 숫자 하나
```

`logits`는 보통 softmax에 들어가기 전 점수를 말함.

```text
logits = a2
```

즉 logits는 정답도 아니고 확률도 아님.
그냥 현재 W, b로 계산한 클래스별 점수임.

## 2. softmax

softmax는 점수를 확률처럼 바꿔주는 함수임.

예를 들어:

```text
logits = [2, 1, 5]
```

이면 5가 가장 크니까 세 번째 클래스 확률이 가장 커짐.

공식은:

```text
y_i = exp(a_i) / sum(exp(a))
```

예:

```text
y0 = exp(a0) / (exp(a0) + exp(a1) + exp(a2))
y1 = exp(a1) / (exp(a0) + exp(a1) + exp(a2))
y2 = exp(a2) / (exp(a0) + exp(a1) + exp(a2))
```

여기서 `exp(a)`는 `e^a`임.

```text
exp(2) = e^2
```

그리고 `log`는 보통 자연로그임.

```text
log(x) = log_e(x)
```

그래서:

```text
log(e^2) = 2
```

이 말은 `e를 몇 제곱해야 e^2가 되나?`를 묻는 것과 같음.
답은 당연히 2임.

## 3. cross entropy

cross entropy는 정답 확률이 낮을수록 큰 벌점을 줌.

공식은:

```text
L = -sum(t_i * log(y_i))
```

정답이 0번 클래스라면 one-hot 정답은:

```text
t = [1, 0, 0]
```

예측 확률이:

```text
y = [0.047, 0.017, 0.936]
```

이면:

```text
L = -(1 * log(0.047) + 0 * log(0.017) + 0 * log(0.936))
L = -log(0.047)
L = 3.05 정도
```

나머지는 정답이 아니니까 `0 * log(...)`가 돼서 사라짐.

그래서 cross entropy는 결국 정답 클래스의 확률만 보고 벌점을 주는 느낌임.

```text
정답 확률 0.10 -> loss 약 2.30
정답 확률 0.50 -> loss 약 0.69
정답 확률 0.90 -> loss 약 0.10
정답 확률 0.99 -> loss 약 0.01
```

정답 확률이 높아질수록 loss는 0에 가까워짐.

## 4. loss는 새 가중치가 아님

여기서 자주 헷갈리는 부분이 있음.

```text
loss는 새 W가 아님.
loss는 현재 W, b가 얼마나 틀렸는지 나타내는 점수 하나임.
```

가중치를 바꾸려면 loss 자체를 W로 쓰는 게 아니라,
loss가 W에 대해 얼마나 변하는지를 구해야 함.

그게 gradient임.

```text
dL/dW
dL/db
```

의미는:

```text
W를 조금 바꾸면 loss가 얼마나 바뀌는가?
b를 조금 바꾸면 loss가 얼마나 바뀌는가?
```

임.

## 5. 수치미분

수치미분은 진짜로 하나씩 살짝 바꿔보는 방식임.

예를 들어 W1의 특정 원소 하나를 보고 싶다면:

```text
W1[0, 0]을 h만큼 올려서 loss를 구함
W1[0, 0]을 h만큼 내려서 loss를 구함
두 loss 차이로 기울기를 계산함
```

공식은:

```text
gradient = (f(x + h) - f(x - h)) / (2h)
```

여기서 `x`는 입력 데이터가 아니라, 편미분하려는 파라미터 하나라고 보면 됨.

즉:

```text
f(W1[0,0] + h, 나머지 W와 b는 그대로)
f(W1[0,0] - h, 나머지 W와 b는 그대로)
```

이렇게 계산하는 것임.

만약 파라미터가 39760개면 이걸 거의 39760번 이상 반복해야 함.
그래서 엄청 느림.

중요한 점:

```text
수치미분도 모든 W, b의 gradient를 구할 수 있음.
다만 하나씩 바꿔보니까 느릴 뿐임.
```

## 6. 역전파

역전파는 수치미분처럼 하나씩 바꿔보지 않음.

대신 순전파 때 계산한 값들을 가지고,
뒤에서 앞으로 미분값을 전달함.

```text
순전파:
x -> W1 -> sigmoid -> W2 -> softmax -> loss

역전파:
loss -> softmax/cross entropy -> W2 -> sigmoid -> W1
```

수치미분은 파라미터 하나마다 loss를 다시 계산함.
역전파는 한 번의 순전파 결과를 이용해서 전체 gradient를 한 번에 구함.

그래서 빠름.

## 7. softmax + cross entropy의 핵심

softmax와 cross entropy를 같이 쓰면 출력층 미분이 깔끔해짐.

```text
da2 = y - t
```

여기서:

```text
y: softmax 결과
t: 정답 one-hot
da2: softmax 이전 점수 a2에 대한 loss의 변화량
```

예:

```text
y = [0.047, 0.017, 0.936]
t = [1, 0, 0]
```

이면:

```text
da2 = y - t
da2 = [0.047, 0.017, 0.936] - [1, 0, 0]
da2 = [-0.953, 0.017, 0.936]
```

의미는:

```text
0번 정답 점수는 올려야 함
1번 오답 점수는 조금 내려야 함
2번 오답 점수는 많이 내려야 함
```

왜냐하면 2번을 0.936으로 너무 강하게 믿고 있기 때문임.

## 8. 왜 y - t가 미분인가

2개 클래스만 있다고 가정해보자.

```text
a = [a0, a1]
y0 = exp(a0) / (exp(a0) + exp(a1))
y1 = exp(a1) / (exp(a0) + exp(a1))
```

정답이 0번이면:

```text
t = [1, 0]
L = -log(y0)
```

이걸 풀면:

```text
L = -log(exp(a0) / (exp(a0) + exp(a1)))
```

로그 성질 때문에:

```text
log(A / B) = log(A) - log(B)
```

그래서:

```text
L = -log(exp(a0)) + log(exp(a0) + exp(a1))
```

그리고:

```text
log(exp(a0)) = a0
```

이므로:

```text
L = -a0 + log(exp(a0) + exp(a1))
```

이제 `a0`를 조금 바꾸면 loss가 얼마나 바뀌는지 봄.

첫 번째 부분:

```text
-a0
```

여기서 a0를 h만큼 바꾸면:

```text
-(a0 + h) - (-a0)
= -a0 - h + a0
= -h
```

h로 나누면:

```text
-h / h = -1
```

그래서 `-a0` 쪽 변화율은 `-1`임.

두 번째 부분:

```text
log(exp(a0) + exp(a1))
```

여기서는 a0가 `exp(a0)` 안에 들어 있음.

```text
S = exp(a0) + exp(a1)
```

라고 잠깐 보면:

```text
log(S)
```

형태임.

a0가 조금 커지면:

```text
exp(a0)가 커짐
S가 커짐
log(S)도 커짐
```

이 변화율은:

```text
exp(a0) / S
```

임.

그런데 softmax에서:

```text
y0 = exp(a0) / S
```

였음.

그래서 전체는:

```text
dL/da0 = -1 + y0
dL/da0 = y0 - 1
dL/da0 = y0 - t0
```

정답이 0번이라서 `t0 = 1`이기 때문임.

정답이 아닌 1번에 대해서는 `t1 = 0`이라서:

```text
dL/da1 = y1 - 0
dL/da1 = y1 - t1
```

그래서 전체 출력층은:

```text
da = y - t
```

가 됨.

## 9. 연쇄법칙

역전파의 핵심은 연쇄법칙임.

합성함수가 있을 때:

```text
f(x) = g(h(x))
```

흐름은:

```text
x -> h(x) -> g(h(x))
```

이때 x가 f에 영향을 주는 길은 중간에 h를 거쳐감.

그래서:

```text
df/dx = dg/dh * dh/dx
```

말로 쓰면:

```text
x가 h를 얼마나 바꾸는지
그리고 h가 f를 얼마나 바꾸는지
둘을 곱하면 x가 f를 얼마나 바꾸는지 알 수 있음
```

예를 들어:

```text
x -> a1 -> sigmoid -> z1 -> W2 -> loss
```

이면, 뒤에서 온 오차를 앞쪽으로 넘길 때 중간에 있던 함수들의 변화율을 계속 곱함.

그래서 sigmoid를 지나서 온 오차를 sigmoid 이전으로 넘기려면:

```text
da1 = dz1 * sigmoid_grad(a1)
```

가 됨.

## 10. sigmoid_grad가 뭔가

순전파에서:

```text
z1 = sigmoid(a1)
```

였음.

sigmoid는 값을 0과 1 사이로 눌러주는 함수임.

```text
sigmoid(a) = 1 / (1 + exp(-a))
```

sigmoid_grad는 sigmoid의 기울기임.

즉:

```text
a1을 조금 바꾸면 z1이 얼마나 바뀌는가?
```

를 말함.

sigmoid 결과를 s라고 두면:

```text
s = sigmoid(a)
```

sigmoid 미분은:

```text
sigmoid_grad(a) = s * (1 - s)
```

주의할 점:

```text
sigmoid(a1)는 z1임.
sigmoid_grad(a1)는 z1이 아니라, sigmoid의 변화율임.
```

예:

```text
a1 = 3
z1 = sigmoid(3) = 0.9526
sigmoid_grad(3) = 0.9526 * (1 - 0.9526)
                = 0.0452
```

즉 sigmoid 값은 0.9526이지만,
그 지점에서의 변화율은 0.0452임.

그래서 뒤에서 온 오차가:

```text
dz1 = -0.2550
```

이면 sigmoid 이전의 오차는:

```text
da1 = dz1 * sigmoid_grad(a1)
da1 = -0.2550 * 0.0452
```

가 됨.

## 11. W2의 gradient

출력층 근처를 보면:

```text
z1 -> W2 -> a2 -> softmax -> loss
```

여기서 이미:

```text
da2 = y - t
```

를 구했음.

W2를 얼마나 바꿔야 하는지는:

```text
dW2 = z1.T @ da2
```

로 구함.

의미는:

```text
dW2[i, j] = z1[i] * da2[j]
```

즉:

```text
은닉층 i번 뉴런이 많이 켜져 있었고
출력 j번 오차가 크면
그 둘을 연결하는 W2[i, j]를 많이 고침
```

이게 꽤 중요한 직관임.

가중치는 두 뉴런 사이의 연결임.
그래서 앞 뉴런이 얼마나 켜졌는지와 뒤쪽 오차가 얼마나 큰지를 같이 봄.

## 12. b의 gradient

b는 입력과 곱해지는 값이 아님.
그냥 점수에 더해지는 값임.

```text
a2 = z1 @ W2 + b2
```

출력 클래스가 2개면:

```text
a2[0] = ... + b2[0]
a2[1] = ... + b2[1]
```

그래서 b2는 출력 오차를 그대로 받음.

입력 x를 하나만 넣었다면:

```text
db2 = da2
```

입력 여러 개를 한 번에 넣는 mini-batch라면 da2가 여러 줄이 됨.

예:

```text
da2 =
[
  [-0.5,  0.5],
  [-0.2,  0.2],
  [ 0.1, -0.1],
]
```

이때 b2[0]에 대한 오차는 0번 클래스 방향으로 모아야 하고,
b2[1]에 대한 오차는 1번 클래스 방향으로 모아야 함.

그래서:

```text
db2 = sum(da2, axis=0)
db2 = [-0.5 + -0.2 + 0.1, 0.5 + 0.2 + -0.1]
db2 = [-0.6, 0.6]
```

즉 전체를 숫자 하나로 합치는 게 아니라,
클래스별로 합치는 것임.

## 13. 전체 역전파 흐름

순전파:

```python
a1 = x @ W1 + b1
z1 = sigmoid(a1)
a2 = z1 @ W2 + b2
y = softmax(a2)
loss = cross_entropy(y, t)
```

역전파:

```python
da2 = y - t

dW2 = z1.T @ da2
db2 = sum(da2, axis=0)

dz1 = da2 @ W2.T
da1 = dz1 * sigmoid_grad(a1)

dW1 = x.T @ da1
db1 = sum(da1, axis=0)
```

업데이트:

```python
W1 = W1 - learning_rate * dW1
b1 = b1 - learning_rate * db1
W2 = W2 - learning_rate * dW2
b2 = b2 - learning_rate * db2
```

여기서 빼는 이유는 gradient가 loss가 커지는 방향을 가리키기 때문임.
우리는 loss를 줄이고 싶으니까 반대 방향으로 감.

## 14. 수치미분과 역전파 차이

둘 다 목적은 같음.

```text
각 W, b를 얼마나 바꿔야 loss가 줄어드는지 구하는 것
```

차이는 gradient를 구하는 방식임.

수치미분:

```text
파라미터 하나를 h만큼 바꿔봄
loss를 다시 계산함
이걸 모든 파라미터에 반복함
느림
```

역전파:

```text
순전파 값을 저장해둠
loss에서 시작해서 뒤에서 앞으로 미분값을 전달함
한 번에 전체 gradient를 구함
빠름
```

그래서 4장에서는 수치미분으로 원리를 확인하고,
5장에서는 역전파로 실제 학습이 가능한 속도를 만드는 것임.

## 15. 현재 코드 기준으로 보면

현재 본소스는 이런 흐름을 가짐.

```text
src/network.py
  NeuralNetwork.forward()
  NeuralNetwork.backward()
  NeuralNetwork.gradient()

src/losses.py
  cross_entropy_loss()
  cross_entropy_gradient()

src/training.py
  train()
  evaluate()
```

학습할 때는 `training.train()`에서:

```python
loss = model.gradient(x_batch, y_batch)
optimizer.update(model.params, model.grads)
```

처럼 동작함.

`model.gradient()` 안에서:

```python
y_pred = self.forward(x, train=True)
loss = cross_entropy_loss(y_pred, y)
dout = cross_entropy_gradient(y_pred, y)
self.backward(dout)
return loss
```

이 순서로 진행됨.

즉 지금 본소스 기준으로는:

```text
순전파 -> cross entropy loss -> y - t -> backward -> optimizer update
```

방식임.

## 16. mnist_lab.ipynb 실행 상태

`mnist_lab.ipynb`는 전체 실습 흐름을 실행하는 노트북임.

노트북 흐름:

```text
1. src 경로 추가
2. MNIST 데이터 로드
3. 테스트 실행
4. 모델 생성
5. 학습
6. 평가
7. loss 그래프 출력
```

로컬 Python 코드 기준으로는 실행 가능함.

확인된 것:

```text
data/mnist.npz 있음
src import 정상
load_mnist() 정상
짧은 학습 정상
테스트 일부 정상 통과
```

다만 노트북에서 바로 커널로 실행하려면 `ipykernel`이 필요함.

```bash
conda install -n mnist-nn -c conda-forge ipykernel
```

처음 확인할 때는 20 epoch 전체 학습보다,
노트북의 학습 셀에서 `epochs=1` 정도로 줄여서 먼저 확인하는 게 좋음.

```python
loss_history = train(model, optimizer, x_train, y_train, epochs=1, batch_size=128)
```

정상 동작을 확인한 뒤 `epochs=20`으로 올리면 됨.

## 17. 한 줄로 다시 정리

MNIST 학습은 결국 이거임.

```text
현재 W, b로 예측한다.
정답과 비교해서 loss를 구한다.
loss를 줄이려면 W, b를 어느 방향으로 바꿔야 하는지 gradient를 구한다.
W, b를 조금 바꾼다.
이걸 반복한다.
```

수치미분은 이 gradient를 하나씩 직접 바꿔보며 구하는 방식이고,
역전파는 연쇄법칙으로 뒤에서 앞으로 한 번에 구하는 방식임.

