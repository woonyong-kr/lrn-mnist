# 09. 현재 MNIST 코드와 개념 연결

이 문서는 지금 repo의 파일들이 책에서 배운 개념과 어떻게 연결되는지 정리한 것임.

## 1. 전체 파일 구조

중요한 파일:

```text
src/data.py
  MNIST 데이터 로드

src/activations.py
  ReLU, Softmax

src/layers.py
  Affine, BatchNorm, Dropout

src/losses.py
  cross entropy loss, softmax+cross entropy gradient

src/optimizers.py
  SGD, Adam

src/network.py
  전체 신경망 조립

src/training.py
  학습 루프, 평가, loss 그래프

mnist_lab.ipynb
  노트북 실행 흐름

src/learning/calculate.py
  역전파 숫자 예제

src/learning/mini_batch.py
  독립 실행 가능한 작은 MNIST 학습 예제
```

## 2. 데이터 로드

`src/data.py`:

```python
with np.load(local_path) as data:
    x_train = data["x_train"].astype(np.float32).reshape(-1, 784) / 255.0
    x_test = data["x_test"].astype(np.float32).reshape(-1, 784) / 255.0
    y_train = data["y_train"]
    y_test = data["y_test"]
```

의미:

```text
28 x 28 이미지를 784개 숫자로 펼침
0~255 픽셀 값을 0~1 범위로 정규화
레이블은 0~9 정수 그대로 사용
```

shape:

```text
x_train: 60000 x 784
y_train: 60000
x_test: 10000 x 784
y_test: 10000
```

## 3. ReLU

`src/activations.py`:

```python
self.mask = x <= 0
out = x.copy()
out[self.mask] = 0
```

순전파:

```text
음수/0은 0
양수는 그대로
```

역전파:

```python
dx = dout.copy()
dx[self.mask] = 0
```

의미:

```text
순전파 때 꺼졌던 위치는 gradient도 막음
```

## 4. Softmax

`src/activations.py`:

```python
shifted = x - np.max(x, axis=1, keepdims=True)
exp_x = np.exp(shifted)
self.out = exp_x / np.sum(exp_x, axis=1, keepdims=True)
```

의미:

```text
logits를 확률로 바꿈
최댓값을 빼서 overflow를 줄임
```

backward:

```python
return dout
```

이유:

```text
softmax + cross entropy 미분값을 losses.py에서 이미 y - t로 만들기 때문
```

## 5. Affine

`src/layers.py`:

```python
return self.x @ self.W + self.b
```

역전파:

```python
self.dW = self.x.T @ dout
self.db = np.sum(dout, axis=0)
dx = dout @ self.W.T
```

개념 연결:

```text
dW = 앞 값.T @ 뒤 오차
db = 뒤 오차를 batch 방향으로 합침
dx = 뒤 오차를 이전 입력 방향으로 넘김
```

## 6. BatchNorm

순전파:

```python
mu = np.mean(x, axis=0)
var = np.var(x, axis=0)
self.x_centered = x - mu
self.std = np.sqrt(var + self.eps)
self.x_norm = self.x_centered / self.std
out = self.gamma * self.x_norm + self.beta
```

의미:

```text
batch 기준으로 평균 0, 분산 1에 가깝게 맞춘 뒤
gamma, beta로 다시 조절함
```

역전파에서:

```python
self.dbeta = np.sum(dout, axis=0)
self.dgamma = np.sum(self.x_norm * dout, axis=0)
```

의미:

```text
beta는 더하기라 dout을 그대로 합침
gamma는 x_norm과 곱해졌으니 x_norm*dout을 합침
```

## 7. Dropout

순전파:

```python
self.mask = np.random.rand(*x.shape) > self.drop_ratio
return x * self.mask
```

역전파:

```python
return dout * self.mask
```

의미:

```text
학습 때 꺼진 뉴런은 역전파 때도 gradient가 흐르지 않음
```

## 8. Cross Entropy Loss

`src/losses.py`:

```python
clipped = np.clip(y_pred, 1e-7, 1.0)
return -np.sum(np.log(clipped[np.arange(batch_size), y_true])) / batch_size
```

의미:

```text
정답 클래스 확률만 뽑음
log(0)을 막기 위해 clip
batch 평균 loss 반환
```

정답이 정수 라벨일 때:

```text
y_true = [3, 0, 8, ...]
```

`y_pred[np.arange(batch_size), y_true]`는 각 샘플의 정답 확률만 뽑음.

## 9. Cross Entropy Gradient

```python
dout = y_pred.copy()
dout[np.arange(batch_size), y_true] -= 1
return dout / batch_size
```

의미:

```text
dout = y - one_hot(t)
```

예:

```text
y = [0.1, 0.7, 0.2]
t = 1
```

정답 1번 위치에서 1을 빼면:

```text
dout = [0.1, -0.3, 0.2]
```

정답 클래스는 확률을 올려야 하므로 음수 gradient가 나오고,
오답 클래스는 확률을 내려야 하므로 양수 gradient가 나옴.

## 10. NeuralNetwork

`src/network.py`는 layer를 순서대로 쌓음.

```python
self.layers[f"Affine{idx}"] = Affine(...)
if use_batchnorm:
    self.layers[f"BatchNorm{idx}"] = BatchNorm(...)
self.layers[f"ReLU{idx}"] = ReLU()
if use_dropout:
    self.layers[f"Dropout{idx}"] = Dropout(...)
self.layers["Softmax"] = Softmax()
```

구조:

```text
784
 -> Affine(512)
 -> BatchNorm
 -> ReLU
 -> Dropout
 -> Affine(256)
 -> BatchNorm
 -> ReLU
 -> Dropout
 -> Affine(10)
 -> Softmax
```

## 11. forward

```python
out = x
for layer in self.layers.values():
    out = layer.forward(out)
return out
```

의미:

```text
입력 x를 첫 layer부터 마지막 layer까지 순서대로 통과시킴
```

결과:

```text
batch_size x 10 확률
```

## 12. backward

```python
for layer in reversed(self.layers.values()):
    dout = layer.backward(dout)
```

의미:

```text
loss 쪽에서 시작한 gradient를 마지막 layer부터 첫 layer까지 거꾸로 넘김
```

Affine layer는 이 과정에서:

```text
dW, db 저장
```

BatchNorm layer는:

```text
dgamma, dbeta 저장
```

## 13. gradient

```python
y_pred = self.forward(x, train=True)
loss = cross_entropy_loss(y_pred, y)
dout = cross_entropy_gradient(y_pred, y)
self.backward(dout)
return loss
```

의미:

```text
순전파로 y_pred 만들기
loss 계산
softmax + cross entropy의 시작 gradient 만들기
backward로 모든 grads 계산
loss 반환
```

즉 `gradient()`는 이름 그대로:

```text
현재 batch에 대한 gradient를 model.grads에 채우는 함수
```

임.

## 14. Optimizer

SGD:

```python
params[key] -= self.lr * grads[key]
```

Adam:

```python
self.m[key] = beta1 * self.m[key] + (1 - beta1) * grads[key]
self.v[key] = beta2 * self.v[key] + (1 - beta2) * (grads[key] ** 2)
m_hat = self.m[key] / (1 - beta1**self.t)
v_hat = self.v[key] / (1 - beta2**self.t)
params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + eps)
```

의미:

```text
grads를 보고 params를 직접 변경함
```

## 15. Training Loop

`src/training.py`:

```python
loss = model.gradient(x_batch, y_batch)
optimizer.update(model.params, model.grads)
```

이 두 줄이 학습 핵심임.

```text
model.gradient:
  gradient 계산

optimizer.update:
  파라미터 업데이트
```

전체:

```text
epoch 반복
 -> 데이터 섞기
 -> mini-batch 뽑기
 -> gradient 계산
 -> optimizer 업데이트
 -> epoch loss 기록
```

## 16. evaluate

```python
y_pred = model.predict(x)
accuracy = np.mean(np.argmax(y_pred, axis=1) == y) * 100
```

의미:

```text
확률 중 가장 큰 index를 예측값으로 보고
정답 y와 비교해서 정확도를 계산
```

## 17. mnist_lab.ipynb

노트북 흐름:

```text
1. src 경로 추가
2. MNIST 데이터 로드
3. 전체 테스트 실행
4. 모델 생성
5. train()
6. evaluate()
7. loss curve 출력
```

처음 실행은 빠르게 확인하도록:

```text
EPOCHS = 1
TRAIN_LIMIT = 5000
TEST_LIMIT = 1000
```

으로 잡혀 있음.

전체 학습을 보고 싶으면:

```text
EPOCHS를 늘림
TRAIN_LIMIT = 0
TEST_LIMIT = 0
```

으로 바꾸면 됨.

## 18. src/learning/calculate.py

이 파일은 실제 학습용이라기보다 이해용임.

하는 일:

```text
작은 숫자 x, W1, W2를 둠
순전파 값을 직접 계산
softmax + cross entropy 미분을 증명
dW2, db2, dz1, da1, dW1, db1 계산
수치미분과 역전파를 비교
```

공부할 때는 이 파일을 먼저 돌려보는 게 좋음.

```bash
python src/learning/calculate.py
```

## 19. src/learning/mini_batch.py

이 파일은 learning 폴더 안에서 독립적으로 MNIST를 학습해보는 예제임.

중요한 점:

```text
본소스의 network.py에 의존하지 않고
data.py만 가져와서 학습 흐름을 확인함
```

환경변수로 조건을 바꿀 수 있음.

```bash
MNIST_ITERS=10 MNIST_BATCH_SIZE=20 python src/learning/mini_batch.py
```

## 20. 전체 연결

개념과 코드를 연결하면:

```text
순전파:
  NeuralNetwork.forward
  Affine.forward
  ReLU.forward
  Softmax.forward

손실:
  cross_entropy_loss

역전파 시작:
  cross_entropy_gradient

역전파:
  NeuralNetwork.backward
  Affine.backward
  ReLU.backward
  BatchNorm.backward
  Dropout.backward

업데이트:
  Adam.update

평가:
  evaluate
```

## 확인 질문

```text
1. model.gradient()는 정확히 어떤 일을 하는가?
2. optimizer.update()는 loss를 다시 계산하는가?
3. Softmax.backward()가 dout을 그대로 반환하는 이유는?
4. cross_entropy_gradient()에서 정답 위치에 1을 빼는 이유는?
5. Affine.backward()에서 dW, db는 어디에 저장되는가?
```

