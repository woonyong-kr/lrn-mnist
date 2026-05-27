# 05. 역전파

역전파는 모든 파라미터의 gradient를 빠르게 구하는 방법임.

수치미분은 파라미터 하나씩 흔들어보지만,
역전파는 순전파 때 저장한 값을 이용해 뒤에서 앞으로 gradient를 전달함.

```text
순전파:
x -> layer1 -> layer2 -> layer3 -> loss

역전파:
loss -> layer3 -> layer2 -> layer1 -> x
```

핵심은 연쇄법칙임.

## 1. 계산 그래프

계산 그래프는 계산을 작은 연산 단위로 쪼갠 것임.

예:

```text
z = x * y
L = z + 5
```

그래프:

```text
x ----*
      |--> z --> +5 --> L
y ----*
```

여기서 L은 최종 결과임.
딥러닝에서는 보통 loss를 L이라고 둠.

## 2. 왜 z가 나오는가

우리가 원래 익숙한 건:

```text
y = f(x)
```

형태임.

그런데 계산이 복잡해지면 중간 결과가 생김.

```text
z = x * y
L = z + 5
```

여기서 z는 최종 목적이 아니라 중간 계산 결과임.

신경망에서도:

```text
a1 = x @ W1 + b1
z1 = ReLU(a1)
a2 = z1 @ W2 + b2
y = softmax(a2)
L = cross_entropy(y, t)
```

처럼 중간값이 계속 생김.

역전파는 이 중간값들을 거꾸로 따라가는 것임.

## 3. 연쇄법칙 다시 보기

```text
x -> h -> L
```

이면:

```text
dL/dx = dL/dh * dh/dx
```

말로:

```text
x가 h를 얼마나 바꾸는지
h가 L을 얼마나 바꾸는지
둘을 곱하면 x가 L을 얼마나 바꾸는지 알 수 있음
```

역전파는 이걸 계속 반복함.

## 4. 덧셈 노드

순전파:

```text
z = x + y
```

미분:

```text
dz/dx = 1
dz/dy = 1
```

뒤에서 `dL/dz`가 왔다면:

```text
dL/dx = dL/dz * dz/dx = dL/dz
dL/dy = dL/dz * dz/dy = dL/dz
```

즉 덧셈 노드는 gradient를 그대로 나눠줌.

## 5. 곱셈 노드

순전파:

```text
z = x * y
```

미분:

```text
dz/dx = y
dz/dy = x
```

뒤에서 `dL/dz`가 왔다면:

```text
dL/dx = dL/dz * y
dL/dy = dL/dz * x
```

즉 곱셈 노드는 상대방 값을 곱해서 gradient를 넘김.

예:

```text
x = 2
y = 3
z = 6
L = z + 5
```

`dL/dz = 1`이므로:

```text
dL/dx = 1 * y = 3
dL/dy = 1 * x = 2
```

## 6. Affine 역전파

순전파:

```text
out = x @ W + b
```

역전파:

```text
dW = x.T @ dout
db = sum(dout, axis=0)
dx = dout @ W.T
```

왜 그런지 하나씩 보면 됨.

### dW

출력 j:

```text
out_j = x0*W0j + x1*W1j + ... + b_j
```

W_ij만 보면:

```text
out_j = x_i * W_ij + ...
```

그래서:

```text
dout_j/dW_ij = x_i
```

loss까지 연결하면:

```text
dL/dW_ij = dL/dout_j * dout_j/dW_ij
          = dout_j_gradient * x_i
```

즉:

```text
dW_ij = x_i * dout_j
```

행렬로:

```text
dW = x.T @ dout
```

### db

```text
out_j = ... + b_j
```

b_j를 1 올리면 out_j도 1 올라감.

```text
dout_j/db_j = 1
```

따라서:

```text
dL/db_j = dL/dout_j
```

batch가 여러 개면 같은 j끼리 더함.

```text
db = sum(dout, axis=0)
```

### dx

x_i는 모든 출력에 영향을 줌.

```text
out_0 = x_i*W_i0 + ...
out_1 = x_i*W_i1 + ...
out_2 = x_i*W_i2 + ...
```

그래서:

```text
dL/dx_i = dL/dout_0 * W_i0
        + dL/dout_1 * W_i1
        + dL/dout_2 * W_i2
        + ...
```

행렬로:

```text
dx = dout @ W.T
```

## 7. ReLU 역전파

순전파:

```text
y = max(0, x)
```

미분:

```text
x > 0이면 1
x <= 0이면 0
```

역전파:

```python
dx = dout.copy()
dx[x <= 0] = 0
```

즉:

```text
순전파 때 꺼졌던 곳은 gradient도 막는다.
```

## 8. Sigmoid 역전파

순전파:

```text
y = sigmoid(x)
```

미분:

```text
dy/dx = y * (1 - y)
```

역전파:

```text
dx = dout * y * (1 - y)
```

여기서 y는 순전파 때 저장해둔 sigmoid 출력임.

## 9. SoftmaxWithLoss 역전파

MNIST 출력층:

```text
a -> softmax -> y -> cross entropy -> L
```

softmax와 cross entropy를 합쳐서 미분하면:

```text
dL/da = y - t
```

batch 평균이면:

```text
dL/da = (y - t) / batch_size
```

현재 코드의 `cross_entropy_gradient`가 이 역할을 함.

```python
dout = y_pred.copy()
dout[np.arange(batch_size), y_true] -= 1
return dout / batch_size
```

정답 위치만 1을 빼기 때문에:

```text
y - one_hot(t)
```

가 됨.

## 10. Softmax + Cross Entropy 증명

2개 클래스 기준:

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

대입:

```text
L = -log(exp(a0) / S)
```

로그 성질:

```text
log(A / B) = log(A) - log(B)
```

따라서:

```text
L = -log(exp(a0)) + log(S)
  = -a0 + log(exp(a0) + exp(a1))
```

a0로 미분:

```text
dL/da0
= d(-a0)/da0 + d(log(S))/da0
= -1 + (1/S) * dS/da0
= -1 + exp(a0)/S
= y0 - 1
= y0 - t0
```

a1로 미분:

```text
dL/da1
= 0 + (1/S) * dS/da1
= exp(a1)/S
= y1
= y1 - t1
```

따라서:

```text
dL/da = y - t
```

## 11. 전체 역전파 흐름

순전파:

```text
a1 = x @ W1 + b1
z1 = ReLU(a1)
a2 = z1 @ W2 + b2
y = softmax(a2)
L = cross_entropy(y, t)
```

역전파:

```text
da2 = y - t

dW2 = z1.T @ da2
db2 = sum(da2, axis=0)
dz1 = da2 @ W2.T

da1 = dz1 * ReLU_grad(a1)

dW1 = x.T @ da1
db1 = sum(da1, axis=0)
dx = da1 @ W1.T
```

업데이트:

```text
W1 = W1 - lr * dW1
b1 = b1 - lr * db1
W2 = W2 - lr * dW2
b2 = b2 - lr * db2
```

## 12. 수치미분과 역전파 비교

수치미분:

```text
파라미터 하나씩 +h, -h로 흔들어봄
매번 순전파를 다시 함
정확하지만 느림
```

역전파:

```text
순전파 때 중간값을 저장함
loss에서 시작해 뒤로 gradient를 전달함
한 번의 forward/backward로 전체 gradient를 구함
```

둘이 구하는 값은 같은 gradient임.

차이는:

```text
어떻게 구하느냐
```

임.

## 13. Layer 구조로 구현하는 이유

각 계층을 객체로 만들면:

```text
forward:
  입력을 받아 출력을 만들고 필요한 값을 저장

backward:
  뒤에서 온 gradient를 받아 앞쪽 gradient를 반환
  동시에 dW, db 같은 파라미터 gradient 저장
```

이 구조가 되면 네트워크 전체는:

```python
for layer in layers:
    x = layer.forward(x)

for layer in reversed(layers):
    dout = layer.backward(dout)
```

처럼 단순하게 구현됨.

## 확인 질문

```text
1. 덧셈 노드는 왜 gradient를 그대로 흘리는가?
2. 곱셈 노드는 왜 상대방 값을 곱하는가?
3. dW = x.T @ dout에서 x와 dout은 각각 무슨 의미인가?
4. Softmax + Cross Entropy에서 왜 y - t가 나오는가?
5. 역전파가 수치미분보다 빠른 이유는?
```

