# 손실 함수: 교차 엔트로피 오차와 오차제곱합

## 1. 손실 함수란

신경망은 입력 \(x\)를 받아 예측값 \(y\)를 만든다.

$$
x \rightarrow model \rightarrow y
$$

학습할 때는 예측값 \(y\)가 정답 \(t\)와 얼마나 다른지 숫자 하나로 계산해야 한다. 이 숫자가 손실 함수이다.

$$
loss = \text{예측이 얼마나 틀렸는지 나타내는 값}
$$

학습의 목표는 이 손실 값을 작게 만드는 것이다.

## 2. 오차제곱합

오차제곱합은 예측값과 정답값의 차이를 제곱해서 더하는 방식이다.

$$
E = \frac{1}{2}\sum_k (y_k - t_k)^2
$$

예를 들어 예측값과 정답값이 다음과 같다고 하자.

$$
y = [0.1,\ 0.2,\ 0.7]
$$

$$
t = [0,\ 0,\ 1]
$$

계산은 다음과 같다.

$$
\begin{aligned}
E
&= \frac{1}{2}\left((0.1 - 0)^2 + (0.2 - 0)^2 + (0.7 - 1)^2\right) \\
&= \frac{1}{2}(0.01 + 0.04 + 0.09) \\
&= 0.07
\end{aligned}
$$

오차제곱합은 예측값 자체가 정답값에 가까워지도록 만든다. 그래서 보통 출력층이 항등 함수인 회귀 문제에서 자연스럽다.

예를 들어 집값 예측 문제에서는 출력이 확률이 아니라 실제 숫자이다.

$$
\text{모델 출력} = 37.2,\quad \text{정답} = 40.0
$$

이 경우에는 예측 숫자와 정답 숫자의 차이를 직접 줄이는 오차제곱합이 잘 맞는다.

$$
E = \frac{1}{2}(37.2 - 40.0)^2
$$

## 3. 교차 엔트로피 오차

교차 엔트로피 오차는 정답 클래스에 대해 모델이 부여한 확률을 본다.

$$
E = -\sum_k t_k \log y_k
$$

정답이 one-hot인 경우를 보자.

$$
y = [0.1,\ 0.2,\ 0.7]
$$

$$
t = [0,\ 0,\ 1]
$$

계산은 다음과 같다.

$$
\begin{aligned}
E
&= -(0 \cdot \log 0.1 + 0 \cdot \log 0.2 + 1 \cdot \log 0.7) \\
&= -\log 0.7
\end{aligned}
$$

즉 정답 클래스의 예측 확률만 손실에 직접 반영된다.

$$
\begin{aligned}
\text{정답 확률 } 0.9 &\Rightarrow -\log 0.9 \quad \text{손실 작음} \\
\text{정답 확률 } 0.1 &\Rightarrow -\log 0.1 \quad \text{손실 큼} \\
\text{정답 확률 } 0.001 &\Rightarrow -\log 0.001 \quad \text{손실 매우 큼}
\end{aligned}
$$

분류 문제에서는 모델 출력이 보통 클래스별 확률이다.

$$
\text{출력} = [P(0),\ P(1),\ \cdots,\ P(9)]
$$

따라서 정답 클래스의 확률을 크게 만드는 교차 엔트로피 오차가 자연스럽다.

## 4. Softmax와 교차 엔트로피

Softmax는 점수(logit)를 확률로 바꾼다.

$$
y_k = \frac{e^{z_k}}{\sum_i e^{z_i}}
$$

예를 들어:

$$
z = [2.0,\ 1.0,\ 0.1]
$$

$$
softmax(z) \approx [0.659,\ 0.242,\ 0.099]
$$

Softmax 출력은 다음 조건을 만족한다.

$$
0 \le y_k \le 1
$$

$$
\sum_k y_k = 1
$$

즉 Softmax 출력은 확률 분포로 볼 수 있다.

교차 엔트로피는 확률 분포와 잘 맞는다. 정답 클래스의 확률이 높아지면 손실이 줄고, 정답 클래스의 확률이 낮으면 손실이 크게 증가한다.

정답 클래스가 2라고 하면:

$$
y = [0.1,\ 0.2,\ 0.7]
$$

$$
loss = -\log 0.7
$$

더 좋은 예측:

$$
y = [0.01,\ 0.04,\ 0.95]
$$

$$
loss = -\log 0.95
$$

더 나쁜 예측:

$$
y = [0.8,\ 0.15,\ 0.05]
$$

$$
loss = -\log 0.05
$$

분류에서는 “정답 클래스 확률을 크게 만들기”가 핵심이므로 Softmax와 교차 엔트로피가 잘 맞는다.

## 5. 항등 함수와 오차제곱합

항등 함수는 입력을 그대로 출력한다.

$$
y = x
$$

출력층에서 항등 함수를 쓴다는 것은 최종 출력값을 확률로 바꾸지 않는다는 뜻이다.

회귀 문제에서는 출력이 그대로 예측값이다.

$$
y = 37.2,\quad t = 40.0
$$

이때는 “정답 클래스 확률”이라는 개념이 없다. 대신 예측 숫자와 정답 숫자의 차이가 중요하다.

그래서 오차제곱합을 쓴다.

$$
E = \frac{1}{2}(y - t)^2
$$

$$
E = \frac{1}{2}(37.2 - 40.0)^2
$$

즉 항등 함수 출력은 실제 수치 예측에 가깝고, 오차제곱합은 실제 수치 차이를 줄이는 데 맞는 손실 함수이다.

## 6. 왜 분류에서 오차제곱합보다 교차 엔트로피가 더 자주 쓰이나

분류에서도 오차제곱합을 아예 못 쓰는 것은 아니다.

$$
Softmax + \text{오차제곱합}
$$

이 조합도 계산은 가능하다. 하지만 일반적으로는 다음 조합이 더 많이 쓰인다.

$$
Softmax + \text{교차 엔트로피}
$$

첫째, 교차 엔트로피는 확률 문제에 직접 맞는 손실이다.

분류 모델의 목표는 정답 클래스의 확률을 높이는 것이다. 교차 엔트로피는 정답 클래스 확률이 낮을 때 강하게 벌점을 준다.

$$
\begin{aligned}
-\log 0.9 &\approx 0.105 \\
-\log 0.1 &\approx 2.303 \\
-\log 0.001 &\approx 6.908
\end{aligned}
$$

둘째, Softmax와 교차 엔트로피를 같이 쓰면 역전파 식이 단순해진다.

Softmax 출력이 \(y\), 정답 one-hot이 \(t\)일 때 출력층 gradient는 다음처럼 깔끔해진다.

$$
\frac{\partial L}{\partial z} = y - t
$$

여기서 \(z\)는 Softmax에 들어가기 전 점수(logit)이다.

예를 들어:

$$
y = [0.1,\ 0.2,\ 0.7]
$$

$$
t = [0,\ 0,\ 1]
$$

그러면:

$$
y - t = [0.1,\ 0.2,\ -0.3]
$$

이 값은 직관적으로도 해석할 수 있다.

$$
\begin{aligned}
\text{틀린 클래스 확률} &\rightarrow \text{줄인다} \\
\text{정답 클래스 확률} &\rightarrow \text{높인다}
\end{aligned}
$$

## 7. Softmax와 교차 엔트로피를 같이 쓰면 왜 미분식이 단순해지는가

Softmax는 logit \(z\)를 확률 \(y\)로 바꾼다.

$$
y_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

교차 엔트로피 오차는 다음과 같다.

$$
L = -\sum_i t_i \log y_i
$$

one-hot 정답에서는 정답 클래스 하나만 손실에 남는다. 예를 들어 정답 클래스가 2라면:

$$
t = [0,\ 0,\ 1]
$$

$$
L = -\log y_2
$$

여기서 중요한 점은 실제 역전파에서 필요한 값이 \(\frac{\partial L}{\partial y_i}\)가 아니라, Softmax에 들어가기 전 값인 logit \(z\)에 대한 미분이라는 것이다.

$$
\frac{\partial L}{\partial z_j}
$$

먼저 교차 엔트로피를 \(y_i\)에 대해 미분하면:

$$
\frac{\partial L}{\partial y_i}
=
-\frac{t_i}{y_i}
$$

여기서 \(\frac{1}{y_i}\)가 나온다.

Softmax의 미분은 다음 형태를 가진다.

$$
\frac{\partial y_i}{\partial z_j}
=
y_i(\delta_{ij} - y_j)
$$

여기서 \(\delta_{ij}\)는 다음을 뜻한다.

$$
\delta_{ij}
=
\begin{cases}
1 & \text{if } i = j \\
0 & \text{if } i \ne j
\end{cases}
$$

체인룰을 적용하면:

$$
\frac{\partial L}{\partial z_j}
=
\sum_i
\frac{\partial L}{\partial y_i}
\frac{\partial y_i}{\partial z_j}
$$

각 항을 대입한다.

$$
\begin{aligned}
\frac{\partial L}{\partial z_j}
&=
\sum_i
\left(-\frac{t_i}{y_i}\right)
y_i(\delta_{ij} - y_j) \\
&=
\sum_i
-t_i(\delta_{ij} - y_j)
\end{aligned}
$$

교차 엔트로피 미분에서 나온 \(\frac{1}{y_i}\)와 Softmax 미분에 들어 있던 \(y_i\)가 약분된다. 이것이 식이 깔끔해지는 핵심이다.

이제 합을 정리하면:

$$
\begin{aligned}
\frac{\partial L}{\partial z_j}
&=
\sum_i -t_i\delta_{ij}
+ \sum_i t_i y_j \\
&=
-t_j + y_j\sum_i t_i
\end{aligned}
$$

one-hot 정답에서는 정답 위치만 1이고 나머지는 0이므로:

$$
\sum_i t_i = 1
$$

따라서:

$$
\frac{\partial L}{\partial z_j}
=
y_j - t_j
$$

즉 벡터 전체로 쓰면:

$$
\frac{\partial L}{\partial z}
=
y - t
$$

이 결과는 역전파 구현에서 매우 중요하다. Softmax 출력 \(y\)와 정답 \(t\)의 차이만 계산하면 출력층 gradient가 바로 나온다.

```python
dout = y_pred.copy()
dout[np.arange(batch_size), y_true] -= 1
dout /= batch_size
```

위 코드는 정답이 정수 라벨일 때 \(y - t\)를 만드는 방식이다.

## 8. Softmax와 오차제곱합을 쓰면 왜 덜 단순한가

오차제곱합은 다음과 같다.

$$
L = \frac{1}{2}\sum_i (y_i - t_i)^2
$$

먼저 \(y_i\)에 대해 미분하면:

$$
\frac{\partial L}{\partial y_i}
=
y_i - t_i
$$

여기까지만 보면 깔끔하다.

하지만 \(y\)는 Softmax 출력이므로, 실제로 필요한 것은 logit \(z\)에 대한 미분이다.

$$
\frac{\partial L}{\partial z_j}
=
\sum_i
\frac{\partial L}{\partial y_i}
\frac{\partial y_i}{\partial z_j}
$$

대입하면:

$$
\frac{\partial L}{\partial z_j}
=
\sum_i
(y_i - t_i)
y_i(\delta_{ij} - y_j)
$$

여기서는 교차 엔트로피 때처럼 약분되는 \(\frac{1}{y_i}\) 항이 없다. 따라서 Softmax 미분의 복잡한 구조가 그대로 남는다.

정리하면:

$$
\begin{aligned}
Softmax + \text{교차 엔트로피}
&\Rightarrow y - t \\
Softmax + \text{오차제곱합}
&\Rightarrow \sum_i (y_i - t_i)y_i(\delta_{ij} - y_j)
\end{aligned}
$$

즉 Softmax와 오차제곱합을 같이 쓰는 것이 불가능한 것은 아니지만, 역전파 식이 \(y - t\)처럼 단순하게 떨어지지 않는다.

## 9. 이 조합은 우연인가

Softmax와 교차 엔트로피 조합은 단순히 \(y - t\)가 나오도록 억지로 만든 조합이라기보다, 확률 모델 관점에서 자연스럽게 나온 조합이다.

분류 문제에서 모델은 다음을 출력한다.

$$
y = [P(class=0),\ P(class=1),\ \cdots,\ P(class=9)]
$$

즉 모델은 “이 입력이 각 클래스일 확률”을 말한다.

정답 클래스가 \(c\)라면 목표는 정답 클래스 확률을 크게 만드는 것이다.

$$
y_c \text{를 크게 만들기}
$$

확률을 최대화하는 문제는 보통 음의 로그우도를 최소화하는 문제로 바꾼다.

$$
L = -\log y_c
$$

이 식이 바로 one-hot 정답에서의 교차 엔트로피이다.

따라서:

$$
\begin{aligned}
Softmax
&\Rightarrow \text{logit을 확률 분포로 바꾼다} \\
\text{교차 엔트로피}
&\Rightarrow \text{정답 확률의 음의 로그를 최소화한다}
\end{aligned}
$$

이 둘은 확률론적으로 자연스럽게 연결된다. 그런데 미분해보면 \(\frac{1}{y_i}\)와 \(y_i\)가 약분되어 \(y - t\)로 깔끔하게 떨어진다.

그래서 이 조합은 다음 두 가지 장점을 동시에 가진다.

$$
\begin{aligned}
\text{분류 문제에 대한 의미} &\Rightarrow \text{자연스럽다} \\
\text{역전파 계산} &\Rightarrow \text{단순하다}
\end{aligned}
$$

## 10. 정리

분류 문제:

$$
\text{출력층 활성화 함수} = Softmax
$$

$$
\text{손실 함수} = \text{교차 엔트로피 오차}
$$

이유는 출력이 클래스별 확률이고, 정답 클래스 확률을 크게 만드는 것이 목표이기 때문이다.

회귀 문제:

$$
\text{출력층 활성화 함수} = \text{항등 함수}
$$

$$
\text{손실 함수} = \text{오차제곱합 또는 평균제곱오차}
$$

이유는 출력이 실제 숫자이고, 예측값과 정답값의 수치 차이를 줄이는 것이 목표이기 때문이다.

따라서 “Softmax를 쓰면 교차 엔트로피, 항등 함수를 쓰면 오차제곱합”이라는 말은 큰 방향에서 맞다.

정확히는 다음처럼 이해하는 것이 좋다.

$$
\begin{aligned}
Softmax &\Rightarrow \text{분류 문제의 확률 출력에 적합} \\
\text{교차 엔트로피} &\Rightarrow \text{정답 클래스 확률을 키우는 데 적합} \\
\text{항등 함수} &\Rightarrow \text{회귀 문제의 숫자 출력에 적합} \\
\text{오차제곱합} &\Rightarrow \text{예측값과 정답값의 차이를 줄이는 데 적합}
\end{aligned}
$$
