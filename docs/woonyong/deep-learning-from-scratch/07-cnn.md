# 07. CNN

CNN은 이미지 같은 격자 구조 데이터를 다루기 좋은 신경망임.

완전연결층은 이미지를 784개 숫자로 펼침.

```text
28 x 28 -> 784
```

이렇게 펼치면 가까운 픽셀끼리의 위치 관계가 약해짐.

CNN은 이미지를 2차원 구조로 보고 작은 필터를 움직이며 특징을 찾음.

```text
이미지 위를 필터가 훑으면서
선, 모서리, 패턴 같은 지역 특징을 찾는다
```

## 1. Fully Connected의 한계

MNIST 이미지를 펼치면:

```text
x: 784
```

완전연결층은 모든 픽셀과 모든 뉴런을 연결함.

문제:

```text
픽셀의 공간 구조를 직접 보존하지 않음
파라미터 수가 많아짐
이미지의 지역 패턴을 효율적으로 쓰지 못함
```

예를 들어 숫자 3의 곡선은 가까운 픽셀들의 관계로 나타남.
CNN은 이런 지역 패턴을 보기 좋음.

## 2. Convolution

Convolution은 작은 필터를 입력 위에 올려놓고 곱의 합을 계산함.

예:

```text
입력 일부:
[
  [1, 2],
  [3, 4],
]

필터:
[
  [0.1, 0.2],
  [0.3, 0.4],
]
```

계산:

```text
1*0.1 + 2*0.2 + 3*0.3 + 4*0.4
= 0.1 + 0.4 + 0.9 + 1.6
= 3.0
```

이 결과가 feature map의 한 칸이 됨.

## 3. 필터

필터는 이미지에서 찾고 싶은 패턴을 담는 가중치임.

초기에는 랜덤이지만 학습을 통해 바뀜.

초기 필터:

```text
랜덤 숫자
```

학습 후 필터:

```text
세로선 감지
가로선 감지
모서리 감지
곡선 감지
```

처럼 특정 패턴에 반응하게 될 수 있음.

## 4. Padding

padding은 입력 주변에 0을 둘러싸는 것임.

예:

```text
입력 4 x 4
필터 3 x 3
padding 없음 -> 출력 2 x 2
padding 1 -> 출력 4 x 4
```

padding을 쓰는 이유:

```text
출력 크기를 조절
가장자리 정보도 더 잘 사용
깊은 층에서 feature map이 너무 빨리 작아지는 것 방지
```

## 5. Stride

stride는 필터를 몇 칸씩 움직일지임.

```text
stride 1:
  한 칸씩 이동

stride 2:
  두 칸씩 이동
```

stride가 커지면 출력 크기가 작아짐.

## 6. 출력 크기 공식

입력 크기 H, W
필터 크기 FH, FW
padding P
stride S

출력:

```text
OH = (H + 2P - FH) / S + 1
OW = (W + 2P - FW) / S + 1
```

예:

```text
H = W = 28
FH = FW = 3
P = 1
S = 1
```

계산:

```text
OH = (28 + 2*1 - 3) / 1 + 1 = 28
OW = 28
```

출력 크기가 입력과 같게 유지됨.

## 7. Channel

이미지는 channel을 가질 수 있음.

```text
흑백 이미지:
  1 channel

RGB 이미지:
  3 channels
```

MNIST는 흑백이라:

```text
1 x 28 x 28
```

컬러 이미지는:

```text
3 x H x W
```

필터도 입력 channel 수를 맞춰야 함.

```text
필터 shape:
  out_channel x in_channel x filter_h x filter_w
```

## 8. Pooling

Pooling은 feature map 크기를 줄임.

대표적으로 max pooling:

```text
2 x 2 영역에서 가장 큰 값만 뽑음
```

예:

```text
[
  [1, 3],
  [2, 4],
]
```

max pooling 결과:

```text
4
```

의미:

```text
작은 위치 변화에 덜 민감하게 만듦
크기를 줄여 계산량을 줄임
강하게 반응한 특징을 남김
```

## 9. Pooling 역전파

max pooling은 forward 때 가장 큰 값의 위치를 기억함.

역전파 때는 그 위치로만 gradient를 보냄.

예:

```text
forward 입력:
[
  [1, 3],
  [2, 4],
]

max 위치: 4
```

뒤에서 gradient 10이 왔다면:

```text
[
  [0, 0],
  [0, 10],
]
```

처럼 max였던 위치에만 전달됨.

## 10. im2col

Convolution을 그대로 구현하면 for문이 많아짐.

im2col은 입력의 지역 영역들을 행렬로 펼쳐서 convolution을 행렬곱으로 바꾸는 기법임.

```text
이미지의 각 필터 적용 영역을 한 줄로 펼침
필터도 한 줄로 펼침
행렬곱으로 한 번에 계산
```

장점:

```text
NumPy의 빠른 행렬곱을 활용할 수 있음
구현이 단순해짐
```

단점:

```text
메모리를 더 많이 쓸 수 있음
```

## 11. CNN 전체 흐름

이미지 분류 CNN은 보통:

```text
Conv
 -> ReLU
 -> Pooling
 -> Conv
 -> ReLU
 -> Pooling
 -> Affine
 -> ReLU
 -> Affine
 -> Softmax
```

처럼 구성됨.

초반 Conv:

```text
선, 모서리 같은 낮은 수준 특징
```

후반 Conv:

```text
더 복잡한 패턴
```

마지막 Affine:

```text
클래스별 점수 계산
```

## 12. CNN도 결국 같은 학습

CNN이 특별해 보여도 학습 흐름은 같음.

```text
순전파
loss 계산
역전파
optimizer 업데이트
```

다른 점은 계층이 Affine만 있는 게 아니라:

```text
Convolution
Pooling
```

이 추가된 것임.

Convolution의 필터도 W임.
Pooling은 학습 파라미터는 없지만 gradient를 어디로 보낼지 결정함.

## 13. CNN이 이미지에 좋은 이유

```text
지역 연결:
  가까운 픽셀 관계를 잘 봄

가중치 공유:
  같은 필터를 이미지 전체에 적용

위치 변화에 어느 정도 강함:
  같은 패턴이 조금 이동해도 필터가 잡을 수 있음

파라미터 효율:
  완전연결보다 적은 파라미터로 이미지 패턴을 표현
```

## 확인 질문

```text
1. 완전연결층은 왜 이미지의 공간 구조를 약하게 만드는가?
2. convolution 필터는 무엇을 학습하는가?
3. padding과 stride는 출력 크기에 어떤 영향을 주는가?
4. max pooling 역전파는 왜 max 위치에만 gradient를 보내는가?
5. im2col은 convolution을 왜 빠르게 만들 수 있는가?
```

