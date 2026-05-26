# 08. 깊은 신경망과 실험 방법

딥러닝 모델은 단순히 층을 많이 쌓는다고 자동으로 좋아지지 않음.

중요한 것은:

```text
표현력
학습 안정성
과적합 관리
실험 기록
```

임.

## 1. 층을 깊게 쌓는 이유

얕은 모델:

```text
간단한 패턴 표현
```

깊은 모델:

```text
단순한 특징을 조합해 복잡한 특징 표현
```

이미지를 예로 들면:

```text
초반:
  선, 모서리

중간:
  곡선, 부분 형태

후반:
  숫자 전체 모양
```

층이 깊으면 이런 계층적 표현이 가능해짐.

## 2. 깊어지면 생기는 문제

층이 깊어지면 gradient가 여러 함수를 지나며 곱해짐.

```text
dL/dx = dL/dh3 * dh3/dh2 * dh2/dh1 * dh1/dx
```

이 곱들이 계속 작으면:

```text
gradient vanishing
```

계속 크면:

```text
gradient explosion
```

이 생길 수 있음.

해결에 도움 되는 것:

```text
ReLU
He initialization
BatchNorm
적절한 optimizer
Residual connection 같은 구조
```

## 3. Train/Test 분리

학습 데이터:

```text
W, b를 업데이트하는 데 사용
```

테스트 데이터:

```text
학습된 모델이 처음 보는 데이터에서 잘 맞는지 확인
```

중요:

```text
test 데이터로 학습하면 안 됨
```

test 성능은 일반화 성능을 보기 위한 것임.

## 4. Overfitting

overfitting은 학습 데이터에만 너무 맞춘 상태임.

증상:

```text
train accuracy 높음
test accuracy 낮음
```

원인:

```text
모델이 너무 큼
데이터가 적음
학습을 너무 오래 함
regularization 부족
```

대응:

```text
Dropout
weight decay
data augmentation
early stopping
모델 크기 줄이기
```

## 5. Underfitting

underfitting은 모델이 학습 데이터도 잘 못 맞추는 상태임.

증상:

```text
train accuracy 낮음
test accuracy도 낮음
loss가 충분히 내려가지 않음
```

대응:

```text
모델 크기 늘리기
epochs 늘리기
learning rate 조정
optimizer 변경
입력 전처리 확인
```

## 6. Loss curve 읽기

loss가 잘 내려가면:

```text
학습이 진행 중
```

loss가 거의 안 내려가면:

```text
learning rate가 너무 작음
gradient가 안 흐름
초기화 문제
코드 문제
```

loss가 심하게 튀면:

```text
learning rate가 너무 큼
batch size가 너무 작아 흔들림
gradient 폭발 가능성
```

train loss는 내려가는데 test accuracy가 안 오르면:

```text
overfitting 가능성
```

## 7. Accuracy 읽기

accuracy는 최종 성능 확인에는 좋음.

하지만 학습 중 gradient 계산에는 안 씀.

이유:

```text
정확도는 계단식 값이라 작은 변화가 반영되지 않음
미분하기 어려움
```

그래서 학습은 loss로 하고,
평가는 accuracy로 함.

## 8. Hyperparameter 실험

바꿔볼 수 있는 값:

```text
learning_rate
batch_size
epochs
hidden_size
layer 수
dropout_ratio
optimizer
weight_decay
initialization
```

한 번에 너무 많이 바꾸면 뭐 때문에 좋아졌는지 모름.

기본 원칙:

```text
한 번에 하나씩 바꾼다.
실험 결과를 기록한다.
train/test를 같이 본다.
```

## 9. 실험 기록 표

예:

```text
실험 1
lr=0.001
optimizer=Adam
dropout=0.5
epochs=5
train acc=...
test acc=...
loss=...
메모=baseline

실험 2
lr=0.0005
optimizer=Adam
dropout=0.5
epochs=5
...
```

기록해야 할 것:

```text
모델 구조
데이터 개수
학습률
batch size
optimizer
regularization
최종 train/test accuracy
loss curve 특징
느낀 점
```

## 10. 좋은 실험의 기준

좋은 실험은 결과만 있는 게 아니라 이유가 있어야 함.

예:

```text
lr을 낮췄더니 loss 진동이 줄었다.
dropout을 넣었더니 train accuracy는 조금 낮아졌지만 test accuracy가 올라갔다.
hidden size를 키웠더니 train은 좋아졌지만 test는 그대로라 overfitting 가능성이 있다.
```

이런 식으로 원인과 결과를 연결해야 함.

## 11. MNIST에서 정확도를 올리는 방향

가능한 개선:

```text
학습 epoch 증가
hidden layer 크기 조정
learning rate 조정
Adam 사용
He initialization
BatchNorm
Dropout 비율 조정
CNN 사용
```

다만 모든 기법을 무조건 넣는다고 좋아지는 것은 아님.

예:

```text
데이터/모델이 작은데 dropout이 너무 강하면 underfitting
learning rate가 너무 크면 Adam이어도 loss가 튐
BatchNorm과 Dropout 위치에 따라 결과가 달라짐
```

## 12. 재현성

실험을 다시 했을 때 비슷한 결과가 나오게 하려면 seed를 고정함.

```python
np.random.seed(0)
```

하지만 완전한 재현은 환경에 따라 어려울 수 있음.

그래도 실험 비교에는 seed 고정이 유용함.

## 13. 모델을 평가할 때 조심할 점

```text
학습 데이터 accuracy만 보고 판단하지 않기
test 데이터로 튜닝을 너무 반복하지 않기
loss와 accuracy를 같이 보기
실행 시간과 모델 크기도 보기
```

포트폴리오나 과제 보고서에서는:

```text
무엇을 바꿨는지
왜 바꿨는지
결과가 어떻게 달라졌는지
해석은 무엇인지
```

가 중요함.

## 확인 질문

```text
1. 깊은 층이 표현력을 높이는 이유는?
2. gradient vanishing은 왜 생기는가?
3. train accuracy는 높은데 test accuracy가 낮으면 무엇을 의심해야 하는가?
4. 실험할 때 한 번에 하나씩 바꿔야 하는 이유는?
5. loss curve와 accuracy는 각각 무엇을 말해주는가?
```

