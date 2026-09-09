# lrn-mnist

숫자 이미지 한 장을 받아 0~9로 분류하는 NumPy 신경망이다. 행렬 연산과 역전파를 직접 구현한 MLP를 학습하고, 저장한 모델로 이미지 파일이나 브라우저에서 그린 숫자를 판별한다. 전처리된 28×28 입력도 함께 보여 주므로 잘못 읽은 숫자가 어떤 모습으로 모델에 들어갔는지 확인할 수 있다.

## 실행

Python 3.12와 uv가 필요하다. `make setup`은 이 저장소의 `.venv`만 준비하며 의존성을 `requirements.lock`으로 고정한다.

```sh
make setup
make demo
make serve
# 브라우저: http://127.0.0.1:8765
```

`make demo`는 제공된 숫자 7 이미지의 예측과 클래스별 점수를 출력한다. `make serve`는 손그림 화면을 열며 Ctrl-C로 종료한다. 둘 다 저장소에 포함된 모델을 사용하므로 재학습이나 MNIST 다운로드가 필요 없다.

다른 이미지나 새로 학습한 모델을 사용하려면 다음 명령을 실행한다.

```sh
.venv/bin/python src/application.py predict examples/digit-7.png
# 새 모델을 학습할 때 (기본 모델은 덮어쓰지 않음)
make train
.venv/bin/python src/application.py evaluate --model .artifacts/model.npz --output .artifacts/my-evaluation/metrics.json
```

## 입력에서 출력까지

이미지 → 반전·crop·20×20 비율 유지·28×28 중심 정렬 → 직접 구현한 MLP → 10개 class probability

기존 NumPy Affine·ReLU·Softmax·Cross Entropy와 직접 구현한 backward, SGD·Adam, BatchNorm·Dropout을 사용한다. Dropout은 학습 때 무작위 mask를 적용하고 추론 때는 유지 비율을 곱하는 방식이다. BatchNorm은 추론 때 저장된 running statistics를 쓴다.

공식 MNIST training 60,000개를 seed 42로 50,000 training / 10,000 validation으로 나눈다. 공식 test 10,000개는 모델 선택에 사용하지 않는다. 모델은 validation accuracy로 선택하며 training 도중 test 평가를 하지 않는다.

`models/reference.npz`에 실제 학습 모델을 포함하므로 demo·웹 추론에는 데이터 다운로드나 재학습이 필요 없다. NumPy arrays와 JSON manifest만 저장하고 pickle을 사용하지 않는다. 기본 제공 모델은 hidden [128,64], BN, dropout 0.1, Adam lr 0.001, batch 128, seed 42, 8 epoch로 학습했다. 2026-09-08 검증에서 validation 97.84%, 공식 test 97.80%였다. test는 선택한 모델에 대해 한 번 평가한 결과이며 미래 입력 정확도 보장이 아니다.

웹은 loopback에서 동작한다. 실제 28×28 입력과 모든 class 점수를 보여 주며 빈 입력은 거절한다. 원본 검증 예제 PNG는 MNIST test에서 각 숫자의 첫 항목을 가져왔다. 사용자 손그림은 자동으로 반전·crop·중심 정렬된다.

학습·저장·전처리·CLI는 [`application.py`](src/application.py), MLP 구성은 [`network.py`](src/network.py), 각 계층의 forward/backward는 [`layers.py`](src/layers.py), 손그림 화면은 [`web/index.html`](web/index.html)에 있다.

## 검증과 관찰

```sh
make test
```

계층과 optimizer의 계산, 수치 미분과 backward의 일치, 저장 전후 예측, 데이터 분리, 이미지 반전과 빈 입력 처리를 검사한다. 모델의 학습 조건·데이터 해시·혼동행렬은 [`models/manifest.json`](models/manifest.json)에 있다. `evaluate`는 test 정확도와 혼동행렬, 오분류 이미지를 출력한다.

## 지원 범위와 한계

숫자 하나만 입력한다. CNN·ViT·여러 숫자·OCR·일반 이미지 분류·PyTorch 대체는 제외한다. MNIST와 손그림의 굵기·위치·필기 형태가 달라 오분류할 수 있다. class probability는 보정된 confidence가 아니다.

`make train`/`make evaluate`는 데이터가 없으면 공개 Keras 배포의 `mnist.npz` 약 11 MiB를 내려받는다. 실제 모델 준비 시간·데이터 SHA-256은 `models/manifest.json`에 기록한다. 다운로드·의존성 설치 시간은 학습 시간에 포함하지 않는다. 기본 모델은 weights+BN 통계 복원용이며 optimizer까지 이어 학습하는 기능은 제공하지 않는다.

## 출처와 기여

`krafton-jungle/mnist-lab` 과제와 `Jungle-12-303/wk13_6_mnist` 팀 구현을 거친 `woonyong-kr/SW_AI-W13-mnist`에서 이어 받은 학습용 파생본이다. 기준 원본 revision은 `6a7e4511eb6b97b1762d32648d638c60b5ab9668`이다. 과제 제공물·팀 구현·이후 확장은 Git author와 diff로 구분하며, 기존 저작권 표시는 유지한다. 원본 주소의 공개 접근이 제한돼 있어 자료는 아래 이력 링크로 확인할 수 있다.

기존 신경망 구현 위에 데이터 분리, 모델 저장·복원, 이미지 CLI와 손그림 화면을 연결했다. 과제 설명과 이전 실험은 [정리 전 이력](https://github.com/woonyong-kr/lrn-mnist/tree/1c28de6983671e891faac453e392f68b354bdb53)에 남아 있다.
