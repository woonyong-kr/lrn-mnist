# lrn-mnist

NumPy로 직접 학습하는 손글씨 숫자 인식기. 핵심 엔진을 실제 입력으로 실행하고 결과와 내부 동작을 확인하는 독립 프로그램이다.

## 실행

Python 3.12와 uv가 필요하다. `make setup`은 이 저장소의 `.venv`만 준비하며 의존성을 `requirements.lock`으로 고정한다.

```sh
make setup
make test
make demo
make serve
# 브라우저: http://127.0.0.1:8765
.venv/bin/python src/application.py predict examples/digit-7.png
# 새 모델을 학습할 때 (기본 모델은 덮어쓰지 않음)
make train
.venv/bin/python src/application.py evaluate --model .artifacts/model.npz --output .artifacts/my-evaluation/metrics.json
```

서버는 Ctrl-C로 종료한다. demo/test는 자신이 만든 프로세스만 종료한다.

## 입력에서 출력까지

이미지 → 반전·crop·20×20 비율 유지·28×28 중심 정렬 → 직접 구현한 MLP → 10개 class probability

기존 NumPy Affine·ReLU·Softmax·Cross Entropy와 직접 구현한 backward, SGD·Adam, BatchNorm·Dropout을 사용한다. 학습 때만 dropout을 적용하고 추론에서는 저장된 BatchNorm running statistics를 쓴다.

공식 MNIST training 60,000개를 seed 42로 50,000 training / 10,000 validation으로 나눈다. 공식 test 10,000개는 모델 선택에 사용하지 않는다. 모델은 validation accuracy로 선택하며 training 도중 test 평가를 하지 않는다.

`models/reference.npz`에 실제 학습 모델을 포함하므로 demo·웹 추론에는 데이터 다운로드나 재학습이 필요 없다. NumPy arrays와 JSON manifest만 저장하고 pickle을 사용하지 않는다. 기본 제공 모델은 hidden [128,64], BN, dropout 0.1, Adam lr 0.001, batch 128, seed 42, 8 epoch로 학습했다. 2026-09-08 검증에서 validation 97.84%, 공식 test 97.80%였다. test는 선택한 모델에 대해 한 번 평가한 결과이며 미래 입력 정확도 보장이 아니다.

웹은 loopback에서 동작한다. 실제 28×28 입력과 모든 class 점수를 보여 주며 빈 입력은 거절한다. 원본 검증 예제 PNG는 MNIST test에서 각 숫자의 첫 항목을 가져왔다. 사용자 손그림은 자동으로 반전·crop·중심 정렬된다.

구현을 읽는 순서: `src/application.py`, `src/network.py`, `src/layers.py`, `web/index.html`.

## 검증과 관찰

기존 layer/optimizer 테스트와 MLP 수치 미분, checkpoint 전후 예측의 정확한 일치, 분리된 split, 극성 반전·빈 입력을 검증한다. confusion matrix와 오분류 PNG를 평가 출력으로 제공한다. 실제 브라우저에서 빈 입력 거절과 손그림 7 → 모델 입력 표시 → 예측 7을 확인했다.

실행 환경·명령·exit code·원본 백업과 전체 결과는 이번 전환의 별도 작업 폴더에 기록한다. 새 기계에서는 같은 명령으로 직접 재검증한다. 수치가 기록되어 있다는 사실과 현재 실행 성공을 구분한다.

## 지원 범위와 한계

숫자 하나만 입력한다. CNN·ViT·여러 숫자·OCR·일반 이미지 분류·PyTorch 대체는 제외한다. MNIST와 손그림의 굵기·위치·필기 형태가 달라 오분류할 수 있다. class probability는 보정된 confidence가 아니다.

`make train`/`make evaluate`는 데이터가 없으면 공개 Keras 배포의 `mnist.npz` 약 11 MiB를 내려받는다. 실제 모델 준비 시간·데이터 SHA-256은 `models/manifest.json`에 기록한다. 다운로드·의존성 설치 시간은 학습 시간에 포함하지 않는다. 기본 모델은 weights+BN 통계 복원용이며 optimizer까지 이어 학습하는 기능은 제공하지 않는다.

## 출처와 기여

[woonyong-kr/SW_AI-W13-mnist](https://github.com/woonyong-kr/SW_AI-W13-mnist), [Jungle-12-303/wk13_6_mnist](https://github.com/Jungle-12-303/wk13_6_mnist), [krafton-jungle/mnist-lab](https://github.com/krafton-jungle/mnist-lab)에서 이어 받은 학습용 파생본이다. 기준 원본 revision은 `6a7e4511eb6b97b1762d32648d638c60b5ab9668`이다. 원본 과제·팀 코드와 이후 개인 확장을 구분하며, 개별 기여는 Git author와 diff로 확인한다. 기존 저작권 표시는 소스에 유지한다.

과거 문서·실험·기여 기록은 [정리 전 이력](https://github.com/woonyong-kr/lrn-mnist/tree/1c28de6983671e891faac453e392f68b354bdb53)에서 확인할 수 있다. 실행법과 지원 계약은 이 README에 모았다. 개념·설계·실험 해석 자료는 개인 WIKI inbox에서 검토한 뒤 기존 정본에 흡수한다.
