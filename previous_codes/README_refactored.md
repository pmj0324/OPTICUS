# OPTICUS - Refactored Version
Optical Property Transformer for IceCube Upgrade Camera System

이 디렉토리는 OPTICUS 프로젝트를 모듈화하고 리팩토링한 버전입니다.

## 📁 프로젝트 구조

```
OPTICUS/
├── models/                    # 모델 정의
│   ├── __init__.py
│   └── vit.py                # ViT50_3block 모델
├── dataloader/                # 데이터 로더
│   ├── __init__.py
│   └── dataset.py            # NoisyImageDataset 및 데이터 로딩 유틸
├── training/                  # 학습 관련
│   ├── __init__.py
│   └── trainer.py            # Trainer 클래스
├── utils/                     # 유틸리티 함수
│   ├── __init__.py
│   ├── seed.py               # 시드 고정
│   ├── metrics.py            # 평가 메트릭
│   └── visualization.py      # 시각화 함수
├── config.yaml                # 설정 파일
├── train.py                   # 메인 학습 스크립트
├── requirements.txt           # 의존성 패키지
└── README_refactored.md       # 이 파일
```

## 🚀 사용 방법

### 1. 환경 설정

```bash
# 패키지 설치
pip install -r requirements.txt
```

### 2. 설정 파일 수정

`config.yaml` 파일을 열어 원하는 설정을 수정하세요:

- **데이터 경로**: `data.train_path`, `data.val_path`, `data.test_path`
- **모델 파라미터**: `model` 섹션
- **학습 설정**: `training` 섹션 (epoch 수, patience, 저장 경로 등)
- **Optimizer**: `optimizer` 섹션 (learning rate, weight decay)
- **Scheduler**: `scheduler` 섹션 (type, factor, patience 등)

### 3. 학습 실행

#### 기본 학습
```bash
cd /home/work/CamSim/OPTICUS
python train.py --config config.yaml
```

#### Learning rate 변경하여 학습
```bash
python train.py --config config.yaml --lr 1e-5
```

#### Batch size 및 epoch 수 변경
```bash
python train.py --config config.yaml --batch_size 64 --epochs 100
```

#### 체크포인트에서 이어서 학습
```bash
python train.py --config config.yaml --resume ./checkpoints/best_vit50_3block.pth --epochs 150 --lr 5e-6
```

#### 체크포인트에서 이어서 학습 + 스케줄러 리셋
```bash
python train.py --config config.yaml --resume ./checkpoints/best_vit50_3block.pth --reset_scheduler
```

## ⚙️ 주요 기능

### 1. 모듈화된 구조
- **models**: 모델 아키텍처 정의
- **dataloader**: 데이터셋 및 데이터 로더 관리
- **training**: 학습 로직 (Trainer 클래스)
- **utils**: 재사용 가능한 유틸리티 함수들

### 2. YAML 기반 설정
- 모든 하이퍼파라미터를 `config.yaml`에서 관리
- 명령줄 인자로 일부 설정 오버라이드 가능

### 3. 체크포인트 관리
- Best validation loss를 가진 모델 자동 저장
- 체크포인트에서 학습 재개 가능

### 4. Early Stopping
- Validation loss가 개선되지 않으면 자동으로 학습 중단

### 5. Learning Rate Scheduling
- ReduceLROnPlateau 또는 CosineAnnealingLR 지원
- YAML에서 쉽게 변경 가능

### 6. 평가 및 시각화
- 학습 완료 후 자동으로 테스트 세트 평가
- 4가지 플롯 생성:
  1. Predicted vs True (±5% band)
  2. Error vs True (±5% band)
  3. Histogram of Errors
  4. Absolute Relative Error by True value
- Training history 플롯 (loss, learning rate)

## 📊 설정 파일 예시

```yaml
# 기본 설정
seed: 14000

# 데이터 경로
data:
  train_path: '/path/to/train_dataset.h5'
  val_path: '/path/to/val_dataset.h5'
  test_path: '/path/to/test_dataset.h5'

# 모델 설정
model:
  name: 'ViT50_3block'
  img_size: 500
  patch_size: 50
  embed_dim: 256
  depth: 3
  num_heads: 8
  mlp_dim: 1024

# 학습 설정
training:
  num_epochs: 300
  patience: 11
  save_path: './checkpoints/best_vit50_3block.pth'

# Optimizer
optimizer:
  lr: 1.0e-4
  weight_decay: 0.01

# Scheduler
scheduler:
  type: 'ReduceLROnPlateau'
  factor: 0.7
  patience: 2
```

## 🔧 커스터마이징

### 새로운 모델 추가
1. `models/` 디렉토리에 새 모델 파일 추가
2. `models/__init__.py`에 import 추가
3. `train.py`의 `create_model()` 함수에 모델 생성 로직 추가
4. `config.yaml`에서 모델 이름 변경

### 새로운 스케줄러 추가
1. `training/trainer.py`의 `__init__()` 메서드에 스케줄러 생성 로직 추가
2. `config.yaml`에서 스케줄러 타입 및 파라미터 설정

## 📝 주의사항

- HDF5 파일은 `images`와 `labels` 키를 포함해야 합니다
- 이미지는 (N, 500, 500) 형태의 grayscale 이미지여야 합니다
- 픽셀 값 범위는 [0, 4095]로 가정합니다
- 레이블은 자동으로 [-1, 1]로 정규화됩니다

## 📄 라이센스

원본 OPTICUS 프로젝트와 동일

## 🙏 기여

버그 리포트나 기능 요청은 이슈로 남겨주세요.

