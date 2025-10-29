# OPTICUS

**Optical Property Transformer for IceCube Upgrade Camera System**

OPTICUS는 IceCube 업그레이드 카메라 시스템을 위한 광학 특성 예측 프레임워크입니다. Vision Transformer (ViT)와 CNN 기반 모델을 사용하여 얼음의 산란 길이(scattering length)를 예측합니다.

## 주요 기능

- **다양한 모델 지원**: Vision Transformer (ViT), ResNet 기반 CNN
- **Hole Ice & Bulk Ice 분석**: 두 가지 유형의 얼음 특성 예측
- **설정 기반 학습**: YAML 설정 파일을 통한 유연한 실험 관리
- **상세한 분석 도구**: 예측 결과 시각화 및 메트릭 계산
- **배포 가능한 구조**: 깔끔한 모듈화 및 패키지 구조

## 설치

### 요구사항

- Python >= 3.8
- PyTorch >= 1.10.0
- CUDA (GPU 사용 시)

### 설치 방법

```bash
# 저장소 클론
git clone https://github.com/yourusername/OPTICUS.git
cd OPTICUS

# 개발 모드로 설치
pip install -e .

# 또는 개발 도구 포함 설치
pip install -e ".[dev]"
```

## 프로젝트 구조

```
OPTICUS/
├── opticus/
│   ├── models/          # 모델 정의 (ViT, ResNet)
│   ├── dataloader/      # 데이터 로딩 및 전처리
│   ├── utils/           # 학습, 체크포인트, 메트릭 유틸리티
│   ├── analysis/        # 시각화 및 분석 도구
│   ├── configs/         # 설정 파일 및 관리
│   └── scripts/         # 학습/평가 스크립트
├── checkpoints/         # 학습된 모델 저장
├── data/                # 데이터 디렉토리
├── plots/               # 생성된 플롯 저장
├── setup.py
└── README.md
```

## 빠른 시작

### 1. 데이터 준비

HDF5 형식의 데이터 파일이 필요합니다:
- `images`: (N, 500, 500) 형태의 이미지 배열
- `labels`: (N,) 형태의 레이블 배열

### 2. 설정 파일 준비

`opticus/configs/` 폴더에 예제 설정 파일이 있습니다:

```bash
# Vision Transformer 사용
cp opticus/configs/default_vit.yaml my_config.yaml

# ResNet 사용
cp opticus/configs/default_cnn.yaml my_config.yaml

# Bulk Ice 설정
cp opticus/configs/bulk_ice.yaml my_config.yaml
```

설정 파일에서 데이터 경로를 수정하세요:

```yaml
data:
  hdf5_path: "/path/to/your/data.h5"
```

### 3. 모델 학습

```bash
# 설정 파일을 사용한 학습
opticus-train --config my_config.yaml

# GPU 지정
opticus-train --config my_config.yaml --gpu 0

# 데이터 경로 오버라이드
opticus-train --config my_config.yaml --data /path/to/data.h5
```

또는 Python 스크립트로:

```bash
python opticus/scripts/train.py --config my_config.yaml
```

### 4. 모델 평가

```bash
# 테스트 세트 평가
opticus-eval --config my_config.yaml --checkpoint checkpoints/best_model.pth

# 검증 세트 평가
opticus-eval --config my_config.yaml --checkpoint checkpoints/best_model.pth --split val
```

## Python API 사용

### 기본 사용법

```python
from opticus.models import ViT50_3block, ResNet4
from opticus.dataloader import load_h5_data, create_dataloaders
from opticus.utils import train_model
from opticus.configs import load_config
import torch.nn as nn
import torch.optim as optim

# 설정 로드
config = load_config('my_config.yaml')

# 데이터 로드
images, labels, lbl_min, lbl_max = load_h5_data(config.data.hdf5_path)

# 데이터로더 생성
train_loader, val_loader, test_loader, dataset = create_dataloaders(
    images, labels, lbl_min, lbl_max,
    batch_size=32
)

# 모델 생성
model = ViT50_3block(
    img_size=500,
    patch_size=50,
    embed_dim=128,
    depth=3,
    num_heads=8,
    mlp_dim=512
)

# 학습
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')
criterion = nn.MSELoss()

trained_model = train_model(
    model, train_loader, val_loader, criterion, 
    optimizer, scheduler, device,
    num_epochs=100, patience=10
)
```

### 평가 및 시각화

```python
from opticus.utils import calculate_metrics
from opticus.utils.metrics import print_metrics
from opticus.analysis import save_all_plots

# 메트릭 계산
metrics = calculate_metrics(model, test_loader, dataset, device)

# 메트릭 출력
print_metrics(metrics, title="Test Set Results")

# 플롯 저장
save_all_plots(metrics, save_dir='./plots/test', unit='cm')
```

## 설정 파일 옵션

### 모델 설정

```yaml
model:
  type: "ViT50_3block"  # 또는 "ResNet4"
  img_size: 500
  patch_size: 50
  embed_dim: 128
  depth: 3
  num_heads: 8
  mlp_dim: 512
```

### 학습 설정

```yaml
training:
  num_epochs: 100
  patience: 10
  
  optimizer:
    type: "AdamW"
    lr: 0.0001
    weight_decay: 0.01
  
  scheduler:
    type: "ReduceLROnPlateau"
    mode: "min"
    factor: 0.5
    patience: 4
```

### 데이터 설정

```yaml
data:
  hdf5_path: "/path/to/data.h5"
  batch_size: 32
  train_split: 0.6
  val_split: 0.2
  test_split: 0.2
```

## 모델

### Vision Transformer (ViT)

- **ViT50_3block**: 50x50 패치 크기, 3개 Transformer 블록
- Hole Ice 및 Bulk Ice 예측에 효과적
- 높은 정확도 (평균 상대 오차 < 0.5%)

### ResNet4

- 4개 레이어를 가진 경량 ResNet
- 빠른 학습 및 추론
- CNN 기반 특징 추출

## 결과 분석

평가 후 다음 메트릭과 플롯이 생성됩니다:

### 메트릭
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- Mean Absolute Relative Error (%)
- 68th/95th Percentile Errors

### 플롯
1. Predicted vs True (±5% 밴드)
2. Error Distribution
3. Error Histogram
4. Absolute Relative Error by True Value

## 기여 방법

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 라이선스

This project is licensed under the MIT License.

## 인용

OPTICUS를 사용하신 경우 다음과 같이 인용해 주세요:

```bibtex
@software{opticus2024,
  title={OPTICUS: Optical Property Transformer for IceCube Upgrade Camera System},
  author={IceCube Collaboration},
  year={2024},
  url={https://github.com/yourusername/OPTICUS}
}
```

## 연락처

IceCube Collaboration - [contact@icecube.wisc.edu](mailto:contact@icecube.wisc.edu)

Project Link: [https://github.com/yourusername/OPTICUS](https://github.com/yourusername/OPTICUS)

## 감사의 말

이 프로젝트는 IceCube Neutrino Observatory의 지원을 받아 개발되었습니다.
