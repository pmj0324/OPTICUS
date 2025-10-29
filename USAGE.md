# OPTICUS 사용 가이드

이 문서는 OPTICUS를 사용하여 모델을 학습하고 평가하는 방법을 단계별로 설명합니다.

## 1. 설치

```bash
cd /home/pmj0324/icecube-camera/OPTICUS
pip install -e .
```

## 2. 데이터 준비

HDF5 파일 형식으로 데이터를 준비합니다:
- `images`: (N, 500, 500) 배열
- `labels`: (N,) 배열

## 3. 설정 파일 준비

### Hole Ice용 ViT 모델

```bash
cp opticus/configs/default_vit.yaml configs/my_hole_ice.yaml
```

`configs/my_hole_ice.yaml` 편집:
```yaml
experiment:
  name: "my_hole_ice_experiment"

data:
  hdf5_path: "/home/work/CamSim/SimData_Hole/ICRC_Data/beam_80/hr_0.5_1000.h5"
  batch_size: 32
```

### Bulk Ice용 ViT 모델

```bash
cp opticus/configs/bulk_ice.yaml configs/my_bulk_ice.yaml
```

설정을 필요에 맞게 수정합니다.

## 4. 모델 학습

### 커맨드라인에서 학습

```bash
# Hole ice 학습
opticus-train --config configs/my_hole_ice.yaml --gpu 0

# Bulk ice 학습
opticus-train --config configs/my_bulk_ice.yaml --gpu 0

# 데이터 경로 오버라이드
opticus-train --config configs/my_hole_ice.yaml \
    --data /path/to/your/data.h5 \
    --gpu 0
```

### Python 스크립트에서 학습

```python
import torch
import torch.nn as nn
from opticus.models import ViT50_3block
from opticus.dataloader import load_h5_data, create_dataloaders
from opticus.utils import train_model

# 데이터 로드
images, labels, lbl_min, lbl_max = load_h5_data('path/to/data.h5')

# 데이터로더 생성
train_loader, val_loader, test_loader, dataset = create_dataloaders(
    images, labels, lbl_min, lbl_max,
    batch_size=32,
    train_split=0.6,
    val_split=0.2,
    test_split=0.2
)

# 모델 생성
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ViT50_3block(
    img_size=500,
    patch_size=50,
    embed_dim=128,
    depth=3,
    num_heads=8,
    mlp_dim=512
).to(device)

# Optimizer와 Scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=4
)
criterion = nn.MSELoss()

# 학습
trained_model = train_model(
    model, train_loader, val_loader, criterion,
    optimizer, scheduler, device,
    num_epochs=100,
    patience=10,
    save_path='checkpoints/best_model.pth'
)
```

## 5. 모델 평가

### 커맨드라인에서 평가

```bash
# 테스트 세트 평가
opticus-eval --config configs/my_hole_ice.yaml \
    --checkpoint checkpoints/my_hole_ice_experiment_best.pth \
    --split test \
    --gpu 0

# 검증 세트 평가
opticus-eval --config configs/my_hole_ice.yaml \
    --checkpoint checkpoints/my_hole_ice_experiment_best.pth \
    --split val \
    --gpu 0
```

### Python 스크립트에서 평가

```python
from opticus.utils import calculate_metrics, load_checkpoint
from opticus.utils.metrics import print_metrics
from opticus.analysis import save_all_plots

# 체크포인트 로드
model, _, _, _, _, _ = load_checkpoint(
    model, 'checkpoints/best_model.pth', device=device
)

# 메트릭 계산
metrics = calculate_metrics(model, test_loader, dataset, device)

# 결과 출력
print_metrics(metrics, title="Test Set Results")

# 플롯 저장
save_all_plots(metrics, save_dir='plots/test', unit='cm')
```

## 6. 결과 확인

평가 후 생성되는 파일들:

```
plots/test/
├── pred_vs_true.png          # 예측 vs 실제값
├── error_distribution.png    # 오차 분포
├── error_histogram.png        # 오차 히스토그램
└── relative_error.png         # 상대 오차
```

## 7. 고급 사용법

### 체크포인트에서 학습 재개

설정 파일에서:
```yaml
checkpoint:
  resume_from: "checkpoints/my_experiment_best.pth"
```

### 커스텀 모델 설정

```yaml
model:
  type: "ViT50_3block"
  img_size: 500
  patch_size: 50
  embed_dim: 256    # 더 큰 임베딩 차원
  depth: 4          # 더 깊은 네트워크
  num_heads: 16     # 더 많은 attention head
  mlp_dim: 1024     # 더 큰 MLP
```

### 학습률 조정

```yaml
training:
  optimizer:
    type: "AdamW"
    lr: 0.0001      # 학습률 조정
    weight_decay: 0.01
  
  scheduler:
    type: "ReduceLROnPlateau"
    patience: 4     # LR 감소 전 대기 epoch 수
    factor: 0.5     # LR 감소 비율
```

## 8. 일반적인 문제 해결

### GPU 메모리 부족
```yaml
data:
  batch_size: 16  # 배치 크기 줄이기
```

### 과적합
```yaml
training:
  optimizer:
    weight_decay: 0.05  # 정규화 강화
```

### 학습이 느림
```yaml
data:
  num_workers: 8  # 데이터 로딩 병렬화
```

## 9. 예제 워크플로우

### Hole Ice 완전한 예제

```bash
# 1. 데이터 확인
ls /home/work/CamSim/SimData_Hole/ICRC_Data/beam_80/

# 2. 설정 파일 생성
cp opticus/configs/default_vit.yaml configs/hole_ice_vit.yaml
# configs/hole_ice_vit.yaml 편집

# 3. 학습
opticus-train --config configs/hole_ice_vit.yaml --gpu 0

# 4. 평가
opticus-eval --config configs/hole_ice_vit.yaml \
    --checkpoint checkpoints/hole_ice_vit_best.pth \
    --split test \
    --gpu 0

# 5. 결과 확인
ls plots/test/
```

## 10. 참고사항

- 학습 전에 충분한 디스크 공간을 확보하세요
- GPU 메모리는 최소 8GB 권장
- 학습 시간은 데이터 크기와 모델에 따라 수 시간에서 하루 정도 소요됩니다
- Early stopping을 사용하므로 `num_epochs`는 최대 epoch 수입니다

더 자세한 정보는 README.md를 참고하세요.

