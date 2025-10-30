# OPTICUS - Quick Start Guide

## 🚀 빠른 시작

### 1단계: 기본 학습 실행
```bash
cd /home/work/CamSim/OPTICUS
python train.py --config config.yaml
```

### 2단계: Learning Rate 조정하여 학습
```bash
python train.py --config config.yaml --lr 5e-5 --epochs 200
```

### 3단계: 체크포인트에서 이어서 학습
```bash
python train.py --config config.yaml \
    --resume ./checkpoints/best_vit50_3block.pth \
    --lr 1e-5 \
    --epochs 100 \
    --reset_scheduler
```

## 📝 주요 명령줄 옵션

| 옵션 | 설명 | 예시 |
|------|------|------|
| `--config` | 설정 파일 경로 | `--config config.yaml` |
| `--lr` | Learning rate | `--lr 1e-5` |
| `--epochs` | Epoch 수 | `--epochs 300` |
| `--batch_size` | Batch size | `--batch_size 64` |
| `--resume` | 체크포인트 경로 | `--resume ./checkpoints/model.pth` |
| `--reset_scheduler` | 스케줄러 리셋 | `--reset_scheduler` |

## 🔧 설정 파일 (config.yaml) 주요 항목

### 데이터 경로 설정
```yaml
data:
  train_path: '/path/to/train_dataset.h5'
  val_path: '/path/to/val_dataset.h5'
  test_path: '/path/to/test_dataset.h5'
```

### 모델 설정
```yaml
model:
  img_size: 500          # 이미지 크기
  patch_size: 50         # 패치 크기
  embed_dim: 256         # 임베딩 차원
  depth: 3               # Transformer 레이어 수
  num_heads: 8           # Attention head 수
  mlp_dim: 1024          # MLP 숨겨진 차원
```

### 학습 설정
```yaml
training:
  num_epochs: 300        # 최대 epoch 수
  patience: 11           # Early stopping patience
  save_path: './checkpoints/best_vit50_3block.pth'
```

### Optimizer 설정
```yaml
optimizer:
  lr: 1.0e-4            # Learning rate
  weight_decay: 0.01     # Weight decay
```

### Scheduler 설정 (ReduceLROnPlateau)
```yaml
scheduler:
  type: 'ReduceLROnPlateau'
  factor: 0.7            # LR 감소 비율
  patience: 2            # LR 감소 전 대기 epoch 수
```

### Scheduler 설정 (CosineAnnealingLR)
```yaml
scheduler:
  type: 'CosineAnnealingLR'
  T_max: 100            # Cosine 주기
  eta_min: 1.0e-7       # 최소 LR
```

## 📊 출력 결과

### 학습 중 출력
```
Epoch   1 | Train Loss: 0.114494 | Val Loss: 0.000540 | LR: 0.0001
  → New best val loss: 0.000540, saved to ./checkpoints/best_vit50_3block.pth
Epoch   2 | Train Loss: 0.000606 | Val Loss: 0.000398 | LR: 0.0001
  → New best val loss: 0.000398, saved to ./checkpoints/best_vit50_3block.pth
```

### 평가 결과
```
================================================================================
EVALUATION RESULTS
================================================================================
Number of samples: 1500

Regression Metrics:
  MSE:  0.000084
  RMSE: 0.009165
  MAE:  0.007123
  MAPE: 2.15%

Absolute Relative Error Percentiles:
  50th percentile: 1.85%
  68th percentile: 2.45%
  95th percentile: 4.92%
================================================================================
```

### 생성되는 플롯
1. **evaluation_results.png**: 
   - Predicted vs True (±5% band)
   - Error vs True (±5% band)
   - Histogram of Errors
   - Absolute Relative Error by True value

2. **training_history.png**:
   - Training/Validation Loss
   - Learning Rate Schedule

## 🔍 체크포인트 파일 위치

- 학습 중 best model: `./checkpoints/best_vit50_3block.pth`
- 결과 플롯: `./results/`

## 💡 팁

### 1. 메모리 부족 시
```bash
python train.py --config config.yaml --batch_size 16
```

### 2. 빠른 실험
```bash
python train.py --config config.yaml --epochs 50
```

### 3. Fine-tuning
```bash
python train.py --config config.yaml \
    --resume ./checkpoints/best_vit50_3block.pth \
    --lr 5e-6 \
    --epochs 50
```

### 4. 학습률 스케줄러 변경
`config.yaml`에서:
```yaml
scheduler:
  type: 'CosineAnnealingLR'
  T_max: 100
  eta_min: 1.0e-7
```

## 🐛 문제 해결

### CUDA out of memory
- `--batch_size`를 줄이세요 (예: 16 또는 8)
- `config.yaml`의 `dataloader.num_workers`를 줄이세요

### 학습이 너무 느림
- `config.yaml`의 `dataloader.num_workers`를 늘리세요
- `dataloader.pin_memory: true`로 설정하세요

### 모델 성능이 좋지 않음
- Learning rate를 조정하세요
- Epoch 수를 늘리세요
- Early stopping patience를 늘리세요

## 📞 도움말

전체 문서는 `README_refactored.md`를 참고하세요.

