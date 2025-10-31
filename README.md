# OPTICUS
[![OT-CFM Preprint](http://img.shields.io/badge/paper-arxiv.2302.00482-B31B1B.svg)](https://arxiv.org/abs/2302.00482)
**Optical Property Transformer for IceCube Upgrade Camera System**

## 📖 Overview / 개요

OPTICUS is a Vision Transformer-based deep learning model for predicting optical properties from camera images in the IceCube Upgrade project. This refactored version provides a modular, YAML-configurable pipeline for training, evaluation, and deployment.

OPTICUS는 IceCube Upgrade 프로젝트의 카메라 이미지에서 광학 특성을 예측하는 Vision Transformer 기반 딥러닝 모델입니다. 이 리팩토링된 버전은 학습, 평가, 배포를 위한 모듈화되고 YAML 설정 가능한 파이프라인을 제공합니다.

### 📄 Related Publication / 관련 논문

**Performance Study of the IceCube Upgrade Camera System**  
*Carsten Rott, Minje Park, Matti Jansson, Garrett Iverson, Seowon Choi (for the IceCube Collaboration)*  
arXiv:2507.18525 [astro-ph.IM]  
Presented at the 39th International Cosmic Ray Conference (ICRC2025)

🔗 **Paper**: [https://arxiv.org/abs/2507.18525](https://arxiv.org/abs/2507.18525)

> The IceCube Upgrade Camera System is a novel calibration system designed to calibrate the IceCube detector by measuring the optical properties of the Antarctic ice. Various image analysis methodologies have been explored, ranging from classical maximum likelihood estimation to AI-based approaches using neural networks.

> IceCube Upgrade Camera System은 남극 얼음의 광학 특성을 측정하여 IceCube 검출기를 보정하기 위해 설계된 새로운 보정 시스템입니다. 고전적인 최대우도추정부터 신경망을 사용하는 AI 기반 접근법까지 다양한 이미지 분석 방법론이 탐구되었습니다.

---

## 📁 Project Structure / 프로젝트 구조

```
OPTICUS/
├── models/                    # Model architectures / 모델 아키텍처
│   ├── __init__.py
│   └── vit.py                # ViT50_3block model / ViT50_3block 모델
│
├── dataloader/                # Data loading utilities / 데이터 로딩 유틸리티
│   ├── __init__.py
│   └── dataset.py            # Dataset & DataLoader / 데이터셋 & 데이터로더
│
├── training/                  # Training logic / 학습 로직
│   ├── __init__.py
│   └── trainer.py            # Trainer class / Trainer 클래스
│
├── utils/                     # Utility functions / 유틸리티 함수
│   ├── __init__.py
│   ├── seed.py               # Seed fixing / 시드 고정
│   ├── metrics.py            # Evaluation metrics / 평가 메트릭
│   └── visualization.py      # Plotting functions / 플롯 함수
│
├── examples/                  # Usage examples / 사용 예제
│   ├── 01_initial_training/  # Training from scratch / 처음부터 학습
│   ├── 02_resume_training/   # Resume training / 이어서 학습
│   └── 03_evaluation_visualization/  # Evaluation / 평가
│
├── checkpoints/               # Model checkpoints / 모델 체크포인트
├── results/                   # Results & plots / 결과 및 플롯
│
├── config.yaml               # Main configuration / 메인 설정 파일
├── train.py                  # Main training script / 메인 학습 스크립트
├── requirements.txt          # Dependencies / 의존성 패키지
└── README.md                 # This file / 이 파일
```

---

## 🚀 Quick Start / 빠른 시작

### 1. Installation / 설치

```bash
# Install dependencies / 의존성 패키지 설치
pip install -r requirements.txt
```

### 2. Basic Training / 기본 학습

```bash
# Train from scratch / 처음부터 학습
python train.py --config config.yaml

# With custom parameters / 커스텀 파라미터로 학습
python train.py --config config.yaml --lr 1e-5 --epochs 200
```

### 3. Resume Training / 이어서 학습

```bash
# Resume from checkpoint / 체크포인트에서 재개
python train.py --config config.yaml \
    --resume ./checkpoints/best_model.pth \
    --lr 5e-6 \
    --epochs 100
```

---

## ✨ Key Features / 주요 기능

### Model Architecture / 모델 아키텍처
- **ViT50_3block**: Vision Transformer with 50×50 patches and 3 transformer blocks
- **ViT50_3block**: 50×50 패치와 3개의 transformer 블록을 가진 Vision Transformer
- Configurable embedding dimension, depth, and attention heads
- 설정 가능한 임베딩 차원, 깊이, attention head 수

### Training Features / 학습 기능
- ✓ **Early Stopping**: Automatic stopping when validation loss stops improving
- ✓ **Early Stopping**: 검증 손실이 개선되지 않으면 자동 중단
- ✓ **Learning Rate Scheduling**: ReduceLROnPlateau or CosineAnnealingLR
- ✓ **Learning Rate Scheduling**: ReduceLROnPlateau 또는 CosineAnnealingLR
- ✓ **Checkpoint Management**: Auto-save best model
- ✓ **Checkpoint Management**: 최고 성능 모델 자동 저장
- ✓ **Resume Training**: Continue from saved checkpoints
- ✓ **Resume Training**: 저장된 체크포인트에서 계속 학습

### Evaluation / 평가
- ✓ **Comprehensive Metrics**: MSE, RMSE, MAE, MAPE
- ✓ **종합 메트릭**: MSE, RMSE, MAE, MAPE
- ✓ **Percentile Statistics**: 50th, 68th, 95th percentiles
- ✓ **백분위 통계**: 50, 68, 95 백분위
- ✓ **Visualization**: 4 comprehensive plots
- ✓ **시각화**: 4가지 종합 플롯
- ✓ **CSV Export**: Detailed prediction results
- ✓ **CSV 내보내기**: 상세 예측 결과

---

## 📚 Examples / 예제

### Example 1: Initial Training / 예제 1: 처음부터 학습

Train a new model from scratch:
새로운 모델을 처음부터 학습:

```bash
cd examples/01_initial_training
python train_from_scratch.py
# Or use Jupyter notebook / 또는 Jupyter 노트북 사용
jupyter notebook train_from_scratch.ipynb
```

### Example 2: Resume Training / 예제 2: 이어서 학습

Fine-tune an existing model:
기존 모델 미세 조정:

```bash
cd examples/02_resume_training
python resume_training.py
```

### Example 3: Evaluation & Visualization / 예제 3: 평가 및 시각화

Evaluate a trained model:
학습된 모델 평가:

```bash
cd examples/03_evaluation_visualization
python evaluate_and_visualize.py
```

**See `examples/README.md` for detailed instructions.**
**자세한 설명은 `examples/README.md`를 참조하세요.**

---

## ⚙️ Configuration / 설정

All settings are managed via YAML files. Main configuration sections:
모든 설정은 YAML 파일로 관리됩니다. 주요 설정 섹션:

### Data / 데이터
```yaml
data:
  train_path: '/path/to/train_dataset.h5'
  val_path: '/path/to/val_dataset.h5'
  test_path: '/path/to/test_dataset.h5'
```

### Model / 모델
```yaml
model:
  name: 'ViT50_3block'
  img_size: 500        # Image size / 이미지 크기
  patch_size: 50       # Patch size / 패치 크기
  embed_dim: 256       # Embedding dimension / 임베딩 차원
  depth: 3             # Number of transformer layers / Transformer 레이어 수
  num_heads: 8         # Attention heads / Attention head 수
  mlp_dim: 1024        # MLP hidden dimension / MLP 숨겨진 차원
```

### Training / 학습
```yaml
training:
  num_epochs: 300      # Maximum epochs / 최대 epoch 수
  patience: 11         # Early stopping patience / Early stopping patience
  save_path: './checkpoints/best_model.pth'
```

### Optimizer / 옵티마이저
```yaml
optimizer:
  lr: 1.0e-4          # Learning rate / 학습률
  weight_decay: 0.01   # Weight decay / Weight decay
```

### Scheduler / 스케줄러
```yaml
scheduler:
  type: 'ReduceLROnPlateau'
  factor: 0.7          # LR reduction factor / LR 감소 비율
  patience: 2          # Epochs before reducing LR / LR 감소 전 대기 epoch
```

---

## 📊 Output / 출력

### Training / 학습 중
```
Epoch   1 | Train Loss: 0.114494 | Val Loss: 0.000540 | LR: 0.0001
  → New best val loss: 0.000540, saved to ./checkpoints/best_model.pth
Epoch   2 | Train Loss: 0.000606 | Val Loss: 0.000398 | LR: 0.0001
  → New best val loss: 0.000398, saved to ./checkpoints/best_model.pth
```

### Evaluation Results / 평가 결과
```
================================================================================
EVALUATION RESULTS
================================================================================
Number of samples: 1500

Regression Metrics:
  MSE:  0.000000
  RMSE: 0.000217
  MAE:  0.000167
  MAPE: 0.17%

Absolute Relative Error Percentiles:
  50th percentile: 0.10%
  68th percentile: 0.16%
  95th percentile: 0.57%
================================================================================
```

### Generated Files / 생성되는 파일
- `checkpoints/best_model.pth` - Best model weights / 최고 성능 모델 가중치
- `results/training_history.png` - Training curves / 학습 곡선
- `results/evaluation_results.png` - Evaluation plots (4 subplots) / 평가 플롯 (4개 서브플롯)
- `results/predictions.csv` - Detailed predictions / 상세 예측 결과

---

## 🔧 Command Line Options / 명령줄 옵션

```bash
python train.py [OPTIONS]

Options / 옵션:
  --config PATH         Configuration file path / 설정 파일 경로 (default: config.yaml)
  --lr FLOAT           Learning rate / 학습률 (overrides config / 설정 덮어쓰기)
  --epochs INT         Number of epochs / Epoch 수 (overrides config)
  --batch_size INT     Batch size / 배치 크기 (overrides config)
  --resume PATH        Checkpoint path to resume / 재개할 체크포인트 경로
  --reset_scheduler    Reset scheduler when resuming / 재개 시 스케줄러 리셋
```

---

## 💡 Tips / 팁

### Memory Issues / 메모리 문제
If you encounter CUDA out of memory errors:
CUDA 메모리 부족 오류가 발생하면:

```yaml
dataloader:
  batch_size: 16      # Reduce batch size / 배치 크기 줄이기
  num_workers: 2      # Reduce workers / 워커 수 줄이기
```

### Fine-tuning / 미세 조정
For fine-tuning, use a lower learning rate:
미세 조정 시 낮은 학습률 사용:

```bash
python train.py --config config.yaml \
    --resume ./checkpoints/best_model.pth \
    --lr 1e-5 \
    --epochs 50
```

### Scheduler Selection / 스케줄러 선택
- **ReduceLROnPlateau**: Reduces LR when validation loss plateaus (recommended)
- **ReduceLROnPlateau**: 검증 손실이 정체될 때 LR 감소 (권장)
- **CosineAnnealingLR**: Smooth cosine decay schedule
- **CosineAnnealingLR**: 부드러운 코사인 감쇠 스케줄

---

## 📖 Documentation / 문서

- **Main README**: This file / 이 파일
- **Examples Guide**: `examples/README.md`
- **Quick Start**: See above / 위 참조
- **API Documentation**: See code docstrings / 코드 docstring 참조

---

## 🐛 Troubleshooting / 문제 해결

### Common Issues / 일반적인 문제

**Q: ImportError for modules / 모듈 import 오류**
```bash
# Make sure you're in the OPTICUS directory
# OPTICUS 디렉토리에 있는지 확인
cd /home/work/CamSim/OPTICUS
python train.py --config config.yaml
```

**Q: Checkpoint size mismatch / 체크포인트 크기 불일치**
```yaml
# Ensure model config matches checkpoint
# 모델 설정이 체크포인트와 일치하는지 확인
model:
  embed_dim: 256  # Must match checkpoint / 체크포인트와 일치해야 함
  mlp_dim: 1024   # Must match checkpoint / 체크포인트와 일치해야 함
```

**Q: Data path not found / 데이터 경로를 찾을 수 없음**
```yaml
# Use absolute paths for data files
# 데이터 파일에 절대 경로 사용
data:
  train_path: '/absolute/path/to/train_dataset.h5'
```

---

## 📄 Requirements / 요구사항

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA-capable GPU (recommended) / CUDA 지원 GPU (권장)
- 8GB+ GPU memory (for batch_size=32) / 8GB+ GPU 메모리 (batch_size=32 기준)

**See `requirements.txt` for full dependencies.**
**전체 의존성은 `requirements.txt` 참조.**

---

## 🙏 Acknowledgments / 감사의 말

This project is part of the IceCube Upgrade Camera System development.
이 프로젝트는 IceCube Upgrade Camera System 개발의 일부입니다.

### Citation / 인용

If you use OPTICUS in your research, please cite:
연구에 OPTICUS를 사용하는 경우 다음을 인용해주세요:

```bibtex
@article{rott2025performance,
  title={Performance Study of the IceCube Upgrade Camera System},
  author={Rott, Carsten and Park, Minje and Jansson, Matti and Iverson, Garrett and Choi, Seowon},
  journal={arXiv preprint arXiv:2507.18525},
  year={2025},
  note={Presented at ICRC2025}
}
```

---

## 📧 Contact / 연락처

For questions or issues, please open an issue on the repository.
질문이나 문제가 있으면 저장소에 이슈를 열어주세요.

For academic inquiries related to the IceCube Upgrade Camera System, please refer to the [paper](https://arxiv.org/abs/2507.18525).
IceCube Upgrade Camera System과 관련된 학술적 문의는 [논문](https://arxiv.org/abs/2507.18525)을 참조하세요.

---

## 📝 License / 라이센스

This project follows the IceCube Collaboration licensing policies.
이 프로젝트는 IceCube Collaboration 라이센스 정책을 따릅니다.
