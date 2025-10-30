# OPTICUS Examples
# 예제 모음

이 디렉토리는 OPTICUS를 사용하는 3가지 주요 시나리오의 예제를 포함합니다.

## 📁 예제 구조

```
examples/
├── 01_initial_training/         # 처음부터 학습
├── 02_resume_training/           # 이어서 학습 (fine-tuning)
├── 03_evaluation_visualization/  # 평가 및 시각화
└── README.md                     # 이 파일
```

## 📚 예제 목록

### 1. Initial Training (처음부터 학습)
**디렉토리**: `01_initial_training/`

새로운 모델을 처음부터 학습하는 예제입니다.

**파일**:
- `config_initial_training.yaml` - 학습 설정 파일
- `train_from_scratch.py` - Python 스크립트
- `train_from_scratch.ipynb` - Jupyter 노트북

**실행 방법**:
```bash
# Python 스크립트로 실행
cd examples/01_initial_training
python train_from_scratch.py

# 또는 Jupyter 노트북 실행
jupyter notebook train_from_scratch.ipynb
```

**주요 내용**:
- 데이터 로딩
- 모델 생성 및 초기화
- 학습 (Early stopping, LR scheduling)
- 테스트 세트 평가
- 결과 시각화

---

### 2. Resume Training (이어서 학습)
**디렉토리**: `02_resume_training/`

기존 체크포인트에서 학습을 이어가는 예제입니다 (fine-tuning).

**파일**:
- `config_resume_training.yaml` - 이어서 학습 설정 파일
- `resume_training.py` - Python 스크립트
- `resume_training.ipynb` - Jupyter 노트북

**실행 방법**:
```bash
# Python 스크립트로 실행
cd examples/02_resume_training
python resume_training.py

# 또는 Jupyter 노트북 실행
jupyter notebook resume_training.ipynb
```

**주요 내용**:
- 체크포인트 로딩
- 낮은 learning rate로 fine-tuning
- 추가 학습 (예: 150 epochs)
- 성능 향상 확인
- 결과 비교

**주의사항**:
- `config_resume_training.yaml`에서 `checkpoint.load_path`를 올바른 경로로 설정하세요
- Learning rate를 낮게 설정하는 것이 fine-tuning에 유리합니다

---

### 3. Evaluation and Visualization (평가 및 시각화)
**디렉토리**: `03_evaluation_visualization/`

학습된 모델을 불러와서 평가하고 결과를 시각화하는 예제입니다.

**파일**:
- `config_evaluation.yaml` - 평가 설정 파일
- `evaluate_and_visualize.py` - Python 스크립트
- `evaluate_and_visualize.ipynb` - Jupyter 노트북

**실행 방법**:
```bash
# Python 스크립트로 실행
cd examples/03_evaluation_visualization
python evaluate_and_visualize.py

# 또는 Jupyter 노트북 실행
jupyter notebook evaluate_and_visualize.ipynb
```

**주요 내용**:
- 학습된 모델 로딩
- Train/Val/Test 중 선택하여 평가
- 상세 메트릭 계산 (MSE, RMSE, MAE, MAPE)
- 4가지 시각화 플롯
- 예측 결과 CSV 저장

**설정 옵션**:
```yaml
evaluation:
  dataset: 'test'  # 'train', 'val', 'test' 중 선택
```

---

## 🚀 빠른 시작

### 1단계: 기본 학습
```bash
cd examples/01_initial_training
python train_from_scratch.py
```

### 2단계: Fine-tuning
```bash
cd ../02_resume_training
# config_resume_training.yaml에서 checkpoint 경로 확인
python resume_training.py
```

### 3단계: 평가
```bash
cd ../03_evaluation_visualization
python evaluate_and_visualize.py
```

---

## ⚙️ 설정 파일 커스터마이징

각 예제의 YAML 파일을 수정하여 설정을 변경할 수 있습니다:

### 공통 설정
```yaml
seed: 14000                    # 랜덤 시드
device: 'cuda'                 # 'cuda' 또는 'cpu'

dataloader:
  batch_size: 32               # 배치 크기
  num_workers: 4               # 워커 수
```

### 학습 설정
```yaml
optimizer:
  lr: 1.0e-4                   # Learning rate
  weight_decay: 0.01           # Weight decay

training:
  num_epochs: 300              # Epoch 수
  patience: 11                 # Early stopping patience
```

### Fine-tuning 설정 (예제 2)
```yaml
optimizer:
  lr: 1.0e-5                   # 낮은 LR

training:
  num_epochs: 150              # 추가 epoch 수
```

---

## 📊 출력 결과

각 예제는 다음과 같은 결과를 생성합니다:

### 체크포인트
- `checkpoints/best_model.pth` - 최고 성능 모델

### 플롯
- `results/training_history.png` - 학습 과정 (loss, LR)
- `results/evaluation_results.png` - 평가 결과 (4가지 플롯)

### CSV (예제 3)
- `results/predictions.csv` - 예측 결과 상세 데이터

---

## 💡 팁

### Python 스크립트 vs Jupyter 노트북
- **Python 스크립트 (`.py`)**: 
  - 빠른 실행
  - 배치 작업에 적합
  - 서버에서 백그라운드 실행 가능
  
- **Jupyter 노트북 (`.ipynb`)**: 
  - 단계별 실행 및 확인
  - 인터랙티브 탐색
  - 중간 결과 시각화

### 경로 설정
- 상대 경로: `'../01_initial_training/checkpoints/best_model.pth'`
- 절대 경로: `'/home/work/CamSim/OPTICUS/checkpoints/best_model.pth'`

### 메모리 부족 시
```yaml
dataloader:
  batch_size: 16  # 배치 크기 줄이기
  num_workers: 2  # 워커 수 줄이기
```

---

## 🔗 관련 문서

- 메인 README: `../../README_refactored.md`
- 빠른 시작 가이드: `../../QUICK_START.md`
- 설정 파일 예제: `../../config.yaml`

---

## ❓ 자주 묻는 질문

**Q: 예제를 내 데이터로 실행하려면?**
A: 각 예제의 `config_*.yaml` 파일에서 `data` 섹션의 경로를 수정하세요.

**Q: GPU 메모리가 부족하다는 에러가 나요**
A: `batch_size`를 줄이거나 `num_workers`를 줄여보세요.

**Q: 체크포인트 경로를 찾을 수 없다는 에러가 나요**
A: 예제 2와 3에서 `checkpoint.load_path`가 올바른지 확인하세요.

**Q: 노트북에서 모듈을 찾을 수 없다는 에러가 나요**
A: 노트북을 해당 예제 디렉토리에서 실행하고 있는지 확인하세요.

