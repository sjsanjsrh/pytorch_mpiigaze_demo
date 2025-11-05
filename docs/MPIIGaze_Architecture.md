# MPIIGaze 모델 구조 상세 설명

## 📐 전체 아키텍처

```
입력: 눈 이미지 (1×36×60) + 머리 자세 (2D)
  ↓
[Pre-activation ResNet-8]
  ↓
출력: 시선 방향 (pitch, yaw)
```

---

## 🏗️ 상세 구조

### 1. 입력 (Input)
- **이미지**: `(batch, 1, 36, 60)` - grayscale 정규화된 눈 이미지
- **Head pose**: `(batch, 2)` - 2D 머리 자세 벡터 (pitch, yaw)

### 2. 컨볼루션 특징 추출 (Feature Extractor)

```
Input (1×36×60)
    ↓
[Conv2d] kernel=3×3, stride=1, padding=1
    1 → 16 channels
    output: (16×36×60)
    ↓
[Stage 1] BasicBlock × 1
    16 → 16 channels, stride=1
    output: (16×36×60)
    ↓
[Stage 2] BasicBlock × 1
    16 → 32 channels, stride=2  ← downsampling
    output: (32×18×30)
    ↓
[Stage 3] BasicBlock × 1
    32 → 64 channels, stride=2  ← downsampling
    output: (64×9×15)
    ↓
[BatchNorm + ReLU]
    ↓
[Adaptive Average Pooling] → (64×1×1)
    ↓
[Flatten] → (64,)
```

### 3. BasicBlock 구조 (Pre-activation ResNet Block)

```
Input (C_in)
    ↓
[BatchNorm] → [ReLU] ← pre-activation
    ↓
[Conv2d] 3×3, stride=s
    C_in → C_out
    ↓
[BatchNorm] → [ReLU]
    ↓
[Conv2d] 3×3, stride=1
    C_out → C_out
    ↓
[Add shortcut]  ← residual connection
    |                    |
    └──[1×1 Conv if C_in≠C_out]
    ↓
Output (C_out)
```

**특징**:
- **Pre-activation**: BN → ReLU → Conv (일반 ResNet은 Conv → BN → ReLU)
- **Shortcut**: 채널 수가 다를 때만 1×1 conv로 조정
- **No bias**: Conv2d에서 bias=False (BatchNorm이 bias 역할)

### 4. 최종 분류 레이어 (Classifier)

```
CNN features (64,) + Head pose (2,) 
    ↓
[Concatenate] → (66,)
    ↓
[Linear(66 → 2)]
    ↓
Output: (pitch, yaw) ← 시선 각도
```

---

## 📊 파라미터 상세

| 레이어 | 입력 shape | 출력 shape | 파라미터 수 |
|--------|-----------|-----------|------------|
| Conv (initial) | 1×36×60 | 16×36×60 | 144 |
| Stage1 (16→16) | 16×36×60 | 16×36×60 | 4,672 |
| Stage2 (16→32) | 16×36×60 | 32×18×30 | 14,432 |
| Stage3 (32→64) | 32×18×30 | 64×9×15 | 57,536 |
| BatchNorm | 64×9×15 | 64×9×15 | 128 |
| Adaptive Pool | 64×9×15 | 64×1×1 | 0 |
| FC | 66 | 2 | 134 |
| **Total** | | | **77,046** |

---

## 🎯 설계 포인트

### 1. 경량화 전략
- **Depth=8**: 매우 얕은 네트워크 (일반 ResNet은 18, 34, 50 등)
- **Base channels=16**: 작은 채널 수 (일반적으로 64 시작)
- **작은 입력**: 36×60 (224×224보다 훨씬 작음)
→ 결과: 77K 파라미터, 실시간 처리 가능

### 2. Pre-activation 사용 이유
```
일반 ResNet:     Conv → BN → ReLU → Conv → BN → Add → ReLU
Pre-activation:  BN → ReLU → Conv → BN → ReLU → Conv → Add
```
- 더 부드러운 그래디언트 흐름
- 정규화 효과 향상
- 작은 네트워크에서 성능 개선

### 3. Multi-modal 입력
- **CNN features**: 눈 이미지에서 추출한 시각적 특징
- **Head pose**: 머리 방향 정보 (보정 역할)
- 두 정보를 concatenate해서 최종 예측 → 정확도 향상

### 4. Stage별 다운샘플링
```
Stage1: 36×60 (stride=1) → 36×60  [같은 해상도, 특징 추출]
Stage2: 36×60 (stride=2) → 18×30  [1/4 크기, 중간 특징]
Stage3: 18×30 (stride=2) → 9×15   [1/16 크기, 고수준 특징]
```

---

## 🔄 Forward Pass 예시

```python
# 입력
eye_image = torch.randn(1, 1, 36, 60)      # 눈 이미지
head_pose = torch.randn(1, 2)              # 머리 자세

# Forward
conv_features = model._forward_conv(eye_image)  # (1, 64, 1, 1)
conv_features = conv_features.view(1, -1)       # (1, 64)
combined = torch.cat([conv_features, head_pose], dim=1)  # (1, 66)
gaze = model.fc(combined)                       # (1, 2) → (pitch, yaw)
```

---

## 📈 비교: MPIIGaze vs MPIIFaceGaze

| 특징 | MPIIGaze | MPIIFaceGaze |
|------|----------|--------------|
| 입력 크기 | 1×36×60 (눈) | 3×224×224 (얼굴) |
| 아키텍처 | ResNet-8 | ResNet-18 backbone |
| 파라미터 | 77K | 2.88M |
| 정확도 | 보통 | 높음 |
| 속도 | 매우 빠름 | 느림 |
| 용도 | 실시간, 경량 장치 | 고정밀 애플리케이션 |

---

## 💡 핵심 요약

1. **초경량**: 77K 파라미터로 실시간 처리
2. **Pre-activation ResNet**: 작은 네트워크에서도 안정적 학습
3. **Multi-modal**: 눈 이미지 + 머리 자세 정보 결합
4. **3-stage 구조**: 점진적 다운샘플링으로 계층적 특징 추출
5. **간단한 출력**: Linear 레이어 하나로 시선 각도 직접 예측

이 구조는 **정확도와 속도의 균형**을 잘 맞춘 설계입니다!
