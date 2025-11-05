
# 전체 파이프라인 구조 및 데이터 흐름도

## 1. 전체 시스템 구조 (High-level)

```mermaid
flowchart TB
    subgraph Input["입력"]
        Camera["카메라<br/>640×480 RGB"]
    end
    
    subgraph Preprocessing["전처리"]
        Undistort["왜곡 보정<br/>(Camera Calibration)"]
    end
    
    subgraph FaceDetection["얼굴 검출 & 랜드마크"]
        MediaPipe["MediaPipe Face Mesh<br/>~3.5M params<br/>478 landmarks"]
    end
    
    subgraph GazeEstimation["시선 추정"]
        direction TB
        HeadPose["머리 자세 추정<br/>(PnP 알고리즘)"]
        EyeNorm["눈 정규화<br/>(36×60 crop)"]
        MPIIGaze["MPIIGaze ResNet-8<br/>77K params"]
        GazeOut["시선 벡터<br/>(pitch, yaw) × 2"]
    end
    
    subgraph MouthEstimation["입술 추정"]
        direction TB
        LipLandmarks["입술 랜드마크<br/>추출 (64점)"]
        MouthReg["MouthRegressor MLP<br/>278 params"]
        MouthOut["입술 상태<br/>(openness, position)"]
    end
    
    subgraph Output["출력"]
        Combine["데이터 결합<br/>(6개 float32)"]
        UDP["UDP 전송<br/>to Unity/Client"]
    end
    
    Camera --> Undistort
    Undistort --> MediaPipe
    MediaPipe --> HeadPose
    MediaPipe --> EyeNorm
    MediaPipe --> LipLandmarks
    HeadPose --> MPIIGaze
    EyeNorm --> MPIIGaze
    MPIIGaze --> GazeOut
    LipLandmarks --> MouthReg
    MouthReg --> MouthOut
    GazeOut --> Combine
    MouthOut --> Combine
    Combine --> UDP
    
    style MediaPipe fill:#ff9999
    style MPIIGaze fill:#99ccff
    style MouthReg fill:#99ff99
```

## 2. MPIIGaze 상세 구조

```mermaid
flowchart TB
    subgraph Input["입력"]
        EyeImg["눈 이미지<br/>(1, 1, 36, 60)"]
        HeadPose["머리 자세<br/>(1, 2)"]
    end
    
    subgraph CNN["CNN Feature Extractor"]
        Conv["Conv2d 3×3<br/>1→16 ch"]
        Stage1["BasicBlock<br/>16→16 ch<br/>stride=1"]
        Stage2["BasicBlock<br/>16→32 ch<br/>stride=2"]
        Stage3["BasicBlock<br/>32→64 ch<br/>stride=2"]
        BN["BatchNorm<br/>+ ReLU"]
        Pool["Adaptive<br/>AvgPool"]
        Flat["Flatten<br/>(64,)"]
    end
    
    subgraph Fusion["Multi-modal Fusion"]
        Cat["Concatenate<br/>(66,)"]
        FC["Linear<br/>66→2"]
    end
    
    subgraph Output["출력"]
        Gaze["시선 각도<br/>(pitch, yaw)"]
    end
    
    EyeImg --> Conv
    Conv --> Stage1
    Stage1 --> Stage2
    Stage2 --> Stage3
    Stage3 --> BN
    BN --> Pool
    Pool --> Flat
    Flat --> Cat
    HeadPose --> Cat
    Cat --> FC
    FC --> Gaze
    
    style Conv fill:#e1f5ff
    style Stage1 fill:#e1f5ff
    style Stage2 fill:#e1f5ff
    style Stage3 fill:#e1f5ff
    style FC fill:#fff9c4
```

## 3. MPIIGaze BasicBlock 구조

```mermaid
flowchart TB
    Input["Input<br/>(C_in, H, W)"]
    
    subgraph PreAct1["Pre-activation"]
        BN1["BatchNorm2d"]
        ReLU1["ReLU"]
    end
    
    Conv1["Conv2d 3×3<br/>C_in → C_out<br/>stride=s"]
    
    subgraph PreAct2["Pre-activation"]
        BN2["BatchNorm2d"]
        ReLU2["ReLU"]
    end
    
    Conv2["Conv2d 3×3<br/>C_out → C_out<br/>stride=1"]
    
    subgraph Shortcut["Shortcut Path"]
        Identity["Identity"]
        Project["1×1 Conv<br/>(if C_in≠C_out)"]
    end
    
    Add["Add<br/>(Residual)"]
    Output["Output<br/>(C_out, H', W')"]
    
    Input --> BN1
    BN1 --> ReLU1
    ReLU1 --> Conv1
    Conv1 --> BN2
    BN2 --> ReLU2
    ReLU2 --> Conv2
    Conv2 --> Add
    
    Input -.-> Identity
    Input -.-> Project
    Identity -.-> Add
    Project -.-> Add
    
    Add --> Output
    
    style Add fill:#ffcccc
    style Identity stroke-dasharray: 5 5
    style Project stroke-dasharray: 5 5
```

## 4. MouthRegressor 상세 구조

```mermaid
flowchart TB
    subgraph Input["입력"]
        Landmarks["랜드마크 좌표<br/>(478×2 from MediaPipe)"]
        Select["선택된 64개 인덱스<br/>(입술 관련)"]
    end
    
    subgraph Preprocessing["전처리"]
        Norm["정규화<br/>(이미지 크기로 나누기)"]
        SelectLM["랜드마크 선택<br/>(64×2 = 128 values)"]
        AddFeatures["특징 추가<br/>(ratio:2 + center:2)"]
        Features["입력 특징<br/>(66,)"]
        Standardize["표준화<br/>(mean, std)"]
    end
    
    subgraph Model["MouthRegressor Model"]
        Backbone["Backbone<br/>Linear(66→4)<br/>+ ReLU + Dropout"]
        Head1["Openness Head<br/>Linear(4→1)<br/>+ Sigmoid"]
        Head2["Lip Position Head<br/>Linear(4→1)<br/>+ Tanh"]
    end
    
    subgraph Output["출력"]
        Open["openness<br/>[0, 1]"]
        LipPos["lip_position<br/>[-1, 1]"]
    end
    
    Landmarks --> Norm
    Select --> SelectLM
    Norm --> SelectLM
    SelectLM --> AddFeatures
    AddFeatures --> Features
    Features --> Standardize
    Standardize --> Backbone
    Backbone --> Head1
    Backbone --> Head2
    Head1 --> Open
    Head2 --> LipPos
    
    style Backbone fill:#c8e6c9
    style Head1 fill:#fff9c4
    style Head2 fill:#fff9c4
```

## 5. 데이터 흐름 (Data Flow with Dimensions)

```mermaid
flowchart LR
    subgraph Frame["프레임 처리"]
        direction TB
        A["카메라<br/>(640, 480, 3)"]
        B["왜곡 보정<br/>(640, 480, 3)"]
    end
    
    subgraph MediaPipe["MediaPipe"]
        direction TB
        C["Face Mesh<br/>478 landmarks<br/>(478, 3)"]
    end
    
    subgraph Gaze["시선 추정 파이프라인"]
        direction TB
        D1["머리 자세<br/>(2,)"]
        D2["좌안 정규화<br/>(1, 36, 60)"]
        D3["우안 정규화<br/>(1, 36, 60)"]
        E["MPIIGaze<br/>CNN"]
        F["좌안 시선<br/>(2,)"]
        G["우안 시선<br/>(2,)"]
    end
    
    subgraph Mouth["입술 추정 파이프라인"]
        direction TB
        H["입술 랜드마크<br/>(64, 2)"]
        I["특징 벡터<br/>(66,)"]
        J["MouthRegressor<br/>MLP"]
        K["입술 상태<br/>(2,)"]
    end
    
    subgraph Out["최종 출력"]
        direction TB
        L["결합<br/>(6,)"]
        M["UDP 패킷<br/>24 bytes"]
    end
    
    A --> B
    B --> C
    C --> D1
    C --> D2
    C --> D3
    C --> H
    D1 --> E
    D2 --> E
    D3 --> E
    E --> F
    E --> G
    H --> I
    I --> J
    J --> K
    F --> L
    G --> L
    K --> L
    L --> M
```

## 6. 성능 비교 (모델 크기)

```mermaid
%%{init: {'theme':'base'}}%%
pie title 모델 파라미터 분포 (MPIIGaze 모드)
    "MediaPipe" : 3500000
    "MPIIGaze" : 77046
    "MouthRegressor" : 278
```

## 7. 처리 시간 분포 (예상)

```mermaid
%%{init: {'theme':'base'}}%%
pie title 추론 시간 분포 (예상, CPU 기준)
    "MediaPipe 랜드마크" : 40
    "MPIIGaze 시선" : 10
    "MouthRegressor 입술" : 1
    "기타 (전처리 등)" : 5
```

---

## 📝 주요 특징 요약

### MediaPipe Face Mesh
- **파라미터**: ~3.5M
- **입력**: RGB 이미지 (640×480)
- **출력**: 478개 3D 랜드마크
- **역할**: 얼굴 전체 랜드마크 추출 (눈, 입술 포함)

### MPIIGaze ResNet-8
- **파라미터**: 77,046
- **입력**: 눈 이미지 (1×36×60) + 머리 자세 (2D)
- **출력**: 시선 각도 (pitch, yaw) × 2 (양쪽 눈)
- **구조**: Pre-activation ResNet, 3-stage

### MouthRegressor MLP
- **파라미터**: 278
- **입력**: 선택된 입술 랜드마크 (64점) + 특징 (ratio, center)
- **출력**: openness [0,1], lip_position [-1,1]
- **구조**: 1-layer MLP with 2 heads

### 전체 시스템
- **총 파라미터**: ~3.58M (MediaPipe가 97.8%)
- **실시간 성능**: 30+ FPS (GPU/DML 사용 시)
- **출력 데이터**: 6개 float32 (24 bytes)
  - 좌안 시선 (2) + 우안 시선 (2) + 입술 (2)
