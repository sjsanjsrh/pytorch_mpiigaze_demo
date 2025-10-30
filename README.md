# ptgaze 설치 및 사용법

## 1. 아나콘다 환경 생성
아나콘다 설치 후,
Anaconda Prompt 실행  *시작 프로그렘 검색*
~~PATH 에 conda 설정될경우 일반 cmd에서 해도됨~~

### 1-1. 새로운 가상환경 생성 (Python 3.10 권장)
`conda create -n ptgaze_env python=3.10 -y`

### 1-2. 가상환경 활성화
`conda activate ptgaze_env`

## 2. 저장소 클론 & 라이브러리 설치

### 2-1. GitHub 저장소 클론 *numpy type문제 해결된 소스임!!*
`git clone https://github.com/sjsanjsrh/pytorch_mpiigaze/`
`cd pytorch_mpiigaze_demo`

### 2-2. 필요한 패키지 설치
`pip install -r requirements.txt`  # 의존성 패키지 설치
`python ./setup.py install`  # ptgaze 설치 *현제 소스에서 설치하여야함!!*

## 3. 실시간 시선 추적 실행
`python ./run.py` # 해당 파일 안에서 실행인수 설정 가능

## 4. 입모양 분석 모델 학습 및 실시간 실행
```
nalysis/landmark_feature_selection.py `
  --dataset data/mouth_landmark_dataset.npz `
  --keep-top 200 `
  --out-prefix artifacts/landmark_selection/mouth_landmarks
```

```
train_mouth_regressor.py --dataset data/mouth_landmark_dataset.npz --indices artifacts/landmark_selection/mouth_landmarks_selected_indices.npy --out artifacts/models/mouth_regressor.pt --epochs 1600 --batch-size 32 --hidden-dim 4 --lr 1e-3
```

```
mouth_regressor_live.py `
  --model artifacts/models/mouth_regressor.pt `
  --draw-mesh `
  --draw-selected
```


# A demo program of gaze estimation models (MPIIGaze, MPIIFaceGaze, ETH-XGaze)

[![PyPI version](https://badge.fury.io/py/ptgaze.svg)](https://pypi.org/project/ptgaze/)
[![Downloads](https://pepy.tech/badge/ptgaze)](https://pepy.tech/project/ptgaze)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hysts/pytorch_mpiigaze_demo/blob/master/demo.ipynb)
[![MIT License](https://img.shields.io/badge/license-MIT-green)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/hysts/pytorch_mpiigaze_demo.svg?style=flat-square&logo=github&label=Stars&logoColor=white)](https://github.com/hysts/pytorch_mpiigaze_demo)

With this program, you can run gaze estimation on images and videos.
By default, the video from a webcam will be used.

![ETH-XGaze video01 result](https://raw.githubusercontent.com/hysts/pytorch_mpiigaze_demo/master/assets/results/eth-xgaze_video01.gif)
![ETH-XGaze video02 result](https://raw.githubusercontent.com/hysts/pytorch_mpiigaze_demo/master/assets/results/eth-xgaze_video02.gif)
![ETH-XGaze video03 result](https://raw.githubusercontent.com/hysts/pytorch_mpiigaze_demo/master/assets/results/eth-xgaze_video03.gif)

![MPIIGaze video00 result](https://raw.githubusercontent.com/hysts/pytorch_mpiigaze_demo/master/assets/results/mpiigaze_video00.gif)
![MPIIFaceGaze video00 result](https://raw.githubusercontent.com/hysts/pytorch_mpiigaze_demo/master/assets/results/mpiifacegaze_video00.gif)

![MPIIGaze image00 result](https://raw.githubusercontent.com/hysts/pytorch_mpiigaze_demo/master/assets/results/mpiigaze_image00.jpg)

To train a model for MPIIGaze and MPIIFaceGaze,
use [this repository](https://github.com/hysts/pytorch_mpiigaze).
You can also use [this repo](https://github.com/hysts/pl_gaze_estimation)
to train a model with ETH-XGaze dataset.

## Quick start

This program is tested only on Ubuntu.

### Installation

```bash
pip install ptgaze
```


### Run demo

```bash
ptgaze --mode eth-xgaze
```


### Usage


```
usage: ptgaze [-h] [--config CONFIG] [--mode {mpiigaze,mpiifacegaze,eth-xgaze}]
              [--face-detector {dlib,face_alignment_dlib,face_alignment_sfd,mediapipe}]
              [--device {cpu,cuda}] [--image IMAGE] [--video VIDEO] [--camera CAMERA]
              [--output-dir OUTPUT_DIR] [--ext {avi,mp4}] [--no-screen] [--debug]

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       Config file. When using a config file, all the other commandline arguments
                        are ignored. See
                        https://github.com/hysts/pytorch_mpiigaze_demo/ptgaze/data/configs/eth-
                        xgaze.yaml
  --mode {mpiigaze,mpiifacegaze,eth-xgaze}
                        With 'mpiigaze', MPIIGaze model will be used. With 'mpiifacegaze',
                        MPIIFaceGaze model will be used. With 'eth-xgaze', ETH-XGaze model will be
                        used.
  --face-detector {dlib,face_alignment_dlib,face_alignment_sfd,mediapipe}
                        The method used to detect faces and find face landmarks (default:
                        'mediapipe')
  --device {cpu,cuda}   Device used for model inference.
  --image IMAGE         Path to an input image file.
  --video VIDEO         Path to an input video file.
  --camera CAMERA       Camera calibration file. See https://github.com/hysts/pytorch_mpiigaze_demo/
                        ptgaze/data/calib/sample_params.yaml
  --output-dir OUTPUT_DIR, -o OUTPUT_DIR
                        If specified, the overlaid video will be saved to this directory.
  --ext {avi,mp4}, -e {avi,mp4}
                        Output video file extension.
  --no-screen           If specified, the video is not displayed on screen, and saved to the output
                        directory.
  --debug
```

While processing an image or video, press the following keys on the window
to show or hide intermediate results:

- `l`: landmarks
- `h`: head pose
- `t`: projected points of 3D face model
- `b`: face bounding box


## References

- Zhang, Xucong, Seonwook Park, Thabo Beeler, Derek Bradley, Siyu Tang, and Otmar Hilliges. "ETH-XGaze: A Large Scale Dataset for Gaze Estimation under Extreme Head Pose and Gaze Variation." In European Conference on Computer Vision (ECCV), 2020. [arXiv:2007.15837](https://arxiv.org/abs/2007.15837), [Project Page](https://ait.ethz.ch/projects/2020/ETH-XGaze/), [GitHub](https://github.com/xucong-zhang/ETH-XGaze)
- Zhang, Xucong, Yusuke Sugano, Mario Fritz, and Andreas Bulling. "Appearance-based Gaze Estimation in the Wild." Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015. [arXiv:1504.02863](https://arxiv.org/abs/1504.02863), [Project Page](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild/)
- Zhang, Xucong, Yusuke Sugano, Mario Fritz, and Andreas Bulling. "It's Written All Over Your Face: Full-Face Appearance-Based Gaze Estimation." Proc. of the IEEE Conference on Computer Vision and Pattern Recognition Workshops(CVPRW), 2017. [arXiv:1611.08860](https://arxiv.org/abs/1611.08860), [Project Page](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/its-written-all-over-your-face-full-face-appearance-based-gaze-estimation/)
- Zhang, Xucong, Yusuke Sugano, Mario Fritz, and Andreas Bulling. "MPIIGaze: Real-World Dataset and Deep Appearance-Based Gaze Estimation." IEEE transactions on pattern analysis and machine intelligence 41 (2017). [arXiv:1711.09017](https://arxiv.org/abs/1711.09017)
- Zhang, Xucong, Yusuke Sugano, and Andreas Bulling. "Evaluation of Appearance-Based Methods and Implications for Gaze-Based Applications." Proc. ACM SIGCHI Conference on Human Factors in Computing Systems (CHI), 2019. [arXiv](https://arxiv.org/abs/1901.10906), [code](https://git.hcics.simtech.uni-stuttgart.de/public-projects/opengaze)
