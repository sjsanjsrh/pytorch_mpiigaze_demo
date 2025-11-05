"""각 추론 단계별 성능 벤치마크 스크립트"""
import sys
import time
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

from ptgaze.gaze_estimator import GazeEstimator
from ptgaze.point.mouth_regression import load_regressor, normalise_landmarks, prepare_regressor_features


def benchmark_pipeline(config_path: str, num_frames: int = 100):
    """전체 파이프라인의 각 단계별 성능 측정"""
    
    config = OmegaConf.load(config_path)
    
    print("=" * 70)
    print(f"Pipeline Benchmark - {config.mode}")
    print(f"Device: {config.device}")
    print("=" * 70)
    
    # Initialize components
    print("\n[1/4] Initializing GazeEstimator...")
    gaze_estimator = GazeEstimator(config)
    
    print("[2/4] Loading mouth regressor...")
    mouth_device = torch.device("cpu")
    mouth_model_path = Path("artifacts/models/mouth_regressor.pt")
    mouth_model = None
    if mouth_model_path.exists():
        mouth_model, mouth_cfg = load_regressor(mouth_model_path, mouth_device)
        mouth_mean = np.asarray(mouth_cfg["mean"], dtype=np.float32)
        mouth_std = np.asarray(mouth_cfg["std"], dtype=np.float32)
        mouth_indices = None
        indices = mouth_cfg.get("indices")
        if indices is not None:
            mouth_indices = np.asarray(indices, dtype=np.int64)
        mouth_dims = int(mouth_cfg.get("dims", 2))
        print(f"   Mouth model loaded: {mouth_model_path}")
    else:
        print(f"   Mouth model not found, skipping...")
    
    print("[3/4] Opening camera...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, gaze_estimator.camera.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, gaze_estimator.camera.height)
    
    # Warm-up
    print("[4/4] Warming up (10 frames)...")
    for _ in range(10):
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            cap.release()
            return
        
        undistorted = cv2.undistort(
            frame, gaze_estimator.camera.camera_matrix,
            gaze_estimator.camera.dist_coefficients)
        faces = gaze_estimator.detect_faces(undistorted)
        if faces:
            face = faces[0]
            gaze_estimator.estimate_gaze(undistorted, face)
            if mouth_model:
                coords_norm = normalise_landmarks(face.landmarks, undistorted.shape[1], undistorted.shape[0], mouth_dims)
                features, _, _ = prepare_regressor_features(coords_norm, mouth_dims, mouth_indices, mouth_mean, mouth_std)
                tensor = torch.from_numpy(features).unsqueeze(0).to(mouth_device)
                with torch.no_grad():
                    _, _ = mouth_model(tensor)
    
    # Benchmark
    print(f"\n{'='*70}")
    print(f"Running benchmark: {num_frames} frames")
    print(f"{'='*70}\n")
    
    timings = {
        'capture': [],
        'undistort': [],
        'detect_faces': [],
        'estimate_gaze': [],
        'mouth_regression': [],
        'total': []
    }
    
    valid_frames = 0
    
    for i in range(num_frames):
        t_start = time.perf_counter()
        
        # 1. Capture
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            continue
        t1 = time.perf_counter()
        timings['capture'].append((t1 - t0) * 1000)
        
        # 2. Undistort
        t0 = time.perf_counter()
        undistorted = cv2.undistort(
            frame, gaze_estimator.camera.camera_matrix,
            gaze_estimator.camera.dist_coefficients)
        t1 = time.perf_counter()
        timings['undistort'].append((t1 - t0) * 1000)
        
        # 3. Face detection (MediaPipe landmark extraction)
        t0 = time.perf_counter()
        faces = gaze_estimator.detect_faces(undistorted)
        t1 = time.perf_counter()
        timings['detect_faces'].append((t1 - t0) * 1000)
        
        if not faces:
            continue
        
        face = faces[0]
        valid_frames += 1
        
        # 4. Gaze estimation
        t0 = time.perf_counter()
        gaze_estimator.estimate_gaze(undistorted, face)
        t1 = time.perf_counter()
        timings['estimate_gaze'].append((t1 - t0) * 1000)
        
        # 5. Mouth regression
        if mouth_model:
            t0 = time.perf_counter()
            coords_norm = normalise_landmarks(face.landmarks, undistorted.shape[1], undistorted.shape[0], mouth_dims)
            features, _, _ = prepare_regressor_features(coords_norm, mouth_dims, mouth_indices, mouth_mean, mouth_std)
            tensor = torch.from_numpy(features).unsqueeze(0).to(mouth_device)
            with torch.no_grad():
                _, _ = mouth_model(tensor)
            t1 = time.perf_counter()
            timings['mouth_regression'].append((t1 - t0) * 1000)
        
        t_end = time.perf_counter()
        timings['total'].append((t_end - t_start) * 1000)
        
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{num_frames} frames ({valid_frames} with faces detected)")
    
    cap.release()
    
    # Results
    print(f"\n{'='*70}")
    print("BENCHMARK RESULTS")
    print(f"{'='*70}")
    print(f"Valid frames (with face): {valid_frames}/{num_frames}\n")
    
    def print_stats(name, times):
        if not times:
            print(f"{name:20s}: N/A")
            return
        arr = np.array(times)
        print(f"{name:20s}: {arr.mean():6.2f} ms (±{arr.std():5.2f}) | "
              f"min: {arr.min():6.2f} | max: {arr.max():6.2f} | "
              f"p95: {np.percentile(arr, 95):6.2f}")
    
    print_stats("Capture", timings['capture'])
    print_stats("Undistort", timings['undistort'])
    print_stats("Landmark Detection", timings['detect_faces'])
    print_stats("Gaze Estimation", timings['estimate_gaze'])
    if mouth_model:
        print_stats("Mouth Regression", timings['mouth_regression'])
    print("-" * 70)
    print_stats("TOTAL (per frame)", timings['total'])
    
    if timings['total']:
        avg_total = np.mean(timings['total'])
        est_fps = 1000.0 / avg_total
        print(f"\nEstimated FPS: {est_fps:.1f}")
    
    # Breakdown percentage
    if timings['detect_faces'] and timings['total']:
        landmark_pct = np.mean(timings['detect_faces']) / np.mean(timings['total']) * 100
        gaze_pct = np.mean(timings['estimate_gaze']) / np.mean(timings['total']) * 100
        mouth_pct = 0
        if mouth_model and timings['mouth_regression']:
            mouth_pct = np.mean(timings['mouth_regression']) / np.mean(timings['total']) * 100
        other_pct = 100 - landmark_pct - gaze_pct - mouth_pct
        
        print(f"\n{'='*70}")
        print("TIME BREAKDOWN (% of total)")
        print(f"{'='*70}")
        print(f"Landmark Detection:  {landmark_pct:5.1f}%")
        print(f"Gaze Estimation:     {gaze_pct:5.1f}%")
        if mouth_model:
            print(f"Mouth Regression:    {mouth_pct:5.1f}%")
        print(f"Other (capture etc): {other_pct:5.1f}%")
    
    print(f"{'='*70}\n")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./ptgaze/data/configs/edge.yaml', help='Config file path')
    parser.add_argument('--frames', type=int, default=100, help='Number of frames to benchmark')
    args = parser.parse_args()
    
    benchmark_pipeline(args.config, args.frames)
