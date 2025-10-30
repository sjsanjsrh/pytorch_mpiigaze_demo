import argparse
from pathlib import Path
from typing import Iterable

import cv2
import mediapipe as mp
import numpy as np
import torch

from ptgaze.point.mouth_regression import (
    load_regressor,
    prepare_regressor_features,
)


def unique_points(indices: np.ndarray | None, dims: int, total: int) -> Iterable[int]:
    if indices is None:
        return []
    pts = np.unique(indices // max(dims, 1))
    return [int(p) for p in pts if 0 <= p < total]


def draw_gauge(frame, x: int, y: int, width: int, height: int, value: float, 
               min_val: float, max_val: float, label: str, color=(0, 255, 0)):
    """Draw a horizontal gauge bar."""
    # Background
    cv2.rectangle(frame, (x, y), (x + width, y + height), (60, 60, 60), -1)
    
    # Value bar
    normalized = (value - min_val) / (max_val - min_val)
    normalized = max(0.0, min(1.0, normalized))
    fill_width = int(width * normalized)
    if fill_width > 0:
        cv2.rectangle(frame, (x, y), (x + fill_width, y + height), color, -1)
    
    # Border
    cv2.rectangle(frame, (x, y), (x + width, y + height), (200, 200, 200), 1)
    
    # Label and value
    text = f"{label}: {value:.3f}"
    cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def get_state_label(openness: float, lip_position: float, 
                    open_threshold: float = 0.5,
                    lip_threshold: float = 0.3) -> str:
    """Convert continuous values to discrete state label.
    
    Note: Due to training data characteristics (lip_lower samples had
    slightly open mouths), predictions may be biased towards positive values.
    Adjust thresholds accordingly.
    """
    if openness >= open_threshold:
        return "mouth_open"
    # Asymmetric thresholds to handle data bias
    elif lip_position >= lip_threshold:
        return "lip_raise"
    elif lip_position <= -lip_threshold * 0.5:  # Lower threshold for lip_lower
        return "lip_lower"
    else:
        return "mouth_closed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live mouth regressor inference")
    parser.add_argument("--model", type=Path, required=True, help="Trained regressor model (.pt)")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--width", type=int, default=640, help="Capture width")
    parser.add_argument("--height", type=int, default=480, help="Capture height")
    parser.add_argument("--mirror", action="store_true", help="Mirror preview horizontally")
    parser.add_argument("--draw-mesh", action="store_true", help="Draw FaceMesh tessellation")
    parser.add_argument("--draw-selected", action="store_true", help="Highlight selected landmarks")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    
    model, config = load_regressor(args.model, device)
    mean = np.asarray(config["mean"], dtype=np.float32)
    std = np.asarray(config["std"], dtype=np.float32)
    indices = config.get("indices")
    if indices is not None:
        indices = np.asarray(indices, dtype=np.int64)
    dims = int(config["dims"])
    print(f"[INFO] Model loaded from {args.model}")
    
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    drawing_utils = mp.solutions.drawing_utils
    drawing_spec = drawing_utils.DrawingSpec(color=(80, 255, 80), thickness=1, circle_radius=1)
    
    with mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Failed to read frame")
                break
            
            if args.mirror:
                frame = cv2.flip(frame, 1)
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            
            openness = 0.0
            lip_position = 0.0
            ratio = 0.0
            state_label = "no_face"
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                coords_norm = np.array(
                    [[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark], dtype=np.float32
                )
                
                try:
                    h, w, _ = frame.shape
                    features, ratio, center = prepare_regressor_features(
                        coords_norm,
                        dims,
                        indices,
                        mean,
                        std,
                    )
                    tensor = torch.from_numpy(features).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        openness_pred, lip_pred = model(tensor)
                        openness = float(openness_pred.cpu().numpy()[0, 0])
                        lip_position = float(lip_pred.cpu().numpy()[0, 0])
                    
                    state_label = get_state_label(openness, lip_position)
                    
                    if args.draw_mesh:
                        drawing_utils.draw_landmarks(
                            frame,
                            face_landmarks,
                            mp.solutions.face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=drawing_spec,
                        )
                    
                    if args.draw_selected and indices is not None:
                        h, w, _ = frame.shape
                        coords_xy = coords_norm[:, :2] * np.array([w, h], dtype=np.float32)
                        for idx in unique_points(indices, dims, coords_xy.shape[0]):
                            pt = coords_xy[idx]
                            cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, (0, 0, 255), -1)
                
                except Exception as exc:
                    state_label = "error"
                    print(f"[WARN] Inference failed: {exc}")
                
                # Draw info panel
                panel_x, panel_y = 20, 20
                panel_width = 400
                cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + 200), (0, 0, 0), -1)
                
                # State label
                cv2.putText(frame, f"State: {state_label}", (panel_x + 10, panel_y + 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Openness gauge
                draw_gauge(frame, panel_x + 10, panel_y + 60, 300, 25, openness, 0.0, 1.0, 
                          "Openness", color=(30, 180, 255))
                
                # Lip position gauge (centered at 0)
                draw_gauge(frame, panel_x + 10, panel_y + 110, 300, 25, lip_position, -1.0, 1.0,
                          "Lip Position", color=(255, 180, 30))
                
                # Center line for lip position
                center_x = panel_x + 10 + 150  # Middle of gauge
                cv2.line(frame, (center_x, panel_y + 110), (center_x, panel_y + 135), (255, 255, 255), 1)
                
                # Ratio info
                cv2.putText(frame, f"Ratio: {ratio:.3f}", (panel_x + 10, panel_y + 170),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            else:
                cv2.putText(frame, "No face", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            cv2.imshow("Mouth Regressor Inference", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
