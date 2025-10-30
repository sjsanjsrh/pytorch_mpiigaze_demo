import socket
import struct
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

import ptgaze.utils as utils
from ptgaze.common import Face
from ptgaze.demo import Demo
from ptgaze.gaze_estimator import GazeEstimator
from ptgaze.point.mouth_regression import (
    load_regressor,
    normalise_landmarks,
    prepare_regressor_features,
)
from ptgaze.utils import (
    download_ethxgaze_model,
    download_mpiifacegaze_model,
    download_mpiigaze_model,
)


HOST, PORT = "0.0.0.0", 25500

DEBUG = True

def _get_ddbox_size(face: Face) -> float:
    """
    Calculate the size of the bounding box for the face.
    """
    bbox = face.bbox
    length =bbox[1] - bbox[0]
    return length[0] * length[1]

class RemoteServer:
    def __init__(self, config):
        self.config = config
        utils.expanduser_all(self.config)
        self.gaze_estimator = GazeEstimator(config)
        self.demo = Demo(config)
        self.server_socket = None
        self.udp_clients = []
        self.running = True

        utils.expanduser_all(self.config)

        self._bind_thread = None
        self._cap = None
        self.mouth_device = torch.device("cpu")
        self.mouth_model = None
        self.mouth_mean = None
        self.mouth_std = None
        self.mouth_indices = None
        self.mouth_dims = 2
        self._load_mouth_regressor()

    def _load_mouth_regressor(self) -> None:
        """Initialise mouth regression model if available."""
        default_path = Path("artifacts/models/mouth_regressor.pt")
        configured_path = OmegaConf.select(self.config, "mouth_regressor.model_path")
        model_path = Path(configured_path) if configured_path is not None else default_path
        if not model_path.exists():
            print(f"[WARN] Mouth regressor missing at {model_path}. Using legacy mouth metrics.")
            return

        try:
            self.mouth_model, cfg = load_regressor(model_path, self.mouth_device)
            self.mouth_mean = np.asarray(cfg["mean"], dtype=np.float32)
            self.mouth_std = np.asarray(cfg["std"], dtype=np.float32)
            indices = cfg.get("indices")
            if indices is not None:
                self.mouth_indices = np.asarray(indices, dtype=np.int64)
            self.mouth_dims = int(cfg.get("dims", 2))
            print(f"[INFO] Mouth regressor loaded from {model_path}")
        except Exception as exc:
            self.mouth_model = None
            print(f"[WARN] Failed to load mouth regressor: {exc}. Falling back to legacy metrics.")
    
    def send_command(self, command: str, addr):
        """
        Send a command to a specific client.
        """
        if self.server_socket is not None:
            data = struct.pack('f',float('NaN')) + command.encode()
            self.server_socket.sendto(data, addr)
            print(f"Sent command '{command}' to {addr}")
        else:
            print("Server socket is not initialized.")

    def client_bind_thread(self):
        while self.running:
            try:
                msg, client_addr = self.server_socket.recvfrom(1024)
                if client_addr not in self.udp_clients:
                    self.udp_clients.append(client_addr)
                    print(f"New client connected: {client_addr}")
                    self.send_command("bind", client_addr)
            except Exception:
                continue

    def run(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server_socket.bind((HOST, PORT))

        # 클라이언트 바인드 쓰레드 시작
        self._bind_thread = threading.Thread(target=self.client_bind_thread, daemon=True)
        self._bind_thread.start()

        self._cap = cv2.VideoCapture(0)

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.gaze_estimator.camera.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.gaze_estimator.camera.height)
        
        print("Server started ")
        
        # FPS 측정 변수
        frame_count = 0
        fps_update_time = time.time()
        fps = 0
        inference_count = 0
        
        try:
            while True:
                ok, frame = self._cap.read()
                if not ok:
                    break

                undistorted = cv2.undistort(
                    frame, self.gaze_estimator.camera.camera_matrix,
                    self.gaze_estimator.camera.dist_coefficients)

                faces = self.gaze_estimator.detect_faces(undistorted)

                face = None
                max_size = 0
                for f in faces:
                    size = _get_ddbox_size(f)
                    if size > max_size:
                        max_size = size
                        face = f
                
                # FPS 계산
                frame_count += 1
                current_time = time.time()
                if current_time - fps_update_time >= 1.0:
                    fps = frame_count / (current_time - fps_update_time)
                    print(f"FPS: {fps:.1f} | Inferences: {inference_count}")
                    frame_count = 0
                    fps_update_time = current_time
                
                if face is None:
                    continue

                
                if DEBUG:
                    cv2.rectangle(
                        undistorted, 
                        tuple(face.bbox[0].astype(int)), 
                        tuple(face.bbox[1].astype(int)), 
                        (0, 255, 0), 2)
                    flip = cv2.flip(undistorted, 1)
                    cv2.imshow("Remote Server", flip)
                    cv2.waitKey(1)

                # GPU 추론 시작
                inference_count += 1
                self.gaze_estimator.estimate_gaze(undistorted, face)

                data = []
                if self.config.mode == 'MPIIGaze':
                    # MPIIGaze: 각 눈마다 시선 벡터가 있음
                    for key in self.gaze_estimator.EYE_KEYS:
                        eye = getattr(face, key.name.lower())
                        data.extend(eye.normalized_gaze_vector)
                else:
                    # ETH-XGaze, MPIIFaceGaze: 얼굴 전체의 시선 벡터 사용
                    data.extend(face.normalized_gaze_vector)
                    # 양쪽 눈에 같은 값 사용 (호환성을 위해)
                    data.extend(face.normalized_gaze_vector)
                
                mouth_values = self._infer_mouth_outputs(face, undistorted.shape[1], undistorted.shape[0])
                if mouth_values is None:
                    mouth_values = (0.0, 0.0)
                data.extend(mouth_values)
                data = np.array(data, dtype=np.float32)

                data = data.astype(np.float32).tobytes()

                # 모든 UDP 클라이언트에게 데이터 전송
                for client_addr in self.udp_clients:
                    self.server_socket.sendto(data, client_addr)
        
        except KeyboardInterrupt:
            print("Server stopped by user.")
        
        finally:
            self.stop()

    def stop(self):
        if self._cap is not None:
            print("Releasing video capture...")
            self._cap.release()
        if self.server_socket is not None:
            self.server_socket.close()
        self.running = False
        print("Server closed.")
        if self._bind_thread is not None:
            self._bind_thread.join()
            print("Bind thread closed.")
        for client_addr in self.udp_clients:
            self.send_command("exit", client_addr)
        print("Exit message sent to all clients.")
        self.udp_clients.clear()
        print("All clients cleared.")
        self.server_socket = None
        print("Server socket set to None.")

    def _infer_mouth_outputs(self, face: Face, width: int, height: int):
        """Run mouth regression if the model is loaded."""
        if self.mouth_model is None:
            return None
        try:
            coords_norm = normalise_landmarks(face.landmarks, width, height, self.mouth_dims)
            features, _, _ = prepare_regressor_features(
                coords_norm,
                self.mouth_dims,
                self.mouth_indices,
                self.mouth_mean,
                self.mouth_std,
            )
            tensor = torch.from_numpy(features).unsqueeze(0).to(self.mouth_device)
            with torch.no_grad():
                openness, lip_position = self.mouth_model(tensor)
            return float(openness.item()), float(lip_position.item())
        except Exception as exc:
            print(f"[WARN] Mouth regression failed: {exc}")
            return None

if __name__ == "__main__":

    # config = OmegaConf.load('./ptgaze/data/configs/edge.yaml')
    config = OmegaConf.load('./ptgaze/data/configs/edge_xgaze.yaml')
    
    # 설정에 따라 자동으로 모델 다운로드
    if config.mode == 'MPIIGaze':
        download_mpiigaze_model()
    elif config.mode == 'MPIIFaceGaze':
        download_mpiifacegaze_model()
    elif config.mode == 'ETH-XGaze':
        download_ethxgaze_model()
    
    # 디바이스 정보 출력
    print(f"\n{'='*50}")
    print(f"Mode: {config.mode}")
    print(f"Device: {config.device}")
    if config.device == 'dml':
        try:
            import torch_directml
            device_count = torch_directml.device_count()
            dml_device = torch_directml.device()
            print(f"DirectML Device: {dml_device}")
            print(f"DirectML Device Count: {device_count}")
            for i in range(device_count):
                try:
                    name = torch_directml.device_name(i)
                    marker = " (Using)" if i == 0 else ""
                    print(f"  Device {i}: {name}{marker}")
                except:
                    print(f"  Device {i}: (Unknown)")
        except Exception as e:
            print(f"DirectML info error: {e}")
    elif config.device == 'cuda':
        import torch
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"{'='*50}\n")

    server = RemoteServer(config)
    server.run()