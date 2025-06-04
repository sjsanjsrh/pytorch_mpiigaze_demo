import socket
import struct
import numpy as np
from omegaconf import OmegaConf
from ptgaze.gaze_estimator import GazeEstimator
import ptgaze.utils as utils
from ptgaze.demo import Demo
import cv2
from ptgaze.common import Face
import threading
from ptgaze.point.mouth_open import mouth_open_ratio


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
    def __init__(self, tflite_path, config):
        self.tflite_path = utils._expanduser(tflite_path)
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

                self.gaze_estimator.estimate_gaze(undistorted, face)

                data = []
                for key in self.gaze_estimator.EYE_KEYS:
                    eye = getattr(face, key.name.lower())
                    data.extend(eye.normalized_gaze_vector)
                data.append(mouth_open_ratio(face))
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

if __name__ == "__main__":

    config = OmegaConf.load('./ptgaze/data/configs/edge.yaml')
    tflite_path = '~/.ptgaze/models/mpiigaze.tflite'

    server = RemoteServer(tflite_path, config)
    server.run()