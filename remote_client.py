import socket
import struct
import numpy as np
import cv2
import time
from ptgaze.point.calibration import Calibration
import pyautogui
from ptgaze.common import Face, FacePartsName

# 서버 주소와 포트
SERVER_IP = '127.0.0.1'  # 예: '192.168.0.10'
SERVER_PORT = 25500

# 화면 해상도
SCREEN_WIDTH = pyautogui.size().width
SCREEN_HEIGHT = pyautogui.size().height

NOISE_R = 5
NOISE_Q = 1e-9

# 캘리브레이션 점
calibration_points = [
    np.array((SCREEN_WIDTH / 10, SCREEN_HEIGHT / 10), dtype=np.float32),
    np.array((SCREEN_WIDTH * 9 / 10, SCREEN_HEIGHT / 10), dtype=np.float32),
    np.array((SCREEN_WIDTH * 9 / 10, SCREEN_HEIGHT * 9 / 10), dtype=np.float32),
    np.array((SCREEN_WIDTH / 10, SCREEN_HEIGHT * 9 / 10), dtype=np.float32)
]

def recv_gaze_vector(sock):
    data, _ = sock.recvfrom(1024)
    if len(data) < 24:
        return None
    gaze = np.frombuffer(data[:24], dtype=np.float32)
    # 오른쪽, 왼쪽 눈 각각 3차원 벡터
    reye = gaze[:3]
    leye = gaze[3:6]
    return reye, leye

def show_fullscreen_image(name, img):
    cv2.namedWindow(name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(name, img)

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(5)
    sock.sendto(b'bind', (SERVER_IP, SERVER_PORT))  # 서버에 바인드 요청

    while True:
        data, _ = sock.recvfrom(1024)
        if data[4:].decode() == 'bind':
            break
        time.sleep(0.1)

    calibration = Calibration(None, SCREEN_WIDTH, SCREEN_HEIGHT, process_noise=NOISE_Q)
    current_calibration_index = -1
    calibration_start_time = time.time()
    collected_points = [[], [], [], []]
    fullscimg = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)

    while True:
        face = Face(np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0]))
        data = recv_gaze_vector(sock)
        for i, key in enumerate([FacePartsName.REYE, FacePartsName.LEYE]):
            eye = getattr(face, key.name.lower())
            eye.normalized_gaze_vector = np.array(data[i], dtype=np.float32)
        
        if current_calibration_index >= len(calibration_points):
            # 캘리브레이션 완료 후 gaze 표시
            fade_factor = 0.02
            fullscimg = cv2.addWeighted(fullscimg, 1 - fade_factor, np.zeros_like(fullscimg), fade_factor, 0)
            try:
                centerd_point = calibration.calc_eye_2d_vector(face)
                point = calibration.calc_trs_transform(centerd_point)
                k_point = calibration.calc_filtered_point(centerd_point)
                k_point = calibration.calc_trs_transform(k_point)
                cv2.circle(fullscimg, (int(k_point[0]), int(k_point[1])), 10, (255, 0, 0), -1)
                cv2.circle(fullscimg, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)
                show_fullscreen_image('calibration', fullscimg)
            except Exception:
                pass
            if cv2.waitKey(1) & 0xFF == 27:
                break

        elif current_calibration_index == -1:
            # n초 대기 후 첫 번째 점으로 이동
            n = 3
            dt = time.time() - calibration_start_time
            img = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
            cv2.putText(img, "Wait "+str(int(n - dt + .5))+"sec", (int(SCREEN_WIDTH / 2) - 50, int(SCREEN_HEIGHT / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            show_fullscreen_image('calibration', img)
            if dt >= n:
                current_calibration_index = 0
                calibration_start_time = time.time()
            if cv2.waitKey(1) & 0xFF == 27:
                break

        else:
            # 현재 점 표시 및 gaze 수집
            current_point = calibration_points[current_calibration_index]
            img = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
            if time.time() - calibration_start_time >= 0.5:
                cv2.circle(img, (int(current_point[0]), int(current_point[1])), 20, (0, 255, 255), -1)
            if time.time() - calibration_start_time >= 1.5:
                cv2.circle(img, (int(current_point[0]), int(current_point[1])), 20, (0, 255, 0), -1)
            for i in range(len(calibration_points)):
                cv2.putText(img, f"{i+1}", (int(calibration_points[i][0]) - 10, int(calibration_points[i][1] + 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            show_fullscreen_image('calibration', img)

            if time.time() - calibration_start_time >= 2:
                try:
                    eye_vector = calibration.calc_eye_2d_vector(face)
                    collected_points[current_calibration_index].append(eye_vector)
                except Exception:
                    pass

            if time.time() - calibration_start_time >= 4:
                current_calibration_index += 1
                calibration_start_time = time.time()
                if current_calibration_index >= len(calibration_points):
                    # 캘리브레이션 행렬 계산
                    points = []
                    for e in collected_points:
                        points.append(calibration.calc_filtered_centers(e))
                    calibration.calc_trs_matrix(calibration_points, points)
                    calibration.calc_noize(points, collected_points, NOISE_R)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    sock.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()