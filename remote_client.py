import socket
import struct
import argparse
import numpy as np
import cv2
import time
from ptgaze.point.calibration import Calibration
import pyautogui
from ptgaze.common import Face, FacePartsName
import platform
import ctypes

# 서버 주소와 포트
SERVER_IP = '127.0.0.1'  # 예: '192.168.0.10'
SERVER_PORT = 25500

# 기본 화면 해상도 (가상 스크린)
SCREEN_WIDTH = pyautogui.size().width
SCREEN_HEIGHT = pyautogui.size().height

NOISE_R = 5
NOISE_Q = 1e-9

def make_calibration_points(sw: int, sh: int):
    return [
        np.array((sw / 10, sh / 10), dtype=np.float32),
        np.array((sw * 9 / 10, sh / 10), dtype=np.float32),
        np.array((sw * 9 / 10, sh * 9 / 10), dtype=np.float32),
        np.array((sw / 10, sh * 9 / 10), dtype=np.float32),
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

def _win_enumerate_monitors():
    """Windows 전용: ctypes로 모니터 목록(x,y,width,height) 반환."""
    monitors = []
    try:
        user32 = ctypes.windll.user32
        class RECT(ctypes.Structure):
            _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long), ("right", ctypes.c_long), ("bottom", ctypes.c_long)]

        class MONITORINFOEX(ctypes.Structure):
            _fields_ = [
                ("cbSize", ctypes.c_ulong),
                ("rcMonitor", RECT),
                ("rcWork", RECT),
                ("dwFlags", ctypes.c_ulong),
                ("szDevice", ctypes.c_wchar * 32),
            ]

        MonitorEnumProc = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.POINTER(RECT), ctypes.c_double)

        def _callback(hMonitor, hdcMonitor, lprcMonitor, dwData):
            mi = MONITORINFOEX()
            mi.cbSize = ctypes.sizeof(MONITORINFOEX)
            user32.GetMonitorInfoW(ctypes.c_void_p(hMonitor), ctypes.byref(mi))
            r = mi.rcMonitor
            monitors.append({
                'x': int(r.left),
                'y': int(r.top),
                'width': int(r.right - r.left),
                'height': int(r.bottom - r.top),
            })
            return 1

        user32.EnumDisplayMonitors(0, 0, MonitorEnumProc(_callback), 0)
    except Exception:
        pass
    return monitors


def setup_window_on_monitor(name: str, monitor_index: int, fullscreen: bool = True):
    """Create and place the OpenCV window on the specified monitor.

    If screeninfo is available, we use the monitor's origin and size. Otherwise,
    we fall back to the primary screen's size and position at (0,0).
    """
    global SCREEN_WIDTH, SCREEN_HEIGHT

    mon_x, mon_y, mon_w, mon_h = 0, 0, SCREEN_WIDTH, SCREEN_HEIGHT
    used = False
    # 1) optional dependency
    try:
        from screeninfo import get_monitors  # type: ignore
        monitors = get_monitors()
        if monitors:
            idx = max(0, min(monitor_index, len(monitors) - 1))
            m = monitors[idx]
            mon_x, mon_y, mon_w, mon_h = int(getattr(m, 'x', 0)), int(getattr(m, 'y', 0)), int(m.width), int(m.height)
            used = True
    except Exception:
        pass
    # 2) Windows fallback via ctypes
    if not used and platform.system().lower() == 'windows':
        mons = _win_enumerate_monitors()
        if mons:
            idx = max(0, min(monitor_index, len(mons) - 1))
            m = mons[idx]
            mon_x, mon_y, mon_w, mon_h = m['x'], m['y'], m['width'], m['height']

    # Update global screen width/height to the selected monitor size
    SCREEN_WIDTH, SCREEN_HEIGHT = mon_w, mon_h

    # Create a movable/resizeable window first, then position and optionally fullscreen it
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(name, mon_x, mon_y)
    if fullscreen:
        cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.resizeWindow(name, mon_w, mon_h)


def show_fullscreen_image(name, img):
    # Assumes the window has been created and placed once via setup_window_on_monitor
    cv2.imshow(name, img)

def parse_args():
    parser = argparse.ArgumentParser(description='Remote gaze client with multi-monitor support')
    parser.add_argument('--monitor', '--screen', type=int, default=0, help='표시할 모니터 인덱스 (0부터 시작). 기본값: 0')
    parser.add_argument('--windowed', action='store_true', help='전체화면 대신 창 모드로 표시')
    parser.add_argument('--no-blue', action='store_true', help='캘리브레이션 완료 후 파란점(필터링된 지점) 표시 생략')
    return parser.parse_args()


def main():
    args = parse_args()
    # 런처 차이에 대비해 모니터 인덱스 안전하게 추출
    monitor_idx = getattr(args, 'monitor', getattr(args, 'screen', 0))
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(5)
    sock.sendto(b'bind', (SERVER_IP, SERVER_PORT))  # 서버에 바인드 요청

    while True:
        data, _ = sock.recvfrom(1024)
        if data[4:].decode() == 'bind':
            break
        time.sleep(0.1)

    # 창을 만들기 전, 선택한 모니터에 맞춰 초기화
    setup_window_on_monitor('calibration', monitor_idx, fullscreen=not args.windowed)

    # 선택된 모니터 크기로 캘리브레이션 구성
    calibration = Calibration(None, SCREEN_WIDTH, SCREEN_HEIGHT, process_noise=NOISE_Q)
    current_calibration_index = -1
    calibration_start_time = time.time()
    collected_points = [[], [], [], []]
    fullscimg = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
    calibration_points = make_calibration_points(SCREEN_WIDTH, SCREEN_HEIGHT)

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
                if not args.no_blue:
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