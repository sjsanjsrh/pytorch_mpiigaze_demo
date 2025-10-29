import sys
import socket
import numpy as np
import time
import keyboard
from ptgaze.point.calibration import Calibration
from ptgaze.common.face import Face
from ptgaze.common.face_parts import FacePartsName
import ctypes
from PyQt5 import QtCore, QtGui, QtWidgets

# 서버 주소와 포트
SERVER_IP = '127.0.0.1'  # 예: '192.168.0.10'
SERVER_PORT = 25500

MOUSE_HOOKING_KEY = '['  # 마우스 위치 업데이트 토글 키
RESET_CALIBRATION_KEY = ']'  # 캘리브레이션 초기화 키

MOUSE_SCROOL_AREA = 0.02  # 마우스 스크롤 인식영역 (0.05 = 5% 화면 높이)
MOUSE_SCROOL_STEP = 100  # 마우스 스크롤 시 이동 거리

MOUSE_CLICK_RATIO = 0.1  # 마우스 클릭 인식 비율

NOISE_R = 5
NOISE_Q = 1e-9

class GazeReceiver(QtCore.QThread):
    # gaze 데이터 전달용 시그널
    gaze_signal = QtCore.pyqtSignal(object)
    mouth_open_signal = QtCore.pyqtSignal(object)

    def __init__(self, server_ip, server_port):
        super().__init__()
        self.server_ip = server_ip
        self.server_port = server_port
        self.running = True

    def run(self):
        # UDP 소켓 생성 및 서버에 바인드 요청
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(5)
        sock.sendto(b'bind', (self.server_ip, self.server_port))  # 서버에 바인드 요청
        while self.running:
            try:
                data, _ = sock.recvfrom(1024)
                if len(data) >= 24:
                    # 오른쪽, 왼쪽 눈 각각 3차원 벡터
                    gaze = np.frombuffer(data[:24], dtype=np.float32)
                    reye = gaze[:3]
                    leye = gaze[3:6]
                    self.gaze_signal.emit((reye, leye))
                if len(data) >= 24+4:
                    mouth_open_ratio = np.frombuffer(data[24:], dtype=np.float32)
                    self.mouth_open_signal.emit(mouth_open_ratio)
            except Exception:
                pass
        sock.close()

    def stop(self):
        self.running = False

class Overlay(QtWidgets.QWidget):
    def __init__(self, screen_width, screen_height):
        super().__init__()
        self.setWindowFlags(
            QtCore.Qt.FramelessWindowHint |
            QtCore.Qt.WindowStaysOnTopHint |
            QtCore.Qt.Tool
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setWindowFlag(QtCore.Qt.WindowTransparentForInput)
        self.setGeometry(0, 0, screen_width, screen_height)

        self.screen_width = screen_width
        self.screen_height = screen_height

        # 캘리브레이션 점
        self.calibration_points = [
            np.array((screen_width / 10, screen_height / 10), dtype=np.float32),
            np.array((screen_width * 9 / 10, screen_height / 10), dtype=np.float32),
            np.array((screen_width * 9 / 10, screen_height * 9 / 10), dtype=np.float32),
            np.array((screen_width / 10, screen_height * 9 / 10), dtype=np.float32)
        ]
        self.current_calibration_index = -1
        self.calibration_start_time = time.time()
        self.collected_points = [[], [], [], []]
        self.calibrated = False
        self.gaze = None
        self.filtered_gaze = None
        self.calibrated_points = []
        self.mouth_open_ratio = None

        self.mouseHooking = False

        # 캘리브레이션 객체 생성
        self.calibration = Calibration(None, screen_width, screen_height, 
                                       process_noise=NOISE_Q)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(30)

        self.last_scroll_time = 0  # 마지막 스크롤 시각
        self.scroll_interval = 0.1  # 최소 스크롤 간격(초)

        # gaze 기반 스크롤 타이머
        self.scroll_timer = QtCore.QTimer(self)
        self.scroll_timer.timeout.connect(self.check_and_scroll)
        self.scroll_timer.start(50)  # 50ms마다 체크

    def set_gaze(self, gaze):
        self.gaze = gaze
    
    def set_mouth_open_ratio(self, mouth_open_ratio):
        self.mouth_open_ratio = mouth_open_ratio

    def mouse_event_scroll(self, x, y, wheel_delta=0):
        # Windows용 마우스 휠 이벤트 발생
        MOUSEEVENTF_WHEEL = 0x0800
        ctypes.windll.user32.mouse_event(MOUSEEVENTF_WHEEL, int(x), int(y), int(wheel_delta), 0)
    
    def mouse_event_leftdown(self, x, y):
        # Windows용 마우스 왼쪽 버튼 누르기 이벤트 발생
        MOUSEEVENTF_LEFTDOWN = 0x0002
        ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTDOWN, int(x), int(y), 0, 0)
    
    def mouse_event_leftup(self, x, y):
        # Windows용 마우스 왼쪽 버튼 누르기 이벤트 발생
        MOUSEEVENTF_LEFTUP = 0x0004
        ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTUP, int(x), int(y), 0, 0)


    def check_and_scroll(self):
        # gaze 위치에 따라 마우스 휠 이벤트 발생 (paintEvent에서 분리)
        if not self.mouseHooking or self.filtered_gaze is None:
            return
        x, y = self.filtered_gaze
        scrollyu = int(self.screen_height * MOUSE_SCROOL_AREA)
        scrollyd = int(self.screen_height * (1 - MOUSE_SCROOL_AREA))
        now = time.time()
        if now - self.last_scroll_time < self.scroll_interval:
            return  # 너무 자주 스크롤 방지
        if y < scrollyu:
            self.mouse_event_scroll(x, y, MOUSE_SCROOL_STEP)
            self.last_scroll_time = now
        elif y > scrollyd:
            self.mouse_event_scroll(x, y, -MOUSE_SCROOL_STEP)
            self.last_scroll_time = now

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setCompositionMode(QtGui.QPainter.CompositionMode_Source)
        painter.fillRect(self.rect(), QtCore.Qt.transparent)

        now = time.time()

        if self.current_calibration_index >= len(self.calibration_points):
            # 캘리브레이션 완료 후 gaze 표시
            painter.setPen(QtGui.QPen(QtCore.Qt.white, 2))
            painter.setFont(QtGui.QFont('Arial', 40))
            painter.drawText(self.rect(), QtCore.Qt.AlignTop | QtCore.Qt.AlignHCenter, "시선 추적 중...")
            if self.gaze is not None:
                try:
                    # gaze 데이터로 Face 객체 생성 및 gaze 벡터 할당
                    face = Face(np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0]))
                    for i, key in enumerate([FacePartsName.REYE, FacePartsName.LEYE]):
                        eye = getattr(face, key.name.lower())
                        eye.normalized_gaze_vector = np.array(self.gaze[i], dtype=np.float32)
                    # gaze 2D 좌표 변환 및 필터링, 변환
                    centerd_point = self.calibration.calc_eye_2d_vector(face)
                    point = self.calibration.calc_filtered_point(centerd_point)
                    self.filtered_gaze = self.calibration.calc_trs_transform(point)
                    
                    x, y = self.filtered_gaze
                    
                    scrollyu = int(self.screen_height * MOUSE_SCROOL_AREA)
                    scrollyd = int(self.screen_height * (1 - MOUSE_SCROOL_AREA))
                    mouse_opened = self.mouth_open_ratio > MOUSE_CLICK_RATIO
                    if self.mouseHooking:
                        # # 마우스 위치 업데이트
                        QtGui.QCursor.setPos(x, y)
                        # 마우스 클릭
                        if mouse_opened:
                            self.mouse_event_leftdown(x, y)
                        else:
                            self.mouse_event_leftup(x, y)
                    else:
                        # 파란색 원으로 gaze 위치 표시
                        if mouse_opened:
                            color = QtCore.Qt.red
                        else:
                            color = QtCore.Qt.blue
                        painter.setBrush(QtGui.QBrush(color))
                        painter.setPen(QtGui.QPen(color, 2))
                        painter.drawEllipse(x-10, y-10, 20, 20)

                    # 마우스 스크롤 영역 표시
                    painter.setPen(QtGui.QPen(QtCore.Qt.red, 2))
                    painter.drawLine(0, scrollyu, self.screen_width, scrollyu)
                    painter.drawLine(0, scrollyd, self.screen_width, scrollyd)

                except Exception:
                    pass

        elif self.current_calibration_index == -1:
            # n초 대기 후 첫 번째 점으로 이동
            n = 3
            dt = now - self.calibration_start_time
            painter.setPen(QtGui.QPen(QtCore.Qt.white, 2))
            painter.setFont(QtGui.QFont('Arial', 40))
            painter.drawText(self.rect(), QtCore.Qt.AlignCenter, f"캘리브레이션 준비 중... {int(n - dt + .5)}초")
            if dt >= n:
                self.current_calibration_index = 0
                self.calibration_start_time = now

        else:
            # 현재 점 표시 및 gaze 수집
            current_point = self.calibration_points[self.current_calibration_index]
            dt = now - self.calibration_start_time
            # 중앙에서부터 화살표 그리기
            painter.setPen(QtGui.QPen(QtCore.Qt.red, 4))
            painter.setBrush(QtCore.Qt.red)
            cx, cy = self.screen_width / 2, self.screen_height / 2
            tx, ty = current_point
            v = np.array([tx - cx, ty - cy])
            v /= np.linalg.norm(v)
            # 화살표 길이 조정
            cv = v * 100
            tv = v * 50
            cx, cy = cx + cv[0], cy + cv[1]
            tx, ty = tx - tv[0], ty - tv[1]
            cx, cy = int(cx), int(cy)
            tx, ty = int(tx), int(ty)
            painter.drawLine(cx, cy, tx, ty)
            # 삼각형 화살표 끝 부분 그리기
            arraw_size = 20
            painter.drawPolygon(QtGui.QPolygon([
                QtCore.QPoint(int(tx + v[1]*arraw_size/2 - v[0]*arraw_size), int(ty - v[0]*arraw_size/2 - v[1]*arraw_size)),
                QtCore.QPoint(int(tx - v[1]*arraw_size/2 - v[0]*arraw_size), int(ty + v[0]*arraw_size/2 - v[1]*arraw_size)),
                QtCore.QPoint(int(tx), int(ty))
            ]))
            # 노란색 점 표시, 기록 안 함
            if dt < 1.0:
                painter.setPen(QtGui.QPen(QtCore.Qt.yellow, 4))
                painter.setBrush(QtGui.QBrush(QtCore.Qt.yellow))
                painter.drawEllipse(int(current_point[0])-20, int(current_point[1])-20, 40, 40)
            # 녹색 점 표시, gaze 기록
            elif dt < 2.0:
                painter.setPen(QtGui.QPen(QtCore.Qt.green, 4))
                painter.setBrush(QtGui.QBrush(QtCore.Qt.green))
                painter.drawEllipse(int(current_point[0])-20, int(current_point[1])-20, 40, 40)
                if self.gaze is not None:
                    try:
                        # gaze 데이터로 Face 객체 생성 및 gaze 벡터 할당
                        face = Face(np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0]))
                        for i, key in enumerate([FacePartsName.REYE, FacePartsName.LEYE]):
                            eye = getattr(face, key.name.lower())
                            eye.normalized_gaze_vector = np.array(self.gaze[i], dtype=np.float32)
                        # 2D gaze 좌표 계산 후 수집
                        eye_vector = self.calibration.calc_eye_2d_vector(face)
                        self.collected_points[self.current_calibration_index].append(eye_vector)
                    except Exception:
                        pass
            # 점 번호 표시
            painter.setPen(QtGui.QPen(QtCore.Qt.magenta, 2))
            painter.setFont(QtGui.QFont('Arial', 30))
            for i, pt in enumerate(self.calibration_points):
                painter.drawText(int(pt[0])-10, int(pt[1])+10, str(i+1))
            # 안내문구
            painter.setPen(QtGui.QPen(QtCore.Qt.white, 2))
            painter.setFont(QtGui.QFont('Arial', 40))
            painter.drawText(self.rect(), QtCore.Qt.AlignVCenter | QtCore.Qt.AlignHCenter, f"점 {self.current_calibration_index+1} 응시")
            # 1.5초 후 다음 점
            if dt >= 2.0:
                self.current_calibration_index += 1
                self.calibration_start_time = now
                if self.current_calibration_index >= len(self.calibration_points):
                    # 캘리브레이션 행렬 계산
                    points = []
                    for e in self.collected_points:
                        points.append(self.calibration.calc_filtered_centers(e))
                    self.calibration.calc_trs_matrix(self.calibration_points, points)
                    self.calibration.calc_noize(points, self.collected_points, NOISE_R)
                    self.calibrated = True

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            QtWidgets.QApplication.quit()

if __name__ == "__main__":
    # 화면 해상도 자동 감지
    app = QtWidgets.QApplication(sys.argv)
    screen = QtWidgets.QApplication.primaryScreen().geometry()
    overlay = Overlay(screen.width(), screen.height())
    overlay.showFullScreen()

    # gaze 수신 스레드 시작
    receiver = GazeReceiver(SERVER_IP, SERVER_PORT)
    receiver.gaze_signal.connect(overlay.set_gaze)
    receiver.mouth_open_signal.connect(overlay.set_mouth_open_ratio)
    receiver.start()

    # 키보드 이벤트 처리
    keyboard.add_hotkey(MOUSE_HOOKING_KEY, 
                        lambda: setattr(overlay, 'mouseHooking', not overlay.mouseHooking))
    keyboard.add_hotkey(RESET_CALIBRATION_KEY,
                        lambda: setattr(overlay, 'current_calibration_index', -1))
    app.exec_()
    receiver.stop()
    receiver.wait()