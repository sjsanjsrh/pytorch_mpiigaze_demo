import sys
import socket
import struct
import numpy as np
import time
from PyQt5 import QtCore, QtGui, QtWidgets
from ptgaze.point.calibration import Calibration
from ptgaze.common.face import Face
from ptgaze.common.face_parts import FacePartsName

# 서버 주소와 포트
SERVER_IP = '127.0.0.1'  # 예: '192.168.0.10'
SERVER_PORT = 25500

class GazeReceiver(QtCore.QThread):
    # gaze 데이터 전달용 시그널
    gaze_signal = QtCore.pyqtSignal(object)

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
        # 캘리브레이션 객체 생성
        self.calibration = Calibration(None, screen_width, screen_height)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(30)

    def set_gaze(self, gaze):
        self.gaze = gaze

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
            painter.drawText(self.rect(), QtCore.Qt.AlignTop | QtCore.Qt.AlignHCenter, "시선 추적 중 (ESC로 종료)")
            if self.gaze is not None:
                try:
                    # gaze 데이터로 Face 객체 생성 및 gaze 벡터 할당
                    face = Face(np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0]))
                    for i, key in enumerate([FacePartsName.REYE, FacePartsName.LEYE]):
                        eye = getattr(face, key.name.lower())
                        eye.normalized_gaze_vector = np.array(self.gaze[i], dtype=np.float32)
                    # gaze 2D 좌표 변환 및 필터링, 변환
                    centerd_point = self.calibration.calc_eye_2d_vector(face)
                    point = self.calibration.calc_trs_transform(centerd_point)
                    k_point = self.calibration.calc_filtered_point(centerd_point)
                    k_point = self.calibration.calc_trs_transform(k_point)
                    self.filtered_gaze = 0.8 * self.filtered_gaze + 0.2 * k_point if self.filtered_gaze is not None else k_point
                    # 파란색 원으로 gaze 위치 표시
                    painter.setBrush(QtGui.QBrush(QtCore.Qt.blue))
                    painter.setPen(QtGui.QPen(QtCore.Qt.blue, 2))
                    painter.drawEllipse(int(self.filtered_gaze[0])-10, int(self.filtered_gaze[1])-10, 20, 20)
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
            # 0.0~0.5초: 노란색 점 표시, 기록 안 함
            if dt < 0.5:
                painter.setPen(QtGui.QPen(QtCore.Qt.yellow, 4))
                painter.setBrush(QtGui.QBrush(QtCore.Qt.yellow))
                painter.drawEllipse(int(current_point[0])-20, int(current_point[1])-20, 40, 40)
            # 0.5~1.5초: 녹색 점 표시, gaze 기록
            elif dt < 1.5:
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
            painter.drawText(self.rect(), QtCore.Qt.AlignTop | QtCore.Qt.AlignHCenter, f"점 {self.current_calibration_index+1} 응시")
            # 1.5초 후 다음 점
            if dt >= 1.5:
                self.current_calibration_index += 1
                self.calibration_start_time = now
                if self.current_calibration_index >= len(self.calibration_points):
                    # 캘리브레이션 행렬 계산
                    points = []
                    for e in self.collected_points:
                        points.append(self.calibration.calc_filtered_centers(e))
                    self.calibration.calc_trs_matrix(self.calibration_points, points)
                    self.calibration.calc_noize(points, self.collected_points)
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
    receiver.start()

    app.exec_()
    receiver.stop()
    receiver.wait()