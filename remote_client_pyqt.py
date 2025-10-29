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

MOUSE_CLICK_RATIO = 0.1  # 마우스 클릭 인식 비율

MOUSE_SCROOL_STEP = 100  # 마우스 스크롤 시 이동 거리

MOUTH_CLOSED_THRESHOLD = 0.045  # 이 값 이하이면 입이 다물린 것으로 간주
MOUTH_CENTER_DELTA_THRESHOLD = 0.01  # 입술 중심 이동량 임계값
MOUTH_CENTER_SMOOTHING = 0.35  # 입술 중심 위치 지수평활 계수
MOUTH_VISUAL_MAX_RATIO = 0.25  # 입벌림 시각화 최대 스케일
MOUTH_CENTER_VISUAL_RANGE = 0.05  # 입술 중심 시각화 범위(+/-)
MOUTH_CENTER_BASELINE_SMOOTHING = 0.1  # 입술 중심 기준 업데이트 계수
MOUTH_DATA_STALE_TIMEOUT = 0.6  # 데이터 수신이 멈췄다고 간주할 시간(초)
MOUTH_CENTER_OPEN_COMPENSATION = 0.3  # 입이 벌어졌을 때 중심 이동 보정 계수
MOUTH_SCROLL_LOCKOUT_AFTER_OPEN = 0.45  # 입을 벌린 직후 스크롤을 잠시 막는 시간(초)
MOUTH_JOYSTICK_MIN_INTERVAL = 0.05  # 조이스틱 최대 속도일 때 스크롤 간격(초)
MOUTH_JOYSTICK_MAX_INTERVAL = 0.2  # 조이스틱 최소 속도일 때 스크롤 간격(초)
MOUTH_JOYSTICK_BASE_STRENGTH = 0.35  # 최소 스크롤 강도 배율
MOUTH_COMMAND_IDLE_SPEED = 0.1  # 입 명령 중 기본 커서 이동 속도 배율
MOUTH_COMMAND_RAMP_INITIAL = 0.25  # 입 명령 가속 시작 시 속도/강도 배율
MOUTH_COMMAND_RAMP_DURATION = 1.2  # 가속이 최고 속도에 도달하는 시간(초)
MOUTH_COMMAND_RAMP_MAX = 0.5  # 입 명령 가속 후 최대 속도/강도 배율
MOUTH_SCROLL_OPEN_TOLERANCE = 0.008  # 스크롤 시 허용할 추가 입벌림 여유

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
        timeout_count = 0
        max_timeouts = 4
        while self.running:
            try:
                data, _ = sock.recvfrom(1024)
                timeout_count = 0
                if len(data) >= 24:
                    # 오른쪽, 왼쪽 눈 각각 3차원 벡터
                    gaze = np.frombuffer(data[:24], dtype=np.float32)
                    reye = gaze[:3]
                    leye = gaze[3:6]
                    self.gaze_signal.emit((reye, leye))
                if len(data) >= 24 + 8:
                    mouth_metrics = np.frombuffer(data[24:], dtype=np.float32)
                    if mouth_metrics.size >= 2:
                        self.mouth_open_signal.emit(mouth_metrics[:2])
            except socket.timeout:
                timeout_count += 1
                if timeout_count >= max_timeouts:
                    self.running = False
                    app = QtWidgets.QApplication.instance()
                    if app is not None:
                        QtCore.QMetaObject.invokeMethod(
                            app,
                            "quit",
                            QtCore.Qt.QueuedConnection,
                        )
                    break
            except Exception:
                self.running = False
                app = QtWidgets.QApplication.instance()
                if app is not None:
                    QtCore.QMetaObject.invokeMethod(
                        app,
                        "quit",
                        QtCore.Qt.QueuedConnection,
                    )
                break
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
        self.gaze = None
        self.filtered_gaze = None
        self.mouth_open_ratio = None
        self.filtered_mouth_center = None
        self.mouse_pressed = False
        self.center_baseline = None
        self.center_ratio_baseline = None
        self.center_offset = 0.0
        self.last_mouth_update = 0.0
        self.last_mouth_open_time = 0.0
        self.mouth_drag_active = False
        self.mouth_drag_start = 0.0
        self.cursor_speed_factor = 1.0
        self.joystick_scroll_active = False
        self.joystick_scroll_start = 0.0

        self.mouseHooking = False

        # 캘리브레이션 객체 생성
        self.calibration = Calibration(None, screen_width, screen_height, 
                                       process_noise=NOISE_Q)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(30)

        self.last_scroll_time = 0  # 마지막 스크롤 시각

        # 입술 기반 스크롤 타이머
        self.scroll_timer = QtCore.QTimer(self)
        self.scroll_timer.timeout.connect(self.check_and_scroll)
        self.scroll_timer.start(50)  # 50ms마다 체크

    def set_gaze(self, gaze):
        self.gaze = gaze
    
    def set_mouth_open_ratio(self, mouth_data):
        now = time.time()
        if mouth_data is None:
            return

        values = np.asarray(mouth_data, dtype=np.float32).flatten()
        if values.size < 2:
            return

        ratio = float(values[0])
        center = float(values[1])
        if not np.isfinite(ratio) or not np.isfinite(center) or ratio < 0:
            return

        ratio = max(0.0, ratio)
        self.mouth_open_ratio = ratio

        if self.filtered_mouth_center is None:
            self.filtered_mouth_center = center
        else:
            self.filtered_mouth_center = (
                (1.0 - MOUTH_CENTER_SMOOTHING) * self.filtered_mouth_center +
                MOUTH_CENTER_SMOOTHING * center
            )

        self.last_mouth_update = now

        if self.filtered_mouth_center is not None:
            if ratio <= MOUTH_CLOSED_THRESHOLD:
                if self.center_baseline is None:
                    self.center_baseline = self.filtered_mouth_center
                else:
                    self.center_baseline = (
                        (1.0 - MOUTH_CENTER_BASELINE_SMOOTHING) * self.center_baseline +
                        MOUTH_CENTER_BASELINE_SMOOTHING * self.filtered_mouth_center
                    )

                if self.center_ratio_baseline is None:
                    self.center_ratio_baseline = ratio
                else:
                    self.center_ratio_baseline = (
                        (1.0 - MOUTH_CENTER_BASELINE_SMOOTHING) * self.center_ratio_baseline +
                        MOUTH_CENTER_BASELINE_SMOOTHING * ratio
                    )
            else:
                self.last_mouth_open_time = now

            base_center = self.center_baseline
            base_ratio = self.center_ratio_baseline
            if base_center is not None:
                ratio_delta = 0.0
                if base_ratio is not None:
                    ratio_delta = max(0.0, ratio - base_ratio)
                compensation = ratio_delta * MOUTH_CENTER_OPEN_COMPENSATION
                adjusted_center = self.filtered_mouth_center - compensation
                new_offset = adjusted_center - base_center
                clamp_range = MOUTH_CENTER_VISUAL_RANGE * 2.0
                self.center_offset = max(-clamp_range, min(new_offset, clamp_range))
            else:
                self.center_offset = 0.0

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

    def _update_cursor_position(self, target_x, target_y, now, mouse_opened, command_active):
        """Move cursor toward the target with adaptive speed during mouth commands."""
        if mouse_opened:
            if not self.mouth_drag_active:
                self.mouth_drag_active = True
                self.mouth_drag_start = now
            elapsed = max(0.0, now - self.mouth_drag_start)
            ramp = min(1.0, elapsed / max(MOUTH_COMMAND_RAMP_DURATION, 1e-6))
            factor = MOUTH_COMMAND_RAMP_INITIAL + ramp * (MOUTH_COMMAND_RAMP_MAX - MOUTH_COMMAND_RAMP_INITIAL)
        else:
            if self.mouth_drag_active:
                self.mouth_drag_active = False
                self.mouth_drag_start = 0.0
            factor = MOUTH_COMMAND_IDLE_SPEED if command_active else 1.0

        factor = max(0.05, min(MOUTH_COMMAND_RAMP_MAX, factor))
        self.cursor_speed_factor = factor

        current_pos = QtGui.QCursor.pos()
        current_x = float(current_pos.x())
        current_y = float(current_pos.y())

        if factor >= 0.999 or (abs(target_x - current_x) < 1.0 and abs(target_y - current_y) < 1.0):
            QtGui.QCursor.setPos(int(round(target_x)), int(round(target_y)))
            return QtGui.QCursor.pos()

        new_x = current_x + (target_x - current_x) * factor
        new_y = current_y + (target_y - current_y) * factor

        QtGui.QCursor.setPos(int(round(new_x)), int(round(new_y)))
        return QtGui.QCursor.pos()

    def draw_mouth_visualization(self, painter, now):
        ratio = float(self.mouth_open_ratio) if self.mouth_open_ratio is not None else 0.0
        ratio = max(0.0, min(ratio, 1.2))
        normalized_ratio = max(0.0, min(ratio / max(MOUTH_VISUAL_MAX_RATIO, 1e-6), 1.0))

        center_value = self.filtered_mouth_center if self.filtered_mouth_center is not None else None
        baseline = self.center_baseline
        offset = self.center_offset if center_value is not None else 0.0
        normalized_center = 0.0
        if center_value is not None and baseline is not None:
            normalized_center = max(
                -1.0,
                min(offset / max(MOUTH_CENTER_VISUAL_RANGE, 1e-6), 1.0)
            )

        data_is_stale = (now - self.last_mouth_update) > MOUTH_DATA_STALE_TIMEOUT

        bar_width = 240
        bar_height = 16
        center_bar_height = 14
        margin = 36
        left = self.screen_width - bar_width - margin
        top = margin

        panel_height = 132

        painter.save()
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(0, 0, 0, 160))
        painter.drawRoundedRect(left - 12, top - 18, bar_width + 24, panel_height, 10, 10)

        # --- Mouth open ratio bar ---
        painter.setBrush(QtGui.QColor(70, 70, 70, 220))
        painter.drawRect(left, top, bar_width, bar_height)

        ratio_fill_alpha = 230 if not data_is_stale else 90
        painter.setBrush(QtGui.QColor(0, 180, 255, ratio_fill_alpha))
        painter.drawRect(left, top, int(bar_width * normalized_ratio), bar_height)

        scale = max(MOUTH_VISUAL_MAX_RATIO, 1e-6)
        closed_ratio = min(1.0, MOUTH_CLOSED_THRESHOLD / scale)
        closed_x = left + int(bar_width * closed_ratio)
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 220, 120, 220), 1, QtCore.Qt.DashLine))
        painter.drawLine(closed_x, top - 4, closed_x, top + bar_height + 4)

        if MOUTH_SCROLL_OPEN_TOLERANCE > 1e-6:
            tol_value = MOUTH_CLOSED_THRESHOLD + MOUTH_SCROLL_OPEN_TOLERANCE
            tol_ratio = min(1.0, tol_value / scale)
            tol_x = left + int(bar_width * tol_ratio)
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 170, 60, 200), 1, QtCore.Qt.DotLine))
            painter.drawLine(tol_x, top - 4, tol_x, top + bar_height + 4)

        click_ratio = min(1.0, MOUSE_CLICK_RATIO / scale)
        click_x = left + int(bar_width * click_ratio)
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 90, 90, 230), 2))
        painter.drawLine(click_x, top - 6, click_x, top + bar_height + 6)

        ratio_pen_color = QtGui.QColor(255, 120, 120) if ratio >= MOUSE_CLICK_RATIO else QtCore.Qt.white
        painter.setPen(QtGui.QPen(ratio_pen_color))
        painter.setFont(QtGui.QFont('Arial', 14))
        painter.drawText(
            left - 4,
            top + bar_height + 24,
            f"입벌림 {ratio:.3f}  닫힘≤{MOUTH_CLOSED_THRESHOLD:.3f}  클릭≥{MOUSE_CLICK_RATIO:.3f}"
        )

        painter.setFont(QtGui.QFont('Arial', 12))
        painter.setPen(QtGui.QPen(QtGui.QColor(200, 200, 200, 200)))
        status_text = "입 모양 데이터 대기 중" if data_is_stale else (
            "입 다물림" if ratio <= MOUTH_CLOSED_THRESHOLD else "입 벌림"
        )
        painter.drawText(left - 4, top + bar_height + 44, status_text)

        # --- Mouth center bar ---
        center_bar_top = top + bar_height + 58
        painter.setBrush(QtGui.QColor(70, 70, 70, 220))
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawRect(left, center_bar_top, bar_width, center_bar_height)

        zero_x = left + bar_width // 2
        painter.setPen(QtGui.QPen(QtGui.QColor(190, 190, 190, 200), 1))
        painter.drawLine(zero_x, center_bar_top - 4, zero_x, center_bar_top + center_bar_height + 4)

        threshold_norm = min(1.0, MOUTH_CENTER_DELTA_THRESHOLD / max(MOUTH_CENTER_VISUAL_RANGE, 1e-6))
        threshold_pixels = int((bar_width / 2) * threshold_norm)
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 160, 0, 220), 1, QtCore.Qt.DashLine))
        painter.drawLine(zero_x + threshold_pixels, center_bar_top, zero_x + threshold_pixels, center_bar_top + center_bar_height)
        painter.drawLine(zero_x - threshold_pixels, center_bar_top, zero_x - threshold_pixels, center_bar_top + center_bar_height)

        center_fill_alpha = 210 if not data_is_stale else 90
        if center_value is not None and baseline is not None:
            fill_width = int((bar_width / 2) * abs(normalized_center))
            fill_width = max(0, min(fill_width, bar_width // 2))
            if fill_width > 0:
                if normalized_center > 0:
                    rect = QtCore.QRect(zero_x, center_bar_top, fill_width, center_bar_height)
                    painter.setBrush(QtGui.QColor(255, 140, 60, center_fill_alpha))
                    painter.drawRect(rect)
                else:
                    rect = QtCore.QRect(zero_x - fill_width, center_bar_top, fill_width, center_bar_height)
                    painter.setBrush(QtGui.QColor(80, 180, 255, center_fill_alpha))
                    painter.drawRect(rect)

        painter.setFont(QtGui.QFont('Arial', 13))
        highlight_center = (
            center_value is not None and baseline is not None and
            abs(offset) >= MOUTH_CENTER_DELTA_THRESHOLD
        )
        center_text_color = QtGui.QColor(255, 200, 90) if highlight_center else QtCore.Qt.white
        painter.setPen(QtGui.QPen(center_text_color))
        if center_value is not None and baseline is not None:
            painter.drawText(
                left - 4,
                center_bar_top + center_bar_height + 24,
                f"중심 Δ {offset:+.4f} / ±{MOUTH_CENTER_DELTA_THRESHOLD:.4f}  (기준 {baseline:.4f})"
            )
        elif center_value is not None:
            painter.drawText(left - 4, center_bar_top + center_bar_height + 24, "중심 기준 계산 중...")
        else:
            painter.drawText(left - 4, center_bar_top + center_bar_height + 24, "중심 데이터 없음")

        painter.setPen(QtGui.QPen(QtGui.QColor(180, 180, 180, 220)))
        painter.setFont(QtGui.QFont('Arial', 11))
        mouse_status = "ON" if self.mouseHooking else "OFF"
        painter.drawText(
            left - 4,
            center_bar_top + center_bar_height + 46,
            f"마우스 제어 {mouse_status} ({MOUSE_HOOKING_KEY} 토글)"
        )

        painter.restore()

    def check_and_scroll(self):
        # 입술 중심의 조이스틱 변위로 마우스 휠 이벤트 발생
        if (
            self.mouth_open_ratio is None or
            self.filtered_mouth_center is None or
            self.center_baseline is None
        ):
            if self.joystick_scroll_active:
                self.joystick_scroll_active = False
                self.joystick_scroll_start = 0.0
            return

        open_excess = max(0.0, self.mouth_open_ratio - MOUTH_CLOSED_THRESHOLD)
        if open_excess > MOUTH_SCROLL_OPEN_TOLERANCE:
            if self.joystick_scroll_active:
                self.joystick_scroll_active = False
                self.joystick_scroll_start = 0.0
            return

        if not self.mouseHooking or self.mouse_pressed:
            if self.joystick_scroll_active:
                self.joystick_scroll_active = False
                self.joystick_scroll_start = 0.0
            return

        now = time.time()
        if now - self.last_mouth_open_time < MOUTH_SCROLL_LOCKOUT_AFTER_OPEN:
            if self.joystick_scroll_active:
                self.joystick_scroll_active = False
                self.joystick_scroll_start = 0.0
            return

        offset = self.center_offset
        abs_offset = abs(offset)
        if abs_offset <= MOUTH_CENTER_DELTA_THRESHOLD:
            if self.joystick_scroll_active:
                self.joystick_scroll_active = False
                self.joystick_scroll_start = 0.0
            return

        if not self.joystick_scroll_active:
            self.joystick_scroll_active = True
            self.joystick_scroll_start = now

        # 기준 임계값을 넘는 정도를 0~1 범위로 정규화
        effective_range = max(MOUTH_CENTER_VISUAL_RANGE - MOUTH_CENTER_DELTA_THRESHOLD, 1e-6)
        normalized = min(1.0, (abs_offset - MOUTH_CENTER_DELTA_THRESHOLD) / effective_range)

        elapsed = max(0.0, now - self.joystick_scroll_start)
        ramp = min(1.0, elapsed / max(MOUTH_COMMAND_RAMP_DURATION, 1e-6))
        gain = MOUTH_COMMAND_RAMP_INITIAL + (MOUTH_COMMAND_RAMP_MAX - MOUTH_COMMAND_RAMP_INITIAL) * ramp
        open_factor = 1.0
        if MOUTH_SCROLL_OPEN_TOLERANCE > 1e-6:
            open_factor = max(0.0, min(1.0, 1.0 - (open_excess / MOUTH_SCROLL_OPEN_TOLERANCE)))
        gain = max(0.0, min(1.0, gain * open_factor))
        effective_normalized = max(0.0, min(1.0, normalized * gain))

        # 강도와 간격을 변형해 조이스틱 감각을 제공
        interval = max(
            MOUTH_JOYSTICK_MIN_INTERVAL,
            MOUTH_JOYSTICK_MAX_INTERVAL - (MOUTH_JOYSTICK_MAX_INTERVAL - MOUTH_JOYSTICK_MIN_INTERVAL) * effective_normalized
        )
        if now - self.last_scroll_time < interval:
            return

        strength = (
            MOUTH_JOYSTICK_BASE_STRENGTH * gain +
            (1.0 - MOUTH_JOYSTICK_BASE_STRENGTH) * effective_normalized
        )
        strength = max(0.05, min(1.0, strength))
        direction = 1 if offset < 0 else -1
        wheel_delta = int(direction * MOUSE_SCROOL_STEP * strength)
        if wheel_delta == 0:
            fallback = MOUSE_SCROOL_STEP * max(0.05, MOUTH_JOYSTICK_BASE_STRENGTH * gain)
            wheel_delta = direction * max(1, int(fallback))

        cursor_pos = QtGui.QCursor.pos()
        self.mouse_event_scroll(cursor_pos.x(), cursor_pos.y(), wheel_delta)
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
                    
                    mouse_opened = (
                        self.mouth_open_ratio is not None and
                        self.mouth_open_ratio > MOUSE_CLICK_RATIO
                    )

                    command_active = False
                    if self.mouth_open_ratio is not None and self.mouth_open_ratio > MOUTH_CLOSED_THRESHOLD:
                        command_active = True
                    elif self.center_baseline is not None and abs(self.center_offset) > MOUTH_CENTER_DELTA_THRESHOLD:
                        command_active = True

                    if self.mouseHooking:
                        cursor_pos = self._update_cursor_position(float(x), float(y), now, mouse_opened, command_active)
                        # 마우스 클릭
                        if mouse_opened and not self.mouse_pressed:
                            self.mouse_event_leftdown(cursor_pos.x(), cursor_pos.y())
                            self.mouse_pressed = True
                        elif not mouse_opened and self.mouse_pressed:
                            self.mouse_event_leftup(cursor_pos.x(), cursor_pos.y())
                            self.mouse_pressed = False
                    else:
                        self.cursor_speed_factor = 1.0
                        if self.mouth_drag_active:
                            self.mouth_drag_active = False
                            self.mouth_drag_start = 0.0
                        if self.mouse_pressed:
                            cursor_pos = QtGui.QCursor.pos()
                            self.mouse_event_leftup(cursor_pos.x(), cursor_pos.y())
                            self.mouse_pressed = False
                        # 파란색 원으로 gaze 위치 표시
                        if mouse_opened:
                            color = QtCore.Qt.red
                        else:
                            color = QtCore.Qt.blue
                        painter.setBrush(QtGui.QBrush(color))
                        painter.setPen(QtGui.QPen(color, 2))
                        painter.drawEllipse(x-10, y-10, 20, 20)

                except Exception:
                    pass

            self.draw_mouth_visualization(painter, now)

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