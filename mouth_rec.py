"""Mouth landmark recorder.

이 스크립트는 `remote_server.py`가 사용하는 파이프라인과 동일한 추정기를 활용해
얼굴 랜드마크와 입 관련 메트릭(입벌림 정도, 입술 중심 위치)을 수집합니다.

수집하고자 하는 클래스
-----------------------
- 1: mouth_closed (입 다물기)
- 2: mouth_open (입 벌리기)
- 3: lip_raise (입술 올리기)
- 4: lip_lower (입술 내리기)

키 사용법
--------
- 1~4 : 해당 라벨 수집 시작 (토글)
- 0   : 수집 일시정지
- s   : 지금까지 수집한 데이터를 저장 (기본: data/mouth_landmark_dataset.npz)
- c   : 누적 데이터 초기화
- q   : 종료 (종료 시 자동 저장 옵션 사용 가능)

저장 형식
--------
- labels        : (N,)     문자열 라벨
- ratios        : (N,)     입벌림 비율
- centers       : (N,)     입술 중심 위치
- landmarks     : (N, K)   평탄화된 랜드마크 좌표(K = 468*3 for mediapipe)
- timestamps    : (N,)     수집 시각(초)

수집된 데이터는 머신러닝 모델 학습을 위한 전처리 단계에 바로 활용할 수 있습니다.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from omegaconf import OmegaConf

try:
    from PIL import Image, ImageDraw, ImageFont

    PIL_AVAILABLE = True
except ImportError:  # Pillow optional for Korean text rendering
    PIL_AVAILABLE = False

from ptgaze.gaze_estimator import GazeEstimator
from ptgaze.point.mouth_open import mouth_metrics
from ptgaze.common.face import Face
import ptgaze.utils as utils

# 모델 다운로드 유틸
from ptgaze.utils import (
    download_ethxgaze_model,
    download_mpiifacegaze_model,
    download_mpiigaze_model,
)

# Visualization thresholds aligned with remote_client_pyqt
MOUTH_CLOSED_THRESHOLD = 0.045
MOUTH_SCROLL_OPEN_TOLERANCE = 0.008
MOUSE_CLICK_RATIO = 0.1
MOUTH_VISUAL_MAX_RATIO = 0.25

LABEL_MAP: Dict[str, Dict[str, str]] = {
    "1": {"id": "mouth_closed", "desc": "입 다물기"},
    "2": {"id": "mouth_open", "desc": "입 벌리기"},
    "3": {"id": "lip_raise", "desc": "입술 올리기"},
    "4": {"id": "lip_lower", "desc": "입술 내리기"},
}


@dataclass
class FontResource:
    path: Optional[Path]
    size: int
    freetype: Optional[Any] = None
    pil_font: Optional[ImageFont.FreeTypeFont] = None


@dataclass
class RecorderState:
    active_label: Optional[str] = None
    last_record_ts: float = 0.0
    capture_interval: float = 0.12
    sample_counts: Dict[str, int] = field(default_factory=lambda: collections.Counter())


@dataclass
class DatasetBuffer:
    labels: List[str] = field(default_factory=list)
    ratios: List[float] = field(default_factory=list)
    centers: List[float] = field(default_factory=list)
    landmarks: List[np.ndarray] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    landmark_shape: Optional[tuple] = None

    def append(self, label: str, ratio: float, center: float, landmarks: np.ndarray, ts: float) -> None:
        if self.landmark_shape is None:
            self.landmark_shape = landmarks.shape
        elif landmarks.shape != self.landmark_shape:
            # Skip samples with mismatched landmark dimensions.
            return

        self.labels.append(label)
        self.ratios.append(float(ratio))
        self.centers.append(float(center))
        self.landmarks.append(landmarks.astype(np.float32).reshape(-1))
        self.timestamps.append(float(ts))

    def clear(self) -> None:
        self.labels.clear()
        self.ratios.clear()
        self.centers.clear()
        self.landmarks.clear()
        self.timestamps.clear()
        self.landmark_shape = None

    def size(self) -> int:
        return len(self.labels)

    def counts_by_label(self) -> Dict[str, int]:
        counter = collections.Counter()
        for label in self.labels:
            counter[label] += 1
        return counter

    def save_npz(self, output_path: Path) -> Path:
        if self.size() == 0:
            raise RuntimeError("저장할 샘플이 없습니다.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output_path,
            labels=np.array(self.labels),
            ratios=np.asarray(self.ratios, dtype=np.float32),
            centers=np.asarray(self.centers, dtype=np.float32),
            landmarks=np.vstack(self.landmarks),
            timestamps=np.asarray(self.timestamps, dtype=np.float64),
        )
        meta = {
            "num_samples": self.size(),
            "counts": self.counts_by_label(),
            "landmark_shape": self.landmark_shape,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        meta_path = output_path.with_suffix(".json")
        with meta_path.open("w", encoding="utf-8") as f_meta:
            json.dump(meta, f_meta, ensure_ascii=False, indent=2)
        return output_path


def detect_default_font() -> Optional[Path]:
    if os.name == "nt":
        windir = Path(os.environ.get("WINDIR", "C:/Windows"))
        candidates = [
            windir / "Fonts" / "malgun.ttf",
            windir / "Fonts" / "malgunbd.ttf",
            windir / "Fonts" / "NanumGothic.ttf",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
    return None


def load_font(font_path: Optional[Path], font_size: int) -> FontResource:
    freetype_handle: Optional[Any] = None
    pil_font: Optional[ImageFont.FreeTypeFont] = None
    resolved_path = font_path

    if resolved_path is not None and not resolved_path.exists():
        print(f"[WARN] 지정한 폰트 파일을 찾을 수 없습니다: {resolved_path}")
        resolved_path = None

    if resolved_path is None:
        resolved_path = detect_default_font()
        if resolved_path is not None:
            print(f"[INFO] 기본 한글 폰트를 사용합니다: {resolved_path}")

    if resolved_path is not None and hasattr(cv2, "freetype"):
        try:
            freetype_handle = cv2.freetype.createFreeType2()
            freetype_handle.loadFontData(str(resolved_path))
            print(f"[INFO] FreeType 폰트를 사용합니다: {resolved_path}")
        except Exception:
            freetype_handle = None
            print("[WARN] FreeType 폰트 로드에 실패했습니다. Pillow로 대체합니다.")

    if resolved_path is not None and PIL_AVAILABLE:
        try:
            pil_font = ImageFont.truetype(str(resolved_path), font_size)
        except Exception:
            pil_font = None
            print("[WARN] Pillow 폰트 로드에 실패했습니다. 기본 영문 폰트를 사용합니다.")
    elif resolved_path is None and PIL_AVAILABLE:
        pil_font = ImageFont.load_default()

    if resolved_path is None and not PIL_AVAILABLE and not hasattr(cv2, "freetype"):
        print("[WARN] 한글 폰트와 Pillow를 찾을 수 없습니다. 영문 폰트만 표시됩니다.")

    return FontResource(path=resolved_path, size=font_size, freetype=freetype_handle, pil_font=pil_font)


def font_scale_from_size(font_size: int) -> float:
    return max(0.4, float(font_size) / 36.0)


def put_text(
    canvas,
    text: str,
    org: Tuple[int, int],
    color=(255, 255, 255),
    font: Optional[FontResource] = None,
    thickness: int = 1,
) -> None:
    if font is not None and font.freetype is not None:
        font.freetype.putText(canvas, text, org, font.size, color, thickness, cv2.LINE_AA, True)
        return

    if font is not None and font.pil_font is not None and PIL_AVAILABLE:
        pil_image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        draw.text(org, text, font=font.pil_font, fill=(color[2], color[1], color[0]))
        updated = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)
        np.copyto(canvas, updated)
        return

    scale = font_scale_from_size(font.size if font else 22)
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def mirror_point(point: Tuple[int, int], width: int) -> Tuple[int, int]:
    x, y = point
    return (max(0, width - x - 1), y)


def mirror_bbox(bbox: Tuple[Tuple[int, int], Tuple[int, int]], width: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    (x0, y0), (x1, y1) = bbox
    left = max(0, width - x1 - 1)
    right = max(0, width - x0 - 1)
    if left > right:
        left, right = right, left
    return ((left, y0), (right, y1))


def render_info_panel(
    state: RecorderState,
    buffer: DatasetBuffer,
    font: FontResource,
    metrics: Optional[Tuple[float, float]] = None,
    margins: Tuple[int, int] = (20, 16),
) -> np.ndarray:
    header_lines = [
        "[1]입다물기 [2]입벌리기 [3]입술올림 [4]입술내림",
        "[0]정지 [S]저장 [C]초기화 [Q]종료",
    ]

    label_order = [info["id"] for info in LABEL_MAP.values()]
    counts = buffer.counts_by_label()
    counts_text = " / ".join(f"{label}:{counts.get(label, 0)}" for label in label_order)

    summary_lines = [
        f"Active label: {state.active_label or '없음'}",
        f"총계: {buffer.size()}",
        f"Samples -> {counts_text}",
    ]

    if metrics is not None:
        ratio, center = metrics
        summary_lines.insert(1, f"현재 측정값: ratio={ratio:.3f}, center={center:.3f}")

    lines = header_lines + summary_lines

    legend_lines = [
        "게이지: 노랑=닫힘, 주황=허용범위, 빨강=클릭",
        f"스케일 상한: {MOUTH_VISUAL_MAX_RATIO:.3f}",
    ]

    if font.pil_font is not None and PIL_AVAILABLE:
        ascent, descent = font.pil_font.getmetrics()
        line_height = ascent + descent + 6
    else:
        line_height = int(font.size * 0.8) + 10

    legend_spacing = max(font.size + 6, line_height)
    bar_height = 18
    gauge_section = bar_height + 12 + legend_spacing * len(legend_lines)
    padding_x, padding_y = margins
    min_width = 480
    est_text_width = max(len(line) for line in lines) * max(font.size // 2, 8)
    width = max(min_width, est_text_width + padding_x * 2)
    height = padding_y * 2 + line_height * len(lines) + gauge_section
    panel = np.zeros((height, width, 3), dtype=np.uint8)

    text_bottom = padding_y
    if font.pil_font is not None and PIL_AVAILABLE:
        pil_image = Image.fromarray(panel)
        draw = ImageDraw.Draw(pil_image)
        y = padding_y
        for line in lines:
            draw.text((padding_x, y), line, font=font.pil_font, fill=(255, 255, 255))
            y += line_height
        text_bottom = y
        panel = np.asarray(pil_image).copy()
    else:
        scale = font_scale_from_size(font.size)
        y = padding_y + int(font.size * 0.8)
        for line in lines:
            cv2.putText(
                panel,
                line,
                (padding_x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            y += line_height
        text_bottom = y

    bar_left = padding_x
    bar_top = text_bottom + 8
    bar_width = width - padding_x * 2
    cv2.rectangle(
        panel,
        (bar_left, bar_top),
        (bar_left + bar_width, bar_top + bar_height),
        (60, 60, 60),
        thickness=-1,
    )

    ratio = metrics[0] if metrics is not None else 0.0
    ratio = float(max(0.0, ratio))
    scale = max(MOUTH_VISUAL_MAX_RATIO, 1e-6)
    fill_ratio = min(ratio / scale, 1.0)
    fill_end = bar_left + int(bar_width * fill_ratio)
    cv2.rectangle(
        panel,
        (bar_left, bar_top),
        (fill_end, bar_top + bar_height),
        (30, 180, 255),
        thickness=-1,
    )

    def draw_marker(value: float, color: Tuple[int, int, int], thickness: int) -> None:
        x_pos = bar_left + int(bar_width * min(max(value / scale, 0.0), 1.0))
        cv2.line(panel, (x_pos, bar_top - 4), (x_pos, bar_top + bar_height + 4), color, thickness)

    draw_marker(MOUTH_CLOSED_THRESHOLD, (0, 215, 255), 1)
    draw_marker(MOUTH_CLOSED_THRESHOLD + MOUTH_SCROLL_OPEN_TOLERANCE, (0, 140, 255), 1)
    draw_marker(MOUSE_CLICK_RATIO, (0, 0, 255), 2)

    legend_y = bar_top + bar_height + 20
    for idx, line in enumerate(legend_lines):
        put_text(
            panel,
            line,
            (padding_x, legend_y + idx * legend_spacing),
            font=font,
        )

    return panel


def draw_overlay(frame) -> None:
    """카메라 프레임 위에는 텍스트를 그리지 않습니다."""
    return None


def load_config(path: Path) -> OmegaConf:
    config = OmegaConf.load(str(path))
    utils.expanduser_all(config)
    mode = config.get("mode", "")
    if mode == "MPIIGaze":
        download_mpiigaze_model()
    elif mode == "MPIIFaceGaze":
        download_mpiifacegaze_model()
    elif mode == "ETH-XGaze":
        download_ethxgaze_model()
    return config


def choose_largest_face(faces) -> Optional[Face]:
    if not faces:
        return None
    max_size = -1.0
    largest = None
    for face in faces:
        bbox = face.bbox
        area = (bbox[1] - bbox[0])
        size = float(area[0] * area[1])
        if size > max_size:
            max_size = size
            largest = face
    return largest

def process_key(key: int, state: RecorderState) -> Optional[str]:
    if key == ord("0") or key == ord(" "):
        state.active_label = None
        return "수집 일시정지"
    if key in (ord("s"), ord("S")):
        return "save"
    if key in (ord("c"), ord("C")):
        return "clear"
    for key_char, info in LABEL_MAP.items():
        if key == ord(key_char):
            label_id = info["id"]
            if state.active_label == label_id:
                state.active_label = None
                return f"{info['desc']} 수집 해제"
            state.active_label = label_id
            state.last_record_ts = 0.0
            state.sample_counts[label_id] = state.sample_counts.get(label_id, 0)
            return f"{info['desc']} 수집 시작"
    return None


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="입술 상태 데이터 수집기")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("./ptgaze/data/configs/edge_xgaze.yaml"),
        help="ptgaze 설정 파일 경로",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/mouth_landmark_dataset.npz"),
        help="저장할 npz 파일 경로",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.12,
        help="프레임 간 수집 간격(초)",
    )
    parser.add_argument(
        "--mirror",
        dest="mirror",
        action="store_true",
        default=True,
        help="화면을 좌우 반전해서 출력합니다.",
    )
    parser.add_argument(
        "--no-mirror",
        dest="mirror",
        action="store_false",
        help="화면 좌우 반전을 비활성화합니다.",
    )
    parser.add_argument(
        "--font-path",
        type=Path,
        default=None,
        help="한국어 출력에 사용할 TrueType 폰트 경로 (기본: 시스템 자동 탐색)",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=22,
        help="폰트 크기(px). FreeType 또는 Pillow가 없으면 기본 영문 폰트를 사용",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    gaze_estimator = GazeEstimator(config)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, gaze_estimator.camera.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, gaze_estimator.camera.height)

    state = RecorderState(capture_interval=max(0.02, args.interval))
    buffer = DatasetBuffer()

    font_size = max(12, args.font_size)
    font_resource = load_font(args.font_path, font_size)
    if (
        font_resource.path is None
        and font_resource.freetype is None
        and (font_resource.pil_font is None or not PIL_AVAILABLE)
    ):
        print("[WARN] 한글 폰트를 렌더링할 수 없습니다. Pillow 설치 또는 --font-path 옵션을 확인하세요.")

    print("[INFO] 데이터 수집을 시작합니다. q 키로 종료하세요.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] 카메라 프레임을 읽을 수 없습니다.")
                break

            undistorted = cv2.undistort(
                frame,
                gaze_estimator.camera.camera_matrix,
                gaze_estimator.camera.dist_coefficients,
            )

            faces = gaze_estimator.detect_faces(undistorted)
            face = choose_largest_face(faces)

            landmarks = None
            ratio = 0.0
            center = 0.0
            metrics: Optional[Tuple[float, float]] = None
            current_ts = time.time()

            display_frame = undistorted.copy()
            face_bbox: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None

            if face is not None:
                # gaze 추정(랜드마크 업데이트)
                gaze_estimator.estimate_gaze(undistorted, face)
                if face.landmarks is not None:
                    landmarks = face.landmarks.copy()
                    ratio, center = mouth_metrics(face)
                    metrics = (ratio, center)

                    bbox = face.bbox.astype(int)
                    p0 = (int(bbox[0][0]), int(bbox[0][1]))
                    p1 = (int(bbox[1][0]), int(bbox[1][1]))
                    face_bbox = (p0, p1)

            if state.active_label and landmarks is not None:
                if current_ts - state.last_record_ts >= state.capture_interval:
                    buffer.append(state.active_label, ratio, center, landmarks, current_ts)
                    state.sample_counts[state.active_label] += 1
                    state.last_record_ts = current_ts

            if args.mirror:
                display_frame = cv2.flip(display_frame, 1)
                if face_bbox is not None:
                    face_bbox = mirror_bbox(face_bbox, display_frame.shape[1])

            if face_bbox is not None:
                cv2.rectangle(display_frame, face_bbox[0], face_bbox[1], (0, 255, 0), 2)

            draw_overlay(display_frame)
            cv2.imshow("Mouth Recorder", display_frame)

            info_panel = render_info_panel(state, buffer, font_resource, metrics=metrics)
            cv2.imshow("Mouth Recorder Info", info_panel)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[INFO] q 입력으로 종료합니다.")
                break
            elif key != 255:
                action = process_key(key, state)
                if action == "save":
                    try:
                        path = buffer.save_npz(args.output)
                        print(f"[INFO] 데이터 저장 완료 -> {path}")
                    except RuntimeError as exc:
                        print(f"[WARN] {exc}")
                elif action == "clear":
                    buffer.clear()
                    state.sample_counts.clear()
                    print("[INFO] 누적 데이터가 초기화되었습니다.")
                elif action is not None:
                    print(f"[INFO] {action}")

    finally:
        cap.release()
        cv2.destroyAllWindows()

        if buffer.size() > 0:
            try:
                path = buffer.save_npz(args.output)
                print(f"[INFO] 종료 시 자동 저장 -> {path}")
            except RuntimeError:
                pass

    return 0


if __name__ == "__main__":
    sys.exit(main())