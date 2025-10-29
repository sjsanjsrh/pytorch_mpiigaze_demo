import logging
from typing import Tuple

from ..common.face import Face

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _extract_landmarks(face: Face):
    if face.landmarks is None:
        logger.warning(
            "Not enough landmarks or landmarks is None for MediaPipe. Available: %s",
            len(face.landmarks) if face.landmarks is not None else "None",
        )
        return None

    # --- MediaPipe Face Mesh 기준 랜드마크 인덱스 ---
    upper_lip_point_index = 13  # 윗입술 안쪽 중앙
    lower_lip_point_index = 14  # 아랫입술 안쪽 중앙
    nose_tip_index = 1          # 코끝
    chin_index = 152            # 턱의 가장 아래쪽 중앙

    try:
        upper_lip_y = face.landmarks[upper_lip_point_index, 1]
        lower_lip_y = face.landmarks[lower_lip_point_index, 1]
        nose_tip_y = face.landmarks[nose_tip_index, 1]
        chin_y = face.landmarks[chin_index, 1]
        return upper_lip_y, lower_lip_y, nose_tip_y, chin_y
    except IndexError:
        logger.error(
            "Landmark INDEX OUT OF BOUNDS for MediaPipe. Landmarks available: %s."
            " Confirm that landmark indices (%s, %s, %s, %s) are valid.",
            len(face.landmarks),
            upper_lip_point_index,
            lower_lip_point_index,
            nose_tip_index,
            chin_index,
        )
        return None


def mouth_metrics(face: Face) -> Tuple[float, float]:
    """Return mouth open ratio and normalized lip-center position."""
    values = _extract_landmarks(face)
    if values is None:
        return 0.0, 0.0

    upper_lip_y, lower_lip_y, nose_tip_y, chin_y = values

    mouth_height = abs(lower_lip_y - upper_lip_y)
    face_segment_height = abs(chin_y - nose_tip_y)
    if face_segment_height < 1e-6:
        logger.warning(
            "FaceSegmentHeight (%s) is too small. Returning default metrics.",
            face_segment_height,
        )
        return 0.0, 0.0

    ratio = mouth_height / face_segment_height
    center_y = (upper_lip_y + lower_lip_y) / 2.0
    normalized_center = (center_y - nose_tip_y) / face_segment_height

    logger.debug(
        "Mouth metrics -> height: %.4f, segment: %.4f, ratio: %.4f, center: %.4f",
        mouth_height,
        face_segment_height,
        ratio,
        normalized_center,
    )
    return ratio, normalized_center


def mouth_open_ratio(face: Face) -> float:
    ratio, _ = mouth_metrics(face)
    return ratio


def mouth_center_position(face: Face) -> float:
    _, center = mouth_metrics(face)
    return center
