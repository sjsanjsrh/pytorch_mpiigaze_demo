import logging
from ..common.face import Face

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def mouth_open_ratio( face: Face) -> float:
    if face.landmarks is None:
        logger.warning(f"Not enough landmarks or landmarks is None for MediaPipe. Available: {len(face.landmarks) if face.landmarks is not None else 'None'}")
        return 0.0

    try:
        # --- MediaPipe Face Mesh 기준 랜드마크 인덱스 ---
        upper_lip_point_index = 13  # 윗입술 안쪽 중앙
        lower_lip_point_index = 14  # 아랫입술 안쪽 중앙
        nose_tip_index = 1        # 코끝
        chin_index = 152          # 턱의 가장 아래쪽 중앙
        # --- 인덱스 변경 끝 ---

        upper_lip_y = face.landmarks[upper_lip_point_index, 1]
        lower_lip_y = face.landmarks[lower_lip_point_index, 1]
        nose_tip_y = face.landmarks[nose_tip_index, 1]
        chin_y = face.landmarks[chin_index, 1]

        mouth_height = abs(lower_lip_y - upper_lip_y)
        # 정규화를 위한 기준 길이: 코끝과 턱끝 사이의 수직 거리
        # 또는 얼굴 전체의 높이를 나타내는 다른 랜드마크 조합을 사용할 수도 있습니다.
        # 예: 얼굴 최상단(10)과 최하단(152) 랜드마크 간의 거리 등
        face_segment_height = abs(chin_y - nose_tip_y)

        # --- 내부 값 상세 로깅 (디버깅에 여전히 유용) ---
        logger.info(f"--- Inside _calculate_mouth_open_ratio (MediaPipe Indices) ---")
        logger.info(f"Landmark Y Coords -> UpperLip(idx {upper_lip_point_index}): {upper_lip_y:.2f}, LowerLip(idx {lower_lip_point_index}): {lower_lip_y:.2f}")
        logger.info(f"Landmark Y Coords for Seg -> NoseTip(idx {nose_tip_index}): {nose_tip_y:.2f}, Chin(idx {chin_index}): {chin_y:.2f}")
        logger.info(f"Calculated Heights -> MouthHeight: {mouth_height:.2f}, FaceSegmentHeight: {face_segment_height:.2f}")

        if face_segment_height < 1e-6: # 매우 작은 값으로 인한 나눗셈 오류 방지
            logger.warning(f"FaceSegmentHeight ({face_segment_height:.2f}) is too small. Returning 0.0 for ratio.")
            return 0.0

        ratio = mouth_height / face_segment_height
        logger.info(f"Calculated Ratio: {ratio:.3f}")
        return ratio
    except IndexError:
        logger.error(f"Landmark INDEX OUT OF BOUNDS for MediaPipe. Landmarks available: {len(face.landmarks)}. "
                    f"Confirm that landmark indices ({upper_lip_point_index}, {lower_lip_point_index}, {nose_tip_index}, {chin_index}) are valid.")
        return -1.0 # 오류 발생 시 음수 값 등으로 구분
    except Exception as e:
        logger.error(f"Exception in _calculate_mouth_open_ratio: {e}")
        return -2.0 # 오류 발생 시 음수 값 등으로 구분