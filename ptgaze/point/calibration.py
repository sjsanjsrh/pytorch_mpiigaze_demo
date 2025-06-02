import cv2
import numpy as np
from scipy.spatial.distance import cdist
from ..common import Face, FacePartsName
from ..gaze_estimator import GazeEstimator
from .kalman_filter import KalmanFilter2D
import time
import math

class Calibration:
    def __init__(self, gaze_estimator: GazeEstimator, screen_width, screen_height, 
                 process_noise=1e-9):
        
        self.gaze_estimator = gaze_estimator 

        self.screen_width = screen_width
        self.screen_height = screen_height

        self.filter = KalmanFilter2D(process_noise=process_noise)
        self.last_time = 0

        self.trs_matrix = None

        self.eye_vector = np.array([0, 0])

    def calc_eye_2d_vector(self, face: Face) -> np.ndarray:
        """
        눈의 각도를 계산합니다.
        """
        eye_angles = np.zeros((2, 3), dtype=np.float32)
        eye_positions = np.zeros((2, 3), dtype=np.float32)
        eye_composition = np.zeros((2, 3), dtype=np.float32)

        for i, key in enumerate([FacePartsName.REYE, FacePartsName.LEYE]):
            eye = getattr(face, key.name.lower())
            eye_angles[i] = eye.normalized_gaze_vector
            eye_angles[i][0] = -eye_angles[i][0]
            eye_composition[i] = eye_angles[i] - eye_positions[i]

        cross_product = np.cross(eye_composition[0], eye_composition[1])
        res = cross_product[:2]/-cross_product[2]

        res = np.sum(eye_composition, axis=0)
        res = res[:2] / -res[2]

        self.eye_vector = res

        return res
    
    def calc_noize(self, origin_points, collected_points, R = 1.0):
        points = np.empty((0, 2), dtype=np.float32)
        for i in range(len(origin_points)):
            points = np.append(points, collected_points[i] - origin_points[i])
        points = np.array(points, dtype=np.float32)
        points = points.reshape(-1, 2)
        self.filter.calc_noize(points, R)



    def calc_filtered_centers(self, points) -> np.ndarray:
        """
        튀어있는 점(outliers)을 제거하고 중앙점을 계산합니다.
        """
        # numpy 배열로 변환
        points = np.array(points, dtype=np.float32)

        # 중앙점 계산 (초기값)
        center = np.mean(points, axis=0)

        # 거리 계산 (중앙점과 각 점 사이의 거리)
        distances = cdist(points, [center])

        # 거리의 IQR(사분위 범위) 계산
        q1, q3 = np.percentile(distances, [25, 75])
        iqr = q3 - q1
        threshold = q3 + 1.5 * iqr  # 이상치 기준

        # 이상치가 아닌 점들만 필터링
        inliers = points[distances.flatten() <= threshold]

        # 필터링된 점들로 새로운 중앙점 계산
        refined_center = np.mean(inliers, axis=0)

        return np.array([refined_center[0], refined_center[1]])


    def calc_trs_matrix(self, origins, dests) -> np.ndarray:
        """
        원본 점과 변환된 점을 기반으로 변환 행렬을 계산합니다.
        """
        # numpy 배열로 변환
        origins = np.array(origins, dtype=np.float32)
        dests = np.array(dests, dtype=np.float32)

        # 변환 행렬 계산
        self.trs_matrix  = cv2.getPerspectiveTransform(dests, origins)
        return self.trs_matrix 

    def calc_trs_transform(self, point) -> np.ndarray:
        """
        변환 행렬을 사용하여 점을 변환합니다.
        """
        # numpy 배열로 변환
        transformed_point = cv2.perspectiveTransform(np.array([[point]], dtype=np.float32), self.trs_matrix )[0][0]
        return np.array([int(transformed_point[0]), int(transformed_point[1])])

    def calc_filtered_point(self, point) -> np.ndarray:
        if self.last_time == 0:
            self.last_time = time.time()
        dt = time.time() - self.last_time
        
        return self.filter.correct(point, dt = dt)[0]
        

