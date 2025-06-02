import numpy as np

class KalmanFilter2D:
    def __init__(self, dt=1.0, process_noise=1e-2, measurement_noise=5.0):
        # 시간 간격
        self.dt = dt

        # 상태 벡터: [px, py, vx, vy]
        self.x = np.zeros((4, 1))

        # 오차 공분산 행렬
        self.P = np.eye(4) * 1000.0

        # 상태 전이 행렬 A
        self.A = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]
        ])

        # 측정 행렬 H
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # 프로세스 잡음 공분산 Q
        self.Q = np.eye(4) * process_noise

        # 측정 잡음 공분산 R
        self.R = np.eye(2) * measurement_noise

        # 항등 행렬
        self.I = np.eye(4)

    def predict(self, dt):
        # 상태 예측
        self.x = self.A @ self.x
        
        # dt에 따라 상태 전이 행렬 A 업데이트
        self.A = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]
        ])

        # 공분산 예측
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x[:2]

    def update(self, z):
        """
        z: np.array([[px], [py]])  측정값
        """
        # 잔차 계산
        y = z - self.H @ self.x

        # 칼만 이득 계산
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # 상태 업데이트
        self.x = self.x + K @ y

        # 공분산 업데이트
        self.P = (self.I - K @ self.H) @ self.P

        return self.x[:2]

    def correct(self, z, dt=1.0):
        """
        예측 + 보정 한 번에 수행
        """
        self.dt = dt
        self.predict(dt)
        return self.update(z)

    def calc_noize(self, points, R = 1.0):
        variances = np.var(points, axis=0)
        self.R = np.diag(variances)*R