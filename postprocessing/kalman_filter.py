import numpy as np

class BasicKalmanFilter:
    def __init__(self, 
                 dt: float = 1.0,               # 시간 간격 (프레임 단위면 1.0)
                 process_noise_std: float = 0.5,# Q: 작을수록 '등속도'를 강하게 믿음 (부드러워짐)
                 measurement_noise_std: float = 3.0): # R: 작을수록 '측정값'을 강하게 믿음 (반응 빨라짐)
        
        self.dt = dt
        self.initialized = False

        # 1. 상태 벡터 (x) [x, y, vx, vy] (4x1)
        self.x = np.zeros((4, 1))
        
        # 2. 오차 공분산 행렬 (P) (4x4)
        # 초기엔 아무것도 모르므로 불확실성을 크게 잡음
        self.P = np.eye(4) * 100.0
        
        # 3. 상태 전이 행렬 (F) (4x4) - 물리 모델
        # x_new = x + vx * dt
        # y_new = y + vy * dt
        self.F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # 4. 관측 행렬 (H) (2x4) - [x, y, vx, vy] -> [x, y]
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # 5. 프로세스 노이즈 (Q) (4x4)
        # 시스템(물리 모델)이 가질 수 있는 급격한 속도 변화 등의 불확실성
        self.Q = np.eye(4) * (process_noise_std ** 2)

        # 6. 측정 잡음 (R) (2x2)
        # 센서(YOLO)가 가지는 오차 범위
        self.R = np.eye(2) * (measurement_noise_std ** 2)

    def predict(self) -> np.ndarray:
        """
        [Time Update] 현재 속도로 다음 위치를 예측합니다.
        """
        # x = Fx
        self.x = self.F @ self.x
        # P = FPF' + Q
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.x[:2].flatten() # 예측된 [x, y] 반환

    def update(self, z: np.ndarray) -> np.ndarray:
        """
        [Measurement Update] 관측값으로 상태를 보정합니다.
        z: 측정된 [x, y] 좌표 (NaN이 들어오면 무시)
        """
        # 0. 초기화 로직 (첫 데이터가 들어오면 위치 설정)
        if not self.initialized:
            if not np.any(np.isnan(z)):
                self.x[:2] = z.reshape(2, 1) # 위치 초기화
                self.x[2:] = 0               # 속도는 0으로 가정
                self.initialized = True
            return self.x[:2].flatten()

        # 1. NaN(결측치) 처리: 업데이트 건너뜀
        if np.any(np.isnan(z)):
            return self.x[:2].flatten()

        # 2. 칼만 게인 계산 및 업데이트
        y = z.reshape(2, 1) - (self.H @ self.x)     # 잔차 (Measurement - Prediction)
        S = self.H @ self.P @ self.H.T + self.R     # 잔차 공분산
        
        try:
            # K = P H' S^-1
            K = self.P @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = np.zeros((4, 2))

        # 상태 보정: x = x + Ky
        self.x = self.x + K @ y
        
        # 오차 공분산 보정: P = (I - KH)P
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P
        
        return self.x[:2].flatten() # 보정된 [x, y] 반환