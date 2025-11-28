import numpy as np
from typing import Dict

class RobustKalmanFilter:
    # 잡음 수준, 이상치 임계값을 인수로 받음
    def __init__(self, process_noise_std: float = 0.5, measurement_noise_std: float = 10.0, outlier_threshold: float = 4.0):
        self.dt = 1.0                           # 시간 간격(프레임 단위, 1로 고정)
        self.threshold = outlier_threshold      # 마할라노비스 거리 제곱 임계값

        self.x = np.zeros((4, 1))               # 상태 벡터 [x,y,vx,vy]로 초기화
        self.P = np.eye(4) * 10                 # 오차 공분산 행렬 (4*4): 초기 불확실성을 설정 (높은 값으로 시작)

        # 상태 전이 행렬 (F): 다음 상태를 예측하는 행렬 (일정 속도 모델)
        self.F = np.array([
            [1, 0, self.dt, 0], [0, 1, 0, self.dt],         # x' = x + vx*dt, y' = y + vy*dt
            [0, 0, 1, 0], [0, 0, 0, 1]                      # vx' = vx, vy' = vy
        ])
        # 관측 행렬 (H): 상태 벡터 [x, y, vx, vy] 중 관측 가능한 [x, y]만 추출 (2x4)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        
        q = process_noise_std**2
        self.Q = np.array([
            [q*self.dt**4/4, 0, q*self.dt**3/2, 0], [0, q*self.dt**4/4, 0, q*self.dt**3/2],
            [q*self.dt**3/2, 0, q*self.dt**2, 0], [0, q*self.dt**3/2, 0, q*self.dt**2]
        ])

        r = measurement_noise_std**2
        self.R = np.eye(2) * r 

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        
    def update(self, z: np.ndarray) -> np.ndarray:
        
        x_prev = self.x.copy() 
        P_prev = self.P.copy() 

        self.predict() 
        
        if np.any(np.isnan(z)):
            return self.x[:2].flatten() 

        y = z.reshape(2, 1) - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        
        try:
            S_inv = np.linalg.inv(S)
            mahalanobis_sq = y.T @ S_inv @ y
        except np.linalg.LinAlgError:
            mahalanobis_sq = self.threshold + 1.0 

        is_outlier = mahalanobis_sq > self.threshold
        
        if is_outlier:
            self.x = x_prev
            self.P = P_prev
            return self.x[:2].flatten()
        else:
            K = self.P @ self.H.T @ S_inv
            self.x = self.x + K @ y 
            I = np.eye(4)
            self.P = (I - K @ self.H) @ self.P 
            
            return self.x[:2].flatten()