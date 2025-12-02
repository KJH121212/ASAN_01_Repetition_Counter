import numpy as np
from typing import List, Dict, Optional

class RobustKalmanFilter:
    # 잡음 수준, 이상치 임계값을 인수로 받음
    def __init__(self, process_noise_std: float = 0.5, measurement_noise_std: float = 10.0, outlier_threshold: float = 15):

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
        
import numpy as np
from typing import Dict, Optional

# ... RobustKalmanFilter 클래스 정의 (이전과 동일) ...

def filter_and_extract_body_keypoints(
    kpts_array: np.ndarray, 
    outlier_threshold: float = 4.0  # <--- [추가] 임계값 인자
) -> np.ndarray:
    """
    키포인트 배열에 칼만 필터를 적용하여 스무딩 및 이상치 제거를 수행합니다.
    
    Args:
        kpts_array: (N, 17, 2) 형태의 입력 키포인트 배열.
        outlier_threshold: 이상치 판단을 위한 Mahalanobis 거리 임계값. (기본값 4.0)
                           낮을수록 더 민감하게 이상치를 제거합니다.
    """
    
    if kpts_array.ndim != 3 or kpts_array.shape[1] != 17 or kpts_array.shape[2] != 2:
        raise ValueError(f"입력 배열 형태가 잘못되었습니다: {kpts_array.shape}. (N, 17, 2)가 필요합니다.")

    num_frames = kpts_array.shape[0]
    target_kp_ids = list(range(5, 17))
    
    filtered_kpts_array = np.full(kpts_array.shape, np.nan, dtype=kpts_array.dtype)
    
    if num_frames < 2:
        print("경고: 프레임 수가 2개 미만입니다. 필터링을 수행하지 않고 5~16번 원본 데이터를 반환합니다.")
        return kpts_array[:, 5:17, :]

    # [수정] outlier_threshold 인자를 RobustKalmanFilter에 전달
    filters: Dict[int, RobustKalmanFilter] = {
        kp_id: RobustKalmanFilter(outlier_threshold=outlier_threshold) 
        for kp_id in target_kp_ids
    }
        
    print(f"--- 칼만 필터링 시작: Threshold={outlier_threshold}, Q=0.5, R=10.0 ---")
    
    # ----------------------------------------------------------------------
    # 1. 초기화 단계 (Frame 0 및 Frame 1)
    # ----------------------------------------------------------------------
    for kp_id in target_kp_ids:
        kf = filters[kp_id]
        z_0 = kpts_array[0, kp_id]
        z_1 = kpts_array[1, kp_id]
        
        # 1-1. Frame 0: 위치 초기화
        if not np.any(np.isnan(z_0)):
            kf.x[:2] = z_0.reshape(2, 1)
            filtered_kpts_array[0, kp_id] = z_0
        
        # 1-2. Frame 1: 속도 초기화
        if not np.any(np.isnan(z_1)):
            if not np.any(np.isnan(z_0)):
                kf.x[2:] = (z_1 - z_0).reshape(2, 1) 
            filtered_kpts_array[1, kp_id] = z_1
            
    # ----------------------------------------------------------------------
    # 2. 메인 필터링 루프 (Frame 2부터 시작)
    # ----------------------------------------------------------------------
    for f in range(2, num_frames):
        for kp_id in target_kp_ids:
            kf = filters[kp_id]
            z_k = kpts_array[f, kp_id]
            
            filtered_coords = kf.update(z_k)
            filtered_kpts_array[f, kp_id] = filtered_coords
            
    print("--- 칼만 필터링 완료 ---")

    final_output_array = filtered_kpts_array[:, 5:17, :]
    
    return final_output_array