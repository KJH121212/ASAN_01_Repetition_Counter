import numpy as np
from typing import List, Dict, Tuple, Any, Optional

def calculate_angle(a, b, c):
    a = np.array(a) # 첫 번째 점을 numpy 배열로 변환
    b = np.array(b) # 중간 점을 numpy 배열로 변환
    c = np.array(c) # 세 번째 점을 numpy 배열로 변환

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0]) # 아크탄젠트를 이용해 라디안 각도 계산
    angle = np.abs(radians*180.0/np.pi) # 라디안을 도(degree) 단위로 변환하고 절대값 취함

    if angle > 180.0: # 각도가 180도를 넘어가면
        angle = 360 - angle # 360도에서 뺀 값을 사용하여 내각을 구함

    return angle # 계산된 각도 반환


# --- 1. 제공해주신 Pose 유사도 검사 함수 ---
def calculate_pose_distance(kpts_t: List[List[float]], kpts_t_plus_1: List[List[float]]) -> float:
    
    # 1. 키포인트 벡터 평탄화 (Flatten)
    # 데이터가 (N, 3)일 수도 있고 (N, 2)일 수도 있으므로 앞의 2개(x, y)만 사용하도록 슬라이싱 처리 추가 권장
    vec_t = np.array(kpts_t)[:, :2].flatten()
    vec_t_plus_1 = np.array(kpts_t_plus_1)[:, :2].flatten()

    # 2. 정규화 (중심점 이동 - Center Shift Normalization)
    if len(vec_t) != len(vec_t_plus_1) or len(vec_t) == 0:
        return float('inf')

    center_t = np.mean(vec_t.reshape(-1, 2), axis=0)
    center_t_plus_1 = np.mean(vec_t_plus_1.reshape(-1, 2), axis=0)
    
    # 정규화된 벡터 생성 (중심점 위치 이동)
    norm_vec_t = vec_t.reshape(-1, 2) - center_t
    norm_vec_t_plus_1 = vec_t_plus_1.reshape(-1, 2) - center_t_plus_1
    
    # 3. L2 거리 (유클리드 거리) 계산
    return euclidean(norm_vec_t.flatten(), norm_vec_t_plus_1.flatten())


# --- 2. 헬퍼 함수: IOU 계산 ---
def calculate_iou(boxA: List[float], boxB: List[float]) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def get_distance_from_center(bbox, frame_w, frame_h):
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    return np.sqrt((cx - frame_w/2)**2 + (cy - frame_h/2)**2)
