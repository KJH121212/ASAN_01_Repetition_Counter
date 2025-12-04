import numpy as np

def kpts_to_bbox(kpts):
    """
    (K, 3) 또는 (K, 2) 키포인트에서 BBox [x1, y1, x2, y2]를 추출합니다.
    유효한(NaN이 아닌) 좌표들의 min/max를 사용합니다.
    """
    # x, y 좌표만 추출
    pts = kpts[:, :2]
    
    # 유효한 좌표 마스크
    valid = ~np.isnan(pts[:, 0])
    
    if np.sum(valid) == 0:
        return None # 유효한 키포인트가 없음

    valid_pts = pts[valid]
    x1, y1 = np.min(valid_pts, axis=0)
    x2, y2 = np.max(valid_pts, axis=0)
    
    return np.array([x1, y1, x2, y2])