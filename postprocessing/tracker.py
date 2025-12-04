import cv2
import numpy as np
import os
import glob
import re

# =============================================================================
# 1. Tracker Class (로직 코어)
# =============================================================================
class HybridTracker:
    def __init__(self, img_width, img_height):
        # 화면 중앙 좌표
        self.center = np.array([img_width / 2, img_height / 2])
        
        # 상태 변수
        self.patient_hist = None  # 환자 색상 정보 (ID)
        self.is_locked = False    # 환자 포착 여부
        self.last_pos = self.center
        
        # 가중치 (색상 60%, 거리 40%)
        self.dist_weight = 0.4
        self.color_weight = 0.6 

    def get_color_hist(self, img, kpts):
        """키포인트 영역의 색상 히스토그램 추출"""
        # 유효한 키포인트만 사용
        valid = kpts[kpts[:, 2] > 0.1, :2]
        if len(valid) < 3: return None
        
        # BBox 계산 (여유 공간 5px)
        x1, y1 = np.min(valid, axis=0) - 5
        x2, y2 = np.max(valid, axis=0) + 5
        
        h, w, _ = img.shape
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        
        crop = img[y1:y2, x1:x2]
        if crop.size == 0: return None
        
        # HSV 히스토그램 계산
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [18, 25], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist

    def update(self, frame, detections):
        """
        한 프레임의 데이터를 받아 환자 키포인트를 반환
        """
        # 1. [초기화 단계] 아직 환자를 못 찾았으면 화면 중앙 검색
        if not self.is_locked:
            best_idx = -1
            best_dist = float('inf')
            
            for i, kpts in enumerate(detections):
                valid = kpts[kpts[:, 2] > 0.1, :2]
                if len(valid) < 3: continue
                
                center_pt = np.mean(valid, axis=0)
                dist = np.linalg.norm(center_pt - self.center)
                
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
            
            # 중앙 반경 200px 이내면 환자로 등록 (Lock)
            if best_idx != -1 and best_dist < 200:
                hist = self.get_color_hist(frame, detections[best_idx])
                if hist is not None:
                    self.patient_hist = hist
                    self.is_locked = True
                    self.last_pos = np.mean(detections[best_idx][:, :2], axis=0)
                    return detections[best_idx]
            return None

        # 2. [추적 단계] 색상 + 거리 비교
        best_score = -1
        best_idx = -1
        
        for i, kpts in enumerate(detections):
            valid = kpts[kpts[:, 2] > 0.1, :2]
            if len(valid) < 3: continue
            
            # A. 거리 점수 (가까울수록 높음)
            curr_pos = np.mean(valid, axis=0)
            dist = np.linalg.norm(curr_pos - self.last_pos)
            dist_score = max(0, 1 - (dist / 300.0))
            
            # B. 색상 점수 (옷 색깔 비슷할수록 높음)
            curr_hist = self.get_color_hist(frame, kpts)
            if curr_hist is None:
                color_score = 0
            else:
                color_score = cv2.compareHist(self.patient_hist, curr_hist, cv2.HISTCMP_CORREL)
                color_score = max(0, color_score)
            
            # C. 종합 점수
            final_score = (self.dist_weight * dist_score) + (self.color_weight * color_score)
            
            if final_score > best_score:
                best_score = final_score
                best_idx = i
        
        # 점수가 0.4 이상이면 환자로 판단
        if best_idx != -1 and best_score > 0.4:
            found = detections[best_idx]
            valid = found[found[:, 2] > 0.1, :2]
            self.last_pos = np.mean(valid, axis=0) # 위치 업데이트
            return found
        else:
            return None # 놓침 (Occlusion 등)


# =============================================================================
# 2. Main Function
# =============================================================================
def extract_patient_data(frame_dir, kpt_data):
    """
    이미지 폴더와 키포인트 데이터를 입력받아 환자 1명의 궤적을 추출합니다.
    
    Args:
        frame_dir (str): 이미지가 들어있는 폴더 경로
        kpt_data (np.ndarray): (Frame, N, 17, 3) 형태의 원본 데이터
        
    Returns:
        patient_data (np.ndarray): (Frame, 1, 17, 3) 형태의 환자 데이터
    """
    
    # 1. 이미지 파일 리스트 로드 및 정렬 (frame_0.jpg, frame_1.jpg ...)
    image_files = sorted(
        glob.glob(os.path.join(frame_dir, "*.jpg")), 
        key=lambda x: int(re.findall(r'\d+', os.path.basename(x))[-1])
    )
    
    if not image_files:
        print("[Error] 해당 폴더에 이미지가 없습니다.")
        return None

    # 2. 첫 이미지로 해상도 확인 및 트래커 초기화
    first_img = cv2.imread(image_files[0])
    h, w, _ = first_img.shape
    
    tracker = HybridTracker(img_width=w, img_height=h)
    
    # 데이터 길이 맞추기 (이미지 수와 데이터 수 중 작은 쪽 기준)
    num_frames = min(len(image_files), kpt_data.shape[0])
    
    # 결과 담을 그릇 (1명 고정)
    patient_data = np.full((num_frames, 1, 17, 3), np.nan)
    
    print(f"[Info] Tracking started using frames... (Total: {num_frames})")

    # 3. 프레임 반복
    for t in range(num_frames):
        # 이미지 로드
        frame = cv2.imread(image_files[t])
        if frame is None: break
        
        # 현재 프레임의 모든 감지 데이터
        detections = kpt_data[t]
        
        # 트래커 업데이트 -> 환자 데이터 반환
        result = tracker.update(frame, detections)
        
        # 결과가 있으면 저장 (없으면 NaN 유지)
        if result is not None:
            patient_data[t, 0] = result
            
        if t % 100 == 0:
            print(f"Processing... {t}/{num_frames}", end='\r')

    print(f"\n[Info] Extraction Complete. Shape: {patient_data.shape}")
    return patient_data