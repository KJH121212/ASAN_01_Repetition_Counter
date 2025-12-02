import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Optional, List, Any, Tuple

# --- 상수 설정 (이전과 동일) ---
COLOR_L = (255, 0, 0)       # Blue
COLOR_R = (0, 0, 255)       # Red
COLOR_NEUTRAL = (0, 255, 0) # Green
COLOR_SK = (50, 50, 50)     # Dark Gray

YOLO12_CONFIG = {
    "LEFT_POINTS": [0, 2, 4, 6, 8, 10],
    "RIGHT_POINTS": [1, 3, 5, 7, 9, 11],
    "EXCLUDE_POINTS": [],
    "SKELETON_LINKS": [
        (10, 8), (8, 6), (11, 9), (9, 7), (6, 7),
        (0, 6), (1, 7), (0, 1), (0, 2), (2, 4), (1, 3), (3, 5)
    ]
}

COCO17_CONFIG = {
    "LEFT_POINTS": [5, 7, 9, 11, 13, 15],
    "RIGHT_POINTS": [6, 8, 10, 12, 14, 16],
    "EXCLUDE_POINTS": [0, 1, 2, 3, 4], 
    "SKELETON_LINKS": [
        (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
        (5, 11), (6, 12), (5, 6),
        (5, 7), (7, 9), (6, 8), (8, 10)
    ]
}

def render_skeleton_video_from_array(
    final_data: np.ndarray,      
    FRAME_PATH: str,           
    OUTPUT_PATH: str,          
    start_frame: int,          
    end_frame: int,            
    fps: int = 30,
    kp_radius: int = 4,
    line_thickness: int = 2,
    flip_horizontal: bool = True
):
    """
    NAS 환경 최적화: 파일 존재 여부를 미리 전수 조사하지 않고 즉시 처리를 시작합니다.
    """
    
    frame_dir_path = Path(FRAME_PATH)
    out_mp4_path = Path(OUTPUT_PATH)
    out_mp4_path.parent.mkdir(parents=True, exist_ok=True)
    
    num_expected_frames = end_frame - start_frame + 1
    num_kpts_frames = final_data.shape[0]
    num_kpts_per_frame = final_data.shape[1]
    
    # --- 설정 선택 ---
    if num_kpts_per_frame == 12:
        config = YOLO12_CONFIG
    elif num_kpts_per_frame == 17:
        config = COCO17_CONFIG
    else:
        print(f"[ERROR] 지원하지 않는 키포인트 개수입니다: {num_kpts_per_frame}")
        return

    LEFT_POINTS = config["LEFT_POINTS"]
    RIGHT_POINTS = config["RIGHT_POINTS"]
    EXCLUDE_POINTS = config["EXCLUDE_POINTS"]
    SKELETON_LINKS = config["SKELETON_LINKS"]

    if num_kpts_frames != num_expected_frames:
        print(f"[ERROR] 프레임 수 불일치! 예상: {num_expected_frames}, 실제: {num_kpts_frames}")
        return

    # --- ⚡ 최적화: 첫 번째 유효 프레임만 빠르게 찾기 (전수 조사 생략) ---
    first_frame_path = None
    sample_img = None
    
    # 앞부분 최대 100프레임까지만 확인해서 첫 이미지를 찾음 (없으면 실패 처리)
    search_limit = min(100, num_kpts_frames)
    for i in range(search_limit):
        chk_idx = start_frame + i
        chk_path = frame_dir_path / f"{chk_idx:06d}.jpg"
        if chk_path.exists():
            sample_img = cv2.imread(str(chk_path))
            if sample_img is not None:
                first_frame_path = chk_path
                break
    
    if sample_img is None:
        print(f"[ERROR] 시작 프레임 근처에서 유효한 이미지를 찾을 수 없습니다.")
        return

    h, w = sample_img.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_mp4_path), fourcc, fps, (w, h))
    
    print(f"[INFO] 렌더링 시작: {out_mp4_path.name}")

    # --- 메인 루프 ---
    for kpt_idx in tqdm(range(num_kpts_frames), desc="Encoding Video", unit="frame"):
        
        frame_num = start_frame + kpt_idx
        # 문자열 연산 최적화 (Path 객체 연산 줄이기)
        frame_path_str = f"{FRAME_PATH}/{frame_num:06d}.jpg"

        # 이미지 읽기
        frame = cv2.imread(frame_path_str)
        
        if frame is None:
            # 이미지가 없으면 빈 화면(검은색)이라도 만들어서 프레임 수를 맞출지, 건너뛸지 결정
            # 여기서는 건너뛰되 경고를 줄이기 위해 pass
            continue
             
        kpts = final_data[kpt_idx] # 원본 좌표 사용 (마지막에 뒤집음)
        
        # 1. 스켈레톤 그리기
        for i, j in SKELETON_LINKS:
            if i in EXCLUDE_POINTS or j in EXCLUDE_POINTS: continue
            if i >= num_kpts_per_frame or j >= num_kpts_per_frame: continue
            
            # NaN 체크
            if np.isnan(kpts[i, 0]) or np.isnan(kpts[i, 1]) or \
               np.isnan(kpts[j, 0]) or np.isnan(kpts[j, 1]):
                continue

            pt1 = (int(kpts[i, 0]), int(kpts[i, 1]))
            pt2 = (int(kpts[j, 0]), int(kpts[j, 1]))
            
            # 화면 밖 좌표 체크 (간단히)
            if pt1[0] < 0 or pt1[1] < 0 or pt2[0] < 0 or pt2[1] < 0: continue

            cv2.line(frame, pt1, pt2, COLOR_SK, line_thickness)

        # 2. 키포인트 그리기
        for idx, pt in enumerate(kpts):
            if idx in EXCLUDE_POINTS: continue
            if np.isnan(pt[0]) or np.isnan(pt[1]): continue
            
            x, y = int(pt[0]), int(pt[1])
            if x < 0 or y < 0: continue
                
            if idx in LEFT_POINTS: color = COLOR_L
            elif idx in RIGHT_POINTS: color = COLOR_R
            else: color = COLOR_NEUTRAL
                
            cv2.circle(frame, (x, y), kp_radius, color, -1)
            
        # 3. 텍스트 및 반전
        legend_text = f"L: Blue | R: Red | Frame: {frame_num}"
        cv2.putText(frame, legend_text, (20, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        if flip_horizontal:
            # 모든 그리기가 끝난 후 이미지 전체를 반전 (텍스트도 반전됨)
            frame = cv2.flip(frame, 1)

        writer.write(frame)

    writer.release()
    print(f"✅ 완료.")