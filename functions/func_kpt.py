import json
import numpy as np
from pathlib import Path
import re # 정규 표현식 모듈 추가
from typing import List, Tuple, Dict, Any, Optional

import os

# load_kpt : json을 읽어와 dict 형태로 저장

def load_kpt(
    json_dir: str,
    start_frame: Optional[int] = None, # Optional로 변경
    end_frame: Optional[int] = None    # Optional로 변경
) -> Dict[str, Any]:
   
    result_data = {
        'meta_info': {},
        'frame_data': []
    }
    
    base_path = Path(json_dir)
    meta_loaded = False
    
    # 1. 프레임 번호 범위 결정 로직
    if start_frame is None or end_frame is None:
        # start_frame 또는 end_frame이 지정되지 않은 경우, 디렉토리의 모든 파일을 탐색
        json_files = sorted(base_path.glob("*.json"))
        
        if not json_files:
            print(f"[WARN] No JSON files found in {json_dir}")
            return result_data

        # 파일명에서 숫자 부분 (000000)을 추출하여 프레임 번호 결정
        frame_numbers = []
        filename_pattern = re.compile(r"(\d{6})\.json$")
        
        for f in json_files:
            match = filename_pattern.search(f.name)
            if match:
                frame_numbers.append(int(match.group(1)))

        if not frame_numbers:
             print(f"[WARN] Could not parse frame numbers from JSON files in {json_dir}")
             return result_data

        determined_start = min(frame_numbers)
        determined_end = max(frame_numbers)
        
        # 실제 반복에 사용할 범위 설정
        frame_range = range(determined_start, determined_end + 1)
        print(f"[INFO] No frame range specified. Loading all JSONs from {determined_start:06d} to {determined_end:06d}.")
    else:
        # start_frame과 end_frame이 지정된 경우
        frame_range = range(start_frame, end_frame + 1)
        
    # 2. 프레임 순회 및 파일 경로 조합
    for frame_num in frame_range: 
        
        filename = f"{frame_num:06d}.json" 
        json_path = base_path / filename

        # 파일이 실제로 존재하는지 확인 (전체 로드 모드일 때 빠진 프레임 건너뛰기)
        if not json_path.exists():
            if start_frame is None or end_frame is None:
                # 전체 로드 모드이고 파일이 없으면 건너뛰고 경고만 출력
                # print(f"[WARN] Skipping missing file: {json_path}")
                continue 
            else:
                # 범위가 지정되었는데 파일이 없으면 오류로 처리하거나, 일단 건너뜁니다.
                print(f"[WARN] Missing file in specified range: {json_path}. Skipping.")
                continue

        # 3. JSON 로드 및 데이터 필터링 (기존 로직 유지)
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # 메타 정보는 첫 번째 프레임에서 한 번만 추출하여 저장
            if not meta_loaded:
                meta_info = data.get('meta_info', {})
                result_data['meta_info'] = {
                    'keypoint_id2name': meta_info.get('keypoint_id2name', {}),
                }
                meta_loaded = True
            
            # 인스턴스 ID 할당
            instance_info = data.get('instance_info', [])
            processed_instances = []
            for i, instance in enumerate(instance_info):
                instance['instance_id'] = i + 1 
                processed_instances.append(instance)

            # (프레임 번호, 인스턴스 정보 리스트) 형태로 저장
            result_data['frame_data'].append((frame_num, processed_instances))
            
        except json.JSONDecodeError:
            print(f"[ERROR] JSON decoding failed for {json_path}")
        except Exception as e:
            print(f"[ERROR] An unexpected error occurred while loading {json_path}: {e}")
            
    return result_data



def save_numpy_to_json(kpt_out_path: str, kpts_array: np.ndarray):
    """
    (N, 12, 2) 형태의 NumPy 배열을 프레임별 JSON 파일로 저장합니다.
    tqdm 없이 print로 진행 상황을 알립니다.
    
    Args:
        kpt_out_path (str): JSON 파일들이 저장될 디렉토리 경로.
        kpts_array (np.ndarray): (Frame, 12, 2) 형태의 키포인트 배열.
    """
    
    # 1. 저장 경로 생성
    out_dir = Path(kpt_out_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    num_frames, num_kpts, _ = kpts_array.shape
    
    # 2. 메타 정보 정의 (YOLO12 기준 12개 키포인트)
    meta_info = {
        "dataset_name": "yolo12_custom",
        "num_keypoints": 12,
        "keypoint_id2name": {
            "0": "left_shoulder", "1": "right_shoulder",
            "2": "left_elbow",    "3": "right_elbow",
            "4": "left_wrist",    "5": "right_wrist",
            "6": "left_hip",      "7": "right_hip",
            "8": "left_knee",     "9": "right_knee",
            "10": "left_ankle",   "11": "right_ankle"
        }
    }

    print(f"[INFO] JSON 변환 및 저장 시작... (총 {num_frames} 프레임)")

    # 3. 프레임별 JSON 생성 및 저장
    for frame_idx in range(num_frames):
        
        # 진행 상황 로그 (1000 프레임마다 출력)
        if (frame_idx + 1) % 1000 == 0:
            print(f"  > Processing frame {frame_idx + 1}/{num_frames}...", end='\r')

        # 현재 프레임의 키포인트 (12, 2)
        kpts = kpts_array[frame_idx]
        
        # 유효한 키포인트 확인 (NaN이 아닌 값)
        valid_mask = ~np.isnan(kpts[:, 0])
        valid_kpts = kpts[valid_mask]
        
        instance_info = []
        
        # 유효한 키포인트가 있는 경우에만 인스턴스 정보 생성
        if valid_kpts.shape[0] > 0:
            # BBox 계산 (min_x, min_y, max_x, max_y)
            min_x, min_y = np.min(valid_kpts, axis=0)
            max_x, max_y = np.max(valid_kpts, axis=0)
            bbox = [float(min_x), float(min_y), float(max_x), float(max_y)]
            
            # Keypoints 리스트 변환 (NaN은 0으로 처리)
            # JSON 직렬화를 위해 float 형으로 변환
            kpts_list = np.nan_to_num(kpts, nan=0.0).tolist()
            
            # Scores (임의로 1.0 부여)
            scores = [1.0 if valid_mask[i] else 0.0 for i in range(num_kpts)]
            
            instance_data = {
                "instance_id": 1,
                "keypoints": kpts_list,
                "keypoint_scores": scores,
                "bbox": [bbox],
                "bbox_score": 1.0
            }
            instance_info.append(instance_data)
            
        # JSON 구조 생성
        json_data = {
            "frame_index": frame_idx,
            "meta_info": meta_info,
            "instance_info": instance_info
        }
        
        # 파일 저장 (예: 000000.json)
        file_path = out_dir / f"{frame_idx:06d}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2)
            
    print(f"\n[INFO] 저장 완료: {out_dir}")