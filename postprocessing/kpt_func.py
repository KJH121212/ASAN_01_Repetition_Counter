import json
import numpy as np
import re
import os
from pathlib import Path

def load_kpt_to_numpy(json_dir):
    """
    JSON 디렉토리에서 키포인트 데이터를 읽어 NumPy 배열로 반환합니다.
    
    Returns:
        kpt_data (np.ndarray): (Frame, Max_Instances, Keypoints, 3) 형태의 배열
                               마지막 차원은 [x, y, confidence] 입니다.
                               데이터가 없는 곳은 NaN으로 채워집니다.
        raw_meta (list): 각 인덱스에 해당하는 원본 메타 데이터 (frame_idx, instance_id 등)
    """
    json_files = sorted(Path(json_dir).glob("*.json"))
    
    if not json_files:
        print("JSON 파일이 없습니다.")
        return None, None

    # 데이터의 전체 크기를 짐작하기 위해 첫 파일을 읽어봅니다.
    # (여기서는 COCO 17개 키포인트라고 가정합니다. YOLO12면 12개로 조정)
    # 실제로는 전체 파일을 훑어서 최대 인원수(Max Instances)를 찾아야 정확합니다.
    # 편의상 최대 인원수를 넉넉하게 10명으로 가정하거나, 1차 스캔을 할 수 있습니다.
    MAX_INSTANCES = 10 
    NUM_KEYPOINTS = 17 # COCO 포맷 기준
    
    # 프레임 번호를 기준으로 정렬 (파일명에 숫자가 있다고 가정)
    # 예: 000000.json -> 0
    json_files.sort(key=lambda f: int(re.findall(r'\d+', f.name)[-1]))
    
    num_frames = len(json_files)
    
    # (프레임 수, 최대 인원 수, 키포인트 수, 3[x,y,score])
    # 초기값은 NaN으로 채워서 데이터가 없는 곳을 구분합니다.
    kpt_data = np.full((num_frames, MAX_INSTANCES, NUM_KEYPOINTS, 3), np.nan, dtype=np.float32)
    
    raw_meta = []

    for i, json_path in enumerate(json_files):
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        frame_idx = data.get('frame_index', i)
        instances = data.get('instance_info', [])
        
        frame_meta = {'frame_index': frame_idx, 'instances': []}

        for j, inst in enumerate(instances):
            if j >= MAX_INSTANCES:
                break # 최대 인원수 초과 시 무시 (필요 시 MAX_INSTANCES 늘리기)
            
            # 키포인트 (N, 2)와 스코어 (N,) 가져오기
            kpts = np.array(inst['keypoints']) # [[x, y], ...]
            scores = np.array(inst['keypoint_scores']) # [s, ...]
            
            # (x, y)와 score를 합쳐서 (N, 3)으로 만듦: [x, y, s]
            # shape: (17, 2) -> (17, 3)
            kpts_with_score = np.column_stack((kpts, scores))
            
            # NumPy 배열에 저장
            kpt_data[i, j, :, :] = kpts_with_score
            
            # 추후 매칭을 위해 원본 instance_id 저장
            frame_meta['instances'].append({
                'raw_idx': j, 
                'instance_id': inst.get('instance_id', j+1)
            })
            
        raw_meta.append(frame_meta)

    return kpt_data, raw_meta


def save_kpts_numpy_to_json(
    kpt_out_path: str, 
    kpts_array: np.ndarray, 
    meta_info: dict = None
):
    """
    (Frame, Max_Instances, Keypoints, 3) 형태의 NumPy 배열을 
    기존 JSON 포맷(instance_info, meta_info 포함)으로 저장합니다.
    
    Args:
        kpt_out_path (str): 저장할 디렉토리 경로
        kpts_array (np.ndarray): (F, N, K, 3) 형태의 배열 [x, y, score]
                                 N은 트랙(인스턴스) 인덱스로 사용됩니다.
        meta_info (dict): 원본 메타 데이터 (없으면 자동 생성)
    """
    
    # 1. 저장 경로 생성
    out_dir = Path(kpt_out_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 배열 차원 확인
    if kpts_array.ndim != 4:
        print(f"[ERROR] 입력 배열은 4차원이어야 합니다. 현재 형태: {kpts_array.shape}")
        return

    num_frames, max_instances, num_kpts, dims = kpts_array.shape
    
    # 2. 메타 정보 자동 설정 (입력이 없을 경우)
    if meta_info is None:
        if num_kpts == 17: # COCO 포맷
            meta_info = {
                "dataset_name": "coco",
                "num_keypoints": 17,
                "keypoint_id2name": {
                    "0": "nose", "1": "left_eye", "2": "right_eye", "3": "left_ear", "4": "right_ear",
                    "5": "left_shoulder", "6": "right_shoulder", "7": "left_elbow", "8": "right_elbow",
                    "9": "left_wrist", "10": "right_wrist", "11": "left_hip", "12": "right_hip",
                    "13": "left_knee", "14": "right_knee", "15": "left_ankle", "16": "right_ankle"
                }
            }
        elif num_kpts == 12: # YOLO12 포맷
            meta_info = {
                "dataset_name": "yolo12_custom",
                "num_keypoints": 12,
                "keypoint_id2name": {
                    "0": "left_shoulder", "1": "right_shoulder", "2": "left_elbow", "3": "right_elbow",
                    "4": "left_wrist", "5": "right_wrist", "6": "left_hip", "7": "right_hip",
                    "8": "left_knee", "9": "right_knee", "10": "left_ankle", "11": "right_ankle"
                }
            }

    print(f"[INFO] JSON 저장 시작... (총 {num_frames} 프레임)")

    # 3. 프레임별 처리
    for frame_idx in range(num_frames):
        
        if (frame_idx + 1) % 100 == 0:
            print(f"  > Saving frame {frame_idx + 1}/{num_frames}...", end='\r')

        instance_info_list = []
        
        # 각 프레임 내의 모든 인스턴스(트랙) 순회
        for inst_idx in range(max_instances):
            # 해당 인스턴스의 데이터 가져오기: (K, 3)
            inst_data = kpts_array[frame_idx, inst_idx]
            
            # 좌표(x,y)와 점수(score) 분리
            kpts_xy = inst_data[:, :2]
            scores = inst_data[:, 2]
            
            # 유효한 데이터인지 확인 (x 좌표가 NaN이 아닌 개수)
            valid_mask = ~np.isnan(kpts_xy[:, 0])
            
            # 유효한 키포인트가 하나도 없으면 해당 인스턴스는 이 프레임에 존재하지 않음 -> 건너뜀
            if np.sum(valid_mask) == 0:
                continue
                
            # --- BBox 자동 계산 ---
            # 유효한 키포인트들의 최소/최대값으로 박스 생성
            valid_xy = kpts_xy[valid_mask]
            min_x, min_y = np.min(valid_xy, axis=0)
            max_x, max_y = np.max(valid_xy, axis=0)
            bbox = [float(min_x), float(min_y), float(max_x), float(max_y)]
            
            # --- 데이터 변환 (NumPy -> List) ---
            # NaN 값은 JSON 표준이 아니므로 0.0 또는 null로 변환해야 함 (여기선 0.0 사용)
            kpts_list = np.nan_to_num(kpts_xy, nan=0.0).tolist()
            scores_list = np.nan_to_num(scores, nan=0.0).tolist()
            
            # 인스턴스 정보 생성
            instance_data = {
                "instance_id": inst_idx + 1,  # 1부터 시작하는 ID 부여 (Track ID와 연동)
                "track_id": inst_idx + 1,     # 트래킹 ID 명시 (선택 사항)
                "keypoints": kpts_list,       # [[x, y], ...]
                "keypoint_scores": scores_list,
                "bbox": [bbox],               # 2D 리스트 형태 [[x1, y1, x2, y2]]
                "bbox_score": 1.0             # 임의의 값 (필요 시 계산된 값 사용)
            }
            instance_info_list.append(instance_data)
            
        # 4. JSON 구조 생성 및 저장
        json_data = {
            "frame_index": frame_idx,
            "meta_info": meta_info,
            "instance_info": instance_info_list
        }
        
        file_name = f"{frame_idx:06d}.json"
        file_path = out_dir / file_name
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2)

    print(f"\n[INFO] 저장 완료: {out_dir}")


# kpt_path = "/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/3_project_HCCmove/data/2_KEYPOINTS/AI_dataset/N01/N01_Treatment/diagonal__hip_extension"
# kpt_array, meta = load_kpt_to_numpy(kpt_path)
# print(kpt_array.shape)  # (11370, 10, 17, 3) 예상
# out_path = "/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_Repeatition_Counter/postproocessing/test"
# save_kpts_numpy_to_json(out_path,kpts_array=kpt_array)