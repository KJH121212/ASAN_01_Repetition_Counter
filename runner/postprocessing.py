import sys
import pandas as pd
import numpy as np
from pathlib import Path
import traceback

# --- 사용자 정의 모듈 경로 설정 ---
PROJECT_ROOT = "/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_Repeatition_Counter"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# --- Custom Modules Import ---
from functions.metadata_parser import extract_info_from_filepath
from functions.func_kpt import load_kpt, save_numpy_to_json
from functions.tracking import extract_patient
from functions.add_noise import add_all_noise
from functions.kalman_filter import filter_and_extract_body_keypoints
from functions.render_skeleton_video_from_array import render_skeleton_video_from_array

# =============================================================================
# 설정 (Configuration)
# =============================================================================
METADATA_PATH = f"{PROJECT_ROOT}/data/metadata.csv"
OUTPUT_DIR = Path(PROJECT_ROOT)

# 노이즈 추가 설정
NOISE_CONFIG = {
    "spike_ratio": 0.00,
    "gaussian_frame_ratio": 0.00,
    "gaussian_kp_ratio": 0.5,
    "gaussian_std": 7.0
}

# =============================================================================
# 메인 실행 함수 (print 로그 버전)
# =============================================================================
def main():
    # 1. 메타데이터 로드
    if not Path(METADATA_PATH).exists():
        print(f"[FATAL] 메타데이터 파일을 찾을 수 없습니다: {METADATA_PATH}")
        return

    df = pd.read_csv(METADATA_PATH)
    total_videos = len(df)
    
    # DataFrame의 모든 행을 순회 (tqdm 제거)
    for index, row in df.iterrows():
        
        # 진행 상황 출력
        print(f"[{index + 1}/{total_videos}] Processing Index {index} ...", end=" ", flush=True)

        try:
            # --- 경로 및 정보 추출 ---
            frame_path = row['frame_path']
            keypoints_path = row['keypoints_path']
            
            # n_json 값 확인 및 예외 처리
            if pd.isna(row['n_json']) or row['n_json'] <= 0:
                print(f"-> [SKIP] 유효하지 않은 n_json 값 ({row.get('n_json')})")
                continue

            start_frame = 0
            end_frame = int(row['n_json']) - 1
            
            # 저장 경로 설정
            json_output_path = row.get("interp_json_path", "")
            
            # 파일명 정보 파싱
            label_string, view_angle, action_name = extract_info_from_filepath(frame_path)
            
            # 영상 저장 경로
            video_output_path = row['mp4_path']

            # --- 2. 키포인트 데이터 로드 ---
            raw_data = load_kpt(keypoints_path, start_frame, end_frame)
            if not raw_data['frame_data']:
                print(f"-> [WARN] 로드된 데이터가 없습니다.")
                continue

            # --- 3. 환자 데이터 추출 (Tracking) ---
            patient_kpts = extract_patient(raw_data)
            
            if np.all(np.isnan(patient_kpts)):
                print(f"-> [WARN] 환자 데이터를 추출하지 못했습니다 (모두 NaN).")
                continue

            # --- 4. 노이즈 추가 (설정에 따라 실행) ---
            if NOISE_CONFIG["spike_ratio"] > 0 or NOISE_CONFIG["gaussian_frame_ratio"] > 0:
                processed_input = add_all_noise(patient_kpts, **NOISE_CONFIG)
            else:
                processed_input = patient_kpts

            # --- 5. 칼만 필터 적용 ---
            final_data = filter_and_extract_body_keypoints(processed_input,outlier_threshold=15)

            # # --- 6. 결과 영상 생성 ---
            # # (내부 함수인 render_skeleton... 에도 tqdm이 있다면 거기서 진행바가 나올 수 있습니다)
            render_skeleton_video_from_array(
                final_data=final_data,
                FRAME_PATH=frame_path,
                OUTPUT_PATH=str(video_output_path),
                start_frame=start_frame,
                end_frame=end_frame,
                flip_horizontal=True
            )
            print(f"{video_output_path}에 비디오를 저장했습니다.")

            # --- 7. 결과 JSON 저장 ---
            if json_output_path:
                Path(json_output_path).parent.mkdir(parents=True, exist_ok=True)
                save_numpy_to_json(json_output_path, final_data)
            
            print("-> [DONE]")
            
        except Exception as e:
            print(f"\n[ERROR] Index {index} 처리 중 오류 발생: {e}")
            traceback.print_exc() # 상세 에러 로그 출력
            continue

if __name__ == "__main__":
    main()