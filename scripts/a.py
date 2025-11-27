import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm 
import numpy as np 
from typing import List, Dict, Any

# 💡 tqdm pandas 확장 활성화 (상단에 한 번만)
tqdm.pandas()

# --- 파일 경로 정의 ---
NEW_PATH = "/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_Repeatition_Counter/data/metadata.csv"
OLD_PATH = "/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_Repeatition_Counter/data/metadata_final.csv"

# --- 헬퍼 함수 정의 ---

def count_files_in_directory(dir_path: str, extension: str = None) -> int:
    """
    주어진 경로 내의 파일 개수를 셉니다. 확장자를 지정할 수 있습니다.
    """
    path_obj = Path(dir_path)
    if not path_obj.is_dir():
        return 0
    
    count = 0
    try:
        if extension:
            count = sum(1 for item in path_obj.iterdir() if item.is_file() and item.suffix.lower() == extension.lower())
        else:
            count = sum(1 for item in path_obj.iterdir() if item.is_file())
    except FileNotFoundError:
        return 0
    except PermissionError:
        print(f"[WARN] Permission denied for path: {dir_path}")
        return 0
        
    return count

def update_file_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    'frame_path'의 파일 개수를 'n_frames'에, 'keypoints_path'의 파일 개수를 'n_json'에 업데이트하고
    진행 상태를 tqdm으로 표시합니다.
    """
    
    print("--- n_frames / n_json 업데이트 시작 (tqdm 적용) ---")
    
    # 1. 'frame_path' 업데이트 -> n_frames
    # 💡 desc 인자 제거! progress_apply에 문제가 발생하지 않도록 합니다.
    df['n_frames'] = df['frame_path'].progress_apply(
        lambda p: count_files_in_directory(p, extension=".jpg")
    )
    
    # 2. 'keypoints_path' 업데이트 -> n_json
    # 💡 desc 인자 제거!
    df['n_json'] = df['keypoints_path'].progress_apply(
        lambda p: count_files_in_directory(p, extension=".json")
    )

    print("--- n_frames / n_json 업데이트 완료 ---")
    return df

# ===============================================================
# --- 메인 실행 로직 ---
# ===============================================================

# --- 데이터 로드 ---
try:
    df_new = pd.read_csv(NEW_PATH)
    df_old = pd.read_csv(OLD_PATH)
except FileNotFoundError as e:
    print(f"[FATAL] 파일을 찾을 수 없습니다: {e}")
    exit()

# (중간 업데이트 로직 생략 - 변경 없음)

# 1. df_old로부터 is_train/is_val 값 복사 및 done 플래그 True로 변경
df_old_subset = df_old[['video_path', 'is_train', 'is_val']].copy()
df_old_subset.columns = ['video_path', 'old_is_train', 'old_is_val']

df_merged = df_new.merge(df_old_subset, on='video_path', how='left')

# 3. 'is_train' 및 'is_val' 열 업데이트
df_new['is_train'] = df_merged['old_is_train'].fillna(df_new['is_train'])
df_new['is_val'] = df_merged['old_is_val'].fillna(df_new['is_val'])

# 4. 모든 '_done' 플래그를 True로 일괄 변경
done_columns = ['frames_done', 'sapiens_done', 'reextract_done', 'overlay_done']
for col in done_columns:
    if col in df_new.columns:
        df_new[col] = True

# --- 2. 파일 개수 카운트 및 n_frames/n_json 업데이트 (수정된 함수 호출) ---
df_new = update_file_counts(df_new)

# 5. 최종 확인
print("\n--- 최종 업데이트된 df_new 상태 (Done 플래그와 카운트 확인) ---")
print(df_new[['video_path', 'is_train', 'is_val', 'frames_done', 'sapiens_done', 'n_frames', 'n_json']].head())
print("-" * 30)

# --- 3. 업데이트된 DataFrame을 원본 CSV 파일에 덮어쓰기 ---
try:
    df_new.to_csv(NEW_PATH, index=False)
    print(f"\n✅ DataFrame 업데이트 완료 및 {NEW_PATH}에 저장되었습니다. (Index 제외)")
except Exception as e:
    print(f"\n[FATAL] CSV 파일 저장 중 오류 발생: {e}")