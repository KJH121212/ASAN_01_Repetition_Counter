import sys
sys.path.append("/workspace/nas203/ds_RehabilitationMedicineData/IDs/Kimjihoo/ASAN_01_Repeatition_Counter")
from functions.constants_skeleton.registry import load_skeleton_constants  # 내부 상수 로드용
from functions.render_skeleton_video import render_skeleton_video                    # overlay mp4 생성 함수
from pathlib import Path
import pandas as pd                        # CSV 파일 처리를 위해 pandas 임포트
from pathlib import Path                   # 경로 처리를 위해 Path 임포트
import json                                # JSON 파일 처리를 위해 json 임포트

# -------------------------------------------------------
# 경로 설정
# -------------------------------------------------------
BASE_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData")  # 루트 경로 설정
CSV_PATH = BASE_DIR / "IDs/Kimjihoo/ASAN_01_Repeatition_Counter/data/metadata_backup.csv"  # metadata.csv 경로

# -------------------------------------------------------
# CSV 로드
# -------------------------------------------------------
df = pd.read_csv(CSV_PATH)                 # metadata.csv 불러오기

# -------------------------------------------------------
# "Nintendo_Therapy" 폴더 & "N06_VISIT6_2" 관련 행 필터링
# -------------------------------------------------------
mask = df["video_path"].str.contains("Nintendo_Therapy", na=False) & df["video_path"].str.contains("N06_VISIT6_2", na=False)  # 조건 필터
filtered = df[mask]                       # 조건 만족 행만 필터링

# -------------------------------------------------------
# 경로 설정
# -------------------------------------------------------
row = filtered.iloc[0]                                 # 현재 비디오 행 선택
frame_dir = Path(row["frame_path"])                    # 프레임 폴더
interp_json_dir = Path(row["interp_json_path"])        # 우리가 방금 생성한 JSON 폴더
out_mp4 = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/Kimjihoo/ASAN_01_Repeatition_Counter/interp_overlay.mp4")       # 출력 mp4 경로

print(f"🎬 영상 생성 시작...")
print(f"📂 Frame dir : {frame_dir}")
print(f"📂 JSON dir  : {interp_json_dir}")
print(f"💾 Output    : {out_mp4}")

# -------------------------------------------------------
# mp4 생성 (render_skeleton_video 함수 사용)
# -------------------------------------------------------
render_skeleton_video(
    frame_dir=str(frame_dir),          # 프레임 경로
    json_dir=str(interp_json_dir),     # JSON 경로
    out_mp4=str(out_mp4),              # 출력 파일
    fps=30,                            # 초당 프레임
    kp_radius=4,                       # 키포인트 점 크기
    line_thickness=2,                  # skeleton 선 두께
    model_type="coco17",               # COCO 17 구조 (12KP면 yolo12)
    flip_horizontal=False              # 좌우 반전 여부
)
