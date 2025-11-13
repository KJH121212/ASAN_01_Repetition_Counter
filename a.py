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
# "Won_Kim_research_at_Bosanjin" 폴더 관련 비디오 필터링
# -------------------------------------------------------
mask = df["video_path"].str.contains("Won_Kim_research_at_Bosanjin", na=False)  # 조건 필터
filtered = df[mask].reset_index(drop=True)  # 조건 만족 행만 필터링

print(f"🎥 총 {len(filtered)}개의 비디오를 처리합니다.")

# -------------------------------------------------------
# 반복 처리 루프
# -------------------------------------------------------
for idx, row in filtered.iterrows():
    # 개별 비디오 경로 추출
    frame_dir = Path(row["frame_path"])              # 프레임 폴더
    json_dir = Path(row["keypoints_path"])  # JSON 폴더
    video_name = Path(row["video_path"]).stem        # 원본 비디오 이름
    
    # 출력 mp4 파일 경로 설정
    out_mp4 = Path(row["mp4_path"])

    # 출력 폴더 없으면 생성
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n[{idx+1}/{len(filtered)}] 🎬 영상 생성 시작")
    print(f"📂 Frame dir : {frame_dir}")
    print(f"📂 JSON dir  : {json_dir}")
    print(f"💾 Output    : {out_mp4}")

    try:
        # -------------------------------------------------------
        # mp4 생성 (render_skeleton_video 함수 사용)
        # -------------------------------------------------------
        render_skeleton_video(
            frame_dir=str(frame_dir),          # 프레임 경로
            json_dir=str(json_dir),            # JSON 경로
            out_mp4=str(out_mp4),              # 출력 파일 경로
            fps=30,                            # 초당 프레임
            model_type="coco17",               # COCO 17 구조 (12KP면 yolo12)
            flip_horizontal=True              # 좌우 반전 여부
        )
        print(f"✅ 완료: {out_mp4.name}")

    except Exception as e:
        print(f"❌ 오류 발생 ({video_name}) → {e}")