#!/usr/bin/env python3
# ================================================================
# script/update_metadata_template.py
# metadata.csv ← 기존 유지 + 신규 비디오 path만 추가 (상태값은 전부 False/0)
# ================================================================
import csv
from pathlib import Path
from tqdm import tqdm

# ------------------------------------------------------------
# 1️⃣ 설정
# ------------------------------------------------------------
VIDEO_ROOT = Path("/workspace/nas203/ds_RehabilitationMedicineData/data/d02")
BASE_OUT   = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/Kimjihoo/3_project_HCCmove/data")
CSV_PATH   = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/Kimjihoo/ASAN_01_Repeatition_Counter/data/metadata.csv")
BACKUP_CSV = CSV_PATH.with_name("metadata_backup.csv")
VIDEO_EXTS = [".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV"]

# ------------------------------------------------------------
# 2️⃣ 유틸 함수
# ------------------------------------------------------------
def find_videos(root: Path):
    """모든 비디오 파일 탐색"""
    videos = []
    for ext in VIDEO_EXTS:
        videos.extend(root.rglob(f"*{ext}"))
    return sorted(videos)

def build_paths(video_path: Path):
    """비디오 경로 기반으로 각 데이터 경로 구성"""
    rel_dir = video_path.parent.relative_to(VIDEO_ROOT)
    video_stem = video_path.stem

    def make(subdir: str):
        return BASE_OUT / subdir / rel_dir / video_stem

    return dict(
        frame_path=make("1_FRAME"),
        keypoints_path=make("2_KEYPOINTS"),
        mp4_path=BASE_OUT / "3_MP4" / rel_dir / f"{video_stem}.mp4",
        interp_json_path=make("4_INTERP_DATA"),
        yolo_bbox_path=make("5_YOLO_BBOX"),
        yolo_pose_path=make("6_YOLO_POSE"),
    )

# ------------------------------------------------------------
# 3️⃣ 메인 함수
# ------------------------------------------------------------
def update_metadata_template():
    """metadata.csv 기반으로 신규 비디오 path만 추가"""
    existing_rows = {}

    # ✅ 기존 CSV 백업 + 읽기
    if CSV_PATH.exists():
        import shutil
        shutil.copy2(CSV_PATH, BACKUP_CSV)
        print(f"[🗂] Backup created → {BACKUP_CSV}")

        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_rows[row["video_path"]] = row
        print(f"[INFO] Loaded existing metadata ({len(existing_rows)} entries)")
    else:
        print("[INFO] No existing metadata found. Creating new one.")

    # ✅ 비디오 탐색
    videos = find_videos(VIDEO_ROOT)
    print(f"[INFO] Found {len(videos)} total videos under {VIDEO_ROOT}")

    # ✅ 신규 추가
    new_rows = {}
    for v in tqdm(videos, desc="🧩 Building template", unit="video"):
        v_str = str(v)
        if v_str in existing_rows:
            continue  # 이미 있는 건 skip

        paths = build_paths(v)
        row = {
            "video_path": v_str,
            "frame_path": str(paths["frame_path"]),
            "keypoints_path": str(paths["keypoints_path"]),
            "mp4_path": str(paths["mp4_path"]),
            "interp_json_path": str(paths["interp_json_path"]),
            "yolo_bbox_path": str(paths["yolo_bbox_path"]),
            "yolo_pose_path": str(paths["yolo_pose_path"]),
            "n_frames": 0,
            "n_json": 0,
            "frames_done": False,
            "sapiens_done": False,
            "reextract_done": False,
            "overlay_done": False,
            "is_train": False,
            "is_val": False,
        }
        new_rows[v_str] = row
        existing_rows[v_str] = row

    # --------------------------------------------------------
    # CSV 저장 (기존 + 신규 병합)
    # --------------------------------------------------------
    fieldnames = [
        "video_path", "frame_path", "keypoints_path", "mp4_path",
        "interp_json_path", "yolo_bbox_path", "yolo_pose_path",
        "n_frames", "n_json",
        "frames_done", "sapiens_done", "reextract_done",
        "overlay_done", "is_train", "is_val"
    ]

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_rows.values())

    print(f"\n[✅] metadata.csv updated → {CSV_PATH}")
    print(f"[INFO] Added {len(new_rows)} new videos, total {len(existing_rows)} entries")

# ------------------------------------------------------------
# 4️⃣ 실행
# ------------------------------------------------------------
if __name__ == "__main__":
    update_metadata_template()
