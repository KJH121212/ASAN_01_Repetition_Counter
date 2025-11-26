#!/usr/bin/env python3
# ================================================================
# script/make_metadata_template.py
# metadata 기본 틀만 생성 (파일 존재 여부/카운트 계산 X)
# ================================================================
import csv
from pathlib import Path
from tqdm import tqdm

# ------------------------------------------------------------
# 1️⃣ 설정
# ------------------------------------------------------------
VIDEO_ROOT = Path("/workspace/nas203/ds_RehabilitationMedicineData/data/d02")
BASE_OUT   = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/3_project_HCCmove/data")
OUT_CSV    = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_Repeatition_Counter/data/metadata_template.csv")

VIDEO_EXTS = [".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV"]

# ------------------------------------------------------------
# 2️⃣ 유틸 함수
# ------------------------------------------------------------
def find_videos(root: Path):
    """비디오 파일 탐색"""
    videos = []
    for ext in VIDEO_EXTS:
        videos.extend(root.rglob(f"*{ext}"))
    return sorted(videos)

def build_paths(video_path: Path):
    """비디오 기준으로 경로 틀 생성"""
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
def make_metadata_template():
    """metadata 기본 틀만 생성 (상태 계산 없음)"""
    videos = find_videos(VIDEO_ROOT)
    print(f"[INFO] Found {len(videos)} total videos under {VIDEO_ROOT}")

    rows = []
    for v in tqdm(videos, desc="🧩 Building template", unit="video"):
        paths = build_paths(v)
        rows.append({
            "video_path": str(v),
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
        })

    # CSV 저장
    fieldnames = [
        "video_path", "frame_path", "keypoints_path", "mp4_path",
        "interp_json_path", "yolo_bbox_path", "yolo_pose_path",
        "n_frames", "n_json",
        "frames_done", "sapiens_done", "reextract_done",
        "overlay_done", "is_train", "is_val"
    ]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[✅] Template CSV created → {OUT_CSV}")
    print(f"[INFO] Total {len(rows)} entries written.")

# ------------------------------------------------------------
# 4️⃣ 실행
# ------------------------------------------------------------
if __name__ == "__main__":
    make_metadata_template()
