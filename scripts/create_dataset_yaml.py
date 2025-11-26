# ============================================================
# build_yolo_staging_public_fast_tqdm.py
# ✅ metadata.csv + public dataset → YOLO 학습용 스테이징 (진행률 표시 버전)
# ============================================================

import os
import hashlib
from pathlib import Path
import pandas as pd
import yaml
from tqdm import tqdm  # ✅ 진행상황 표시용

BASE_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_Repeatition_Counter")
CSV_PATH = BASE_DIR / "data/metadata_backup.csv"
STAGE_ROOT = BASE_DIR / "data/_yolo_stage"
OUT_YAML = BASE_DIR / "data/dataset.yml"

def short_hash(text: str, n: int = 8) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def symlink_force(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src, dst)

# ------------------------------------------------------------
# ⚡ frame_dir 내 파일명 미리 인덱싱
# ------------------------------------------------------------
def index_images(frame_dir: Path):
    """frame_dir 내 이미지 파일을 {stem: Path} 딕셔너리로 반환"""
    exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    index = {}
    for ext in exts:
        for f in frame_dir.glob(f"*{ext}"):
            index[f.stem] = f
    return index

# ------------------------------------------------------------
# public dataset 정의
# ------------------------------------------------------------
public_data = [
    {
        "frame_path": "/workspace/nas203/ds_RehabilitationMedicineData/data/d01/body_key_point_Public_Data/COCO/train2017",
        "yolo_pose_path": "/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/3_project_HCCmove/data/6_YOLO_POSE/public_data/coco/train",
        "is_train": True, "is_val": False
    },
    {
        "frame_path": "/workspace/nas203/ds_RehabilitationMedicineData/data/d01/body_key_point_Public_Data/COCO/val2017",
        "yolo_pose_path": "/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/3_project_HCCmove/data/6_YOLO_POSE/public_data/coco/val",
        "is_train": False, "is_val": True
    },
    {
        "frame_path": "/workspace/nas203/ds_RehabilitationMedicineData/data/d01/body_key_point_Public_Data/DWPOSE/images",
        "yolo_pose_path": "/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/3_project_HCCmove/data/6_YOLO_POSE/public_data/dwpose/train",
        "is_train": True, "is_val": False
    },
    {
        "frame_path": "/workspace/nas203/ds_RehabilitationMedicineData/data/d01/body_key_point_Public_Data/DWPOSE/images",
        "yolo_pose_path": "/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/3_project_HCCmove/data/6_YOLO_POSE/public_data/dwpose/val",
        "is_train": False, "is_val": True
    },
    {
        "frame_path": "/workspace/nas203/ds_RehabilitationMedicineData/data/d01/body_key_point_Public_Data/MPII/mpii/images",
        "yolo_pose_path": "/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/3_project_HCCmove/data/6_YOLO_POSE/public_data/mpii/train",
        "is_train": True, "is_val": False
    },
    {
        "frame_path": "/workspace/nas203/ds_RehabilitationMedicineData/data/d01/body_key_point_Public_Data/MPII/mpii/images",
        "yolo_pose_path": "/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/3_project_HCCmove/data/6_YOLO_POSE/public_data/mpii/val",
        "is_train": False, "is_val": True
    },
]

# ------------------------------------------------------------
# 메인 빌드 함수
# ------------------------------------------------------------
def build_stage_from_metadata(csv_path: Path, stage_root: Path, out_yaml: Path):
    df_meta = pd.read_csv(csv_path)
    df_public = pd.DataFrame(public_data)
    df_all = pd.concat([df_meta, df_public], ignore_index=True)

    required = ["frame_path", "yolo_pose_path", "is_train", "is_val"]
    for c in required:
        if c not in df_all.columns:
            raise ValueError(f"metadata.csv에 '{c}' 컬럼이 없습니다.")

    # 디렉토리 준비
    img_tr = stage_root / "images" / "train"
    img_va = stage_root / "images" / "val"
    lb_tr = stage_root / "labels" / "train"
    lb_va = stage_root / "labels" / "val"
    for d in [img_tr, img_va, lb_tr, lb_va]:
        ensure_dir(d)

    n_tr, n_va = 0, 0

    # ✅ tqdm으로 전체 진행률 표시
    for _, row in tqdm(df_all.iterrows(), total=len(df_all), desc="Building YOLO Stage", unit="set"):
        frame_dir = Path(row["frame_path"])
        label_dir = Path(row["yolo_pose_path"])
        use_train = bool(row["is_train"])
        use_val = bool(row["is_val"])

        if not (use_train or use_val):
            continue
        if not label_dir.exists():
            print(f"[WARN] 라벨 폴더 없음 → {label_dir}")
            continue

        image_index = index_images(frame_dir)
        prefix = short_hash(str(label_dir.resolve()))

        # ✅ 내부 루프 진행률도 tqdm으로 감싸기
        for lb_file in tqdm(label_dir.glob("*.txt"), desc=f"{label_dir.name}", leave=False, unit="file"):
            stem = lb_file.stem
            if stem not in image_index:
                continue

            im_file = image_index[stem]
            im_name = f"{prefix}__{stem}{im_file.suffix}"
            lb_name = f"{prefix}__{stem}.txt"

            if use_train:
                symlink_force(im_file, img_tr / im_name)
                symlink_force(lb_file, lb_tr / lb_name)
                n_tr += 1
            if use_val:
                symlink_force(im_file, img_va / im_name)
                symlink_force(lb_file, lb_va / lb_name)
                n_va += 1

    # dataset.yml 생성
    data_yaml = {
        "path": str(stage_root),
        "train": "images/train",
        "val": "images/val",
        "kpt_shape": [12, 3],
        "names": {0: "patient"},
    }

    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, sort_keys=False, allow_unicode=True)

    print(f"\n✅ 스테이징 완료: train={n_tr}, val={n_va}")
    print(f"📦 STAGE_ROOT: {stage_root}")
    print(f"📝 dataset.yml: {out_yaml}")

# ------------------------------------------------------------
# 실행
# ------------------------------------------------------------
if __name__ == "__main__":
    build_stage_from_metadata(CSV_PATH, STAGE_ROOT, OUT_YAML)
