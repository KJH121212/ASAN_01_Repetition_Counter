# ============================================================
# train_yolo_pose.py  (Ultralytics v8.3.22x, Pose-12KP)
# - ✅ YAML 설정 (--cfg) + CLI 인자 병합 지원
# - dataset.yml 검증(kpt_shape, 경로)
# - W&B 연동 (환경변수 기반)
# - 체크포인트 최소 저장 및 오래된 런 정리
# ============================================================

import argparse
import json
import os
import sys
import shutil
from glob import glob
from pathlib import Path
import yaml
from ultralytics import YOLO

# ------------------------------------------------------------
# (선택) .env 로드: WANDB_API_KEY 등 환경변수 주입
# ------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv("/workspace/nas203/ds_RehabilitationMedicineData/IDs/Kimjihoo/ASAN_01_Repeatition_Counter/env/.env")
except Exception:
    pass

# ------------------------------------------------------------
# ✅ YAML 설정 병합 로더
# ------------------------------------------------------------
def _load_cfg_yaml(args):
    """--cfg 인자가 주어지면 YAML 파일 내용을 argparse args에 덮어씌움."""
    if getattr(args, "cfg", None):
        cfg_path = Path(args.cfg)
        if not cfg_path.exists():
            sys.exit(f"[ERR] 설정 파일 없음: {cfg_path}")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        for k, v in cfg.items():
            setattr(args, k, v)
        print(f"[CFG] Loaded training config from {cfg_path}")
    return args

# ------------------------------------------------------------
# 기본 경로
# ------------------------------------------------------------
BASE_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/Kimjihoo/ASAN_01_Repeatition_Counter")
DEFAULT_DATA = BASE_DIR / "data/dataset.yml"
DEFAULT_MODEL = BASE_DIR / "checkpoints/yolo_pose/yolo11m-pose.pt"
DEFAULT_PROJECT = BASE_DIR / "checkpoints/yolo_pose"
DEFAULT_SOURCE = BASE_DIR / "data/_yolo_stage/images/val"

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _load_yaml(p: Path) -> dict:
    if not p.exists():
        sys.exit(f"[ERR] dataset.yml이 없습니다: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _check_dataset_structure(data_yaml: Path):
    data_yaml = Path(data_yaml)
    data = _load_yaml(data_yaml)
    
    kpt = data.get("kpt_shape")
    if not (isinstance(kpt, (list, tuple)) and len(kpt) == 2 and kpt[0] == 12):
        sys.exit(f"[ERR] kpt_shape가 [12,3]이어야 합니다. 현재: {kpt}")

    root = Path(data.get("path", Path(data_yaml).parent))
    for split in ["train", "val"]:
        img_dir = root / data.get(split, f"images/{split}")
        lbl_dir = root / f"labels/{split}"
        if not img_dir.exists() or not lbl_dir.exists():
            sys.exit(f"[ERR] {split} 데이터 경로 누락: {img_dir} / {lbl_dir}")
    print("[OK] dataset.yml 및 스테이징 구조 점검 완료 ✅")

def _apply_wandb_env(args):
    if getattr(args, "wandb_off", False):
        os.environ["WANDB_MODE"] = "disabled"
        print("[W&B] disabled")
        return
    try:
        import wandb
        api_key = os.environ.get("WANDB_API_KEY", "")
        if api_key:
            wandb.login(key=api_key, relogin=True)
        print(f"[W&B] ✅ 연결 완료 (ver={wandb.__version__})")
    except Exception as e:
        print(f"[W&B][WARN] wandb 초기화 실패: {e}")

def _cleanup_checkpoints(run_dir: Path, keep="best"):
    wdir = run_dir / "weights"
    if not wdir.exists():
        return
    keep_list = []
    if keep in ("best", "both"):
        keep_list.append(wdir / "best.pt")
    if keep in ("last", "both"):
        keep_list.append(wdir / "last.pt")
    for p in wdir.glob("*.pt"):
        if p not in keep_list and p.exists():
            p.unlink(missing_ok=True)

def _prune_old_runs(project: Path, prefix: str, keep_n: int = 1):
    runs = sorted(
        [p for p in project.glob(f"{prefix}*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for p in runs[keep_n:]:
        try:
            shutil.rmtree(p)
            print(f"[CLEANUP] 오래된 런 삭제: {p}")
        except Exception as e:
            print(f"[WARN] 런 삭제 실패: {p} ({e})")

# ------------------------------------------------------------
# Commands
# ------------------------------------------------------------
def cmd_train(args):
    _check_dataset_structure(args.data)
    _apply_wandb_env(args)

    model_path = str(args.model)
    resume_arg = str(args.resume_from) if args.resume_from else (True if args.resume else False)

    model = YOLO(model_path)
    if isinstance(args.batch, str) and args.batch.lower() == "auto":
        args.batch = -1

    train_kwargs = dict(
        data=str(args.data),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(args.project),
        name=args.name,
        resume=resume_arg,
        pretrained=not args.from_scratch if not args.init_weights else False,
        lr0=args.lr0,
        lrf=args.lrf,
        patience=args.patience,
        verbose=True,
        exist_ok=True,
        save_period=0,
        save=not args.no_save_artifacts,
        deterministic=True,
    )

    # 설정 기록
    run_dir = Path(args.project) / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {k: v for k, v in vars(args).items() if not callable(v)},  # func 같은 callable 제외
            f, indent=2, ensure_ascii=False
        )

    print("[INFO] 🚀 Training 시작...")
    model.train(**train_kwargs)

    _cleanup_checkpoints(run_dir, keep=args.keep)
    if args.retention > 0:
        _prune_old_runs(Path(args.project), args.name, keep_n=args.retention)
    print("[DONE] ✅ Training 종료")

def cmd_val(args):
    _check_dataset_structure(args.data)
    model = YOLO(str(args.model))
    metrics = model.val(
        data=str(args.data),
        imgsz=args.imgsz,
        device=args.device,
        split="val",
        workers=args.workers,
        project=str(args.project),
        name=f"{args.name}_val",
    )
    print("[DONE] ✅ Validation 완료")
    print(metrics)

def cmd_predict(args):
    model = YOLO(str(args.model))
    out = model.predict(
        source=str(args.source),
        imgsz=args.imgsz,
        device=args.device,
        save=True,
        project=str(args.project),
        name=f"{args.name}_pred",
    )
    print("[DONE] ✅ Prediction 완료")
    print(out)

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(description="Ultralytics YOLO Pose 12KP Trainer")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp):
        sp.add_argument("--cfg", type=str, help="설정 YAML 경로", default=None)
        sp.add_argument("--data", type=Path, default=DEFAULT_DATA)
        sp.add_argument("--model", type=Path, default=DEFAULT_MODEL)
        sp.add_argument("--imgsz", type=int, default=768)
        sp.add_argument("--device", type=str, default="0")
        sp.add_argument("--workers", type=int, default=8)
        sp.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
        sp.add_argument("--name", type=str, default="yolo11_pose_12kp_ft")
        sp.add_argument("--wandb_off", action="store_true")

    sp_tr = sub.add_parser("train", help="학습 실행")
    add_common(sp_tr)
    sp_tr.add_argument("--epochs", type=int, default=100)
    sp_tr.add_argument("--batch", type=str, default="auto")
    sp_tr.add_argument("--resume", action="store_true")
    sp_tr.add_argument("--resume_from", type=Path, default=None)
    sp_tr.add_argument("--init_weights", type=Path, default=None)
    sp_tr.add_argument("--from_scratch", action="store_true")
    sp_tr.add_argument("--lr0", type=float, default=0.001)
    sp_tr.add_argument("--lrf", type=float, default=0.01)
    sp_tr.add_argument("--patience", type=int, default=50)
    sp_tr.add_argument("--no_save_artifacts", action="store_true")
    sp_tr.add_argument("--keep", choices=["best", "last", "both"], default="best")
    sp_tr.add_argument("--retention", type=int, default=1)
    sp_tr.set_defaults(func=cmd_train)

    sp_val = sub.add_parser("val", help="검증 실행")
    add_common(sp_val)
    sp_val.set_defaults(func=cmd_val)

    sp_pred = sub.add_parser("predict", help="추론 실행")
    add_common(sp_pred)
    sp_pred.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    sp_pred.set_defaults(func=cmd_predict)

    return p

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    parser = build_parser()
    args = parser.parse_args()
    args = _load_cfg_yaml(args)  # ✅ YAML 적용
    args.func(args)

if __name__ == "__main__":
    main()
