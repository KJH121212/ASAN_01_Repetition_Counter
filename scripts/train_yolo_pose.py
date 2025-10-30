# ============================================================
# train_yolo_pose.py  (Ultralytics v8.3.22x, Pose-12KP)
# - train / val / predict 서브커맨드
# - dataset.yml 검증(kpt_shape, 경로), 스테이징 구조 점검
# - W&B 연동(환경변수 기반), 체크포인트 최소 저장 + 정리
# - 특정 ckpt 재시작(--resume_from) / 가중치만 로드(--init_weights)
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
# 기본 경로 (필요시 CLI로 override)
# ------------------------------------------------------------
DEFAULT_DATA = Path(
    "/workspace/nas203/ds_RehabilitationMedicineData/IDs/Kimjihoo/ASAN_01_Repeatition_Counter/data/dataset.yml"
)
DEFAULT_MODEL = Path(
    "/workspace/nas203/ds_RehabilitationMedicineData/IDs/Kimjihoo/ASAN_01_Repeatition_Counter/checkpoints/yolo_pose/yolo11m-pose.pt"
)
DEFAULT_PROJECT = Path(
    "/workspace/nas203/ds_RehabilitationMedicineData/IDs/Kimjihoo/ASAN_01_Repeatition_Counter/checkpoints/yolo_pose"
)
DEFAULT_SOURCE = Path(
    "/workspace/nas203/ds_RehabilitationMedicineData/IDs/Kimjihoo/ASAN_01_Repeatition_Counter/data/_yolo_stage/images/val"
)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _load_yaml(p: Path) -> dict:
    if not p.exists():
        sys.exit(f"[ERR] dataset.yml이 없습니다: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _check_dataset_structure(data_yaml: Path):
    """
    - kpt_shape: [12, 3] 또는 [12, 2]
    - images/{train,val}, labels/{train,val} 디렉토리 존재 확인
    - path 키가 있으면 그 하위에 상대경로가 붙도록 처리
    """
    data = _load_yaml(data_yaml)

    # kpt_shape 확인 (12개 keypoints, 3채널 권장)
    kpt = data.get("kpt_shape")
    if not (isinstance(kpt, (list, tuple)) and len(kpt) == 2 and kpt[0] == 12 and kpt[1] in (2, 3)):
        sys.exit(f"[ERR] kpt_shape가 올바르지 않습니다. 기대값 [12, 3], 현재: {kpt}")

    # path / train / val
    root = Path(data.get("path", Path(data_yaml).parent))
    train_rel = data.get("train")
    val_rel = data.get("val")
    if train_rel is None or val_rel is None:
        sys.exit("[ERR] dataset.yml에 'train' 또는 'val' 항목이 없습니다.")

    # 절대/상대 모두 대응
    def _resolve(p):
        p = Path(p)
        return p if p.is_absolute() else (root / p)

    train_img_dir = _resolve(train_rel)
    val_img_dir = _resolve(val_rel)

    for d, tag in [(train_img_dir, "train images"), (val_img_dir, "val images")]:
        if not d.exists():
            sys.exit(f"[ERR] {tag} 폴더가 없습니다: {d}")

    # labels 디렉토리: <root>/labels/<split>
    train_split = Path(train_rel).name  # "train"
    val_split = Path(val_rel).name      # "val"
    train_lbl_dir = root / "labels" / train_split
    val_lbl_dir = root / "labels" / val_split

    for d, tag in [(train_lbl_dir, "train labels"), (val_lbl_dir, "val labels")]:
        if not d.exists():
            sys.exit(f"[ERR] {tag} 폴더가 없습니다: {d}")

    print("[OK] dataset.yml 및 스테이징 구조 점검 완료 ✅")
    return {
        "root": root,
        "train_imgs": train_img_dir,
        "val_imgs": val_img_dir,
        "train_labels": train_lbl_dir,
        "val_labels": val_lbl_dir,
        "data_dict": data,
    }


def _apply_wandb_env(args):
    """
    W&B 사용 여부 및 설정을 환경변수로 전달.
    - .env에 WANDB_API_KEY가 있으면 자동 로그인 시도.
    - --wandb_off / --wandb_project / --wandb_entity / --wandb_mode 지원.
    """
    if args.wandb_off:
        os.environ["WANDB_MODE"] = "disabled"
        print("[W&B] disabled")
        return

    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project
    if args.wandb_entity:
        os.environ["WANDB_ENTITY"] = args.wandb_entity
    if args.wandb_mode:
        # online / offline / disabled
        os.environ["WANDB_MODE"] = args.wandb_mode

    # (선택) run name 보강
    if "WANDB_RUN_NAME" not in os.environ:
        os.environ["WANDB_RUN_NAME"] = getattr(args, "name", "ultralytics-run")

    # 로그인/버전 출력
    try:
        import wandb
        api_key = os.environ.get("WANDB_API_KEY", "")
        if api_key:
            wandb.login(key=api_key, relogin=True)
        print(
            f"[W&B] ok ver={wandb.__version__} "
            f"project={os.environ.get('WANDB_PROJECT')} "
            f"entity={os.environ.get('WANDB_ENTITY')} "
            f"mode={os.environ.get('WANDB_MODE','online')} "
            f"run_name={os.environ.get('WANDB_RUN_NAME')}"
        )
    except Exception as e:
        print(f"[W&B][WARN] wandb 초기화 실패: {e}")


def _cleanup_checkpoints(run_dir: Path, keep: str = "best"):
    """
    run_dir/weights 안에서 best/last만 남기고 나머지 *.pt 삭제.
    """
    wdir = run_dir / "weights"
    if not wdir.exists():
        return
    keep_files = []
    if keep in ("best", "both"):
        keep_files.append(wdir / "best.pt")
    if keep in ("last", "both"):
        keep_files.append(wdir / "last.pt")

    for p in wdir.glob("*.pt"):
        if p not in keep_files and p.exists():
            try:
                p.unlink()
            except Exception as e:
                print(f"[WARN] 삭제 실패: {p} ({e})")


def _prune_old_runs(project: Path, name_prefix: str, keep_n: int = 1):
    """
    같은 프로젝트 아래 name_prefix로 시작하는 런 디렉토리 중 최신 keep_n개만 남기고 삭제.
    예: name_prefix='yolo11_pose_12kp_ft' → ft, ft2, ft3 중 최신 N개 보관
    """
    if keep_n <= 0 or not project.exists():
        return

    runs = sorted(
        [Path(p) for p in glob(str(project / f"{name_prefix}*")) if Path(p).is_dir()],
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

    # W&B 환경 설정/로그인
    _apply_wandb_env(args)

    # init weights / resume 처리
    model_path = str(args.model)
    resume_arg = False  # 기본: 새 학습

    if args.init_weights:
        # 가중치만 초기화(optimizer/epoch 미포함)
        model_path = str(args.init_weights)
        print(f"[INIT] weights from: {model_path}")

    if args.resume_from:
        # 특정 체크포인트에서 재시작 (optimizer/epoch 포함)
        resume_arg = str(args.resume_from)
        print(f"[RESUME] from checkpoint: {resume_arg}")
    elif args.resume:
        # 같은 project/name의 최근 런에서 자동 재시작
        resume_arg = True
        print("[RESUME] auto from last run in project/name")

    model = YOLO(model_path)

    # AutoBatch 문자열 처리
    if isinstance(args.batch, str) and args.batch.lower() == "auto":
        args.batch = -1  # enable AutoBatch

    # 하이퍼파라미터
    train_kwargs = dict(
        data=str(args.data),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(args.project),
        name=args.name,
        resume=resume_arg,  # 재시작 경로 또는 True
        pretrained=not args.from_scratch if not args.init_weights else False,
        lr0=args.lr0,
        lrf=args.lrf,
        patience=args.patience,
        verbose=True,
        exist_ok=True,       # 같은 폴더 재사용(덮어쓰기)
        save_period=0,       # 에폭별 .pt 저장 비활성화 (용량 절감)
        save=not args.no_save_artifacts,  # 시각화 등 부가 산출물 저장 최소화 옵션
        deterministic=True,  # 재현성 보정(필요시)
    )

    # 실행 설정 저장
    run_cfg_out = Path(args.project) / args.name / "run_config.json"
    run_cfg_out.parent.mkdir(parents=True, exist_ok=True)
    with open(run_cfg_out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "command": "train",
                "base_model": str(args.model),
                "init_weights": str(args.init_weights) if args.init_weights else None,
                "resume_from": str(args.resume_from) if args.resume_from else None,
                **train_kwargs,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("[INFO] Train 시작")
    model.train(**train_kwargs)

    # 체크포인트 정리 (best/last 선택 보관)
    run_dir = Path(args.project) / args.name
    _cleanup_checkpoints(run_dir, keep=args.keep)

    # 오래된 런 폴더 정리
    if args.retention > 0:
        _prune_old_runs(Path(args.project), args.name, keep_n=args.retention)

    print("[DONE] Train 종료")


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
    print("[DONE] Val 종료")
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
    print("[DONE] Predict 종료")
    print(out)


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(description="Ultralytics YOLO Pose 12KP finetune helper")
    sub = p.add_subparsers(dest="cmd", required=True)

    # 공통 옵션
    def add_common(sp):
        sp.add_argument("--data", type=Path, default=DEFAULT_DATA, help="dataset.yml 경로")
        sp.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="초기 가중치(.pt)")
        sp.add_argument("--imgsz", type=int, default=768)
        sp.add_argument("--device", type=str, default="0")
        sp.add_argument("--workers", type=int, default=8)
        sp.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
        sp.add_argument("--name", type=str, default="yolo11_pose_12kp_ft")

        # W&B
        sp.add_argument("--wandb_project", type=str, default=None, help="W&B project override")
        sp.add_argument("--wandb_entity", type=str, default=None, help="W&B entity override")
        sp.add_argument("--wandb_mode", type=str, default=None, choices=["online", "offline", "disabled"])
        sp.add_argument("--wandb_off", action="store_true", help="W&B 끄기")

    sp_tr = sub.add_parser("train", help="Finetune pose model")
    add_common(sp_tr)
    sp_tr.add_argument("--epochs", type=int, default=100)
    sp_tr.add_argument("--batch", type=str, default="auto")
    sp_tr.add_argument("--resume", action="store_true", help="project/name의 마지막 상태에서 자동 재시작")
    sp_tr.add_argument("--resume_from", type=Path, default=None, help="특정 ckpt에서 재시작(last.pt 등, 옵티마이저/스케줄 포함)")
    sp_tr.add_argument("--init_weights", type=Path, default=None, help="가중치만 불러와 새 학습 시작(optimizer/epoch 미포함)")
    sp_tr.add_argument("--from_scratch", action="store_true", help="pretrained 사용 안함(기본 모델만)")
    sp_tr.add_argument("--lr0", type=float, default=0.01, help="initial LR")
    sp_tr.add_argument("--lrf", type=float, default=0.01, help="final LR ratio")
    sp_tr.add_argument("--patience", type=int, default=50)
    sp_tr.add_argument("--no_save_artifacts", action="store_true", help="학습 중 이미지/결과물 저장 최소화")
    # 체크포인트/런 보관 정책
    sp_tr.add_argument("--keep", choices=["best", "last", "both"], default="best", help="학습 종료 후 어떤 체크포인트만 남길지")
    sp_tr.add_argument(
        "--retention",
        type=int,
        default=1,
        help="동일 이름(prefix)의 런 폴더 보관 개수 (최신 N개 보관, 0이면 비활성화)",
    )
    sp_tr.set_defaults(func=cmd_train)

    sp_val = sub.add_parser("val", help="Validate pose model")
    add_common(sp_val)
    sp_val.set_defaults(func=cmd_val)

    sp_pred = sub.add_parser("predict", help="Predict on images/dir")
    add_common(sp_pred)
    sp_pred.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    sp_pred.set_defaults(func=cmd_predict)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
