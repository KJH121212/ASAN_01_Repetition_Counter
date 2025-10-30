# ============================================================
# build_yolo_staging.py
# ✅ metadata.csv 기반 → YOLO 학습용 스테이징(심볼릭 링크) + dataset.yml 생성
#    - 원본 폴더 구조/파일은 변경하지 않음
#    - 파일명 충돌 방지: <dirhash>__<basename> 형식으로 링크 생성
#    - Ultralytics v8 규칙 준수: path/images/* 와 path/labels/* 짝맞춤
#    - 라벨 기준으로 이미지를 찾아 매칭하므로 "이미지 많음/경로 흩어짐" 문제를 회피
# ============================================================

import os                   # 심볼릭 링크 생성 및 OS 기능을 사용하기 위한 모듈
import hashlib              # 경로 문자열을 짧은 해시로 변환하기 위한 모듈
from pathlib import Path    # 경로 처리를 편리하게 하기 위한 모듈
import pandas as pd         # metadata.csv 로드를 위한 모듈
import yaml                 # dataset.yml 생성을 위한 모듈

# ------------------------------------------------------------
# 경로 설정
# ------------------------------------------------------------
BASE_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/Kimjihoo/ASAN_01_Repeatition_Counter")  # 프로젝트 루트 경로
CSV_PATH = BASE_DIR / "data/metadata_backup.csv"       # 메타데이터 CSV 경로
STAGE_ROOT = BASE_DIR / "data/_yolo_stage"      # YOLO가 읽을 스테이징 루트(심볼릭 링크 모음) 경로
OUT_YAML = BASE_DIR / "data/dataset.yml"        # YOLO dataset.yml 출력 경로

# ------------------------------------------------------------
# 유틸 함수
# ------------------------------------------------------------
def short_hash(text: str, n: int = 8) -> str:
    """주어진 문자열에 대해 n글자 길이의 짧은 해시를 반환한다."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]  # SHA1 해시를 계산해 앞 n글자를 반환한다.

def ensure_dir(p: Path) -> None:
    """디렉토리가 없으면 생성한다."""
    p.mkdir(parents=True, exist_ok=True)  # 상위 폴더까지 포함해 안전하게 디렉토리를 만든다.

def try_find_image(frame_dir: Path, stem: str) -> Path:
    """라벨 파일명과 동일한 stem을 가진 이미지를 확장자 우선순위로 찾아 반환한다."""
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:  # 일반적인 이미지 확장자 후보를 나열한다.
        cand = frame_dir / f"{stem}{ext}"                   # 후보 이미지 경로를 만든다.
        if cand.exists():                                   # 후보가 실제로 존재하는지 확인한다.
            return cand                                     # 존재하면 해당 경로를 반환한다.
    raise FileNotFoundError(f"이미지 파일을 찾을 수 없음: {frame_dir}/{stem}.*")  # 없으면 예외를 발생시킨다.

def symlink_force(src: Path, dst: Path) -> None:
    """dst가 있으면 지우고 src로 향하는 심볼릭 링크를 생성한다."""
    if dst.exists() or dst.is_symlink():                    # 대상 경로에 파일 또는 링크가 이미 있는지 확인한다.
        dst.unlink()                                        # 기존 것을 삭제해 충돌을 방지한다.
    os.symlink(src, dst)                                    # 소스 파일을 가리키는 심볼릭 링크를 생성한다.

# ------------------------------------------------------------
# 메인 빌드 함수
# ------------------------------------------------------------
def build_stage_from_metadata(csv_path: Path, stage_root: Path, out_yaml: Path) -> None:
    """metadata.csv를 읽어 스테이징(links)과 dataset.yml을 생성한다."""
    df = pd.read_csv(csv_path)                                          # CSV 파일을 로드한다.
    required = ["frame_path", "yolo_pose_path", "is_train", "is_val"]   # 필요한 컬럼 목록을 정의한다.
    for c in required:                                                  # 각 필수 컬럼을 순회한다.
        if c not in df.columns:                                         # 컬럼이 없으면 예외를 던진다.
            raise ValueError(f"metadata.csv에 '{c}' 컬럼이 없습니다.")   # 명확한 에러 메시지를 제공한다.

    # 스테이징 디렉토리 준비
    img_tr = stage_root / "images" / "train"    # 학습 이미지 링크 폴더 경로를 정의한다.
    img_va = stage_root / "images" / "val"      # 검증 이미지 링크 폴더 경로를 정의한다.
    lb_tr = stage_root / "labels" / "train"     # 학습 라벨 링크 폴더 경로를 정의한다.
    lb_va = stage_root / "labels" / "val"       # 검증 라벨 링크 폴더 경로를 정의한다.
    for d in [img_tr, img_va, lb_tr, lb_va]:    # 네 경로를 모두 순회한다.
        ensure_dir(d)                           # 없으면 생성한다.

    # 생성 통계 카운터
    n_tr, n_va = 0, 0  # 학습/검증 링크 생성 개수를 집계한다.

    # 각 행(frame_dir ↔ label_dir 쌍)을 처리
    for _, row in df.iterrows():                    # 메타데이터의 모든 행을 순회한다.
        frame_dir = Path(row["frame_path"])         # 해당 샘플의 이미지 폴더를 가져온다.
        label_dir = Path(row["yolo_pose_path"])     # 해당 샘플의 YOLO 포즈 라벨 폴더를 가져온다.
        use_train = bool(row["is_train"])           # 학습에 사용할지 여부를 얻는다.
        use_val = bool(row["is_val"])               # 검증에 사용할지 여부를 얻는다.

        if not (use_train or use_val):                  # 둘 다 아니면 이 행을 건너뛴다.
            continue
        if not label_dir.exists():                      # 라벨 폴더가 없으면 경고하고 건너뛴다.
            print(f"[WARN] 라벨 폴더 없음 → {label_dir}")    # 사용자에게 경고를 출력한다.
            continue

        prefix = short_hash(str(label_dir.resolve()))  # 라벨 폴더 경로로부터 8자리 해시 prefix를 만든다.

        for lb_file in sorted(label_dir.glob("*.txt")):  # 라벨 폴더 내 모든 .txt를 순회한다.
            stem = lb_file.stem  # 라벨 파일의 베이스 이름을 얻는다.
            try:
                im_file = try_find_image(frame_dir, stem)   # 같은 stem의 이미지 파일을 찾는다.
            except FileNotFoundError as e:                  # 매칭 이미지가 없으면 경고하고 건너뛴다.
                print(f"[WARN] 매칭 이미지 없음 → {e}")         # 어떤 파일이 누락됐는지 로그를 남긴다.
                continue

            im_name = f"{prefix}__{stem}{im_file.suffix}"   # 충돌방지용 이미지 링크 파일명을 만든다.
            lb_name = f"{prefix}__{stem}.txt"               # 충돌방지용 라벨 링크 파일명을 만든다.

            if use_train:  # 학습 분할로 사용하면
                symlink_force(im_file, img_tr / im_name)    # 학습 이미지 링크를 생성/갱신한다.
                symlink_force(lb_file, lb_tr / lb_name)     # 학습 라벨 링크를 생성/갱신한다.
                n_tr += 1                                   # 생성 개수를 증가시킨다.
            if use_val:  # 검증 분할로 사용하면
                symlink_force(im_file, img_va / im_name)    # 검증 이미지 링크를 생성/갱신한다.
                symlink_force(lb_file, lb_va / lb_name)     # 검증 라벨 링크를 생성/갱신한다.
                n_va += 1                                   # 생성 개수를 증가시킨다.

    # Ultralytics v8 규칙을 따르는 dataset.yml을 생성한다.
    data_yaml = {  # YAML 구조를 딕셔너리로 구성한다.
        "path": str(stage_root),        # 스테이징 루트를 path로 지정한다.
        "train": "images/train",        # 학습 이미지는 path/images/train으로 둔다.
        "val": "images/val",            # 검증 이미지는 path/images/val으로 둔다.
        "kpt_shape": [12, 3],           # 12개 키포인트와 (x,y,v) 포맷을 명시한다.
        "names": {0: "patient"},        # 단일 클래스명을 정의한다.
        # "skeleton": [[10,8],[8,6],[11,9],[9,7],[6,7],[0,6],[1,7],[0,1],[0,2],[2,4],[1,3],[3,5]],  # 필요시 스켈레톤을 추가한다.
        # "flip_idx": [...],  # 좌우 반전 매핑이 필요하면 yolo12 순서에 맞게 채운다.
    }  

    with open(out_yaml, "w", encoding="utf-8") as f:  # dataset.yml 파일을 연다.
        yaml.dump(data_yaml, f, sort_keys=False, allow_unicode=True)  # 정렬을 유지하고 유니코드를 허용해 저장한다.

    print(f"✅ 스테이징 완료: train={n_tr} val={n_va}")  # 생성된 링크 개수를 요약 출력한다.
    print(f"📦 STAGE_ROOT: {stage_root}")  # 스테이징 루트 경로를 알려준다.
    print(f"📝 dataset.yml: {out_yaml}")  # 생성된 YAML 경로를 알려준다.

# ------------------------------------------------------------
# 실행 스크립트
# ------------------------------------------------------------
if __name__ == "__main__":  # 스크립트로 직접 실행될 때만 동작한다.
    build_stage_from_metadata(CSV_PATH, STAGE_ROOT, OUT_YAML)  # 메인 함수를 호출해 모든 작업을 수행한다.
