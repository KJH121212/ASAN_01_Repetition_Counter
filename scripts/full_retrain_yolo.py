# ================================================================
# full_retrain_yolo.py
# ✅ YOLOv11-Pose (12 keypoints 구조) 완전 재학습 스크립트
#    - 기존 17kp pretrained backbone만 transfer
#    - pose head는 12kp로 재초기화
#    - Ultralytics YOLOv11 기반 (pose)
# ================================================================

from ultralytics import YOLO                     # YOLO 라이브러리 (Ultralytics v8 이상)
from pathlib import Path                         # 경로 처리를 위한 Path
import time                                      # 실행 시간 측정용
import os                                        # 환경 변수 설정용

# ------------------------------------------------
# 1️⃣ 기본 경로 설정
# ------------------------------------------------
BASE_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/Kimjihoo/ASAN_01_Repeatition_Counter")
DATA_YAML = BASE_DIR / "data/dataset.yml"         # YOLO dataset.yml (kpt_shape=[12,3])
PRETRAINED = BASE_DIR / "checkpoints/yolo_pose/yolo11m-pose.pt"  # COCO 17kp pretrained
OUTPUT_DIR = BASE_DIR / "checkpoints/yolo_pose/yolo11_pose_12kp_fullretrain"

# ------------------------------------------------
# 2️⃣ 학습 환경 설정
# ------------------------------------------------
os.environ["YOLO_VERBOSE"] = "True"              # YOLO 로그 자세히 출력
os.environ["CUDA_VISIBLE_DEVICES"] = "0"         # 사용할 GPU 지정 (예: GPU0)

EPOCHS = 100                                    # 학습 epoch 수
IMG_SIZE = 640                                  # 입력 이미지 크기
BATCH = 16                                      # 배치 크기 (GPU VRAM에 맞게 조정)
LR = 0.0005                                     # 학습률
FREEZE = 10                                     # backbone freeze 단계 수 (0=전체 학습, 10=neck 이후만 학습)

# ------------------------------------------------
# 3️⃣ YOLO 모델 로드 및 구조 확인
# ------------------------------------------------
print(f"[INFO] ✅ Pretrained Backbone 로드 중... → {PRETRAINED}")
model = YOLO(str(PRETRAINED))                   # COCO17kp 기반 pretrained 로드

# 모델 구조 출력
model.info(verbose=True)

# ------------------------------------------------
# 4️⃣ 12 Keypoints 구조로 맞춤 재설정
# ------------------------------------------------
# dataset.yml 내부 kpt_shape=[12,3]을 따르도록 head 자동 조정
print("[INFO] ✅ Pose head를 12 keypoints 구조로 재초기화합니다.")
model.model.model[-1].kpt_shape = [12, 3]       # keypoint 개수 변경
model.model.model[-1].nc = 1                    # 클래스 개수 (patient)

# ✅ initialize_biases()는 Pose head에는 없음 → hasattr로 보호
if hasattr(model.model.model[-1], "initialize_biases"):
    model.model.model[-1].initialize_biases()
else:
    print("[INFO] (skip) Pose head에는 initialize_biases() 없음, 무시합니다.")
# ------------------------------------------------
# 5️⃣ 학습 실행
# ------------------------------------------------
print(f"[INFO] 🚀 YOLOv11 Pose 12kp Full Retrain 시작 ({EPOCHS} epochs)")
start = time.time()

results = model.train(
    data=str(DATA_YAML),                        # dataset.yml 경로
    epochs=EPOCHS,                              # 학습 epoch 수
    imgsz=IMG_SIZE,                             # 입력 이미지 크기
    lr0=LR,                                     # 초기 학습률
    batch=BATCH,                                # 배치 크기
    device=0,                                   # GPU 선택
    project=str(OUTPUT_DIR.parent),             # 결과 상위 폴더
    name=OUTPUT_DIR.name,                       # 세부 폴더 이름
    exist_ok=False,                              # 동일 폴더 덮어쓰기 허용
    pretrained=False,                           # 헤드는 새로 학습하므로 False
    freeze=FREEZE,                              # backbone 일부 freeze
    optimizer="SGD",                            # SGD 또는 AdamW
    verbose=True,                               # 로그 자세히 출력

    workers=4,                # DataLoader 병렬 처리 (GPU 1개라도 OK)
    cache=False,               # 이미지 캐시 (다음 epoch부터 빠름)
)

end = time.time()
elapsed = (end - start) / 60
print(f"\n✅ 학습 완료 | 총 소요 시간: {elapsed:.1f}분")
print(f"📦 결과 저장 경로: {OUTPUT_DIR}")

# ------------------------------------------------
# 6️⃣ 최종 모델 평가 (Validation)
# ------------------------------------------------
print("\n[INFO] ✅ Validation 평가 시작...")
val_results = model.val(
    data=str(DATA_YAML),
    imgsz=IMG_SIZE,
    batch=BATCH,
    split="val",
    device=0,
)
print("📊 Validation 완료:", val_results)
