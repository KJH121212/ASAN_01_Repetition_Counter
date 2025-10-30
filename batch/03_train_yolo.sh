#!/bin/bash
#SBATCH -J tojihoo_yolo
#SBATCH -t 7-00:00:00
#SBATCH -o /mnt/nas203/ds_RehabilitationMedicineData/IDs/Kimjihoo/ASAN_01_Repeatition_Counter/batch/logs/%A.out
#SBATCH --mail-type END,TIME_LIMIT_90,REQUEUE,INVALID_DEPEND
#SBATCH --mail-user jihu6033@gmail.com
#SBATCH -p RTX3090
#SBATCH --gpus 1

# ------------------------------------------------------------
# 환경 설정
# ------------------------------------------------------------
export HTTP_PROXY=http://192.168.45.108:3128
export HTTPS_PROXY=http://192.168.45.108:3128
export http_proxy=http://192.168.45.108:3128
export https_proxy=http://192.168.45.108:3128

DOCKER_IMAGE_NAME="kimjihoo/repetition-counter"
DOCKER_CONTAINER_NAME="kimjihoo_running-yolo"
DOCKERFILE_PATH="/mnt/nas203/ds_RehabilitationMedicineData/IDs/Kimjihoo/ASAN_01_Repeatition_Counter/docker/Dockerfile.sapiens"
WORKSPACE_PATH="/mnt/nas203/ds_RehabilitationMedicineData/IDs/Kimjihoo/ASAN_01_Repeatition_Counter"
RANDOM_PORT=$(( (RANDOM % 101) + 8000 ))  # 8000~8100 사이 포트
LOG_PATH="/mnt/nas203/ds_RehabilitationMedicineData/IDs/Kimjihoo/ASAN_01_Repeatition_Counter/batch/logs"

# ------------------------------------------------------------
# Docker 이미지 빌드
# ------------------------------------------------------------
echo "[INFO] Building Docker image: ${DOCKER_IMAGE_NAME}"
docker build -t ${DOCKER_IMAGE_NAME} -f ${DOCKERFILE_PATH} ${WORKSPACE_PATH}
if [ $? -ne 0 ]; then
    echo "[❌ ERROR] Docker build failed."
    exit 1
fi

# ------------------------------------------------------------
# Docker 컨테이너 실행
# ------------------------------------------------------------
echo "[INFO] Running container: ${DOCKER_CONTAINER_NAME}"
docker run -it --rm --device=nvidia.com/gpu=all --shm-size 1TB \
    --name "${DOCKER_CONTAINER_NAME}" \
    -e JUPYTER_ENABLE_LAB=yes \
    -p ${RANDOM_PORT}:${RANDOM_PORT} \
    -v /mnt:/workspace \
    -e HTTP_PROXY=${HTTP_PROXY} \
    -e HTTPS_PROXY=${HTTPS_PROXY} \
    -e http_proxy=${http_proxy} \
    -e https_proxy=${https_proxy} \
    ${DOCKER_IMAGE_NAME} \
    bash -lc "
        cd /workspace/nas203/ds_RehabilitationMedicineData/IDs/Kimjihoo/ASAN_01_Repeatition_Counter/scripts && \
        python3 train_yolo_pose.py train \
        --imgsz 640 \
        --device 0 \
        --patience 50 \
        --epochs 150 \
        --name yolo11_pose_12kp_ft2 \
        --resume \
        --wandb_project RepeatitionCounter
        "
