#!/bin/bash
#SBATCH -J tojihoo_metadata_update
#SBATCH -t 7-00:00:00
#SBATCH -o /mnt/nas203/ds_RehabilitationMedicineData/IDs/Kimjihoo/ASAN_01_Repeatition_Counter/batch/logs/%A.out
#SBATCH --mail-type END,TIME_LIMIT_90,REQUEUE,INVALID_DEPEND
#SBATCH --mail-user jihu6033@gmail.com
#SBATCH -p RTX3090
#SBATCH --gpus 1

export HTTP_PROXY=http://192.168.45.108:3128
export HTTPS_PROXY=http://192.168.45.108:3128
export http_proxy=http://192.168.45.108:3128
export https_proxy=http://192.168.45.108:3128

# Define vars
DOCKER_IMAGE_NAME="kimjihoo/repetition-counter-yolo"
DOCKER_CONTAINER_NAME="kimjihoo_metadata-update"
RANDOM_PORT=$(( (RANDOM % 101) + 8888 ))  # Random port between 8888 and 8988

# ✅ Docker build
docker build \
  -t ${DOCKER_IMAGE_NAME} \
  -f /mnt/nas203/ds_RehabilitationMedicineData/IDs/Kimjihoo/ASAN_01_Repeatition_Counter/docker/Dockerfile.sapiens \
  /mnt/nas203/ds_RehabilitationMedicineData/IDs/Kimjihoo/ASAN_01_Repeatition_Counter/docker

# ✅ Docker run
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
    bash -c "
        cd /workspace/nas203/ds_RehabilitationMedicineData/IDs/Kimjihoo/ASAN_01_Repeatition_Counter/scripts && \
        python3 synk_metadata.py
    "
