import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob
import re
from pathlib import Path
from tqdm import tqdm



def plot_instance_keypoints(kpt_data, instance_num=0, conf_thr=0.07, num_keypoints=17):
    """
    특정 인스턴스의 모든 키포인트(x, y) 궤적을 서브플롯으로 시각화합니다.
    신뢰도가 낮은 구간은 빨간색으로 표시합니다.

    Args:
        kpt_data (np.ndarray): (Frame, Max_Instances, Keypoints, 3) 형태의 배열
        instance_num (int): 시각화할 인스턴스(사람) 인덱스 (기본값: 0)
        conf_thr (float): 신뢰도 임계값. 이보다 낮으면 빨간색으로 표시 (기본값: 0.3)
        num_keypoints (int): 전체 키포인트 개수 (COCO=17, YOLO=12 등)
    """
    
    # 1. 데이터 유효성 검사
    frames, max_instances, kpts, dims = kpt_data.shape
    if instance_num >= max_instances:
        print(f"[Error] instance_num({instance_num})이 Max Instances({max_instances})를 초과했습니다.")
        return
    
    if kpts != num_keypoints:
        print(f"[Warn] 데이터의 키포인트 개수({kpts})와 설정된 개수({num_keypoints})가 다릅니다.")

    # 2. 서브플롯 생성 (행: 키포인트 수, 열: 2 [X, Y])
    # 크기를 넉넉하게 잡습니다 (가로 12, 세로 키포인트 개수 * 2)
    fig, axes = plt.subplots(num_keypoints, 2, figsize=(12, 2 * num_keypoints), sharex=True)
    
    # 간격 조절
    plt.subplots_adjust(hspace=0.4)
    fig.suptitle(f"Trajectories of Instance {instance_num} (Threshold={conf_thr})", fontsize=16, y=0.99)

    # 시간축 (Frame Index)
    frame_indices = np.arange(frames)

    # 3. 각 키포인트별 반복 시각화
    for k in range(num_keypoints):
        # 데이터 추출: (Frames, 3) -> [x, y, score]
        data = kpt_data[:, instance_num, k, :]
        
        x_vals = data[:, 0]
        y_vals = data[:, 1]
        scores = data[:, 2]

        # --- [왼쪽 그래프] X 좌표 ---
        ax_x = axes[k, 0]
        
        # 3-1. 전체 궤적 (파란 실선) - NaN이 있으면 끊겨서 보임
        ax_x.plot(frame_indices, x_vals, 'b-', alpha=0.5, linewidth=1, label='Trajectory')
        
        # 3-2. 신뢰도에 따른 색상 구분 (산점도)
        # 정상 (High Confidence): 파란 점
        mask_high = scores > conf_thr
        ax_x.scatter(frame_indices[mask_high], x_vals[mask_high], c='blue', s=2, alpha=0.6)
        
        # 이상 (Low Confidence): 빨간 점
        mask_low = scores <= conf_thr
        ax_x.scatter(frame_indices[mask_low], x_vals[mask_low], c='red', s=5, label=f'Low Conf (<{conf_thr})')

        ax_x.set_ylabel(f'KP {k} (X)')
        ax_x.grid(True, linestyle='--', alpha=0.5)
        if k == 0: ax_x.legend(loc='upper right', fontsize='small')

        # --- [오른쪽 그래프] Y 좌표 ---
        ax_y = axes[k, 1]
        
        # 전체 궤적
        ax_y.plot(frame_indices, y_vals, 'g-', alpha=0.5, linewidth=1, label='Trajectory')
        
        # 정상: 초록 점
        ax_y.scatter(frame_indices[mask_high], y_vals[mask_high], c='green', s=2, alpha=0.6)
        
        # 이상: 빨간 점
        ax_y.scatter(frame_indices[mask_low], y_vals[mask_low], c='red', s=5)

        ax_y.set_ylabel(f'KP {k} (Y)')
        ax_y.grid(True, linestyle='--', alpha=0.5)

    # 마지막 행에만 X축 레이블 추가
    axes[-1, 0].set_xlabel('Frame Index')
    axes[-1, 1].set_xlabel('Frame Index')

    plt.tight_layout(rect=[0, 0, 1, 0.99]) # 타이틀 공간 확보
    plt.show()


def render_video_with_keypoints(
    frame_dir: str,
    keypoints_data: np.ndarray, 
    output_path: str = "output_video.mp4",
    fps: int = 30,
    conf_threshold: float = 0.0
):
    """
    이미지 폴더와 키포인트 배열(Numpy)을 입력받아 시각화 영상을 생성합니다.
    
    Args:
        frame_dir (str): 원본 이미지가 들어있는 폴더 경로
        keypoints_data (np.ndarray): (Frame, 1, 17, 3) 또는 (Frame, 17, 3) 형태의 데이터
                                     [x, y, confidence]
        output_path (str): 저장할 동영상 파일 경로 (.mp4)
        fps (int): 초당 프레임 수
        conf_threshold (float): 시각화할 최소 신뢰도 (이보다 낮으면 안 그림)
    """
    
    # 1. 이미지 파일 리스트업 및 정렬 (숫자 기준)
    frame_dir = Path(frame_dir)
    image_files = sorted(
        glob.glob(str(frame_dir / "*.jpg")), 
        key=lambda x: int(re.findall(r'\d+', os.path.basename(x))[-1])
    )
    
    if not image_files:
        print(f"[Error] '{frame_dir}' 폴더에 이미지가 없습니다.")
        return

    # 2. 데이터와 이미지 개수 매칭
    num_frames = min(len(image_files), keypoints_data.shape[0])
    
    # 3. 비디오 설정 (첫 번째 이미지로 크기 결정)
    first_img = cv2.imread(image_files[0])
    if first_img is None:
        print("[Error] 이미지를 읽을 수 없습니다.")
        return
    h, w, _ = first_img.shape
    
    # 출력 폴더 자동 생성
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # VideoWriter 초기화
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    # 4. 스켈레톤 연결 정보 (COCO Format 17 Keypoints)
    # (시작점, 끝점) 인덱스 튜플
    skeleton_links = [
        (0,1), (0,2), (1,3), (2,4),      # 얼굴
        (5,6),                           # 어깨 연결
        (5,7), (7,9),                    # 왼팔
        (6,8), (8,10),                   # 오른팔
        (5,11), (6,12),                  # 몸통
        (11,12),                         # 골반 연결
        (11,13), (13,15),                # 왼쪽 다리
        (12,14), (14,16)                 # 오른쪽 다리
    ]
    
    # 색상 설정 (BGR)
    color_point = (0, 0, 255)    # 빨간색 점
    color_line = (255, 0, 0)     # 파란색 선
    
    print(f" >> [Rendering] Creating video: {output_path} ({num_frames} frames)")

    for i in tqdm(range(num_frames)):
        # 이미지 로드
        frame = cv2.imread(image_files[i])
        if frame is None: break
        
        # 현재 프레임의 키포인트 추출
        # 입력이 (F, 1, 17, 3)인 경우 -> (17, 3)으로 차원 축소
        if keypoints_data.ndim == 4:
            kpts = keypoints_data[i, 0] 
        else:
            kpts = keypoints_data[i] # (17, 3) 가정
            
        # --- 그리기 로직 ---
        # 1. 선(Limb) 그리기
        for u, v in skeleton_links:
            # u, v 인덱스의 (x, y, conf) 가져오기
            if u < len(kpts) and v < len(kpts):
                x1, y1, c1 = kpts[u]
                x2, y2, c2 = kpts[v]
                
                # NaN이 아니고 신뢰도가 높을 때만 그리기
                if (not np.isnan(x1) and not np.isnan(x2) and 
                    c1 > conf_threshold and c2 > conf_threshold):
                    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color_line, 2)
        
        # 2. 점(Point) 그리기
        for k in range(len(kpts)):
            x, y, c = kpts[k]
            if not np.isnan(x) and not np.isnan(y) and c > conf_threshold:
                cv2.circle(frame, (int(x), int(y)), 4, color_point, -1)
                
        # 3. 프레임 번호 표시
        cv2.putText(frame, f"Frame: {i}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # 비디오에 쓰기
        out.write(frame)

    out.release()
    print(" >> [Done] Video saved.")