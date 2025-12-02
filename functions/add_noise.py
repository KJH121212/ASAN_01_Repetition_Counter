import numpy as np

# def : add_all_noise 

# --- 1. 독립 스파이크 노이즈 함수 (이전과 동일) ---
def add_independent_spike_noise(kpts_array: np.ndarray, frame_spike_ratio: float = 0.005, start_frame: int = 30) -> np.ndarray:
    """
    각 키포인트(K)별로 독립적이고 무작위적인 프레임을 선택하여 X, Y 좌표를 0으로 대체합니다.
    """
    if kpts_array.ndim != 3:
        raise ValueError("입력 배열은 (N, K, 2+) 형태여야 합니다.")
    
    noisy_kpts_array = kpts_array.copy()
    num_frames, num_kpts, _ = noisy_kpts_array.shape
    
    if start_frame >= num_frames:
        print("경고: 시작 프레임이 총 프레임 수보다 크거나 같습니다. 스파이크 노이즈를 추가하지 않습니다.")
        return noisy_kpts_array
    
    num_spike_frames_per_kpt = int(num_frames * frame_spike_ratio)
    available_frames_for_spiking = num_frames - start_frame
    
    if num_spike_frames_per_kpt > available_frames_for_spiking:
        num_spike_frames_per_kpt = available_frames_for_spiking
    
    if num_spike_frames_per_kpt == 0:
        print("경고: 각 키포인트당 스파이크 노이즈를 적용할 프레임이 0개입니다. 노이즈를 추가하지 않습니다.")
        return noisy_kpts_array

    frame_indices_pool = np.arange(start_frame, num_frames)
    total_spikes_applied = 0
    
    for kp_id in range(num_kpts):
        spike_frame_indices = np.random.choice(
            frame_indices_pool, 
            size=num_spike_frames_per_kpt, 
            replace=False
        )
        
        for f in spike_frame_indices:
            target_slice = noisy_kpts_array[f, kp_id, :2]
            valid_mask = ~np.isnan(target_slice)
            target_slice[valid_mask] = 0.0
            total_spikes_applied += 1
            
    print(f"--- 💥 독립 스파이크 노이즈 추가 완료: 총 {total_spikes_applied}개 스파이크 적용 ---")
    return noisy_kpts_array

# --- 2. 선택적 가우시안 노이즈 함수 (랜덤 프레임 및 랜덤 키포인트 적용) ---
def add_selective_gaussian_noise(
    kpts_array: np.ndarray, 
    frame_noise_ratio: float = 0.1, 
    kp_ratio: float = 0.5, 
    noise_std: float = 5.0
) -> np.ndarray:
    """
    무작위로 선택된 프레임(frame_noise_ratio)과 무작위로 선택된 키포인트(kp_ratio)에만
    가우시안 노이즈를 추가합니다.

    Args:
        kpts_array (np.ndarray): 입력 키포인트 배열. (N, K, 2+) 형태.
        frame_noise_ratio (float): 노이즈를 적용할 프레임의 비율 (0.0 ~ 1.0). (기본값 0.1)
        kp_ratio (float): 노이즈를 적용할 키포인트의 비율 (0.0 ~ 1.0). (기본값 0.5)
        noise_std (float): 가우시안 노이즈의 표준편차 (Standard Deviation). (기본값 5.0)

    Returns:
        np.ndarray: 노이즈가 추가된 배열.
    """
    
    if kpts_array.ndim != 3:
        raise ValueError("입력 배열은 (N, K, 2+) 형태여야 합니다.")
    
    noisy_kpts_array = kpts_array.copy()
    num_frames, num_kpts, num_coords = noisy_kpts_array.shape
    
    if num_coords < 2:
        raise ValueError("좌표 개수가 2개 미만입니다. (X, Y) 좌표가 필요합니다.")

    # 1. 노이즈를 적용할 프레임 선택
    num_noisy_frames = int(num_frames * frame_noise_ratio)
    if num_noisy_frames == 0:
        print("경고: 노이즈를 적용할 프레임이 0개입니다. 가우시안 노이즈를 추가하지 않습니다.")
        return noisy_kpts_array
        
    all_frame_indices = np.arange(num_frames)
    noisy_frame_indices = np.random.choice(
        all_frame_indices, 
        size=num_noisy_frames, 
        replace=False
    )
    
    # 2. 노이즈를 적용할 키포인트 선택 (모든 프레임에서 동일한 키포인트를 선택하지 않음)
    num_noisy_kpts = int(num_kpts * kp_ratio)
    if num_noisy_kpts == 0:
        print("경고: 노이즈를 적용할 키포인트가 0개입니다. 가우시안 노이즈를 추가하지 않습니다.")
        return noisy_kpts_array
        
    kpt_indices_pool = np.arange(num_kpts)
    
    total_noise_points = 0
    
    # 3. 선택된 프레임을 반복하며 키포인트에 노이즈 적용
    for f in noisy_frame_indices:
        # 해당 프레임에서 무작위로 num_noisy_kpts 개 키포인트 선택
        noisy_kpt_indices = np.random.choice(
            kpt_indices_pool, 
            size=num_noisy_kpts, 
            replace=False
        )
        
        for kp_id in noisy_kpt_indices:
            # 해당 프레임, 해당 키포인트의 (X, Y) 슬라이스
            target_coords = noisy_kpts_array[f, kp_id, :2]
            
            # 가우시안 노이즈 생성 (1, 2) 크기
            gaussian_noise = np.random.normal(loc=0.0, scale=noise_std, size=target_coords.shape)
            
            # 유효한 값(NaN이 아닌 값)에만 노이즈 추가
            valid_mask = ~np.isnan(target_coords)
            target_coords[valid_mask] += gaussian_noise[valid_mask]
            total_noise_points += np.sum(valid_mask)

            noisy_kpts_array[f, kp_id, :2] = target_coords
    
    print(f"--- 🌊 선택적 가우시안 노이즈 추가 완료: {num_noisy_frames}개 프레임({frame_noise_ratio*100:.1f}%)에 노이즈 적용 ---")
    print(f"--- 📢 프레임당 {num_noisy_kpts}개 키포인트({kp_ratio*100:.1f}%)에 노이즈 적용 ---")
    
    return noisy_kpts_array

# --- 3. 최종 통합 함수 (이전과 동일) ---
def add_all_noise(
    kpts_array: np.ndarray, 
    spike_ratio: float = 0.005, 
    spike_start_frame: int = 30, 
    gaussian_frame_ratio: float = 0.1,  # 프레임 비율 추가
    gaussian_kp_ratio: float = 0.5, 
    gaussian_std: float = 5.0
) -> np.ndarray:
    """
    스파이크 노이즈와 가우시안 노이즈를 순차적으로 적용하여 최종 노이즈 배열을 생성합니다.
    """
    
    # 1. 독립 스파이크 노이즈 적용 (0 스파이크)
    intermediate_data = add_independent_spike_noise(
        kpts_array, 
        frame_spike_ratio=spike_ratio, 
        start_frame=spike_start_frame
    )
    
    # 2. 선택적 가우시안 노이즈 적용 (잡음)
    final_noisy_data = add_selective_gaussian_noise(
        intermediate_data, 
        frame_noise_ratio=gaussian_frame_ratio,  # 프레임 비율 전달
        kp_ratio=gaussian_kp_ratio, 
        noise_std=gaussian_std
    )
    
    print("--- ✅ 모든 노이즈 통합 적용 완료 ---")
    return final_noisy_data
