import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from typing import List, Dict, Any, Tuple

# --- 1. PoseTracker 클래스 (ID 할당용) ---
class Track:
    def __init__(self, track_id, instance):
        self.track_id = track_id
        self.update(instance)
    def update(self, instance):
        self.bbox = instance['bbox'][0]
        self.keypoints = np.array(instance['keypoints'])
    def mark_missed(self): pass

class PoseTracker:
    def __init__(self, max_lost=5):
        self.next_id = 1
        self.tracks = []
        self.max_lost = max_lost

    def _calculate_iou(self, bbox1, bbox2):
        x1, y1, x2, y2 = max(bbox1[0], bbox2[0]), max(bbox1[1], bbox2[1]), min(bbox1[2], bbox2[2]), min(bbox1[3], bbox2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1, area2 = (bbox1[2]-bbox1[0])*(bbox1[3]-bbox1[1]), (bbox2[2]-bbox2[0])*(bbox2[3]-bbox2[1])
        return inter / (area1 + area2 - inter + 1e-6)

    def update(self, instances):
        # (간소화된 헝가리안 매칭 로직)
        if not self.tracks:
            for inst in instances:
                self.tracks.append(Track(self.next_id, inst))
                inst['track_id'] = self.next_id
                self.next_id += 1
            return instances

        cost_matrix = np.zeros((len(self.tracks), len(instances)))
        for t, track in enumerate(self.tracks):
            for d, det in enumerate(instances):
                cost_matrix[t, d] = 1.0 - self._calculate_iou(track.bbox, det['bbox'][0])
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matched_track_indices = set()
        matched_det_indices = set()

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 0.8:
                self.tracks[r].update(instances[c])
                instances[c]['track_id'] = self.tracks[r].track_id
                matched_track_indices.add(r)
                matched_det_indices.add(c)

        # Unmatched detections -> New Tracks
        for i, inst in enumerate(instances):
            if i not in matched_det_indices:
                self.tracks.append(Track(self.next_id, inst))
                inst['track_id'] = self.next_id
                self.next_id += 1
                
        # (Lost Track 삭제 로직은 생략)
        return instances

# --- 2. 헬퍼 함수 ---
def get_distance_from_center(bbox, frame_w, frame_h):
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    return np.sqrt((cx - frame_w/2)**2 + (cy - frame_h/2)**2)

# --- 3. 메인 함수: Center-Priority Tracking ---
def extract_patient(
    data: Dict[str, Any], 
    frame_width: int = 1280, 
    frame_height: int = 720,
    switch_threshold: float = 100.0 # 픽셀 단위: 이만큼 더 가까워야 타겟을 바꿈 (빈번한 교체 방지)
) -> np.ndarray:
    """
    1. 모든 객체를 Tracking하여 ID를 부여합니다.
    2. 현재 추적 중인 Target ID가 있더라도, 
       다른 객체가 화면 중앙에 '훨씬 더(switch_threshold)' 가깝다면 Target ID를 변경합니다.
    """
    
    frame_data_list = data.get('frame_data', [])
    num_frames = len(frame_data_list)
    
    # 결과 배열 (Frame, 17, 2)
    patient_kpts_array = np.full((num_frames, 17, 2), np.nan)
    
    # 트래커 인스턴스 생성
    tracker = PoseTracker(max_lost=5)
    
    # 상태 변수
    current_target_id = None
    
    print(f"[INFO] Center-Priority Tracking 시작 (총 {num_frames} 프레임)")
    
    # 프레임 정렬
    sorted_frames = sorted(frame_data_list, key=lambda x: x[0])
    if not sorted_frames:
        return patient_kpts_array
        
    start_frame_num = sorted_frames[0][0]

    for frame_num, instances in sorted_frames:
        idx = frame_num - start_frame_num
        if idx >= num_frames: break
        
        if not instances:
            continue

        # 1. 트래킹 수행 (ID 할당)
        # tracked_instances의 각 요소에는 이제 'track_id'가 포함됨
        tracked_instances = tracker.update(instances)
        
        # 2. 후보군 분석 (ID, 거리 계산)
        candidates = []
        for inst in tracked_instances:
            dist = get_distance_from_center(inst['bbox'][0], frame_width, frame_height)
            candidates.append({
                'id': inst['track_id'],
                'dist': dist,
                'inst': inst
            })
        
        # 거리 순으로 정렬 (가장 가까운게 0번)
        candidates.sort(key=lambda x: x['dist'])
        best_candidate = candidates[0] # 이번 프레임의 중앙 챔피언
        
        # 3. 타겟 결정 로직 (Switching Logic)
        
        # Case A: 아직 타겟이 없는 경우 -> 가장 가까운 놈을 타겟으로
        if current_target_id is None:
            current_target_id = best_candidate['id']
            target_inst = best_candidate['inst']
            # print(f"Frame {frame_num}: 초기 타겟 설정 ID {current_target_id}")

        # Case B: 타겟이 있는 경우
        else:
            # 현재 타겟이 이번 프레임에도 존재하는지 확인
            curr_target_candidate = next((c for c in candidates if c['id'] == current_target_id), None)
            
            if curr_target_candidate:
                # 현재 타겟이 여전히 존재함.
                # 그러나 다른 누군가가 중앙에 '훨씬' 더 가까운가?
                if best_candidate['id'] != current_target_id:
                    # 조건: (현재 타겟 거리) - (도전자 거리) > 임계값
                    # 즉, 도전자가 기존 타겟보다 100픽셀 이상 더 중앙에 있어야 바꿈 (떨림 방지)
                    if (curr_target_candidate['dist'] - best_candidate['dist']) > switch_threshold:
                        # print(f"Frame {frame_num}: 타겟 변경! ID {current_target_id} -> {best_candidate['id']} (더 중앙으로 옴)")
                        current_target_id = best_candidate['id']
                        target_inst = best_candidate['inst']
                    else:
                        # 별 차이 없으면 의리 지킴 (기존 타겟 유지)
                        target_inst = curr_target_candidate['inst']
                else:
                    # 현재 타겟이 여전히 1등임
                    target_inst = curr_target_candidate['inst']
            else:
                # 현재 타겟이 화면에서 사라짐 (Lost)
                # print(f"Frame {frame_num}: 타겟 ID {current_target_id} 사라짐. 새로운 타겟 ID {best_candidate['id']} 선정.")
                current_target_id = best_candidate['id']
                target_inst = best_candidate['inst']

        # 4. 데이터 추출
        if target_inst:
            kpts = np.array(target_inst['keypoints'])
            if kpts.shape[0] == 17:
                patient_kpts_array[idx] = kpts[:, :2]

    print(f"[INFO] 추출 완료. shape: {patient_kpts_array.shape}")
    return patient_kpts_array

