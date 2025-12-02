import json
from pathlib import Path
from typing import Tuple, List, Any, Union

# 1. 파일명에서 정보 추출 헬퍼 함수
def extract_info_from_filepath(
    filepath: str,
    delimiter: str = '__'
) -> Tuple[str, str, str]:
    """
    절대 경로 또는 파일명에서 정보를 파싱합니다.
    """
    path_obj = Path(filepath)
    filename = path_obj.stem

    parts = filename.split(delimiter)

    if len(parts) >= 2:
        view_angle = parts[0]
        action_name = parts[1]
    else:
        view_angle = "UNKNOWN"
        action_name = "UNKNOWN"
    
    return filename, view_angle, action_name

# 2. 세그먼트 파일 파싱 메인 함수
def parse_segment_frames_data(
    segment_file_path: Union[str, Path], 
    primary_delimiter: str = ',',
    label_delimiter: str = '__'
) -> List[List[Any]]:
    """
    segment.txt 파일의 경로를 받아 내용을 읽고, 모든 세그먼트 정보를 파싱하여 리스트의 리스트로 반환합니다.
    
    Args:
        segment_file_path: segment.txt 파일의 경로 (str 또는 Path 객체)
        
    Returns:
        List[List]: [[filename, start, end, angle, action], ...] 형태의 전체 리스트
    """
    parsed_data = []
    
    # Path 객체로 변환
    file_path = Path(segment_file_path)
    
    if not file_path.exists():
        print(f"[ERROR] 파일을 찾을 수 없습니다: {file_path}")
        return []

    try:
        # 파일을 직접 엽니다 (기존 오류가 발생하던 지점 해결)
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            
            # 헤더나 주석 행 건너뛰기
            if not line or line.startswith('#'):
                continue
                
            parts = line.split(primary_delimiter)
            
            if len(parts) < 3:
                continue

            try:
                label = parts[0].strip()
                start_frame = int(parts[1].strip())
                end_frame = int(parts[2].strip())
                
                # 헬퍼 함수를 활용하여 각도와 동작 이름 추출
                _, view_angle, action_name = extract_info_from_filepath(label, delimiter=label_delimiter)
                
                # 리스트 형태로 저장
                parsed_data.append([
                    label,           # 0: filename
                    start_frame,     # 1: start_frame
                    end_frame,       # 2: end_frame
                    view_angle,      # 3: view_angle
                    action_name,     # 4: action_name
                ])
                
            except ValueError:
                print(f"[WARN] 데이터 파싱 오류 (숫자 변환 실패): {line}")
                continue 

    except Exception as e:
        print(f"[ERROR] 파일 읽기 중 오류 발생: {e}")
        return []

    return parsed_data