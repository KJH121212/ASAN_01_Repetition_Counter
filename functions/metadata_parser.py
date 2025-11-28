from pathlib import Path
from typing import Tuple, List, Dict, Any

# 파일명에서 view_angle, action_name 추출 하는 함수
def extract_info_from_filepath(
    filepath: str,
    delimiter: str = '__'
) -> Tuple[str,str,str]:
    path_obj = Path(filepath)
    filename = path_obj.stem            # .stem: Path 내부 함수로 파일 확장자를 제외한 파일명을 나타냄

    parts = filename.split(delimiter) # 파일명을 Delimiter을 기준으로 나눔.

    if len(parts) >= 2:
        view_angle = parts[0]
        action_name = parts[1]
    else:
        view_angle = "UNKNOWN"
        action_name = "UNKNOWN"
    
    return filename, view_angle, action_name

# segment_frames.txt에서 label, start_frame, end_frame, view_angle, action_name, 반환
# result[0][0] 형태로 결과 나옴.
def parse_segment_frames_data(
    segment_content: str, 
    primary_delimiter: str = ',',
    label_delimiter: str = '__'
) -> List[Dict[str, Any]]:
    parsed_data = []
    
    # 텍스트 내용을 줄 단위로 분리하여 처리
    for line in segment_content.strip().split('\n'):
        line = line.strip()
        
        # 헤더나 주석 행 건너뛰기
        if not line or line.startswith('#'):
            continue
            
        # 1. start_frame, end_frame 분리
        parts = line.split(primary_delimiter)
        
        if len(parts) < 3:
            # 필수 데이터(label, start, end) 부족 시 건너뛰기
            continue

        try:
            # 쉼표로 분리된 세그먼트 정보 추출
            label = parts[0].strip()
            start_frame = int(parts[1].strip())
            end_frame = int(parts[2].strip())
            
            _, view_angle, action_name = extract_info_from_filepath(label, delimiter=label_delimiter)

            parsed_data.append([
                label,
                start_frame,
                end_frame,
                view_angle,
                action_name,
                ])
            
        except ValueError:
            # 프레임 숫자로 변환 실패 시 (데이터 오류) 건너뛰기
            continue 

    return parsed_data