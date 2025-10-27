# ===============================================================
# coco17.py - COCO 17 keypoints 구조 (Sapiens 기반)
# ===============================================================

COLOR_SK       = (50, 50, 50)
COLOR_L        = (255, 0, 0)
COLOR_R        = (0, 0, 255)
COLOR_NEUTRAL  = (0, 255, 0)

LEFT_POINTS    = [5, 7, 9, 11, 13, 15]
RIGHT_POINTS   = [6, 8, 10, 12, 14, 16]
EXCLUDE_POINTS = [0, 1, 2, 3, 4]

SKELETON_LINKS = [
    (15, 13), (13, 11),
    (16, 14), (14, 12),
    (11, 12),
    (5, 11), (6, 12),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (1, 2), (0, 1), (0, 2),
    (1, 3), (2, 4), (3, 5), (4, 6)
]

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle"
]
