import numpy as np
import math

def calculate_angle(a, b, c):
    """
    Tính góc giữa 3 điểm (a, b, c) — điểm b là đỉnh.
    Góc trả về đơn vị độ (°).
    """
    try:
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba = a - b
        bc = c - b
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        return angle
    except Exception as e:
        print(f"⚠️ Lỗi tính góc: {e}")
        return 0.0


def euclidean_distance(p1, p2):
    """Tính khoảng cách Euclidean giữa hai điểm."""
    p1, p2 = np.array(p1), np.array(p2)
    return np.linalg.norm(p1 - p2)


def normalize_keypoints(keypoints, image_size=(640, 480)):
    """
    Chuẩn hóa tọa độ keypoint về [0, 1] dựa trên kích thước ảnh.
    Dùng cho bước huấn luyện hoặc tính feature ML.
    """
    h, w = image_size
    return [(x / w, y / h) for (x, y) in keypoints]
