import time
import math
from utils.geometry import calculate_angle

DELTA_THRESHOLD = 5


def _valid_pt(pt):
    try:
        x, y = pt
        return x is not None and y is not None and not (math.isnan(x) or math.isnan(y))
    except Exception:
        return False


def _safe_angle(a, default=180.0):
    try:
        a = float(a)
        if math.isnan(a) or a is None: return default
        return a
    except Exception:
        return default


# === CẬP NHẬT TẤT CẢ CÁC HÀM ĐẾM ĐỂ NHẬN current_timestamp ===

def count_squat(kps, state, current_timestamp=None):
    # Squat không dùng time, nhưng cần nhận tham số để không lỗi
    left_hip, left_knee, left_ankle = kps[11], kps[13], kps[15]
    angle = calculate_angle(left_hip, left_knee, left_ankle)
    prev_angle = state.get("prev_angle", angle)
    delta = angle - prev_angle

    if abs(delta) > DELTA_THRESHOLD:
        direction = "down" if delta < 0 else "up"
        state["direction"] = direction

    if angle > 150 and state.get("stage") == "down":
        state["counter"] += 1
        state["stage"] = "up"
    elif angle < 100 and state.get("stage") != "down":
        state["stage"] = "down"

    state["prev_angle"] = angle
    return state["counter"], state.get("direction", "up"), angle


def count_pushup(kps, state, current_timestamp=None):
    # Pushup logic
    left_shoulder, left_elbow, left_wrist = kps[5], kps[7], kps[9]
    left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_shoulder, right_elbow, right_wrist = kps[6], kps[8], kps[10]
    right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    angle = min(left_angle, right_angle)

    prev_angle = state.get("prev_angle", angle)
    delta = angle - prev_angle

    if abs(delta) > DELTA_THRESHOLD:
        direction = "down" if delta < 0 else "up"
        state["direction"] = direction

    if angle > 160 and state.get("stage") == "down":
        state["counter"] += 1
        state["stage"] = "up"
    elif angle < 90 and state.get("stage") != "down":
        state["stage"] = "down"

    state["prev_angle"] = angle
    return state["counter"], state.get("direction", "up"), angle


# === 🔥 LOGIC PLANK ĐƯỢC SỬA LẠI TẠI ĐÂY 🔥 ===
def count_plank(kps, state, current_timestamp=None):
    """
    Sửa lỗi: Sử dụng current_timestamp (nếu có) thay vì time.time()
    """
    # 1. Xác định thời gian hiện tại (Video Time hoặc Real Time)
    if current_timestamp is not None:
        now = current_timestamp
    else:
        now = time.time()

    state.setdefault("start_time", None)
    state.setdefault("good_time", 0.0)
    state.setdefault("bad_time", 0.0)
    state.setdefault("is_good", False)
    state.setdefault("last_time", now)  # Init last_time bằng now hiện tại
    state.setdefault("elapsed", 0.0)
    state.setdefault("angle", 0.0)

    # Nếu mới bắt đầu bài tập (start_time chưa set), gán mốc thời gian
    if state.get("start_time") is None:
        state["start_time"] = now
        state["last_time"] = now

    # Tính dt (delta time) dựa trên thời điểm trước đó
    dt = now - state["last_time"]
    state["last_time"] = now  # Cập nhật cho frame sau

    # (Logic cũ giữ nguyên)
    try:
        left_shoulder, left_hip, left_ankle = kps[5], kps[11], kps[15]
        if not (_valid_pt(left_shoulder) and _valid_pt(left_hip) and _valid_pt(left_ankle)):
            raise ValueError("Invalid Points")
        angle = _safe_angle(calculate_angle(left_shoulder, left_hip, left_ankle))
    except Exception:
        return float(state.get("elapsed", 0.0)), "holding", float(state.get("angle", 0.0))

    # Logic đánh giá
    is_good = 160 <= angle <= 190
    if is_good:
        state["good_time"] += dt
    else:
        state["bad_time"] += dt

    state["angle"] = angle
    # Elapsed tính bằng tổng good + bad (chính xác hơn trừ start_time khi video bị nhảy cóc)
    state["elapsed"] = state["good_time"] + state["bad_time"]
    state["feedback"] = "Form chuẩn" if is_good else ("Hông thấp" if angle < 160 else "Lưng cong")
    state["is_good"] = is_good

    return float(state["elapsed"]), "holding", float(angle)


def count_situp(kps, state, current_timestamp=None):
    # Situp logic
    left_shoulder, left_hip, left_knee = kps[5], kps[11], kps[13]
    right_shoulder, right_hip, right_knee = kps[6], kps[12], kps[14]
    mean_angle = (calculate_angle(left_shoulder, left_hip, left_knee) +
                  calculate_angle(right_shoulder, right_hip, right_knee)) / 2

    prev_angle = state.get("prev_angle", mean_angle)
    delta = mean_angle - prev_angle
    if abs(delta) > DELTA_THRESHOLD:
        state["direction"] = "up" if delta < 0 else "down"

    if mean_angle > 140 and state.get("stage") == "up":
        state["stage"] = "down"
    elif mean_angle < 80 and state.get("stage") == "down":
        state["counter"] += 1
        state["stage"] = "up"

    state["prev_angle"] = mean_angle
    return state["counter"], state.get("direction", "down"), mean_angle