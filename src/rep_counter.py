import time
from utils.geometry import calculate_angle

DELTA_THRESHOLD = 5  # độ thay đổi nhỏ thì bỏ qua (để tránh rung hình)

# ======================================
# 🦵 SQUAT
# ======================================
KNEE_ANGLE_DOWN = 100   # squat xuống khi nhỏ hơn giá trị này
KNEE_ANGLE_UP = 150     # đứng thẳng khi lớn hơn giá trị này

def count_squat(kps, state):
    """
    Đếm số rep + xác định hướng di chuyển dựa trên biến thiên góc đầu gối
    """
    left_hip = kps[11]
    left_knee = kps[13]
    left_ankle = kps[15]

    angle = calculate_angle(left_hip, left_knee, left_ankle)
    prev_angle = state.get("prev_angle", angle)
    delta = angle - prev_angle

    # Phát hiện hướng di chuyển (tăng/giảm góc)
    if abs(delta) > DELTA_THRESHOLD:
        direction = "down" if delta < 0 else "up"
        state["direction"] = direction

    # Xử lý đếm rep khi đi lên qua góc 150°
    if angle > KNEE_ANGLE_UP and state.get("stage") == "down":
        state["counter"] += 1
        state["stage"] = "up"
        print(f"✅ Đếm được {state['counter']} squat")
    elif angle < KNEE_ANGLE_DOWN and state.get("stage") != "down":
        state["stage"] = "down"

    state["prev_angle"] = angle
    return state["counter"], state.get("direction", "up"), angle

# ======================================
# 💪 PUSH-UP
# ======================================
ELBOW_ANGLE_DOWN = 90    # cúi người xuống khi góc khuỷu tay < 90°
ELBOW_ANGLE_UP = 160     # duỗi thẳng khi góc khuỷu tay > 160°

def count_pushup(kps, state):
    """
    Đếm số rep push-up dựa trên cả hai tay để phù hợp nhiều góc quay.
    kps: keypoints từ mô hình pose estimation
    state: dict lưu trạng thái hiện tại {'counter', 'stage', 'prev_angle', 'direction'}
    """
    # Tay trái
    left_shoulder, left_elbow, left_wrist = kps[5], kps[7], kps[9]
    left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

    # Tay phải
    right_shoulder, right_elbow, right_wrist = kps[6], kps[8], kps[10]
    right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

    # Chọn góc nhỏ hơn (tay nào gập nhiều hơn)
    angle = min(left_angle, right_angle)
    prev_angle = state.get("prev_angle", angle)
    delta = angle - prev_angle

    # Phát hiện hướng di chuyển
    if abs(delta) > DELTA_THRESHOLD:
        direction = "down" if delta < 0 else "up"
        state["direction"] = direction

    # Xử lý đếm rep khi đi lên qua góc ELBOW_ANGLE_UP
    if angle > ELBOW_ANGLE_UP and state.get("stage") == "down":
        state["counter"] += 1
        state["stage"] = "up"
        print(f"✅ Đếm được {state['counter']} push-up")
    elif angle < ELBOW_ANGLE_DOWN and state.get("stage") != "down":
        state["stage"] = "down"

    state["prev_angle"] = angle
    return state["counter"], state.get("direction", "up"), angle

# ======================================
# 🧍‍♀️ PLANK
# ======================================
PLANK_MIN_ANGLE = 160  # lưng-hông thẳng
PLANK_MAX_ANGLE = 190

def count_plank(kps, state):
    """
    Đếm thời gian giữ plank với form đúng.
    - kps: keypoints
    - state: {'start_time', 'good_time', 'bad_time', 'is_good'}
    """
    left_shoulder, left_hip, left_ankle = kps[5], kps[11], kps[15]
    angle = calculate_angle(left_shoulder, left_hip, left_ankle)

    # Xác định form có tốt không
    is_good = PLANK_MIN_ANGLE <= angle <= PLANK_MAX_ANGLE
    now = time.time()

    if "start_time" not in state:
        state["start_time"] = now
        state["good_time"] = 0
        state["bad_time"] = 0
        state["is_good"] = is_good

    elapsed = now - state["start_time"]
    if is_good:
        state["good_time"] += 1 / 30.0  # ~30 FPS giả định
    else:
        state["bad_time"] += 1 / 30.0

    # Form ổn định trong ≥1s thì báo OK
    feedback = "Form đúng" if is_good else "Hông bị xệ" if angle < PLANK_MIN_ANGLE else "Lưng cong"

    state["angle"] = angle
    state["elapsed"] = elapsed
    state["feedback"] = feedback
    state["is_good"] = is_good

    return int(state["good_time"]), "holding", angle

# ======================================
# 🤸 SIT-UP
# ======================================
SITUP_DOWN_ANGLE = 140   # nằm ngả ra sau
SITUP_UP_ANGLE = 80      # gập người lên

def count_situp(kps, state):
    """
    Đếm số rep sit-up dựa trên góc giữa vai – hông – gối.
    """
    left_shoulder, left_hip, left_knee = kps[5], kps[11], kps[13]
    right_shoulder, right_hip, right_knee = kps[6], kps[12], kps[14]

    left_angle = calculate_angle(left_shoulder, left_hip, left_knee)
    right_angle = calculate_angle(right_shoulder, right_hip, right_knee)
    mean_angle = (left_angle + right_angle) / 2

    prev_angle = state.get("prev_angle", mean_angle)
    delta = mean_angle - prev_angle

    if abs(delta) > DELTA_THRESHOLD:
        direction = "up" if delta < 0 else "down"  # Khi góc giảm → gập người lên
        state["direction"] = direction

    if mean_angle > SITUP_DOWN_ANGLE and state.get("stage") == "up":
        state["stage"] = "down"
    elif mean_angle < SITUP_UP_ANGLE and state.get("stage") == "down":
        state["counter"] += 1
        state["stage"] = "up"
        print(f"✅ Đếm được {state['counter']} sit-up")

    state["prev_angle"] = mean_angle
    return state["counter"], state.get("direction", "down"), mean_angle