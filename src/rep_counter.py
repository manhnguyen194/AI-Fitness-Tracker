from utils.geometry import calculate_angle

DELTA_THRESHOLD = 5  # độ thay đổi nhỏ thì bỏ qua (để tránh rung hình)

# Ngưỡng động tác (tùy chỉnh)
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

# Ngưỡng động tác push-up (tùy chỉnh)
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