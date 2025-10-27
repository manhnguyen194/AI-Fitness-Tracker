import numpy as np
import random
import time
from utils.geometry import calculate_angle
from utils.draw_utils import draw_colored_line
from utils.feedback_utils import AIFeedbackManager

# ======================
# ⚙️ Khởi tạo Feedback Manager toàn cục
# ======================
feedback_manager = AIFeedbackManager(cooldown=2.0)

form_memory = {
    "lowest_angle": None,
    "rep_active": False,
}

# ======================
# 🏋️‍♂️ Squat Evaluation
# ======================
def evaluate_squat(keypoints, frame=None, stage=None, counter=None):
    left_hip, left_knee, left_ankle = keypoints[11], keypoints[13], keypoints[15]
    right_hip, right_knee, right_ankle = keypoints[12], keypoints[14], keypoints[16]
    left_shoulder, right_shoulder = keypoints[5], keypoints[6]

    # Tính góc trung bình đầu gối
    left_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_angle = calculate_angle(right_hip, right_knee, right_ankle)
    mean_angle = (left_angle + right_angle) / 2

    issues, score = [], 100
    ok_color, warn_color, err_color = (0, 255, 0), (0, 255, 255), (0, 0, 255)

    if stage == "down":
        form_memory["rep_active"] = True
        if form_memory["lowest_angle"] is None or mean_angle < form_memory["lowest_angle"]:
            form_memory["lowest_angle"] = mean_angle

    if stage == "up" and form_memory["rep_active"]:
        lowest_angle = form_memory["lowest_angle"] or mean_angle

        if lowest_angle > 130:
            issues.append("Cần hạ thấp hơn ở lần sau.")
            score -= 25
            if frame is not None:
                draw_colored_line(frame, left_hip, left_knee, err_color)

        shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
        hip_y = (left_hip[1] + right_hip[1]) / 2
        if shoulder_y - hip_y < 40:
            issues.append("Giữ lưng thẳng hơn.")
            score -= 20
            if frame is not None:
                draw_colored_line(frame, left_shoulder, left_hip, warn_color)

        feedback, tone = feedback_manager.get_feedback("squat", score, issues)
        form_memory["rep_active"] = False
        form_memory["lowest_angle"] = None
        return score, feedback, tone

    return 100, feedback_manager.last_feedback, "neutral"


# ======================
# 🤸‍♀️ Push-up Evaluation
# ======================
def evaluate_pushup(keypoints, frame=None, stage=None, counter=None):
    left_shoulder, left_elbow, left_wrist = keypoints[5], keypoints[7], keypoints[9]
    right_shoulder, right_elbow, right_wrist = keypoints[6], keypoints[8], keypoints[10]
    left_hip, right_hip = keypoints[11], keypoints[12]

    left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    mean_angle = (left_angle + right_angle) / 2

    issues, score = [], 100
    ok_color, warn_color, err_color = (0, 255, 0), (0, 255, 255), (0, 0, 255)

    if stage == "down":
        form_memory["rep_active"] = True
        if form_memory["lowest_angle"] is None or mean_angle < form_memory["lowest_angle"]:
            form_memory["lowest_angle"] = mean_angle

    if stage == "up" and form_memory["rep_active"]:
        lowest_angle = form_memory["lowest_angle"] or mean_angle

        if lowest_angle > 100:
            issues.append("Chưa hạ người đủ sâu.")
            score -= 20

        shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
        hip_y = (left_hip[1] + right_hip[1]) / 2
        if hip_y - shoulder_y > 80:
            issues.append("Giữ hông cao hơn nhé.")
            score -= 25
            if frame is not None:
                draw_colored_line(frame, left_hip, left_shoulder, warn_color)

        feedback, tone = feedback_manager.get_feedback("pushup", score, issues)
        form_memory["rep_active"] = False
        form_memory["lowest_angle"] = None
        return score, feedback, tone

    return 100, feedback_manager.last_feedback, "neutral"


# ======================
# 🧍‍♀️ Plank Evaluation
# ======================
def evaluate_plank(keypoints, frame=None, stage=None, counter=None):
    left_shoulder, left_hip, left_ankle = keypoints[5], keypoints[11], keypoints[15]
    angle = calculate_angle(left_shoulder, left_hip, left_ankle)

    issues, score = [], 100
    color = (0, 255, 0)

    if angle < 160:
        issues.append("Hông bị xệ, nâng cao hơn chút.")
        score -= 20
        color = (0, 0, 255)
    elif angle > 190:
        issues.append("Lưng cong quá, siết core lại.")
        score -= 15
        color = (0, 255, 255)

    if frame is not None and issues:
        draw_colored_line(frame, left_shoulder, left_hip, color)
        draw_colored_line(frame, left_hip, left_ankle, color)

    feedback, tone = feedback_manager.get_feedback("plank", score, issues)
    return score, feedback, tone

# ======================
# 🪶 Sit-up Evaluation
# ======================
def evaluate_situp(keypoints, frame=None, stage=None, counter=None):
    """
    Đánh giá động tác Sit-up:
    - Kiểm tra độ sâu khi gập người (vai – hông – gối)
    - Kiểm tra độ thẳng khi nằm xuống
    - Phản hồi khi hoàn thành rep (up)
    """

    left_shoulder, right_shoulder = keypoints[5], keypoints[6]
    left_hip, right_hip = keypoints[11], keypoints[12]
    left_knee, right_knee = keypoints[13], keypoints[14]
    left_ear, right_ear = keypoints[3], keypoints[4] if len(keypoints) > 4 else (left_shoulder, right_shoulder)

    # Góc giữa thân và đùi
    left_angle = calculate_angle(left_shoulder, left_hip, left_knee)
    right_angle = calculate_angle(right_shoulder, right_hip, right_knee)
    mean_angle = (left_angle + right_angle) / 2

    issues, score = [], 100
    ok_color, warn_color, err_color = (0, 255, 0), (0, 255, 255), (0, 0, 255)

    # Ghi nhận độ sâu khi đang xuống
    if stage == "down":
        form_memory["rep_active"] = True
        if (form_memory.get("lowest_angle") is None) or (mean_angle < form_memory["lowest_angle"]):
            form_memory["lowest_angle"] = mean_angle

    # Khi vừa ngồi dậy (stage == "up")
    if stage == "up" and form_memory.get("rep_active", False):
        lowest_angle = form_memory.get("lowest_angle", mean_angle)

        # 1️⃣ Độ sâu: gập chưa đủ
        if lowest_angle > 130:
            issues.append("Chưa gập người đủ sâu, cố gắng chạm gối hoặc cao hơn.")
            score -= 25
            if frame is not None:
                draw_colored_line(frame, left_shoulder, left_hip, err_color)
                draw_colored_line(frame, right_shoulder, right_hip, err_color)

        # 2️⃣ Độ thẳng khi nằm xuống: kiểm tra head-hip-knee khi duỗi ra
        head_y = (left_ear[1] + right_ear[1]) / 2
        hip_y = (left_hip[1] + right_hip[1]) / 2
        if head_y < hip_y - 40:  # đầu không cùng mặt phẳng khi nằm xuống
            issues.append("Thả đầu quá mạnh, kiểm soát hạ người xuống.")
            score -= 15

        # 3️⃣ Độ ổn định: kiểm tra vai không lệch
        shoulder_diff = abs(left_shoulder[1] - right_shoulder[1])
        if shoulder_diff > 40:
            issues.append("Giữ vai cân bằng khi gập người.")
            score -= 10

        # AI Feedback
        feedback, tone = feedback_manager.get_feedback("situp", score, issues)

        # Reset trạng thái cho rep tiếp theo
        form_memory["rep_active"] = False
        form_memory["lowest_angle"] = None

        return score, feedback, tone

    # Nếu chưa hoàn thành rep, giữ nguyên feedback cũ
    return 100, feedback_manager.last_feedback, "neutral"