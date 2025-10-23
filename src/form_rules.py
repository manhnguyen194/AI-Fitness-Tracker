import numpy as np
import random
import time
from utils.geometry import calculate_angle
from utils.draw_utils import draw_colored_line

# ======================
# Biến trạng thái toàn cục (có thể chuyển thành class sau)
# ======================
last_feedback = "Bắt đầu bài tập nào!"
last_feedback_time = 0
form_memory = {
    "lowest_angle": None,
    "rep_active": False,
}


# ======================
# Đánh giá Squat
# ======================
def evaluate_squat(keypoints, frame=None, stage=None, counter=None):
    """
    AI-based squat evaluation: chỉ đánh giá khi hoàn thành rep (stage chuyển từ down → up)
    """
    global last_feedback, last_feedback_time, form_memory

    left_hip, left_knee, left_ankle = keypoints[11], keypoints[13], keypoints[15]
    right_hip, right_knee, right_ankle = keypoints[12], keypoints[14], keypoints[16]
    left_shoulder, right_shoulder = keypoints[5], keypoints[6]

    # Tính góc đầu gối trung bình
    left_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_angle = calculate_angle(right_hip, right_knee, right_ankle)
    mean_angle = (left_angle + right_angle) / 2

    ok_color, warn_color, err_color = (0, 255, 0), (0, 255, 255), (0, 0, 255)

    # Ghi nhận điểm thấp nhất khi đang xuống
    if stage == "down":
        form_memory["rep_active"] = True
        if (form_memory["lowest_angle"] is None) or (mean_angle < form_memory["lowest_angle"]):
            form_memory["lowest_angle"] = mean_angle

    # Khi rep hoàn thành (up)
    if stage == "up" and form_memory["rep_active"]:
        lowest_angle = form_memory["lowest_angle"] or mean_angle
        score, feedbacks = 100, []

        # 1️⃣ Độ sâu
        if lowest_angle > 130:
            feedbacks.append("Cần hạ thấp hơn ở lần sau.")
            score -= 25
            if frame is not None:
                draw_colored_line(frame, left_hip, left_knee, err_color)
                draw_colored_line(frame, right_hip, right_knee, err_color)

        # 2️⃣ Lưng thẳng (dựa trên snapshot)
        shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
        hip_y = (left_hip[1] + right_hip[1]) / 2
        if shoulder_y - hip_y < 40:
            feedbacks.append("Giữ lưng thẳng hơn.")
            score -= 20
            if frame is not None:
                draw_colored_line(frame, left_shoulder, left_hip, warn_color)
                draw_colored_line(frame, right_shoulder, right_hip, warn_color)

        # 3️⃣ Feedback AI
        if score > 90:
            ai_feedback = random.choice(["Rất tốt!", "Form chuẩn rồi!", "Hoàn hảo!"])
        elif score > 70:
            ai_feedback = random.choice(["Ổn, hạ sâu hơn nhé.", "Giữ lưng thẳng hơn một chút."])
        else:
            ai_feedback = random.choice(["Chưa ổn, chậm lại để kiểm soát form.", "Sai form, tập lại chậm hơn."])

        # Gộp phản hồi
        final_feedback = ai_feedback + " " + " ".join(feedbacks)
        last_feedback = final_feedback
        last_feedback_time = time.time()

        # Reset cho rep kế tiếp
        form_memory["rep_active"] = False
        form_memory["lowest_angle"] = None

    # Nếu không hoàn thành rep → giữ feedback cũ trong 2 giây
    elif time.time() - last_feedback_time > 2:
        pass  # không thay đổi gì

    return 100, last_feedback  # score hiển thị luôn 100, feedback ổn định


# ======================
# Đánh giá Push-up
# ======================
def evaluate_pushup(keypoints, frame=None, stage=None, counter=None):
    global last_feedback, last_feedback_time, form_memory

    left_shoulder, left_elbow, left_wrist = keypoints[5], keypoints[7], keypoints[9]
    right_shoulder, right_elbow, right_wrist = keypoints[6], keypoints[8], keypoints[10]
    left_hip, right_hip = keypoints[11], keypoints[12]

    left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    mean_angle = (left_angle + right_angle) / 2

    ok_color, warn_color, err_color = (0, 255, 0), (0, 255, 255), (0, 0, 255)

    # Ghi nhận điểm thấp nhất (khi đang hạ)
    if stage == "down":
        form_memory["rep_active"] = True
        if (form_memory["lowest_angle"] is None) or (mean_angle < form_memory["lowest_angle"]):
            form_memory["lowest_angle"] = mean_angle

    # Khi lên (rep kết thúc)
    if stage == "up" and form_memory["rep_active"]:
        lowest_angle = form_memory["lowest_angle"] or mean_angle
        score, feedbacks = 100, []

        # Tay chưa duỗi hết
        if lowest_angle > 100:
            feedbacks.append("Chưa chạm đủ sâu.")
            score -= 20

        # Hông xệ
        shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
        hip_y = (left_hip[1] + right_hip[1]) / 2
        if hip_y - shoulder_y > 80:
            feedbacks.append("Giữ hông cao hơn nhé.")
            score -= 25
            if frame is not None:
                draw_colored_line(frame, left_hip, left_shoulder, warn_color)
                draw_colored_line(frame, right_hip, right_shoulder, warn_color)

        # AI feedback
        if score > 90:
            ai_feedback = random.choice(["Xuất sắc!", "Form rất chuẩn!"])
        elif score > 70:
            ai_feedback = random.choice(["Ổn, cố duỗi tay mạnh hơn.", "Giữ hông cao hơn một chút."])
        else:
            ai_feedback = random.choice(["Sai form rồi, tập lại chậm hơn.", "Cần kiểm soát chuyển động."])

        final_feedback = ai_feedback + " " + " ".join(feedbacks)
        last_feedback = final_feedback
        last_feedback_time = time.time()

        form_memory["rep_active"] = False
        form_memory["lowest_angle"] = None

    return 100, last_feedback
