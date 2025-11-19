"""
Tạo phản hồi động cho AI Fitness Tracker dựa trên điểm form và bài tập.
"""

import random
import time

# ========================================
# 🧠 AI Coach Personality (Preset messages)
# ========================================
FEEDBACK_SETS = {
    "positive": {
        "squat": [
            "Rất tốt! Form squat đang cực kỳ ổn định!",
            "Tuyệt vời! Giữ lưng và nhịp hạ rất chuẩn!",
            "Form squat chuẩn không cần chỉnh!",
        ],
        "pushup": [
            "Xuất sắc! Đẩy tay đều và chắc chắn!",
            "Rất tốt — lực hai tay cân bằng hoàn hảo!",
            "Đẩy mạnh và đều, form chuẩn lắm!",
        ],
        "plank": [
            "Giữ form plank rất chắc! Core siết tốt!",
            "Tư thế plank của bạn cực kỳ ổn định!",
            "Tuyệt! Duy trì nhịp thở đều như vậy nhé!",
        ],
        "situp": [
            "Rất tốt! Cảm nhận rõ lực ở cơ bụng!",
            "Tốt lắm! Sit-up nhịp đều và chuẩn!",
            "Form sit-up rất tốt, tiếp tục nào!",
        ],
    },
    "neutral": {
        "squat": [
            "Ổn rồi, nhưng hạ sâu hơn chút nhé.",
            "Giữ lưng thẳng hơn một chút nữa.",
            "Tập trung vào việc kiểm soát hông khi lên.",
        ],
        "pushup": [
            "Ổn, thử hạ thấp hơn chút để hiệu quả hơn.",
            "Giữ hông cao hơn một chút khi đẩy.",
            "Đừng vội, cảm nhận chuyển động đều hai tay.",
        ],
        "plank": [
            "Giữ vững tư thế thêm chút nữa!",
            "Cố gắng siết cơ core mạnh hơn!",
            "Duy trì đường thẳng từ vai tới gót nhé.",
        ],
        "situp": [
            "Ổn, nhưng nên giữ nhịp đều hơn.",
            "Thử gập người chậm và có kiểm soát hơn.",
            "Giữ mắt nhìn lên để bảo vệ cổ nhé.",
        ],
    },
    "negative": {
        "squat": [
            "Chưa đủ sâu, cố hạ thấp hơn!",
            "Sai form rồi, chậm lại để kiểm soát hông!",
            "Hông đang lệch, chỉnh lại tư thế nhé!",
        ],
        "pushup": [
            "Hông bị xệ, siết core lại ngay!",
            "Chưa duỗi hết tay, đẩy mạnh hơn!",
            "Form chưa ổn, hãy chậm lại và giữ nhịp đều.",
        ],
        "plank": [
            "Hông đang xệ xuống, nâng lên chút!",
            "Lưng cong rồi, siết bụng và chỉnh thẳng lại!",
            "Vai hơi cao, giữ thẳng người nào!",
        ],
        "situp": [
            "Chưa đủ gập người, cố thêm chút nữa!",
            "Sai form, cổ đang cúi quá sâu!",
            "Gập nhanh quá, hãy kiểm soát nhịp chậm lại!",
        ],
    },
}

# ========================================
# 🗣️ Main Function: AI Feedback Generator
# ========================================
def ai_feedback_generator(exercise, score, issues=None):
    """
    Sinh phản hồi AI tự nhiên dựa trên điểm và bài tập.
    Args:
        exercise (str): tên bài tập ("squat", "pushup", "plank", "lunge", "situp")
        score (float): điểm form (0–100)
        issues (list[str]): danh sách vấn đề phát hiện
    Returns:
        tuple[str, str]: (feedback, tone)
    """
    if issues is None:
        issues = []

    # Xác định tone phản hồi
    if score > 90:
        tone = "positive"
    elif score > 70:
        tone = "neutral"
    else:
        tone = "negative"

    base = random.choice(FEEDBACK_SETS[tone].get(exercise, ["Tốt lắm!"]))
    joined_issues = " ".join(issues)
    feedback = f"{base} {joined_issues}".strip()

    return feedback, tone


# ========================================
# 🧭 Smart Feedback Manager
# ========================================
class AIFeedbackManager:
    """
    Quản lý phản hồi AI với thời gian cooldown để tránh spam.
    Đồng bộ tốt với hệ thống Text-to-Speech hoặc hiển thị overlay.
    """
    def __init__(self, cooldown=2.0):
        self.last_feedback = "Bắt đầu bài tập nào!"
        self.last_time = 0
        self.last_tone = "neutral"
        self.cooldown = cooldown

    def get_feedback(self, exercise, score, issues=None):
        now = time.time()
        if now - self.last_time < self.cooldown:
            return self.last_feedback, self.last_tone

        feedback, tone = ai_feedback_generator(exercise, score, issues)
        self.last_feedback = feedback
        self.last_tone = tone
        self.last_time = now
        return feedback, tone
