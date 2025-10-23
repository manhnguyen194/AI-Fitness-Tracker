import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
qq
import cv2
import time
from ultralytics import YOLO

# Import module hỗ trợ
from utils.draw_utils import draw_text_pil
from utils.video_utils import setup_window, compute_fps
from rep_counter import count_squat, count_pushup
from form_rules import evaluate_squat, evaluate_pushup

# -----------------------------
# ⚙️ Cấu hình
# -----------------------------
EXERCISE =  "squat" # hoặc "squat"
VIDEO_PATH = "data/raw/squat_ok_01.mp4"  # hoặc 0 nếu dùng webcam
FONT_PATH = os.path.join(os.path.dirname(__file__), "..", "fonts", "Roboto.ttf")

# -----------------------------
# 🚀 Khởi tạo model
# -----------------------------
model = YOLO("yolo11n-pose.pt")
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("❌ Không thể mở video hoặc webcam:", VIDEO_PATH)
    exit()

state = {"stage": "up", "counter": 0}
prev_time = time.time()
frame_idx = 0
print("▶️ Bắt đầu. Nhấn 'q' để thoát.")

exercise_registry = {
    "squat": {
        "counter_func": count_squat,
        "form_func": evaluate_squat,
        "state": {"stage": "up", "counter": 0, "prev_angle": 160, "direction": "up"},
    },
    "pushup": {
        "counter_func": count_pushup,
        "form_func": evaluate_pushup,
        "state": {"stage": "up", "counter": 0, "prev_angle": 160, "direction": "up"},
    },
    # Thêm bài tập mới ở đây:
    # "lunges": {"counter_func": count_lunges, "form_func": evaluate_lunges, "state": {...}}
}

if EXERCISE not in exercise_registry:
    raise ValueError(f"❌ Bài tập '{EXERCISE}' chưa được đăng ký trong exercise_registry!")

counter_func = exercise_registry[EXERCISE]["counter_func"]
form_func = exercise_registry[EXERCISE]["form_func"]
state = exercise_registry[EXERCISE]["state"]

# -----------------------------
# 🔁 Vòng lặp chính
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("🎬 Hết video hoặc lỗi đọc frame.")
        break

    results = model(frame, verbose=False)
    res = results[0]
    annotated = res.plot()

    counter = 0
    stage = "up"
    angle = 0
    feedback = "..."

    # Nếu có keypoints → xử lý
    if res.keypoints is not None and len(res.keypoints.xy) > 0:
        kps = res.keypoints.xy[0].tolist()

        # Gọi hàm đếm và đánh giá form tương ứng bài tập
        counter, stage, angle = counter_func(kps, state)
        form_score, feedback = form_func(kps, annotated, stage, counter)

    # FPS
    fps, prev_time = compute_fps(prev_time)

    # -----------------------------
    # 🖼️ Overlay text
    # -----------------------------
    form_color = (0, 255, 0) if "tốt" in feedback or "chuẩn" in feedback else (255, 255, 0) if "có thể" in feedback else (255, 80, 80)
    lines = [
        (f"Số lần: {counter}", (255, 215, 0)),
        (f"Trạng thái: {stage}", (255, 255, 255)),
        (f"Góc: {int(angle)}°", (144, 238, 144)),
        (f"Đánh giá: {feedback}", form_color),
        (f"FPS: {fps:.1f}", (200, 200, 200)),
    ]

    annotated = draw_text_pil(annotated, lines, font_path=FONT_PATH, font_scale=26, pos=(20, 20))

    # -----------------------------
    # 🖥️ Hiển thị video auto-scale
    # -----------------------------
    if frame_idx == 0:
        setup_window("AI Fitness Tracker", annotated, max_height=720)

    cv2.imshow("AI Fitness Tracker", annotated)
    frame_idx += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
