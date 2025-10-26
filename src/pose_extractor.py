import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import cv2
import time
import torch
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
VIDEO_REL = os.path.join("data", "raw", "squat_ok_01.mp4")
# file data/ nằm bên trong src/, không phải ở project root -> không cần ".."
VIDEO_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), VIDEO_REL))

# Nếu không tìm thấy file thì thông báo rõ ràng và fallback sang webcam (0)
if not os.path.exists(VIDEO_PATH):
    print(f"❌ Video không tìm thấy tại: {VIDEO_PATH}")
    print("➜ Đặt file vào data/raw/ hoặc đổi VIDEO_PATH. Tự động chuyển sang webcam (0).")
    VIDEO_PATH = 0

print(f"▶️ Sử dụng video/webcam: {VIDEO_PATH}")

FONT_PATH = os.path.join(os.path.dirname(__file__), "..", "fonts", "Roboto.ttf")

# Add/modify these configurations at the top
BATCH_SIZE = 1
IMG_SIZE = 640  # or 480 for faster processing
DRAW_EVERY_N_FRAMES = 3  # Increase to 3 or 4 for higher FPS

# Thêm cấu hình sau phần CONFIG
CONF_THRESHOLD = 0.5     # Lọc bớt detection có độ tin cậy thấp

# -----------------------------
# 🚀 Khởi tạo model (với GPU nếu có)
# -----------------------------
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"▶️ Device: {device}")

# Debug CUDA status
print("\n=== 🔍 GPU/CUDA Status ===")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name()}")
    print(f"Device memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("⚠️ CUDA không khả dụng - model đang chạy trên CPU")
print("=====================\n")

# Tối ưu thêm cho CUDA
if device.startswith("cuda"):
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Modify model initialization
model = YOLO("yolo11n-pose.pt")
model.conf = CONF_THRESHOLD
model.to(device)

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
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

    # Use model.predict instead of direct call for better GPU utilization
    results = model.predict(frame, 
                       verbose=False,
                       conf=CONF_THRESHOLD,
                       device=device,
                       batch=BATCH_SIZE)
    res = results[0]
    
    # Chỉ vẽ annotation mỗi N frame
    if frame_idx % DRAW_EVERY_N_FRAMES == 0:
        annotated = res.plot()
    else:
        annotated = frame.copy()

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
