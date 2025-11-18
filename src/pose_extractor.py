import os
import cv2
import time
import torch
from ultralytics import YOLO

# -----------------------------
# 📦 Import module hỗ trợ
# -----------------------------
from utils.draw_utils import draw_text_pil
from utils.video_utils import setup_window, compute_fps
from rep_counter import (
    count_squat, count_pushup, count_plank, count_situp
)
from form_rules import (
    evaluate_squat, evaluate_pushup, evaluate_plank, evaluate_situp
)

import voice_player

# -----------------------------
# ⚙️ Cấu hình
# -----------------------------
EXERCISE =  "pushup"  # hoặc "squat", "plank", "situp"

# Chọn nguồn vào: webcam hay video
USE_WEBCAM = False      # Đổi True/False để chọn nguồn
WEBCAM_INDEX = 0       # Chỉ số webcam (mặc định 0)

# Đường dẫn video dùng khi USE_WEBCAM = False
VIDEO_REL = os.path.join("../data", "raw", "pushup_ok_01.mp4")
# file data/ nằm bên trong src/, không phải ở project root -> không cần ".."
VIDEO_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), VIDEO_REL))

# Xác định nguồn cho VideoCapture
if USE_WEBCAM:
    CAP_SOURCE = WEBCAM_INDEX
    print(f"▶️ Nguồn: Webcam({WEBCAM_INDEX})")
else:
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ Video không tìm thấy tại: {VIDEO_PATH}")
        print(f"➜ Tự động chuyển sang webcam ({WEBCAM_INDEX}).")
        CAP_SOURCE = WEBCAM_INDEX
    else:
        CAP_SOURCE = VIDEO_PATH
        print(f"▶️ Nguồn: Video → {VIDEO_PATH}")

FONT_PATH = os.path.join(os.path.dirname(__file__), "..", "fonts", "Roboto.ttf")

# Add/modify these configurations at the top
BATCH_SIZE = 1
IMG_SIZE = 640  # or 480 for faster processing
DRAW_EVERY_N_FRAMES = 3  # Increase to 3 or 4 for higher FPS
INFERENCE_EVERY = 3  # chạy model.predict mỗi N frames (giảm inference)
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
    # allow_tf32 attributes may not exist on some torch versions; guard with try
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

# Modify model initialization
model = YOLO("yolo11n-pose.pt")
model.conf = CONF_THRESHOLD
try:
    model.to(device)
except Exception:
    # some ultralytics versions auto-handle device; ignore if .to() fails
    pass

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("❌ Không thể mở video hoặc webcam:", VIDEO_PATH)
    exit()
print("▶️ Bắt đầu. Nhấn 'q' để thoát.")

# -----------------------------
# 🧩 Đăng ký bài tập
# -----------------------------
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
    "plank": {
        "counter_func": count_plank,
        "form_func": evaluate_plank,
        "state": {"good_time": 0, "bad_time": 0, "is_good": False},
    },
    "situp": {
        "counter_func": count_situp,
        "form_func": evaluate_situp,
        "state": {"stage": "down", "counter": 0, "prev_angle": 140, "direction": "down"},
    },
    # Thêm bài tập mới ở đây:
    # "situp": {"counter_func": count_situp,
    #            "form_func": evaluate_situp,
    #            "state": {...}},
}

if EXERCISE not in exercise_registry:
    raise ValueError(f"❌ Bài tập '{EXERCISE}' chưa được đăng ký trong exercise_registry!")

counter_func = exercise_registry[EXERCISE]["counter_func"]
form_func = exercise_registry[EXERCISE]["form_func"]
state = exercise_registry[EXERCISE]["state"]


# --- Voice player init ---
# Build the path dynamically relative to the current script
BASE_DIR = os.path.dirname(__file__)  # folder containing this file (e.g. src/)
VOICES_DIR = os.path.join(BASE_DIR, "data", "voices")

# phát welcome.mp3 → đợi 2s → lần đầu theo tone là 5s, sau đó giữ nguyên tone thì 4s
voice_player.init(VOICES_DIR, base_interval_first=6.0, base_interval_same=5.0)

# --- Voice player init (periodic-only) ---

# Khởi tạo periodic player: mỗi 3 giây đọc tone hiện tại và phát
voice_player.init(VOICES_DIR, base_interval_first=10.0, base_interval_same=10.0)
# ------------------------------------------------



# -----------------------------
# 🔁 Vòng lặp chính
# -----------------------------
prev_time = time.time()
frame_idx = 0
last_annotated = None     # cache frame đã vẽ skeleton/annotation
last_results = None       # cache kết quả model (ultralytics 'result' object)
last_kps = None           # cache keypoints list để reuse cho rep counter
while True:
    ret, frame = cap.read()
    if not ret:
        print("🎬 Hết video hoặc lỗi đọc frame.")
        break
    # Resize frame for consistent input size (trade-off: smaller -> faster)
    frame_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

    # --- Chỉ chạy inference mỗi INFERENCE_EVERY frames ---
    do_inference = (frame_idx % INFERENCE_EVERY == 0)

    if do_inference:
        try:
            results = model.predict(frame_resized,
                                    verbose=False,
                                    conf=CONF_THRESHOLD,
                                    device=device,
                                    batch=BATCH_SIZE)
            # results returned as list-like; lấy phần tử đầu
            res = results[0] if results and len(results) > 0 else None
            last_results = res
        except Exception as e:
            # Trong trường hợp predict lỗi, giữ last_results (nếu có)
            print("⚠️ Lỗi khi predict:", e)
            res = last_results
    else:
        # reuse last inference result
        res = last_results

    # -----------------------------
    # Chỉ vẽ annotation mỗi DRAW_EVERY_N_FRAMES; reuse ảnh đã vẽ nếu có
    # -----------------------------
    if frame_idx % DRAW_EVERY_N_FRAMES == 0 and res is not None:
        try:
            annotated = res.plot()  # ultralytics result.plot()
            last_annotated = annotated
        except Exception:
            # nếu res.plot() fail, fallback to drawing on resized frame
            annotated = frame_resized.copy()
            last_annotated = annotated
    else:
        # reuse last annotated image nếu có, else dùng resized frame
        annotated = last_annotated if last_annotated is not None else frame_resized.copy()

    counter = 0
    stage = "up"
    angle = 0
    feedback = "..."

    # Nếu có keypoints → xử lý (dùng last_results nếu không predict frame này)
    if res is not None and getattr(res, "keypoints", None) is not None:
        try:
            # res.keypoints.xy có thể trả về mảng Nx( ... ), lấy bộ đầu tiên nếu có nhiều người
            kps_arr = res.keypoints.xy
            if kps_arr is not None and len(kps_arr) > 0:
                kps = kps_arr[0].tolist()
                last_kps = kps
            else:
                # không có keypoints trong kết quả hiện tại → fallback dùng last_kps
                kps = last_kps
        except Exception:
            kps = last_kps
    else:
        kps = last_kps

    if kps is not None:
        # Gọi hàm đếm và đánh giá form tương ứng bài tập
        try:
            counter, stage, angle = counter_func(kps, state)
            form_score, feedback, tone = form_func(kps, annotated, stage, counter)
            # Cập nhật voice player với tone hiện tại (tone mapping do bạn design)
            voice_player.set_tone(tone)
            form_color = (0, 255, 0) if tone == "good" else (0, 0, 255)
        except Exception as e:
            # in ra lỗi xử lý rep/form nhưng không crash vòng lặp
            print("⚠️ Lỗi xử lý rep/form:", e)
            form_color = (255, 255, 255)
            feedback = "Lỗi xử lý"
    else:
        form_color = (255, 255, 255)
        feedback = "Không phát hiện người"

    # -----------------------------
    # 🧮 Tính FPS
    # -----------------------------
    fps, prev_time = compute_fps(prev_time)

    # -----------------------------
    # 🖼️ Overlay text
    # -----------------------------
    # For plank display, use elapsed time (counter represents elapsed now)
    if EXERCISE == "plank":
        elapsed = state.get("elapsed", float(counter) if counter is not None else 0.0)
        lines = [
            (f"Thời gian giữ: {elapsed:.1f}s", (255, 215, 0)),
            (f"Tư thế: {'Chuẩn' if state.get('is_good') else 'Chưa đúng'}", (255, 255, 255)),
            (f"Góc: {int(angle)}°", (144, 238, 144)),
            (f"Đánh giá: {feedback}", form_color),
            (f"FPS: {fps:.1f}", (200, 200, 200)),
        ]
    else:
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
voice_player.stop()
