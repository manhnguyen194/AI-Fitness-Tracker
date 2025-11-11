import streamlit as st
import os
import cv2
import time
import torch
import av
import numpy as np
import copy
from ultralytics import YOLO
from tempfile import NamedTemporaryFile

import torch
print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda device count:", torch.cuda.device_count())
    try:
        print("device name:", torch.cuda.get_device_name(0))
    except Exception as e:
        print("device name error:", e)


# Sử dụng streamlit-webrtc cho webcam realtime
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# --- IMPORT CÁC MODULE CUSTOM CỦA BẠN ---
# Chú ý: Các file này phải nằm trong cùng thư mục hoặc trong PYTHONPATH
try:
    from utils.draw_utils import draw_text_pil
    from utils.video_utils import compute_fps
    from rep_counter import (
        count_squat, count_pushup, count_plank, count_situp
    )
    from form_rules import (
        evaluate_squat, evaluate_pushup, evaluate_plank, evaluate_situp
    )
    # Tắt voice_player vì server-side voice playback rất khó khăn trong Streamlit
    # import voice_player
    # print("Đã tải các module hỗ trợ.")
except ImportError as e:
    st.error(f"Lỗi tải module tùy chỉnh: {e}")
    st.info("Vui lòng đảm bảo các file 'rep_counter.py', 'form_rules.py', và 'utils/' tồn tại trong thư mục.")


    # Định nghĩa dummy functions để app vẫn chạy (chỉ vẽ, không tính toán)
    def dummy_func(*args, **kwargs):
        return (0, "up", 0)


    def dummy_eval(*args, **kwargs):
        return (0, "Module lỗi!", "neutral")


    count_squat, count_pushup, count_plank, count_situp = [dummy_func] * 4
    evaluate_squat, evaluate_pushup, evaluate_plank, evaluate_situp = [dummy_eval] * 4

# -----------------------------
# ⚙️ Cấu hình và Caching Model
# -----------------------------
IMG_SIZE = 480
CONF_THRESHOLD = 0.5
DRAW_EVERY_N_FRAMES = 3
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Target output resolution (1080p)
TARGET_W, TARGET_H = 1920, 1080

# Xác định đường dẫn font (ưu tiên Noto Sans có hỗ trợ tiếng Việt)
FONTS_DIR = os.path.join(BASE_DIR, "fonts")
os.makedirs(FONTS_DIR, exist_ok=True)
FONT_PATH = os.path.join(FONTS_DIR, "NotoSans-Regular.ttf")
if not os.path.exists(FONT_PATH):
    # fallback to Roboto if Noto not found
    FONT_PATH = os.path.join(FONTS_DIR, "Roboto-Regular.ttf")
if not os.path.exists(FONT_PATH):
    st.warning("Không tìm thấy font Unicode hỗ trợ tiếng Việt trong ./fonts/. Sẽ dùng font mặc định của OpenCV.")
    FONT_PATH = None


@st.cache_resource
def load_yolo_model():
    """Tải và cache model YOLO Pose."""
    # Use canonical device strings accepted by ultralytics/torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.info(f"Đang tải model trên device: {device} (cuda available: {torch.cuda.is_available()})")
    try:
        model = YOLO("yolo11n-pose.pt")
        model.conf = CONF_THRESHOLD
        # move model to device and set model args if needed
        try:
            model.to(device)
        except Exception as _:
            # ultralytics YOLO handles device via predict parameter; fallback ok
            pass
        return model, device
    except Exception as e:
        st.error(f"Lỗi tải model: {e}")
        return None, device


model, device = load_yolo_model()

# -----------------------------
# 🧩 Đăng ký Bài tập (Đồng bộ với code gốc)
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
        "state": {"good_time": 0, "bad_time": 0, "is_good": False, "elapsed": 0.0},
    },
    "situp": {
        "counter_func": count_situp,
        "form_func": evaluate_situp,
        "state": {"stage": "down", "counter": 0, "prev_angle": 140, "direction": "down"},
    },
}


# -----------------------------
# 🎥 CLASS XỬ LÝ VIDEO REALTIME (cho Webcam)
# -----------------------------

class PoseProcessor(VideoProcessorBase):
    """Xử lý từng khung hình video từ webcam."""

    def __init__(self, exercise_key, model, device):
        self.model = model
        self.device = device
        self.exercise_key = exercise_key

        if exercise_key not in exercise_registry:
            raise ValueError(f"Bài tập {exercise_key} không hợp lệ.")

        self.counter_func = exercise_registry[exercise_key]["counter_func"]
        self.form_func = exercise_registry[exercise_key]["form_func"]

        # Deep copy state để mỗi phiên web có state riêng
        self.state = copy.deepcopy(exercise_registry[exercise_key]["state"])

        self.prev_time = time.time()
        self.frame_idx = 0
        self.last_annotated = None
        self.FONT_PATH = FONT_PATH
        self.start_time = time.time()

    def process_frame(self, img):
        """Logic xử lý chính (tương tự vòng lặp while trong code gốc)"""

        # 1. Resize cho inference (giữ IMG_SIZE để model nhanh)
        infer_frame = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        results = self.model.predict(infer_frame,
                                     verbose=False,
                                     conf=CONF_THRESHOLD,
                                     device=self.device,
                                     batch=1)
        res = results[0]

        # 2. Vẽ annotation (dùng plot từ result; kích thước dựa trên infer_frame)
        if self.frame_idx % DRAW_EVERY_N_FRAMES == 0:
            annotated = res.plot()
            self.last_annotated = annotated
        else:
            annotated = self.last_annotated if self.last_annotated is not None else infer_frame.copy()

        counter = 0
        stage = "..."
        angle = 0
        feedback = "..."
        form_color = (255, 255, 255)

        # 3. Xử lý Keypoints
        if res.keypoints is not None and len(res.keypoints.xy) > 0:
            kps = res.keypoints.xy[0].tolist()

            try:
                # Cập nhật elapsed time cho Plank
                if self.exercise_key == "plank":
                    self.state['elapsed'] = time.time() - self.start_time

                counter, stage, angle = self.counter_func(kps, self.state)
                form_score, feedback, tone = self.form_func(kps, annotated, stage, counter)

                form_color = (0, 255, 0) if tone == "positive" or tone == "good" else (0, 0, 255)
            except Exception as e:
                # Xử lý lỗi nếu logic bị crash
                feedback = f"Lỗi xử lý: {e.__class__.__name__}"
                form_color = (0, 165, 255)  # Màu cam

        else:
            feedback = "Không phát hiện người"

        # 4. Tính FPS
        fps, self.prev_time = compute_fps(self.prev_time)

        # 5. Overlay Text
        if self.exercise_key == "plank":
            elapsed = self.state.get("elapsed", 0.0)
            lines = [
                (f"Bài tập: {self.exercise_key.upper()}", (255, 105, 180)),
                (f"Thời gian giữ: {elapsed:.1f}s", (255, 215, 0)),
                (f"Tư thế: {'Chuẩn' if self.state.get('is_good') else 'Chưa đúng'}", (255, 255, 255)),
            ]
        else:
            lines = [
                (f"Bài tập: {self.exercise_key.upper()}", (255, 105, 180)),
                (f"Số lần: {self.state.get('counter', 0)}", (255, 215, 0)),
                (f"Trạng thái: {stage}", (255, 255, 255)),
            ]

        lines.extend([
            (f"Góc: {int(angle)}°", (144, 238, 144)),
            (f"Đánh giá: {feedback}", form_color),
            (f"FPS: {fps:.1f}", (200, 200, 200)),
        ])

        annotated = draw_text_pil(annotated, lines, font_path=self.FONT_PATH, font_scale=26, pos=(20, 20))

        # 3. Resize annotated -> TARGET (1080p) before returning (preserve aspect by scaling then padding)
        try:
            h, w = annotated.shape[:2]
            if w == 0 or h == 0:
                final = cv2.resize(infer_frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)
            else:
                scale = min(TARGET_W / w, TARGET_H / h)
                new_w = max(1, int(w * scale))
                new_h = max(1, int(h * scale))
                interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
                resized = cv2.resize(annotated, (new_w, new_h), interpolation=interp)
                top = (TARGET_H - new_h) // 2
                bottom = TARGET_H - new_h - top
                left = (TARGET_W - new_w) // 2
                right = TARGET_W - new_w - left
                final = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])
        except Exception:
            final = cv2.resize(annotated, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)

        self.frame_idx += 1
        return final

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Callback nhận khung hình từ webrtc"""
        img = frame.to_ndarray(format="bgr24")
        processed_img = self.process_frame(img)

        # Chuyển lại về AV VideoFrame để hiển thị (processed_img đảm bảo là TARGET size)
        return av.VideoFrame.from_ndarray(processed_img, format="bgr24")


# -----------------------------
# 🌐 GIAO DIỆN STREAMLIT CHÍNH
# -----------------------------

st.title("🏋️ AI Fitness Tracker")
st.sidebar.title("Cấu hình")

# 1. Chọn Bài tập
exercise_choice = st.sidebar.selectbox(
    "Chọn bài tập:",
    list(exercise_registry.keys()),
    key='exercise'
)

# 2. Chọn Chế độ
mode = st.sidebar.radio(
    "Chọn chế độ hoạt động:",
    ["Webcam Realtime", "Phân tích Video Upload"]
)

if model is None:
    st.error("Không thể khởi tạo model. Vui lòng kiểm tra file yolo11n-pose.pt.")
elif mode == "Webcam Realtime":

    st.header("🔴 Webcam Realtime (Sử dụng `streamlit-webrtc`)")

    # Yêu cầu camera client capture ở 1080p (nếu trình duyệt/thiết bị hỗ trợ)
    media_constraints = {
        "video": {
            "width": {"ideal": TARGET_W},
            "height": {"ideal": TARGET_H},
            "frameRate": {"ideal": 30}
        },
        "audio": False
    }

    ctx = webrtc_streamer(
        key="webcam_processor",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        video_processor_factory=lambda: PoseProcessor(exercise_choice, model, device),
        media_stream_constraints=media_constraints,
        async_processing=True,
    )

    if ctx.state.playing:
        st.success(f"Đang phân tích tư thế {exercise_choice.upper()}...")
        # (Thông tin và feedback sẽ được hiển thị trực tiếp trên luồng video)


elif mode == "Phân tích Video Upload":

    st.header("⬆️ Phân tích Video Đã Tải Lên")

    uploaded_file = st.file_uploader(
        "Tải lên một file video (.mp4, .mov)",
        type=['mp4', 'mov']
    )

    if uploaded_file is not None:

        # 1. Lưu file tạm thời
        tfile = NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        tfile.close()
        video_path = tfile.name

        st.video(uploaded_file, format="video/mp4", start_time=0)

        start_button = st.button("Bắt đầu Phân tích Video")

        if start_button:
            st.info("Đang xử lý video... Quá trình này có thể mất thời gian tùy thuộc vào độ dài video.")

            # Khởi tạo logic xử lý
            counter_func = exercise_registry[exercise_choice]["counter_func"]
            form_func = exercise_registry[exercise_choice]["form_func"]
            state = copy.deepcopy(exercise_registry[exercise_choice]["state"])

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                st.error("Không thể mở video.")
            else:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps_input = cap.get(cv2.CAP_PROP_FPS)

                # Streamlit placeholder để cập nhật frame liên tục
                frame_placeholder = st.empty()
                progress_bar = st.progress(0)

                prev_time = time.time()
                frame_idx = 0
                last_annotated = None
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                if exercise_choice == "plank":
                    state['start_time'] = time.time()  # Giả định start_time của video là 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

                    # Xử lý tương tự như trong PoseProcessor
                    results = model.predict(frame, verbose=False, conf=CONF_THRESHOLD, device=device, batch=1)
                    res = results[0]

                    if frame_idx % DRAW_EVERY_N_FRAMES == 0:
                        annotated = res.plot()
                        last_annotated = annotated
                    else:
                        annotated = last_annotated if last_annotated is not None else frame.copy()

                    counter, stage, angle, feedback, form_color = 0, "...", 0, "...", (255, 255, 255)

                    if res.keypoints is not None and len(res.keypoints.xy) > 0:
                        kps = res.keypoints.xy[0].tolist()

                        if exercise_choice == "plank":
                            # Tính thời gian thực trong video
                            time_in_video = frame_idx / fps_input
                            state['elapsed'] = time_in_video

                        counter, stage, angle = counter_func(kps, state)
                        form_score, feedback, tone = form_func(kps, annotated, stage, counter)
                        form_color = (0, 255, 0) if tone == "positive" or tone == "good" else (0, 0, 255)
                    else:
                        feedback = "Không phát hiện người"

                    # Tính FPS (dựa trên tốc độ xử lý của máy tính)
                    fps, prev_time = compute_fps(prev_time)

                    # Overlay Text
                    if exercise_choice == "plank":
                        elapsed = state.get("elapsed", 0.0)
                        lines = [
                            (f"Bài tập: {exercise_choice.upper()}", (255, 105, 180)),
                            (f"Thời gian giữ: {elapsed:.1f}s", (255, 215, 0)),
                            (f"Tư thế: {'Chuẩn' if state.get('is_good') else 'Chưa đúng'}", (255, 255, 255)),
                        ]
                    else:
                        lines = [
                            (f"Bài tập: {exercise_choice.upper()}", (255, 105, 180)),
                            (f"Số lần: {state.get('counter', 0)}", (255, 215, 0)),
                            (f"Trạng thái: {stage}", (255, 255, 255)),
                        ]

                    lines.extend([
                        (f"Góc: {int(angle)}°", (144, 238, 144)),
                        (f"Đánh giá: {feedback}", form_color),
                        (f"FPS: {fps:.1f} (Xử lý)", (200, 200, 200)),
                    ])

                    annotated = draw_text_pil(annotated, lines, font_path=FONT_PATH, font_scale=26, pos=(20, 20))

                    # Resize annotated -> 1080p before display / writing
                    try:
                        annotated = cv2.resize(annotated, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)
                    except Exception:
                        pass

                    # Hiển thị trên Streamlit
                    frame_placeholder.image(
                        annotated,
                        channels="BGR",
                        caption=f"Frame {frame_idx}/{total_frames}",
                        use_column_width=True
                    )

                    frame_idx += 1
                    progress_bar.progress(frame_idx / total_frames)

                cap.release()
                st.success(f"Phân tích video hoàn tất. Tổng số lần: {state.get('counter', 0)}.")