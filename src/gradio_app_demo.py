#!/usr/bin/env python3
"""
gradio_app_demo.py
Version: Fix Video Playback (Switched to .webm/VP8 for Browser Support)
"""

import os
import gradio as gr
import cv2
import time
import threading
import numpy as np
import torch
from PIL import Image, ImageFont, ImageDraw

# [QUAN TRỌNG] Import hàm vẽ chuẩn từ utils
from utils.draw_utils import draw_text_pil

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FONT_NAME = "Roboto.ttf"
FONT_PATH = os.path.join(BASE_DIR, "fonts", FONT_NAME)


# ==========================================
# 🖥️ LOG DEVICE INFO
# ==========================================
def print_startup_log():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\n{'=' * 40}")
    print(f"🚀 AI Fitness Gradio App Initialized")
    print(f"{'=' * 40}")
    print(f"► Device    : {device.upper()}")
    if os.path.exists(FONT_PATH):
        print(f"► Font      : Loaded '{FONT_NAME}'")
    else:
        print(f"► Warning   : Font '{FONT_NAME}' NOT FOUND.")
    print(f"{'=' * 40}\n")


print_startup_log()


# === IMPORT AND MOCK VOICE PLAYER ===
class MockVoicePlayer:
    def init(self, *args, **kwargs): pass

    def set_tone(self, tone): pass

    def play_feedback(self, *args, **kwargs): pass

    def stop(self): print("[VoicePlayer] Stopping audio queue...")


try:
    from pose_extractor import FitnessTracker
    import voice_player

    if not hasattr(voice_player, 'stop'):
        voice_player.stop = MockVoicePlayer().stop
except ImportError as e:
    raise ImportError("Could not find 'pose_extractor.py'. Please ensure the file is present.")

# === GLOBAL VARIABLES ===
WEBCAM_THREAD = None
WEBCAM_STOP_EVENT = threading.Event()
CAP_DEVICE_INDEX = 0


def stop_external_webcam_logic():
    global WEBCAM_THREAD, WEBCAM_STOP_EVENT
    if WEBCAM_THREAD is not None and WEBCAM_THREAD.is_alive():
        WEBCAM_STOP_EVENT.set()
        WEBCAM_THREAD.join(timeout=2.0)
        WEBCAM_THREAD = None
    voice_player.stop()
    return "⏹️ Webcam Stopped."


def auto_stop_webcam():
    stop_external_webcam_logic()
    return "🔴 Auto-stopped (Tab Switch)"


# === WEBCAM LOGIC ===
def webcam_thread_target(exercise_name, weights_path):
    global WEBCAM_STOP_EVENT
    try:
        tracker = FitnessTracker(
            exercise=exercise_name.lower().replace("-", ""),
            data_dir="data",
            fonts_dir="fonts",
            img_size=640,
            conf=0.5,
            inference_every=3,
            draw_every=3
        )
    except Exception as e:
        print(f"Error initializing tracker: {e}")
        return

    cap = cv2.VideoCapture(CAP_DEVICE_INDEX, cv2.CAP_DSHOW) if os.name == 'nt' else cv2.VideoCapture(CAP_DEVICE_INDEX)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    window_name = "AI Fitness Coach - Live Tracking (Press 'q' to quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    print(f"--- Webcam Started for {exercise_name} ---")

    while not WEBCAM_STOP_EVENT.is_set():
        ret, frame = cap.read()
        if not ret: break

        try:
            annotated = tracker.process_frame(frame)
            h_img = annotated.shape[0]
            estimated_size = max(14, int(26 * (h_img / 720)))
            pos_y = h_img - int(estimated_size * 2.5)
            annotated = draw_text_pil(
                annotated,
                [(f"Mode: {exercise_name}", (0, 255, 127))],
                font_path=FONT_PATH,
                font_scale=26,
                pos=(20, pos_y),
                wrap_text=False
            )
        except Exception as e:
            annotated = frame

        if annotated is not None:
            cv2.imshow(window_name, annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27: break
        try:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1: break
        except:
            break

    cap.release()
    cv2.destroyAllWindows()
    voice_player.stop()
    print("--- Webcam Thread Finished ---")


def start_webcam_btn(exercise_name, weights_path):
    global WEBCAM_THREAD, WEBCAM_STOP_EVENT
    if WEBCAM_THREAD is not None and WEBCAM_THREAD.is_alive():
        return "⚠️ Webcam is already running in an external window."
    WEBCAM_STOP_EVENT.clear()
    WEBCAM_THREAD = threading.Thread(
        target=webcam_thread_target,
        args=(exercise_name, weights_path),
        daemon=True
    )
    WEBCAM_THREAD.start()
    return "🚀 Webcam running in external window! Check your taskbar."


# === VIDEO STREAM FUNCTION (FIXED CODEC & SAVE) ===
def stream_video_analysis(video_file, exercise_name, weights_path, output_res="1280x720"):
    stop_external_webcam_logic()

    if video_file is None:
        yield None, "⚠️ Please upload a video first.", None
        return

    video_path = video_file.name if hasattr(video_file, 'name') else video_file

    # 1. Resolution Setup
    try:
        if 'x' in output_res:
            w_out, h_out = map(int, output_res.split('x'))
        else:
            w_out, h_out = 1280, 720
    except:
        w_out, h_out = 1280, 720

    # 2. Tracker Setup
    try:
        tracker = FitnessTracker(
            exercise=exercise_name.lower().replace("-", ""),
            img_size=640,
            conf=0.5,
            inference_every=3,
            fonts_dir="fonts"
        )
    except Exception as e:
        yield None, f"Model Error: {e}", None
        return

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0: fps = 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 3. Video Writer Setup (QUAN TRỌNG: Chuyển sang .webm / VP8)
    # Tạo tên file duy nhất để tránh cache trình duyệt
    output_filename = f"output_{int(time.time())}.webm"
    output_path = os.path.abspath(output_filename)

    # Codec 'vp80' cho định dạng .webm -> Trình duyệt hỗ trợ 100%
    fourcc = cv2.VideoWriter_fourcc(*'vp80')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w_out, h_out))

    frame_count = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # Tính timestamp chính xác cho video
            current_video_timestamp = frame_count / fps

            # Xử lý frame
            annotated_sq = tracker.process_frame(frame, timestamp=current_video_timestamp)

            if annotated_sq is not None:
                # Vẽ Text
                sq_h = annotated_sq.shape[0]
                estimated_size = max(14, int(26 * (sq_h / 720)))
                pos_y = sq_h - int(estimated_size * 2.5)
                annotated_sq = draw_text_pil(
                    annotated_sq,
                    [(f"{exercise_name}", (255, 215, 0))],
                    font_path=FONT_PATH,
                    font_scale=26,
                    pos=(30, pos_y),
                    wrap_text=False
                )

                # Resize và căn giữa (Padding)
                sq_w = annotated_sq.shape[1]
                scale = min(w_out / sq_w, h_out / sq_h)
                nw, nh = int(sq_w * scale), int(sq_h * scale)
                resized = cv2.resize(annotated_sq, (nw, nh))

                canvas = np.zeros((h_out, w_out, 3), dtype=np.uint8)
                x_off, y_off = (w_out - nw) // 2, (h_out - nh) // 2
                canvas[y_off:y_off + nh, x_off:x_off + nw] = resized
                final_frame = canvas
            else:
                final_frame = np.zeros((h_out, w_out, 3), dtype=np.uint8)

            # Ghi vào file (BGR format)
            writer.write(final_frame)

            # Convert sang RGB để hiển thị Live Preview
            rgb_frame = cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB)
            frame_count += 1

            pct = int((frame_count / total_frames) * 100) if total_frames > 0 else 0
            msg = f"▶️ Processing: {pct}% (Frame {frame_count}/{total_frames})"

            if frame_count % 3 == 0:  # Giảm tần suất yield để UI mượt hơn
                yield rgb_frame, msg, None

    except Exception as e:
        print(f"Stream Error: {e}")
        yield None, "❌ Error during processing", None

    finally:
        cap.release()
        writer.release()  # Đóng file hoàn tất
        voice_player.stop()

        if frame_count > 0:
            # Kiểm tra file có tồn tại không trước khi trả về
            if os.path.exists(output_path):
                yield rgb_frame, "✅ Complete! Video ready below.", output_path
            else:
                yield rgb_frame, "⚠️ Error: File not saved.", None
        else:
            yield None, "⚠️ No frames processed.", None


# === UI BUILDER ===
theme = gr.themes.Soft(
    primary_hue="emerald",
    neutral_hue="slate",
).set(
    body_background_fill="*neutral_50",
    block_title_text_weight="600"
)

css = """
h1 { text-align: center; color: #10b981; font-weight: 800; margin-bottom: 0; }
.sub-title { text-align: center; color: #64748b; margin-top: 5px; }
"""


def build_app():
    EXERCISES = ["Squat", "Push-up", "Plank", "Sit-up"]

    with gr.Blocks(theme=theme, css=css, title="AI Fitness Coach") as demo:
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("# 🏋️‍♂️ AI Fitness Coach")
                gr.Markdown("<p class='sub-title'>Real-time Form Correction & Rep Counting</p>")

        with gr.Row(variant="panel"):
            ex_drop = gr.Dropdown(EXERCISES, value="Squat", label="Choose Exercise", interactive=True, scale=2)
            w_txt = gr.Textbox("yolo11n-pose.pt", visible=False)

        with gr.Tabs():
            # TAB 1: WEBCAM
            with gr.TabItem("📷 Webcam (External Window)"):
                with gr.Row():
                    with gr.Column(scale=1, variant="panel"):
                        gr.Markdown("### ⚙️ Controls")
                        with gr.Group():
                            b_start_gr = gr.Button("▶️ Start Camera", variant="primary", size="lg")
                            b_stop_gr = gr.Button("⏹️ Stop Camera", variant="stop", size="lg")
                        st_cam_gr = gr.Textbox(label="System Status", value="Ready", interactive=False)

                    with gr.Column(scale=3):
                        gr.Markdown("### 🖥️ External Display Mode\nVideo opens in a separate window for high FPS.")
                        dummy_img = np.zeros((400, 600, 3), dtype=np.uint8) + 240
                        stream_out_cam = gr.Image(value=dummy_img, label="Window Status", interactive=False,
                                                  show_label=False, height=400)

                b_start_gr.click(start_webcam_btn, [ex_drop, w_txt], [st_cam_gr])
                b_stop_gr.click(stop_external_webcam_logic, None, [st_cam_gr])

            # TAB 2: VIDEO
            with gr.TabItem("📁 Video Analysis"):
                with gr.Row():
                    with gr.Column(scale=1, variant="panel"):
                        gr.Markdown("### 📤 Upload")
                        v_in = gr.File(label="Video File", file_count="single", file_types=[".mp4", ".mov", ".avi"])
                        btn_analyze = gr.Button("✨ Analyze Video", variant="primary", size="lg")
                        st_box = gr.Textbox(label="Analysis Status", value="Waiting for file...", interactive=False)
                        res_vid = gr.Dropdown(["1280x720", "640x640"], value="1280x720", label="Output Size")

                    with gr.Column(scale=3):
                        stream_out_vid = gr.Image(label="Live Preview", interactive=False, show_label=False, height=400)

                        # [OUTPUT VIDEO DOWNLOAD]
                        gr.Markdown("### 🎬 Final Result (Downloadable)")
                        # format='mp4' để Gradio biết cách hiển thị player, dù file là .webm
                        final_vid_out = gr.Video(label="Processed Video", interactive=False)

                # Cập nhật outputs cho nút Analyze
                btn_analyze.click(
                    stream_video_analysis,
                    inputs=[v_in, ex_drop, w_txt, res_vid],
                    outputs=[stream_out_vid, st_box, final_vid_out]
                )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860)