import os
import gradio as gr
from ultralytics import YOLO
import cv2
import torch
import time
import threading
import numpy as np
from rep_counter import count_squat, count_pushup, count_plank, count_situp
from form_rules import evaluate_squat, evaluate_pushup, evaluate_plank, evaluate_situp
from utils.draw_utils import draw_text_pil

try:
    from moviepy.editor import ImageSequenceClip
    USE_MOVIEPY = True
except Exception:
    ImageSequenceClip = None
    USE_MOVIEPY = False

# Exercise registry
EXERCISE_REGISTRY = {
    "Squat": {
        "counter_func": count_squat,
        "form_func": evaluate_squat,
        "state": {"stage": "up", "counter": 0, "prev_angle": 160, "direction": "up"},
    },
    "Push-up": {
        "counter_func": count_pushup,
        "form_func": evaluate_pushup,
        "state": {"stage": "up", "counter": 0, "prev_angle": 160, "direction": "up"},
    },
    "Plank": {
        "counter_func": count_plank,
        "form_func": evaluate_plank,
        "state": {"good_time": 0, "bad_time": 0, "is_good": False},
    },
    "Sit-up": {
        "counter_func": count_situp,
        "form_func": evaluate_situp,
        "state": {"stage": "down", "counter": 0, "prev_angle": 140, "direction": "down"},
    }
}


def process_video(video_path, exercise_type, device=None, imgsz=480, conf=0.5, draw_every_n=1):
    """Process a video file and return list of annotated frames (RGB uint8 numpy arrays).
    This function is robust against per-frame model errors and will log why it stops.
    """
    print(f"Processing video: {video_path}")
    print(f"Exercise type: {exercise_type}")

    # Khởi tạo font path (robust)
    font_path = os.path.join(os.path.dirname(__file__), "..", "fonts", "Roboto.ttf")
    if not os.path.exists(font_path):
        font_path = "arial.ttf"

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Load model once
    model = YOLO("yolo11n-pose.pt")
    model.conf = conf
    try:
        model.to(device)
    except Exception:
        print("Warning: could not move model to device, continuing on default")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Không thể mở video: {video_path}")

    output_frames = []
    # mỗi exercise có state riêng ở registry; copy để tránh thay đổi toàn cục
    local_state = {k: v["state"].copy() for k, v in EXERCISE_REGISTRY.items()}

    frame_idx = 0
    last_print = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Frame không đọc được — video kết thúc hoặc nguồn mất tín hiệu.")
            break

        # ensure we have a 3-channel BGR frame
        if frame is None:
            print("⚠️ Frame đọc được là None, bỏ qua frame này.")
            continue

        # Optionally resize to speed up inference
        try:
            h, w = frame.shape[:2]
            if max(h, w) > imgsz:
                scale = imgsz / max(h, w)
                new_w = max(1, int(w * scale))
                new_h = max(1, int(h * scale))
                frame_in = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                frame_in = frame
        except Exception as e:
            print(f"Resize error: {e}")
            frame_in = frame

        annotated = None
        try:
            # perform prediction on the frame (YOLO supports passing ndarray)
            results = model.predict(frame_in, verbose=False, device=device, imgsz=imgsz)
            res = results[0]
            annotated = res.plot()
        except Exception as e:
            print(f"Model error on frame {frame_idx}: {e}")
            # Attempt to free GPU memory and continue
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            # fallback: use original frame for output
            annotated = frame_in

        # Process keypoints and overlay text if available
        try:
            counter = 0
            stage_or_good = None
            angle = None
            feedback = "..."

            if hasattr(res, 'keypoints') and res.keypoints is not None and len(res.keypoints.xy) > 0:
                kps = res.keypoints.xy[0].tolist()
                exercise = EXERCISE_REGISTRY[exercise_type]
                counter_func = exercise["counter_func"]
                form_func = exercise["form_func"]
                state = local_state[exercise_type]

                # gọi counter_func, unpack linh hoạt
                result = counter_func(kps, state)
                print(f"[DEBUG] counter_func returned: {result} for exercise {exercise_type} (frame {frame_idx})")
                if isinstance(result, (tuple, list)):
                    if len(result) == 3:
                        counter, stage_or_good, angle = result
                    elif len(result) == 2:
                        counter, stage_or_good = result
                        angle = None
                    else:
                        counter = result[0]
                else:
                    counter = result

                # gọi form_func với nhiều signature khả dĩ (robust)
                ret = None
                try:
                    ret = form_func(kps, annotated, stage_or_good, counter)
                except TypeError:
                    try:
                        ret = form_func(kps, annotated, counter)
                    except TypeError:
                        try:
                            ret = form_func(kps, counter)
                        except Exception as e:
                            print(f"Form eval error: {e}")
                            ret = None
                except Exception as e:
                    print(f"Form eval error: {e}")
                    ret = None

                form_score, feedback = None, ""
                if isinstance(ret, (tuple, list)):
                    if len(ret) >= 2:
                        form_score, feedback = ret[0], ret[1]
                    elif len(ret) == 1:
                        form_score = ret[0]
                elif isinstance(ret, (int, float, str)):
                    form_score = ret

            # build overlay lines
            if exercise_type == "Plank":
                elapsed = float(counter) if counter is not None else 0.0
                lines = [
                    (f"Thời gian giữ: {elapsed:.1f}s", (255, 215, 0)),
                    (f"Tư thế: {'Chuẩn' if stage_or_good else 'Chưa đúng'}", (255,255,255)),
                    (f"Góc: {int(angle) if angle is not None else '?'}°", (144,238,144)),
                    (f"Đánh giá: {feedback}", (0,255,0)),
                ]
            else:
                angle_text = f"{int(angle)}°" if angle is not None else "?"
                stage_text = stage_or_good if stage_or_good is not None else "?"
                lines = [
                    (f"Số lần: {counter}", (255,215,0)),
                    (f"Trạng thái: {stage_text}", (255,255,255)),
                    (f"Góc: {angle_text}", (144,238,144)),
                    (f"Đánh giá: {feedback}", (0,255,0)),
                ]

            try:
                if annotated is None:
                    annotated = frame_in
                annotated = draw_text_pil(annotated, lines, font_path)
            except Exception as e:
                print(f"Draw text error: {e}")

        except Exception as e:
            print(f"Error processing frame {frame_idx}: {str(e)}")
            if annotated is None:
                annotated = frame_in

        # Ensure RGB uint8
        try:
            arr = np.asarray(annotated)
            if arr.dtype != np.uint8:
                arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            if arr.ndim == 2:
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
            if arr.shape[2] == 4:
                arr = arr[:, :, :3]
        except Exception as e:
            print(f"Frame normalization error: {e}")
            arr = np.zeros((480, 640, 3), dtype=np.uint8)

        output_frames.append(arr)

        frame_idx += 1
        # periodic debug print so user knows progress
        if time.time() - last_print > 5:
            print(f"Processed {frame_idx} frames...")
            last_print = time.time()

        # avoid GPU hogging
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    cap.release()
    # free model memory
    try:
        del model
        torch.cuda.empty_cache()
    except Exception:
        pass

    return output_frames


# Tạo giao diện Gradio

def fitness_tracker(video, exercise_type):
    # Guard: no input
    if video is None:
        raise Exception("No video provided. Please upload a video file before processing.")

    # video can be a path (str) from Gradio or a dict with 'name' key
    if isinstance(video, dict) and 'name' in video and video['name']:
        temp_path = video['name']
    elif isinstance(video, str) and os.path.exists(video):
        temp_path = video
    else:
        # Gradio sometimes supplies a file-like object or bytes; write to temp
        temp_path = 'temp_input.mp4'
        try:
            with open(temp_path, 'wb') as f:
                if isinstance(video, (bytes, bytearray)):
                    f.write(video)
                elif hasattr(video, 'read') and callable(video.read):
                    data = video.read()
                    if data is None:
                        raise Exception('Uploaded file is empty')
                    f.write(data)
                else:
                    raise Exception('Unsupported video input type')
        except Exception as e:
            raise Exception(f"Could not save uploaded video: {e}")

    try:
        output_frames = process_video(temp_path, exercise_type)
        if not output_frames:
            raise Exception("No frames processed")

        # Normalize frames: ensure uint8 RGB numpy arrays
        TARGET_W, TARGET_H = 1920, 1080
        norm_frames = []
        for i, f in enumerate(output_frames):
            frame = np.asarray(f)
            if frame.dtype != np.uint8:
                frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            if frame.shape[2] == 4:
                frame = frame[:, :, :3]

            h, w = frame.shape[:2]
            if w == 0 or h == 0:
                continue

            scale = min(TARGET_W / w, TARGET_H / h)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            resized = cv2.resize(frame, (new_w, new_h), interpolation=interp)

            # pad to exact TARGET_W x TARGET_H (centered)
            top = (TARGET_H - new_h) // 2
            bottom = TARGET_H - new_h - top
            left = (TARGET_W - new_w) // 2
            right = TARGET_W - new_w - left
            padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            norm_frames.append(padded)

        output_path = "output.mp4"
        fps = 30

        if USE_MOVIEPY and ImageSequenceClip is not None:
            print("Using moviepy -> writing mp4 (libx264)...")
            clip = ImageSequenceClip([f[:, :, ::-1] for f in norm_frames], fps=fps)
            clip.write_videofile(
                output_path,
                codec='libx264',
                audio=False,
                verbose=False,
                threads=4,
                preset='ultrafast',
                ffmpeg_params=['-crf', '23']
            )
        else:
            print("Moviepy not available — fallback to OpenCV mp4v...")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (TARGET_W, TARGET_H))
            if not out.isOpened():
                raise Exception("Could not open VideoWriter with mp4v codec")
            for idx, fr in enumerate(norm_frames):
                # frames are RGB -> convert to BGR for OpenCV
                bgr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
                out.write(bgr)
                if idx % 50 == 0:
                    print(f"Writing frame {idx}/{len(norm_frames)}")
            out.release()

        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f"Output created: {output_path} ({os.path.getsize(output_path)} bytes)")
            return output_path
        else:
            raise Exception("Output video file is empty after writing")

    except Exception as e:
        print(f"Lỗi xử lý video: {str(e)}")
        return None

    finally:
        if os.path.exists('temp_input.mp4'):
            try:
                os.remove('temp_input.mp4')
            except Exception:
                pass


# --- Simple Gradio UI ---

def build_ui():
    with gr.Blocks(title="AI Fitness Tracker") as demo:
        gr.Markdown("# 🏋️‍♂️ AI Fitness Tracker")
        with gr.Row():
            video_input = gr.Video(label="Upload video (MP4) or paste path")
            exercise = gr.Dropdown(list(EXERCISE_REGISTRY.keys()), label="Exercise", value=list(EXERCISE_REGISTRY.keys())[0])
        run_btn = gr.Button("Start Processing")
        output_file = gr.File(label="Output video (downloadable)")

        def _run(video, exercise_type):
            print("User requested processing...")
            return fitness_tracker(video, exercise_type)

        run_btn.click(_run, inputs=[video_input, exercise], outputs=[output_file])

    return demo


if __name__ == "__main__":
    app = build_ui()
    # launch Gradio app in blocking mode so the script does not exit
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
