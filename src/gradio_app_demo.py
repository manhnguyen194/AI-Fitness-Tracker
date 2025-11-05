#!/usr/bin/env python3
"""
gradio_app_final4.py

Final revision:
- Supports YOLO11 pose weights (default: yolo11n-pose.pt).
- External OpenCV webcam window using model.predict(..., show=False) and results[0].plot() for skeleton rendering.
- Robust keypoint extraction supporting various YOLO versions (xy, xyn, data, cpu tensors).
- Overlay rep counter (bottom-left) + FPS and terminal log: [Webcam] FPS: .. | Detected: N person(s)
- Video upload processing split into 3 parts; fixes for `failed to write frame` (resize and dtype fix before write).
- English UI via Gradio; keeps video analyze + external webcam features.
"""
import os, time, threading, tempfile, traceback, math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import gradio as gr

# Torch + ultralytics detection
try:
    import torch
    torch.backends.cudnn.benchmark = True
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False

# rep counter (local file expected)
try:
    from rep_counter import count_squat, count_pushup, count_plank, count_situp
except Exception:
    def count_squat(kps, state): return 0, "up", 0.0
    def count_pushup(kps, state): return 0, "up", 0.0
    def count_plank(kps, state): return 0.0, "holding", 0.0
    def count_situp(kps, state): return 0, "down", 0.0

# ---------------- Config ----------------
MODEL_PATH_DEFAULT = "yolo11n-pose.pt"
CAP_DEVICE_INDEX = 0
IMG_SIZE = 224
DISPLAY_MAX_WIDTH = 1280
INFER_EVERY_N = 2

# Thread controls and caches
EXTERNAL_THREAD = None
EXTERNAL_STOP = threading.Event()
MODEL = None
MODEL_LOCK = threading.Lock()

# Background video processing tracker
BG_TASK = {"thread": None, "status": "idle", "part1_path": None, "final_path": None, "tmp_dir": None, "error": None}

# ---------------- Utilities ----------------

def ensure_bgr_uint8(img):
    """Ensure image is uint8 BGR numpy array for OpenCV display."""
    # If PIL image
    try:
        from PIL import Image
        if isinstance(img, Image.Image):
            arr = np.array(img)
        else:
            arr = img
    except Exception:
        arr = img
    if not isinstance(arr, np.ndarray):
        return None
    # if float image in [0,1]
    if arr.dtype == np.float32 or arr.dtype == np.float64:
        if arr.max() <= 1.01:
            arr = (arr * 255.0).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    # if 3-channel assume RGB -> convert to BGR (most ultralytics plot returns RGB)
    if arr.ndim == 3 and arr.shape[2] == 3:
        # Heuristic: if red channel mean > blue channel mean, likely RGB
        try:
            mean_r = float(arr[:,:,0].mean())
            mean_b = float(arr[:,:,2].mean())
            if mean_r > mean_b:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        except Exception:
            # fallback - try convert anyway inside try/except
            try:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            except Exception:
                pass
    return arr

def kps_to_pixel_coords(kps, frame_w, frame_h):
    """If kps are normalized (0-1), convert to pixel coords using frame width/height.
       Returns copy in float."""
    if kps is None:
        return None
    try:
        arr = np.array(kps, dtype=float).copy()
        if arr.size == 0:
            return None
        # if values are normalized (<=1.01), scale up
        if arr.max() <= 1.01:
            arr[:,0] = arr[:,0] * frame_w
            arr[:,1] = arr[:,1] * frame_h
        return arr
    except Exception:
        return None

def get_model(weights_path: Optional[str] = None):
    global MODEL
    with MODEL_LOCK:
        if MODEL is not None:
            return MODEL
        if not YOLO_AVAILABLE:
            raise RuntimeError("Ultralytics YOLO not installed. Please install ultralytics.")
        if weights_path is None:
            weights_path = MODEL_PATH_DEFAULT
        MODEL = YOLO(weights_path)
        try:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                MODEL.model.to("cuda")
        except Exception:
            pass
        # warm-up tiny image
        try:
            dummy = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            MODEL.predict(dummy, imgsz=IMG_SIZE, device="cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else None, verbose=False)
        except Exception:
            pass
        return MODEL

def safe_extract_kps(res):
    """
    Robust keypoint extractor supporting YOLOv8/v11 variations.
    Returns Nx2 numpy array (float) in image pixel coordinates or None.
    """
    try:
        if res is None or len(res) == 0:
            return None
        r0 = res[0]
        # r0.keypoints may be various types/attributes across versions
        kp_candidates = []
        if hasattr(r0, "keypoints") and r0.keypoints is not None:
            kp_obj = r0.keypoints
            # try common attributes
            for attr in ("xy", "xyn", "data", "pts", "xyxy"):
                val = getattr(kp_obj, attr, None)
                if val is not None:
                    kp_candidates.append(val)
        # also try direct properties sometimes present
        for attr in ("keypoints", "kps"):
            val = getattr(r0, attr, None)
            if val is not None:
                kp_candidates.append(val)

        for candidate in kp_candidates:
            try:
                # tensor with cpu()
                if hasattr(candidate, "cpu"):
                    arr = candidate.cpu().numpy()
                else:
                    arr = np.array(candidate)
                # arr may be (N_persons, K, 3) or (K,3) or (K,2)
                if arr.ndim == 3 and arr.shape[0] >= 1:
                    arr0 = arr[0]
                elif arr.ndim == 2:
                    arr0 = arr
                else:
                    continue
                if arr0.shape[1] >= 2:
                    return arr0[:, :2].astype(float)
            except Exception:
                continue
    except Exception as e:
        print("safe_extract_kps top-level error:", e)
    return None

def overlay_rep_bottom_left(img_bgr, exercise, kps, state, fps=None):
    overlay = img_bgr.copy()
    h, w = img_bgr.shape[:2]
    if kps is None or (isinstance(kps, np.ndarray) and kps.size == 0):
        label = f"{exercise} | No person detected"
        rect_w = min(420, w-20)
        rect_h = 50
        x0, y0 = 10, h - rect_h - 10
        cv2.rectangle(overlay, (x0,y0), (x0+rect_w, y0+rect_h), (60,60,60), -1)
        cv2.putText(overlay, label, (x0+10, y0+32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        if fps is not None:
            cv2.putText(overlay, f"FPS: {fps:.1f}", (w-140, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
        cv2.addWeighted(overlay, 0.75, img_bgr, 0.25, 0, img_bgr)
        return img_bgr

    # compute rep info based on exercise
    # ensure prev_angle is initialized to avoid TypeError in rep_counter
    try:
        import rep_counter as _rc
    except Exception:
        _rc = None
    # if prev_angle in state is None, set it to current joint angle before first call
    try:
        if exercise == "Squat":
            if _rc is not None and (state.get("prev_angle") is None):
                try:
                    left_hip, left_knee, left_ankle = kps[11], kps[13], kps[15]
                    state["prev_angle"] = _rc.calculate_angle(left_hip, left_knee, left_ankle)
                except Exception:
                    state.setdefault("prev_angle", 0.0)
        elif exercise == "Push-up":
            if _rc is not None and (state.get("prev_angle") is None):
                try:
                    l_shoulder, l_elbow, l_wrist = kps[5], kps[7], kps[9]
                    state["prev_angle"] = _rc.calculate_angle(l_shoulder, l_elbow, l_wrist)
                except Exception:
                    state.setdefault("prev_angle", 0.0)
        elif exercise == "Sit-up":
            if _rc is not None and (state.get("prev_angle") is None):
                try:
                    left_shoulder, left_hip, left_ankle = kps[5], kps[11], kps[15]
                    state["prev_angle"] = _rc.calculate_angle(left_shoulder, left_hip, left_ankle)
                except Exception:
                    state.setdefault("prev_angle", 0.0)
    except Exception:
        pass

    if exercise == "Squat":
        counter, direction, angle = count_squat(kps.tolist(), state)
    elif exercise == "Push-up":
        counter, direction, angle = count_pushup(kps.tolist(), state)
    elif exercise == "Sit-up":
        counter, direction, angle = count_situp(kps.tolist(), state)
    else:
        elapsed, label_state, angle = count_plank(kps.tolist(), state)
        counter = None
        direction = label_state

    is_good = state.get("is_good", True)
    bg_color = (0,140,0) if is_good else (0,80,160)
    rect_w = min(420, w-20); rect_h = 60
    x0, y0 = 10, h - rect_h - 10
    cv2.rectangle(overlay, (x0,y0), (x0+rect_w, y0+rect_h), bg_color, -1)
    if counter is not None:
        label = f"{exercise} | Reps: {counter} | Angle: {angle:.0f}° | {direction}"
    else:
        label = f"{exercise} | Elapsed: {state.get('elapsed', 0.0):.1f}s | {state.get('feedback','')} | Angle: {state.get('angle',0.0):.0f}°"
    cv2.putText(overlay, label, (x0+12, y0+38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    if fps is not None:
        cv2.putText(overlay, f"FPS: {fps:.1f}", (w-140, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
    cv2.addWeighted(overlay, 0.75, img_bgr, 0.25, 0, img_bgr)
    return img_bgr

# ---------------- External webcam loop ----------------
def external_webcam_loop(exercise="Squat", weights_path=None, log_every=10):
    global EXTERNAL_STOP, EXTERNAL_THREAD
    EXTERNAL_STOP.clear()
    try:
        model = get_model(weights_path)
    except Exception as e:
        print("Model load error:", e)
        return "model_error"

    cap = cv2.VideoCapture(CAP_DEVICE_INDEX, cv2.CAP_DSHOW) if os.name=='nt' else cv2.VideoCapture(CAP_DEVICE_INDEX)
    # try to set 1080p
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    except Exception:
        pass

    # read sample to determine actual resolution
    ret, sample = cap.read()
    if not ret:
        print("Cannot read from webcam")
        return "cam_error"
    actual_h, actual_w = sample.shape[:2]
    display_w = min(actual_w, DISPLAY_MAX_WIDTH)
    display_h = int(display_w * (actual_h / actual_w))

    window_name = "Webcam - Realtime Extractor"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, display_w, display_h)

    last_time = time.time()
    frame_idx = 0
    last_res = None
    rep_state = {"counter":0, "stage":None, "prev_angle":None, "direction":"up",
                 "start_time":None, "last_time":None, "good_time":0.0, "bad_time":0.0,
                 "is_good":True, "feedback":"", "elapsed":0.0, "angle":0.0}

    detected_count_log = 0
    while not EXTERNAL_STOP.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        frame_idx += 1

        do_infer = (frame_idx % INFER_EVERY_N) == 0
        if do_infer:
            try:
                kwargs = {"imgsz": IMG_SIZE, "verbose": False}
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    kwargs["device"] = "cuda"
                res = model.predict(frame, **kwargs)
                last_res = res
            except Exception as e:
                print("Predict error:", e)
                last_res = None

        annotated = frame.copy()
        detected = 0
        if last_res is not None:
            try:
                # Use YOLO's plotting to match extractor visuals
                plotted = last_res[0].plot()
                # ensure bgr uint8
                annotated_candidate = ensure_bgr_uint8(plotted)
                if isinstance(annotated_candidate, np.ndarray):
                    annotated = annotated_candidate.copy()
                # extract keypoints robustly
                kps = safe_extract_kps(last_res)
                if kps is not None and kps.size > 0:
                    # convert normalized coords to pixel coords if needed (frame is original size)
                    kps_pixels = kps_to_pixel_coords(kps, frame.shape[1], frame.shape[0])
                    detected = 1 if kps_pixels is not None and kps_pixels.shape[0] > 0 else 0
                    kps_scaled = kps_pixels
                else:
                    kps_scaled = None
                now = time.time()
                dt = now - last_time if last_time else 0.01
                fps = 1.0 / dt if dt > 0 else 0.0
                last_time = now
                annotated = overlay_rep_bottom_left(annotated, exercise, kps_scaled, rep_state, fps)
            except Exception as e:
                print("Annotation error:", e, traceback.format_exc())
                annotated = frame.copy()

        # show, scaling for display only (preserve aspect)
        display = annotated
        if display.shape[1] > DISPLAY_MAX_WIDTH:
            display = cv2.resize(display, (display_w, display_h), interpolation=cv2.INTER_AREA)
        cv2.imshow(window_name, display)

        # periodic terminal log
        if frame_idx % log_every == 0:
            print(f"[Webcam] FPS: {fps:.1f} | Detected: {detected} person(s)")

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            EXTERNAL_STOP.set()
            break
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            EXTERNAL_STOP.set()
            break

    cap.release()
    cv2.destroyWindow(window_name)
    EXTERNAL_THREAD = None
    return "stopped"

def start_external_webcam_thread(exercise="Squat", weights_path=None):
    global EXTERNAL_THREAD, EXTERNAL_STOP
    if EXTERNAL_THREAD is not None and EXTERNAL_THREAD.is_alive():
        return "Webcam already running."
    EXTERNAL_STOP.clear()
    EXTERNAL_THREAD = threading.Thread(target=external_webcam_loop, args=(exercise, weights_path), daemon=True)
    EXTERNAL_THREAD.start()
    return "Launched external webcam window."

def stop_external_webcam_thread():
    global EXTERNAL_THREAD, EXTERNAL_STOP
    EXTERNAL_STOP.set()
    if EXTERNAL_THREAD is not None:
        EXTERNAL_THREAD.join(timeout=2.0)
    return "Stopped external webcam window."

# ---------------- Video processing (split into 3 parts) ----------------
def process_video_split_parts(input_path, exercise="Squat", weights_path=None, output_resolution=(1920,1080)):
    try:
        model = get_model(weights_path)
    except Exception as e:
        print("Model load failed:", e)
        BG_TASK["status"] = "error"
        BG_TASK["error"] = str(e)
        return None, None, None

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print("Cannot open uploaded video:", input_path)
        BG_TASK["status"] = "error"
        BG_TASK["error"] = "cannot_open_video"
        return None, None, None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # compute parts
    if total_frames <= 0:
        parts = [(0, None)]
    else:
        per = max(1, total_frames // 3)
        parts = []
        start = 0
        for i in range(3):
            end = start + per - 1 if i < 2 else total_frames - 1
            parts.append((start, end))
            start = end + 1

    tmp_dir = tempfile.mkdtemp(prefix="video_parts_")
    part_paths = [os.path.join(tmp_dir, f"part_{i+1}.mp4") for i in range(len(parts))]
    final_path = os.path.join(tmp_dir, "final_annotated.mp4")

    out_w, out_h = output_resolution
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    def process_range(start_frame, end_frame, out_path):
        cap_local = cv2.VideoCapture(str(input_path))
        cap_local.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        writer = cv2.VideoWriter(out_path, fourcc, fps, (out_w, out_h))
        frame_idx = start_frame
        last_res = None
        last_time = time.time()
        rep_state = {"counter":0, "stage":None, "prev_angle":None, "direction":"up",
                     "start_time":None, "last_time":None, "good_time":0.0, "bad_time":0.0,
                     "is_good":True, "feedback":"", "elapsed":0.0, "angle":0.0}
        while True:
            if end_frame is not None and frame_idx > end_frame:
                break
            ret, frame = cap_local.read()
            if not ret:
                break
            # inference on raw frame (let model handle resize internally)
            if (frame_idx % INFER_EVERY_N) == 0:
                try:
                    kwargs = {"imgsz": IMG_SIZE, "verbose": False}
                    if TORCH_AVAILABLE and torch.cuda.is_available():
                        kwargs["device"] = "cuda"
                    res = model.predict(frame, **kwargs)
                    last_res = res
                except Exception as e:
                    last_res = None
            annotated = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
            if last_res is not None:
                try:
                    plotted = last_res[0].plot()
                    # ensure bgr uint8 and resize to out resolution
                    plotted_candidate = ensure_bgr_uint8(plotted)
                    if isinstance(plotted_candidate, np.ndarray):
                        plotted_resized = cv2.resize(plotted_candidate, (out_w, out_h), interpolation=cv2.INTER_AREA)
                        annotated = plotted_resized.copy()
                    kps = safe_extract_kps(last_res)
                    if kps is not None and kps.size > 0:
                        in_h, in_w = frame.shape[:2]
                        sx = out_w / in_w; sy = out_h / in_h
                        kps_scaled = np.zeros_like(kps)
                        kps_scaled[:,0] = kps[:,0] * sx
                        kps_scaled[:,1] = kps[:,1] * sy
                    else:
                        kps_scaled = None
                    now = time.time()
                    dt = now - last_time if last_time else 0.01
                    fps_est = 1.0 / dt if dt > 0 else 0.0
                    last_time = now
                    annotated = overlay_rep_bottom_left(annotated, exercise, kps_scaled, rep_state, fps_est)
                except Exception as e:
                    print("Error annotating frame in video processing:", e)
            # ensure correct dtype and size before write
            if annotated.shape[1] != out_w or annotated.shape[0] != out_h:
                annotated = cv2.resize(annotated, (out_w, out_h), interpolation=cv2.INTER_AREA)
            annotated = np.ascontiguousarray(annotated, dtype=np.uint8)
            try:
                writer.write(annotated)
            except Exception as e:
                print(f"Write frame failed at {frame_idx}: {e} shape={annotated.shape} dtype={annotated.dtype}")
            frame_idx += 1
        cap_local.release()
        writer.release()

    # process first part synchronously
    start0, end0 = parts[0]
    process_range(start0, end0, part_paths[0])
    BG_TASK["part1_path"] = part_paths[0]
    BG_TASK["tmp_dir"] = tmp_dir

    def bg_job():
        BG_TASK["status"] = "processing_rest"
        try:
            for i in range(1, len(parts)):
                s,e = parts[i]
                process_range(s, e, part_paths[i])
            # concatenate parts to final_path
            out = cv2.VideoWriter(final_path, fourcc, fps, (out_w, out_h))
            for p in part_paths:
                cap_p = cv2.VideoCapture(p)
                while True:
                    ret, frm = cap_p.read()
                    if not ret:
                        break
                    # ensure dtype/size
                    if frm.shape[1] != out_w or frm.shape[0] != out_h:
                        frm = cv2.resize(frm, (out_w, out_h), interpolation=cv2.INTER_AREA)
                    frm = np.ascontiguousarray(frm, dtype=np.uint8)
                    out.write(frm)
                cap_p.release()
            out.release()
            BG_TASK["final_path"] = final_path
            BG_TASK["status"] = "done"
        except Exception as e:
            BG_TASK["status"] = "error"
            BG_TASK["error"] = str(e)
            traceback.print_exc()

    bg_thread = threading.Thread(target=bg_job, daemon=True)
    bg_thread.start()
    BG_TASK["thread"] = bg_thread
    BG_TASK["status"] = "part1_ready"
    return part_paths[0], tmp_dir, bg_thread

# ---------------- Gradio callbacks ----------------
def analyze_video_click(uploaded_file, exercise, weights, resolution):
    if uploaded_file is None:
        return None, "No file uploaded."
    weights = weights or MODEL_PATH_DEFAULT
    out_res = (1920,1080)
    if isinstance(resolution, str):
        if 'x' in resolution:
            try:
                w,h = resolution.split('x')
                out_res = (int(w), int(h))
            except Exception:
                out_res = (1920,1080)
        elif resolution.endswith('p'):
            try:
                h = int(resolution[:-1])
                w = int(h * 16 / 9)
                out_res = (w,h)
            except Exception:
                out_res = (1920,1080)
    BG_TASK["status"] = "starting"
    part1, tmpd, thr = process_video_split_parts(uploaded_file.name, exercise, weights, output_resolution=out_res)
    if part1 is None:
        BG_TASK["status"] = "error"
        return None, "Processing failed (model/load error)."
    return part1, f"Part 1 ready. Processing remaining parts in background (tmp: {tmpd}). Use 'View Remaining' to check when done."

def view_remaining_click():
    st = BG_TASK.get("status", "idle")
    if st == "done" and BG_TASK.get("final_path"):
        return BG_TASK["final_path"], "Final video ready."
    elif st in ("processing_rest","part1_ready","starting"):
        return None, f"Still processing (status: {st}). Please wait or check later."
    elif st == "error":
        return None, f"Processing error: {BG_TASK.get('error','unknown')}"
    else:
        return None, "No processing in progress."

# ---------------- Gradio UI ----------------
EXERCISES = ["Squat", "Push-up", "Sit-up", "Plank"]
DEFAULT_RES_OPTIONS = ["1920x1080", "1280x720", "854x480", "1080p"]

def build_ui():
    with gr.Blocks(title="AI Fitness Tracker - Final (YOLO11 pose)") as demo:
        gr.Markdown("## AI Fitness Tracker\nExternal webcam window for realtime pose; upload video to analyze in 3 parts (fast preview).")
        with gr.Row():
            exercise = gr.Dropdown(EXERCISES, value=EXERCISES[0], label="Select Exercise (choose before Start)")
            start_btn = gr.Button("Start External Webcam Window")
            stop_btn = gr.Button("Stop External Webcam Window")
        with gr.Row():
            weights_input = gr.Textbox(value=MODEL_PATH_DEFAULT, label="Model weights path (local file)") 
            status = gr.Textbox(label="Status", value="Ready", interactive=False)
        start_btn.click(fn=lambda ex, w: start_external_webcam_thread(ex, w), inputs=[exercise, weights_input], outputs=[status])
        stop_btn.click(fn=lambda: stop_external_webcam_thread(), outputs=[status])

        gr.Markdown("### Analyze uploaded video (split into 3 parts, first part returned immediately)")
        with gr.Row():
            upload = gr.File(label="Upload video file (.mp4, .mov)")
            res_choice = gr.Dropdown(DEFAULT_RES_OPTIONS, value="1920x1080", label="Output resolution")
            analyze_btn = gr.Button("Analyze Video")
            view_btn = gr.Button("View Remaining (final)")
        out_video = gr.Video(label="Annotated output (first part or final)")
        message = gr.Textbox(label="Message", value="", interactive=False)

        analyze_btn.click(fn=analyze_video_click, inputs=[upload, exercise, weights_input, res_choice], outputs=[out_video, message])
        view_btn.click(fn=view_remaining_click, inputs=None, outputs=[out_video, message])

        gr.Markdown("Notes:\n- External webcam window uses your machine's webcam and shows realtime overlays.\n- Close that window or press 'q' to stop.\n- This app must be run on the machine with the webcam.\n")
    return demo

if __name__ == "__main__":
    app = build_ui()
    app.launch(share=False, server_name="localhost", server_port=7860)
