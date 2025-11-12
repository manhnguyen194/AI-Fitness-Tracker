#!/usr/bin/env python3
"""
gradio_app_demo.py (ĐÃ SỬA LỖI HOÀN CHỈNH)

- FIX 1 (Lỗi kéo giãn): Sửa lỗi gõ nhầm 'out_h, out_h' thành 'out_h, out_w'
  trong hàm process_range (dòng 535).
- FIX 7 (Lỗi Webcam): Sửa lỗi 'UnboundLocalError' bằng cách
  đổi 'frame_count' thành 'frame_idx' (dòng 373-380).
- FIX 8 (Lỗi Video): Sửa lỗi 'KeyError' bằng cách gán kết quả
  vào 'local_state[exercise]' thay vì 'local_state' (dòng 515).
- FIX 9 (Yêu cầu): ĐÃ BẬT LẠI (un-comment) dòng "Đánh giá: {feedback}"
  trong _process_frame_logic (dòng 310 và 318).
- FIX 10 (Tương thích): Sửa lỗi gọi 'form_func' để khớp với
  file form_rules.py mới (truyền 5 tham số, bao gồm 'state').
"""

import os
import gradio as gr
from ultralytics import YOLO
import cv2
import torch
import time
import threading
import numpy as np
import tempfile
import traceback
from pathlib import Path
from typing import Optional, Callable
import shutil
import atexit

# === Import logic mới ===
from rep_counter import count_squat, count_pushup, count_plank, count_situp
# 🛠️ SỬA: Import các hàm đánh giá cụ thể
from form_rules import evaluate_squat, evaluate_pushup, evaluate_plank, evaluate_situp
from utils.draw_utils import draw_text_pil
from utils.video_utils import compute_fps

# === Các biến toàn cục cho Webcam Ngoài (từ code cũ) ===
EXTERNAL_THREAD = None
EXTERNAL_STOP = threading.Event()
# (MODEL và MODEL_LOCK sẽ dùng chung với logic mới)

# === Các biến toàn cục cho Video 3 phần (từ code cũ) ===
BG_TASK = {"thread": None, "status": "idle", "part1_path": None, "final_path": None, "tmp_dir": None, "error": None}

# === Cấu hình chung (từ code mới) ===
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("=== 🔍 GPU/CUDA Status ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"Device memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("=====================")
    # Optimize CUDA backends for real-time
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else:
    device = torch.device("cpu")
    print("⚠️ CUDA not available. Running on CPU.")

try:
    import importlib
    import importlib.util
    spec = importlib.util.find_spec("moviepy.editor")
    if spec is not None:
        moviepy = importlib.import_module("moviepy.editor")
        ImageSequenceClip = getattr(moviepy, "ImageSequenceClip", None)
        USE_MOVIEPY = ImageSequenceClip is not None
    else:
        ImageSequenceClip = None
        USE_MOVIEPY = False
except Exception:
    ImageSequenceClip = None
    USE_MOVIEPY = False

# Global model and stream state for webcam processing
GLOBAL_MODEL = None
MODEL_LOCK = threading.Lock() # Thêm lock từ logic mới
GLOBAL_CONF = 0.5

# === CẬP NHẬT: Đồng bộ từ pose_extractor.py ===
GLOBAL_IMG_SIZE = 640  # Kích thước input cho model (thay vì 384)
INFER_EVERY_N = 3 # Đồng bộ với DRAW_EVERY_N_FRAMES (thay vì 2)
# === KẾT THÚC CẬP NHẬT ===

GLOBAL_USE_HALF = bool(torch.cuda.is_available())  # dùng FP16 nếu có CUDA
DISPLAY_MAX_WIDTH = 1280 # Cho cửa sổ webcam ngoài
CAP_DEVICE_INDEX = 0

# Đường dẫn font (từ logic mới)
font_path = Path(__file__).parent.parent / "fonts" / "Roboto.ttf"
if not font_path.exists():
    # Fallback nếu không tìm thấy font
    try:
        # Thử arial trên Windows
        font_path = "arial.ttf"
        from PIL import ImageFont
        ImageFont.truetype(font_path, 10)
    except Exception:
        # Fallback cuối cùng
        print("⚠️ Không tìm thấy font 'Roboto.ttf' hoặc 'arial.ttf'. Overlay văn bản có thể bị lỗi.")
        font_path = None


def get_model(weights_path: Optional[str] = "yolo11n-pose.pt"):
    """
    Hợp nhất từ _get_model (mới) và get_model (cũ).
    Lazily initialize, dùng lock, hỗ trợ custom weights, và warm-up.
    """
    global GLOBAL_MODEL, MODEL_LOCK
    with MODEL_LOCK:
        if GLOBAL_MODEL is None:
            try:
                m = YOLO(weights_path)
            except Exception as e:
                print(f"Error loading {weights_path}: {e}. Fallback to default.")
                m = YOLO("yolo11n-pose.pt")
            m.conf = GLOBAL_CONF
            try:
                m.to(device)
            except Exception:
                pass
            # Fuse Conv+BN để tăng tốc nếu được hỗ trợ
            try:
                m.fuse()
            except Exception:
                pass
            # Warm-up
            try:
                import numpy as _np
                dummy = _np.zeros((GLOBAL_IMG_SIZE, GLOBAL_IMG_SIZE, 3), dtype=_np.uint8)
                with torch.inference_mode():
                    _ = m.predict(dummy, verbose=False, device=str(device), imgsz=GLOBAL_IMG_SIZE, half=GLOBAL_USE_HALF, max_det=1, conf=GLOBAL_CONF)
            except Exception:
                pass
            GLOBAL_MODEL = m
        return GLOBAL_MODEL

# Lấy hàm trích xuất KPS tốt nhất (từ code local_webcam trong file mới)
def safe_extract_kps(res):
    """
    Trích xuất keypoints (robust) từ kết quả YOLO.
    (Đổi tên từ safe_get_kps_from_res_local)
    """
    try:
        r0 = res[0]
        if hasattr(r0, "keypoints") and r0.keypoints is not None:
            # 🛠️ SỬA (FIX 4): Logic này được cập nhật để chỉ lấy 'xy'
            # và người có conf cao nhất, khớp với logic mong muốn.
            if not res or not res.keypoints or not res.boxes:
                return None
            
            confs = res.boxes.conf.cpu().numpy() if res.boxes.conf is not None else []
            if len(confs) == 0:
                return None
            
            max_idx = np.argmax(confs)
            if confs[max_idx] < 0.6:  # Threshold để bỏ low-conf
                return None

            kps_tensor = getattr(res.keypoints, "xy", None)
            if kps_tensor is None:
                return None
                
            kps_list = kps_tensor.cpu().numpy()
            if max_idx >= len(kps_list):
                return None # Index out of bounds
                
            kps = kps_list[max_idx] # Lấy KPS của người có conf cao nhất
            
            if kps.ndim == 2 and kps.shape[0] > 0 and kps.shape[1] >= 2:
                 return kps[:, :2]  # Trả array cho dễ dùng
    except Exception:
        pass
    return None

# Registry bài tập (từ code mới)
EXERCISE_REGISTRY = {
    "Squat": {
        "counter_func": count_squat,
        "form_func": evaluate_squat, # 🛠️ SỬA: Dùng hàm evaluate_squat trực tiếp
        "state": {"stage": "up", "counter": 0, "prev_angle": 160, "direction": "up"},
    },
    "Push-up": {
        "counter_func": count_pushup,
        "form_func": evaluate_pushup, # 🛠️ SỬA: Dùng hàm evaluate_pushup trực tiếp
        "state": {"stage": "up", "counter": 0, "prev_angle": 160, "direction": "up"},
    },
    "Plank": {
        "counter_func": count_plank,
        "form_func": evaluate_plank, # 🛠️ SỬA: Dùng hàm evaluate_plank trực tiếp
        "state": {"good_time": 0, "bad_time": 0, "is_good": False, "start_time": None, "elapsed": 0.0},
    },
    "Sit-up": {
        "counter_func": count_situp,
        "form_func": evaluate_situp, # 🛠️ SỬA: Dùng hàm evaluate_situp trực tiếp
        "state": {"stage": "down", "counter": 0, "prev_angle": 140, "direction": "down"},
    }
}

# Thêm hàm reset_state
def reset_state(exercise_type, state_dict):
    if exercise_type in EXERCISE_REGISTRY:
        state_dict[exercise_type] = EXERCISE_REGISTRY[exercise_type]["state"].copy()
    return state_dict

# --- Logic Webcam Ngoài (Lấy từ code cũ, cập nhật logic) ---

def _process_frame_logic(frame_bgr, exercise_type, state_dict, prev_time):
    """
    Hàm logic lõi, dùng chung cho webcam và video.
    Xử lý 1 frame, trả về (annotated_frame, kps, state, fps, prev_time)
    """
    global GLOBAL_MODEL
    model = get_model() # Lấy model đã khởi tạo

    # 1. Chuẩn bị frame cho inference
    # 🛠️ SỬA (FIX 4): Ép (squash) frame về 640x640, giống hệt pose_extractor.py
    frame_in = cv2.resize(frame_bgr, (GLOBAL_IMG_SIZE, GLOBAL_IMG_SIZE), interpolation=cv2.INTER_AREA)

    # 2. Chạy inference
    try:
        with torch.inference_mode():
            results = model.predict(
                frame_in, # model nhận BGR
                verbose=False,
                device=str(device),
                imgsz=GLOBAL_IMG_SIZE,
                half=GLOBAL_USE_HALF,
                max_det=1, # Giới hạn 1 người
                conf=GLOBAL_CONF,
            )
        res = results[0]
        annotated = res.plot() # Trả về BGR (đã là 640x640)
    except Exception as e:
        print(f"Lỗi model predict: {e}")
        res = None
        annotated = frame_in.copy() # Dùng frame_in (640x640) nếu lỗi

    # 3. Trích xuất KPS
    # 🛠️ SỬA (FIX 4): Lấy KPS trực tiếp, không cần scale
    kps_scaled = None
    if res:
        kps_scaled_candidate = safe_extract_kps(res) # KPS đã ở tọa độ 640x640
        if kps_scaled_candidate is not None and kps_scaled_candidate.size > 0:
            kps_scaled = kps_scaled_candidate
    
    # 4. Tính toán (Counter & Form)
    # === SỬA LỖI (FIX 2): Khởi tạo biến TRƯỚC khối if/else ===
    current_state = state_dict[exercise_type]
    counter = current_state.get('counter', 0)
    angle = current_state.get('prev_angle', 180.0) # Dùng prev_angle hoặc angle từ state
    if exercise_type == "Plank":
        stage_or_good = current_state.get('is_good', False)
        counter = current_state.get('elapsed', 0.0) # counter là thời gian cho plank
    else:
        stage_or_good = current_state.get('stage', 'up') # stage cho các bài khác
    
    feedback = "..."
    form_color = (0, 255, 0) # Mặc định là 'tốt' (BGR)
    # === KẾT THÚC SỬA LỖI ===

    if kps_scaled is not None:
        try:
            exercise = EXERCISE_REGISTRY[exercise_type]
            counter_func = exercise["counter_func"]
            form_func = exercise["form_func"]
            state = current_state # Dùng state đã lấy ở trên

            # Hàm counter_func sẽ cập nhật 'state' và trả về giá trị MỚI
            result = counter_func(kps_scaled.tolist(), state) 
            
            # === FIX 3: Thêm khối giải nén (unpack) kết quả ===
            if isinstance(result, (tuple, list)):
                if len(result) == 3:
                    counter, stage_or_good, angle = result
                elif len(result) == 2:
                    counter, stage_or_good = result
                    angle = None # Hoặc gán giá trị mặc định
                else:
                    counter = result[0]
            else:
                counter = result
            # === KẾT THÚC FIX 3 ===

            # Gọi form_func (robust)
            ret = None
            try: 
                # 🛠️ SỬA (FIX 10): Truyền 'state' vào form_func (để khớp với form_rules.py ĐÃ SỬA)
                ret = form_func(kps_scaled.tolist(), state, annotated, stage_or_good, counter)
            except Exception as e_form:
                # In lỗi TypeError nếu form_rules.py CHƯA được cập nhật
                print(f"Lỗi khi gọi form_func (Kiểm tra form_rules.py): {e_form}")
                ret = None

            if ret is not None and isinstance(ret, tuple) and len(ret) >= 3:
                form_score, feedback, tone = ret
                if tone == "bad": form_color = (0, 0, 255) # Xấu (BGR)
                elif tone == "warn": form_color = (0, 165, 255) # Cam (BGR)
            elif ret is not None:
                feedback = str(ret)
                
        except Exception as e:
            print(f"Lỗi counter/form: {e}")
            traceback.print_exc()
            feedback = "Lỗi xử lý"

    # 5. Tính FPS
    fps, prev_time = compute_fps(prev_time)

    # 6. Overlay text với tách plank/reps
    if font_path is not None:
        if exercise_type == "Plank":
            lines = [
                (f"Thời gian giữ: {counter:.1f}s", (255, 215, 0)),
                (f"Tư thế: {'Chuẩn' if stage_or_good else 'Chưa đúng'}", (255, 255, 255)),
                (f"Góc: {int(angle or 0)}°", (144, 238, 144)),
                (f"Đánh giá: {feedback}", form_color), # 🛠️ SỬA (FIX 9): Bật lại dòng này
                (f"FPS: {fps:.1f}", (200, 200, 200)),
            ]
        else:
            lines = [
                (f"Số lần: {counter}", (255, 215, 0)),
                (f"Trạng thái: {stage_or_good}", (255, 255, 255)),
                (f"Góc: {int(angle or 0)}°", (144, 238, 144)),
                (f"Đánh giá: {feedback}", form_color), # 🛠️ SỬA (FIX 9): Bật lại dòng này
                (f"FPS: {fps:.1f}", (200, 200, 200)),
            ]
        annotated = draw_text_pil(annotated, lines, font_path=str(font_path), font_scale=26, pos=(20, 20))

    return annotated, kps_scaled, current_state, fps, prev_time

# --- External Webcam Thread (từ code cũ) ---

def external_webcam_loop(exercise, weights):
    global EXTERNAL_STOP, EXTERNAL_THREAD
    EXTERNAL_STOP.clear()
    
    try:
        model = get_model(weights) # Warm-up
    except Exception as e:
        print(f"Lỗi tải model: {e}")
        return "model_error"

    cap = cv2.VideoCapture(CAP_DEVICE_INDEX, cv2.CAP_DSHOW) if os.name=='nt' else cv2.VideoCapture(CAP_DEVICE_INDEX)
    if not cap.isOpened():
        print(f"Lỗi mở webcam index {CAP_DEVICE_INDEX}.")
        return "Lỗi mở webcam."

    # Lấy độ phân giải thực tế
    ret, sample = cap.read()
    if not ret:
        print("Không thể đọc frame từ webcam")
        cap.release()
        return "cam_error"
    
    actual_h, actual_w = sample.shape[:2]
    display_w = min(actual_w, DISPLAY_MAX_WIDTH)
    display_h = int(display_w * (actual_h / actual_w))
    
    window_name = "AI Fitness Tracker - Webcam (Nhan 'q' de thoat)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, display_w, display_h)

    prev_time = time.time()
    
    # 🛠️ SỬA (FIX 7): Khởi tạo frame_idx (đổi từ frame_count)
    frame_idx = 0 
    
    last_annotated = None
    state_dict = {k: v["state"].copy() for k, v in EXERCISE_REGISTRY.items()}
    state_dict = reset_state(exercise, state_dict)  # Reset state

    while not EXTERNAL_STOP.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        # Đồng bộ với INFER_EVERY_N
        if (frame_idx % INFER_EVERY_N) == 0:
            annotated, kps, state_dict[exercise], fps, prev_time = _process_frame_logic(frame, exercise, state_dict, prev_time)
            last_annotated = annotated
            
            # 🛠️ SỬA (FIX 7): Thay 'frame_count' thành 'frame_idx'
            if (frame_idx % 30) == 0: # Log mỗi 30 frames
                detected_count = 1 if kps is not None else 0
                print(f"[Webcam] FPS: {fps:.1f} | Detected: {detected_count} | Exercise: {exercise}")

        else:
            annotated = last_annotated if last_annotated is not None else frame.copy()

        # 🛠️ SỬA (FIX 4): Resize frame 640x640 lên kích thước cửa sổ
        if annotated is not None:
            display_frame = cv2.resize(annotated, (display_w, display_h), interpolation=cv2.INTER_AREA)
            cv2.imshow(window_name, display_frame)

        # 🛠️ SỬA (FIX 7): Thay 'frame_count' thành 'frame_idx'
        frame_idx += 1
        
        # Bỏ logic tự động dừng
        # 🛠️ SỬA (FIX 7): Thay 'frame_count' thành 'frame_idx'
        # if frame_idx > 300:
        #     break

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            EXTERNAL_STOP.set()
            break
        try:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                EXTERNAL_STOP.set()
                break
        except Exception:
             EXTERNAL_STOP.set()
             break

    cap.release()
    cv2.destroyAllWindows()
    EXTERNAL_THREAD = None # Thêm dòng này
    return "Webcam stopped."

def start_external_webcam_thread(exercise, weights):
    global EXTERNAL_THREAD
    if EXTERNAL_THREAD and EXTERNAL_THREAD.is_alive():
        return "Webcam đang chạy."
    EXTERNAL_THREAD = threading.Thread(target=external_webcam_loop, args=(exercise, weights), daemon=True)
    EXTERNAL_THREAD.start()
    return "Webcam started (external window)."

def stop_external_webcam_thread():
    global EXTERNAL_STOP, EXTERNAL_THREAD # Thêm EXTERNAL_THREAD
    EXTERNAL_STOP.set()
    if EXTERNAL_THREAD:
        EXTERNAL_THREAD.join(timeout=5.0)
    return "Webcam stopped."

# --- Video Processing (chia 3 phần, từ code cũ) ---

def process_video_split_parts(input_path: str, exercise: str, weights: str, output_resolution=(1920, 1080)):
    global BG_TASK
    
    if not Path(input_path).exists():
        BG_TASK["status"] = "error"
        BG_TASK["error"] = "Input file not found"
        return None, None, None

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"Không thể mở video: {input_path}")
        BG_TASK["status"] = "error"
        BG_TASK["error"] = "cannot_open_video"
        return None, None, None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    
    # 🛠️ SỬA: Lấy kích thước gốc của video
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cap.release()

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

    # 🛠️ SỬA: Nếu output_resolution mặc định (1920x1080), 
    # kiểm tra xem video gốc nhỏ hơn thì dùng kích thước gốc
    out_w, out_h = output_resolution
    if orig_w > 0 and orig_h > 0:
        # Nếu video gốc nhỏ hơn, dùng kích thước gốc (tránh upscale quá nhiều)
        if orig_w < out_w or orig_h < out_h:
            out_w, out_h = orig_w, orig_h
            print(f"[VideoProcess] Dùng kích thước gốc ({out_w}x{out_h}) thay vì ({output_resolution[0]}x{output_resolution[1]})")

    tmp_dir = tempfile.mkdtemp(prefix="video_parts_")
    part_paths = [os.path.join(tmp_dir, f"part_{i+1}.mp4") for i in range(len(parts))]
    final_path = os.path.join(tmp_dir, "final_annotated.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    def process_range(start_frame, end_frame, out_path):
        cap_local = cv2.VideoCapture(str(input_path))
        cap_local.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        writer = cv2.VideoWriter(out_path, fourcc, fps, (out_w, out_h))
        
        # Khởi tạo state cho process_range
        local_state = {k: v["state"].copy() for k, v in EXERCISE_REGISTRY.items()}
        local_state = reset_state(exercise, local_state)
        prev_time = time.time()
        
        frame_idx = start_frame
        last_log_time = time.time()
        
        last_annotated_canvas = None

        while True:
            if end_frame is not None and frame_idx > end_frame:
                break
            ret, frame = cap_local.read()
            if not ret:
                break
            
            annotated = None
            if (frame_idx % INFER_EVERY_N) == 0:
                try:
                    annotated_orig_size, kps, local_state[exercise], _, prev_time = _process_frame_logic(
                        frame, exercise, local_state, prev_time
                    )
                    
                    # 🛠️ SỬA: Giữ tỉ lệ bằng letterbox (không kéo giãn)
                    in_h, in_w = annotated_orig_size.shape[:2]
                    scale = min(out_w / in_w, out_h / in_h)
                    new_w = int(in_w * scale)
                    new_h = int(in_h * scale)

                    resized_frame = cv2.resize(annotated_orig_size, (new_w, new_h), interpolation=cv2.INTER_AREA)

                    # Canvas với màu đen (hoặc có thể dùng màu khác)
                    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)

                    # Căn giữa frame trên canvas
                    x_offset = (out_w - new_w) // 2
                    y_offset = (out_h - new_h) // 2

                    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_frame
                    
                    annotated = canvas
                    last_annotated_canvas = annotated

                except Exception as e:
                    print(f"Lỗi xử lý frame {frame_idx}: {e}")
                    traceback.print_exc()
                    try:
                        if 'annotated_orig_size' in locals():
                            annotated = cv2.resize(annotated_orig_size, (out_w, out_h), interpolation=cv2.INTER_AREA)
                        else:
                            annotated = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
                    except Exception:
                        annotated = np.zeros((out_h, out_w, 3), dtype=np.uint8)
                    last_annotated_canvas = annotated
            else:
                annotated = last_annotated_canvas

            if annotated is None:
                annotated = np.zeros((out_h, out_w, 3), dtype=np.uint8)
                
            annotated = np.ascontiguousarray(annotated, dtype=np.uint8)
            
            try:
                writer.write(annotated)
            except Exception as e:
                print(f"Lỗi ghi frame {frame_idx}: {e} shape={annotated.shape} dtype={annotated.dtype}")
            
            frame_idx += 1
            
            if time.time() - last_log_time > 5.0:
                print(f"[VideoProcess] Đã xử lý {frame_idx} frames... (Đang ở part: {out_path})")
                last_log_time = time.time()

        cap_local.release()
        writer.release()
        print(f"[VideoProcess] Đã hoàn thành part: {out_path}")

    # Xử lý part 1 đồng bộ
    print("[VideoProcess] Bắt đầu xử lý Part 1...")
    start0, end0 = parts[0]
    process_range(start0, end0, part_paths[0])
    BG_TASK["part1_path"] = part_paths[0]
    BG_TASK["tmp_dir"] = tmp_dir
    print("[VideoProcess] Hoàn thành Part 1.")

    def bg_job():
        BG_TASK["status"] = "processing_rest"
        print("[VideoProcess] Bắt đầu xử lý Part 2 & 3 (background)...")
        try:
            for i in range(1, len(parts)):
                s, e = parts[i]
                process_range(s, e, part_paths[i])
            
            print("[VideoProcess] Đang nối các part...")
            out = cv2.VideoWriter(final_path, fourcc, fps, (out_w, out_h))
            for p in part_paths:
                cap_p = cv2.VideoCapture(p)
                while True:
                    ret, frm = cap_p.read()
                    if not ret:
                        break
                    frm = np.ascontiguousarray(frm, dtype=np.uint8)
                    out.write(frm)
                cap_p.release()
            out.release()
            BG_TASK["final_path"] = final_path
            BG_TASK["status"] = "done"
            print("[VideoProcess] Đã xử lý xong video (final).")
        except Exception as e:
            BG_TASK["status"] = "error"
            BG_TASK["error"] = str(e)
            traceback.print_exc()
            print(f"[VideoProcess] Lỗi background: {e}")

    bg_thread = threading.Thread(target=bg_job, daemon=True)
    bg_thread.start()
    BG_TASK["thread"] = bg_thread
    BG_TASK["status"] = "part1_ready"
    return part_paths[0], tmp_dir, bg_thread

# --- Gradio Callbacks (từ code cũ) ---
def analyze_video_click(uploaded_file, exercise, weights, resolution):
    if uploaded_file is None:
        return None, "Chưa tải file lên."
    
    # Đảm bảo uploaded_file là đường dẫn
    video_path = uploaded_file
    if hasattr(uploaded_file, 'name'):
        video_path = uploaded_file.name
    elif not isinstance(uploaded_file, str):
         return None, "Lỗi định dạng file tải lên."

    weights = weights or "yolo11n-pose.pt"
    out_res = (1920, 1080) # Default
    if isinstance(resolution, str):
        try:
            if 'x' in resolution: w, h = map(int, resolution.split('x'))
            elif resolution.endswith('p'): h = int(resolution[:-1]); w = int(h * 16 / 9)
            else: w, h = 1920, 1080
            out_res = (w, h)
        except Exception:
            out_res = (1920, 1080)
            
    BG_TASK["status"] = "starting"
    print(f"[Gradio] Bắt đầu analyze_video: {video_path} | Ex: {exercise} | Res: {out_res}")
    
    part1, tmpd, thr = process_video_split_parts(video_path, exercise, weights, output_resolution=out_res)
    
    if part1 is None:
        BG_TASK["status"] = "error"
        return None, f"Xử lý thất bại. Lỗi: {BG_TASK.get('error', 'Không rõ')}"
        
    return part1, f"Part 1 đã sẵn sàng. Đang xử lý các phần còn lại... (tmp: {tmpd}). Dùng 'Xem Video (final)' để kiểm tra."

def view_remaining_click():
    st = BG_TASK.get("status", "idle")
    if st == "done" and BG_TASK.get("final_path"):
        return BG_TASK["final_path"], "Video (final) đã sẵn sàng."
    elif st in ("processing_rest","part1_ready","starting"):
        return None, f"Đang xử lý (status: {st}). Vui lòng đợi."
    elif st == "error":
        return None, f"Lỗi xử lý: {BG_TASK.get('error','unknown')}"
    else:
        return None, "Không có tác vụ nào đang chạy."

# Thêm cleanup
def cleanup():
    if BG_TASK.get("tmp_dir") and os.path.exists(BG_TASK["tmp_dir"]):
        try:
            shutil.rmtree(BG_TASK["tmp_dir"])
            print(f"Cleaned up temp dir: {BG_TASK['tmp_dir']}")
        except Exception as e:
            print(f"Error cleaning up temp dir: {e}")

atexit.register(cleanup)

# --- Giao diện Gradio (từ code cũ) ---
DEFAULT_RES_OPTIONS = ["1920x1080", "1280x720", "854x480", "1080p", "720p"]

def build_ui():
    with gr.Blocks(title="AI Fitness Tracker (Merged)") as demo:
        gr.Markdown("# 🏋️‍♂️ AI Fitness Tracker (External Cam + Split Video)")
        gr.Markdown("Sử dụng logic `form_rules` mới. Webcam chạy ở cửa sổ ngoài. Video upload được chia 3 phần.")

        with gr.Row():
            exercise = gr.Dropdown(
                list(EXERCISE_REGISTRY.keys()),
                label="Chọn bài tập (chọn trước khi Start/Analyze)",
                value=list(EXERCISE_REGISTRY.keys())[0]
            )
            weights_input = gr.Textbox(value="yolo11n-pose.pt", label="Model weights path (local file)")

        gr.Markdown("---")
        gr.Markdown("### 🎥 Webcam Trực Tiếp (Cửa sổ ngoài)")
        with gr.Row():
            start_btn = gr.Button("Bắt đầu Webcam ngoài")
            stop_btn = gr.Button("Dừng Webcam ngoài")
        status = gr.Textbox(label="Trạng thái Webcam", value="Sẵn sàng", interactive=False)
        
        start_btn.click(fn=start_external_webcam_thread, inputs=[exercise, weights_input], outputs=[status])
        stop_btn.click(fn=stop_external_webcam_thread, inputs=None, outputs=[status])

        gr.Markdown("---")
        gr.Markdown("### 📁 Phân tích Video (Chia 3 phần)")
        with gr.Row():
            upload = gr.File(label="Tải video file (.mp4, .mov)")
            # 🛠️ SỬA (FIX 6): Đặt giá trị mặc định là 1080p (1920x1080)
            res_choice = gr.Dropdown(DEFAULT_RES_OPTIONS, value="1920x1080", label="Độ phân giải đầu ra")
        with gr.Row():
            analyze_btn = gr.Button("🎬 Phân tích Video (Part 1)")
            view_btn = gr.Button("🍿 Xem Video (final)")
            
        out_video = gr.Video(
            label="Video kết quả (part 1 hoặc final)",
            scale=1,  # Cho phép tự điều chỉnh theo tỉ lệ gốc
            format="mp4"
        )
        message = gr.Textbox(label="Trạng thái Video", value="", interactive=False)

        analyze_btn.click(fn=analyze_video_click, inputs=[upload, exercise, weights_input, res_choice], outputs=[out_video, message])
        view_btn.click(fn=view_remaining_click, inputs=None, outputs=[out_video, message])

    return demo

if __name__ == "__main__":
    app = build_ui()
    try:
        app.launch(server_name="localhost", server_port=7860, share=False)
    except Exception as e:
        print(f"Failed to launch Gradio app: {e}")
        raise