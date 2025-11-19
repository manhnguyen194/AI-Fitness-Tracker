import os
import cv2
import time
import torch
import contextlib
import io
from ultralytics import YOLO

# Import các module hỗ trợ
from utils.draw_utils import draw_text_pil
from utils.video_utils import compute_fps
from rep_counter import count_squat, count_pushup, count_plank, count_situp
from form_rules import evaluate_squat, evaluate_pushup, evaluate_plank, evaluate_situp
import voice_player


class FitnessTracker:
    def __init__(
            self,
            exercise="squat",
            data_dir="data",
            fonts_dir="fonts",
            voices_dir="data/voices",
            img_size=640,
            inference_every=3,
            draw_every=3,
            conf=0.5
    ):
        self.exercise = exercise
        self.img_size = img_size
        self.inference_every = inference_every
        self.draw_every = draw_every
        self.conf = conf

        # --- Paths ---
        self.exercise = self.exercise.lower().replace('-', '')
        self.voices_dir = voices_dir
        self.font_path = os.path.join(fonts_dir, "Roboto.ttf")

        # --- Register exercises ---
        self.exercise_registry = {
            "squat": {
                "counter_func": count_squat,
                "form_func": evaluate_squat,
                "state": {"stage": "up", "counter": 0, "prev_angle": 160, "direction": "up", "active_frames": 0},
            },
            "pushup": {
                "counter_func": count_pushup,
                "form_func": evaluate_pushup,
                "state": {"stage": "up", "counter": 0, "prev_angle": 160, "direction": "up", "active_frames": 0},
            },
            "plank": {
                "counter_func": count_plank,
                "form_func": evaluate_plank,
                "state": {"good_time": 0, "bad_time": 0, "is_good": False, "start_time": None, "elapsed": 0.0,
                          "active_frames": 0, "last_time": None},  # Thêm last_time vào init
            },
            "situp": {
                "counter_func": count_situp,
                "form_func": evaluate_situp,
                "state": {"stage": "down", "counter": 0, "prev_angle": 140, "direction": "down", "active_frames": 0},
            },
        }

        if self.exercise not in self.exercise_registry:
            raise ValueError(f"Exercise '{self.exercise}' not supported.")

        cfg = self.exercise_registry[self.exercise]
        self.counter_func = cfg["counter_func"]
        self.form_func = cfg["form_func"]
        self.state = cfg["state"].copy()

        # --- Variables cho Logic Âm Thanh ---
        self.last_counter = 0
        self.override_speech_until = 0.0
        self.override_tone = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # === LOG DEVICE INFO ===
        print(f"\n{'=' * 40}")
        print(f"🚀 AI Fitness Tracker Initialized")
        print(f"► Device    : {self.device.upper()}")
        print(f"► Exercise  : {self.exercise.upper()}")
        print(f"{'=' * 40}\n")

        self.model = YOLO("yolo11n-pose.pt")
        self.model.conf = conf
        try:
            self.model.to(self.device)
        except:
            pass

        try:
            with contextlib.redirect_stdout(io.StringIO()):
                self.model.fuse()
            # Warmup
            import numpy as _np
            dummy = _np.zeros((self.img_size, self.img_size, 3), dtype=_np.uint8)
            with torch.inference_mode():
                _ = self.model.predict(dummy, verbose=False, imgsz=self.img_size, max_det=1, conf=self.conf)
        except Exception:
            pass

        voice_player.init(voices_dir, base_interval_first=10.0, base_interval_same=8.0)

        # --- Internal states ---
        self.frame_idx = 0
        self.last_results = None
        self.last_annotated = None
        self.last_kps = None
        self.prev_time = time.time()

    # [SỬA ĐỔI QUAN TRỌNG]: Thêm tham số timestamp=None
    def process_frame(self, frame, timestamp=None):
        if frame is None:
            voice_player.set_active(False)
            return None

        frame_resized = cv2.resize(frame, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)

        # 1. Inference
        do_inference = (self.frame_idx % self.inference_every == 0)
        if do_inference:
            try:
                with torch.inference_mode():
                    results = self.model.predict(
                        frame_resized,
                        verbose=False,
                        conf=self.conf,
                        device=self.device,
                        imgsz=self.img_size,
                        max_det=1,
                    )
                res = results[0] if results else None
                self.last_results = res
            except Exception:
                res = self.last_results
        else:
            res = self.last_results

        # 2. Draw
        if self.frame_idx % self.draw_every == 0 and res is not None:
            try:
                annotated = res.plot()
                self.last_annotated = annotated
            except:
                annotated = frame_resized.copy()
                self.last_annotated = annotated
        else:
            annotated = (
                self.last_annotated if self.last_annotated is not None else frame_resized.copy()
            )

        # 3. Get Keypoints
        kps = self.last_kps
        has_kps_detected = False

        if res is not None and getattr(res, "keypoints", None) and res.boxes.conf is not None:
            if len(res.boxes.conf) > 0:
                max_idx = torch.argmax(res.boxes.conf).item()
                if res.boxes.conf[max_idx] > self.conf:
                    arr = getattr(res.keypoints, "xy", None)
                    if arr is not None and len(arr) > 0 and max_idx < len(arr):
                        kps = arr[max_idx].cpu().numpy()[:, :2].tolist()
                        self.last_kps = kps
                        has_kps_detected = True

        # --- CORE LOGIC ---
        counter = self.state.get("counter", 0)
        stage = self.state.get("stage", "up")
        feedback = "..."
        feedback_color = (255, 255, 255)
        current_angle = 180.0
        calculated_tone = "neutral"

        if kps is not None:
            try:
                # [SỬA ĐỔI QUAN TRỌNG]: Truyền timestamp vào hàm counter
                # Nếu timestamp là None (webcam), rep_counter sẽ tự dùng time.time()
                result = self.counter_func(kps, self.state, current_timestamp=timestamp)

                if isinstance(result, (tuple, list)):
                    counter = result[0]
                    if len(result) >= 3:
                        current_angle = result[2]
                else:
                    counter = result
                    current_angle = self.state.get('prev_angle', 180.0)

                stage = self.state.get('stage', stage)

                # Form Eval
                eval_res = self.form_func(kps, annotated, stage, counter)
                if isinstance(eval_res, (tuple, list)) and len(eval_res) >= 3:
                    if len(eval_res) == 4:
                        _, feedback, calculated_tone, _ = eval_res
                    else:
                        _, feedback, calculated_tone = eval_res[:3]
                else:
                    feedback, calculated_tone = "Analyzing...", "neutral"

            except Exception as e:
                print(f"Logic Error: {e}")
                pass

        # ... (Phần Logic Âm thanh giữ nguyên không đổi) ...
        final_tone = calculated_tone
        final_active = False

        # Logic âm thanh plank/squat/etc giữ nguyên...
        if counter > self.last_counter:
            self.override_speech_until = time.time() + 2.5
            self.override_tone = "positive"
            self.last_counter = counter

        if time.time() < self.override_speech_until:
            final_active = True
            final_tone = self.override_tone
        else:
            is_motion_active = False
            if has_kps_detected:
                if self.exercise == "plank":
                    if self.state.get('is_good', False) or self.state.get('elapsed', 0.0) > 1.0:
                        is_motion_active = True
                # Các bài tập khác giữ nguyên logic
                elif self.exercise in ["squat", "pushup", "situp"]:
                    # Logic cũ
                    if self.exercise == "squat" and current_angle < 165:
                        is_motion_active = True
                    elif self.exercise == "pushup" and current_angle < 160:
                        is_motion_active = True
                    elif self.exercise == "situp" and current_angle < 135:
                        is_motion_active = True

            current_active_frames = self.state.get("active_frames", 0)
            if is_motion_active:
                current_active_frames = min(current_active_frames + 1, 30)
            else:
                current_active_frames = max(current_active_frames - 1, 0)
            self.state["active_frames"] = current_active_frames

            if current_active_frames >= 5:
                final_active = True
            if final_active and final_tone == "neutral" and self.exercise != "plank":
                final_active = False
            if final_tone == "negative" and current_active_frames >= 5:
                final_active = True

        voice_player.set_tone(final_tone)
        voice_player.set_active(final_active)

        if final_tone == "positive":
            feedback_color = (0, 255, 0)
        elif final_tone == "neutral":
            feedback_color = (0, 255, 255)
        elif final_tone == "negative":
            feedback_color = (0, 0, 255)

        # Visuals
        fps, self.prev_time = compute_fps(self.prev_time)
        display_counter = self.state.get("elapsed", self.state.get("counter", counter))
        display_stage = self.state.get("stage", stage)

        if self.exercise == "plank":
            val_str = f"{display_counter:.1f}s"
            lbl_str = "Time"
        else:
            val_str = f"{int(display_counter)}"
            lbl_str = "Count"

        lines = [
            (f"{lbl_str}: {val_str}", (255, 215, 0)),
            (f"Stage: {display_stage}", (255, 255, 255)),
            (f"Eval: {feedback}", feedback_color),
            (f"FPS: {fps:.1f}", (200, 200, 200)),
        ]

        current_font_path = self.font_path if os.path.exists(self.font_path) else None
        annotated = draw_text_pil(annotated, lines, font_path=current_font_path, font_scale=26, pos=(20, 20))

        self.frame_idx += 1
        return annotated