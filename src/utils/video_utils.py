import cv2
import time

def setup_window(window_name, frame, max_height=720):
    """
    Tự động điều chỉnh kích thước cửa sổ theo video input.
    """
    h, w = frame.shape[:2]
    aspect = w / h
    window_h = min(h, max_height)
    window_w = int(window_h * aspect)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, window_w, window_h)


def compute_fps(prev_time):
    """Tính FPS (frames per second)."""
    now = time.time()
    fps = 1.0 / (now - prev_time) if now != prev_time else 0.0
    return fps, now


def draw_fps(frame, fps):
    """Vẽ FPS lên góc phải trên."""
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
