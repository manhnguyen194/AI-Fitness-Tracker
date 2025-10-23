import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import textwrap
import os

# -----------------------------
# 🎨 Hỗ trợ vẽ text tiếng Việt & overlay
# -----------------------------

def load_font(font_path, size):
    """
    Load font có hỗ trợ tiếng Việt.
    Nếu không tìm thấy file, dùng font mặc định của PIL.
    """
    try:
        return ImageFont.truetype(font_path, size=size)
    except Exception:
        print("⚠️ Font không tồn tại hoặc không hỗ trợ Unicode — dùng mặc định.")
        return ImageFont.load_default()


def draw_colored_line(frame, p1, p2, color=(0, 255, 0), thickness=3):
    """
    Vẽ đoạn thẳng màu giữa 2 điểm (x, y).
    """
    if p1 and p2:
        p1, p2 = tuple(map(int, p1)), tuple(map(int, p2))
        cv2.line(frame, p1, p2, color, thickness)


def draw_text_pil(frame_bgr, lines, font_path, font_scale=28, pos=(20, 20)):
    """
    Vẽ nhiều dòng text tiếng Việt lên frame, tự động xuống dòng theo chiều rộng thật.
    - lines: [(text, color)]
    """
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    h_img, w_img = pil_img.height, pil_img.width
    computed_size = max(14, int(font_scale * (h_img / 720)))
    font = load_font(font_path, computed_size)

    x, y = pos
    max_text_width = int(w_img * 0.9)  # Giới hạn chiều rộng dòng = 90% frame

    for text, color in lines:
        words = text.split(" ")
        current_line = ""
        for word in words:
            test_line = f"{current_line} {word}".strip()
            # Dùng textbbox để tính bề ngang dòng text thực tế
            bbox = draw.textbbox((0, 0), test_line, font=font)
            line_width = bbox[2] - bbox[0]
            if line_width <= max_text_width:
                current_line = test_line
            else:
                # Vẽ dòng đầy
                draw.text((x, y), current_line, font=font, fill=color,
                          stroke_width=max(1, computed_size // 14),
                          stroke_fill=(0, 0, 0))
                y += int(computed_size * 1.4)
                current_line = word
        # Vẽ dòng cuối cùng
        if current_line:
            draw.text((x, y), current_line, font=font, fill=color,
                      stroke_width=max(1, computed_size // 14),
                      stroke_fill=(0, 0, 0))
            y += int(computed_size * 1.6)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
