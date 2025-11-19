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
    Load font thông minh:
    1. Thử load font theo đường dẫn chỉ định.
    2. Nếu lỗi, thử tìm các font hệ thống phổ biến (Arial, Segoe UI...).
    3. Cùng đường mới dùng font mặc định (lưu ý: font mặc định KHÔNG resize được).
    """
    # 1. Thử font chỉ định
    try:
        return ImageFont.truetype(font_path, size=size)
    except (IOError, OSError):
        pass  # Thất bại thì bỏ qua, xuống bước 2

    # 2. Thử các font hệ thống (Windows/Linux/Mac)
    # Các font này hỗ trợ thay đổi kích thước tốt
    common_fonts = [
        "arial.ttf",  # Windows/Standard
        "segoeui.ttf",  # Windows Modern
        "calibri.ttf",  # Windows Office
        "Roboto-Regular.ttf",  # Android/Web
        "DejaVuSans.ttf",  # Linux
        "FreeSans.ttf"  # Linux
    ]

    for name in common_fonts:
        try:
            return ImageFont.truetype(name, size=size)
        except (IOError, OSError):
            continue

    # 3. Fallback cuối cùng (Chữ sẽ rất bé)
    print("⚠️ CẢNH BÁO: Không tìm thấy bất kỳ font nào hỗ trợ resize. Dùng font Bitmap mặc định.")
    return ImageFont.load_default()


def draw_colored_line(frame, p1, p2, color=(0, 255, 0), thickness=3):
    """
    Vẽ đoạn thẳng màu giữa 2 điểm (x, y).
    """
    if p1 and p2:
        p1, p2 = tuple(map(int, p1)), tuple(map(int, p2))
        cv2.line(frame, p1, p2, color, thickness)


def draw_text_pil(frame_bgr, lines, font_path, font_scale=28, pos=(20, 20), wrap_text=True):
    """
    Vẽ nhiều dòng text tiếng Việt lên frame.
    - lines: [(text, color)]
    - wrap_text: True (tự xuống dòng), False (vẽ trực tiếp 1 dòng - dùng cho tiêu đề).
    """
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    h_img, w_img = pil_img.height, pil_img.width

    # Tính toán size chữ dựa trên chiều cao ảnh
    # Ví dụ: 720p -> size ~28 | 1080p -> size ~42
    computed_size = max(14, int(font_scale * (h_img / 720)))

    # Load font (Đã nâng cấp logic tìm font)
    font = load_font(font_path, computed_size)

    x, y = pos

    # Tính độ dày viền chữ (Stroke) giúp chữ nổi bật hơn trên nền video
    # Tăng độ dày lên một chút so với bản cũ (chia 10 thay vì 14)
    stroke_w = max(1, computed_size // 10)

    if not wrap_text:
        # Chế độ Overlay (Tiêu đề, Mode, Count...)
        line_spacing = int(computed_size * 1.3)
        for text, color in lines:
            # Chuyển màu BGR -> RGB
            color_rgb = tuple(reversed(color)) if len(color) == 3 else color

            draw.text((x, y), text, font=font, fill=color_rgb,
                      stroke_width=stroke_w,
                      stroke_fill=(0, 0, 0))  # Viền đen
            y += line_spacing
    else:
        # Chế độ Wrap text (Đoạn văn hướng dẫn dài)
        max_text_width = int(w_img * 0.9)
        for text, color in lines:
            words = text.split(" ")
            current_line = ""
            color_rgb = tuple(reversed(color)) if len(color) == 3 else color

            for word in words:
                test_line = f"{current_line} {word}".strip()
                bbox = draw.textbbox((0, 0), test_line, font=font)
                line_width = bbox[2] - bbox[0]

                if line_width <= max_text_width:
                    current_line = test_line
                else:
                    draw.text((x, y), current_line, font=font, fill=color_rgb,
                              stroke_width=stroke_w,
                              stroke_fill=(0, 0, 0))
                    y += int(computed_size * 1.4)
                    current_line = word

            if current_line:
                draw.text((x, y), current_line, font=font, fill=color_rgb,
                          stroke_width=stroke_w,
                          stroke_fill=(0, 0, 0))
                y += int(computed_size * 1.6)

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)