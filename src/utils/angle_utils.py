import math

def calculate_angle(a, b, c):
    """Tính góc giữa ba điểm (x, y)"""
    try:
        ba = (a[0] - b[0], a[1] - b[1])
        bc = (c[0] - b[0], c[1] - b[1])
        cosine_angle = (ba[0] * bc[0] + ba[1] * bc[1]) / (
                math.sqrt(ba[0] ** 2 + ba[1] ** 2) * math.sqrt(bc[0] ** 2 + bc[1] ** 2)
        )
        return math.degrees(math.acos(max(-1, min(1, cosine_angle))))
    except:
        return 0