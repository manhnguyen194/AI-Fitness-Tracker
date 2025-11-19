# src/voice_player.py
"""
AI-Fitness-Tracker Voice Player (Pygame Backend)

- Backend: Dùng 'pygame.mixer' thay vì 'playsound' để sửa lỗi MCI/Path trên Windows.
- Logic:
  + Single worker queue (không chồng tiếng).
  + Welcome message -> Wait -> Periodic loop.
  + Debounce tone change (tránh đổi giọng liên tục khi AI nhận diện chập chờn).
  + Silent window (không phát ngay sau khi vừa đổi giọng để tránh spam).
  + Thêm cờ `_is_active` để tắt âm thanh định kỳ khi không phát hiện người/bài tập.
"""

import os
import time
import threading
import queue

# --- Pygame backend init ---
try:
    # Tắt thông báo chào của pygame
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"
    import pygame

    _HAVE_PYGAME = True
except ImportError:
    _HAVE_PYGAME = False
    print("⚠️ Lỗi: Chưa cài thư viện pygame. Hãy chạy: pip install pygame")

VALID_TONES = ("positive", "neutral", "negative", "good", "bad")


# ---------- Helpers ----------
def _mp3_exists(path: str) -> bool:
    return bool(path) and path.lower().endswith(".mp3") and os.path.exists(path)


# ---------- VoicePlayer ----------
class VoicePlayer:
    def __init__(
            self,
            voices_dir: str,
            base_interval_first: float = 6.0,
            base_interval_same: float = 5.0,
            tone_change_debounce_ms: int = 1100,
            require_stable_frames: int = 8,
            edge_cooldown_sec: float = 3.0,
            no_play_if_recent_change_sec: float = 2.5,
    ):
        self.voices_dir = voices_dir
        self.base_interval_first = float(base_interval_first)
        self.base_interval_same = float(base_interval_same)

        # Init mixer nếu chưa init
        if _HAVE_PYGAME:
            try:
                if not pygame.mixer.get_init():
                    pygame.mixer.init()
            except Exception as e:
                print(f"⚠️ Không thể khởi tạo âm thanh: {e}")

        # Tone state
        self.current_tone: str = "neutral"
        self._last_periodic_tone: str | None = None
        self._cycle = {t: 1 for t in VALID_TONES}  # Index xoay vòng 1->2->3

        # Debounce / cooldown
        self.tone_change_debounce_ms = int(tone_change_debounce_ms)
        self.require_stable_frames = int(require_stable_frames)
        self._edge_cooldown_sec = float(edge_cooldown_sec)
        self.no_play_if_recent_change_sec = float(no_play_if_recent_change_sec)

        # Debounce pending var
        self._pending_tone: str | None = None
        self._pending_since_ts: float | None = None
        self._pending_count: int = 0

        # Trạng thái Hoạt động MỚI
        self._is_active: bool = False

        # Scheduling
        self._next_play_ts: float = float("inf")
        self._last_tone_change_ts: float = 0.0

        # Threading
        self._stop_flag = threading.Event()
        self._lock = threading.Lock()

        # Playback Queue
        self._play_queue: "queue.Queue[tuple[str,str]]" = queue.Queue()
        self._play_thread: threading.Thread | None = None

    # ---------- Public ----------
    def start(self):
        if self._play_thread and self._play_thread.is_alive():
            return
        self._stop_flag.clear()

        # Worker thread (phát âm thanh từ hàng đợi)
        self._play_thread = threading.Thread(target=self._play_worker, daemon=True)
        self._play_thread.start()

        # Scheduler thread (tính toán thời điểm phát tiếp theo)
        threading.Thread(target=self._scheduler_loop, daemon=True).start()

    def stop(self):
        self._stop_flag.set()
        try:
            # Gửi lệnh dừng tới hàng đợi
            self._play_queue.put_nowait(("", "stop"))
        except Exception:
            pass

        # Dừng nhạc ngay lập tức
        if _HAVE_PYGAME and pygame.mixer.get_init():
            pygame.mixer.music.stop()

        if self._play_thread:
            self._play_thread.join(timeout=1.0)

    def set_active(self, is_active: bool):
        """Cập nhật cờ hoạt động. Gọi khi KPS được phát hiện."""
        with self._lock:
            self._is_active = is_active
            # Nếu vừa chuyển sang trạng thái inactive, reset thời gian chờ
            if not is_active:
                self._next_play_ts = time.time() + 1.0

    def set_tone(self, tone: str):
        """
        Nhận tone từ AI model -> Debounce -> Quyết định có đổi giọng hay không.
        """
        t = (tone or "").lower()
        if t not in VALID_TONES:
            # Xử lý trường hợp `tone` được gán là 'good'/'bad' nhưng thư mục chưa có file
            # Ví dụ: đổi 'good'/'bad' thành 'positive'/'negative' để dùng file chung
            if t == 'good': t = 'positive'
            if t == 'bad': t = 'negative'
            if t not in VALID_TONES:
                return

        now = time.time()
        with self._lock:
            # Nếu tone giống hiện tại -> reset pending
            if t == self.current_tone:
                self._pending_tone = None
                self._pending_since_ts = None
                self._pending_count = 0
                return

            # Nếu bắt đầu đổi sang tone mới
            if self._pending_tone != t:
                self._pending_tone = t
                self._pending_since_ts = now
                self._pending_count = 1
                return

            # Đang pending tone t
            self._pending_count += 1
            elapsed_ms = (now - (self._pending_since_ts or now)) * 1000.0

            # Kiểm tra đủ điều kiện debounce chưa
            if (elapsed_ms < self.tone_change_debounce_ms) or (self._pending_count < self.require_stable_frames):
                return

            # === CHẤP NHẬN ĐỔI TONE ===
            new_tone = t
            self.current_tone = new_tone
            self._pending_tone = None
            self._pending_since_ts = None
            self._pending_count = 0

            # Reset cycle
            self._cycle[new_tone] = 1

            # Xử lý Edge Trigger (Phát ngay khi đổi, trừ khi vừa đổi xong)
            if (now - self._last_tone_change_ts) >= self.no_play_if_recent_change_sec:
                path = self._tone_file(new_tone, 1)
                self._enqueue(path, reason="edge-change")
                self._cycle[new_tone] = 2  # Lần tới sẽ là câu 2
            else:
                # Trong vùng im lặng -> bỏ qua edge, giữ index 1
                self._cycle[new_tone] = 1

            # Cập nhật thời gian
            self._last_tone_change_ts = now
            self._next_play_ts = max(self._next_play_ts, now + self._edge_cooldown_sec)
            self._last_periodic_tone = new_tone

    # ---------- Internal Workers ----------
    def _play_worker(self):
        """
        Lấy file từ hàng đợi và phát bằng Pygame (Blocking logic)
        """
        while not self._stop_flag.is_set():
            try:
                # Đặt timeout ngắn để thread có thể thoát khi cờ dừng được set
                path, reason = self._play_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if not path or reason == "stop":
                continue

            if not _HAVE_PYGAME:
                continue

            if not _mp3_exists(path):
                # print(f"[VoicePlayer] ❌ File not found: {path}")
                continue

            # Phát nhạc
            try:
                # print(f"[VoicePlayer] 🔊 Play: {os.path.basename(path)} ({reason})")
                pygame.mixer.music.load(path)
                pygame.mixer.music.play()

                # Chờ nhạc chạy xong (Blocking giả lập)
                while pygame.mixer.music.get_busy() and not self._stop_flag.is_set():
                    time.sleep(0.1)

            except Exception as e:
                print(f"[VoicePlayer] Error playing: {e}")

    def _enqueue(self, path: str, reason: str):
        if _mp3_exists(path):
            try:
                self._play_queue.put_nowait((path, reason))
            except Exception:
                pass

    def _scheduler_loop(self):
        """
        Điều phối việc phát định kỳ (Periodic)
        """
        # 1. Phát Welcome
        welcome_path = os.path.join(self.voices_dir, "welcome.mp3")
        if _mp3_exists(welcome_path):
            self._enqueue(welcome_path, reason="welcome")

        # Đợi 2s sau khi start
        start_ts = time.time()
        while not self._stop_flag.is_set() and (time.time() - start_ts) < 2.0:
            time.sleep(0.05)

        # Hẹn giờ lần đầu
        with self._lock:
            self._next_play_ts = time.time() + self.base_interval_first
            self._last_periodic_tone = None

        # 2. Vòng lặp chính
        while not self._stop_flag.is_set():
            now = time.time()
            with self._lock:

                # 💡 LOGIC SỬA LỖI: CHỈ PHÁT ĐỊNH KỲ NẾU self._is_active LÀ TRUE
                if not self._is_active:
                    # Kiểm tra lại sau khoảng thời gian ngắn
                    self._next_play_ts = now + 1.0
                    time.sleep(0.05)
                    continue

                if now >= self._next_play_ts:
                    # Nếu vừa đổi tone gần đây -> Skip periodic này
                    if (now - self._last_tone_change_ts) < self.no_play_if_recent_change_sec:
                        # Tính lại thời gian chờ
                        interval = self.base_interval_same if (self._last_periodic_tone == self.current_tone) \
                            else self.base_interval_first
                        self._next_play_ts = max(now + interval, now + self._edge_cooldown_sec)
                    else:
                        # Phát định kỳ
                        tone = self.current_tone
                        idx = self._cycle.get(tone, 1)  # Đảm bảo key tồn tại
                        path = self._tone_file(tone, idx)
                        self._enqueue(path, reason="periodic")

                        # Tăng index (1->2->3->1)
                        self._cycle[tone] = 1 if idx >= 3 else (idx + 1)

                        # Tính thời gian chờ lần tới
                        interval = self.base_interval_same if (self._last_periodic_tone == tone) \
                            else self.base_interval_first
                        self._last_periodic_tone = tone
                        self._next_play_ts = max(now + interval, now + self._edge_cooldown_sec)

            time.sleep(0.05)

    def _tone_file(self, tone: str, index: int) -> str:
        return os.path.join(self.voices_dir, f"{tone}_{index}.mp3")


# -------- Singleton API --------
_player: VoicePlayer | None = None


def init(voices_dir: str,
         base_interval_first: float = 6.0,
         base_interval_same: float = 5.0):
    global _player
    if _player:
        _player.stop()
    _player = VoicePlayer(
        voices_dir=voices_dir,
        base_interval_first=base_interval_first,
        base_interval_same=base_interval_same,
    )
    _player.start()
    return _player


def set_tone(tone: str):
    if _player:
        _player.set_tone(tone)


def set_active(is_active: bool):
    """API mới: Kích hoạt/Vô hiệu hóa phản hồi định kỳ."""
    if _player:
        _player.set_active(is_active)


def stop():
    global _player
    if _player:
        _player.stop()
        _player = None