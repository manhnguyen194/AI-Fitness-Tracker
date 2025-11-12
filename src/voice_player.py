# src/voice_player.py
"""
AI-Fitness-Tracker Voice Player (MP3-only, queued, no overlap, tone-silence window)

- Single playback worker (queue) -> KHÔNG chồng tiếng:
  Nếu đang phát mà tone đổi, edge clip sẽ vào hàng đợi, phát SAU khi clip hiện tại kết thúc.
- Plays 'welcome.mp3' 1 lần, đợi 2s, rồi vào logic theo tone.
- Periodic:
    * Lần đầu theo tone: base_interval_first (mặc định 6s).
    * Nếu giữ nguyên tone: base_interval_same (mặc định 5s).
    * Thứ tự mỗi tone: <tone>_1.mp3 -> _2.mp3 -> _3.mp3 -> _1...
- Edge trigger:
    * Khi tone đổi (qua debounce), enqueue ngay <tone>_1.
    * Sau đó block >= edge_cooldown_sec (mặc định 2.5s) để tránh dày.
- "Silent window" sau khi đổi tone:
    * Nếu đổi tone rồi NHẢY tiếp trong <= no_play_if_recent_change_sec (mặc định 2.0s)
      thì KHÔNG phát (bỏ qua edge/periodic trong khoảng này).
- MP3 only. Nếu thiếu file -> print("Audio not found").
"""

import os
import sys
import time
import threading
import queue
import tempfile
import shutil
import uuid
import atexit
import ctypes

# --- playsound backend (blocking) ---
try:
    from playsound import playsound  # pip install playsound==1.3.0
    _HAVE_PLAYSOUND = True
except Exception:
    _HAVE_PLAYSOUND = False

VALID_TONES = ("positive", "neutral", "negative")

# ---------- Safe-path helpers (fix MCI issues on Windows) ----------
def _mp3_exists(path: str) -> bool:
    return bool(path) and path.lower().endswith(".mp3") and os.path.exists(path)

# Windows 8.3 short path (ASCII-ish, no spaces)
def _win_short_path(path: str) -> str | None:
    try:
        GetShortPathNameW = ctypes.windll.kernel32.GetShortPathNameW
        GetShortPathNameW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint]
        GetShortPathNameW.restype = ctypes.c_uint
        buf = ctypes.create_unicode_buffer(260)
        res = GetShortPathNameW(path, buf, 260)
        return buf.value if res > 0 else None
    except Exception:
        return None

_TEMP_DIR = os.path.join(tempfile.gettempdir(), "aft_voice_tmp")
os.makedirs(_TEMP_DIR, exist_ok=True)
def _cleanup_tmp_dir():
    try:
        shutil.rmtree(_TEMP_DIR, ignore_errors=True)
    except Exception:
        pass
atexit.register(_cleanup_tmp_dir)

def _playsound_blocking_safe(path: str):
    """
    Gọi playsound blocking theo cách an toàn trên Windows:
    - Thử 8.3 short path; nếu fail -> copy sang temp ASCII/no-space rồi playsound.
    - Non-Windows: playsound(path) trực tiếp.
    """
    if not _HAVE_PLAYSOUND:
        raise RuntimeError("playsound is not available")

    if not sys.platform.startswith("win"):
        playsound(path)
        return

    sp = _win_short_path(path)
    if sp and os.path.exists(sp):
        try:
            playsound(sp)
            return
        except Exception:
            pass

    fname = f"voice_{uuid.uuid4().hex}.mp3"
    safe_path = os.path.join(_TEMP_DIR, fname)
    shutil.copyfile(path, safe_path)
    try:
        playsound(safe_path)
    finally:
        try: os.remove(safe_path)
        except Exception: pass

# ---------- VoicePlayer ----------
class VoicePlayer:
    def __init__(
        self,
        voices_dir: str,
        base_interval_first: float = 6.0,
        base_interval_same: float  = 5.0,
        tone_change_debounce_ms: int = 1100,  # yêu cầu tone mới ổn định tối thiểu 1.1s
        require_stable_frames: int = 8,       # và >= 8 frames liên tiếp
        edge_cooldown_sec: float   = 2.5,     # sau edge, chờ >= 2.5s
        no_play_if_recent_change_sec: float = 2.0,  # vùng yên lặng 2s sau khi đổi tone
    ):
        self.voices_dir = voices_dir
        self.base_interval_first = float(base_interval_first)
        self.base_interval_same  = float(base_interval_same)

        # Tone state
        self.current_tone: str = "neutral"
        self._last_periodic_tone: str | None = None
        self._cycle = {t: 1 for t in VALID_TONES}  # next index 1..3 per tone

        # Debounce / cooldown / silent window
        self.tone_change_debounce_ms = int(tone_change_debounce_ms)
        self.require_stable_frames   = int(require_stable_frames)
        self._edge_cooldown_sec      = float(edge_cooldown_sec)
        self.no_play_if_recent_change_sec = float(no_play_if_recent_change_sec)

        # Debounce pending
        self._pending_tone: str | None = None
        self._pending_since_ts: float | None = None
        self._pending_count: int = 0

        # Scheduling
        self._next_play_ts: float = float("inf")
        self._last_tone_change_ts: float = 0.0  # mốc khi CHẤP NHẬN đổi tone

        # Threading
        self._stop_flag = threading.Event()
        self._lock = threading.Lock()

        # Playback worker (single)
        self._play_queue: "queue.Queue[tuple[str,str]]" = queue.Queue()
        self._play_thread: threading.Thread | None = None
        self._is_playing: bool = False

        # Welcome flag
        self._did_welcome = False

    # ---------- Public ----------
    def start(self):
        if self._play_thread and self._play_thread.is_alive():
            return
        self._stop_flag.clear()
        # worker trước
        self._play_thread = threading.Thread(target=self._play_worker, daemon=True)
        self._play_thread.start()
        # main loop sau
        threading.Thread(target=self._main_loop, daemon=True).start()

    def stop(self):
        self._stop_flag.set()
        try: self._play_queue.put_nowait(("", "stop"))
        except Exception: pass
        if self._play_thread:
            self._play_thread.join(timeout=1.0)

    def set_tone(self, tone: str):
        """
        Debounce đổi tone:
          - Chỉ nhận đổi nếu tone mới ổn định >= tone_change_debounce_ms
            VÀ liên tiếp >= require_stable_frames.
          - Khi nhận, enqueue <tone>_1 (edge-change) TRỪ KHI đang trong silent window.
          - Sau đó block >= edge_cooldown_sec.
        """
        t = (tone or "").lower()
        if t not in VALID_TONES:
            return

        now = time.time()
        with self._lock:
            if t == self.current_tone:
                # giữ nguyên tone -> xoá pending
                self._pending_tone = None
                self._pending_since_ts = None
                self._pending_count = 0
                return

            if self._pending_tone != t:
                # bắt đầu pending mới
                self._pending_tone = t
                self._pending_since_ts = now
                self._pending_count = 1
                return

            # vẫn pending cùng một t
            self._pending_count += 1
            elapsed_ms = (now - (self._pending_since_ts or now)) * 1000.0
            if (elapsed_ms < self.tone_change_debounce_ms) or (self._pending_count < self.require_stable_frames):
                return  # chưa đủ điều kiện

            # === CHẤP NHẬN đổi tone ===
            new_tone = t
            self.current_tone = new_tone
            self._pending_tone = None
            self._pending_since_ts = None
            self._pending_count = 0

            # reset thứ tự về _1
            self._cycle[new_tone] = 1

            # Silent window: nếu vừa đổi tone gần đây (<= window) -> KHÔNG phát edge
            if (now - self._last_tone_change_ts) >= self.no_play_if_recent_change_sec:
                path = self._tone_file(new_tone, 1)
                self._enqueue(path, reason="edge-change")
                # lần sau của tone này sẽ là _2
                self._cycle[new_tone] = 2
            else:
                # bỏ qua edge (giữ _1 để kỳ sau phát)
                self._cycle[new_tone] = 1

            # cập nhật mốc đổi tone & block periodic
            self._last_tone_change_ts = now
            self._next_play_ts = max(self._next_play_ts, now + self._edge_cooldown_sec)
            self._last_periodic_tone = new_tone

    # ---------- Internal ----------
    def _play_worker(self):
        """Single playback worker: playsound() BLOCKING -> không chồng tiếng."""
        while not self._stop_flag.is_set():
            try:
                path, reason = self._play_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if not path or reason == "stop":
                continue

            if not _mp3_exists(path) or not _HAVE_PLAYSOUND:
                print("Audio not found")
                continue

            self._is_playing = True
            try:
                _playsound_blocking_safe(path)
                print(f"[VoicePlayer] play {os.path.splitext(os.path.basename(path))[0]} ({reason})")
            except Exception:
                print("Audio not found")
            finally:
                self._is_playing = False

    def _enqueue(self, path: str, reason: str):
        if not _mp3_exists(path):
            print("Audio not found")
            return
        try:
            self._play_queue.put_nowait((path, reason))
        except Exception:
            pass

    def _main_loop(self):
        """
        1) Welcome nếu có, đợi 2s.
        2) Hẹn lần periodic đầu: base_interval_first.
        3) Nếu giữ nguyên tone từ lần periodic trước -> dùng base_interval_same; ngược lại base_interval_first.
        4) Silent window: nếu vừa đổi tone trong cửa sổ -> SKIP periodic.
        """
        # Step 1: welcome
        welcome_path = os.path.join(self.voices_dir, "welcome.mp3")
        if _mp3_exists(welcome_path):
            self._enqueue(welcome_path, reason="welcome")
        else:
            print("Audio not found")
        # đợi 2s (không block worker)
        start_ts = time.time()
        while not self._stop_flag.is_set() and (time.time() - start_ts) < 2.0:
            time.sleep(0.05)

        # First periodic
        with self._lock:
            self._next_play_ts = time.time() + self.base_interval_first
            self._last_periodic_tone = None

        # Main loop
        while not self._stop_flag.is_set():
            now = time.time()
            with self._lock:
                if now >= self._next_play_ts:
                    # Silent window sau lần đổi tone gần nhất -> bỏ periodic
                    if (now - self._last_tone_change_ts) < self.no_play_if_recent_change_sec:
                        interval = self.base_interval_same if (self._last_periodic_tone == self.current_tone) \
                                   else self.base_interval_first
                        self._next_play_ts = max(now + interval, now + self._edge_cooldown_sec)
                    else:
                        tone = self.current_tone
                        idx = self._cycle[tone]
                        path = self._tone_file(tone, idx)
                        self._enqueue(path, reason="periodic")

                        # 1->2->3->1
                        self._cycle[tone] = 1 if idx >= 3 else (idx + 1)

                        # chọn interval theo việc giữ tone
                        interval = self.base_interval_same if (self._last_periodic_tone == tone) \
                                   else self.base_interval_first
                        self._last_periodic_tone = tone
                        self._next_play_ts = max(now + interval, now + self._edge_cooldown_sec)
            time.sleep(0.05)

    def _tone_file(self, tone: str, index: int) -> str:
        return os.path.join(self.voices_dir, f"{tone}_{index}.mp3")

# -------- Module-level API --------
_player: VoicePlayer | None = None

def init(voices_dir: str,
         base_interval_first: float = 6.0,
         base_interval_same: float  = 5.0):
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

def stop():
    global _player
    if _player:
        _player.stop()
        _player = None
