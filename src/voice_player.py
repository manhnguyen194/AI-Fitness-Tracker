# # src/voice_player.py
# """
# Periodic voice player (MP3-first).
# - Mỗi `interval` giây phát file theo `current_tone`: positive / neutral / negative.
# - Ưu tiên .mp3 (kể cả dạng *_voice_mp3.mp3), fallback .wav.
# - Windows: WAV dùng winsound (non-blocking với SND_ASYNC); MP3/WAV dùng playsound nếu có.
# """
# import os
# import threading
# import sys
#
# # Winsound chỉ phát WAV (Windows)
# _HAVE_WINSOUND = False
# try:
#     if sys.platform.startswith("win"):
#         import winsound
#         _HAVE_WINSOUND = True
# except Exception:
#     _HAVE_WINSOUND = False
#
# # playsound để phát MP3/WAV (pip install playsound==1.3.0)
# try:
#     from playsound import playsound
#     _HAVE_PLAYSOUND = True
# except Exception:
#     _HAVE_PLAYSOUND = False
#
#
# def _resolve_path(voices_dir: str, tone: str) -> str | None:
#     """Tìm file theo ưu tiên MP3 trước, rồi WAV."""
#     tone = (tone or "").lower()
#     candidates = [
#         f"{tone}_voice_mp3.mp3",   # tên bạn đang dùng
#         f"{tone}_voice.mp3",
#         f"{tone}.mp3",
#         f"{tone}-voice.mp3",
#         # WAV fallbacks
#         f"{tone}_voice.wav",
#         f"{tone}_voice_pcm.wav",
#         f"{tone}.wav",
#     ]
#     for name in candidates:
#         path = os.path.join(voices_dir, name)
#         if os.path.exists(path):
#             return path
#     return None
#
#
# class VoicePlayer:
#     def __init__(self, voices_dir: str, interval: float = 3.0, allow_overlap: bool = False):
#         self.voices_dir = voices_dir
#         self.interval = float(interval)
#         self.allow_overlap = bool(allow_overlap)
#         self.current_tone = "neutral"
#         self._stop_flag = threading.Event()
#         self._thread = None
#
#     def set_tone(self, tone: str):
#         t = (tone or "").lower()
#         if t in ("positive", "neutral", "negative"):
#             self.current_tone = t
#
#     def _play_path(self, path: str):
#         if not path or not os.path.exists(path):
#             return False, f"Not found: {path}"
#         try:
#             ext = os.path.splitext(path)[1].lower()
#
#             # WAV trên Windows → winsound (non-blocking khi dùng SND_ASYNC)
#             if _HAVE_WINSOUND and ext == ".wav":
#                 if not self.allow_overlap:
#                     try:
#                         winsound.PlaySound(None, winsound.SND_PURGE)
#                     except Exception:
#                         pass
#                 winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
#                 return True, None
#
#             # playsound: MP3/WAV (blocking) → chạy thread riêng để không khóa loop
#             if _HAVE_PLAYSOUND:
#                 threading.Thread(target=playsound, args=(path,), daemon=True).start()
#                 return True, None
#
#             # Fallback: mở bằng app mặc định hệ điều hành
#             if sys.platform.startswith("win"):
#                 os.startfile(path)
#             else:
#                 opener = "open" if sys.platform == "darwin" else "xdg-open"
#                 import subprocess
#                 subprocess.Popen([opener, path],
#                                  stdout=subprocess.DEVNULL,
#                                  stderr=subprocess.DEVNULL)
#             return True, None
#
#         except Exception as e:
#             return False, str(e)
#
#     def _loop(self):
#         while not self._stop_flag.is_set():
#             tone = self.current_tone or "neutral"
#             path = _resolve_path(self.voices_dir, tone)
#             if path:
#                 ok, err = self._play_path(path)
#                 if ok:
#                     print(f"[VoicePlayer] ▶️ {tone} -> {os.path.basename(path)}")
#                 else:
#                     print(f"[VoicePlayer] ⚠️ Lỗi phát '{tone}': {err}")
#             else:
#                 print(f"[VoicePlayer] ⚠️ Không thấy file cho tone '{tone}' trong {self.voices_dir}")
#             self._stop_flag.wait(self.interval)
#
#     def start(self):
#         if getattr(self, "_thread", None) and self._thread.is_alive():
#             return
#         self._stop_flag = threading.Event()
#         self._thread = threading.Thread(target=self._loop, daemon=True)
#         self._thread.start()
#         print(f"[VoicePlayer] Bắt đầu (interval={self.interval}s)")
#
#     def stop(self):
#         self._stop_flag.set()
#         if getattr(self, "_thread", None):
#             self._thread.join(timeout=1.0)
#         print("[VoicePlayer] Đã dừng")
#
# # API mức module
# _player = None
#
# def init(voices_dir: str, interval: float = 3.0, allow_overlap: bool = False):
#     global _player
#     if _player:
#         _player.stop()
#     _player = VoicePlayer(voices_dir, interval=interval, allow_overlap=allow_overlap)
#     _player.start()
#     return _player
#
# def set_tone(tone: str):
#     if _player:
#         _player.set_tone(tone)
#
# def stop():
#     global _player
#     if _player:
#         _player.stop()
#         _player = None












# src/voice_player.py
"""
AI-Fitness-Tracker Voice Player (MP3-only)

Features:
- Plays 'welcome.mp3' once on init, waits 2s, then starts tone-driven logic.
- Periodic logic:
    * First play: every 5s based on current tone.
    * If the same tone is maintained, subsequent plays are every 4s.
    * For each tone, plays files in order: <tone>_1.mp3 -> _2.mp3 -> _3.mp3 -> _1...
- Edge trigger:
    * When tone changes, play immediately <tone>_1.mp3.
    * After that, next play won't occur for at least 2s to avoid overlap.
- MP3 only. If a needed file is missing, prints: "Audio not found".
- voices_dir example:
    C:\\Users\\Admin\\Downloads\\AI-Fitness-Tracker\\src\\data\\voices

Public API:
    init(voices_dir, base_interval_first=5.0, base_interval_same=4.0)
    set_tone(tone)  # "positive" | "neutral" | "negative"
    stop()
"""

import os
import sys
import time
import threading
import subprocess

# Optional MP3 playback backend: playsound (blocking). We'll call it in a thread.
try:
    from playsound import playsound  # pip install playsound==1.3.0
    _HAVE_PLAYSOUND = True
except Exception:
    _HAVE_PLAYSOUND = False

VALID_TONES = ("positive", "neutral", "negative")

def _mp3_exists(path: str) -> bool:
    return bool(path) and path.lower().endswith(".mp3") and os.path.exists(path)

def _play_mp3_async(path: str):
    """
    Play MP3 without blocking the main loop.
    On Windows, avoid playsound/MCI to prevent Error 305 -> use default app (os.startfile / PowerShell).
    On macOS/Linux, use 'open' / 'xdg-open'.
    """
    if not _mp3_exists(path):
        print("Audio not found")
        return

    def _runner():
        try:
            if sys.platform.startswith("win"):
                # Try startfile (default associated player). This is non-blocking for us.
                try:
                    os.startfile(path)  # type: ignore[attr-defined]
                except Exception:
                    # Hidden window via PowerShell as a fallback
                    quoted = '"' + path + '"'
                    cmd = ['powershell', '-NoProfile', '-WindowStyle', 'Hidden',
                           '-Command', f'Start-Process -WindowStyle Hidden {quoted}']
                    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                opener = "open" if sys.platform == "darwin" else "xdg-open"
                subprocess.Popen([opener, path],
                                 stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
        except Exception:
            print("Audio not found")

    threading.Thread(target=_runner, daemon=True).start()


class VoicePlayer:
    def __init__(
        self,
        voices_dir: str,
        base_interval_first: float = 6.0,   # first interval when (re)entering a tone
        base_interval_same: float = 5.0,    # subsequent interval while staying in same tone
        tone_change_debounce_ms: int = 900,  # NEW: yêu cầu tone mới ổn định tối thiểu 0.9s
        require_stable_frames: int = 6,  # NEW: và ổn định tối thiểu 6 frames liên tiếp
        edge_cooldown_sec: float = 2.5,  # NEW: sau edge, chờ >=2.5s mới cho phát tiếp
    ):
        self.voices_dir = voices_dir
        self.base_interval_first = float(base_interval_first)
        self.base_interval_same = float(base_interval_same)

        # Tone state
        self.current_tone: str = "neutral"
        self._last_periodic_tone: str | None = None

        # Per-tone cycle index (next to play): 1..3
        self._cycle = {t: 1 for t in VALID_TONES}

        # Scheduler state
        self._stop_flag = threading.Event()
        self._thread: threading.Thread | None = None
        self._next_play_ts: float = float("inf")  # when periodic is allowed next
        self._edge_cooldown_sec = 3.0             # after edge play, block next for >=2s

        # Internal guard to avoid overlap
        self._lock = threading.Lock()

        # Welcome
        self._did_welcome = False

        # --- NEW: cấu hình debounce / cooldown ---
        self.tone_change_debounce_ms = int(tone_change_debounce_ms)
        self.require_stable_frames = int(require_stable_frames)
        self._edge_cooldown_sec = float(edge_cooldown_sec)

        # --- NEW: trạng thái pending (chờ xác nhận) ---
        self._pending_tone: str | None = None
        self._pending_since_ts: float | None = None
        self._pending_count: int = 0

    # ---------- Public ----------
    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_flag.clear()
        self._thread = threading.Thread(target=self._main_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_flag.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    def set_tone(self, tone: str):
        """
        Debounce đổi tone:
          - Chỉ chấp nhận đổi nếu tone mới ổn định >= tone_change_debounce_ms
            VÀ xuất hiện liên tiếp >= require_stable_frames.
          - Khi chấp nhận, phát ngay <tone>_1 (edge-change), sau đó chặn phát tiếp >= edge_cooldown_sec.
        """
        t = (tone or "").lower()
        if t not in ("positive", "neutral", "negative"):
            return

        now = time.time()
        with self._lock:
            # Nếu giống hệt current_tone -> xoá pending
            if t == self.current_tone:
                self._pending_tone = None
                self._pending_since_ts = None
                self._pending_count = 0
                return

            # Nếu khác current_tone:
            if self._pending_tone != t:
                # bắt đầu một pending mới
                self._pending_tone = t
                self._pending_since_ts = now
                self._pending_count = 1
                return
            else:
                # vẫn cùng pending_tone -> tăng đếm & kiểm tra thời gian
                self._pending_count += 1
                elapsed_ms = (now - (self._pending_since_ts or now)) * 1000.0
                if (elapsed_ms < self.tone_change_debounce_ms) or (self._pending_count < self.require_stable_frames):
                    return  # chưa đủ điều kiện

                # === ĐỦ điều kiện: chấp nhận đổi tone ===
                new_tone = t
                self.current_tone = new_tone
                self._pending_tone = None
                self._pending_since_ts = None
                self._pending_count = 0

                # reset vòng phát cho tone mới: bắt đầu từ _1
                self._cycle[new_tone] = 1
                self._play_tone_clip_now(new_tone, index=1, reason="edge-change")
                self._cycle[new_tone] = 2  # lần tiếp theo của tone này

                # chặn phát kế tiếp tối thiểu edge_cooldown_sec
                self._next_play_ts = max(self._next_play_ts, now + self._edge_cooldown_sec)
                self._last_periodic_tone = new_tone

    # ---------- Internal ----------
    def _main_loop(self):
        """
        1) Play welcome once if available, then wait 2s.
        2) Schedule first periodic at now + 5s.
        3) If tone stays the same since last periodic, use 4s; else 5s.
        """
        # Step 1: welcome
        welcome_path = os.path.join(self.voices_dir, "welcome.mp3")
        if _mp3_exists(welcome_path):
            _play_mp3_async(welcome_path)
        else:
            # Not a hard error; just inform
            print("Audio not found")
        self._did_welcome = True

        # Step 2: wait 2s before tone-driven logic
        start_ts = time.time()
        while not self._stop_flag.is_set() and (time.time() - start_ts) < 2.0:
            time.sleep(0.05)

        # First periodic in 5s
        with self._lock:
            self._next_play_ts = time.time() + self.base_interval_first
            self._last_periodic_tone = None

        # Main loop
        while not self._stop_flag.is_set():
            now = time.time()
            with self._lock:
                if now >= self._next_play_ts:
                    tone = self.current_tone

                    # Decide which index to play for this tone
                    idx = self._cycle[tone]
                    self._play_tone_clip_now(tone, idx, reason="periodic")

                    # Advance cycle for this tone (1->2->3->1)
                    self._cycle[tone] = 1 if idx >= 3 else (idx + 1)

                    # Next interval: 4s if tone unchanged since last periodic; else 5s
                    if self._last_periodic_tone == tone:
                        interval = self.base_interval_same  # 4s
                    else:
                        interval = self.base_interval_first # 5s

                    self._last_periodic_tone = tone
                    # Ensure next play at least after edge-cooldown if something just played
                    self._next_play_ts = max(now + interval, now + self._edge_cooldown_sec)
            # Sleep a little to avoid busy wait
            time.sleep(0.05)

    def _tone_file(self, tone: str, index: int) -> str:
        """
        Build the exact filename <tone>_<index>.mp3 in voices_dir.
        """
        name = f"{tone}_{index}.mp3"
        return os.path.join(self.voices_dir, name)

    def _play_tone_clip_now(self, tone: str, index: int, reason: str):
        """
        Try to play a specific tone clip now. If missing -> print "Audio not found".
        """
        path = self._tone_file(tone, index)
        if _mp3_exists(path):
            _play_mp3_async(path)
            print(f"[VoicePlayer] play {tone}_{index} ({reason})")
        else:
            print("Audio not found")

# -------- Module-level API --------
_player: VoicePlayer | None = None

def init(voices_dir: str,
         base_interval_first: float = 5.0,
         base_interval_same: float = 4.0):
    """
    Init the player and start its background loop.
    - base_interval_first: seconds for first periodic play when entering a tone (default 5s)
    - base_interval_same: seconds for subsequent plays while staying in the same tone (default 4s)
    """
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
