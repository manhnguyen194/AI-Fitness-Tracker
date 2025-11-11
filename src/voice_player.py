# src/voice_player.py
"""
Minimal periodic voice player.
Mỗi `interval` giây sẽ phát file tương ứng với `current_tone`.
File phải có tên: positive_voice_pcm.wav, neutral_voice_pcm.wav, negative_voice_pcm.wav
(đã convert sang PCM 16-bit).
"""
import os
import threading
import time
import sys

# Prefer winsound on Windows (no external deps)
_HAVE_WINSOUND = False
try:
    if sys.platform.startswith("win"):
        import winsound
        _HAVE_WINSOUND = True
except Exception:
    _HAVE_WINSOUND = False

# Fallback: playsound (pip install playsound)
try:
    from playsound import playsound
    _HAVE_PLAYSOUND = True
except Exception:
    _HAVE_PLAYSOUND = False

class VoicePlayer:
    def __init__(self, voices_dir: str, interval: float = 3.0, allow_overlap: bool = False):
        self.voices_dir = voices_dir
        self.interval = float(interval)
        self.allow_overlap = bool(allow_overlap)
        # default tone
        self.current_tone = "neutral"
        self.voice_map = {
            "positive": os.path.join(voices_dir, "positive_voice_pcm.wav"),
            "neutral":  os.path.join(voices_dir, "neutral_voice_pcm.wav"),
            "negative": os.path.join(voices_dir, "negative_voice_pcm.wav"),
        }
        self._stop_flag = threading.Event()
        self._thread = None

    def set_tone(self, tone: str):
        t = (tone or "").lower()
        if t in ("positive", "neutral", "negative"):
            self.current_tone = t

    def _play_path(self, path):
        if not path or not os.path.exists(path):
            return False, f"Not found: {path}"
        try:
            if _HAVE_WINSOUND and path.lower().endswith(".wav"):
                if not self.allow_overlap:
                    try:
                        winsound.PlaySound(None, winsound.SND_PURGE)
                    except Exception:
                        pass
                winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
                return True, None
            elif _HAVE_PLAYSOUND:
                # playsound blocks — spawn thread
                threading.Thread(target=playsound, args=(path,), daemon=True).start()
                return True, None
            else:
                # fallback open with default app (may show UI)
                if sys.platform.startswith("win"):
                    os.startfile(path)
                else:
                    opener = "open" if sys.platform == "darwin" else "xdg-open"
                    import subprocess
                    subprocess.Popen([opener, path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True, None
        except Exception as e:
            return False, str(e)

    def _loop(self):
        while not self._stop_flag.is_set():
            tone = self.current_tone or "neutral"
            path = self.voice_map.get(tone)
            if path:
                ok, err = self._play_path(path)
                if ok:
                    print(f"[VoicePlayer] ▶️ Phát tone '{tone}' -> {os.path.basename(path)}")
                else:
                    print(f"[VoicePlayer] ⚠️ Lỗi phát tone '{tone}': {err}")
            else:
                print(f"[VoicePlayer] ⚠️ Không có mapping cho tone '{tone}'")
            # sleep interval
            # use wait so stop() can break quickly
            self._stop_flag.wait(self.interval)

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_flag.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[VoicePlayer] Bắt đầu periodic player (interval={self.interval}s)")

    def stop(self):
        self._stop_flag.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        print("[VoicePlayer] Đã dừng player")

# module-level simple API
_player = None

def init(voices_dir: str, interval: float = 3.0, allow_overlap: bool = False):
    global _player
    if _player:
        _player.stop()
    _player = VoicePlayer(voices_dir, interval=interval, allow_overlap=allow_overlap)
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
