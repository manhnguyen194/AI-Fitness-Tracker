# check_wav.py
import wave
from pathlib import Path

voices_dir = Path(r"C:\Users\Admin\Downloads\AI-Fitness-Tracker\src\data\voices")
files = ["positive_voice.wav", "neutral_voice.wav", "negative_voice.wav"]

for name in files:
    p = voices_dir / name
    print("=>", p)
    if not p.exists():
        print("   NOT FOUND")
        continue
    try:
        with wave.open(str(p), 'rb') as wf:
            nch = wf.getnchannels()
            sampw = wf.getsampwidth()
            fr = wf.getframerate()
            comptype = wf.getcomptype()
            compname = wf.getcompname()
            nframes = wf.getnframes()
            print(f"   channels={nch}, sampwidth={sampw}, fr={fr}, comptype={comptype}, compname={compname}, frames={nframes}")
            if comptype != 'NONE' or sampw != 2:
                print("   --> NOT standard PCM 16-bit (needs conversion).")
            else:
                print("   --> OK: PCM 16-bit WAV.")
    except wave.Error as e:
        print("   wave.Error:", e)
    except Exception as e:
        print("   other error:", e)
