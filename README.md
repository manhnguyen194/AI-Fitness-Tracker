# AI Fitness Tracker
Realtime exercise tracking & video analysis powered by YOLO11 Pose and OpenCV.

---

## Features
- ***Realtime webcam tracking:*** Xem skeleton, số reps, góc khớp, và feedback trực tiếp. 
- ***Video upload analysis:*** Phân tích video lớn theo 3 phần, preview phần đầu ngay, phần còn lại xử lý background.
- ***4 exercises supported:*** Squat, Push-up, Sit-up, Plank.
- ***User-friendly UI:*** Chọn bài tập, model, resolution ngay trong trình duyệt qua Gradio.
- ***Lightweight YOLO11 Pose model*** (`yolo11n-pose.pt`) → tốc độ inference nhanh, hỗ trợ GPU nếu có.
- ***FPS & form feedback overlay*** trên video và webcam.

---

## Folder Structure
```angular2html
AI-Fitness-Tracker/
│
├─ src/                          # Source code
│   ├─ main.py                   # Entry point (optional, gọi pose_extractor / gradio_app_demo)
│   ├─ pose_extractor.py         # Core pose extraction & rep counter
│   ├─ gradio_app_demo.py        # Gradio UI + webcam & video handling
│   ├─ rep_counter.py             # Logic đếm reps + stage
│   ├─ form_rules.py              # Logic đánh giá form
│   ├─ voice_player.py            # Voice feedback handler
│   └─ utils/                     # Helper modules
│       ├─ draw_utils.py          # Vẽ text, skeleton
│       ├─ video_utils.py         # FPS calculation, window setup
│       ├─ angle_utils.py
│       ├─ feedback_utils.py
│       └─ geometry.py
│
├─ data/                          # User & raw data
│   ├─ raw/                       # Sample videos
│   └─ voices/                    # Voice feedback files
│
├─ models/                        # Trained weights
│   └─ yolo11n-pose.pt            # Default model weights
│
├─ fonts/                         # Fonts for overlay text
│   └─ Roboto.ttf
│
├─ requirements.txt
│
└─ README.md
```

---

## Installation
1. **Clone repository**

```bash
git clone https://github.com/manhnguyen194/AI-Fitness-Tracker.git
cd AI-Fitness-Tracker
```

2. **Install Miniconda**
- Download Miniconda from the official website.
- During installation, check the first two options:
  - Add Miniconda to my PATH
  - Register Miniconda as default Python
3. **Open Conda PowerShell**
- Open the Start Menu → search for “Conda PowerShell Prompt” → open it.
4. **Create and activate the environment**
```bash
conda create -n ai_fitness python=3.11
```
- Press `y` when prompted.
```bash
conda activate ai_fitness
```
6. **Install dependencies**
- **GPU (CUDA 12.1) recommended:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
- **CPU only:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```
- **Other required libraries** (from `requirements.txt`):
```bash
pip install -r requirements.txt
```
4. **Verify GPU (optional)**
```angular2html
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")
```

## Running the App
**Navigate to the project directory**
```bash
cd path/to/AI-Fitness-Tracker
```
**Gradio Web Interface**
```angular2html
python src/gradio_app_demo.py
```
- Open your browser at `http://localhost:7860`
- Features:
  - Launch external webcam window
  - Upload video for 3-part processing (first part available immediately)
  - View final annotated video when processing is done

## Tips & Notes
- **Webcam window** shows live pose skeleton + rep counter + feedback
- **Video analysis** returns first part quickly, remaining parts processed in background
- **Voice feedback** is configurable via `data/voices/`
- Recommended resolution for output: **1920x1080**
- Close webcam window or press `q` to stop the thread safely
