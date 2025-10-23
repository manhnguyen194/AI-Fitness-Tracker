# AI Fitness Tracker — MVP Spec

## 1. Bài tập hỗ trợ
- Squat
- Push-up
- Plank

## 2. Chức năng
- Phát hiện tư thế người (pose keypoints)
- Đếm số lần lặp (reps)
- Cảnh báo form sai cơ bản
- Giao diện đơn giản qua Gradio

## 3. Tech Stack
- Pose Estimation: YOLO11-pose (Ultralytics)
- AI Pipeline: Python
- UI: Gradio (Blocks)

## 4. Output
- Video có overlay skeleton
- Counter reps (in text)
- Feedback: “OK” / “Sai tư thế”
