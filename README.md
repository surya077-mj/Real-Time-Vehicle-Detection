# Real-Time Vehicle Detection 🚗🎯

This project performs **real-time vehicle detection and tracking** using **YOLOv5** for object detection and **DeepSort** for multi-object tracking.

It detects vehicles from live video or webcam streams and assigns persistent IDs for each vehicle moving through the frame.

---

## 🧠 Tech Stack

- Python 🐍
- YOLOv5 🚀 (Ultralytics)
- DeepSort 📍 (Tracking)
- OpenCV 🎥
- Torch / CUDA 🔥

---

## 🎥 Demo

> 📽️ *[Insert YouTube/GIF link if available]*  
> Example: "Watch the real-time vehicle tracking demo in action!"

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/surya077-mj/Real-Time-Vehicle-Detection.git
cd Real-Time-Vehicle-Detection

# Install dependencies
pip install -r requirements.txt

# Download YOLOv5 weights
wget https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt
🚀 Usage
bash
Copy
Edit
# Run on webcam
python track.py --source 0 --yolo_model yolov5s.pt --img 640

# Run on video file
python track.py --source path/to/video.mp4 --yolo_model yolov5s.pt --img 640
--source: Input source (0 = webcam, or path to video)

--img: Input image size (default = 640)

##🙌 Acknowledgements
Ultralytics YOLOv5

nwojke/deep_sort

