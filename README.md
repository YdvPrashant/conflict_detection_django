# 🛡️ Conflict Detection System using Django

This project is a real-time **Conflict and Weapon Detection System** built using Django and deep learning. It detects the number of people using YOLOv8 and further analyzes frames for violent behavior using a Swin Transformer model — helping automate surveillance and ensure public safety.

---

## 🔍 Features

- 🎯 Person detection using YOLOv8. (`yolov8x.pt`)
- 🧠 Conflict detection using Swin Transformer. (`swin_transformer_tiny_v3.pth`)
- 🖥️ Real-time video feed processing. (OpenCV)
- 🌐 Web interface built using Django templates.
- ⚠️ Automatically detects and flags potential human conflicts.

---

## 🛠️ Tech Stack

| Layer       | Tech Used                        |
|-------------|----------------------------------|
| Backend     | Django (Python)                  |
| Models      | YOLOv8, Swin Transformer         |
| Frontend    | HTML / CSS (Django Templates)    |
| Core Libs   | OpenCV, Torch, Ultralytics YOLO  |

---

## 🚀 Setup & Installation
  ```bash
git clone https://github.com/YdvPrashant/conflict_detection_system_django.git
cd conflict_detection_system_django
