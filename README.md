# 😷 Face Mask Detection & Face Recognition System

This project is a real-time system that both identifies individuals using facial recognition and analyzes whether they are wearing a face mask. It is developed to facilitate health compliance monitoring in environments such as public buildings or institutions. The system integrates camera, artificial intelligence, and user interface components.

---

## 📌 Core Features

### 🎯 Objectives:

- Facial recognition and identity matching
- Mask detection
- User registration and login functionality
- Logging of unmasked individuals
- User-friendly desktop GUI

### 🔒 Security & Monitoring:

- Unmasked individuals are automatically logged with **timestamp and name**
- The system saves relevant data and images to local storage automatically

---

## 🧠 System Architecture

### 1. Face Detection
- Utilizes **OpenCV DNN (Deep Neural Network)** module
- Uses `deploy.prototxt` and `res10_300x300_ssd_iter_140000.caffemodel`
- Detects face coordinates in real-time camera frames

### 2. Face Recognition
- Employs **Keras-FaceNet** model
- Generates 128-dimensional embeddings from both masked and unmasked face samples
- Matches incoming embeddings against stored `.pkl` files under `users/`
- Recognition threshold: **Euclidean distance < 0.8**

### 3. Mask Detection
- Based on a custom **MobileNetV2** deep learning classifier
- Classes:
  - `with_mask`
  - `without_mask`
- Applies **data augmentation** during training:
  - Rotation, zoom, shift, horizontal flip
- Output: 2-class softmax prediction with confidence percentages

### 4. Graphical Interface (GUI with PyQt5)
- Buttons: **Register**, **Login**, **Logout**
- Displays real-time camera feed
- Shows recognized user name and mask status live on screen
- Colored borders:
  - Green = masked
  - Red = unmasked

### 5. User Registration
- User enters their name
- The system captures two face images: one with mask, one without
- Extracted embeddings are saved to `.pkl` under `users/`
- Having both masked/unmasked references improves recognition accuracy

### 6. Logging System
- When a known user is detected **without a mask**:
  - Logs name, date, time, and cropped face image to `.csv` and `.jpg`
  - Prevents repeated logs using time-based caching
- Logs are generated automatically, no manual action required

---

## 📚 Model Training Details

### Training Command:
```bash
python train_mask_detector.py --dataset dataset/
```

### Dataset Structure:
```
dataset/
├── with_mask/
│   └── img1.jpg, img2.jpg, ...
├── without_mask/
    └── img1.jpg, img2.jpg, ...
```

### Training Configuration:
- Model: MobileNetV2 (`include_top=False`)
- Input shape: `224x224x3`
- Layers:
  - AveragePooling2D
  - Flatten
  - Dense (128) + ReLU
  - Dropout (0.5)
  - Dense (2) + Softmax
- Optimizer: Adam (LR = 1e-4)
- Loss: Binary Crossentropy
- Epochs: 20
- Batch size: 32
- Augmentation: rotation, zoom, shift, shear, flip
- Outputs:
  - `mask_detector_model.h5` (trained model)
  - `plot.png` (accuracy/loss graph)
  - Console classification report (precision, recall, F1)

---

## 🚀 Running the Application

### Installation:
```bash
pip install -r requirements.txt
```

### Launch GUI:
```bash
python gui.py
```

> Once launched, the GUI activates your webcam. If the user is recognized and their mask status is identified, this is displayed in real time. If the user is unmasked, a log entry is created automatically.

---

## 💡 Key Functionalities Overview

| Feature              | Description |
|----------------------|-------------|
| 👤 Face Recognition   | Recognizes individuals using FaceNet, even when masked |
| 😷 Mask Detection     | MobileNetV2 classifier with high accuracy |
| 💾 Registration       | Captures both masked & unmasked samples |
| 📊 Logging            | Logs unmasked users with timestamp and image |
| 🧠 Model Training     | Custom dataset can be used for training |
| 🖥️ PyQt5 Interface     | Real-time display with interactive GUI |

---

## 🛠️ Future Enhancements

- Support for improper mask use (e.g., nose exposed)
- Multi-face detection and tracking
- Remote/cloud-based logging support
- Camera calibration and low-light improvement
- Export to ONNX for embedded device deployment

---

This project merges artificial intelligence, computer vision, and real-time interaction into a practical and scalable health compliance tool. It can be easily adapted and integrated into any environment requiring identity verification and safety enforcement.

