# Face Mask Detection System

## 📌 Overview
A real-time face mask detection system that uses deep learning and computer vision to detect whether a person is wearing a face mask via webcam feed. This project demonstrates the application of transfer learning with MobileNetV2 for binary classification of masked and unmasked faces.

## ✨ Features
- **Real-time Detection**: Live webcam feed processing with face detection and mask classification
- **Deep Learning Model**: Utilizes MobileNetV2 for efficient and accurate mask detection
- **Visual Feedback**: Displays bounding boxes and confidence scores in real-time
- **Easy to Use**: Simple command-line interface for both training and inference
- **Lightweight**: Optimized for performance on standard hardware

## 🚀 Quick Start

### Prerequisites
- Python 3.6 or higher
- Webcam
- pip (Python package manager)

### Installation
1. **Clone the repository**
   ```bash
   git clone https://github.com/Joel-Shibu/Facemask_Detection
   cd facemask_detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Haar Cascade XML** (if not already present)
   ```bash
   wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
   ```

## 📂 Project Structure
```
facemask-detection/
├── dataset/                      # Training dataset
│   ├── with_mask/               # Images with masks
│   └── without_mask/            # Images without masks
├── model/                       # Saved models and training history
├── detect_mask_webcam.py        # Real-time detection script
├── mask_detector.py             # Model training script
├── haarcascade_frontalface_default.xml  # Haar Cascade classifier
└── requirements.txt             # Project dependencies
```

## 🛠️ Usage

### Training the Model
To train the model with your dataset:
```bash
python mask_detector.py --dataset dataset --model model/mask_detector.model --plot model/plot.png
```

### Running Real-time Detection
To start the face mask detection system:
```bash
python detect_mask_webcam.py --model model/mask_detector.model --face haarcascade_frontalface_default.xml
```

### Controls
- Press `q` to quit the application
- Ensure proper lighting conditions for best results
- Position your face clearly in the webcam frame

## 📊 Performance
- Model: MobileNetV2 (fine-tuned)
- Input Size: 224x224 pixels
- Framework: TensorFlow/Keras
- Face Detection: OpenCV Haar Cascade Classifier

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

