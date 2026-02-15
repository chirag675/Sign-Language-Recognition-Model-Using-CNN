# 🤟 Sign Language Recognition using CNN (Custom Dataset)

An end-to-end **Computer Vision + Deep Learning** project that recognizes hand gestures from static images and real-time webcam video using a **Convolutional Neural Network (CNN)** built with TensorFlow/Keras.

This project was developed as an interactive and research-oriented system to demonstrate practical applications of deep learning in accessibility technologies. The model is trained on a **self-curated and labeled dataset**, ensuring full control over data quality, preprocessing, and class balance.

---

## 🚀 Project Overview

Sign languages are essential communication systems for the deaf and hard-of-hearing community. However, communication barriers still exist between signers and non-signers.

This project addresses that challenge by building a **real-time Sign Language Recognition system** capable of:

- Detecting hands from images and live video streams  
- Classifying hand gestures using a trained CNN model  
- Providing real-time predictions  
- Generating performance metrics and evaluation reports  

The system integrates **Computer Vision** and **Deep Learning** into a deployable, interactive application.

---

## 🧠 Technical Architecture

The pipeline follows a structured ML workflow:

### 1️⃣ Data Collection (Self-Created Dataset)
- Captured hand gesture images manually  
- Created structured class folders  
- Ensured consistent background and lighting conditions  
- Applied augmentation techniques (rotation, flipping, zooming)  
- Balanced dataset across all gesture classes  

### 2️⃣ Preprocessing (OpenCV)
- Frame extraction from webcam  
- ROI (Region of Interest) cropping  
- Background noise reduction  
- Image resizing (e.g., 64x64 / 128x128)  
- Normalization (pixel scaling 0–1)  

### 3️⃣ Model Architecture (CNN - TensorFlow/Keras)

Typical architecture used:

    Conv2D → ReLU → MaxPooling  
    Conv2D → ReLU → MaxPooling  
    Conv2D → ReLU → MaxPooling  
    Flatten  
    Dense → ReLU  
    Dropout  
    Dense → Softmax (Output Layer)

Key Features:
- Multiple convolutional layers for feature extraction  
- Dropout for regularization  
- Softmax activation for multi-class classification  
- Categorical Crossentropy Loss  
- Adam Optimizer  

### 4️⃣ Training & Evaluation
- Train/Test split  
- Accuracy and Loss tracking  
- Confusion Matrix generation  
- Classification report  
- Model checkpoint saving  

### 5️⃣ Real-Time Prediction
- Live webcam input using OpenCV  
- Continuous frame processing  
- Gesture prediction overlay on video stream  
- Interactive prediction window  

---

## 🛠️ Tech Stack

- Python  
- TensorFlow / Keras  
- OpenCV  
- NumPy  
- Matplotlib  
- Scikit-learn  

---


## 📊 Model Performance


Evaluation metrics include:
- Accuracy  
- Precision  
- Recall  
- F1-Score  
- Confusion Matrix Visualization  

---

## 🎥 Real-Time Demo Features

- Live camera gesture recognition  
- Real-time classification overlay  
- Smooth prediction transitions  
- Interactive quit functionality  
- Lightweight and fast inference  

---



⭐ If you find this project useful, consider giving it a star on GitHub!
