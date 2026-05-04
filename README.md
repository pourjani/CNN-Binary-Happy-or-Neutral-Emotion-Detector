# Facial Emotion Recognition (Happy vs Neutral)

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-90.2%25-brightgreen)
![Status](https://img.shields.io/badge/Status-Completed-success)

---

## 📌 Overview
This project is a deep‑learning based Facial Emotion Recognition (FER) system that classifies faces into two emotions: **Happy** and **Neutral**.  
A lightweight Convolutional Neural Network (CNN) was trained on grayscale 48×48 images to allow **fast inference and real‑time usage**.

---

## 📊 Dataset
Dataset Source: [GitHub - muxspace/facial_expressions](https://github.com/muxspace/facial_expressions)

### Distribution:
| Emotion | Count | Percentage |
|---------|-------|------------|
| Neutral | 6,868 | 54.7%      |
| Happy   | 5,696 | 45.3%      |

---

## 📈 Model Performance & Training Logs
The model was trained for **15 Epochs**. Below is the detailed performance log extracted from the training process:

| Epoch | Loss | Accuracy | Val Loss | Val Accuracy | Time/Step |
|-------|------|----------|----------|--------------|-----------|
| 1/15  | 0.4721 | 76.11% | 0.3768 | 82.69% | 31s 47ms |
| 5/15  | 0.2393 | 89.88% | 0.2735 | 88.10% | 33s 53ms |
| 10/15 | 0.1724 | 92.89% | 0.2726 | 90.05% | 33s 52ms |
| 15/15 | 0.1170 | **95.15%** | 0.2760 | **90.21%** | 32s 51ms |

### 🔑 Key Takeaways:
- **Final Validation Accuracy:** ~90.2%
- **Inference Speed:** Fast execution (~32s per epoch on a standard CPU/GPU setup).
- **Stability:** The model shows consistent improvement, reaching over 90% accuracy within just 10 epochs.

---

## 🎯 Motivation
The project was designed as a **Binary Classification** system to ensure high reliability. By focusing on two primary classes (Happy vs. Neutral), we achieved:
- **Higher Precision:** Reduced the noise caused by similar-looking subtle emotions.
- **Robustness:** Better performance in real-time scenarios with varying lighting conditions.

---

## 🧠 Model Architecture
A custom CNN was built from scratch. Transfer learning was avoided to:
1. Prevent overfitting on a relatively small dataset.
2. Keep the model lightweight for real-time webcam inference.

### Why Softmax?
Even for binary classification, **Softmax (2 neurons)** was used instead of Sigmoid. This allows:
- Direct output of probability scores for each class.
- Compatibility with categorical cross-entropy loss.
- Scalability for future multi-emotion expansion.

---
