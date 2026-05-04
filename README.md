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
## ❓ Why Only Two Emotions (Happy vs Neutral)?

Although the original dataset includes multiple emotion categories, this project intentionally focuses on **only two emotional states: Happy and Neutral**.  
This decision was made based on both **data-driven analysis** and **practical engineering considerations**, not simplification.

### Key Reasons:

### 1️⃣ Dataset Size & Class Distribution
The dataset contains approximately 13,000 images in total. While this may seem sufficient, it is **relatively small for reliable multi-class deep learning**, especially when some emotion classes contain very few samples.

In contrast, **Happy** and **Neutral** are the two most frequent and well-represented classes, providing enough data to train a stable and generalizable model.

---

### 2️⃣ Severe Class Imbalance in Other Emotions
Other emotions such as *anger, fear, sadness,* and *disgust* are significantly underrepresented. Training a multi-class classifier under these conditions often leads to:
- Biased predictions toward majority classes  
- Poor recall for minority emotions  
- Unstable training and misleading accuracy metrics  

Restricting the problem to two dominant classes helps avoid these pitfalls.

---

### 3️⃣ Label Noise & Annotation Inconsistency
The dataset contains inconsistencies such as:
- Variations in label naming (e.g., `happy`, `HAPPINESS`)
- Ambiguous facial expressions that overlap between emotional categories

By narrowing the task to Happy vs Neutral, label ambiguity is significantly reduced, resulting in **cleaner supervision and more reliable learning**.

---

### 4️⃣ Real‑World Practicality
In many real-world applications (e.g., user engagement analysis, attention detection, basic mood estimation), distinguishing between:
- **Neutral (no strong emotion)** and  
- **Happy (positive emotion)**  

is often sufficient and more robust than attempting to classify subtle emotional differences.

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
