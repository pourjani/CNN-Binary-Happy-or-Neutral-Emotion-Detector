# Facial Emotion Recognition (Happy vs Neutral)

## 📌 Project Overview
This project implements a deep learning system designed to classify facial expressions into two primary categories: **Happy** and **Neutral**. Using a custom-built Convolutional Neural Network (CNN), the model processes 48x48 grayscale images to provide fast and efficient emotion detection.

---

## 📊 Dataset Details
The model is trained on the **MUXSPACE Facial Expressions** dataset.
- **Source:** [GitHub - muxspace/facial_expressions](https://github.com/muxspace/facial_expressions)
- **Total Samples:** Approximately 13,690 images.
- **Format:** Grayscale images with corresponding labels in `legend.csv`.

---

## 🎯 Motivation & Challenges

### Why Binary Classification?
While the original dataset contains multiple emotion labels, this project focuses on **Binary Classification** (Happy vs. Neutral) for the following reasons:

1. **Dataset Size Constraints:** Although the dataset has ~13k images, this is considered **relatively small** for robust multi-class deep learning (which typically requires tens of thousands of samples per class).
2. **Class Imbalance:** A significant majority of the data is concentrated in the "Happy" and "Neutral" categories. Other emotions (like anger or disgust) have too few samples for the model to learn effectively.
3. **Label Noise:** The dataset contained inconsistent labeling (e.g., `happy` vs `HAPPINESS`). By focusing on two classes, we could perform rigorous data cleaning to ensure high model reliability.

### Why Softmax instead of Sigmoid?
Despite being a binary task, the model uses **Softmax activation with two output neurons**. This approach:
- Allows the model to output a clear probability distribution across the two classes.
- Makes the architecture easily scalable for future multi-class expansions.
- Works seamlessly with One-Hot encoding and Categorical Cross-Entropy loss.

---

## 🛠️ Data Preparation & Cleaning
To ensure high accuracy, the following preprocessing steps were taken:
1. **Label Normalization:** Standardized all variations of labels (e.g., merging "happiness" into "happy").
2. **Filtering:** Extracted only the "Happy" and "Neutral" samples.
3. **Image Processing:** Images were converted to grayscale, resized to 48x48 pixels, and normalized to a [0, 1] range.

**Final Dataset Distribution:**
| Emotion | Count | Percentage |
|---------|-------|------------|
| Neutral | 6,868 | 54.7%      |
| Happy   | 5,696 | 45.3%      |

---

## 🧠 Model Architecture
A custom, lightweight CNN was designed to extract features without the overhead of massive architectures. 
- **Transfer Learning Avoidance:** Large pre-trained models (like VGG or ResNet) were intentionally avoided to prevent **overfitting**, as the current dataset size is insufficient to fine-tune hundreds of millions of parameters effectively.

---
