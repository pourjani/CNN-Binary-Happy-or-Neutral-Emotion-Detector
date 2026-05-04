# Facial Emotion Recognition (Happy vs Neutral) - Binary Classification

## Project Overview

This project implements a deep learning-based facial emotion recognition system that classifies facial images into two primary emotional states: **Happy** and **Neutral**. The model architecture is a lightweight Convolutional Neural Network (CNN) trained from scratch on a curated dataset of grayscale 48x48 pixel face images.

---

## Dataset Source

The dataset used in this project is sourced from the publicly available repository:

[https://github.com/muxspace/facial_expressions](https://github.com/muxspace/facial_expressions)

This dataset contains labeled facial images along with metadata that annotates each image with the corresponding emotion.

---

## Motivation & Challenges

### Why Binary Classification?

The original dataset includes multiple emotional categories. However, a thorough inspection revealed several challenges:

- **Limited Dataset Size:** The dataset size is relatively small for robust multi-class deep learning classification.  
- **Label Imbalance:** The majority of samples belong predominantly to two classes: **Neutral** and **Happy**, with very few images representing other emotions.  
- **Label Noise and Inconsistency:** Variations in emotion labels (e.g., `happy`, `HAPPINESS`, `happy` in different letter cases) introduce noise and ambiguity.

Due to these reasons, the project focuses on a simplified **binary classification** task distinguishing between Happy and Neutral expressions. This approach enables:

- More effective training on the available data  
- Better class balance to avoid bias  
- Reduced noise impact through rigorous label cleaning and normalization

---

## Data Preparation & Processing

1. **Label Normalization:** Emotion labels were standardized (lowercased, synonymous labels merged — e.g., "happiness" combined into "happy").  
2. **Filtering:** Only samples labeled as "happy" and "neutral" were retained.  
3. **Image Preprocessing:** All images were converted to grayscale, resized to 48x48 pixels, and normalized to the [0,1] range for pixel intensities.

Resulting dataset size after cleaning:

| Class   | Number of Samples | Percentage of Total |
|---------|-------------------|--------------------|
| Neutral | 6,868             | 54.7%              |
| Happy   | 5,696             | 45.3%              |

---

## Model Architecture

A custom lightweight CNN was designed to effectively learn features from the relatively small dataset. Transfer learning with large pre-trained models was intentionally avoided due to:

- Dataset size limitation — insufficient data to fine-tune deep architectures without overfitting  
- Noise and imbalance which complicate transfer learning stability

---

## Installation

This project requires the following Python packages:

- `numpy`  
- `pandas`  
- `tensorflow` (or `tensorflow-gpu`)  
- `opencv-python`  
- `scikit-learn`  

