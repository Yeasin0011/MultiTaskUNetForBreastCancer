# Breast Cancer Detection and Lesion Segmentation using Multi-Task Deep Learning

This repository contains a **PyTorch** implementation of a multi-task deep learning model designed to analyze breast ultrasound images for **lesion segmentation** and **classification**. The model is built using a **MultiTaskUNet** architecture, capable of performing both tasks simultaneously for more efficient and accurate results. The goal of this project is to demonstrate the power of deep learning in automating diagnostic tasks, assisting in early **breast cancer detection**.

## Overview

This project uses **deep learning** to analyze breast ultrasound images, performing both **lesion segmentation** and **classification**. By combining these two tasks, the model provides a comprehensive solution for diagnosing breast cancer based on ultrasound images. The segmentation part focuses on identifying the location of lesions, while the classification part determines whether the lesion is **Normal**, **Benign**, or **Malignant**.

### Key Features:
- **Multi-task learning** for simultaneous lesion segmentation and classification.
- **Custom preprocessing pipeline** to handle grayscale conversion, resizing, and normalization.
- **Data augmentation** for enhancing model generalization and reducing overfitting.
- **Class balancing** through computed class weights to address class imbalance in the dataset.
- Performance evaluation using metrics like **IoU**, **Accuracy**, and **F1-Score**.

---

## Technologies Used

- **PyTorch** - Deep learning framework for building and training the model.
- **NumPy** & **Pandas** - Data manipulation and array handling.
- **Matplotlib** & **Seaborn** - Data visualization and plotting.
- **skimage** - Image processing tools.
- **Scikit-learn** - For model evaluation and splitting the dataset.
- **tqdm** - Progress bar for loops during training.

---

## Dataset

The dataset used in this project is the **BUSI (Breast Ultrasound Images)** dataset, which contains ultrasound images for breast cancer detection. The dataset includes:
- **Images**: Grayscale images of breast ultrasound scans.
- **Masks**: Ground truth segmentation masks, indicating the tumor regions.
- **Labels**: Class labels for each image indicating whether the tumor is **Normal**, **Benign**, or **Malignant**.

### Dataset Structure:
The dataset is organized into the following directories:
