# BrainTumor_classification

# 🧠 Brain Tumor MRI Image Classification

This project uses deep learning to classify brain MRI images into four categories: **Glioma**, **Meningioma**, **Pituitary**, and **No Tumor**. It leverages both a custom-built Convolutional Neural Network (CNN) and pretrained transfer learning models (e.g., ResNet50, MobileNetV2, EfficientNetB0) to achieve reliable classification accuracy.

---

## 📌 Project Objective

To develop an image classification system that can automatically detect and classify types of brain tumors using MRI scans — assisting radiologists and reducing diagnostic errors through machine intelligence.

---

## Dataset

The dataset includes **2,443 labeled MRI images** distributed across 4 categories:

- **Glioma Tumor**
- **Meningioma Tumor**
- **Pituitary Tumor**
- **No Tumor**

### Directory Structure

dataset/
├── train/
├── val/
└── test/

Training: 1695 images

Validation: 502 images

Testing: 246 images

🔄 Project Workflow
1. 📊 Dataset Understanding
Checked class distribution and sample images.

Ensured image resolution consistency.

Verified dataset was already split into train/val/test.

2. 🧼 Data Preprocessing
Normalized image pixel values to [0, 1].

Resized all images to 224x224.

3. 📈 Data Augmentation
Applied rotation, zoom, brightness, flipping, and shifting to avoid overfitting.

4. 🧠 Model Building
✅ Custom CNN
Built from scratch with:

3 convolutional layers

MaxPooling

Batch Normalization

Dropout and Dense layers

✅ Transfer Learning Models
Used pretrained models from ImageNet:

ResNet50

MobileNetV2

InceptionV3

EfficientNetB0

Each model’s top layers were removed and replaced with new classification heads for 4 categories.

5. ⚙️ Model Training
Used EarlyStopping and ModelCheckpoint.

Monitored validation loss and accuracy.

6. 📊 Model Evaluation
Evaluated on test set using:

Accuracy

Precision

Recall

F1-Score

Confusion Matrix

7. 📊 Model Comparison

.

🚀 Streamlit Web App
An interactive UI allows users to upload MRI images and get real-time tumor predictions.

Features:
Upload .jpg, .jpeg, .png brain scan

View predicted tumor type

See model confidence for each class

## Files in Repo
File	Description
best_brain_tumor_model.h5	Final trained model for deployment
class_indices.pkl	Mapping of labels to class indices
app.py	Streamlit prediction app
notebook.ipynb / .py	Training and model comparison code
README.md	Project summary

🛠️ Tech Stack
Python 🐍

TensorFlow / Keras

OpenCV

Scikit-learn

Streamlit

Matplotlib / Seaborn

✅ Future Improvements
Real-time webcam classification

Grad-CAM visualization for interpretability

Deploy as API using FastAPI or Flask

Add support for DICOM format (medical standard)

🙋‍♂️ Author
Akash Siddharth


