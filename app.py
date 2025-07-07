import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import pickle
import cv2

# Load model and class indices
model = load_model("best_brain_tumor_model.h5")

with open("class_indices.pkl", "rb") as f:
    class_indices = pickle.load(f)

# Reverse mapping: index to class name
idx_to_class = {v: k for k, v in class_indices.items()}

# Image preprocessing
def preprocess_image(image, target_size=(224, 224)):
    image = image.convert("RGB")
    img = np.array(image)
    img = cv2.resize(img, target_size)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Streamlit UI
st.set_page_config(page_title="Brain Tumor MRI Classifier", layout="centered")

st.title("🧠 Brain Tumor MRI Classification")
st.write("Upload an MRI brain scan image and the model will predict the tumor type (if any).")

uploaded_file = st.file_uploader("📤 Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    if st.button("🧪 Predict"):
        with st.spinner("Analyzing the image..."):
            processed_img = preprocess_image(image)
            predictions = model.predict(processed_img)
            predicted_class = idx_to_class[np.argmax(predictions)]
            confidence_scores = predictions[0]

        st.success(f"✅ Predicted Tumor Type: **{predicted_class.upper()}**")
        
        st.subheader("📊 Confidence Scores:")
        for class_name, score in zip(class_indices.keys(), confidence_scores):
            st.write(f"- {class_name.title()}: `{score*100:.2f}%`")
