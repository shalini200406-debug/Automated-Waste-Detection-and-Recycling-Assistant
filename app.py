import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

st.title("♻️ Automated Waste Detection and Recycling Assistant")

model = tf.keras.models.load_model("your_model.h5")

class_names = ["plastic", "glass", "paper", "metal", "organic"]

uploaded_file = st.file_uploader("Upload an image of waste", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    img_resized = cv2.resize(img, (128, 128))
    img_resized = np.expand_dims(img_resized, axis=0)
    prediction = model.predict(img_resized)
    result = class_names[np.argmax(prediction)]

    st.success(f"Predicted Waste Type: {result}")
