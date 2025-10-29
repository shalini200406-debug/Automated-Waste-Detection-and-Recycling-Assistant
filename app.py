import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

# Load model
model = tf.keras.models.load_model("your_model.h5")

# Class names
class_names = ['plastic', 'glass', 'paper', 'metal', 'organic']

st.title("♻️ Automated Waste Detection and Recycling Assistant")

uploaded_file = st.file_uploader("Upload an image of waste", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # ✅ Correct preprocessing for your model
    img_resized = cv2.resize(img, (128, 128))
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    img_flatten = img_gray.flatten() / 255.0
    img_array = np.expand_dims(img_flatten, axis=0)

    try:
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        st.success(f"✅ Predicted Waste Type: **{predicted_class}**")
    except Exception as e:
        st.error(f"⚠️ Error during prediction: {str(e)}")
