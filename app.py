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
    # Read and show image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess for model
    img_resized = cv2.resize(img, (128, 128))
    img_array = np.expand_dims(img_resized, axis=0)
    img_array = img_array / 255.0  # ✅ normalize

    # Predict
    try:
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        st.success(f"✅ Predicted Waste Type: **{predicted_class}**")
    except Exception as e:
        st.error(f"⚠️ Error during prediction: {str(e)}")
