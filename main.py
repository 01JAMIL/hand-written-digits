import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("handwritten_digit_recognition.keras")

# Streamlit app
st.title("Handwritten Digit Recognition")

st.write("Upload an image of a handwritten digit to get a prediction.")

# File uploader
uploaded_file = st.file_uploader("Choose a digit image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    image = np.array(image)
    
    # Resize and preprocess the image
    image_resized = cv2.resize(image, (28, 28))
    image_resized = cv2.bitwise_not(image_resized)  # Invert colors
    image_normalized = image_resized / 255.0

    # Display the image
    st.image(image_resized, caption="Processed Image", use_column_width=True)

    # Prepare for prediction
    image_normalized = np.expand_dims(image_normalized, axis=0)  # Add batch dimension
    
    # Make prediction
    prediction = model.predict(image_normalized)
    predicted_digit = np.argmax(prediction)

    st.write(f"Predicted Digit: **{predicted_digit}**")
    
    
    