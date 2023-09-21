import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

# Load the trained model
model = tf.keras.models.load_model("mnist_model.h5")

st.title("MNIST Digit Recognition")

# Allow user to upload an image
uploaded_file = st.file_uploader("Upload a drawn digit", type="png")

if uploaded_file:
    image = Image.open(uploaded_file).convert("L")
    image = ImageOps.invert(image)
    image = image.resize((28, 28))

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image and predict
    image = np.array(image) / 255.0
    image = image.reshape(1, 28, 28)
    prediction = model.predict(image)

    st.write(f"Predicted Digit: {np.argmax(prediction)}")
    st.write(f"Confidence: {100 * np.max(prediction):.2f}%")
