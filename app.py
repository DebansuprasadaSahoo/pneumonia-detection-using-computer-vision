import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Streamlit deployment
st.markdown("<h1 style='color:White;'>Pneumonia Detection from Chest X-ray</h1>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded X-ray', use_column_width=True)

    # Convert the image to RGB (if not already in RGB format)
    image = image.convert("RGB")
    
    # Preprocess the image
    image = image.resize((224, 224))
    image = np.array(image) / 255.0  # Normalize the pixel values to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Load the trained model
    model = load_model(r'C:\Users\deban\Downloads\pneumonia detection\pneumonia_model.h5')
    prediction = model.predict(image)

    # Display the prediction
    if prediction > 0.5:
        st.write("Prediction: Pneumonia Detected")
    else:
        st.write("Prediction: Healthy")
        
    # Add custom CSS for background image
bg_image_url = "https://wallpaperbat.com/img/304560-doctor-wallpaper.jpg"  # Replace with your image URL
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("{bg_image_url}");
        background-size: cover;
        background-position: center center;
        background-repeat: no-repeat;
        height: 100vh;
    }}
    </style>
    """, unsafe_allow_html=True)
