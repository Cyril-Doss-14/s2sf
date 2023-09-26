import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from gtts import gTTS
from io import BytesIO

# Load the trained model
model = load_model('sign2speech_best_model.h5')

# Define the classes corresponding to the signs ('0' to 'Z')
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Function to convert text to speech and play in real-time
def text_to_speech(text):
    tts = gTTS(text, lang='en')
    audio_data = BytesIO()
    tts.write_to_fp(audio_data)
    audio_data.seek(0)
    st.audio(audio_data, format='audio/mp3')

# Create a Streamlit web application
st.title('Sign Language Prediction')

# Upload an image for prediction
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image_data = uploaded_image.read()
    st.image(image_data, caption='Uploaded Image', use_column_width=True)

    # Perform image processing and make predictions
    img = image.load_img(uploaded_image, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    prediction = model.predict(img)
    predicted_class = classes[np.argmax(prediction)]

    st.write(f"Predicted sign: {predicted_class}")

    # Automatically speak the predicted sign
    text_to_speech(predicted_class)

