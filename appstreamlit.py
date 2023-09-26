import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from gtts import gTTS
from io import BytesIO
import time

# Load the trained model
model = load_model('sign2speech_best_model.h5')

# Define the classes corresponding to the signs ('0' to 'Z')
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Function to convert text to speech and play in real-time
def text_to_speech(text):
    tts = gTTS(text, lang='en')
    audio_data = BytesIO()
    tts.write_to_fp(audio_data)
    st.audio(audio_data.getvalue(), format='audio/mp3')

# Create a Streamlit web application
st.title('Sign Language Prediction')

# Option to predict using an image
st.sidebar.header('Choose an option')
option = st.sidebar.selectbox('Select an option', ['Predict sign using image', 'Predict sign using camera'])

# Create OpenCV video capture
cap = None

if option == 'Predict sign using camera':
    cap = cv2.VideoCapture(0)
    st.text("Camera is on. Press 'c' to capture a sign.")

while True:
    if cap:
        ret, frame = cap.read()

        if not ret:
            continue

        # Display the camera feed within the Streamlit frontend
        st.image(frame, channels='BGR', use_column_width=True, caption='Camera Feed')

        # Generate a unique key for the button using a timestamp
        capture_button_key = f"capture_button_{int(time.time())}"

        # Capture a new image when 'c' is pressed
        if st.button("Capture Sign ('c')", key=capture_button_key):
            cv2.imwrite('captured_sign.jpg', frame)
            break

# Release the camera when done
if cap:
    cap.release()

# Rest of the code for processing the captured image and making predictions
if 'captured_sign.jpg' in os.listdir():
    image_path = 'captured_sign.jpg'
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)
    predicted_class = classes[np.argmax(prediction)]

    st.image(image, caption='Captured Image', use_column_width=True)
    st.write(f"Predicted sign: {predicted_class}")
    text_to_speech(predicted_class)