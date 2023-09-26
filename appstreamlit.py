import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from io import BytesIO
from gtts import gTTS
import mediapipe as mp

st.set_page_config(page_title="Sign2Speech Translation", page_icon="🤟")

# Load the trained model for image prediction
image_model = load_model('sign2speech_best_model.h5')

# Load the trained model for real-time prediction
realtime_model = tf.keras.models.load_model('sign2speech_best_model.h5')

# Define the classes corresponding to the signs ('0' to 'Z')
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Function to convert text to speech and play in real-time
def text_to_speech(text):
    tts = gTTS(text, lang='en')
    audio_data = BytesIO()
    tts.write_to_fp(audio_data)
    audio_data.seek(0)
    st.audio(audio_data, format='audio/mp3')

# Function for image prediction
def predict_image(image_data):
    img = image.load_img(BytesIO(image_data), target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    prediction = image_model.predict(img)
    predicted_class = classes[np.argmax(prediction)]

    st.write(f"Predicted sign: {predicted_class}")

    # Automatically speak the predicted sign
    text_to_speech(predicted_class)

# Function for real-time prediction
def predict_realtime():
    frame_width = 800
    frame_height = 600
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(0)  # Set the camera index to the desired value

    if not cap.isOpened():
        st.write("Error: Camera not found.")
    else:
    # Load the model here
        image_model = load_model('sign2speech_best_model.h5')
    
    # Rest of your code for real-time prediction


    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (frame_width, frame_height))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            min_x, max_x, min_y, max_y = frame_width, 0, frame_height, 0

            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * frame_width), int(landmark.y * frame_height)
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)

            hand_frame = frame[min_y:max_y, min_x:max_x]

            if not hand_frame.size:
                continue

            hand_frame_resized = cv2.resize(hand_frame, (224, 224))
            hand_frame_normalized = hand_frame_resized / 255.0

            prediction = realtime_model.predict(np.expand_dims(hand_frame_normalized, axis=0))
            predicted_label = np.argmax(prediction)
            predicted_sign = classes[predicted_label]
            confidence = prediction[0][predicted_label]

            text = f'Predicted Sign: {predicted_sign} (Confidence: {confidence:.2f})'
            cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            for connection in mp_hands.HAND_CONNECTIONS:
                x1, y1 = int(hand_landmarks.landmark[connection[0]].x * frame_width), int(hand_landmarks.landmark[connection[0]].y * frame_height)
                x2, y2 = int(hand_landmarks.landmark[connection[1]].x * frame_width), int(hand_landmarks.landmark[connection[1]].y * frame_height)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow('Sign Language Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Create a Streamlit web application
# Create a Streamlit web application
st.title('Sign Language Prediction')

# Choose an option
option = st.sidebar.selectbox('Select an option', ['Predict using Image', 'Predict in Real Time'])

if option == 'Predict using Image':
    st.sidebar.header('Image Prediction')
    uploaded_image = st.file_uploader("Upload an image")

    if uploaded_image is not None:
        image_data = uploaded_image.read()
        st.image(image_data, caption='Uploaded Image', use_column_width=True)
        predict_image(image_data)

elif option == 'Predict in Real Time':
    st.sidebar.header('Real-Time Prediction')
    st.text("Camera is on. Hold up a sign for prediction. Press q to close the camera.👍")
    predict_realtime()


