import speech_recognition as sr
from gtts import gTTS
from playsound import playsound
import os
import requests # Import the requests library
import re # Import regex for finding numbers
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import pickle
import numpy as np

# --- 1. Load Saved AI Model and Assets ---
print("Loading saved model and assets...")
MODEL_DIR = 'intent_model'
try:
    model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'intent_classifier.keras'))
    with open(os.path.join(MODEL_DIR, 'tokenizer.json')) as f:
        tokenizer = tokenizer_from_json(json.load(f))
    with open(os.path.join(MODEL_DIR, 'label_encoder.pkl'), 'rb') as f:
        label_encoder = pickle.load(f)
except FileNotFoundError:
    print("Error: Model assets not found. Please run the training script first.")
    exit()
print("Assets loaded successfully.")

# --- 2. Create Prediction and Response Logic ---
def predict_intent(text):
    max_length = 120
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_length, padding='post', truncating='post')
    prediction = model.predict(padded, verbose=0)
    predicted_class_index = np.argmax(prediction)
    return label_encoder.inverse_transform([predicted_class_index])[0]

def get_response_for_intent(intent, text):
    """
    Returns a text response based on the intent.
    For 'check_status', it calls our mock API.
    """
    if intent == 'check_status':
        # Try to find a number in the user's speech to use as an order ID
        order_id = re.search(r'\d+', text)
        if order_id:
            order_id = order_id.group(0)
            # Call our local API
            try:
                api_url = f"http://127.0.0.1:5000/get_order_status/{order_id}"
                response = requests.get(api_url)
                if response.status_code == 200:
                    data = response.json()
                    return f"The status for order {order_id} is {data['status']}. The estimated delivery date is {data['delivery_date']}."
                else:
                    return "Sorry, I couldn't connect to the order system right now."
            except requests.exceptions.ConnectionError:
                return "Sorry, I am unable to connect to the backend service. Please make sure the API server is running."
        else:
            return "I can help with that. Please tell me the order number."

    # Default responses for other intents
    responses = {
        'ask_help': "Of course, I am here to assist you. What do you need help with?",
        'complain': "I'm sorry to hear you're having an issue. Please tell me more so I can help.",
        'other': "Thank you for your message. How can I help you today?"
    }
    return responses.get(intent, "I'm not sure how to respond to that, but I'm here to help.")

# --- 3. Create Voice Functions ---
def speak(text):
    print(f"Agent: {text}")
    try:
        tts = gTTS(text=text, lang='en')
        filename = 'response.mp3'
        tts.save(filename)
        playsound(filename)
        os.remove(filename)
    except Exception as e:
        print(f"Error during text-to-speech: {e}")

def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("\nListening...")
        r.pause_threshold = 1
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)
    try:
        print("Recognizing...")
        query = r.recognize_google(audio, language='en-us')
        print(f"You: {query}")
        return query
    except sr.UnknownValueError:
        speak("Sorry, I didn't catch that. Could you please repeat?")
        return None
    except Exception as e:
        print(f"Error during speech recognition: {e}")
        return None

# --- 4. Main Conversation Loop ---
if __name__ == "__main__":
    speak("Hello, I am your voice assistant. How can I help you today?")
    
    while True:
        user_input = listen()
        
        if user_input:
            if "goodbye" in user_input.lower() or "exit" in user_input.lower():
                speak("Goodbye!")
                break
            
            intent = predict_intent(user_input)
            # Pass the original user_input to the response function
            response = get_response_for_intent(intent, user_input)
            speak(response)
