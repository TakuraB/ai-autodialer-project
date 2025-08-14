from flask import Flask, request
from twilio.twiml.voice_response import VoiceResponse, Gather
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import pickle
import numpy as np
import os

# --- 1. Initialize Flask App and Load AI Model ---
app = Flask(__name__)

# Load the saved AI model and assets once when the server starts
MODEL_DIR = 'intent_model'
model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'intent_classifier.keras'))
with open(os.path.join(MODEL_DIR, 'tokenizer.json')) as f:
    tokenizer = tokenizer_from_json(json.load(f))
with open(os.path.join(MODEL_DIR, 'label_encoder.pkl'), 'rb') as f:
    label_encoder = pickle.load(f)
print("AI model and assets loaded successfully.")

def predict_intent(text):
    """Takes a sentence and returns the predicted intent."""
    max_length = 120
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_length, padding='post', truncating='post')
    prediction = model.predict(padded, verbose=0)
    predicted_class_index = np.argmax(prediction)
    return label_encoder.inverse_transform([predicted_class_index])[0]

# --- 2. Define API Endpoints for Twilio ---

@app.route("/voice", methods=['GET', 'POST'])
def voice():
    """This endpoint handles the initial incoming call from Twilio."""
    response = VoiceResponse()
    
    # Use <Gather> to greet the user and listen for their speech
    gather = Gather(input='speech', action='/gather', speechTimeout='auto')
    gather.say('Hello, you have reached the AI assistant. How can I help you today?')
    response.append(gather)

    # If the user doesn't say anything, redirect back to the start
    response.redirect('/voice')

    return str(response)

@app.route("/gather", methods=['GET', 'POST'])
def gather():
    """This endpoint processes the speech transcribed by Twilio."""
    response = VoiceResponse()

    # Check if Twilio provided transcribed speech
    if 'SpeechResult' in request.values:
        user_speech = request.values['SpeechResult']
        print(f"User said: {user_speech}")
        
        # Predict the intent using our TensorFlow model
        intent = predict_intent(user_speech)
        print(f"Predicted intent: {intent}")
        
        # Generate a dynamic response based on the intent
        if intent == 'ask_help':
            response.say('It sounds like you need assistance. I am connecting you to our support documentation.')
        elif intent == 'complain':
            response.say('I understand you are having an issue. A support ticket has been created and an agent will contact you shortly.')
        elif intent == 'check_status':
            response.say('To check the status of an order, please visit our website and enter your tracking number.')
        else:
            response.say('Thank you for your message. If you need further assistance, please visit our website.')
            
    else:
        # If no speech was detected, redirect to the start
        response.redirect('/voice')
        
    # Hang up the call after responding
    response.hangup()

    return str(response)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
