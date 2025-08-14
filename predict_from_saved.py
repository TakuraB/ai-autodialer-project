import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import pickle
import numpy as np
import os

# --- 1. Define file paths ---
MODEL_DIR = 'intent_model'
MODEL_PATH = os.path.join(MODEL_DIR, 'intent_classifier.keras')
TOKENIZER_PATH = os.path.join(MODEL_DIR, 'tokenizer.json')
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.pkl')

# --- 2. Load all saved assets ---
print("Loading saved model and assets...")

# Check if the directory exists
if not os.path.isdir(MODEL_DIR):
    print(f"Error: The directory '{MODEL_DIR}' was not found.")
    print("Please run the training script first to save the model.")
    exit()

# Load the model
model = tf.keras.models.load_model(MODEL_PATH)

# Load the tokenizer
with open(TOKENIZER_PATH) as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)

# Load the label encoder
with open(LABEL_ENCODER_PATH, 'rb') as f:
    label_encoder = pickle.load(f)

print("Assets loaded successfully.")

# --- 3. Create the prediction function ---
def predict_intent(text):
    """
    Takes a sentence as input and returns the predicted intent.
    """
    # The max_length must be the same as used during training
    max_length = 120
    
    # Prepare the text using the loaded tokenizer
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_length, padding='post', truncating='post')
    
    # Get the prediction from the loaded model
    prediction = model.predict(padded, verbose=0)
    
    # Decode the prediction using the loaded label encoder
    predicted_class_index = np.argmax(prediction)
    predicted_intent = label_encoder.inverse_transform([predicted_class_index])
    
    return predicted_intent[0]

# --- 4. Use the function to make predictions ---
if __name__ == "__main__":
    print("\n--- Testing the loaded model ---")

    test_sentence_1 = "i still have no connection"
    print(f"'{test_sentence_1}' -> Predicted Intent: {predict_intent(test_sentence_1)}")

    test_sentence_2 = "this is the worst experience i have ever had"
    print(f"'{test_sentence_2}' -> Predicted Intent: {predict_intent(test_sentence_2)}")

    test_sentence_3 = "can you show me how to reset my password"
    print(f"'{test_sentence_3}' -> Predicted Intent: {predict_intent(test_sentence_3)}")
