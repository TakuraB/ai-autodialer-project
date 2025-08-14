import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --- 1. Load and Prepare the Data ---
print("Loading and preparing data...")

try:
    df = pd.read_csv('data/cleaned_customer_support_data.csv')
except FileNotFoundError:
    print("Error: 'data/cleaned_customer_support_data.csv' not found.")
    print("Please ensure you've run Project 1 and moved the file to the 'data' folder.")
    exit()

# Ensure all messages are treated as strings to prevent errors.
df['customer_message'] = df['customer_message'].astype(str)

# --- IMPROVED INTENT ASSIGNMENT FUNCTION ---
def assign_intent(message):
    """
    Assigns an intent to a message based on keywords.
    The function checks for keywords in a specific order of priority.
    """
    message = message.lower()

    # Define keyword lists for clarity and easy modification
    complain_keywords = [
        'not working', 'issue', 'problem', 'disappointed', 'worst',
        'terrible', 'unacceptable', 'frustrated', 'broken', 'fail', 'complaint'
    ]
    status_keywords = ['update', 'status', 'track', 'where is', 'delivery']
    help_keywords = ['help', 'assist', 'how do i', 'can you', 'guide', 'support']

    # Check intents in a specific order of priority
    if any(keyword in message for keyword in complain_keywords):
        return 'complain'
    if any(keyword in message for keyword in status_keywords):
        return 'check_status'
    if any(keyword in message for keyword in help_keywords):
        return 'ask_help'

    return 'other' # Default category if no keywords match

df['intent'] = df['customer_message'].apply(assign_intent)

df = df[['customer_message', 'intent']]
print("Data sample with intents:")
print(df.head())
print(f"\nIntent distribution:\n{df['intent'].value_counts()}")

# --- 2. Convert Text to Numbers ---
vocab_size = 10000
max_length = 120
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(df['customer_message'])

sequences = tokenizer.texts_to_sequences(df['customer_message'])
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(df['intent'])
labels = np.expand_dims(labels, axis=-1)

# --- 3. Split Data for Training and Testing ---
X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences, labels, test_size=0.2, random_state=42
)

print(f"\nTraining data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# --- 4. Build the AI Model ---
print("\nBuilding the model...")
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 16), # No need for input_length
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# --- 5. Train the Model ---
print("\nTraining the model...")
num_epochs = 10
history = model.fit(
    X_train,
    y_train,
    epochs=num_epochs,
    validation_data=(X_test, y_test),
    verbose=2
)

# --- 6. Test the Model ---
print("\nTesting the model with new sentences...")

def predict_intent(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_length, padding='post', truncating='post')
    prediction = model.predict(padded, verbose=0) # Set verbose to 0 for cleaner output
    predicted_class_index = np.argmax(prediction)
    predicted_intent = label_encoder.inverse_transform([predicted_class_index])
    return predicted_intent[0]

print(f"'how do i track my package?' -> Predicted Intent: {predict_intent('how do i track my package?')}")
print(f"'this is the worst service ever' -> Predicted Intent: {predict_intent('this is the worst service ever')}")
print(f"'can you please help me with my account' -> Predicted Intent: {predict_intent('can you please help me with my account')}")

# --- 7. Save the Model and Supporting Files ---
print("\nSaving model and supporting files...")

import os
os.makedirs('intent_model', exist_ok=True)

model.save('intent_model/intent_classifier.keras')

import json
tokenizer_json = tokenizer.to_json()
with open('intent_model/tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(tokenizer_json, ensure_ascii=False))

import pickle
with open('intent_model/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("All files saved successfully in the 'intent_model' directory.")
