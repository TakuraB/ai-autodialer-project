# AI-Powered Voice Agent & Auto-Dialer System

This repository contains the complete source code for an AI-powered auto-dialer and interactive voice agent. The system is capable of initiating outbound phone calls, understanding spoken user intent through a custom-trained TensorFlow model, and providing dynamic, data-driven responses.

## Core Technologies
- **Backend:** Python, Flask
- **AI/ML:** TensorFlow (Keras), Scikit-learn, Pandas
- **Telephony:** Twilio Voice API
- **Voice Processing:** SpeechRecognition, gTTS
- **Environment:** Git, python-dotenv

## Project Structure
    ├── intent_model/           # Saved TensorFlow model, tokenizer, and label encoder
    ├── data/                   # Raw and cleaned datasets (ignored by Git)
    ├── .env                    # Environment variables for credentials (ignored by Git)
    ├── .gitignore              # Specifies files and directories for Git to ignore
    ├── requirements.txt        # Project dependencies for pip
    ├── train_intent_model.py   # Script to train and save the AI model
    ├── interactive_api.py      # Flask server to handle calls and make predictions
    ├── auto_dialer.py          # Script to initiate the outbound call
    └── README.md               # This file

## Setup and Installation

Follow these steps to set up the project on your local machine.

### 1. Clone the Repository
    git clone https://github.com/takurab/ai-autodialer-project.git
    cd ai-autodialer-project

### 2. Create and Activate a Virtual Environment
**Windows**
    
    python -m venv venv
    .\venv\Scripts\activate

**macOS/Linux**
    
    python3 -m venv venv
    source venv/bin/activate

### 3. Install Dependencies
Install all the required Python packages using the `requirements.txt` file.
    
    pip install -r requirements.txt

### 4. Configure Credentials
Create a file named `.env` in the root of the project. Add the following variables, filling in your actual credentials from Twilio:
    
    ACCOUNT_SID="ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    AUTH_TOKEN="your_auth_token"
    TWILIO_PHONE_NUMBER="+15017122661"
    YOUR_CELL_PHONE="+15558675309"

## How to Run the System

This project requires three separate terminal processes to be running simultaneously.

### Terminal 1: Start the Interactive API Server
This Flask server loads the AI model and handles the interactive call logic.
    
    python interactive_api.py

### Terminal 2: Expose the API with ngrok
Twilio needs a public URL to connect to your local server.
    
    ngrok http 5000

Copy the `https://<random-string>.ngrok-free.app` URL provided by ngrok.

### Terminal 3: Initiate the Call
Open the `auto_dialer.py` file and paste your ngrok URL into the `NGROK_URL` variable. Run the script to make the call.
    
    python auto_dialer.py

Your phone will ring. When you answer, you will be connected to your AI agent.
