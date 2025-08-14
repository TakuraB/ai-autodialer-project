from twilio.rest import Client
import time
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# --- Load Your Credentials from .env ---
ACCOUNT_SID = os.getenv("ACCOUNT_SID")
AUTH_TOKEN = os.getenv("AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
YOUR_CELL_PHONE = os.getenv("YOUR_CELL_PHONE")

# --- Webhook URL ---
# IMPORTANT: Replace with your ngrok forwarding URL each time you run it.
NGROK_URL = "https://70b61d5bbb8f.ngrok-free.app"
# This URL now points to the '/voice' endpoint on your interactive API server
VOICE_URL = f"{NGROK_URL}/voice" 

# Initialize the Twilio client
client = Client(ACCOUNT_SID, AUTH_TOKEN)

def make_call():
    """
    Initiates a phone call and directs it to our interactive voice API.
    """
    # Check if essential variables are loaded
    if not all([ACCOUNT_SID, AUTH_TOKEN, TWILIO_PHONE_NUMBER, YOUR_CELL_PHONE]):
        print("Error: Missing one or more environment variables.")
        print("Please check your .env file.")
        return

    print(f"Initiating call to {VOICE_URL}...")
    try:
        call = client.calls.create(
            # This is the key change: we provide a URL for Twilio to fetch instructions from.
            url=VOICE_URL,
            to=YOUR_CELL_PHONE,
            from_=TWILIO_PHONE_NUMBER
        )
        
        print(f"Call initiated with SID: {call.sid}")
        
        # Monitor the call status
        while True:
            time.sleep(5)
            updated_call = client.calls(call.sid).fetch()
            print(f"Call status: {updated_call.status}")
            if updated_call.status in ["completed", "failed", "canceled", "no-answer"]:
                print("Call finished.")
                break

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    make_call()
