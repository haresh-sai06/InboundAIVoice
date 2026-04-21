import argparse
import os
import json
from dotenv import load_dotenv
from twilio.rest import Client

# Load environment variables
load_dotenv(".env")

# Get Twilio credentials from env
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
NGROK_URL = os.getenv("NGROK_URL", "")

def make_call(to_number: str):
    """
    Make an outbound call using Twilio.
    Routes to Sarvam AI for TTS, STT, and LLM.
    """
    
    # Validation
    if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
        print("ERROR: Please set TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN in .env")
        return
    
    if not TWILIO_PHONE_NUMBER:
        print("ERROR: Please set TWILIO_PHONE_NUMBER in .env")
        return
    
    if not NGROK_URL:
        print("ERROR: Please set NGROK_URL in .env (e.g., https://your-ngrok.ngrok-free.app)")
        return
    
    if not to_number.startswith("+"):
        print("ERROR: Phone number must start with '+' and country code (e.g., +919876543210)")
        return
    
    # Initialize Twilio client
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    
    # TwiML to connect to our Sarvam-powered WebSocket endpoint
    # Uses <Connect> for bidirectional streaming
    ngrok_host = NGROK_URL.rstrip('/').replace('https://', '').replace('http://', '')
    websocket_url = f"wss://{ngrok_host}/ws/twilio/sarvam"
    
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{websocket_url}" />
    </Connect>
</Response>"""
    
    print(f"📞 Making call to: {to_number}")
    print(f"📞 From: {TWILIO_PHONE_NUMBER}")
    print(f"🔗 WebSocket: {websocket_url}")
    print(f"🤖 Using: Sarvam STT + Sarvam TTS + Sarvam LLM (FREE)")
    print()
    
    try:
        # Create the call
        call = client.calls.create(
            to=to_number,
            from_=TWILIO_PHONE_NUMBER,
            twiml=twiml
        )
        
        print("✅ Call initiated successfully!")
        print(f"   Call SID: {call.sid}")
        print(f"   Status: {call.status}")
        print()
        print("📱 The phone should ring shortly!")
        print()
        print("Prerequisites:")
        print("   1. Start ngrok: ngrok http 8000")
        print("   2. Start server: python twilio_sarvam_server.py")
        print("   3. Ensure SARVAM_API_KEY is set in .env")
        
    except Exception as e:
        print(f"\n❌ Error making call: {e}")
        print()
        print("Troubleshooting:")
        print("   - Check Twilio credentials in .env")
        print("   - Verify phone number format (+91 for India)")
        print("   - Check Twilio console for error details")
        print("   - Ensure Twilio phone number is verified for outbound calls")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Make an outbound call via Twilio with Sarvam AI (STT + TTS + LLM)"
    )
    parser.add_argument(
        "--to", 
        required=True, 
        help="The phone number to call (e.g., +919876543210)"
    )
    args = parser.parse_args()
    
    make_call(args.to)
