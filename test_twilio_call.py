import sys
from twilio_client import TwilioClient
from config import settings

# Get ngrok URL from settings
NGROK_URL = settings.NGROK_URL

# Daily.co SIP configuration
DAILY_DOMAIN = "6e418e1622aa4b9094883b68923186cd"
DAILY_ROOM_NAME = settings.DAILY_ROOM_NAME
DAILY_SIP_URI = f"sip:{DAILY_ROOM_NAME}@{DAILY_DOMAIN}.sip.daily.co"


def make_test_call(to_number: str, mode: str = "daily"):
    """
    Make a test outbound call using Twilio with AI voice.
    
    Args:
        to_number: Phone number to call (with country code, e.g., +919876543210)
        mode: 'gather' (basic), 'daily' (Daily.co SIP - RECOMMENDED), or 'stream' (Media Stream)
    """
    
    # Check if Twilio is configured
    if not settings.TWILIO_ACCOUNT_SID or not settings.TWILIO_AUTH_TOKEN:
        print("ERROR: Please set TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN in .env")
        return
    
    if not settings.TWILIO_PHONE_NUMBER:
        print("ERROR: Please set TWILIO_PHONE_NUMBER in .env")
        return
    
    client = TwilioClient()
    
    if not client.validate_credentials():
        print("ERROR: Twilio credentials not properly configured")
        return
    
    try:
        print(f"Making call to {to_number}...")
        print(f"From: {client.phone_number}")
        print(f"Mode: {mode}")
        print()
        print("🎉 Using Gemini API for AI conversation!")
        print(f"   Model: {settings.MODEL}")
        print(f"   Voice: {settings.VOICE}")
        print()
        
        if mode == "daily":
            # Option 1: Daily.co SIP dial-in (BEST QUALITY - RECOMMENDED)
            # This routes the call through Daily.co for WebRTC-quality audio
            # Then connects to Gemini Live API
            twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Aditi">Namaste! Connecting you to ZEE School admission counselor. Please wait a moment.</Say>
    <Dial>
        <Sip>{DAILY_SIP_URI}</Sip>
    </Dial>
</Response>"""
            print(f"📞 Routing through Daily.co SIP: {DAILY_SIP_URI}")
            print("   This provides low-latency, natural AI voice conversation")
            
        elif mode == "stream":
            # Option 2: Media Stream for real-time bidirectional audio
            # Strip any trailing slash and protocol for the wss:// URL
            ngrok_host = NGROK_URL.rstrip('/').replace('https://', '').replace('http://', '')
            websocket_url = f"wss://{ngrok_host}/ws/twilio"
            
            # IMPORTANT: Use <Connect> for bidirectional audio.
            # <Start> is unidirectional (Twilio sends to us, drops what we send back!).
            twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{websocket_url}" />
    </Connect>
</Response>"""
            print(f"📡 Using Media Stream WebSocket: {websocket_url}")
            print("   Using <Connect> for full bidirectional voice streaming")
            
        else:
            # Option 3: Gather + Say (CURRENT WORKING SETUP)
            # Uses Twilio's speech recognition + Gemini text processing
            ngrok_url = NGROK_URL
            
            twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Aditi">Hi! This is Priya from ZEE School. Do you have a school-going child?</Say>
    <Gather action="{ngrok_url}/twiml/gather" method="POST" input="speech" timeout="15" language="en-IN" speechTimeout="auto">
    </Gather>
    <Redirect>{ngrok_url}/twiml/gather</Redirect>
</Response>"""
            print(f"🎯 Using Gather + Say with webhook: {ngrok_url}/twiml/gather")
            print("   This uses Twilio's built-in speech recognition")
        
        # Make the call
        result = client.make_call(
            to_number=to_number,
            twiml=twiml
        )
        
        if result.get("success"):
            print(f"\n✅ Call initiated successfully!")
            print(f"   Call SID: {result.get('call_sid')}")
            print(f"   Status: {result.get('status')}")
            print()
            print("📱 Your phone should ring shortly!")
            print()
            
            if mode == "daily":
                print("💡 For Daily.co mode:")
                print("   1. Make sure your server is running (python main.py)")
                print("   2. Make sure ngrok is running (ngrok http 8000)")
                print("   3. The call will route through Daily for best quality")
                
        else:
            print(f"\n❌ Call failed: {result.get('error')}")
            print()
            print("Troubleshooting:")
            print("   - Check Twilio credentials in .env")
            print("   - Verify phone number format (+91 for India)")
            print("   - Check Twilio console for error details")
        
    except Exception as e:
        print(f"Error making call: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("=" * 70)
        print("📞 Twilio Test Call - ZEE School AI Receptionist")
        print("=" * 70)
        print()
        print("Usage:")
        print("  python test_twilio_call.py <phone_number> [mode]")
        print()
        print("Examples:")
        print("  # Daily.co SIP mode (RECOMMENDED - Best quality)")
        print("  python test_twilio_call.py +919876543210 daily")
        print()
        print("  # Media Stream mode (Real-time bidirectional)")
        print("  python test_twilio_call.py +919876543210 stream")
        print()
        print("  # Gather mode (Basic - uses Twilio speech recognition)")
        print("  python test_twilio_call.py +919876543210 gather")
        print()
        print("Modes:")
        print("  daily   - Daily.co SIP dial-in (WebRTC quality, low latency)")
        print("  stream  - Media Stream WebSocket (Real-time streaming)")
        print("  gather  - Gather + Say (Basic, uses Twilio STT)")
        print()
        print("Prerequisites:")
        print("  1. Update NGROK_URL in .env file")
        print("  2. Start ngrok: ngrok http 8000")
        print("  3. Start server: python main.py")
        print()
        print("For Indian numbers: Use +91 prefix (e.g., +919876543210)")
        print("=" * 70)
        sys.exit(1)
    
    phone_number = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "daily"
    
    # Validate mode
    if mode not in ["daily", "stream", "gather"]:
        print(f"❌ Invalid mode: {mode}")
        print("Valid modes: daily, stream, gather")
        sys.exit(1)
    
    make_test_call(phone_number, mode)
