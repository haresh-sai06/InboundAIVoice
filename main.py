import asyncio
import audioop
import base64
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from sqlmodel import SQLModel, create_engine, Session, select
from sqlmodel.ext.asyncio.session import AsyncSession

from gemini_live import GeminiLive
from config import settings
from database import engine, get_session, Call, CallStatus
from receptionist_tools import get_receptionist_tool_mapping, get_receptionist_tool_definitions
from twilio_audio_bridge import TwilioAudioBridgeFixed as TwilioAudioBridge
from azure_tts import AzureTTS
from azure_stt import AzureSTT
from conversation_manager import ConversationManager
from google import genai
from google.genai import types

# Configure logging - DEBUG for our modules, INFO for everything else
logging.basicConfig(level=logging.INFO)
logging.getLogger("gemini_live").setLevel(logging.DEBUG)
logging.getLogger(__name__).setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)




@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: create database tables
    SQLModel.metadata.create_all(engine)
    yield
    # Shutdown: cleanup if needed


# Initialize FastAPI
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.get("/")
async def root():
    return FileResponse("frontend/index.html")


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "service": "AI Voice Receptionist",
        "model": settings.MODEL,
        "voice": settings.VOICE
    }


@app.post("/twiml")
async def twiml_endpoint(request: Request):
    """
    TwiML endpoint for Twilio calls.
    Returns TwiML to connect to Media Stream for audio streaming with Gemini Live.
    """
    # Get call details from Twilio request
    call_sid = request.query_params.get("CallSid", "")
    from_number = request.query_params.get("From", "")
    to_number = request.query_params.get("To", "")
    
    logger.info(f"TwiML request: CallSid={call_sid}, From={from_number}, To={to_number}")
    
    # Get the ngrok URL from the request
    scheme = request.url.scheme
    host = request.headers.get("host", "localhost:8000")
    base_url = f"{scheme}://{host}"
    
    # Return TwiML to connect to Media Stream for audio
    # Use wss:// for secure WebSocket connection
    websocket_url = f"wss://{host}/ws/twilio"
    
    # IMPORTANT: After <Start>, we need a <Pause> to keep the call alive
    # while the WebSocket stream handles the audio. Without this, Twilio
    # would end the call immediately after starting the stream.
    # Reduced from 5 minutes to 2 minutes for faster initialization
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Start>
        <Stream url="{websocket_url}" />
    </Start>
    <Pause length="60"/>
    <Pause length="60"/>
</Response>"""
    
    logger.info(f"Returning TwiML with WebSocket URL: {websocket_url}")
    return Response(content=twiml, media_type="application/xml")


@app.post("/twiml/azure")
async def twiml_azure_endpoint(request: Request):
    """
    TwiML endpoint for Twilio calls with Azure Speech Services.
    Returns TwiML to connect to Media Stream for audio streaming with Azure STT + TTS.
    """
    # Get call details from Twilio request
    call_sid = request.query_params.get("CallSid", "")
    from_number = request.query_params.get("From", "")
    to_number = request.query_params.get("To", "")
    
    logger.info(f"TwiML Azure request: CallSid={call_sid}, From={from_number}, To={to_number}")
    
    # Get the ngrok URL from the request
    scheme = request.url.scheme
    host = request.headers.get("host", "localhost:8000")
    base_url = f"{scheme}://{host}"
    
    # Return TwiML to connect to Media Stream for audio
    # Use wss:// for secure WebSocket connection
    websocket_url = f"wss://{host}/ws/twilio/azure"
    
    # IMPORTANT: After <Start>, we need a <Pause> to keep the call alive
    # while the WebSocket stream handles the audio. Without this, Twilio
    # would end the call immediately after starting the stream.
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Start>
        <Stream url="{websocket_url}" />
    </Start>
    <Pause length="60"/>
    <Pause length="60"/>
</Response>"""
    
    logger.info(f"Returning Azure TwiML with WebSocket URL: {websocket_url}")
    return Response(content=twiml, media_type="application/xml")


@app.post("/twiml/gather")
async def twiml_gather(request: Request):
    """
    Process speech input from Twilio Gather and return AI response.
    Uses Gemini for text processing and Twilio TTS for voice output.
    """
    # Get speech result from Twilio
    form_data = await request.form()
    speech_result = form_data.get("SpeechResult", "")
    call_sid = form_data.get("CallSid", "")
    
    logger.info(f"🎤 Speech input received: {speech_result}")
    logger.info(f"📞 Call SID: {call_sid}")
    
    if not speech_result:
        # No speech detected
        twiml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>I didn't catch that. Could you please repeat?</Say>
    <Gather action="/twiml/gather" method="POST" input="speech" timeout="10" language="en-IN" speechTimeout="auto">
        <Say>Please speak after the beep.</Say>
    </Gather>
    <Say>Goodbye.</Say>
</Response>"""
        return Response(content=twiml, media_type="application/xml")
    
    try:
        logger.info("🔧 Initializing Gemini client...")
        # Initialize Gemini client for text processing
        client = genai.Client(api_key=settings.GEMINI_API_KEY)
        
        logger.info("📝 Creating system prompt...")
        # Create system prompt for ZEE school admission promotional caller
        system_prompt = """You are Priya, an admission counselor from ZEE School, making a friendly outbound call.

IMPORTANT - BE SHORT AND NATURAL:
- Keep EVERY response under 2 sentences max
- Sound like a real human, not a script reader
- Ask only ONE question at a time
- Wait for them to respond before moving on
- Use natural Indian English conversational style

ZEE School key points (only share when asked or relevant):
- Hosur location, smart classrooms, experienced teachers
- Sports, arts, robotics clubs
- Bus facility, affordable fees, scholarships
- 100% board pass rate

Flow:
1. Greet and ask if they have a school-age child
2. If yes, ask which grade and current school
3. Share 1-2 relevant ZEE features based on their need
4. Ask if they'd like to visit or get more info
5. If interested: tell them counselor will call shortly
6. If not interested: say thanks warmly and end

NEVER: list bullet points aloud, give long speeches, repeat yourself.
ALWAYS: be warm, brief, and conversational."""
        
        # Use model from settings (gemini-3-flash-live from .env)
        text_model = settings.MODEL
        logger.info(f"🤖 Calling Gemini model: {text_model}")
        
        # Generate AI response
        response = client.models.generate_content(
            model=text_model,
            contents=[
                types.Content(parts=[types.Part(text=system_prompt)]),
                types.Content(parts=[types.Part(text=f"Caller said: {speech_result}\n\nRespond naturally:")])
            ],
            config=types.GenerateContentConfig(
                temperature=0.8,
                max_output_tokens=500
            )
        )
        
        ai_response = response.text.strip() if response.text else ""
        logger.info(f"✅ AI response: {ai_response}")
        
        if not ai_response:
            logger.warning("⚠️ Empty response from Gemini, using fallback")
            raise Exception("Empty response from Gemini")
        
        # Escape XML special characters in AI response
        ai_response_escaped = ai_response.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
        
        # Return TwiML with AI response and loop back to gather — NEVER hang up here
        twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Aditi">{ai_response_escaped}</Say>
    <Gather action="/twiml/gather" method="POST" input="speech" timeout="15" language="en-IN" speechTimeout="auto">
    </Gather>
    <Redirect>/twiml/gather</Redirect>
</Response>"""
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        logger.error(f"❌ Error processing with Gemini: {e}")
        logger.error(f"📋 Full traceback: {traceback.format_exc()}")
        
        # ALWAYS return valid TwiML - use rule-based fallback
        logger.warning("⚠️ Using rule-based fallback due to error")
        
        speech_lower = speech_result.lower() if speech_result else ""
        if "hello" in speech_lower or "hi" in speech_lower:
            ai_response = "Hello! Welcome to our business. How can I assist you today?"
        elif "how are you" in speech_lower:
            ai_response = "I'm doing well, thank you for asking! How can I help you today?"
        elif "services" in speech_lower or "what do you do" in speech_lower:
            ai_response = "We offer various services. What would you like to know about?"
        elif "hours" in speech_lower or "open" in speech_lower:
            ai_response = "We are open Monday through Saturday, 9 AM to 6 PM."
        elif "appointment" in speech_lower or "book" in speech_lower:
            ai_response = "I can help you book an appointment. What date and time works for you?"
        elif "bye" in speech_lower or "goodbye" in speech_lower:
            ai_response = "Goodbye! Thank you for calling. Have a wonderful day!"
        else:
            ai_response = "I apologize, I'm having technical difficulties. Please try again or ask about our services, hours, or appointments."
        
        twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Aditi">{ai_response}</Say>
    <Gather action="/twiml/gather" method="POST" input="speech" timeout="15" language="en-IN" speechTimeout="auto">
    </Gather>
    <Redirect>/twiml/gather</Redirect>
</Response>"""
    
    return Response(content=twiml, media_type="application/xml")


@app.post("/twilio/webhook")
async def twilio_webhook(request: Request):
    """
    Webhook endpoint for Twilio call events.
    Handles call status updates (ringing, answered, completed, etc).
    """
    # Get call event data
    call_sid = request.query_params.get("CallSid", "")
    call_status = request.query_params.get("CallStatus", "")
    from_number = request.query_params.get("From", "")
    to_number = request.query_params.get("To", "")
    call_direction = request.query_params.get("Direction", "")
    
    logger.info(f"Twilio webhook: CallSid={call_sid}, Status={call_status}, From={from_number}, Direction={call_direction}")
    
    # Update call record in database
    session = Session(engine)
    try:
        call = session.exec(
            select(Call).where(Call.call_sid == call_sid)
        ).first()
        
        if call:
            # Update call status
            if call_status == "ringing":
                call.status = CallStatus.RINGING
            elif call_status == "in-progress" or call_status == "answered":
                call.status = CallStatus.CONNECTED
                if not call.started_at:
                    call.started_at = datetime.utcnow()
            elif call_status == "completed":
                call.status = CallStatus.COMPLETED
                call.ended_at = datetime.utcnow()
                if call.started_at:
                    call.duration = int((call.ended_at - call.started_at).total_seconds())
            elif call_status == "failed" or call_status == "busy" or call_status == "no-answer":
                call.status = CallStatus.FAILED
                call.ended_at = datetime.utcnow()
                if call.started_at:
                    call.duration = int((call.ended_at - call.started_at).total_seconds())
            
            session.commit()
            logger.info(f"Updated call {call.id} to status {call_status}")
        else:
            logger.warning(f"Call {call_sid} not found in database")
    
    except Exception as e:
        logger.error(f"Error processing Twilio webhook: {e}")
        session.rollback()
    finally:
        session.close()
    
    # Return 200 OK to acknowledge receipt
    return Response(status_code=200)


@app.websocket("/ws/twilio")
async def twilio_media_stream(websocket: WebSocket):
    """
    WebSocket endpoint for Twilio Media Stream.
    Handles bidirectional audio streaming with Twilio and bridges to Gemini Live.
    """
    logger.info("Twilio Media Stream WebSocket connection attempt...")
    await websocket.accept()
    logger.info("Twilio Media Stream connection accepted")
    
    # Create database session
    session = Session(engine)
    
    # Create call record
    call = Call(
        call_sid="twilio_" + str(datetime.utcnow().timestamp()),
        status=CallStatus.CONNECTED,
        started_at=datetime.utcnow()
    )
    session.add(call)
    session.commit()
    session.refresh(call)
    
    logger.info(f"Twilio call record created: {call.id}")
    
    # Initialize audio bridge (24000Hz for Gemini output)
    audio_bridge = TwilioAudioBridge(gemini_sample_rate=24000)
    current_audio_rate = 24000  # Track actual sample rate from Gemini
    
    # Call management state
    call_state = {
        "rejected_attempts": 0,
        "should_end_call": False,
        "is_ai_speaking": False,
        "audio_output_queue": asyncio.Queue(),  # Track pending audio
    }
    
    # End call detection keywords — include satisfaction indicators
    END_CALL_KEYWORDS = [
        "bye", "goodbye", "see you later", "talk later", "got to go",
        "stop calling", "don't call", "remove my number", "take me off",
        "thank you", "thanks", "okay", "fine", "that's all", "nothing else"
    ]
    
    # Rejection keywords for re-engagement
    REJECTION_KEYWORDS = [
        "not interested", "no thanks", "not now", "busy", "don't want",
        "already decided", "not looking", "happy with current"
    ]
    
    # Initialize Gemini Live client
    audio_input_queue = asyncio.Queue()
    audio_output_queue = asyncio.Queue()
    text_input_queue = asyncio.Queue()
    video_input_queue = asyncio.Queue()
    
    # Get tool mapping
    tool_mapping = get_receptionist_tool_mapping(session, call.id)
    tool_definitions = get_receptionist_tool_definitions()
    
    logger.info("Initializing Gemini Live client...")
    gemini_client = GeminiLive(
        api_key=settings.GEMINI_API_KEY,
        model=settings.MODEL,
        input_sample_rate=16000,
        voice=settings.VOICE,
        tools=tool_definitions,
        tool_mapping=tool_mapping
    )
    
    stream_sid = None  # Will be set when we receive the 'start' event from Twilio
    current_audio_rate = 24000  # Gemini outputs at 24kHz by default
    
    async def twilio_audio_callback(pcm_audio, sample_rate=None):
        """Send audio from Gemini Live to Twilio."""
        if not stream_sid:
            logger.warning("⚠️ stream_sid not yet available, cannot send audio to Twilio")
            return
        
        # Check if call should end - don't send more audio
        if call_state["should_end_call"]:
            logger.info("🛑 Call ending, skipping audio send")
            return
        
        # Clear interruption flag when AI starts speaking again
        audio_bridge.set_interrupted(False)
        
        # Mark AI as speaking
        call_state["is_ai_speaking"] = True
        
        # Use detected sample rate or default
        rate = sample_rate or current_audio_rate
        
        logger.info(f"🔊 Sending audio to Twilio: {len(pcm_audio)} bytes @ {rate}Hz")
        try:
            await audio_bridge.send_audio_to_twilio(websocket, pcm_audio, stream_sid, rate)
            logger.info(f"✅ Audio sent to Twilio successfully")
        except Exception as e:
            logger.error(f"❌ Error sending audio to Twilio: {e}")
        finally:
            call_state["is_ai_speaking"] = False
    
    def check_end_call_intent(text: str) -> bool:
        """Check if user wants to end the call."""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in END_CALL_KEYWORDS)
    
    def check_rejection_intent(text: str) -> bool:
        """Check if user is rejecting/not interested."""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in REJECTION_KEYWORDS)
    
    async def handle_end_call():
        """Gracefully end the call."""
        call_state["should_end_call"] = True
        logger.info("📞 Ending call based on user intent")
        
        # Close WebSocket immediately to end the call
        try:
            logger.info("🔌 Closing WebSocket connection")
            await websocket.close()
        except Exception as e:
            logger.error(f"Error closing WebSocket: {e}")
        
        # Break out of the session loop immediately
        raise asyncio.CancelledError("Call ended by user intent")
    
    async def handle_rejection():
        """Handle rejection - try to re-engage once, then end if still rejected."""
        call_state["rejected_attempts"] += 1
        logger.info(f"🚫 Rejection detected (attempt #{call_state['rejected_attempts']})")
        
        if call_state["rejected_attempts"] == 1:
            # Try re-engagement once - professional tone
            re_engagement = "Early bird discounts available. Interested?"
            await text_input_queue.put(re_engagement)
            return False  # Don't end call yet
        else:
            # Second rejection - end politely
            await text_input_queue.put("Understood. Thanks and good luck!")
            await asyncio.sleep(3)
            await handle_end_call()
            return True  # Call ended
    
    def clear_pending_audio():
        """Clear pending audio queue when user interrupts."""
        logger.info("🔄 Clearing pending audio due to user interruption")
        # Note: In a more sophisticated implementation, you'd track and cancel
        # pending audio sends. For now, we rely on the natural flow.
    
    async def twilio_interrupt_callback():
        """Handle user interrupt - stop AI from speaking."""
        logger.info("🛑 User interrupted AI - stopping audio transmission")
        # Set interruption flag on audio bridge to stop sending audio
        audio_bridge.set_interrupted(True)
        clear_pending_audio()
        # Send interrupt signal to Gemini Live to stop current generation
        await gemini_client.interrupt()
        # Signal that AI should stop current speech
        call_state["is_ai_speaking"] = False
    
    async def receive_from_twilio():
        """Receive audio from Twilio and send to Gemini."""
        nonlocal stream_sid
        try:
            message_count = 0
            while True:
                message = await websocket.receive_text()
                message_count += 1
                
                # Parse message to check for start event
                try:
                    data = json.loads(message)
                    if data.get("event") == "start":
                        stream_sid = data.get("start", {}).get("streamSid")
                        logger.info(f"📞 Twilio Media Stream started with streamSid: {stream_sid}")
                        
                        # Send short text prompt to trigger AI greeting
                        greeting = "Start."
                        await text_input_queue.put(greeting)
                        logger.info("📤 Sent short prompt to trigger AI greeting")
                        
                    elif data.get("event") == "media":
                        # Check for interruption - if AI is speaking and we receive media, interrupt
                        if call_state["is_ai_speaking"] and message_count % 50 == 0:
                            logger.info(f"🛑 User speech detected while AI speaking (message #{message_count})")
                            await twilio_interrupt_callback()
                        
                        if message_count % 100 == 0:
                            logger.info(f"Received message #{message_count} from Twilio")
                except:
                    pass
                
                # Process Twilio message and convert audio
                await audio_bridge.process_twilio_message(
                    message,
                    lambda audio: audio_input_queue.put_nowait(audio)
                )
        except WebSocketDisconnect:
            logger.info("Twilio Media Stream disconnected")
        except Exception as e:
            logger.error(f"Error receiving from Twilio: {e}")
    
    receive_task = asyncio.create_task(receive_from_twilio())
    
    async def run_gemini_session():
        """Run Gemini Live session and consume events with intent detection."""
        async for event in gemini_client.start_session(
            audio_input_queue=audio_input_queue,
            video_input_queue=video_input_queue,
            text_input_queue=text_input_queue,
            audio_output_callback=twilio_audio_callback,
            audio_interrupt_callback=twilio_interrupt_callback,
        ):
            if event:
                logger.debug(f"Gemini event: {event}")
                
                # Process user speech for intent detection
                if event.get("type") == "user":
                    user_text = event.get("text", "").lower()
                    logger.info(f"🎤 User said: {user_text}")
                    
                    # Check for interruption - if AI is speaking, clear pending audio
                    if call_state["is_ai_speaking"]:
                        clear_pending_audio()
                    
                    # Check if user wants to end call
                    if check_end_call_intent(user_text):
                        logger.info("👋 User wants to end call detected")
                        await handle_end_call()
                        break
                    
                    # Check for rejection intent
                    if check_rejection_intent(user_text):
                        call_ended = await handle_rejection()
                        if call_ended:
                            break
                
                # Process AI responses
                elif event.get("type") == "gemini":
                    ai_text = event.get("text", "")
                    logger.info(f"🤖 AI said: {ai_text}")
    
    try:
        logger.info("Starting Gemini Live session...")
        await run_gemini_session()
    except Exception as e:
        logger.error(f"Error in Gemini session: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cancel receive task
        receive_task.cancel()
        
        # Update call record
        call.ended_at = datetime.utcnow()
        if call.started_at:
            call.duration = int((call.ended_at - call.started_at).total_seconds())
        call.status = CallStatus.COMPLETED
        session.commit()
        
        logger.info(f"Twilio call record updated: {call.id}, duration: {call.duration}s")
        session.close()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for Gemini Live with call logging."""
    await websocket.accept()

    logger.info("WebSocket connection accepted")

    # Create database session
    session = Session(engine)
    
    # Create call record
    call = Call(
        call_sid=f"ws_{datetime.utcnow().timestamp()}",
        status=CallStatus.CONNECTED,
        started_at=datetime.utcnow()
    )
    session.add(call)
    session.commit()
    session.refresh(call)
    
    logger.info(f"Call record created: {call.id}")

    audio_input_queue = asyncio.Queue()
    video_input_queue = asyncio.Queue()
    text_input_queue = asyncio.Queue()
    
    # Accumulate transcript
    transcript_parts = []

    async def audio_output_callback(data):
        await websocket.send_bytes(data)

    async def audio_interrupt_callback():
        pass

    # Get tool mapping with call ID
    tool_mapping = get_receptionist_tool_mapping(session, call.id)
    tool_definitions = get_receptionist_tool_definitions()

    gemini_client = GeminiLive(
        api_key=settings.GEMINI_API_KEY, 
        model=settings.MODEL, 
        input_sample_rate=16000, 
        voice=settings.VOICE,
        tools=tool_definitions,
        tool_mapping=tool_mapping
    )

    async def receive_from_client():
        try:
            while True:
                message = await websocket.receive()

                if message.get("bytes"):
                    await audio_input_queue.put(message["bytes"])
                elif message.get("text"):
                    text = message["text"]
                    try:
                        payload = json.loads(text)
                        if isinstance(payload, dict) and payload.get("type") == "image":
                            logger.info(f"Received image chunk from client: {len(payload['data'])} base64 chars")
                            image_data = base64.b64decode(payload["data"])
                            await video_input_queue.put(image_data)
                            continue
                    except json.JSONDecodeError:
                        pass

                    await text_input_queue.put(text)
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected")
        except Exception as e:
            logger.error(f"Error receiving from client: {e}")

    receive_task = asyncio.create_task(receive_from_client())

    async def run_session():
        async for event in gemini_client.start_session(
            audio_input_queue=audio_input_queue,
            video_input_queue=video_input_queue,
            text_input_queue=text_input_queue,
            audio_output_callback=audio_output_callback,
            audio_interrupt_callback=audio_interrupt_callback,
        ):
            if event:
                # Log transcript
                if event.get("type") == "user":
                    transcript_parts.append(f"User: {event.get('text')}")
                elif event.get("type") == "gemini":
                    transcript_parts.append(f"AI: {event.get('text')}")
                
                # Forward events to client
                await websocket.send_json(event)

    try:
        await run_session()
    except Exception as e:
        import traceback
        logger.error(f"Error in Gemini session: {type(e).__name__}: {e}\n{traceback.format_exc()}")
    finally:
        receive_task.cancel()
        
        # Update call record
        call.ended_at = datetime.utcnow()
        if call.started_at:
            call.duration = int((call.ended_at - call.started_at).total_seconds())
        call.transcript = "\n".join(transcript_parts)
        call.status = CallStatus.COMPLETED
        session.commit()
        
        logger.info(f"Call record updated: {call.id}, duration: {call.duration}s")
        
        session.close()


@app.websocket("/ws/twilio/azure")
async def twilio_azure_media_stream(websocket: WebSocket):
    """
    WebSocket endpoint for Twilio Media Stream with Azure Speech Services.
    Uses Azure STT + Azure TTS + Standard Gemini API instead of Gemini Live.
    """
    logger.info("Twilio Azure Media Stream WebSocket connection attempt...")
    await websocket.accept()
    logger.info("Twilio Azure Media Stream connection accepted")
    
    # Create database session
    session = Session(engine)
    
    # Create call record
    call = Call(
        call_sid="azure_" + str(datetime.utcnow().timestamp()),
        status=CallStatus.CONNECTED,
        started_at=datetime.utcnow()
    )
    session.add(call)
    session.commit()
    session.refresh(call)
    
    logger.info(f"Twilio Azure call record created: {call.id}")
    
    # Initialize audio bridge (24000Hz for Azure output)
    audio_bridge = TwilioAudioBridge(gemini_sample_rate=24000)
    stream_sid = None
    current_audio_rate = 24000
    
    # Call management state
    call_state = {
        "is_ai_speaking": False,
        "should_end_call": False,
    }
    
    # Initialize Azure services if credentials provided
    azure_stt_client = None
    azure_tts_client = None
    conversation_manager = None
    use_azure = False
    
    if settings.AZURE_SPEECH_KEY:
        try:
            # Initialize Azure STT
            azure_stt_client = AzureSTT(
                speech_key=settings.AZURE_SPEECH_KEY,
                speech_region=settings.AZURE_SPEECH_REGION,
                speech_endpoint=settings.AZURE_SPEECH_ENDPOINT,
                language="en-IN"
            )
            
            # Initialize Azure TTS
            azure_tts_client = AzureTTS(
                speech_key=settings.AZURE_SPEECH_KEY,
                speech_region=settings.AZURE_SPEECH_REGION,
                speech_endpoint=settings.AZURE_SPEECH_ENDPOINT,
                voice=settings.AZURE_VOICE or "en-IN-PriyaNeural"
            )
            
            # Initialize Conversation Manager with system prompt
            system_prompt = """You are Priya, a warm and friendly admission counselor from ZEE School making an outbound call.

CRITICAL - YOU ARE MAKING AN OUTBOUND CALL - READ CAREFULLY:
YOU are CALLING a parent on their phone. They are ANSWERING your call. YOU dialed their number. YOU are the caller. THEY are the receiver.

OPENING: When the person answers, say exactly this:
"Vanakkam. This is Priya from ZEE School in Hosur. I'm calling about school admissions. Summer holidays are a great time to consider school options. Which grade will your child enter after holidays? You can ask me any details about ZEE School."

NEVER EVER say anything that implies they called you:
❌ "Thank you for calling" - WRONG! You called them!
❌ "How can I help you" - WRONG! You're calling to tell them about admissions!
❌ "Welcome to ZEE School" - WRONG! You're calling them, not welcoming them!

STYLE — SHORT AND NATURAL:
- Speak like a real human, not a script
- 1-2 short sentences per turn, MAX
- Ask only ONE question at a time
- Use natural Indian English (can say "na", "actually", "so basically")
- Never list features — weave them in naturally when needed

IMPORTANT — SCHOOL INFORMATION REQUESTS:
When the user asks about the school, facilities, curriculum, or wants to know more details:
- Provide comprehensive but clear explanations (2-4 sentences)
- Cover relevant aspects: location, classrooms, teachers, facilities, transport, achievements
- Be informative but not overwhelming
- Still sound natural and conversational

FEE STRUCTURE QUESTIONS:
When user asks about fees, fee structure, or cost:
- Say: "I'll ask the academic counselor to schedule a call with you to discuss the fee structure."
- Do NOT give specific fee amounts
- Then ask: "Are you interested in visiting the campus?"

CAMPUS VISIT:
Always ask if they are interested in visiting the campus after answering their questions.

ZEE SCHOOL FACTS:
- Location: Hosur with modern infrastructure
- Academics: Smart classrooms, qualified experienced teachers, 100% board pass rate
- Facilities: Sports complex, arts & music rooms, robotics lab, library
- Transport: Bus service covering major areas
- Activities: Clubs, competitions, field trips, cultural events

CALL ENDING:
When user says "thank you", "thanks", "okay", "fine", or shows satisfaction/conclusion:
- Respond warmly with "Thank you for your time. Have a great day."
- Then stop — don't ask more questions

NEVER: list bullet points out loud, repeat yourself unnecessarily.
ALWAYS: sound warm, be brief for general questions, detailed for school queries, and recognize when to end the call."""
            
            conversation_manager = ConversationManager(
                api_key=settings.GEMINI_API_KEY,
                model="gemini-2.5-flash",
                system_instruction=system_prompt
            )
            
            use_azure = True
            logger.info("✅ Azure services initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Azure services: {e}")
            use_azure = False
    else:
        logger.warning("⚠️ Azure Speech credentials not provided, Azure integration disabled")
    
    async def twilio_audio_callback(audio_data, sample_rate=24000):
        """Send audio from Azure TTS to Twilio.
        Audio is PCM 24kHz 16-bit mono - convert to mulaw 8kHz for Twilio."""
        if not stream_sid:
            logger.warning("⚠️ stream_sid not yet available, cannot send audio to Twilio")
            return
        
        if call_state["should_end_call"]:
            logger.info("🛑 Call ending, skipping audio send")
            return
        
        try:
            # Convert PCM 24kHz → 8kHz → mulaw (same as Gemini Live pipeline)
            import audioop
            
            # Skip leading silence in PCM (all zero bytes)
            start_idx = 0
            for i in range(0, min(len(audio_data), 10000), 2):  # Check in 16-bit samples
                if i + 1 < len(audio_data):
                    sample = int.from_bytes(audio_data[i:i+2], 'little', signed=True)
                    if abs(sample) > 100:  # Non-silence threshold
                        start_idx = max(0, i - 320)  # Keep small leading silence
                        break
            
            if start_idx > 0:
                logger.info(f"📤 Skipping {start_idx} bytes of leading PCM silence")
            
            pcm_data = audio_data[start_idx:]
            
            # Downsample 24kHz → 8kHz
            pcm_8k = audioop.ratecv(pcm_data, 2, 1, sample_rate, 8000, None)[0]
            
            # Convert PCM 16-bit to mulaw
            mulaw_data = audioop.lin2ulaw(pcm_8k, 2)
            
            logger.info(f"📤 Sending {len(mulaw_data)} bytes mulaw to Twilio (from {len(pcm_data)} PCM)")
            
            # Send in 160-byte chunks (20ms at 8kHz)
            chunk_size = 160
            chunk_count = 0
            
            for i in range(0, len(mulaw_data), chunk_size):
                if call_state["should_end_call"]:
                    break
                
                chunk = mulaw_data[i:i + chunk_size]
                payload = base64.b64encode(chunk).decode('utf-8')
                
                message = {
                    "event": "media",
                    "streamSid": stream_sid,
                    "media": {
                        "payload": payload
                    }
                }
                
                try:
                    await websocket.send_json(message)
                except Exception as send_err:
                    logger.error(f"❌ WebSocket send error at chunk {chunk_count}: {send_err}")
                    break
                
                chunk_count += 1
                
                # Yield every 50 chunks (~1 second) to allow receive loop
                if chunk_count % 50 == 0:
                    await asyncio.sleep(0.01)
            
            logger.info(f"✅ Sent {chunk_count} audio chunks to Twilio")
        except Exception as e:
            logger.error(f"❌ Error sending audio to Twilio: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        finally:
            call_state["is_ai_speaking"] = False
    
    async def handle_user_speech(transcribed_text: str):
        """Handle transcribed user speech - get AI response and speak it."""
        logger.info(f"🎤 User said: {transcribed_text}")
        
        # Check for call ending
        text_lower = transcribed_text.lower()
        end_keywords = ["bye", "goodbye", "thank you", "thanks", "okay", "fine"]
        if any(keyword in text_lower for keyword in end_keywords):
            logger.info("👋 User wants to end call")
            call_state["should_end_call"] = True
            goodbye_msg = "Thank you for your time. Have a great day!"
            audio_data = azure_tts_client.text_to_speech(goodbye_msg)
            if audio_data:
                await twilio_audio_callback(audio_data)
            await asyncio.sleep(3)
            await websocket.close()
            return
        
        # Get AI response
        try:
            ai_response = conversation_manager.get_ai_response(transcribed_text)
            logger.info(f"🤖 AI response: {ai_response}")
            
            # Convert AI response to speech
            if ai_response and azure_tts_client:
                call_state["is_ai_speaking"] = True
                audio_data = azure_tts_client.text_to_speech(ai_response)
                if audio_data:
                    await twilio_audio_callback(audio_data)
                else:
                    logger.warning("⚠️ Azure TTS returned no audio")
        except Exception as e:
            logger.error(f"❌ Error handling user speech: {e}")
    
    async def receive_from_twilio():
        """Receive audio from Twilio and send to Azure STT."""
        nonlocal stream_sid
        try:
            message_count = 0
            audio_buffer = bytearray()
            
            while True:
                message = await websocket.receive_text()
                message_count += 1
                
                # Parse message to check for start event
                try:
                    data = json.loads(message)
                    if data.get("event") == "start":
                        stream_sid = data.get("start", {}).get("streamSid")
                        logger.info(f"📞 Twilio Media Stream started with streamSid: {stream_sid}")
                        
                        # Send greeting as separate task to not block receive loop
                        async def send_greeting():
                            nonlocal stream_sid
                            logger.info("⏳ Waiting 3 seconds for user to pick up phone...")
                            await asyncio.sleep(3)
                            
                            if use_azure and conversation_manager and stream_sid:
                                greeting = "Vanakkam. This is Priya from ZEE School in Hosur. I'm calling about school admissions. Summer holidays are a great time to consider school options. Which grade will your child enter after holidays? You can ask me any details about ZEE School."
                                logger.info(f"🎙️ Generating greeting audio...")
                                audio_data = azure_tts_client.text_to_speech(greeting)
                                if audio_data:
                                    logger.info(f"🎙️ Sending greeting audio ({len(audio_data)} bytes) to Twilio...")
                                    await twilio_audio_callback(audio_data)
                                    logger.info(f"🎙️ Greeting audio sent successfully")
                                else:
                                    logger.error("❌ Failed to generate greeting audio")
                        
                        asyncio.create_task(send_greeting())
                        
                except:
                    pass
                
                # Process Twilio message and convert audio
                await audio_bridge.process_twilio_message(
                    message,
                    lambda audio: audio_buffer.extend(audio)
                )
                
                # Accumulate audio and send to Azure STT periodically
                if len(audio_buffer) >= 16000 * 2:  # 1 second of audio at 16kHz
                    if use_azure and azure_stt_client:
                        try:
                            # audio_buffer already contains PCM from process_twilio_message
                            # Send directly to Azure STT
                            transcribed_text = azure_stt_client.transcribe_audio_stream(bytes(audio_buffer))
                            if transcribed_text:
                                await handle_user_speech(transcribed_text)
                        except Exception as e:
                            logger.error(f"❌ Error with Azure STT: {e}")
                    audio_buffer.clear()
                
                if message_count % 100 == 0:
                    logger.info(f"Received message #{message_count} from Twilio")
                    
        except WebSocketDisconnect:
            logger.info("Twilio Media Stream disconnected")
        except Exception as e:
            logger.error(f"Error receiving from Twilio: {e}")
    
    receive_task = asyncio.create_task(receive_from_twilio())
    
    try:
        await receive_task
    except Exception as e:
        logger.error(f"Error in Azure session: {e}")
    finally:
        receive_task.cancel()
        
        # Update call record
        call.ended_at = datetime.utcnow()
        if call.started_at:
            call.duration = int((call.ended_at - call.started_at).total_seconds())
        call.status = CallStatus.COMPLETED
        session.commit()
        
        logger.info(f"Twilio Azure call record updated: {call.id}, duration: {call.duration}s")
        session.close()



        
        # Ensure websocket is closed if not already
        try:
            await websocket.close()
        except:
            pass


@app.websocket("/ws/daily")
async def daily_websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for Daily.co SIP/WebRTC connections.
    Handles real-time bidirectional audio with Gemini Live.
    """
    await websocket.accept()
    
    # Get connection details
    caller_id = websocket.query_params.get("caller_id", "unknown")
    room_name = websocket.query_params.get("room", settings.DAILY_ROOM_NAME)
    
    logger.info(f"🔌 Daily WebSocket connection accepted")
    logger.info(f"   Caller: {caller_id}")
    logger.info(f"   Room: {room_name}")
    logger.info(f"   Client: {websocket.client}")

    # Create database session
    session = Session(engine)
    
    # Create call record
    call = Call(
        call_sid=f"daily_{datetime.utcnow().timestamp()}",
        from_number=caller_id,
        status=CallStatus.CONNECTED,
        started_at=datetime.utcnow()
    )
    session.add(call)
    session.commit()
    session.refresh(call)
    
    logger.info(f"Daily call record created: {call.id}")

    audio_input_queue = asyncio.Queue()
    video_input_queue = asyncio.Queue()
    text_input_queue = asyncio.Queue()
    
    transcript_parts = []
    
    # Track AI speaking state for interruption
    call_state = {
        "is_ai_speaking": False,
    }

    async def audio_output_callback(data):
        """Send AI audio back to Daily."""
        call_state["is_ai_speaking"] = True
        await websocket.send_bytes(data)
        call_state["is_ai_speaking"] = False

    async def audio_interrupt_callback():
        """Handle user interrupt - stop AI from speaking."""
        logger.info("🛑 User interrupted AI in Daily WebSocket")
        # Send interrupt signal to Gemini Live to stop current generation
        await gemini_client.interrupt()
        call_state["is_ai_speaking"] = False

    # Get tool mapping
    tool_mapping = get_receptionist_tool_mapping(session, call.id)
    tool_definitions = get_receptionist_tool_definitions()

    gemini_client = GeminiLive(
        api_key=settings.GEMINI_API_KEY, 
        model=settings.MODEL, 
        input_sample_rate=16000, 
        voice=settings.VOICE,
        tools=tool_definitions,
        tool_mapping=tool_mapping
    )

    async def receive_from_daily():
        """Receive audio from Daily WebSocket."""
        try:
            while True:
                message = await websocket.receive()
                
                if message.get("bytes"):
                    # Audio data from Daily (16kHz PCM)
                    await audio_input_queue.put(message["bytes"])
                elif message.get("text"):
                    # Text commands if any
                    text = message["text"]
                    await text_input_queue.put(text)
        except WebSocketDisconnect:
            logger.info("Daily WebSocket disconnected")
        except Exception as e:
            logger.error(f"Error receiving from Daily: {e}")

    receive_task = asyncio.create_task(receive_from_daily())

    async def run_session():
        async for event in gemini_client.start_session(
            audio_input_queue=audio_input_queue,
            video_input_queue=video_input_queue,
            text_input_queue=text_input_queue,
            audio_output_callback=audio_output_callback,
            audio_interrupt_callback=audio_interrupt_callback,
        ):
            if event:
                # Process user speech for interruption detection
                if event.get("type") == "user":
                    transcript_parts.append(f"User: {event.get('text')}")
                    # Check for interruption - if AI is speaking, trigger interrupt callback
                    if call_state["is_ai_speaking"]:
                        logger.info("🛑 User interrupted AI speech in Daily WebSocket")
                        await audio_interrupt_callback()
                elif event.get("type") == "gemini":
                    transcript_parts.append(f"AI: {event.get('text')}")

    try:
        await run_session()
    except Exception as e:
        logger.error(f"Error in Daily Gemini session: {e}")
    finally:
        receive_task.cancel()
        
        # Update call record
        call.ended_at = datetime.utcnow()
        if call.started_at:
            call.duration = int((call.ended_at - call.started_at).total_seconds())
        call.transcript = "\n".join(transcript_parts)
        call.status = CallStatus.COMPLETED
        session.commit()
        
        logger.info(f"Daily call completed: {call.id}, duration: {call.duration}s")
        session.close()
        
        try:
            await websocket.close()
        except:
            pass


@app.post("/daily/setup")
async def setup_daily_room():
    """
    Setup Daily.co room with SIP dial-in enabled.
    Returns SIP URI for phone calls.
    """
    from daily_client import DailyClient
    
    if not settings.DAILY_API_KEY:
        return {"error": "Daily API key not configured"}
    
    daily = DailyClient()
    
    # Create room
    room_result = await daily.create_room()
    if "error" in room_result:
        return room_result
    
    # Enable SIP dial-in
    sip_result = await daily.enable_sip_dialin()
    if "error" in sip_result:
        return sip_result
    
    return {
        "success": True,
        "room_name": room_result.get("name"),
        "sip_uri": sip_result.get("sip_uri"),
        "room_url": room_result.get("url"),
        "message": "Configure your Twilio number to forward to this SIP URI"
    }


@app.get("/daily/sip-info")
async def get_daily_sip_info():
    """Get SIP information for the configured room."""
    from daily_client import DailyClient
    
    if not settings.DAILY_API_KEY:
        return {"error": "Daily API key not configured"}
    
    daily = DailyClient()
    sip_result = await daily.enable_sip_dialin()
    
    return sip_result


@app.post("/twiml/sip-dial")
async def twiml_sip_dial(request: Request):
    """
    Return TwiML to dial Daily SIP URI.
    Use this URL in your Twilio TwiML App.
    Twilio makes POST requests to webhooks.
    """
    # Get call info from Twilio
    form_data = await request.form()
    call_sid = form_data.get("CallSid", "unknown")
    from_number = form_data.get("From", "unknown")
    to_number = form_data.get("To", "unknown")
    
    # Daily SIP URI from Pipecat Cloud
    daily_domain = "6e418e1622aa4b9094883b68923186cd"
    room_name = settings.DAILY_ROOM_NAME
    sip_uri = f"sip:{room_name}@{daily_domain}.sip.daily.co"
    
    logger.info(f"📞 Twilio call {call_sid} from {from_number} -> routing to Daily SIP: {sip_uri}")
    
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>Connecting you to the A I receptionist. Please wait.</Say>
    <Dial>
        <Sip>{sip_uri}</Sip>
    </Dial>
</Response>"""
    
    return Response(content=twiml, media_type="application/xml")


@app.get("/twiml/sip-dial")
async def twiml_sip_dial_get():
    """GET version for testing in browser."""
    daily_domain = "6e418e1622aa4b9094883b68923186cd"
    room_name = settings.DAILY_ROOM_NAME
    sip_uri = f"sip:{room_name}@{daily_domain}.sip.daily.co"
    
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Dial>
        <Sip>{sip_uri}</Sip>
    </Dial>
</Response>"""
    
    return Response(content=twiml, media_type="application/xml")


@app.post("/twiml/incoming")
async def twiml_incoming(request: Request):
    """
    Initial TwiML for incoming calls.
    This is what the TwiML App should point to.
    """
    # Get call info
    form_data = await request.form()
    call_sid = form_data.get("CallSid", "unknown")
    from_number = form_data.get("From", "unknown")
    
    logger.info(f"📞 Incoming call {call_sid} from {from_number}")
    
    # Get server URL from request
    server_url = str(request.base_url).rstrip('/')
    
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Aditi">Hi! This is Priya from ZEE School. Do you have a school-going child?</Say>
    <Gather action="{server_url}/twiml/gather" method="POST" input="speech" timeout="15" language="en-IN" speechTimeout="auto">
    </Gather>
    <Redirect>{server_url}/twiml/gather</Redirect>
</Response>"""
    
    return Response(content=twiml, media_type="application/xml")


@app.get("/twiml/incoming")
async def twiml_incoming_get():
    """GET version for browser testing."""
    twiml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Aditi">Hi! This is Priya from ZEE School. Do you have a school-going child?</Say>
    <Gather action="/twiml/gather" method="POST" input="speech" timeout="15" language="en-IN" speechTimeout="auto">
    </Gather>
    <Redirect>/twiml/gather</Redirect>
</Response>"""
    
    return Response(content=twiml, media_type="application/xml")


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="localhost", port=port)
