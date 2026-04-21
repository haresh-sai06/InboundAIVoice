"""
Twilio + Sarvam AI Voice Server
Handles outbound calls with Sarvam STT, TTS, and LLM (FREE)
No storage - just call processing
"""

import asyncio
import base64
import json
import logging
import os
import audioop
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import websockets
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load env
from dotenv import load_dotenv
load_dotenv(".env")

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY", "")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "centralindia")

# ═══════════════════════════════════════════════════════════════════════════════
# DEEPGRAM STT CLIENT (Low Latency)
# ═══════════════════════════════════════════════════════════════════════════════

class DeepgramSTT:
    """Fast STT using Deepgram Nova-2"""
    
    BASE_URL = "https://api.deepgram.com/v1/listen"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Token {api_key}",
            "Content-Type": "audio/wav"
        }
    
    async def transcribe(self, audio_bytes: bytes, language: str = "en-IN") -> str:
        """Transcribe audio to text with enhanced settings"""
        import aiohttp
        
        # Enhanced parameters for better Indian English recognition
        params = {
            "model": "nova-2",  # Best for multilingual/Indian English
            "language": language,
            "smart_format": "true",
            "punctuate": "true",
            "diarize": "false",
            "ner": "false",
            "utt_split": "0.5",  # Shutter utterance splitting
            "filler_words": "false",
            "numerals": "false"
        }
        
        # Build URL with params
        query = "&".join([f"{k}={v}" for k, v in params.items()])
        url = f"{self.BASE_URL}?{query}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=audio_bytes, headers=self.headers, timeout=10) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        channels = result.get("results", {}).get("channels", [])
                        if channels:
                            alternatives = channels[0].get("alternatives", [{}])
                            transcript = alternatives[0].get("transcript", "")
                            # Log confidence if available
                            confidence = alternatives[0].get("confidence", 0)
                            logger.info(f"🎤 Deepgram confidence: {confidence:.2f}")
                            return transcript
                        return ""
                    else:
                        error_text = await resp.text()
                        logger.error(f"Deepgram STT error: {resp.status} - {error_text[:200]}")
                        return ""
        except Exception as e:
            logger.error(f"Deepgram STT failed: {e}")
            return ""


# ═══════════════════════════════════════════════════════════════════════════════
# AZURE TTS CLIENT (Low Latency, Indian Voices)
# ═══════════════════════════════════════════════════════════════════════════════

class AzureTTS:
    """Fast TTS using Azure Speech Services with Indian voices"""
    
    def __init__(self, subscription_key: str, region: str = "centralindia"):
        self.subscription_key = subscription_key
        self.region = region
    
    async def text_to_speech(self, text: str, voice: str = "en-IN-NeerjaNeural") -> bytes:
        """Convert text to speech using Azure with SSML for expressiveness"""
        import asyncio
        
        def _synthesize():
            """Run Azure TTS in a thread with SSML"""
            try:
                import azure.cognitiveservices.speech as speechsdk
                
                speech_config = speechsdk.SpeechConfig(
                    subscription=self.subscription_key, 
                    region=self.region
                )
                speech_config.speech_synthesis_voice_name = voice
                
                synthesizer = speechsdk.SpeechSynthesizer(
                    speech_config=speech_config, 
                    audio_config=None
                )
                
                # Build SSML for mature, professional Indian-accented speech
                # Reduced gaps: comma=0ms break, mature style, warm but professional
                # Replace periods with shorter breaks to reduce gaps between sentences
                processed_text = text.replace('. ', '<break time="50ms"/>').replace('.', '<break time="50ms"/>')
                
                ssml = f'''<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' 
                    xmlns:mstts='http://www.w3.org/2001/mstts' xml:lang='en-IN'>
                    <voice name='{voice}'>
                        <mstts:express-as style='friendly' styledegree='1.2' role='YoungAdultFemale'>
                            <prosody rate='1.2' pitch='+8%' volume='+5%' range='+10%'>
                                {processed_text}
                            </prosody>
                        </mstts:express-as>
                    </voice>
                </speak>'''
                
                result = synthesizer.speak_ssml_async(ssml).get()
                
                if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                    return result.audio_data
                else:
                    logger.error(f"Azure TTS failed: {result.reason}")
                    return b""
                    
            except Exception as e:
                logger.error(f"Azure TTS error: {e}")
                return b""
        
        # Run in thread pool with 8 second timeout
        try:
            loop = asyncio.get_event_loop()
            return await asyncio.wait_for(
                loop.run_in_executor(None, _synthesize),
                timeout=8.0
            )
        except asyncio.TimeoutError:
            logger.error("Azure TTS timed out")
            return b""


# ═══════════════════════════════════════════════════════════════════════════════
# SARVAM API CLIENT (Fallback)
# ═══════════════════════════════════════════════════════════════════════════════

class SarvamClient:
    """Simple client for Sarvam AI APIs"""
    
    BASE_URL = "https://api.sarvam.ai"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "api-subscription-key": api_key,
            "Content-Type": "application/json"
        }
    
    async def speech_to_text(self, audio_bytes: bytes, language: str = "unknown") -> str:
        """Convert audio to text using Sarvam STT"""
        import aiohttp
        
        # Convert audio if needed (Sarvam expects specific format)
        # Sarvam accepts wav, mp3, etc.
        
        url = f"{self.BASE_URL}/speech-to-text"
        
        # Prepare multipart form data
        data = aiohttp.FormData()
        data.add_field('file', audio_bytes, filename='audio.wav', content_type='audio/wav')
        data.add_field('language_code', language if language else "unknown")
        data.add_field('model', 'saaras:v3')
        
        headers = {"api-subscription-key": self.api_key}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=data, headers=headers) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        return result.get("transcript", "")
                    else:
                        logger.error(f"STT error: {resp.status} - {await resp.text()}")
                        return ""
        except Exception as e:
            logger.error(f"STT request failed: {e}")
            return ""
    
    async def text_to_speech(self, text: str, language: str = "hi-IN", voice: str = "kavya") -> bytes:
        """Convert text to speech using Sarvam TTS"""
        import aiohttp
        
        url = f"{self.BASE_URL}/text-to-speech"
        
        payload = {
            "inputs": [text],
            "target_language_code": language,
            "speaker": voice,
            "model": "bulbul:v3",  # Use v3 for kavya voice
            "speech_sample_rate": 16000  # Better quality, we'll downsample
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=self.headers) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        # Decode base64 audio
                        audio_b64 = result.get("audios", [""])[0]
                        return base64.b64decode(audio_b64) if audio_b64 else b""
                    else:
                        logger.error(f"TTS error: {resp.status} - {await resp.text()}")
                        return b""
        except Exception as e:
            logger.error(f"TTS request failed: {e}")
            return b""


class GeminiClient:
    """Simple client for Gemini API via REST"""
    
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        logger.info("✅ Gemini REST client initialized")
    
    async def chat_completion(self, messages: list, model: str = "gemini-1.5-flash") -> str:
        """Get chat completion from Gemini via REST API"""
        import aiohttp
        
        # Convert messages to Gemini format
        contents = []
        system_instruction = None
        
        for msg in messages:
            role = msg["role"]
            if role == "system":
                system_instruction = msg["content"]
                continue
            elif role == "assistant":
                role = "model"
            
            contents.append({
                "role": role,
                "parts": [{"text": msg["content"]}]
            })
        
        # Add system instruction as first user message
        if system_instruction:
            contents.insert(0, {
                "role": "user",
                "parts": [{"text": f"System instruction: {system_instruction}"}]
            })
        
        url = f"{self.BASE_URL}/models/{model}:generateContent?key={self.api_key}"
        
        payload = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": 500,
                "temperature": 0.7
            }
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=15) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        candidates = result.get("candidates", [])
                        if candidates:
                            content = candidates[0].get("content", {})
                            parts = content.get("parts", [{}])
                            response_text = parts[0].get("text", "").strip()
                            if response_text:
                                # Enforce word limit - truncate to 12 words max
                                words = response_text.split()
                                if len(words) > 12:
                                    logger.warning(f"🤖 AI response too long ({len(words)} words), truncating to 12")
                                    response_text = " ".join(words[:12])
                                logger.info(f"🤖 AI response: {response_text[:50]}...")
                                return response_text
                        return ""
                    else:
                        error_text = await resp.text()
                        logger.error(f"Gemini API error: {resp.status} - {error_text[:200]}")
                        return "I apologize, I'm having technical difficulties."
        except Exception as e:
            logger.error(f"Gemini API failed: {e}")
            return "I apologize, I'm having technical difficulties."


# ═══════════════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ═══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Sarvam AI Voice Server starting...")
    if not SARVAM_API_KEY:
        logger.warning("⚠️ SARVAM_API_KEY not set! Calls will fail.")
    else:
        logger.info("✅ Sarvam API configured")
    yield
    logger.info("🛑 Server shutting down...")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize clients
stt_client = DeepgramSTT(DEEPGRAM_API_KEY) if DEEPGRAM_API_KEY else None
tts_client = AzureTTS(AZURE_SPEECH_KEY, AZURE_SPEECH_REGION) if AZURE_SPEECH_KEY else None
gemini = GeminiClient(GEMINI_API_KEY) if GEMINI_API_KEY else None

# Fallback to Sarvam if new keys not available
sarvam_fallback = SarvamClient(SARVAM_API_KEY) if SARVAM_API_KEY else None

# Log which services are active
logger.info(f"🔊 STT: {'Deepgram' if stt_client else 'Sarvam (fallback)' if sarvam_fallback else 'None'}")
logger.info(f"🗣️  TTS: {'Azure' if tts_client else 'Sarvam (fallback)' if sarvam_fallback else 'None'}")
logger.info(f"🤖 LLM: {'Gemini' if gemini else 'None'}")

# ═══════════════════════════════════════════════════════════════════════════════
# ZEE SCHOOL AGENT CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are Priya, a professional admission counselor from ZEE School in Hosur.

CRITICAL RULE - WORD COUNT ENFORCEMENT:
You MUST count words in your response. NEVER exceed 12 words. If your response is longer than 12 words, you have FAILED.

GOOD EXAMPLES (under 12 words):
- "Which grade is your child in?"
- "We have excellent sports facilities."
- "Would you like to visit?"
- "Our teachers are very experienced."
- "Bus service covers your area."

BAD EXAMPLES (too long - NEVER DO THIS):
- "Certainly! To help me understand better, could you please tell me which grade level you are considering for your child?" (22 words - TOO LONG)
- "Of course. To provide the most relevant information, could you please share your child's current grade level and what you are looking for in a school?" (24 words - TOO LONG)

CONVERSATION FLOW:
1. GREET: "Hi! Do you have a school-going child?"
2. DISCOVER: "Which grade?" / "What are you looking for?"
3. INFORM: Share ONE fact in 5-6 words
4. ENGAGE: "Would you like to visit?"
5. CLOSE: "I'll have our counselor call you."

FEES RESPONSE (use exactly):
"I'll connect you with our academic counselor for fees."

GOODBYE RESPONSE (use exactly):
"Thank you! Have a wonderful day!"

ZEE SCHOOL FACTS (pick ONE, 4-6 words):
- Smart classrooms and labs
- 100% board exam pass
- Sports and robotics facilities
- Bus service available
- Scholarships for merit students

RULE: Before responding, count your words. If more than 12, delete words until under 12."""


def build_messages(user_text: str, conversation_history: list = None) -> list:
    """Build message list for Sarvam LLM"""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    if conversation_history:
        messages.extend(conversation_history)
    
    messages.append({"role": "user", "content": user_text})
    return messages


# ═══════════════════════════════════════════════════════════════════════════════
# TWILIO MEDIA STREAM HANDLER
# ═══════════════════════════════════════════════════════════════════════════════

class TwilioAudioBridge:
    """Handles audio conversion between Twilio (mulaw 8kHz) and PCM"""
    
    @staticmethod
    def decode_mulaw(mulaw_data: bytes) -> bytes:
        """Decode mulaw to PCM 16-bit"""
        return audioop.ulaw2lin(mulaw_data, 2)
    
    @staticmethod
    def encode_mulaw(pcm_data: bytes) -> bytes:
        """Encode PCM 16-bit to mulaw"""
        return audioop.lin2ulaw(pcm_data, 2)
    
    @staticmethod
    def resample(pcm_data: bytes, from_rate: int, to_rate: int) -> bytes:
        """Resample PCM audio"""
        if from_rate == to_rate:
            return pcm_data
        return audioop.ratecv(pcm_data, 2, 1, from_rate, to_rate, None)[0]
    
    @staticmethod
    def create_wav_header(data_len: int, sample_rate: int = 16000, channels: int = 1, bits: int = 16) -> bytes:
        """Create WAV header for raw PCM data"""
        byte_rate = sample_rate * channels * bits // 8
        block_align = channels * bits // 8
        
        header = b'RIFF'
        header += (data_len + 36).to_bytes(4, 'little')  # File size - 8
        header += b'WAVE'
        header += b'fmt '
        header += (16).to_bytes(4, 'little')  # Subchunk size
        header += (1).to_bytes(2, 'little')     # Audio format (PCM)
        header += channels.to_bytes(2, 'little')
        header += sample_rate.to_bytes(4, 'little')
        header += byte_rate.to_bytes(4, 'little')
        header += block_align.to_bytes(2, 'little')
        header += bits.to_bytes(2, 'little')
        header += b'data'
        header += data_len.to_bytes(4, 'little')
        
        return header


@app.websocket("/ws/twilio/sarvam")
async def twilio_sarvam_stream(websocket: WebSocket):
    """
    WebSocket endpoint for Twilio Media Stream with Sarvam AI.
    Bidirectional audio: Twilio ↔ Sarvam
    """
    await websocket.accept()
    logger.info("📞 Twilio Media Stream connected (Sarvam mode)")
    
    stream_sid = None
    conversation_history = []
    is_speaking = False
    should_end_call = False
    is_processing = False
    last_ai_speak_time = 0
    
    # VAD (Voice Activity Detection) state
    vad_state = {
        'is_listening': False,      # Currently detecting speech
        'speech_buffer': bytearray(), # Buffer for current utterance
        'silence_frames': 0,          # Consecutive silent frames
        'speech_frames': 0,         # Consecutive speech frames
        'last_process_time': 0
    }
    
    import time
    
    async def process_captured_speech(pcm_data: bytes):
        """Process captured speech from VAD using Deepgram + Azure"""
        nonlocal is_speaking, conversation_history, should_end_call, last_ai_speak_time
        
        # Skip if currently speaking (avoid interrupting ourselves)
        if is_speaking:
            return
        
        logger.info("🎤 Sending to Deepgram STT...")
        
        # Add WAV header for STT (expects WAV format)
        wav_header = TwilioAudioBridge.create_wav_header(len(pcm_data), sample_rate=8000, channels=1, bits=16)
        wav_data = wav_header + pcm_data
        
        # STT with Deepgram primary, Sarvam fallback if suspicious result
        transcript = ""
        
        if stt_client:
            result = await stt_client.transcribe(wav_data, language="en-IN")
            if result:
                transcript = result
                # Check if result seems wrong (contains nonsense words)
                suspicious_words = ["mhmm", "uhh", "umm", "right?", "okay?", "rule", "going", "time"]
                if any(word in transcript.lower() for word in suspicious_words):
                    # Check average word length
                    words = transcript.split()
                    if words:
                        avg_len = sum(len(w) for w in words) / len(words)
                        if avg_len < 3.0:
                            logger.warning(f"Deepgram seems confused (words: {transcript}), trying Sarvam")
                            transcript = ""
        
        # Fallback to Sarvam if no result or seems wrong
        if not transcript and sarvam_fallback:
            logger.info("🎤 Falling back to Sarvam STT...")
            transcript = await sarvam_fallback.speech_to_text(wav_data, language="en-IN")
        
        if not transcript:
            logger.error("No speech detected")
            return
        
        if not transcript:
            logger.info("🤷 No speech detected")
            return
        
        logger.info(f"🎤 User said: {transcript}")
        
        # Check for end call
        if any(word in transcript.lower() for word in ["bye", "goodbye", "thank you", "thanks"]):
            logger.info("👋 User wants to end call")
            # Send goodbye message
            goodbye_text = "Thank you for your time. Have a great day!"
            
            if tts_client:
                audio_data = await tts_client.text_to_speech(goodbye_text, voice="en-IN-NeerjaNeural")
                if not audio_data and sarvam_fallback:
                    logger.warning("Azure TTS failed, falling back to Sarvam")
                    audio_data = await sarvam_fallback.text_to_speech(goodbye_text, language="en-IN", voice="kavya")
            elif sarvam_fallback:
                audio_data = await sarvam_fallback.text_to_speech(goodbye_text, language="en-IN", voice="kavya")
            else:
                audio_data = None
            if audio_data and stream_sid:
                is_speaking = True
                await send_audio_to_twilio(audio_data)
                is_speaking = False
                # Wait for audio to play (rough estimate: 2-3 seconds)
                await asyncio.sleep(3)
            should_end_call = True  # Signal main loop to end call
            last_ai_speak_time = time.time()
        
        # Get AI response from Gemini LLM
        messages = build_messages(transcript, conversation_history)
        
        logger.info("🤖 Getting response from Gemini LLM...")
        if gemini:
            ai_response = await gemini.chat_completion(messages, model="gemini-2.5-flash")
        else:
            ai_response = "I apologize, I'm having trouble responding. Please try again."
        
        logger.info(f"🤖 AI said: {ai_response}")
        
        # Update conversation history
        conversation_history.append({"role": "user", "content": transcript})
        conversation_history.append({"role": "assistant", "content": ai_response})
        
        # Limit history to last 10 exchanges
        if len(conversation_history) > 20:
            conversation_history = conversation_history[-20:]
        
        # TTS with Azure (or fallback to Sarvam)
        logger.info("🔊 Converting to speech with Azure...")
        
        if tts_client:
            # Use Azure TTS with Indian voice (Neerja or Priya)
            audio_data = await tts_client.text_to_speech(ai_response, voice="en-IN-NeerjaNeural")
            if not audio_data and sarvam_fallback:
                logger.warning("Azure TTS failed, falling back to Sarvam")
                audio_data = await sarvam_fallback.text_to_speech(ai_response, language="en-IN", voice="kavya")
        elif sarvam_fallback:
            audio_data = await sarvam_fallback.text_to_speech(ai_response, language="en-IN", voice="kavya")
        else:
            logger.error("No TTS client available")
            return
        
        if audio_data and stream_sid:
            is_speaking = True
            await send_audio_to_twilio(audio_data)
            is_speaking = False
            last_ai_speak_time = time.time()
    
    async def send_audio_to_twilio(audio_data: bytes):
        """Send audio back to Twilio"""
        try:
            # Azure returns WAV with header, Sarvam returns raw PCM
            # Strip WAV header if present (starts with 'RIFF')
            if audio_data[:4] == b'RIFF':
                # Find 'data' chunk and skip header (44 bytes typically)
                data_pos = audio_data.find(b'data')
                if data_pos != -1:
                    # Skip 'data' marker (4 bytes) + size (4 bytes)
                    pcm_data = audio_data[data_pos + 8:]
                else:
                    pcm_data = audio_data[44:]  # Default WAV header size
            else:
                pcm_data = audio_data
            
            # Resample from 16kHz to 8kHz for Twilio, then encode to mulaw
            pcm_8khz = TwilioAudioBridge.resample(pcm_data, 16000, 8000)
            mulaw_data = TwilioAudioBridge.encode_mulaw(pcm_8khz)
            
            # Send in 320-byte chunks (40ms at 8kHz) - larger chunks for smoother playback
            chunk_size = 320
            for i in range(0, len(mulaw_data), chunk_size):
                chunk = mulaw_data[i:i + chunk_size]
                payload = base64.b64encode(chunk).decode('utf-8')
                
                message = {
                    "event": "media",
                    "streamSid": stream_sid,
                    "media": {"payload": payload}
                }
                
                await websocket.send_json(message)
                
                # Longer delay between chunks for smoother playback (20ms)
                await asyncio.sleep(0.02)
            
            logger.info(f"✅ Sent {len(mulaw_data)} bytes to Twilio")
            
            # Send mark message to indicate audio finished
            mark_message = {
                "event": "mark",
                "streamSid": stream_sid,
                "mark": {"name": "audio_finished"}
            }
            await websocket.send_json(mark_message)
            
        except Exception as e:
            logger.error(f"Error sending audio: {e}")
    
    try:
        while True:
            message = await websocket.receive_text()
            
            try:
                data = json.loads(message)
                event_type = data.get("event")
                
                if event_type == "start":
                    stream_sid = data.get("start", {}).get("streamSid")
                    logger.info(f"📞 Stream started: {stream_sid}")
                    
                    # Wait 1 second for connection to stabilize
                    logger.info("⏳ Starting call...")
                    await asyncio.sleep(1)
                    
                    greeting = "Vanakkam! This is Priya from ZEE School in Hosur. I'm calling about school admissions. Do you have a school-going child?"
                    logger.info(f"🎙️ Sending greeting: {greeting}")
                    
                    # Use Azure TTS for greeting (or fallback)
                    if tts_client:
                        audio_data = await tts_client.text_to_speech(greeting, voice="en-IN-NeerjaNeural")
                        if not audio_data and sarvam_fallback:
                            logger.warning("Azure TTS failed, falling back to Sarvam")
                            audio_data = await sarvam_fallback.text_to_speech(greeting, language="en-IN", voice="kavya")
                    elif sarvam_fallback:
                        audio_data = await sarvam_fallback.text_to_speech(greeting, language="en-IN", voice="kavya")
                    else:
                        audio_data = None
                    if audio_data:
                        is_speaking = True
                        await send_audio_to_twilio(audio_data)
                        is_speaking = False
                        last_ai_speak_time = time.time()
                        logger.info("✅ Greeting sent successfully")
                
                elif event_type == "media":
                    # Skip if AI is speaking or still processing (prevent self-interruption)
                    if is_speaking or is_processing:
                        # Reset VAD state to discard echo
                        vad_state['speech_buffer'].clear()
                        vad_state['is_listening'] = False
                        vad_state['silence_frames'] = 0
                        vad_state['speech_frames'] = 0
                        continue
                    
                    # Skip if AI just spoke (0.5 second cooldown for faster response)
                    if time.time() - last_ai_speak_time < 0.5:
                        continue
                    
                    # Receive audio from Twilio
                    payload = data.get("media", {}).get("payload", "")
                    mulaw_chunk = base64.b64decode(payload)
                    
                    # Decode mulaw to PCM
                    pcm_chunk = TwilioAudioBridge.decode_mulaw(mulaw_chunk)
                    
                    # VAD: Check energy level
                    try:
                        rms = audioop.rms(pcm_chunk, 2)
                        is_speech = rms >= 500  # Speech threshold
                    except:
                        is_speech = True  # Assume speech if can't calculate
                    
                    # VAD State Machine
                    if is_speech:
                        vad_state['speech_frames'] += 1
                        vad_state['silence_frames'] = 0
                        
                        # Start listening if we have enough consecutive speech (100ms = 2 frames)
                        if vad_state['speech_frames'] >= 2:
                            vad_state['is_listening'] = True
                            
                        # If listening, add to speech buffer
                        if vad_state['is_listening']:
                            vad_state['speech_buffer'].extend(pcm_chunk)
                            
                    else:  # Silence
                        vad_state['silence_frames'] += 1
                        vad_state['speech_frames'] = 0
                        
                        # If listening and silence for ~1.5 seconds, process speech
                        if vad_state['is_listening'] and vad_state['silence_frames'] >= 75:  # ~1.5 seconds of silence
                            if len(vad_state['speech_buffer']) >= 4000 and not is_processing:  # At least 0.25s of speech
                                is_processing = True
                                try:
                                    # Copy buffer and clear
                                    speech_data = bytes(vad_state['speech_buffer'])
                                    vad_state['speech_buffer'].clear()
                                    vad_state['is_listening'] = False
                                    vad_state['silence_frames'] = 0
                                    
                                    # Process the captured speech
                                    await process_captured_speech(speech_data)
                                finally:
                                    is_processing = False
                            else:
                                # Too short, discard
                                vad_state['speech_buffer'].clear()
                                vad_state['is_listening'] = False
                                vad_state['silence_frames'] = 0
                
                # Check if we should end the call
                if should_end_call:
                    logger.info("📞 Ending call as requested")
                    break
                
                elif event_type == "stop":
                    logger.info("📞 Stream stopped")
                    break
                    
            except json.JSONDecodeError:
                logger.warning(f"Received non-JSON message: {message[:100]}")
                
    except WebSocketDisconnect:
        logger.info("📞 Twilio WebSocket disconnected")
    except Exception as e:
        logger.error(f"Error in WebSocket: {e}")
    finally:
        logger.info("📞 Call ended")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Twilio + Deepgram STT + Azure TTS + Gemini LLM Server",
        "stt": "Deepgram" if stt_client else "Sarvam (fallback)" if sarvam_fallback else "None",
        "tts": "Azure" if tts_client else "Sarvam (fallback)" if sarvam_fallback else "None",
        "llm": "Gemini" if gemini else "None",
        "latency_target": "~800ms with Deepgram+Azure vs ~3s with Sarvam"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
