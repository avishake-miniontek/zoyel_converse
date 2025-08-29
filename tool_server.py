#!/usr/bin/env python3
"""
Voice AI Server - Complete WebSocket server with OpenAI-compatible API integration
Features:
- Multi-client session management
- Database storage for messages
- Voice processing (Whisper + Kokoro TTS)
- Tool call handling
- Model Context Protocol (MCP)
- Reconnection handling
- Error recovery
"""

import asyncio
import websockets
import json
import sqlite3
import uuid
import logging
import traceback
import time
import io
import base64
import wave
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager
import aiohttp
import threading
from pathlib import Path
from tools import TOOL_FUNCTIONS, TOOL_SCHEMAS

# External dependencies (install with: pip install openai transformers torch torchaudio kokoro-tts)
try:
    from openai import AsyncOpenAI
    from transformers import pipeline
    import torch
    import torchaudio
    from kokoro import KPipeline
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install openai transformers torch torchaudio kokoro-tts")
    exit(1)

# Register SQLite adapters for datetime
sqlite3.register_adapter(datetime, lambda val: val.isoformat())
sqlite3.register_converter("TIMESTAMP", lambda val: datetime.fromisoformat(val.decode('utf-8') if isinstance(val, bytes) else val))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('voice_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# System instruction to be applied to every chat completion
SYSTEM_PROMPT = """
You are Zoya, a helpful medical AI assistant who communicates through voice. Generally its a patient that you will be talking to.

You are part of a secure medical software system with strict privacy and non-disclosure policies. Collecting and storing patient data is necessary and safe for providing accurate medical assistance, differential diagnosis, and recalling information later.

At the beginning of a new conversation, or if patient data hasn't been collected yet, politely greet the patient and start asking for their personal and medical information step by step in a natural, conversational way. Gather details including age (in years, months, days), gender, city, country, current date, complaints (with symptom, severity, and since when for each), vitals (weight in kg, height in cm, blood pressure systolic/diastolic, temperature in C, heart rate, respiration rate, SpO2, LMP if applicable), physical examination findings, comorbidities, past history (illnesses and procedures), current medications, family history, allergies (drug and food), test documents, and test results. Ask follow-up questions to fill in any missing details or clarify ambiguities. Keep lists empty if no information is available. Once you have all required details, use the save_patient_data tool to store it without mentioning the tool to the user.

If the user asks to recall or confirm any of their previously provided information, use the fetch_patient_data tool to retrieve only the relevant fields and share them naturally in your response.

You have access to tools that can help answer questions. These tools will be executed automatically when needed with or without paramters depending on the use case. 
You should never output tool calls, code blocks, or markdown — only respond naturally with the results.

For example:
If the user asks for the current time, and the tool provides "10:22 AM", you should simply say:
"The current time is 10:22 AM."

After the tool executes, you'll see the result and can respond to the user.

Important instructions for your responses:
1) Provide only plain text that will be converted to speech - never use markdown, asterisk *, code blocks, or special formatting in your final response.
2) Use natural, conversational language as if you're speaking to someone.
3) Never use bullet points, numbered lists, or special characters in your final response.
4) Keep responses concise and clear since they will be spoken aloud.
5) Express lists or multiple points in a natural spoken way using words like 'first', 'also', 'finally', etc.
6) Use punctuation only for natural speech pauses (periods, commas, question marks).
7) Never show the tool calls to the user - they are for internal use only
8) After a tool executes, respond naturally with the result
"""

@dataclass
class Message:
    id: str
    session_id: str
    role: str  # 'user', 'assistant', 'system'
    content: str
    timestamp: datetime
    message_type: str  # 'text', 'audio', 'tool_call', 'tool_response'
    metadata: Optional[Dict] = None
    audio_data: Optional[bytes] = None

@dataclass
class Session:
    id: str
    user_id: str
    created_at: datetime
    last_activity: datetime
    context: List[Dict]
    model_name: str
    temperature: float = 0.3
    max_tokens: int = 256
    context_max_tokens: int = 128000
    is_active: bool = True

class DatabaseManager:
    def __init__(self, db_path: str = "voice_ai.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database with required tables"""
        with sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    created_at TIMESTAMP,
                    last_activity TIMESTAMP,
                    context TEXT,
                    model_name TEXT,
                    temperature REAL,
                    max_tokens INTEGER,
                    is_active BOOLEAN
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    session_id TEXT,
                    role TEXT,
                    content TEXT,
                    timestamp TIMESTAMP,
                    message_type TEXT,
                    metadata TEXT,
                    audio_data BLOB,
                    FOREIGN KEY (session_id) REFERENCES sessions (id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tool_calls (
                    id TEXT PRIMARY KEY,
                    session_id TEXT,
                    message_id TEXT,
                    function_name TEXT,
                    arguments TEXT,
                    result TEXT,
                    timestamp TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions (id),
                    FOREIGN KEY (message_id) REFERENCES messages (id)
                )
            """)
            
            conn.commit()
    
    def save_session(self, session: Session):
        """Save or update session in database"""
        with sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO sessions 
                (id, user_id, created_at, last_activity, context, model_name, temperature, max_tokens, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session.id, session.user_id, session.created_at, session.last_activity,
                json.dumps(session.context), session.model_name, session.temperature,
                session.max_tokens, session.is_active
            ))
            conn.commit()
    
    def load_session(self, session_id: str) -> Optional[Session]:
        """Load session from database"""
        with sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
            cursor = conn.execute("""
                SELECT * FROM sessions WHERE id = ? AND is_active = 1
            """, (session_id,))
            row = cursor.fetchone()
            
            if row:
                return Session(
                    id=row[0],
                    user_id=row[1],
                    created_at=row[2],
                    last_activity=row[3],
                    context=json.loads(row[4]) if row[4] else [],
                    model_name=row[5],
                    temperature=row[6],
                    max_tokens=row[7],
                    is_active=bool(row[8])
                )
        return None
    
    def save_message(self, message: Message):
        """Save message to database"""
        with sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
            conn.execute("""
                INSERT INTO messages 
                (id, session_id, role, content, timestamp, message_type, metadata, audio_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                message.id, message.session_id, message.role, message.content,
                message.timestamp, message.message_type,
                json.dumps(message.metadata) if message.metadata else None,
                message.audio_data
            ))
            conn.commit()
    
    def get_session_messages(self, session_id: str, limit: int = 50) -> List[Message]:
        """Get recent messages for a session"""
        with sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
            cursor = conn.execute("""
                SELECT * FROM messages 
                WHERE session_id = ? 
                ORDER BY timestamp DESC LIMIT ?
            """, (session_id, limit))
            
            messages = []
            for row in cursor.fetchall():
                messages.append(Message(
                    id=row[0], session_id=row[1], role=row[2], content=row[3],
                    timestamp=row[4], message_type=row[5],
                    metadata=json.loads(row[6]) if row[6] else None,
                    audio_data=row[7]
                ))
            
            return list(reversed(messages))

    def save_tool_call(
        self,
        session_id: str,
        function_name: str,
        arguments: str,
        result: str,
        message_id: Optional[str] = None,
    ) -> str:
        """Persist a tool/function call record to the database and return its id."""
        tool_call_id = str(uuid.uuid4())
        with sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
            conn.execute(
                """
                INSERT INTO tool_calls 
                (id, session_id, message_id, function_name, arguments, result, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tool_call_id,
                    session_id,
                    message_id,
                    function_name,
                    arguments,
                    result,
                    datetime.now(),
                ),
            )
            conn.commit()
        return tool_call_id

class VoiceProcessor:
    def __init__(self, stt_model: str = "openai/whisper-large-v3-turbo"):
        """Initialize voice processing models"""
        logger.info("Initializing voice processing models...")
        
        # Initialize Hugging Face Whisper pipeline for STT
        try:
            self.stt_pipeline = pipeline("automatic-speech-recognition", model=stt_model, generate_kwargs={"language": "en"})
            logger.info(f"Hugging Face Whisper model '{stt_model}' loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
        
        # Initialize Kokoro TTS
        try:
            self.tts_pipeline = KPipeline(lang_code='a')
            logger.info("Kokoro TTS model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Kokoro TTS model: {e}")
            raise
    
    async def speech_to_text(self, audio_data: bytes) -> str:
        """Convert speech audio to text using Hugging Face Whisper pipeline"""
        try:
            # Process with Whisper in thread to avoid blocking
            loop = asyncio.get_event_loop()
            
            def transcribe():
                audio_io = io.BytesIO(audio_data)
                waveform, sample_rate = torchaudio.load(audio_io)
                audio_array = waveform.squeeze().numpy()
                result = self.stt_pipeline(audio_array)
                return result['text'].strip()
            
            text = await loop.run_in_executor(None, transcribe)
            
            return text
        
        except Exception as e:
            logger.error(f"Speech-to-text error: {e}")
            return ""
    
    async def text_to_speech(self, text: str, voice: Optional[str] = "af_heart") -> bytes:
        """Convert text to speech using Kokoro via KPipeline."""
        try:
            loop = asyncio.get_event_loop()

            def synth():
                # voice parameter optional; KPipeline may accept voice kwargs
                generator = self.tts_pipeline(text, voice=voice) if voice else self.tts_pipeline(text)
                # collect all generated chunks into one audio array
                audio_arrays = [audio for (_, _, audio) in generator]
                import numpy as np
                return np.concatenate(audio_arrays, axis=0)

            audio = await loop.run_in_executor(None, synth)

            # Convert to WAV bytes
            audio_io = io.BytesIO()
            # torchaudio.save(audio_io, torch.tensor(audio).unsqueeze(0), 24000, format="wav")
            pcm16 = (audio * 32767).astype(np.int16)
            torchaudio.save(audio_io, torch.from_numpy(pcm16).unsqueeze(0), 24000, format="wav")
            return audio_io.getvalue()

        except Exception as e:
            logger.error(f"Text-to-speech error (kokoro): {e}")
            return b""

class ModelContextProtocol:
    """Handle Model Context Protocol (MCP) for tool integration"""
    
    def __init__(self):
        self.available_tools = {
            "get_time": self.get_current_time,
            "calculate": self.calculate,
            "search_web": self.search_web,
            "get_weather": self.get_weather
        }
    
    async def handle_tool_call(self, function_name: str, arguments: Dict) -> Dict:
        """Handle tool function calls"""
        try:
            if function_name in self.available_tools:
                result = await self.available_tools[function_name](**arguments)
                return {"success": True, "result": result}
            else:
                return {"success": False, "error": f"Unknown function: {function_name}"}
        except Exception as e:
            logger.error(f"Tool call error: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_current_time(self) -> str:
        """Get current time"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    async def calculate(self, expression: str) -> str:
        """Safe calculator"""
        try:
            # Simple safe evaluation
            result = eval(expression, {"__builtins__": {}}, {})
            return str(result)
        except:
            return "Invalid expression"
    
    async def search_web(self, query: str) -> str:
        """Mock web search"""
        return f"Search results for: {query} (Mock implementation)"
    
    async def get_weather(self, location: str) -> str:
        """Mock weather service"""
        return f"Weather in {location}: Sunny, 22°C (Mock implementation)"

class VoiceAIServer:
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.db = DatabaseManager()
        self.voice_processor = VoiceProcessor()  # You can pass a different model here, e.g., VoiceProcessor("openai/whisper-small")
        # Register tool functions and schemas for OpenAI function calling
        self.tool_functions = TOOL_FUNCTIONS
        self.tool_schemas = TOOL_SCHEMAS
        self.sessions: Dict[str, Session] = {}
        self.connected_clients: Dict[str, websockets.WebSocketServerProtocol] = {}
        # Per-session locks to serialize processing and avoid role alternation errors
        self.session_locks: Dict[str, asyncio.Lock] = {}
        
        # OpenAI client configuration (update with your settings)
        self.openai_client = AsyncOpenAI(
            base_url="http://51.159.132.110/v1",
            api_key="zoyel-medgemma-27b-it-miniontek"
        )

    def _get_session_lock(self, session_id: str) -> asyncio.Lock:
        """Get or create an asyncio.Lock for a session id."""
        lock = self.session_locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            self.session_locks[session_id] = lock
        return lock

    def _normalize_context(self, context: List[Dict]) -> List[Dict]:
        """Normalize context to strictly alternate roles (user <-> assistant),
        drop leading assistant messages, merge consecutive same-role messages,
        without applying a fixed history cap.
        """
        normalized: List[Dict] = []
        for msg in context:
            role = msg.get("role")
            content = msg.get("content")
            if role not in ("user", "assistant"):
                continue
            if not content:
                continue
            if not normalized:
                if role == "assistant":
                    # Drop leading assistant
                    continue
                normalized.append({"role": role, "content": content})
            else:
                last_role = normalized[-1]["role"]
                if last_role == role:
                    # Merge consecutive same-role messages to avoid alternation errors
                    normalized[-1]["content"] = normalized[-1]["content"] + "\n\n" + content
                else:
                    normalized.append({"role": role, "content": content})
        return normalized

    def _estimate_tokens(self, text: str) -> int:
        """Lightweight token estimator: ~4 chars per token (conservative)."""
        if not text:
            return 0
        # Add 1 to avoid zero for very short strings
        return max(1, len(text) // 4)

    def _truncate_messages_to_token_limit(self, messages: List[Dict], max_tokens: int) -> List[Dict]:
        """Keep the tail of messages so that the estimated total tokens <= max_tokens.
        Ensures we don't start with an assistant message after truncation.
        """
        total = 0
        kept_rev: List[Dict] = []
        for msg in reversed(messages):
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role not in ("user", "assistant"):
                continue
            t = self._estimate_tokens(content) + 4  # small per-message overhead
            if total + t > max_tokens:
                # If nothing kept yet and this single message exceeds limit, truncate its tail
                if not kept_rev and max_tokens > 0:
                    char_limit = max_tokens * 4
                    kept_rev.append({"role": role, "content": content[-char_limit:]})
                    total = max_tokens
                break
            kept_rev.append({"role": role, "content": content})
            total += t

        kept = list(reversed(kept_rev))
        if kept and kept[0].get("role") == "assistant":
            kept = kept[1:]
        return kept

    def _build_messages_for_api(self, session: Session) -> List[Dict]:
        """Prepend system prompt and include normalized context, truncated to fit
        the configured context window minus reserved output tokens.
        """
        convo = self._normalize_context(session.context)

        # Reserve output tokens and some overhead so input+output stays within context
        overhead_tokens = self._estimate_tokens(SYSTEM_PROMPT) + 64
        available_for_context = max(
            0,
            session.context_max_tokens - session.max_tokens - overhead_tokens
        )

        convo = self._truncate_messages_to_token_limit(convo, available_for_context)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(convo)
        return messages
    
    async def register_client(self, websocket, session_id: str):
        """Register a new client connection"""
        self.connected_clients[session_id] = websocket
        logger.info(f"Client registered: {session_id}")
    
    async def unregister_client(self, session_id: str):
        """Unregister client connection"""
        if session_id in self.connected_clients:
            del self.connected_clients[session_id]
        logger.info(f"Client unregistered: {session_id}")
    
    async def get_or_create_session(self, user_id: str, session_id: Optional[str] = None) -> Session:
        """Get existing session or create new one"""
        if session_id and session_id in self.sessions:
            session = self.sessions[session_id]
            session.last_activity = datetime.now()
            return session
        
        # Try to load from database
        if session_id:
            session = self.db.load_session(session_id)
            if session:
                self.sessions[session_id] = session
                session.last_activity = datetime.now()
                # Normalize any previously stored context to ensure alternation
                session.context = self._normalize_context(session.context)
                # Enforce configured generation token cap for this model
                if session.model_name == "google/medgemma-27b-it" and session.max_tokens != 256:
                    session.max_tokens = 256
                    self.db.save_session(session)
                return session
        
        # Create new session
        new_session_id = session_id or str(uuid.uuid4())
        session = Session(
            id=new_session_id,
            user_id=user_id,
            created_at=datetime.now(),
            last_activity=datetime.now(),
            context=[],
            model_name="google/medgemma-27b-it"  # Default model
        )
        
        self.sessions[new_session_id] = session
        self.db.save_session(session)
        
        return session
    
    async def process_message(self, session: Session, message: Message) -> Optional[Message]:
        """Process user message and generate AI response using function calling."""
        try:
            # Add user message to context
            session.context.append({
                "role": "user",
                "content": message.content,
            })
            
            # Build base messages for API
            messages = self._build_messages_for_api(session)
            final_response = None
            max_tool_calls = 3
            tool_call_count = 0

            # Create a copy of messages for the API call
            api_messages = messages.copy()

            # Tool-calling loop
            while tool_call_count < max_tool_calls:
                # Make the API call
                response = await self.openai_client.chat.completions.create(
                    model=session.model_name,
                    messages=api_messages,
                    temperature=session.temperature,
                    max_tokens=session.max_tokens,
                    tools=self.tool_schemas,
                    tool_choice="auto",
                )

                choice = response.choices[0]
                msg = choice.message
                
                # If this is a final response (no tool calls), use it as the final response
                if not hasattr(msg, 'tool_calls') or not msg.tool_calls:
                    final_response = msg.content or ""
                    break
            
                # Add assistant's message with tool calls to the conversation
                tool_calls = []
                for tool_call in msg.tool_calls:
                    tool_calls.append({
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        }
                    })
            
                api_messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tool_calls
                })
            
                # Process tool calls
                tool_call_count += 1
                for tool_call in msg.tool_calls:
                    fn_name = tool_call.function.name
                    try:
                        # Parse arguments
                        args = json.loads(tool_call.function.arguments)
                    
                        # Execute the function
                        if fn_name in self.tool_functions:
                            fn = self.tool_functions[fn_name]
                            if asyncio.iscoroutinefunction(fn):
                                result = await fn(**args)
                            else:
                                result = fn(**args)
                            
                            # Add tool response to messages
                            api_messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": fn_name,
                                "content": str(result)
                            })
                    
                    except Exception as e:
                        logger.error(f"Error executing tool {fn_name}: {e}")
                        api_messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": fn_name,
                            "content": f"Error: {str(e)}"
                        })
        
            # If we don't have a final response yet, get one from the model
            if final_response is None:
                response = await self.openai_client.chat.completions.create(
                    model=session.model_name,
                    messages=api_messages,
                    temperature=session.temperature,
                    max_tokens=session.max_tokens,
                )
                final_response = response.choices[0].message.content or "I'm sorry, I couldn't complete the request."

            # Create response message
            response_msg = Message(
                id=str(uuid.uuid4()),
                session_id=session.id,
                role="assistant",
                content=final_response,
                timestamp=datetime.utcnow(),
                message_type="text"
            )
        
            # Add assistant's final response to context
            session.context.append({
                "role": "assistant",
                "content": final_response,
            })
        
            # Save session to update context
            self.db.save_session(session)
        
            return response_msg
        
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            logger.error(traceback.format_exc())
            return None
    
    async def handle_client_message(self, websocket, message_data: Dict):
        """Handle incoming client message"""
        try:
            message_type = message_data.get("type")
            session_id = message_data.get("session_id")
            user_id = message_data.get("user_id", "anonymous")
            
            # Get or create session
            session = await self.get_or_create_session(user_id, session_id)
            
            # Serialize processing per session to avoid concurrent completions
            async with self._get_session_lock(session.id):
                if message_type == "text_message":
                    # Handle text message
                    content = message_data.get("content", "")
                    
                    user_message = Message(
                        id=str(uuid.uuid4()),
                        session_id=session.id,
                        role="user",
                        content=content,
                        timestamp=datetime.now(),
                        message_type="text"
                    )
                    
                    # Save user message
                    self.db.save_message(user_message)
                    
                    # Process and get AI response
                    ai_response = await self.process_message(session, user_message)
                    
                    if ai_response:
                        # Send text response
                        await websocket.send(json.dumps({
                            "type": "text_response",
                            "content": ai_response.content,
                            "session_id": session.id,
                            "message_id": ai_response.id
                        }))
                        
                        # Generate and send audio response
                        audio_data = await self.voice_processor.text_to_speech(ai_response.content)
                        if audio_data:
                            audio_b64 = base64.b64encode(audio_data).decode('utf-8')
                            await websocket.send(json.dumps({
                                "type": "audio_response",
                                "audio_data": audio_b64,
                                "session_id": session.id,
                                "message_id": ai_response.id
                            }))
                
                elif message_type == "audio_message":
                    # Handle audio message
                    audio_b64 = message_data.get("audio_data", "")
                    audio_data = base64.b64decode(audio_b64)
                    
                    # Convert speech to text
                    text_content = await self.voice_processor.speech_to_text(audio_data)
                    
                    if text_content:
                        user_message = Message(
                            id=str(uuid.uuid4()),
                            session_id=session.id,
                            role="user",
                            content=text_content,
                            timestamp=datetime.now(),
                            message_type="audio",
                            audio_data=audio_data
                        )
                        
                        # Save user message
                        self.db.save_message(user_message)
                        
                        # Send transcription back to client
                        await websocket.send(json.dumps({
                            "type": "transcription",
                            "content": text_content,
                            "session_id": session.id,
                            "message_id": user_message.id
                        }))
                        
                        # Process and get AI response
                        ai_response = await self.process_message(session, user_message)
                        
                        if ai_response:
                            # Send text response
                            await websocket.send(json.dumps({
                                "type": "text_response",
                                "content": ai_response.content,
                                "session_id": session.id,
                                "message_id": ai_response.id
                            }))
                            
                            # Generate and send audio response
                            audio_data = await self.voice_processor.text_to_speech(ai_response.content)
                            if audio_data:
                                audio_b64 = base64.b64encode(audio_data).decode('utf-8')
                                await websocket.send(json.dumps({
                                    "type": "audio_response",
                                    "audio_data": audio_b64,
                                    "session_id": session.id,
                                    "message_id": ai_response.id
                                }))
                
                elif message_type == "get_history":
                    # Send conversation history
                    messages = self.db.get_session_messages(session.id)
                    history = []
                    for msg in messages:
                        history.append({
                            "role": msg.role,
                            "content": msg.content,
                            "timestamp": msg.timestamp.isoformat(),
                            "message_type": msg.message_type
                        })
                    
                    await websocket.send(json.dumps({
                        "type": "history",
                        "messages": history,
                        "session_id": session.id
                    }))
            
        except Exception as e:
            logger.error(f"Error handling client message: {e}")
            traceback.print_exc()
            await websocket.send(json.dumps({
                "type": "error",
                "message": "Internal server error"
            }))
    
    async def handle_client(self, websocket):
        """Handle individual client connection"""
        session_id = None
        try:
            # Send welcome message
            await websocket.send(json.dumps({
                "type": "connected",
                "message": "Connected to Voice AI Server"
            }))
            
            async for message in websocket:
                try:
                    data = json.loads(message)
                    
                    # Register client on first message
                    if session_id is None:
                        session_id = data.get("session_id", str(uuid.uuid4()))
                        await self.register_client(websocket, session_id)
                    
                    await self.handle_client_message(websocket, data)
                
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": "Invalid JSON"
                    }))
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": "Error processing message"
                    }))
        
        except websockets.exceptions.ConnectionClosed as e:
            logger.info(f"Client {session_id} disconnected (code={getattr(e, 'code', 'unknown')}, reason={getattr(e, 'reason', '')})")
        except Exception as e:
            logger.error(f"Connection error: {e}")
        finally:
            if session_id:
                await self.unregister_client(session_id)
    
    async def cleanup_inactive_sessions(self):
        """Periodic cleanup of inactive sessions"""
        while True:
            try:
                cutoff_time = datetime.now() - timedelta(hours=1)
                inactive_sessions = [
                    sid for sid, session in self.sessions.items()
                    if session.last_activity < cutoff_time
                ]
                
                for session_id in inactive_sessions:
                    session = self.sessions[session_id]
                    session.is_active = False
                    self.db.save_session(session)
                    del self.sessions[session_id]
                    logger.info(f"Cleaned up inactive session: {session_id}")
                
                await asyncio.sleep(300)  # Run every 5 minutes
            
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
                await asyncio.sleep(60)
    
    async def start_server(self):
        """Start the WebSocket server"""
        logger.info(f"Starting Voice AI Server on ws://{self.host}:{self.port}")
        
        # Start cleanup task
        cleanup_task = asyncio.create_task(self.cleanup_inactive_sessions())
        
        # Start WebSocket server
        server = await websockets.serve(
            self.handle_client,
            self.host,
            self.port,
            ping_interval=30,
            ping_timeout=20,
            max_size=16 * 1024 * 1024
        )
        
        logger.info("Voice AI Server started successfully!")
        
        try:
            await server.wait_closed()
        except KeyboardInterrupt:
            logger.info("Shutting down server...")
        finally:
            cleanup_task.cancel()
            server.close()
            await server.wait_closed()

def main():
    """Main entry point"""
    # Configuration
    HOST = "0.0.0.0"  # Listen on all interfaces
    PORT = 8765
    
    # Create and start server
    server = VoiceAIServer(HOST, PORT)
    
    try:
        asyncio.run(server.start_server())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")

if __name__ == "__main__":
    main()