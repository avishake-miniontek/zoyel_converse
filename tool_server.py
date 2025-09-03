#!/usr/bin/env python3
"""
Voice AI Server - Complete WebSocket server with manual tool call processing
Features:
- Multi-client session management
- Database storage for messages
- Voice processing (Whisper + Kokoro TTS)
- Manual tool call parsing and execution
- Reconnection handling
- Error recovery
"""

import asyncio
import websockets
import json
import re
from sqlalchemy import create_engine, Column, String, DateTime, Boolean, Text, LargeBinary, ForeignKey, Float, Integer
from sqlalchemy.orm import DeclarativeBase, sessionmaker
import uuid
import logging
import traceback
import io
import base64
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
from tools import TOOL_FUNCTIONS

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
You are Zoya, a helpful medical AI assistant who communicates through voice. Generally it's a patient that you will be talking to.
You are part of a secure medical software system with strict privacy and non-disclosure policies. Collecting and storing patient data is necessary and safe for providing accurate medical assistance, differential diagnosis, and recalling information later.

You have access to tools that can help answer questions. When you need to use a tool, wrap the function call in <tool_call></tool_call> tags like this:

<tool_call>get_time()</tool_call>
<tool_call>get_weather('New York')</tool_call>
<tool_call>calculate('2 + 2')</tool_call>
<tool_call>save_patient_data(age={'years': 25, 'months': 0, 'days': 0}, complaints=[{'symptom': 'headache', 'severity': 'moderate', 'since': '2 days'}], vitals={'weight_kg': 60, 'height_cm': 162, 'bp_systolic': 110, 'bp_diastolic': 80, 'temperature_c': 37.6, 'heart_rate': 88, 'respiration_rate': 18, 'spo2': 99, 'lmp': '15-07-2025'})</tool_call>
<tool_call>save_patient_data(age_years=25, complaints=[{'symptom': 'headache', 'severity': 'moderate', 'since': '2 days'}])</tool_call>
<tool_call>fetch_patient_data()</tool_call>
<tool_call>fetch_patient_data(fields=['allergies', 'vitals'])</tool_call>

Available tools:
- get_time(): Get current time
- get_weather(location): Get weather for a location
- calculate(expression): Perform calculations
- save_patient_data(**kwargs): Save/update patient medical data (supports nested JSON structure) - session_id is automatically provided
- fetch_patient_data(fields=None): Retrieve patient medical data (optionally specific fields only) - session_id is automatically provided
- search_web(query): Search the internet

At the beginning of every conversation, first use the fetch_patient_data tool to check for and load any existing patient data for this session. If no data exists or it's incomplete, politely greet the patient and start asking for their personal and medical information step by step in a natural, conversational way. 

Gather details including:
- Age (in years, months, days - can be passed as nested object or individual values)
- Gender, city, country
- Complaints (each with symptom, severity, and duration)
- Vitals (weight in kg, height in cm, blood pressure systolic/diastolic, temperature in Celsius with decimals, heart rate, respiration rate, SpO2, LMP if applicable - can be passed as nested object or individual values)
- Physical examination findings
- Comorbidities
- Past history including illnesses and procedures with dates when available
- Current medications
- Family history
- Allergies (both drug and food allergies - can be passed as nested object or individual lists)
- Test documents (URLs or file paths)
- Test results (structured with test name and parameter details including values, units, and reference ranges)

Do not ask for the current date; use the get_time tool to obtain it when needed for the date field in save_patient_data. The date will be automatically formatted as DD-MM-YYYY. Ask follow-up questions to fill in any missing details or clarify ambiguities. Keep lists empty if no information is available. 

Once you have collected new or updated details, ALWAYS use the save_patient_data tool to store or update it immediately. Patient data persistence is CRITICAL for medical safety and continuity of care. If a save operation fails, retry it immediately with the same data. You can use either the nested JSON structure (recommended) or flat parameters:

CRITICAL DATA STRUCTURE REQUIREMENTS:
When using save_patient_data, you MUST follow these exact formats:

For vitals, use individual fields: bp_systolic, bp_diastolic (NOT blood_pressure)
For family_history, use a list of strings: ['Hypertension (Mother)', 'Breast Cancer (Maternal Aunt)']
For current_medications, use simple strings: ['Levothyroxine 75mcg once daily']
For past_history, use the nested format with separate arrays: past_history={'past_illnesses': [{'illness': 'UTI', 'date': '05-09-2023'}], 'previous_procedures': [{'procedure': 'Cystoscopy', 'date': '10-10-2023'}]}
For allergies, use the nested format: allergies={'drug_allergies': ['Ibuprofen'], 'food_allergies': ['Shellfish']}
For physical_examination, use a simple TEXT STRING, not a dictionary
For test_results, use the correct structure: test_results=[{'test_name': 'Test Name', 'test_result_values': [{'parameterName': 'Parameter', 'parameterValue': 'Value', 'parameterUnit': 'Unit', 'parameterRefRange': 'Range'}]}]

CORRECT example:
save_patient_data(vitals={'bp_systolic': 110, 'bp_diastolic': 80}, family_history=['Hypertension (Mother)'], current_medications=['Levothyroxine 75mcg once daily'], allergies={'drug_allergies': ['Ibuprofen'], 'food_allergies': ['Shellfish']}, physical_examination='Mild tenderness in suprapubic region, no guarding')

WRONG examples to avoid:
- physical_examination={'abdomen': 'findings'} (use physical_examination='Abdomen: findings')
- allergies={'drug': ['X'], 'food': ['Y']} (use allergies={'drug_allergies': ['X'], 'food_allergies': ['Y']})
- test_results with 'parameters' key (use 'test_result_values' key)



IMPORTANT: After collecting ANY new patient information (even a single piece of data like age or one symptom), immediately save it using save_patient_data. Do not wait to collect all information before saving. Save incrementally as you gather data to ensure no information is lost. If you receive an error when saving, retry the save operation immediately. Patient data must never be lost due to technical issues.

If the user asks to recall or confirm any of their previously provided information, use the fetch_patient_data tool to retrieve only the relevant fields and share them naturally in your response. The fetch function returns JSON with success status and structured data matching the target format.

Important instructions for your responses:

1) Provide only plain text that will be converted to speech - never use markdown, asterisk *, code blocks, or special formatting in your final response.
2) Use natural, conversational language as if you're speaking to someone.
3) Never use bullet points, numbered lists, or special characters in your final response.
4) Keep responses concise and clear since they will be spoken aloud.
5) Express lists or multiple points in a natural spoken way using words like 'first', 'also', 'finally', etc.
6) Use punctuation only for natural speech pauses (periods, commas, question marks).
7) The tool calls will be executed automatically and removed from your response before it's spoken.
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
    max_tokens: int = 4096
    context_max_tokens: int = 128000
    is_active: bool = True

class Base(DeclarativeBase):
    pass

class SessionModel(Base):
    __tablename__ = 'sessions'
    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False)
    created_at = Column(DateTime)
    last_activity = Column(DateTime)
    context = Column(Text)
    model_name = Column(Text)
    temperature = Column(Float)
    max_tokens = Column(Integer)
    is_active = Column(Boolean)

class MessageModel(Base):
    __tablename__ = 'messages'
    id = Column(String, primary_key=True)
    session_id = Column(String, ForeignKey('sessions.id'))
    role = Column(String)
    content = Column(Text)
    timestamp = Column(DateTime)
    message_type = Column(String)

class ToolCallModel(Base):
    __tablename__ = 'tool_calls'
    id = Column(String, primary_key=True)
    session_id = Column(String, ForeignKey('sessions.id'))
    message_id = Column(String, ForeignKey('messages.id'))
    function_name = Column(String)
    arguments = Column(Text)
    result = Column(Text)
    timestamp = Column(DateTime)

class DatabaseManager:
    def __init__(self, db_path: str = "voice_ai.db"):
        self.db_path = db_path
        self.engine = create_engine(f"sqlite:///{self.db_path}")
        self.Session = sessionmaker(bind=self.engine)
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database with required tables"""
        Base.metadata.create_all(bind=self.engine)
    
    def save_session(self, session: Session):
        """Save or update session in database"""
        with self.Session() as sess:
            context_str = json.dumps(session.context)
            model = sess.query(SessionModel).filter_by(id=session.id).first()
            if model:
                model.user_id = session.user_id
                model.created_at = session.created_at
                model.last_activity = session.last_activity
                model.context = context_str
                model.model_name = session.model_name
                model.temperature = session.temperature
                model.max_tokens = session.max_tokens
                model.is_active = session.is_active
            else:
                model = SessionModel(
                    id=session.id,
                    user_id=session.user_id,
                    created_at=session.created_at,
                    last_activity=session.last_activity,
                    context=context_str,
                    model_name=session.model_name,
                    temperature=session.temperature,
                    max_tokens=session.max_tokens,
                    is_active=session.is_active
                )
                sess.add(model)
            sess.commit()
    
    def load_session(self, session_id: str) -> Optional[Session]:
        """Load session from database"""
        with self.Session() as sess:
            row = sess.query(SessionModel).filter_by(id=session_id, is_active=True).first()
            if row:
                return Session(
                    id=row.id,
                    user_id=row.user_id,
                    created_at=row.created_at,
                    last_activity=row.last_activity,
                    context=json.loads(row.context) if row.context else [],
                    model_name=row.model_name,
                    temperature=row.temperature,
                    max_tokens=row.max_tokens,
                    is_active=row.is_active
                )
        return None
    
    def save_message(self, message: Message):
        """Save message to database"""
        with self.Session() as sess:
            metadata_str = json.dumps(message.metadata) if message.metadata else None
            model = MessageModel(
                id=message.id,
                session_id=message.session_id,
                role=message.role,
                content=message.content,
                timestamp=message.timestamp,
                message_type=message.message_type,
            )
            sess.add(model)
            sess.commit()
    
    def get_session_messages(self, session_id: str, limit: int = 50) -> List[Message]:
        """Get recent messages for a session"""
        with self.Session() as sess:
            rows = sess.query(MessageModel).filter_by(session_id=session_id)\
                .order_by(MessageModel.timestamp.desc()).limit(limit).all()
            messages = []
            for row in rows:
                messages.append(Message(
                    id=row.id,
                    session_id=row.session_id,
                    role=row.role,
                    content=row.content,
                    timestamp=row.timestamp,
                    message_type=row.message_type,
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
        with self.Session() as sess:
            model = ToolCallModel(
                id=tool_call_id,
                session_id=session_id,
                message_id=message_id,
                function_name=function_name,
                arguments=arguments,
                result=result,
                timestamp=datetime.now()
            )
            sess.add(model)
            sess.commit()
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

class VoiceAIServer:
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.db = DatabaseManager()
        self.voice_processor = VoiceProcessor()  # You can pass a different model here, e.g., VoiceProcessor("openai/whisper-small")
        # Register tool functions for manual execution
        self.tool_functions = TOOL_FUNCTIONS
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
    
    def _parse_tool_calls(self, text: str) -> List[Dict]:
        """Parse tool calls from text using regex to find <tool_call>function_name(args)</tool_call> patterns."""
        tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
        matches = re.findall(tool_call_pattern, text, re.DOTALL)
        
        tool_calls = []
        for match in matches:
            match = match.strip()
            # Parse function name and arguments
            func_pattern = r'(\w+)\((.*)\)'
            func_match = re.match(func_pattern, match)
            if func_match:
                func_name = func_match.group(1)
                args_str = func_match.group(2).strip()
                
                tool_calls.append({
                    'function_name': func_name,
                    'arguments_str': args_str,
                    'raw_call': match
                })
        
        return tool_calls
    
    def _remove_tool_calls(self, text: str) -> str:
        """Remove all <tool_call>...</tool_call> blocks from text."""
        tool_call_pattern = r'<tool_call>.*?</tool_call>'
        cleaned_text = re.sub(tool_call_pattern, '', text, flags=re.DOTALL)
        # Clean up extra whitespace
        cleaned_text = re.sub(r'\n\s*\n', '\n', cleaned_text)
        return cleaned_text.strip()
    
    def _parse_function_args(self, args_str: str, session_id: str) -> Dict:
        """Parse function arguments from string, handling various formats including nested objects."""
        if not args_str:
            return {}
        
        try:
            import ast
            
            # Handle session_id injection for patient data functions
            if 'session_id' not in args_str and session_id:
                if args_str:
                    args_str = f"session_id='{session_id}', {args_str}"
                else:
                    args_str = f"session_id='{session_id}'"
            
            # For complex nested structures, try to evaluate the entire args_str as a function call
            try:
                # Create a safe evaluation environment
                def safe_eval_dict(node):
                    """Safely evaluate dictionary/list literals"""
                    if isinstance(node, ast.Dict):
                        return {safe_eval_dict(k): safe_eval_dict(v) for k, v in zip(node.keys, node.values)}
                    elif isinstance(node, ast.List):
                        return [safe_eval_dict(item) for item in node.elts]
                    elif isinstance(node, ast.Constant):
                        return node.value
                    elif isinstance(node, ast.Str):  # For older Python versions
                        return node.s
                    elif isinstance(node, ast.Num):  # For older Python versions
                        return node.n
                    elif isinstance(node, ast.NameConstant):  # For older Python versions
                        return node.value
                    else:
                        raise ValueError(f"Unsupported node type: {type(node)}")

                # Parse the arguments string manually with better handling of nested structures
                result = {}
                
                # Split by commas, but respect nested structures
                args_parts = []
                current_part = ""
                paren_count = 0
                bracket_count = 0
                brace_count = 0
                quote_char = None
                
                i = 0
                while i < len(args_str):
                    char = args_str[i]
                    
                    if quote_char:
                        current_part += char
                        if char == quote_char and (i == 0 or args_str[i-1] != '\\'):
                            quote_char = None
                    elif char in ['"', "'"]:
                        quote_char = char
                        current_part += char
                    elif char == '(':
                        paren_count += 1
                        current_part += char
                    elif char == ')':
                        paren_count -= 1
                        current_part += char
                    elif char == '[':
                        bracket_count += 1
                        current_part += char
                    elif char == ']':
                        bracket_count -= 1
                        current_part += char
                    elif char == '{':
                        brace_count += 1
                        current_part += char
                    elif char == '}':
                        brace_count -= 1
                        current_part += char
                    elif char == ',' and paren_count == 0 and bracket_count == 0 and brace_count == 0:
                        args_parts.append(current_part.strip())
                        current_part = ""
                    else:
                        current_part += char
                    
                    i += 1
                
                if current_part.strip():
                    args_parts.append(current_part.strip())
                
                # Parse each argument
                for arg_part in args_parts:
                    if '=' in arg_part:
                        key, value_str = arg_part.split('=', 1)
                        key = key.strip()
                        value_str = value_str.strip()
                        
                        try:
                            # Parse the value using AST
                            parsed = ast.parse(value_str, mode='eval')
                            value = safe_eval_dict(parsed.body)
                            result[key] = value
                        except Exception as parse_error:
                            logger.warning(f"Failed to parse value '{value_str}' for key '{key}': {parse_error}")
                            # Fallback: treat as string, but handle common cases
                            cleaned_value = value_str.strip('"\'')
                            
                            # Try to handle common list formats that might have failed
                            if cleaned_value.startswith('[') and cleaned_value.endswith(']'):
                                try:
                                    # Simple list parsing for strings
                                    list_content = cleaned_value[1:-1]
                                    if list_content:
                                        # Split by comma and clean up quotes
                                        items = [item.strip().strip('"\'') for item in list_content.split(',')]
                                        result[key] = items
                                    else:
                                        result[key] = []
                                except:
                                    result[key] = cleaned_value
                            else:
                                result[key] = cleaned_value
                
                return result
                
            except Exception as e:
                logger.error(f"Complex parsing failed: {e}")
                # Fallback to simpler parsing for basic cases
                
                if '=' in args_str:
                    # Simple keyword arguments with improved nested structure handling
                    pairs = []
                    current_pair = ""
                    paren_count = 0
                    bracket_count = 0
                    brace_count = 0
                    quote_char = None
                    
                    for char in args_str:
                        if quote_char:
                            current_pair += char
                            if char == quote_char:
                                quote_char = None
                        elif char in ['"', "'"]:
                            quote_char = char
                            current_pair += char
                        elif char == '(':
                            paren_count += 1
                            current_pair += char
                        elif char == ')':
                            paren_count -= 1
                            current_pair += char
                        elif char == '[':
                            bracket_count += 1
                            current_pair += char
                        elif char == ']':
                            bracket_count -= 1
                            current_pair += char
                        elif char == '{':
                            brace_count += 1
                            current_pair += char
                        elif char == '}':
                            brace_count -= 1
                            current_pair += char
                        elif char == ',' and paren_count == 0 and bracket_count == 0 and brace_count == 0:
                            pairs.append(current_pair.strip())
                            current_pair = ""
                        else:
                            current_pair += char
                    
                    if current_pair.strip():
                        pairs.append(current_pair.strip())
                    
                    result = {}
                    for pair in pairs:
                        if '=' in pair:
                            key, value = pair.split('=', 1)
                            key = key.strip()
                            value = value.strip()
                            
                            # Try to evaluate the value
                            try:
                                result[key] = ast.literal_eval(value)
                            except:
                                # If literal_eval fails, treat as string
                                result[key] = value.strip('"\'')
                        
                    return result
                else:
                    # Positional argument
                    try:
                        value = ast.literal_eval(args_str)
                        return {'arg': value}
                    except:
                        return {'arg': args_str.strip('"\''), 'location': args_str.strip('"\''), 'expression': args_str.strip('"\''), 'query': args_str.strip('"\'')}
            
        except Exception as e:
            logger.error(f"Error parsing function args '{args_str}': {e}")
            return {'arg': args_str}
    
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
    
    async def execute_tool_call(self, function_name: str, args: Dict, session_id: str, message_id: str = None) -> str:
        """Execute a single tool call and return the result."""
        try:
            if function_name not in self.tool_functions:
                return f"Error: Unknown function '{function_name}'"
            
            func = self.tool_functions[function_name]
            
            # Add session_id for patient data functions if not already present
            if function_name in ["save_patient_data", "fetch_patient_data"] and "session_id" not in args:
                args["session_id"] = session_id
            
            # Handle different argument patterns for different functions
            if function_name == "get_time":
                result = await func() if asyncio.iscoroutinefunction(func) else func()
            elif function_name in ["get_weather", "search_web", "calculate"]:
                # These typically take a single argument
                if 'location' in args:
                    result = await func(args['location']) if asyncio.iscoroutinefunction(func) else func(args['location'])
                elif 'query' in args:
                    result = await func(args['query']) if asyncio.iscoroutinefunction(func) else func(args['query'])
                elif 'expression' in args:
                    result = await func(args['expression']) if asyncio.iscoroutinefunction(func) else func(args['expression'])
                elif 'arg' in args:
                    result = await func(args['arg']) if asyncio.iscoroutinefunction(func) else func(args['arg'])
                else:
                    result = await func(**args) if asyncio.iscoroutinefunction(func) else func(**args)
            else:
                # Patient data functions and others that take keyword arguments
                result = await func(**args) if asyncio.iscoroutinefunction(func) else func(**args)
            
            # Log the tool call
            self.db.save_tool_call(
                session_id=session_id,
                message_id=message_id,
                function_name=function_name,
                arguments=json.dumps(args),
                result=str(result)
            )
            
            return str(result)
            
        except Exception as e:
            error_msg = f"Error executing {function_name}: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            self.db.save_tool_call(
                session_id=session_id,
                message_id=message_id,
                function_name=function_name,
                arguments=json.dumps(args),
                result=error_msg
            )
            
            return error_msg
    
    async def process_message(self, session: Session, message: Message) -> Optional[Message]:
        """Process user message and generate AI response with manual tool call handling."""
        try:
            # Add user message to context
            session.context.append({
                "role": "user",
                "content": message.content,
            })
            
            # Build messages for API
            messages = self._build_messages_for_api(session)
            
            max_iterations = 5  # Prevent infinite loops
            iteration_count = 0
            
            while iteration_count < max_iterations:
                iteration_count += 1
                
                # Make the API call (without function calling - just regular completion)
                response = await self.openai_client.chat.completions.create(
                    model=session.model_name,
                    messages=messages,
                    temperature=session.temperature,
                    max_tokens=session.max_tokens,
                )
                
                ai_response = response.choices[0].message.content or ""
                
                # Parse tool calls from the response
                tool_calls = self._parse_tool_calls(ai_response)
                
                if not tool_calls:
                    # No tool calls, this is the final response
                    # Remove any remaining tool call tags and use as final response
                    final_response = self._remove_tool_calls(ai_response)
                    break
                
                # Execute tool calls
                tool_results = []
                for tool_call in tool_calls:
                    function_name = tool_call['function_name']
                    args_str = tool_call['arguments_str']
                    raw_call = tool_call['raw_call']
                    
                    logger.info(f"Executing tool call: {function_name}({args_str})")
                    
                    # Parse arguments
                    args = self._parse_function_args(args_str, session.id)
                    
                    # Execute the tool call
                    result = await self.execute_tool_call(function_name, args, session.id, message.id)
                    tool_results.append(f"Tool call {raw_call} returned: {result}")
                
                # Add the assistant's response with tool calls to messages
                messages.append({
                    "role": "assistant",
                    "content": ai_response
                })
                
                # Add tool results as a user message (simulating the results being provided back)
                tool_results_text = "\n".join(tool_results)
                messages.append({
                    "role": "user",
                    "content": f"Tool execution results:\n{tool_results_text}\n\nPlease provide your response based on these results."
                })
            
            # If we exhausted iterations without a clean response, use the last response
            if iteration_count >= max_iterations:
                final_response = self._remove_tool_calls(ai_response)
                logger.warning(f"Reached maximum iterations ({max_iterations}) for tool calls")
            
            # Create response message
            response_msg = Message(
                id=str(uuid.uuid4()),
                session_id=session.id,
                role="assistant",
                content=final_response,
                timestamp=datetime.now(),
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
                        
                        # Generate and send audio response (only if content is not empty)
                        if ai_response.content.strip():
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
                            
                            # Generate and send audio response (only if content is not empty)
                            if ai_response.content.strip():
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