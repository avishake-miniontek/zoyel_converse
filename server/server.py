#!/usr/bin/env python3

from dotenv import load_dotenv
load_dotenv()

import asyncio
import websockets
import numpy as np
import time
import sys
import json
import os
import struct
from collections import deque
from urllib.parse import urlparse, parse_qs
import datetime
import re

################################################################################
# GPU SELECTION
################################################################################

def select_gpu():
    """Select best GPU and set CUDA_VISIBLE_DEVICES."""
    try:
        import subprocess
        
        result = subprocess.run(
            ['nvidia-smi',
             '--query-gpu=index,memory.total,memory.used,memory.free',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )
        
        best_gpu = None
        max_free_memory = 0
        min_required_gb = 4
        
        for line in result.stdout.strip().split('\n'):
            gpu_id, total, used, free = map(int, line.strip().split(', '))
            free_memory_gb = free / 1024.0
            print(f"GPU {gpu_id}: {free_memory_gb:.2f}GB free VRAM")
            
            if free >= (min_required_gb * 1024) and free > max_free_memory:
                max_free_memory = free
                best_gpu = gpu_id
        
        if best_gpu is not None:
            print(f"Setting CUDA_VISIBLE_DEVICES={best_gpu}")
            os.environ['CUDA_VISIBLE_DEVICES'] = str(best_gpu)
            return True
            
    except Exception as e:
        print(f"Error selecting GPU: {e}")
    return False

has_gpu = select_gpu()

import torch
import gc
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from kokoro import KPipeline

# Import our modules
from src.server_audio_core import ServerAudioCore, map_language_code
from src import audio_utils

# Pipeline manager for handling multiple language pipelines
class KokoroPipelineManager:
    def __init__(self, max_pipelines=3, device="cuda:0"):
        self.pipelines = {}  # Map of language codes to pipelines
        self.last_used = {}  # Map of language codes to last usage timestamp
        self.max_pipelines = max_pipelines
        self.device = device
        self.default_lang_code = "en"  # Default language (ISO code)
        print(f"Initialized KokoroPipelineManager with max_pipelines={max_pipelines}")
        
    def get_pipeline(self, iso_lang_code):
        """Get or create a pipeline for the specified language."""
        # Update last used timestamp for this language
        current_time = time.time()
        self.last_used[iso_lang_code] = current_time
        
        if iso_lang_code in self.pipelines:
            print(f"Using existing pipeline for language {iso_lang_code}")
        else:
            # Check if we need to evict a pipeline
            self._cleanup_if_needed()
            
            # Convert ISO code to Kokoro code
            kokoro_lang_code = map_language_code(iso_lang_code)
            print(f"Creating new Kokoro pipeline for language: {iso_lang_code} (Kokoro code: {kokoro_lang_code})")
            print(f"Pipeline cache status: {len(self.pipelines)}/{self.max_pipelines} pipelines in use")
            
            # Initialize new pipeline
            pipeline = KPipeline(lang_code=kokoro_lang_code)
            pipeline.model = pipeline.model.to(device=self.device, dtype=torch.float32)
            
            # Store in cache
            self.pipelines[iso_lang_code] = pipeline
            
        return self.pipelines[iso_lang_code]
    
    def _cleanup_if_needed(self):
        """Remove least recently used pipeline if we exceed the maximum."""
        if len(self.pipelines) < self.max_pipelines:
            return
            
        # Find the least recently used language
        lru_lang = min(self.last_used.items(), key=lambda x: x[1])[0]
        
        # Remove it from our collections
        if lru_lang in self.pipelines:
            print(f"Evicting pipeline for language {lru_lang} (least recently used)")
            
            # Explicitly move model to CPU before deletion to free GPU memory
            try:
                self.pipelines[lru_lang].model = self.pipelines[lru_lang].model.to("cpu")
                # Force garbage collection
                del self.pipelines[lru_lang]
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"Successfully freed resources for {lru_lang} pipeline")
            except Exception as e:
                print(f"Error while freeing pipeline resources: {e}")
                # Still remove from dictionary even if cleanup fails
                del self.pipelines[lru_lang]

# Load configuration
with open('config.json', 'r') as f:
    CONFIG = json.load(f)
    CONFIG['server']['websocket']['host'] = os.getenv("WEBSOCKET_HOST", CONFIG['server']['websocket']['host'])
    CONFIG['server']['websocket']['port'] = os.getenv("WEBSOCKET_PORT", CONFIG['server']['websocket']['port'])
    CONFIG['server']['websocket']['api_key'] = os.getenv("WEBSOCKET_API_SECRET_KEY", CONFIG['server']['websocket']['api_key'])
    CONFIG['server']['models']['whisper']['path'] = os.getenv("WHISPER_PATH", CONFIG['server']['models']['whisper']['path'])
    CONFIG['server']['models']['kokoro']['path'] = os.getenv("KOKORO_PATH", CONFIG['server']['models']['kokoro']['path'])
    CONFIG['server']['models']['kokoro']['voice_name'] = os.getenv("KOKORO_VOICE_NAME", CONFIG['server']['models']['kokoro']['voice_name'])
    CONFIG['server']['models']['kokoro']['language_code'] = os.getenv("KOKORO_LANGUAGE_CODE", CONFIG['server']['models']['kokoro']['language_code'])

# Check if model paths exist
whisper_path = CONFIG['server']['models']['whisper']['path']
kokoro_path = CONFIG['server']['models']['kokoro']['path']

if not os.path.exists(whisper_path):
    print(f"Error: Whisper model path does not exist: {whisper_path}")
    sys.exit(1)
if not os.path.exists(kokoro_path):
    print(f"Error: Kokoro model path does not exist: {kokoro_path}")
    sys.exit(1)

# Define Mira alternatives for trigger word detection (excluding 'mira' since it's handled by exact match)
MIRA_ALTERNATIVES = frozenset(['nira', 'neera', 'miro', 'munira', 'miura', 'mihiro'])

def find_trigger_word(msg: str, trigger: str) -> tuple[bool, str, str]:
    """
    Find the trigger word and return both the trigger portion and everything after it.
    
    This function handles both exact matches and alternative spellings of the trigger word.
    For example, if the trigger word is "mira", it will also detect variants like "nira", 
    "neera", etc. - improving the assistant's responsiveness to mispronunciations.
    
    Args:
        msg (str): The transcribed message to search in
        trigger (str): The trigger word to search for
        
    Returns:
        tuple[bool, str, str]: A tuple containing:
            - found (bool): True if trigger or alternative was found
            - trigger_portion (str): The actual trigger word as it appeared in the text
            - text_after_trigger (str): The text after the trigger word
    """
    msg_lower = msg.lower()
    trigger_lower = trigger.lower()
    pos = msg_lower.find(trigger_lower)
    trigger_len = None
    
    # If exact trigger not found and trigger is "mira", try alternatives
    if pos == -1 and trigger_lower == "mira":
        for alt in MIRA_ALTERNATIVES:
            alt_lower = alt.lower()
            alt_pos = msg_lower.find(alt_lower)
            if alt_pos != -1:
                # Found alternative, use its position and length
                pos = alt_pos
                trigger_len = len(alt)
                break
    
    if pos != -1:
        # Match found (either exact trigger or alternative)
        if trigger_len is None:  # No alternative was used
            trigger_len = len(trigger)
        # Extract both parts: the trigger portion and everything after
        trigger_portion = msg[pos:pos + trigger_len]
        text_after = msg[pos + trigger_len:].strip()
        return True, trigger_portion, text_after
    
    return False, "", ""

# Default trigger word from config
DEFAULT_TRIGGER_WORD = CONFIG['assistant']['name'].lower()

device = "cuda:0" if has_gpu else "cpu"
torch_dtype = torch.float16 if has_gpu else torch.float32

MODEL_PATH = CONFIG['server']['models']['whisper']['path']

print(f"Device set to {device}")
print(f"Loading ASR model & processor from: {MODEL_PATH}")

try:
    print("Loading Whisper model...")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True
    )
    print(f"Model loaded successfully. Moving to device {device}")
    model = model.to(device)
    print("Model moved to device successfully")

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    print("Processor loaded successfully")

    print("Creating ASR pipeline...")
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        batch_size=1,  # Process one utterance at a time
        return_timestamps=False,  # Don't need timestamps for basic transcription
        chunk_length_s=30,  # Process in 30-second chunks
    )
    print("ASR pipeline created successfully")
    
    # Verify the pipeline is working with a simple test
    print("Testing ASR pipeline with silence...")
    test_audio = np.zeros((16000,), dtype=np.float32)  # 1 second of silence
    test_result = asr_pipeline(
        {"array": test_audio, "sampling_rate": 16000},
        generate_kwargs={
            "task": "transcribe",
            "language": "en",
            "use_cache": True,
            "num_beams": 1,
            "do_sample": False
        }
    )
    print(f"ASR pipeline test result: {test_result}")
    
except Exception as e:
    print(f"Error loading ASR components: {str(e)}")
    import traceback
    print(f"Full traceback:\n{traceback.format_exc()}")
    sys.exit(1)

print("Loading Kokoro TTS model...")
voice_name = CONFIG['server']['models']['kokoro']['voice_name']

# Get the 2-char language code from configuration and map it to Kokoro's 1-char code
iso_lang_code = CONFIG['server']['models']['kokoro'].get('language_code', 'en')
kokoro_lang_code = map_language_code(iso_lang_code)

print(f"Using voice name: {voice_name}, language code: {iso_lang_code} (Kokoro code: {kokoro_lang_code})")

# Initialize the pipeline manager with max_pipelines from config
max_pipelines = CONFIG['server']['models']['kokoro'].get('max_pipelines', 3)
pipeline_manager = KokoroPipelineManager(max_pipelines=max_pipelines, device=device)

# Initialize default pipeline
tts_pipeline = pipeline_manager.get_pipeline(iso_lang_code)

# Batch processing configuration
ASR_BATCH_TIMEOUT = 0.3  # Process ASR batch after 300ms of no new data
ASR_MAX_BATCH_SIZE = 4   # Maximum items in ASR batch before forcing processing
ASR_SUB_BATCH_SIZE = 2   # Process ASR in smaller sub-batches for responsiveness

TTS_BATCH_TIMEOUT = 0.2  # Process TTS batch after 200ms of no new data
TTS_MAX_BATCH_SIZE = 4   # Maximum items in TTS batch before forcing processing
TTS_BUFFER_TIMEOUT = 0.2  # Flush TTS buffer after 200ms of no new data

################################################################################
# AUDIO SERVER
################################################################################

# Frame format:
# Magic bytes (4 bytes): 0x4D495241 ("MIRA")
# Frame type (1 byte): 
#   0x01 = Audio data
#   0x02 = End of utterance
#   0x03 = VAD status
# Frame length (4 bytes): Length of payload in bytes
# Payload: Audio data (for type 0x01) or empty (for type 0x02)

MAGIC_BYTES = b'MIRA'
FRAME_TYPE_AUDIO = 0x01
FRAME_TYPE_END = 0x02
FRAME_TYPE_VAD = 0x03

def create_frame(frame_type, payload=b''):
    """Create a framed message."""
    frame = bytearray()
    frame.extend(MAGIC_BYTES)
    frame.append(frame_type)
    frame.extend(struct.pack('>I', len(payload)))  # 4 bytes for length
    frame.extend(payload)
    return bytes(frame)

class AudioServer:
    def __init__(self):
        with open('config.json', 'r') as f:
            self.config = json.load(f)
            
        # Initialize the ServerAudioCore; all VAD processing is done within it.
        self.audio_core = ServerAudioCore()
        # (No noise floor calibration is applied here.)
        
        self.enable_voice_filtering = False

        self.transcript_history = deque(maxlen=10)
        self.min_confidence = 0.4
        self.short_phrase_confidence = 0.8
        self.last_transcript = ""
        self.last_transcript_time = 0
        self.min_repeat_interval = 2.0

        # Batch processing buffers
        self.asr_batch = []
        self.asr_batch_clients = []
        self.asr_batch_last_update = time.time()
        self.tts_batch = []
        self.tts_batch_clients = []
        self.tts_batch_last_update = time.time()

    def should_process_transcript(self, transcript, confidence, speech_duration):
        if not transcript:
            return False
        transcript = transcript.strip().lower()
        now = time.time()
        if len(transcript) <= 1:
            return False
        skip_phrases = ["thank you."]
        if transcript in skip_phrases:
            return False

        # Only check for exact duplicates within the repeat interval
        if transcript == self.last_transcript:
            if now - self.last_transcript_time < self.min_repeat_interval * 2:
                return False

        self.transcript_history.append(transcript)
        self.last_transcript = transcript
        self.last_transcript_time = now
        return True

class ClientSettingsManager:
    def __init__(self):
        self.client_settings = {}
        self.client_buffer_times = {}  # Track last update time for each client's buffer
        self.client_languages = {}  # Store language code for each client
        self.client_trigger_words = {}  # Store trigger word for each client

    def get_audio_server(self, client_id):
        if client_id not in self.client_settings:
            print(f"Creating new AudioServer for client {client_id}")
            self.client_settings[client_id] = AudioServer()
        return self.client_settings[client_id]
        
    def set_client_language(self, client_id, language_code):
        """Set the language code for a client."""
        if not language_code:
            language_code = "en"  # Default to English
        self.client_languages[client_id] = language_code
        print(f"Set language for client {client_id} to {language_code}")
        
    def get_client_language(self, client_id):
        """Get the language code for a client."""
        return self.client_languages.get(client_id, "en")  # Default to English
        
    def set_client_trigger_word(self, client_id, trigger_word):
        """Set the trigger word for a client."""
        if not trigger_word:
            trigger_word = DEFAULT_TRIGGER_WORD  # Default to config value
        self.client_trigger_words[client_id] = trigger_word.lower()
        print(f"Set trigger word for client {client_id} to '{trigger_word}'")
        
    def get_client_trigger_word(self, client_id):
        """Get the trigger word for a client."""
        return self.client_trigger_words.get(client_id, DEFAULT_TRIGGER_WORD)  # Default to config value

    def remove_client(self, client_id):
        if client_id in self.client_settings:
            print(f"Removing AudioServer for client {client_id}")
            del self.client_settings[client_id]
        if client_id in self.client_buffer_times:
            del self.client_buffer_times[client_id]
        if client_id in self.client_languages:
            del self.client_languages[client_id]
        if client_id in self.client_trigger_words:
            del self.client_trigger_words[client_id]

client_settings_manager = ClientSettingsManager()

################################################################################
# TTS HELPER FUNCTIONS
################################################################################

async def send_audio_frame(websocket, audio_data):
    """Send audio data using the framing protocol."""
    try:
        # Convert float32 to int16
        audio_int16 = np.clip(audio_data * 32768.0, -32768, 32767).astype(np.int16)
        frame = create_frame(FRAME_TYPE_AUDIO, audio_int16.tobytes())
        await websocket.send(frame)
    except Exception as e:
        print(f"Error sending audio frame: {e}")
        import traceback
        print(traceback.format_exc())

async def send_end_frame(websocket):
    """Send an end of utterance frame."""
    try:
        frame = create_frame(FRAME_TYPE_END)
        await websocket.send(frame)
        print("[TTS] Binary end frame sent")
    except Exception as e:
        print(f"Error sending end frame: {e}")
        import traceback
        print(traceback.format_exc())

# Keep track of accumulated text for each client
client_tts_buffers = {}

def find_sentence_split_index(text: str) -> int:
    """
    Returns the index of a valid sentence-ending punctuation character in text.
    This function considers '.', '!', and '?' as potential sentence terminators,
    but skips periods that appear to be part of a decimal number (i.e. when the
    period is both preceded and followed by a digit).
    If no valid sentence end is found, returns -1.
    """
    pattern = re.compile(r'(?<!\d)([.!?])(?=\s|$)')
    matches = list(pattern.finditer(text))
    if matches:
        return matches[-1].start()
    # Fallback: manually scan from the end
    for i in range(len(text) - 1, -1, -1):
        ch = text[i]
        if ch in ".!?":
            if ch == '.' and i > 0 and i < len(text) - 1 and text[i - 1].isdigit() and text[i + 1].isdigit():
                continue
            return i
    return -1

async def process_tts_batch(server, force=False):
    """Process TTS requests sequentially."""
    if not server.tts_batch:
        return

    try:
        # Process each TTS request sequentially
        while server.tts_batch:
            text, websocket, client_id = server.tts_batch.pop(0)
            
            # Get the client's language
            language_code = client_settings_manager.get_client_language(client_id)
            print(f"[TTS] Generating audio for client {client_id} in language {language_code}: {text}")
            
            try:
                # Get the appropriate pipeline for this language
                current_pipeline = pipeline_manager.get_pipeline(language_code)
                
                # Generate audio using pipeline call (which uses infer internally)
                generator = current_pipeline(
                    text,
                    voice=f'{voice_name}', # _heart
                    speed=1.0
                )
                
                audio_list = []
                for _, _, audio in generator:
                    if audio is not None:
                        if torch.is_tensor(audio):
                            audio = audio.detach().cpu().numpy()
                        if isinstance(audio, np.ndarray):
                            audio_list.append(audio.astype(np.float32))

                if audio_list:
                    try:
                        audio = np.concatenate(audio_list)
                        audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
                    except Exception as e:
                        continue
                    FRAME_SIZE = 8192
                    tasks = []

                    if len(audio) >= FRAME_SIZE:
                        tasks.append(send_audio_frame(websocket, audio[:FRAME_SIZE].copy()))

                    for i in range(FRAME_SIZE, len(audio) - FRAME_SIZE, FRAME_SIZE):
                        tasks.append(send_audio_frame(websocket, audio[i:i + FRAME_SIZE].copy()))
                        if len(tasks) >= 8:
                            await asyncio.gather(*tasks)
                            tasks = []

                    if tasks:
                        await asyncio.gather(*tasks)

                    await send_end_frame(websocket)

            except Exception as e:
                print(f"[TTS] Error processing item: {e}")
                continue

    except Exception as e:
        print(f"[TTS] Processing error: {e}")
    finally:
        # Clear any remaining items
        server.tts_batch.clear()
        server.tts_batch_clients = []
        server.tts_batch_last_update = time.time()

async def handle_tts_multi_utterances(websocket, text, client_id):
    """
    Accumulates text until we have a complete sentence, then adds to TTS batch.
    A sentence is considered complete if it ends with ., !, or ?.
    Text will also be flushed if it has been in the buffer for longer than TTS_BUFFER_TIMEOUT.
    """
    try:
        # Initialize or get buffer for this client
        if client_id not in client_tts_buffers:
            client_tts_buffers[client_id] = ""
        
        # Force flush any existing buffer before adding new text
        if client_tts_buffers[client_id].strip():
            buffer = client_tts_buffers[client_id].strip()
            if not buffer.endswith(('.', '!', '?')):
                buffer += "."
            server = client_settings_manager.get_audio_server(client_id)
            server.tts_batch.append((buffer, websocket, client_id))
            server.tts_batch_last_update = time.time()
            await process_tts_batch(server)
            client_tts_buffers[client_id] = ""
        
        # Append new text to buffer and update timestamp
        client_tts_buffers[client_id] += text
        buffer = client_tts_buffers[client_id]
        client_settings_manager.client_buffer_times[client_id] = time.time()
        
        # Process any complete sentences
        while True:
            split_index = find_sentence_split_index(buffer)
            if split_index < 0:
                # If no sentence boundary found, check for timeout
                now = time.time()
                last_update = client_settings_manager.client_buffer_times.get(client_id, now)
                if buffer.strip() and (now - last_update) > TTS_BUFFER_TIMEOUT:
                    # Force flush with a period
                    sentence = buffer.strip() + "."
                    client_tts_buffers[client_id] = ""
                    print(f"[TTS] Timeout flush: '{sentence}'")
                    
                    server = client_settings_manager.get_audio_server(client_id)
                    server.tts_batch.append((sentence, websocket, client_id))
                    server.tts_batch_last_update = time.time()
                    await process_tts_batch(server)
                break
                
            # Flush the sentence
            sentence = buffer[:split_index + 1].strip()
            remainder = buffer[split_index + 1:].strip()
            
            if sentence:
                server = client_settings_manager.get_audio_server(client_id)
                server.tts_batch.append((sentence, websocket, client_id))
                server.tts_batch_last_update = time.time()
                await process_tts_batch(server)
            
            buffer = remainder
            client_tts_buffers[client_id] = remainder
            
            if remainder:
                print(f"[TTS] Buffering remaining text: '{remainder}'")
                client_settings_manager.client_buffer_times[client_id] = time.time()
    
    except Exception as e:
        print(f"TTS Error: {e}")
        import traceback
        print(traceback.format_exc())

################################################################################
# SECURITY CHECK
################################################################################

def verify_api_key(websocket, client_id):
    try:
        server_api_key = CONFIG['server']['websocket']['api_key']
        if not server_api_key:
            print("No API key configured.")
            return False

        path_string = getattr(websocket.request, 'path', None) or getattr(websocket, 'path', None)
        print(f"Path from websocket: {path_string}")

        parsed_uri = urlparse(path_string)
        query_params = parse_qs(parsed_uri.query)

        client_api_key_list = query_params.get('api_key', [])
        if not client_api_key_list:
            print("No API key in query params.")
            return False
        client_api_key = client_api_key_list[0]
        if client_api_key != server_api_key:
            print("Client API key mismatch.")
            return False

        client_id_list = query_params.get('client_id', [])
        if not client_id_list:
            print("No Client ID in query params.")
            return False
        client_id_uri = client_id_list[0]
        if client_id_uri != str(client_id):
            print(f"Client ID mismatch: expected {client_id}, got {client_id_uri}")
            return False

        return True

    except Exception as e:
        print(f"Error verifying API key/ID: {e}")
        return False

################################################################################
# MAIN ASR HANDLER
################################################################################

async def process_asr_batch(server, force=False):
    """Process a batch of ASR requests."""
    if not server.asr_batch:
        return

    # Check if we should process the batch
    now = time.time()
    should_process = (
        force or
        len(server.asr_batch) >= ASR_MAX_BATCH_SIZE or
        (now - server.asr_batch_last_update) > ASR_BATCH_TIMEOUT
    )
    
    if not should_process:
        return

    try:
        print(f"Processing ASR batch of size {len(server.asr_batch)}")
        
        # Process in smaller sub-batches for better responsiveness
        for i in range(0, len(server.asr_batch), ASR_SUB_BATCH_SIZE):
            sub_batch = server.asr_batch[i:i + ASR_SUB_BATCH_SIZE]
            sub_batch_clients = server.asr_batch_clients[i:i + ASR_SUB_BATCH_SIZE]
            
            print(f"[ASR] Processing sub-batch of size {len(sub_batch)}")
            for item in sub_batch:
                print(f"[ASR DEBUG] Input shape: {item['array'].shape}, dtype: {item['array'].dtype}, "
                      f"range: [{item['array'].min():.3f}, {item['array'].max():.3f}]")
            
            try:
                print("[ASR] Running Whisper model on audio...")
                print(f"[ASR DEBUG] Processing batch of size: {len(sub_batch)}")
                for idx, item in enumerate(sub_batch):
                    print(f"[ASR DEBUG] Batch item {idx} - Shape: {item['array'].shape}, "
                          f"Sample rate: {item['sampling_rate']}, "
                          f"Duration: {len(item['array'])/item['sampling_rate']:.2f}s")
                
                try:
                    print("[ASR DEBUG] Pipeline input format:")
                    if isinstance(sub_batch, list):
                        print(f"[ASR DEBUG] Batch is a list of {len(sub_batch)} items")
                        for i, item in enumerate(sub_batch):
                            print(f"[ASR DEBUG] Item {i} keys: {item.keys()}")
                    else:
                        print(f"[ASR DEBUG] Batch is type: {type(sub_batch)}")
                    
                    # Process each audio input individually
                    asr_results = []
                    for item in sub_batch:
                        try:
                            print(f"[ASR DEBUG] Processing single audio input - Shape: {item['array'].shape}, "
                                  f"Duration: {len(item['array'])/item['sampling_rate']:.2f}s")
                            
                            # Ensure we're working with numpy arrays
                            audio_array = item["array"]
                            if torch.is_tensor(audio_array):
                                audio_array = audio_array.cpu().numpy()
                            
                            print(f"[ASR DEBUG] Input array shape: {audio_array.shape}")
                            
                            result = asr_pipeline(
                                {"array": audio_array, "sampling_rate": item["sampling_rate"]},
                                generate_kwargs={
                                    "task": "transcribe",
                                    "language": "en",
                                    "use_cache": True,
                                    "num_beams": 1,
                                    "do_sample": False
                                }
                            )
                            print(f"[ASR DEBUG] Single result: {result}")
                            asr_results.append(result)
                        except Exception as e:
                            print(f"[ASR ERROR] Failed to process audio input: {e}")
                            import traceback
                            print(f"[ASR ERROR] Traceback:\n{traceback.format_exc()}")
                except Exception as e:
                    print(f"[ASR ERROR] Pipeline execution failed with error: {str(e)}")
                    import traceback
                    print(f"[ASR ERROR] Full traceback:\n{traceback.format_exc()}")
                    raise
                print(f"[ASR DEBUG] Pipeline completed")
                print(f"[ASR DEBUG] Results type: {type(asr_results)}")
                print(f"[ASR DEBUG] Results content: {asr_results}")
            except Exception as e:
                print(f"[ASR ERROR] Pipeline failed: {e}")
                import traceback
                print(traceback.format_exc())
                continue

            # Process results for this sub-batch
            for idx, asr_result in enumerate(asr_results):
                try:
                    client_id, websocket, speech_duration = sub_batch_clients[idx]
                    transcript = asr_result["text"].strip()
                    confidence = asr_result.get("confidence", 0.0)
                    print(f"[ASR] Result {idx}: text='{transcript}', confidence={confidence}")

                    if transcript and server.should_process_transcript(
                        transcript,
                        confidence,
                        speech_duration
                    ):
                        # Get client's trigger word
                        client_trigger_word = client_settings_manager.get_client_trigger_word(client_id)
                        
                        # Check if transcript contains the trigger word
                        found, trigger_portion, text_after = find_trigger_word(transcript, client_trigger_word)
                        
                        if found:
                            print(f"[ASR] Trigger word '{trigger_portion}' detected for client {client_id}")
                            # Send the full transcript to the client
                            print(f"[ASR] {client_id}: '{transcript}'")
                            await websocket.send(transcript)
                        else:
                            print(f"[ASR] No trigger word detected in transcript for client {client_id}")
                except Exception as e:
                    print(f"[ASR] Error processing result {idx}: {e}")
                    continue

    except Exception as e:
        print(f"ASR Batch processing error: {e}")
    finally:
        # Clear batch
        server.asr_batch.clear()
        server.asr_batch_clients.clear()
        server.asr_batch_last_update = time.time()

async def transcribe_audio(websocket):
    client_id = None
    server = None

    try:
        path_string = getattr(websocket.request, 'path', None) or getattr(websocket, 'path', None)
        parsed_uri = urlparse(path_string)
        query_params = parse_qs(parsed_uri.query)
        
        # Extract client ID
        client_id_list = query_params.get('client_id', [])
        if not client_id_list:
            print("No client_id in URI.")
            await websocket.close(code=4000, reason="Client ID required.")
            return
        client_id = client_id_list[0]
        
        # Extract language code
        language_list = query_params.get('language', [])
        language_code = language_list[0] if language_list else "en"
        
        # Extract trigger word
        trigger_word_list = query_params.get('trigger_word', [])
        trigger_word = trigger_word_list[0] if trigger_word_list else DEFAULT_TRIGGER_WORD
        
        # Set client language and trigger word
        client_settings_manager.set_client_language(client_id, language_code)
        client_settings_manager.set_client_trigger_word(client_id, trigger_word)
        
        print(f"Client {client_id} using trigger word: '{trigger_word}'")

        if not verify_api_key(websocket, client_id):
            print("Client rejected: invalid key/ID.")
            await websocket.send("AUTH_FAILED")
            return
        
        await websocket.send("AUTH_OK")
        print(f"Client authenticated: {client_id}, Language: {language_code}")

        server = client_settings_manager.get_audio_server(client_id)

        async for message in websocket:
            if isinstance(message, bytes):
                # Microphone audio
                try:
                    # Process incoming audio through VAD
                    result = server.audio_core.process_audio(message)

                    # Send VAD status to client for GUI
                    vad_frame = create_frame(FRAME_TYPE_VAD, bytes([1 if result['is_speech'] else 0]))
                    await websocket.send(vad_frame)

                    # Process complete utterances
                    if result.get('is_complete'):
                        audio = result.get('audio')
                        if audio is not None and len(audio) > 0:
                            try:
                                print("[ASR] Processing complete utterance...")
                                print(f"[ASR DEBUG] Audio shape: {audio.shape}, duration: {len(audio)/16000:.2f}s")
                                
                                # Debug: Check utterance duration
                                utterance_duration = len(audio) / 16000.0
                                print(f"[DEBUG] Detected utterance duration: {utterance_duration:.2f}s")
                                if utterance_duration < 1.0:
                                    print("[WARNING] Utterance duration is very short; possible premature cutoff due to VAD settings.")
                                
                                try:
                                    # Ensure audio is in the correct format and shape
                                    if audio.dtype != np.float32:
                                        audio = audio.astype(np.float32)
                                    
                                    # Ensure audio is mono (average if multi-channel)
                                    if len(audio.shape) > 1:
                                        if audio.shape[1] > 1:
                                            audio = np.mean(audio, axis=1)
                                        else:
                                            audio = audio.squeeze()
                                    
                                    # Verify audio is 1D
                                    if len(audio.shape) != 1:
                                        print(f"[ASR ERROR] Unexpected audio shape after processing: {audio.shape}")
                                        continue
                                    
                                    print(f"[ASR DEBUG] Processed audio shape: {audio.shape}, range: [{audio.min():.3f}, {audio.max():.3f}]")
                                    
                                    print("[ASR] Running Whisper inference...")
                                    transcription = asr_pipeline(
                                        {"array": audio, "sampling_rate": 16000},
                                        generate_kwargs={
                                            "task": "transcribe",
                                            "language": "en",
                                            "use_cache": True,
                                            "num_beams": 1,
                                            "do_sample": False
                                        }
                                    )
                                    
                                    print(f"[ASR] Raw transcription: {transcription}")
                                    
                                    if transcription and server.should_process_transcript(
                                        transcription["text"],
                                        transcription.get("confidence", 0.0),
                                        result['speech_duration']
                                    ):
                                        transcript_text = transcription["text"]
                                        print(f"[ASR] Final transcript for {client_id}: '{transcript_text}'")
                                        
                                        # Get client's trigger word
                                        client_trigger_word = client_settings_manager.get_client_trigger_word(client_id)
                                        
                                        # Check if transcript contains the trigger word
                                        found, trigger_portion, text_after = find_trigger_word(transcript_text, client_trigger_word)
                                        
                                        if found:
                                            print(f"[ASR] Trigger word '{trigger_portion}' detected for client {client_id}")
                                            # Send the full transcript to the client
                                            await websocket.send(transcript_text)
                                        else:
                                            print(f"[ASR] No trigger word detected in transcript for client {client_id}")
                                        
                                except Exception as e:
                                    print(f"[ASR ERROR] Pipeline execution failed: {e}")
                                    import traceback
                                    print(f"[ASR ERROR] Full traceback:\n{traceback.format_exc()}")
                            except Exception as e:
                                print(f"ASR Error: {e}")
                                server.asr_batch.clear()
                                server.asr_batch_clients.clear()
                except Exception as e:
                    print(f"Error processing audio from {client_id}: {e}")

            elif isinstance(message, str):
                # Control / TTS
                if message.strip() == "VOICE_FILTER_ON":
                    server.enable_voice_filtering = True
                    print(f"Voice filter ON for {client_id}")
                elif message.strip() == "VOICE_FILTER_OFF":
                    server.enable_voice_filtering = False
                    print(f"Voice filter OFF for {client_id}")
                elif message.strip() == "RESET":
                    print(f"Buffer reset by {client_id}")
                elif message.strip() == "EXIT":
                    print(f"Client {client_id} exit requested.")
                    break
                elif message.startswith("TTS:"):
                    # Handle TTS request
                    text = message[4:].strip()
                    print(f"[TTS] Request from {client_id}: {text}")
                    asyncio.create_task(handle_tts_multi_utterances(websocket, text, client_id))
                else:
                    print(f"Unknown message from {client_id}: {message}")

    except websockets.ConnectionClosed as e:
        print(f"Client {client_id} disconnected: {e}")
    except Exception as e:
        print(f"Server error for {client_id}: {e}")
        import traceback
        print(traceback.format_exc())
    finally:
        if server and server.audio_core:
            server.audio_core.close()
        if client_id:
            # Process any remaining buffered text
            if client_id in client_tts_buffers and client_tts_buffers[client_id].strip():
                remaining = client_tts_buffers[client_id].strip()
                print(f"[TTS] Processing remaining buffered text before disconnect: '{remaining}'")
                try:
                    await handle_tts_multi_utterances(websocket, remaining + ".", client_id)
                    if server.tts_batch:
                        await process_tts_batch(server, force=True)
                except Exception as e:
                    print(f"[TTS] Error processing final buffer: {e}")
            
            if server.asr_batch:
                await process_asr_batch(server, force=True)
            
            if client_id in client_tts_buffers:
                del client_tts_buffers[client_id]
            client_settings_manager.remove_client(client_id)
        print(f"Cleaned up for client {client_id}")

################################################################################
# MAIN
################################################################################

async def cleanup_unused_pipelines_task():
    """Periodically check for and clean up unused pipelines."""
    while True:
        try:
            current_time = time.time()
            # Check every 10 minutes
            await asyncio.sleep(600)
            
            # Consider pipelines unused if not accessed in the last hour
            unused_threshold = current_time - 3600  # 1 hour
            
            for lang, last_used in list(pipeline_manager.last_used.items()):
                if last_used < unused_threshold and lang in pipeline_manager.pipelines:
                    print(f"Cleaning up unused pipeline for {lang}, not used for {(current_time - last_used)/60:.1f} minutes")
                    # Use the same cleanup logic as in _cleanup_if_needed
                    try:
                        pipeline_manager.pipelines[lang].model = pipeline_manager.pipelines[lang].model.to("cpu")
                        del pipeline_manager.pipelines[lang]
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        print(f"Successfully freed resources for {lang} pipeline")
                    except Exception as e:
                        print(f"Error cleaning up unused pipeline: {e}")
                        del pipeline_manager.pipelines[lang]
        except Exception as e:
            print(f"Error in cleanup task: {e}")
            await asyncio.sleep(600)  # Wait before trying again

async def main():
    try:
        # Start the cleanup task
        cleanup_task = asyncio.create_task(cleanup_unused_pipelines_task())
        
        host = CONFIG['server']['websocket']['host']
        port = CONFIG['server']['websocket']['port']
        async with websockets.serve(transcribe_audio, host, port):
            print(f"WebSocket server on ws://{host}:{port}")
            await asyncio.Future()  # keep running
    except Exception as e:
        print(f"Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
