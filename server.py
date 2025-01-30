#!/usr/bin/env python3

import asyncio
import websockets
import numpy as np
import time
import sys
import json
import os
from collections import deque
from urllib.parse import urlparse, parse_qs

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
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from kokoro import KPipeline
from src.audio_core import AudioCore

# Load configuration
with open('config.json', 'r') as f:
    CONFIG = json.load(f)

# Check if model paths exist
whisper_path = CONFIG['server']['models']['whisper']['path']
kokoro_path = CONFIG['server']['models']['kokoro']['path']

if not os.path.exists(whisper_path):
    print(f"Error: Whisper model path does not exist: {whisper_path}")
    sys.exit(1)
if not os.path.exists(kokoro_path):
    print(f"Error: Kokoro model path does not exist: {kokoro_path}")
    sys.exit(1)

# Trigger word
TRIGGER_WORD = CONFIG['assistant']['name'].lower()

device = "cuda:0" if has_gpu else "cpu"
torch_dtype = torch.float16 if has_gpu else torch.float32

MODEL_PATH = CONFIG['server']['models']['whisper']['path']

print(f"Device set to {device}")
print("Loading ASR model & processor...")

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True
).to(device)

processor = AutoProcessor.from_pretrained(MODEL_PATH)
asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device
)

print("Loading Kokoro TTS model...")
voice_name = CONFIG['server']['models']['kokoro']['voice_name']
print(f"Using voice name: {voice_name}")
tts_pipeline = KPipeline(lang_code='a')
tts_pipeline.model = tts_pipeline.model.to(device=device, dtype=torch.float32)

################################################################################
# AUDIO SERVER
################################################################################

class AudioServer:
    def __init__(self):
        with open('config.json', 'r') as f:
            self.config = json.load(f)
            
        self.audio_core = AudioCore()
        self.audio_core.noise_floor = -96.0
        self.audio_core.min_floor = -96.0
        self.audio_core.max_floor = -36.0
        self.audio_core.rms_level = -96.0
        self.audio_core.peak_level = -96.0

        self.enable_voice_filtering = False
        self.client_noise_floor = -96.0
        self.current_voice_profile = None
        self.voice_profile_timestamp = None

        self.transcript_history = deque(maxlen=10)
        self.min_confidence = 0.4
        self.short_phrase_confidence = 0.8
        self.last_transcript = ""
        self.last_transcript_time = 0
        self.min_repeat_interval = 2.0

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

        if speech_duration < self.audio_core.min_speech_duration:
            word_count = len(transcript.split())
            if word_count <= 2 and TRIGGER_WORD not in transcript:
                return False

        if transcript == self.last_transcript:
            if now - self.last_transcript_time < self.min_repeat_interval * 2:
                return False

        for past_transcript in self.transcript_history:
            past_words = set(past_transcript.lower().split())
            current_words = set(transcript.split())
            if past_words and current_words:
                common_words = past_words.intersection(current_words)
                similarity = len(common_words) / max(len(past_words), len(current_words))
                if similarity > 0.8:
                    return False

        self.transcript_history.append(transcript)
        self.last_transcript = transcript
        self.last_transcript_time = now
        return True

class ClientSettingsManager:
    def __init__(self):
        self.client_settings = {}

    def get_audio_server(self, client_id):
        if client_id not in self.client_settings:
            print(f"Creating new AudioServer for client {client_id}")
            self.client_settings[client_id] = AudioServer()
        return self.client_settings[client_id]

    def remove_client(self, client_id):
        if client_id in self.client_settings:
            print(f"Removing AudioServer for client {client_id}")
            del self.client_settings[client_id]

client_settings_manager = ClientSettingsManager()

################################################################################
# TTS HELPER FUNCTIONS
################################################################################

async def process_audio_chunk(websocket, chunk, fade_in=None, fade_out=None, fade_samples=32):
    try:
        if fade_in is not None:
            chunk[:fade_samples] *= fade_in
        if fade_out is not None:
            chunk[-fade_samples:] *= fade_out
            
        chunk_int16 = np.clip(chunk * 32768.0, -32768, 32767).astype(np.int16)
        await websocket.send(b'TTS:' + chunk_int16.tobytes())
    except Exception as e:
        print(f"Error sending TTS audio chunk: {e}")

# Keep track of accumulated text for each client
client_tts_buffers = {}

async def handle_tts_multi_utterances(websocket, text, client_id):
    """
    Accumulates text until we have a complete sentence, then generates and sends audio.
    A sentence is considered complete if it ends with ., !, or ?.
    """
    try:
        # Initialize or get buffer for this client
        if client_id not in client_tts_buffers:
            client_tts_buffers[client_id] = ""
        
        # Append new text to buffer
        client_tts_buffers[client_id] += text
        buffer = client_tts_buffers[client_id]
        
        # Check if we have any complete sentences
        sentence_end_chars = ".!?"
        last_end = -1
        for char in sentence_end_chars:
            pos = buffer.rfind(char)
            if pos > last_end:
                last_end = pos
        
        # If we have a complete sentence
        if last_end >= 0:
            # Extract the complete sentence(s)
            sentence = buffer[:last_end + 1].strip()
            # Keep the remainder in the buffer
            remainder = buffer[last_end + 1:].strip()
            client_tts_buffers[client_id] = remainder
            if remainder:
                print(f"[TTS] Buffering remaining text: '{remainder}'")
            
            if sentence:
                print(f"[TTS] Generating audio for sentence: '{sentence}'")
                generator = tts_pipeline(sentence, voice=f'{voice_name}_bella', speed=1)

                # Collect the entire TTS audio for this sentence
                audio_list = []
                for _, _, audio in generator:
                    if torch.is_tensor(audio):
                        audio = audio.detach().cpu().numpy()
                    audio_list.append(audio.astype(np.float32))

                # Concatenate partial TTS segments into one array
                if not audio_list:
                    return
                audio = np.concatenate(audio_list)
                audio = np.clip(audio, -1.0, 1.0).astype(np.float32)

                # Send chunked frames
                FRAME_SIZE = 512
                fade_samples = 32
                fade_in = np.linspace(0, 1, fade_samples).astype(np.float32)
                fade_out = np.linspace(1, 0, fade_samples).astype(np.float32)

                tasks = []

                if len(audio) >= FRAME_SIZE:
                    first_chunk = audio[:FRAME_SIZE].copy()
                    tasks.append(process_audio_chunk(websocket, first_chunk, fade_in=fade_in))

                for i in range(FRAME_SIZE, len(audio) - FRAME_SIZE, FRAME_SIZE):
                    chunk = audio[i:i + FRAME_SIZE].copy()
                    if i + FRAME_SIZE >= len(audio) - FRAME_SIZE:
                        tasks.append(process_audio_chunk(websocket, chunk, fade_out=fade_out))
                    else:
                        tasks.append(process_audio_chunk(websocket, chunk))

                    if len(tasks) >= 8:
                        await asyncio.gather(*tasks)
                        tasks = []

                if tasks:
                    await asyncio.gather(*tasks)

                # Send TTS_END after the complete sentence
                print(f"[TTS] Sending TTS_END for sentence: '{sentence}'")
                await websocket.send(b"TTS_END")
                print("[TTS] TTS_END sent")

    except Exception as e:
        print(f"TTS Error: {e}")
        await websocket.send(b"TTS_ERROR")

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

async def transcribe_audio(websocket):
    client_id = None
    server = None

    try:
        path_string = getattr(websocket.request, 'path', None) or getattr(websocket, 'path', None)
        parsed_uri = urlparse(path_string)
        query_params = parse_qs(parsed_uri.query)
        client_id_list = query_params.get('client_id', [])
        if not client_id_list:
            print("No client_id in URI.")
            await websocket.close(code=4000, reason="Client ID required.")
            return
        client_id = client_id_list[0]

        if not verify_api_key(websocket, client_id):
            print("Client rejected: invalid key/ID.")
            await websocket.send("AUTH_FAILED")
            return
        
        await websocket.send("AUTH_OK")
        print(f"Client authenticated: {client_id}")

        server = client_settings_manager.get_audio_server(client_id)

        async for message in websocket:
            if isinstance(message, bytes):
                # Microphone audio
                try:
                    chunk_data, sr = server.audio_core.bytes_to_float32_audio(message, sample_rate=16000)
                    result = server.audio_core.process_audio(chunk_data)

                    if result.get('is_complete') and result.get('is_speech'):
                        audio = result.get('audio')
                        if audio is not None and len(audio) > 0:
                            try:
                                asr_result = asr_pipeline(
                                    {"array": audio, "sampling_rate": sr},
                                    return_timestamps=True,
                                    generate_kwargs={"task": "transcribe",
                                                     "language": "english",
                                                     "use_cache": False}
                                )
                                transcript = asr_result["text"].strip()
                                confidence = asr_result.get("confidence", 0.0)

                                if transcript and server.should_process_transcript(
                                    transcript,
                                    confidence,
                                    result['speech_duration']
                                ):
                                    print(f"[ASR] {client_id}: '{transcript}'")
                                    await websocket.send(transcript)

                            except Exception as e:
                                print(f"ASR Error: {e}")

                except Exception as e:
                    print(f"Error processing audio from {client_id}: {e}")

            elif isinstance(message, str):
                # Control / TTS
                if message.startswith("NOISE_FLOOR:"):
                    try:
                        parts = message.split(":")
                        noise_floor = float(parts[1])
                        msg_client_id = parts[2] if len(parts) > 2 else None

                        if msg_client_id != client_id:
                            print(f"NOISE_FLOOR client mismatch. Expected {client_id}, got {msg_client_id}")
                            continue

                        if -120 < noise_floor < 0:
                            server.client_noise_floor = noise_floor
                            server.audio_core.noise_floor = noise_floor
                            server.audio_core.min_floor = noise_floor - 5
                            server.audio_core.max_floor = noise_floor + 45
                            print(f"Set {client_id} noise floor to {noise_floor:.1f} dB")
                            await websocket.send("READY")
                        else:
                            print(f"Invalid noise floor: {noise_floor}")
                            await websocket.send("ERROR:Invalid noise floor")

                    except Exception as e:
                        print(f"Error with noise floor from {client_id}: {e}")
                        await websocket.send("ERROR:Noise floor fail")

                elif message.strip() == "VOICE_FILTER_ON":
                    server.enable_voice_filtering = True
                    print(f"Voice filter ON for {client_id}")
                elif message.strip() == "VOICE_FILTER_OFF":
                    server.enable_voice_filtering = False
                    server.current_voice_profile = None
                    server.voice_profile_timestamp = None
                    print(f"Voice filter OFF for {client_id}")
                elif message.strip() == "RESET":
                    print(f"Buffer reset by {client_id}")
                elif message.strip() == "EXIT":
                    print(f"Client {client_id} exit requested.")
                    break
                elif message.startswith("TTS:"):
                    # **Now handle multi-utterance TTS**:
                    text = message[4:].strip()
                    print(f"[TTS] Request from {client_id}: {text}")
                    asyncio.create_task(handle_tts_multi_utterances(websocket, text, client_id))
                else:
                    print(f"Unknown message from {client_id}: {message}")

    except websockets.ConnectionClosed as e:
        print(f"Client {client_id} disconnected: {e}")
    except Exception as e:
        print(f"Server error for {client_id}: {e}")
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
                except Exception as e:
                    print(f"[TTS] Error processing final buffer: {e}")
            
            # Clean up client's TTS buffer
            if client_id in client_tts_buffers:
                del client_tts_buffers[client_id]
            client_settings_manager.remove_client(client_id)
        print(f"Cleaned up for client {client_id}")

################################################################################
# MAIN
################################################################################

async def main():
    try:
        host = "0.0.0.0"
        port = CONFIG['server']['websocket']['port']
        async with websockets.serve(transcribe_audio, host, port):
            print(f"WebSocket server on ws://{host}:{port}")
            await asyncio.Future()  # keep running
    except Exception as e:
        print(f"Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
