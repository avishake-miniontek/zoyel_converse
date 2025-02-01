#!/usr/bin/env python3
"""
Audio Chat Client

This script captures audio from your microphone, sends it to a server for
real-time transcription, and displays the transcribed text. It also receives
TTS audio data from the server, which it plays via AudioOutput.

Usage:
    python client.py [--no-gui]

In addition to its normal operation, this version saves a local WAV file every
10 seconds containing the post-processed audio being sent to the server. This
helps you inspect the original client-side audio for debugging.
"""

import asyncio
import websockets
import numpy as np
import samplerate
import sys
import os
import platform
import uuid
from collections import deque
import threading
import queue
import json
import time
import argparse
import sounddevice as sd
import datetime
from scipy.io.wavfile import write as wav_write

# Conditionally import GUI modules
gui_available = True
try:
    import tkinter as tk
    from src.graphical_interface import AudioInterface as GraphicalInterface
except ImportError:
    gui_available = False

from src.headless_interface import HeadlessAudioInterface
from src.audio_core import AudioCore
from src.llm_client import LLMClient
from src.audio_output import AudioOutput  # The updated file from above

# Load configuration
with open('config.json', 'r') as f:
    CONFIG = json.load(f)

# Server configuration
API_KEY = CONFIG['server']['websocket']['api_key']
SERVER_HOST = CONFIG['server']['websocket']['host']
SERVER_PORT = CONFIG['server']['websocket']['port']

# Generate unique client ID
CLIENT_ID = uuid.uuid4()
SERVER_URI = f"ws://{SERVER_HOST}:{SERVER_PORT}?api_key={API_KEY}&client_id={CLIENT_ID}"

# Trigger word configuration
TRIGGER_WORD = CONFIG['assistant']['name']

################################################################################
# ASYNCHRONOUS TASKS
################################################################################

async def record_and_send_audio(websocket, audio_interface, audio_core):
    """
    Continuously read audio from the microphone, send raw PCM frames to the server,
    and (for debugging) save a WAV file locally every 10 seconds.
    The audio is automatically resampled to 16kHz if needed.
    """
    error_count = 0
    max_errors = 3

    # Buffer to accumulate audio for local saving
    save_buffer = []  # Each element is a numpy array (int16)
    last_save_time = time.time()
    
    try:
        if not audio_core or not audio_core.stream:
            print("Error: Audio core not properly initialized")
            return

        stream = audio_core.stream
        rate = audio_core.rate
        needs_resampling = audio_core.needs_resampling

        print("\nStart speaking...")

        # Create resampler if needed
        if needs_resampling:
            resampler = samplerate.Resampler('sinc_best')
            ratio = 16000 / rate
        else:
            resampler = None

        while True:
            try:
                # Read audio chunk from mic with error handling
                try:
                    # Read returns tuple (data, overflowed)
                    # data shape is (frames, channels) for multi-channel
                    audio_data = stream.read(audio_core.CHUNK)[0]
                except Exception as e:
                    if isinstance(e, sd.PortAudioError) and "Invalid stream pointer" in str(e):
                        print(f"Fatal stream error: {e}")
                        raise  # Trigger reconnection
                    print(f"Stream read error: {e}")
                    await asyncio.sleep(0.1)
                    continue

                # Convert stereo to mono by averaging the channels
                if len(audio_data.shape) == 2 and audio_data.shape[1] > 1:
                    audio_data = np.mean(audio_data, axis=1)

                # Resample to 16kHz if needed
                if needs_resampling and resampler:
                    audio_data = resampler.process(audio_data, ratio, end_of_input=False)

                # Scale to int16
                final_data = np.clip(audio_data * 32767.0, -32767, 32767).astype(np.int16)
                
                # Append to local save buffer (for debugging)
                save_buffer.append(final_data.copy())

                # Save a WAV file every 10 seconds
                current_time = time.time()
                if current_time - last_save_time >= 10:
                    try:
                        combined = np.concatenate(save_buffer)
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"client_record_{timestamp}.wav"
                        wav_write(filename, 16000, combined)
                        print(f"[DEBUG] Saved local recording to {filename}")
                    except Exception as e:
                        print(f"[DEBUG] Error saving WAV file: {e}")
                    save_buffer = []
                    last_save_time = current_time

                # Send the audio to the server
                try:
                    await websocket.send(final_data.tobytes())
                    error_count = 0
                except websockets.ConnectionClosed:
                    raise
                except Exception as e:
                    error_count += 1
                    if error_count >= max_errors:
                        print("\nToo many consecutive errors, triggering reconnection...")
                        raise
                    print(f"Error sending audio chunk (attempt {error_count}/{max_errors}): {e}")
                    await asyncio.sleep(0.1)
                    continue

            except websockets.ConnectionClosed:
                raise
            except Exception as e:
                error_count += 1
                if error_count >= max_errors:
                    print("\nToo many consecutive errors, triggering reconnection...")
                    raise
                print(f"Error processing audio chunk: {e}")
                await asyncio.sleep(0.1)
                continue

            await asyncio.sleep(0.001)

    except asyncio.CancelledError:
        print("Audio recording cancelled.")
    except Exception as e:
        print(f"Error in record_and_send_audio: {e}")
        raise
    finally:
        # Do not close the stream here; audio_core manages it.
        pass

async def receive_transcripts(websocket, audio_interface):
    """
    Continuously receive transcripts (text) and TTS frames (bytes) from the server.
    If text is received, handle potential LLM triggers. If binary frames are received,
    pass them to audio_output for playback.
    """
    error_count = 0
    max_errors = 3

    llm_client = LLMClient()

    global audio_output

    async def handle_llm_chunk(text):
        """When LLM yields text, send it to the server as a TTS request."""
        await websocket.send(f"TTS:{text}")

    async def process_text(text: str):
        """Send text to the LLM for processing."""
        try:
            if not text.lower().startswith(TRIGGER_WORD.lower()):
                text = f"{TRIGGER_WORD}, {text}"
            await audio_output.start_stream()
            await llm_client.process_trigger(text, callback=handle_llm_chunk)
        except Exception as e:
            print(f"[ERROR] Failed to process text input: {e}")
            import traceback
            print(traceback.format_exc())

    while True:
        try:
            msg = await asyncio.wait_for(websocket.recv(), timeout=0.1)
            error_count = 0
        except asyncio.TimeoutError:
            if audio_interface and audio_interface.has_gui:
                text_input = audio_interface.get_text_input()
                if text_input is not None:
                    asyncio.create_task(process_text(text_input))
            continue
        except websockets.ConnectionClosed:
            print("Server connection closed (transcript reader).")
            raise
        except Exception as e:
            error_count += 1
            if error_count >= max_errors:
                print("\nToo many consecutive errors in receive_transcripts, reconnecting...")
                raise
            print(f"Error receiving message (attempt {error_count}/{max_errors}): {e}")
            await asyncio.sleep(0.1)
            continue

        if isinstance(msg, bytes):
            try:
                if len(msg) >= 9:
                    magic = msg[:4]
                    frame_type = msg[4]
                    if magic == b'MIRA':
                        if frame_type == 0x03:
                            if audio_interface and audio_interface.has_gui:
                                is_speech = bool(msg[9])
                                audio_interface.process_vad(is_speech)
                        else:
                            asyncio.create_task(audio_output.play_chunk(msg))
            except Exception as e:
                print(f"[ERROR] Failed to process frame: {e}")
                import traceback
                print(traceback.format_exc())
            continue

        if msg == "TTS_ERROR":
            print("[ERROR] Server TTS generation failed.")
            continue

        msg_lower = msg.lower()
        trigger_pos = msg_lower.find(TRIGGER_WORD.lower())
        if trigger_pos != -1:
            print(f"\n[TRANSCRIPT] {msg}")
            trigger_text = msg[trigger_pos:]
            try:
                await audio_output.start_stream()
                asyncio.create_task(llm_client.process_trigger(trigger_text, callback=handle_llm_chunk))
            except Exception as e:
                print(f"[ERROR] LLM trigger processing: {e}")

class AsyncThread(threading.Thread):
    """
    Runs the main async loop to connect to the server, handle microphone audio,
    and receive transcripts.
    """
    def __init__(self, audio_interface, audio_core):
        super().__init__()
        self.audio_interface = audio_interface
        self.audio_core = audio_core
        self.loop = None
        self.websocket = None
        self.tasks = []
        self.running = True
        self.daemon = True

    async def connect_to_server(self):
        """Connect to the WebSocket server and perform basic auth."""
        try:
            self.websocket = await websockets.connect(SERVER_URI)
            print(f"Connected to {SERVER_URI}")

            # Wait for auth confirmation from server
            auth_response = await asyncio.wait_for(self.websocket.recv(), timeout=2.0)
            if auth_response != "AUTH_OK":
                raise Exception(f"Auth failed: {auth_response}")
            return True
        except Exception as e:
            print(f"[ERROR] connect_to_server: {e}")
            if self.websocket:
                await self.websocket.close()
            return False

    async def handle_server_connection(self):
        """
        Attempt to connect and then start tasks.
        (Note: Noise floor functionality has been removed.)
        """
        retry_count = 0
        max_retries = CONFIG['client']['retry']['max_attempts']
        delay_seconds = CONFIG['client']['retry']['delay_seconds']

        while retry_count < max_retries and self.running:
            print(f"\nAttempting to connect... (attempt {retry_count+1}/{max_retries})")
            if not await self.connect_to_server():
                retry_count += 1
                if retry_count < max_retries:
                    print(f"Retry in {delay_seconds} seconds...")
                    await asyncio.sleep(delay_seconds)
                continue

            print("[CLIENT] Server ready.")

            self.tasks = [
                asyncio.create_task(record_and_send_audio(self.websocket, self.audio_interface, self.audio_core)),
                asyncio.create_task(receive_transcripts(self.websocket, self.audio_interface))
            ]

            done, pending = await asyncio.wait(self.tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

            if not self.running:
                return

            print("[CLIENT] Connection lost, will retry...")
            retry_count += 1
            if retry_count < max_retries:
                print(f"Retrying in {delay_seconds} seconds...")
                await asyncio.sleep(delay_seconds)

    async def initialize_audio(self):
        """Initialize microphone and speaker."""
        try:
            await audio_output.initialize()
            stream, device_info, rate, needs_resampling = self.audio_core.init_audio_device()
            if None in (stream, device_info, rate, needs_resampling):
                print("[CLIENT] Error initializing microphone.")
                return False

            if self.audio_interface and self.audio_interface.has_gui:
                self.audio_interface.input_device_queue.put(device_info['name'])

            print(f"[CLIENT] Audio initialized. Device: {device_info['name']}")
            return True
        except Exception as e:
            print(f"[CLIENT] Audio init error: {e}")
            return False

    async def async_main(self):
        """Main async loop for the client."""
        try:
            if not await self.initialize_audio():
                return
            await self.handle_server_connection()
        except Exception as e:
            print(f"[CLIENT] Error in async main: {e}")
        finally:
            await self.cleanup()

    async def cleanup(self):
        """Clean up resources."""
        try:
            if audio_output:
                await audio_output.close()
            if self.audio_core:
                self.audio_core.close()
            if self.websocket:
                await self.websocket.close()
            for t in self.tasks:
                if not t.done():
                    t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass
        except Exception as e:
            print(f"[CLIENT] Error during cleanup: {e}")

    def stop(self):
        """Stop everything gracefully."""
        self.running = False
        if self.loop and not self.loop.is_closed():
            async def shutdown():
                await self.cleanup()
            future = asyncio.run_coroutine_threadsafe(shutdown(), self.loop)
            try:
                future.result(timeout=5)
            except Exception as e:
                print(f"[CLIENT] Error stopping: {e}")

    def run(self):
        """Thread run: create an event loop and run async_main."""
        try:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self.async_main())
        except asyncio.CancelledError:
            print("[CLIENT] Async ops cancelled.")
        except Exception as e:
            print(f"[CLIENT] Exception in async thread: {e}")
        finally:
            pending = asyncio.all_tasks(self.loop)
            for task in pending:
                task.cancel()
            self.loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            self.loop.stop()
            self.loop.close()

def run_client():
    """
    Entrypoint for the client. Handles CLI args, selects GUI or headless mode,
    spawns AsyncThread, and runs until termination.
    """
    global audio_output

    async_thread = None
    audio_interface = None
    audio_core = None

    try:
        parser = argparse.ArgumentParser(description='Audio Chat Client')
        parser.add_argument('--no-gui', action='store_true', help='Run in headless mode')
        args = parser.parse_args()

        use_gui = not args.no_gui and gui_available

        if use_gui and platform.system() == 'Darwin':
            os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
            os.environ['TK_SILENCE_DEPRECATION'] = '1'

        audio_output = AudioOutput()
        audio_output.initialize_sync()

        audio_core = AudioCore()

        interface_params = {
            'input_device_name': "Initializing...",
            'output_device_name': audio_output.get_device_name(),
            'on_input_change': None,
            'on_output_change': lambda name: audio_output.set_device_by_name(name)
        }
        if use_gui:
            from src.graphical_interface import AudioInterface as GraphicalInterface
            audio_interface = GraphicalInterface(**interface_params)
        else:
            from src.headless_interface import HeadlessAudioInterface
            audio_interface = HeadlessAudioInterface(**interface_params)
            print("[CLIENT] Running in headless mode.")

        async_thread = AsyncThread(audio_interface, audio_core)
        async_thread.start()

        if use_gui:
            def on_window_close():
                print("[CLIENT] Window close -> stopping.")
                if async_thread:
                    async_thread.stop()
                if audio_core:
                    audio_core.close()
                audio_interface.close()

            audio_interface.root.protocol("WM_DELETE_WINDOW", on_window_close)
            audio_interface.root.mainloop()
        else:
            while async_thread.is_alive():
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n[CLIENT] Ctrl-C -> shutting down...")
        if async_thread:
            async_thread.stop()
        if audio_interface:
            audio_interface.close()
        if audio_core:
            audio_core.close()
        if audio_output:
            asyncio.run(audio_output.close())
    except Exception as e:
        print(f"[CLIENT] Error: {e}")
        if "--no-gui" in sys.argv:
            import traceback
            print("\nTraceback:")
            print(traceback.format_exc())
    finally:
        if async_thread:
            async_thread.join(timeout=5)

if __name__ == "__main__":
    try:
        run_client()
    except KeyboardInterrupt:
        print("\n[CLIENT] KeyboardInterrupt -> shutting down.")
    except Exception as e:
        print(f"\n[CLIENT] Fatal error: {e}")
