#!/usr/bin/env python3
"""
Audio Chat Client

This script captures audio from your microphone, sends it to a server for
real-time transcription, and displays the transcribed text. It also receives
TTS audio data from the server, which it plays via AudioOutput.

Usage:
    python client.py

It will:
1. Connect to the transcription server
2. Select a microphone
3. Start streaming audio
4. Display any transcripts and handle TTS playback
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
    Continuously read audio from the microphone and send raw PCM frames to the server.
    The audio is automatically resampled to 16kHz if needed.
    """
    error_count = 0
    max_errors = 3
    
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
                        raise  # This will trigger reconnection
                    print(f"Stream read error: {e}")
                    await asyncio.sleep(0.1)
                    continue

                # First convert to mono if needed
                if len(audio_data.shape) == 2 and audio_data.shape[1] > 1:
                    audio_data = audio_data[:, 0]  # Take left channel

                # Then resample to 16kHz if needed
                if needs_resampling and resampler:
                    audio_data = resampler.process(audio_data, ratio, end_of_input=False)

                # Convert to int16 and send to server
                final_data = np.clip(audio_data * 32767.0, -32767, 32767).astype(np.int16)

                # Send to server
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
        # Do not close the stream here; audio_core manages it
        pass

async def receive_transcripts(websocket, audio_interface):
    """
    Continuously receive transcripts (text) and TTS frames (bytes) from the server.
    If it's text, we handle possible LLM triggers. If it's bytes, we hand it to audio_output
    for playback. This approach uses 'audio_output.play_chunk(...)' for the TTS frames.
    """
    error_count = 0
    max_errors = 3

    llm_client = LLMClient()

    # Access the globally-initialized audio_output (we'll init it below main).
    global audio_output

    async def handle_llm_chunk(text):
        """When LLM yields text, send it back to server as TTS request."""
        await websocket.send(f"TTS:{text}")

    async def process_text(text: str):
        """Send text to the LLM, cause TTS output, etc."""
        try:
            # Prepend the trigger if missing
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
            # Check if user typed text in GUI
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

        # Process binary messages and text messages separately
        if isinstance(msg, bytes):
            try:
                if len(msg) >= 9:  # Minimum frame size (4 magic + 1 type + 4 length)
                    magic = msg[:4]
                    frame_type = msg[4]
                    if magic == b'MIRA':
                        if frame_type == 0x03:
                            # VAD status frame; process it directly (payload is at msg[9])
                            if audio_interface and audio_interface.has_gui:
                                is_speech = bool(msg[9])
                                audio_interface.process_vad(is_speech)
                        else:
                            # For audio data (0x01) and end-of-utterance (0x02) frames,
                            # pass the full frame (header + payload) to the audio output.
                            asyncio.create_task(audio_output.play_chunk(msg))
            except Exception as e:
                print(f"[ERROR] Failed to process frame: {e}")
                import traceback
                print(traceback.format_exc())
            continue

        # Otherwise it's text
        if msg == "TTS_ERROR":
            print("[ERROR] Server TTS generation failed.")
            continue

        # Check for the trigger word in text messages
        msg_lower = msg.lower()
        trigger_pos = msg_lower.find(TRIGGER_WORD.lower())
        if trigger_pos != -1:
            print(f"\n[TRANSCRIPT] {msg}")
            # Everything from trigger to end
            trigger_text = msg[trigger_pos:]
            try:
                await audio_output.start_stream()
                asyncio.create_task(llm_client.process_trigger(trigger_text, callback=handle_llm_chunk))
            except Exception as e:
                print(f"[ERROR] LLM trigger processing: {e}")

class AsyncThread(threading.Thread):
    """
    Runs the main async loop to connect to server, handle mic audio, and transcripts.
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
        """Connect to the WebSocket server and handle auth."""
        try:
            self.websocket = await websockets.connect(SERVER_URI)
            print(f"Connected to {SERVER_URI}")

            # Wait for auth
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
        """Attempt to connect + start tasks, with retries."""
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

            # Send calibrated noise floor
            try:
                nf_str = f"NOISE_FLOOR:{self.audio_core.noise_floor}:{CLIENT_ID}"
                print(f"Sending noise floor: {nf_str}")
                await self.websocket.send(nf_str)

                response = await asyncio.wait_for(self.websocket.recv(), timeout=5.0)
                if response != "READY":
                    print(f"[ERROR] Unexpected server response: {response}")
                    continue
                print("[CLIENT] Server ready.")

                # Start tasks: mic->server, transcripts->console
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

            except asyncio.TimeoutError:
                print("[CLIENT] Timed out waiting for server.")
                continue
            except websockets.ConnectionClosed:
                print("[CLIENT] WebSocket closed.")
                continue
            except Exception as e:
                print(f"[CLIENT] Server comm error: {e}")
                continue

        if retry_count >= max_retries:
            print("[CLIENT] Max retries exceeded. Shutting down.")
            self.running = False
            if self.audio_interface and self.audio_interface.has_gui:
                self.audio_interface.root.after(0, self.audio_interface._on_closing)

    async def initialize_audio(self):
        """Initialize mic + speaker."""
        try:
            # Initialize audio output (speaker)
            await audio_output.initialize()

            # Initialize mic (audio_core)
            stream, device_info, rate, needs_resampling = self.audio_core.init_audio_device()
            if None in (stream, device_info, rate, needs_resampling):
                print("[CLIENT] Error init mic.")
                return False

            # If you have a GUI, show selected device
            if self.audio_interface and self.audio_interface.has_gui:
                self.audio_interface.input_device_queue.put(device_info['name'])

            if self.audio_core.noise_floor is None:
                print("[CLIENT] Error: No noise floor calibration.")
                return False

            print(f"[CLIENT] Noise floor: {self.audio_core.noise_floor:.1f} dB")
            return True
        except Exception as e:
            print(f"[CLIENT] Audio init error: {e}")
            return False

    async def async_main(self):
        """Main async loop for the client."""
        try:
            # Initialize audio (mic + output)
            if not await self.initialize_audio():
                return

            # Handle server connect + run tasks
            await self.handle_server_connection()
        except Exception as e:
            print(f"[CLIENT] Error in async main: {e}")
        finally:
            await self.cleanup()

    async def cleanup(self):
        """Clean up resources (audio, tasks)."""
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
    Entrypoint for the client. Handles CLI args, chooses GUI or headless,
    spawns AsyncThread, and runs until done.
    """
    global audio_output

    async_thread = None
    audio_interface = None
    audio_core = None

    try:
        parser = argparse.ArgumentParser(description='Audio Chat Client')
        parser.add_argument('--no-gui', action='store_true',
                            help='Run in headless mode')
        args = parser.parse_args()

        use_gui = not args.no_gui and gui_available

        # For macOS + tkinter
        if use_gui and platform.system() == 'Darwin':
            os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
            os.environ['TK_SILENCE_DEPRECATION'] = '1'

        # 1) Create our AudioOutput global instance
        audio_output = AudioOutput()

        # 2) Initialize it synchronously just to confirm device
        audio_output.initialize_sync()

        # 3) Create the AudioCore for microphone
        audio_core = AudioCore()

        # 4) Create interface
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

        # 5) Start our main async thread
        async_thread = AsyncThread(audio_interface, audio_core)
        async_thread.start()

        # 6) If GUI mode, run mainloop
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
            # Headless -> wait for ctrl+C
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
