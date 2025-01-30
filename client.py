#!/usr/bin/env python3
"""
Audio Chat Client

This script captures audio from your microphone, sends it to a server for real-time
transcription, and displays the transcribed text. It automatically selects the most
appropriate input device and handles audio capture and streaming.

Usage:
    python client.py

The script will automatically:
1. Connect to the transcription server
2. Select the best available microphone
3. Start capturing and streaming audio
4. Display transcribed text as it becomes available
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

# Conditionally import GUI-related modules
gui_available = True
try:
    import tkinter as tk
    from src.graphical_interface import AudioInterface as GraphicalInterface
except ImportError:
    gui_available = False

from src.headless_interface import HeadlessAudioInterface
from src.audio_core import AudioCore
from src.llm_client import LLMClient
from src.audio_output import AudioOutput

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
    The audio is automatically resampled to 16kHz if the device doesn't support it directly.
    For local GUI display, we call audio_core.process_audio(...) to get the current
    'is_speech' state and update our rolling speech-detection graph.
    """
    error_count = 0
    max_errors = 3  # Maximum consecutive errors before raising exception
    
    try:
        if not audio_core or not audio_core.stream:
            print("\nError: Audio core not properly initialized")
            return

        stream = audio_core.stream
        rate = audio_core.rate
        needs_resampling = audio_core.needs_resampling

        print("\nStart speaking...")

        # Create resampler only if needed
        resampler = None
        if needs_resampling:
            # Use scipy's resample_poly for more efficient resampling
            from scipy import signal
            gcd = np.gcd(16000, rate)
            up = 16000 // gcd
            down = rate // gcd
            
        while True:
            try:
                # Read audio from mic
                try:
                    audio_data = stream.read(audio_core.CHUNK)[0]  # float32 array
                except Exception as e:
                    print(f"Stream read error (trying to recover): {e}")
                    await asyncio.sleep(0.1)
                    continue

                # Process audio for server-side transcription
                result = audio_core.process_audio(audio_data)
                
                # Separate VAD processing for GUI visualization
                if audio_interface and audio_interface.has_gui:
                    is_speech = audio_core.process_audio_vad(audio_data)
                    audio_interface.process_vad(is_speech)

                # Resample and convert to int16 for sending to server
                if needs_resampling:
                    try:
                        # Use resample_poly for more efficient resampling
                        resampled_data = signal.resample_poly(audio_data, up, down)
                        final_data = np.clip(resampled_data * 32768.0, -32768, 32767).astype(np.int16)
                    except Exception as e:
                        print(f"Error during resampling: {e}")
                        continue
                else:
                    final_data = np.clip(audio_data * 32768.0, -32768, 32767).astype(np.int16)

                try:
                    # Send audio data (16-bit PCM) to the server
                    await websocket.send(final_data.tobytes())
                    error_count = 0  # Reset error count on successful send
                except websockets.ConnectionClosed:
                    raise  # Re-raise to trigger reconnection
                except Exception as e:
                    error_count += 1
                    if error_count >= max_errors:
                        print("\nToo many consecutive errors, triggering reconnection...")
                        raise
                    print(f"Error sending audio chunk (attempt {error_count}/{max_errors}): {e}")
                    await asyncio.sleep(0.1)
                    continue

            except websockets.ConnectionClosed:
                raise  # Re-raise to trigger reconnection
            except Exception as e:
                error_count += 1
                if error_count >= max_errors:
                    print("\nToo many consecutive errors, triggering reconnection...")
                    raise
                print(f"Error processing audio chunk: {e}")
                await asyncio.sleep(0.1)
                continue

            # Small delay to prevent high CPU usage
            await asyncio.sleep(0.001)

    except asyncio.CancelledError:
        print("\nAudio recording cancelled")
    except Exception as e:
        print(f"\nError in record_and_send_audio: {e}")
        raise  # Re-raise to trigger reconnection
    finally:
        # Don't close the stream here, it's managed by audio_core
        pass

# Initialize audio output globally
audio_output = AudioOutput()

async def receive_transcripts(websocket, audio_interface):
    """
    Continuously receive transcripts from the server and print them in the client console.
    Also detects if the trigger word appears in the transcript.
    When triggered, sends the transcript to the LLM for processing and handles TTS playback.
    Additionally processes text input from the GUI.
    """
    error_count = 0
    max_errors = 3  # Maximum consecutive errors before raising exception
    
    try:
        llm_client = LLMClient()

        # Define callback for LLM to send TTS requests
        async def handle_llm_chunk(text):
            await websocket.send(f"TTS:{text}")

        async def process_text(text: str):
            """Process text input through LLM and TTS."""
            try:
                # Prepend trigger word if not present
                if not text.lower().startswith(TRIGGER_WORD.lower()):
                    text = f"{TRIGGER_WORD}, {text}"
                
                await audio_output.start_stream()
                await llm_client.process_trigger(text, callback=handle_llm_chunk)
            except Exception as e:
                print(f"\n[ERROR] Failed to process text input: {e}")
                import traceback
                print(traceback.format_exc())

        while True:
            try:
                # Check for websocket messages with a short timeout
                msg = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                error_count = 0  # Reset error count on successful receive
            except asyncio.TimeoutError:
                # If no websocket message, check for text input
                if audio_interface and audio_interface.has_gui:
                    text_input = audio_interface.get_text_input()
                    if text_input is not None:
                        asyncio.create_task(process_text(text_input))
                continue
            except websockets.ConnectionClosed:
                raise  # Re-raise to trigger reconnection
            except Exception as e:
                error_count += 1
                if error_count >= max_errors:
                    print("\nToo many consecutive errors, triggering reconnection...")
                    raise
                print(f"Error receiving message (attempt {error_count}/{max_errors}): {e}")
                await asyncio.sleep(0.1)
                continue

            if isinstance(msg, bytes):
                try:
                    await audio_output.play_chunk(msg)
                    continue
                except Exception as e:
                    print(f"\n[ERROR] Failed to process TTS chunk: {e}")
                    import traceback
                    print(traceback.format_exc())
                    continue

            # Handle text messages
            if msg == "TTS_ERROR":
                print("\n[ERROR] TTS generation failed")
                continue

            # Check if trigger word appears anywhere in the message
            msg_lower = msg.lower()
            trigger_pos = msg_lower.find(TRIGGER_WORD.lower())

            if trigger_pos != -1:
                # Print transcript only when trigger word is detected
                print(f"\n[TRANSCRIPT] {msg}")
                
                # Extract everything from the trigger word to the end
                trigger_text = msg[trigger_pos:]

                try:
                    await audio_output.start_stream()
                    asyncio.create_task(llm_client.process_trigger(trigger_text, callback=handle_llm_chunk))
                except Exception as e:
                    print(f"\n[ERROR] Failed to process trigger: {e}")
    except websockets.ConnectionClosed:
        print("\nServer connection closed, attempting to reconnect...")
        raise  # Re-raise to trigger reconnection
    except Exception as e:
        print(f"\nError in receive_transcripts: {e}")
        raise  # Re-raise to trigger reconnection

class AsyncThread(threading.Thread):
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
        """Connect to WebSocket server"""
        try:
            # Try to connect
            self.websocket = await websockets.connect(SERVER_URI)
            print(f"Connected to server at {SERVER_URI}.")

            # Wait for authentication response
            try:
                auth_response = await asyncio.wait_for(
                    self.websocket.recv(),
                    timeout=2.0  # 2 second timeout
                )
                if auth_response != "AUTH_OK":
                    raise Exception(f"Authentication failed: {auth_response}")
                
                return True
                
            except Exception as e:
                print(f"Authentication failed: {e}")
                if self.websocket:
                    await self.websocket.close()
                raise

        except Exception as e:
            print(f"Connection failed: {str(e)}")
            return False

    async def handle_server_connection(self):
        """Handle server connection and reconnection"""
        retry_count = 0
        max_retries = CONFIG['client']['retry']['max_attempts']
        delay_seconds = CONFIG['client']['retry']['delay_seconds']

        while retry_count < max_retries and self.running:
            try:
                print(f"\nAttempting to connect to server... (attempt {retry_count + 1}/{max_retries})")

                # Connect to server
                if not await self.connect_to_server():
                    retry_count += 1
                    if retry_count < max_retries:
                        print(f"\nRetrying connection in {delay_seconds} seconds...")
                        await asyncio.sleep(delay_seconds)
                    continue

                # Send calibrated noise floor
                try:
                    print(f"\nSending calibrated noise floor to server: {self.audio_core.noise_floor:.1f} dB")
                    await self.websocket.send(f"NOISE_FLOOR:{self.audio_core.noise_floor}:{CLIENT_ID}")

                    response = await asyncio.wait_for(
                        self.websocket.recv(),
                        timeout=5.0
                    )
                    if response != "READY":
                        print(f"\nUnexpected server response: {response}")
                        continue
                    print("\nServer ready to receive audio")

                    # Create tasks
                    self.tasks = [
                        asyncio.create_task(record_and_send_audio(self.websocket, self.audio_interface, self.audio_core)),
                        asyncio.create_task(receive_transcripts(self.websocket, self.audio_interface))
                    ]
                    
                    # Wait for any task to complete (which likely means a disconnection)
                    done, pending = await asyncio.wait(
                        self.tasks,
                        return_when=asyncio.FIRST_COMPLETED
                    )

                    # Cancel remaining tasks
                    for task in pending:
                        task.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)

                    # Check if we should retry
                    if not self.running:
                        return
                    
                    print("\nServer connection lost, attempting to reconnect...")
                    retry_count += 1
                    if retry_count < max_retries:
                        print(f"Retrying in {delay_seconds} seconds...")
                        await asyncio.sleep(delay_seconds)
                    
                except asyncio.TimeoutError:
                    print("\nTimeout waiting for server ready response")
                    continue
                except websockets.ConnectionClosed:
                    print("\nServer connection closed")
                    continue
                except Exception as e:
                    print(f"\nError in server communication: {e}")
                    continue

            except Exception as e:
                print(f"\nConnection error: {e}")
                retry_count += 1
                if retry_count < max_retries:
                    print(f"Retrying in {delay_seconds} seconds...")
                    await asyncio.sleep(delay_seconds)

        if retry_count >= max_retries:
            print(f"\nFailed to maintain server connection after {max_retries} attempts. Shutting down...")
            self.running = False
            # Trigger shutdown if in GUI mode
            if self.audio_interface and self.audio_interface.has_gui:
                self.audio_interface.root.after(0, self.audio_interface._on_closing)

    async def initialize_audio(self):
        """Initialize audio devices"""
        try:
            # Initialize audio output
            await audio_output.initialize()

            # Initialize audio core (already passed from main thread)
            stream, device_info, rate, needs_resampling = self.audio_core.init_audio_device()

            # Check if initialization succeeded
            if None in (stream, device_info, rate, needs_resampling):
                print("\nError: Failed to initialize audio device")
                return False

            # Update audio interface with device name
            if self.audio_interface and self.audio_interface.has_gui:
                self.audio_interface.input_device_queue.put(device_info['name'])

            # Double check noise floor calibration (still used for server message)
            if self.audio_core.noise_floor is None:
                print("\nError: Failed to get initial noise floor calibration")
                return False

            # Verify audio levels
            if None in (self.audio_core.rms_level, self.audio_core.peak_level,
                        self.audio_core.min_floor, self.audio_core.max_floor):
                print("\nError: Audio levels not properly initialized")
                return False

            print(f"\nNoise floor calibrated to: {self.audio_core.noise_floor:.1f} dB")
            return True

        except Exception as e:
            print(f"Error initializing audio: {e}")
            return False

    async def async_main(self):
        """Main async loop"""
        try:
            # Initialize audio once at startup
            if not await self.initialize_audio():
                return

            # Handle server connection and reconnection
            await self.handle_server_connection()

        except Exception as e:
            print(f"Error in async main: {e}")
        finally:
            await self.cleanup()

    async def cleanup(self):
        """Clean up resources"""
        try:
            # Clean up audio resources
            if audio_output and audio_output.stream:
                audio_output.close()
            if self.audio_interface and hasattr(self.audio_interface, 'audio_core'):
                # If the new graphical_interface doesn't store it, skip
                pass
            if self.audio_core:
                self.audio_core.close()

            # Close websocket
            if self.websocket:
                await self.websocket.close()

            # Cancel tasks
            for task in self.tasks:
                if not task.done():
                    task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        except Exception as e:
            print(f"Error during cleanup: {e}")

    def stop(self):
        """Stop the async thread gracefully"""
        self.running = False
        if self.loop and not self.loop.is_closed():
            async def shutdown():
                await self.cleanup()
            
            future = asyncio.run_coroutine_threadsafe(shutdown(), self.loop)
            try:
                future.result(timeout=5)  # Wait up to 5 seconds for cleanup
            except Exception as e:
                print(f"Error during shutdown: {e}")

    def run(self):
        """Run the async event loop in this thread"""
        try:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self.async_main())
        except asyncio.CancelledError:
            print("\nAsync operations cancelled")
        except Exception as e:
            print(f"Error in async thread: {e}")
        finally:
            try:
                pending = asyncio.all_tasks(self.loop)
                for task in pending:
                    task.cancel()
                self.loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                
                if self.loop and not self.loop.is_closed():
                    self.loop.stop()
                    self.loop.close()
            except Exception as e:
                print(f"Error closing event loop: {e}")

def run_client():
    """Run the client"""
    async_thread = None
    audio_interface = None
    audio_core = None
    
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Audio Chat Client')
        parser.add_argument('--no-gui', action='store_true',
                          help='Run in headless mode without GUI')
        args = parser.parse_args()

        use_gui = not args.no_gui and gui_available

        # Set up macOS specific configurations if using GUI
        if use_gui and platform.system() == 'Darwin':
            os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
            os.environ['TK_SILENCE_DEPRECATION'] = '1'

        # Initialize audio output
        audio_output.initialize_sync()

        # Create audio core in main thread
        audio_core = AudioCore()

        # Create appropriate interface based on mode
        interface_params = {
            'input_device_name': "Initializing...",
            'output_device_name': audio_output.get_device_name(),
            'on_input_change': None,
            'on_output_change': audio_output.set_device_by_name
        }

        if use_gui:
            audio_interface = GraphicalInterface(**interface_params)
        else:
            audio_interface = HeadlessAudioInterface(**interface_params)
            print("\nRunning in headless mode")

        # Create and start async thread
        async_thread = AsyncThread(audio_interface, audio_core)
        async_thread.start()

        if use_gui:
            # Set up window close handler for GUI mode
            def on_window_close():
                print("\nWindow close detected, initiating shutdown...")
                if async_thread:
                    async_thread.stop()  # Stop async operations first
                if audio_core:
                    audio_core.close()  # Clean up audio resources
                if audio_output:
                    audio_output.close()  # Clean up audio output
                audio_interface.close()  # Then close the GUI
                
            # Set the window close handler
            audio_interface.root.protocol("WM_DELETE_WINDOW", on_window_close)
            
            # Run tkinter mainloop in main thread
            audio_interface.root.mainloop()
        else:
            # For headless mode, just wait for the async thread
            try:
                while async_thread.is_alive():
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\nShutting down...")
                raise

    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        if async_thread:
            async_thread.stop()  # Gracefully stop async operations
        if audio_interface and hasattr(audio_interface, 'close'):
            audio_interface.close()  # Close GUI if it exists
        if audio_core:
            audio_core.close()  # Clean up audio resources
        if audio_output:
            audio_output.close()  # Clean up audio output
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        # Ensure thread is stopped
        if async_thread:
            async_thread.join(timeout=5)  # Wait up to 5 seconds for thread to finish

if __name__ == "__main__":
    try:
        run_client()
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"\nError: {e}")
        if "--no-gui" in sys.argv:
            # In headless mode, show the full traceback
            import traceback
            print("\nTraceback:")
            print(traceback.format_exc())
