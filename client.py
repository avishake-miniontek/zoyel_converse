#!/usr/bin/env python3
"""
Audio Chat Client

This script captures audio from your microphone, sends it to a server for
real-time transcription, and displays the transcribed text. It also receives
TTS audio data from the server, which it plays via AudioOutput.

Usage:
    python client.py [--no-gui]
"""

import sys
import asyncio
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
import websockets
import numpy as np
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
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
from src.audio_output import AudioOutput

# Load configuration
with open('config.json', 'r') as f:
    CONFIG = json.load(f)

# Server configuration
API_KEY = CONFIG['server']['websocket']['api_key']
SERVER_HOST = CONFIG['server']['websocket']['host']
SERVER_PORT = CONFIG['server']['websocket']['port']

# Generate unique client ID and build the server URI.
CLIENT_ID = uuid.uuid4()
SERVER_URI = f"ws://{SERVER_HOST}:{SERVER_PORT}?api_key={API_KEY}&client_id={CLIENT_ID}"

# Trigger word configuration
TRIGGER_WORD = CONFIG['assistant']['name']

################################################################################
# ASYNCHRONOUS TASKS
################################################################################

class MessageHandler:
    def __init__(self, audio_output, audio_interface, llm_client):
        self.text_queue = asyncio.Queue()
        self.audio_queue = asyncio.Queue()
        self.audio_output = audio_output
        self.audio_interface = audio_interface
        self.llm_client = llm_client
        self.websocket = None

    async def handle_llm_chunk(self, text):
        """Send LLM-generated text to the server as a TTS request."""
        if self.websocket:
            await self.websocket.send(f"TTS:{text}")

    async def process_text(self, text: str):
        """Process text input with the LLM."""
        try:
            if not text.lower().startswith(TRIGGER_WORD.lower()):
                text = f"{TRIGGER_WORD}, {text}"
            await self.audio_output.start_stream()
            await self.llm_client.process_trigger(text, callback=self.handle_llm_chunk)
        except Exception as e:
            logger.error("Failed to process text input: %s", e, exc_info=True)

async def websocket_receiver(websocket, handler):
    """Dedicated task for receiving from WebSocket and distributing messages"""
    handler.websocket = websocket
    error_count = 0
    max_errors = 3

    while True:
        try:
            msg = await websocket.recv()
            error_count = 0
            
            if isinstance(msg, bytes):
                await handler.audio_queue.put(msg)
            else:
                await handler.text_queue.put(msg)
                
        except websockets.ConnectionClosed:
            logger.info("WebSocket connection closed")
            raise
        except Exception as e:
            error_count += 1
            if error_count >= max_errors:
                logger.error("Too many consecutive errors in receiver, reconnecting...")
                raise
            logger.warning("Error in receiver: %s", e)
            await asyncio.sleep(0.1)

async def process_audio_messages(handler):
    """Process audio frames"""
    while True:
        try:
            msg = await handler.audio_queue.get()
            if len(msg) >= 9:  # Minimum frame size (4 magic + 1 type + 4 length)
                magic = msg[:4]
                frame_type = msg[4]
                if magic == b'MIRA':
                    if frame_type == 0x03:
                        if handler.audio_interface and handler.audio_interface.has_gui:
                            is_speech = bool(msg[9])
                            handler.audio_interface.process_vad(is_speech)
                    else:
                        await handler.audio_output.play_chunk(msg)
        except Exception as e:
            logger.error("Failed to process audio frame: %s", e)
            await asyncio.sleep(0.001)

async def process_text_messages(handler):
    """Process text messages"""
    while True:
        try:
            msg = await handler.text_queue.get()
            
            if msg == "TTS_ERROR":
                logger.error("Server TTS generation failed.")
                continue

            msg_lower = msg.lower()
            trigger_pos = msg_lower.find(TRIGGER_WORD.lower())
            if trigger_pos != -1:
                print(f"\n[TRANSCRIPT] {msg}")
                trigger_text = msg[trigger_pos:]
                await handler.audio_output.start_stream()
                asyncio.create_task(handler.process_text(trigger_text))
                
        except Exception as e:
            logger.error("Failed to process text message: %s", e)
            await asyncio.sleep(0.001)

async def check_gui_input(handler):
    """Fallback: Check for GUI text input (used only in headless mode)"""
    while True:
        try:
            if handler.audio_interface and not handler.audio_interface.has_gui:
                text_input = handler.audio_interface.get_text_input()
                if text_input is not None:
                    asyncio.create_task(handler.process_text(text_input))
            await asyncio.sleep(0.1)
        except Exception as e:
            logger.error("Failed to check GUI input: %s", e)
            await asyncio.sleep(0.1)

async def record_and_send_audio(websocket, audio_interface, audio_core):
    """
    Continuously read audio from the microphone and send raw PCM frames to the server.
    Audio is resampled to 16kHz if needed.
    """
    error_count = 0
    max_errors = 3

    try:
        if not audio_core or not audio_core.stream:
            logger.error("Error: Audio core not properly initialized")
            return

        stream = audio_core.stream
        print("\nStart speaking...")

        loop = asyncio.get_running_loop()

        while True:
            try:
                # Read a chunk of audio from the microphone using an executor to avoid blocking.
                try:
                    result = await loop.run_in_executor(None, stream.read, audio_core.CHUNK)
                    audio_data = result[0]
                except Exception as e:
                    if isinstance(e, sd.PortAudioError) and "Invalid stream pointer" in str(e):
                        logger.error("Fatal stream error: %s", e)
                        raise
                    logger.warning("Stream read error: %s", e)
                    await asyncio.sleep(0.1)
                    continue

                # Process audio through audio_core which handles mono conversion and resampling
                processed = audio_core.process_audio(audio_data)
                audio_data = processed['audio']

                # Scale to int16.
                final_data = np.clip(audio_data * 32767.0, -32767, 32767).astype(np.int16)
                
                # Send the audio chunk to the server.
                try:
                    await websocket.send(final_data.tobytes())
                    error_count = 0
                except websockets.ConnectionClosed:
                    raise
                except Exception as e:
                    error_count += 1
                    if error_count >= max_errors:
                        logger.error("Too many consecutive errors, triggering reconnection...")
                        raise
                    logger.warning("Error sending audio chunk (attempt %d/%d): %s", error_count, max_errors, e)
                    await asyncio.sleep(0.1)
                    continue

            except websockets.ConnectionClosed:
                raise
            except Exception as e:
                error_count += 1
                if error_count >= max_errors:
                    logger.error("Too many consecutive errors, triggering reconnection...")
                    raise
                logger.warning("Error processing audio chunk: %s", e)
                await asyncio.sleep(0.1)
                continue

            await asyncio.sleep(0.001)

    except asyncio.CancelledError:
        logger.info("Audio recording cancelled.")
    except Exception as e:
        logger.error("Error in record_and_send_audio: %s", e)
        raise
    finally:
        # The audio stream is managed by audio_core; do not close here.
        pass

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
        self.handler = None  # Will hold the MessageHandler instance
        self.daemon = True

    async def connect_to_server(self):
        """Connect to the WebSocket server and wait for authentication."""
        try:
            self.websocket = await websockets.connect(SERVER_URI)
            logger.info(f"Connected to {SERVER_URI}")

            # Wait for server auth
            auth_response = await asyncio.wait_for(self.websocket.recv(), timeout=2.0)
            if auth_response != "AUTH_OK":
                raise Exception(f"Auth failed: {auth_response}")
            logger.info("[CLIENT] Server ready.")
            return True
        except Exception as e:
            logger.error("connect_to_server: %s", e)
            if self.websocket:
                await self.websocket.close()
            return False

    async def handle_server_connection(self):
        """
        Attempt to connect and, once connected, start the tasks for sending and receiving audio.
        """
        retry_count = 0
        max_retries = CONFIG['client']['retry']['max_attempts']
        delay_seconds = CONFIG['client']['retry']['delay_seconds']

        while retry_count < max_retries and self.running:
            logger.info(f"\nAttempting to connect... (attempt {retry_count+1}/{max_retries})")
            if not await self.connect_to_server():
                retry_count += 1
                if retry_count < max_retries:
                    logger.info(f"Retry in {delay_seconds} seconds...")
                    await asyncio.sleep(delay_seconds)
                continue

            # Initialize message handler
            handler = MessageHandler(audio_output, self.audio_interface, LLMClient())
            self.handler = handler  # Store for GUI callbacks

            tasks = [
                asyncio.create_task(record_and_send_audio(self.websocket, self.audio_interface, self.audio_core)),
                asyncio.create_task(websocket_receiver(self.websocket, handler)),
                asyncio.create_task(process_audio_messages(handler)),
                asyncio.create_task(process_text_messages(handler))
            ]

            # In headless mode, add the polling for GUI input; in GUI mode, use callback
            if not self.audio_interface.has_gui:
                tasks.append(asyncio.create_task(check_gui_input(handler)))
            else:
                # Register the GUI callback to process text input
                def gui_text_callback(text):
                    if self.loop is not None and self.handler is not None:
                        asyncio.run_coroutine_threadsafe(self.handler.process_text(text), self.loop)
                    else:
                        logger.warning("GUI text callback invoked but async loop or handler not ready.")
                try:
                    self.audio_interface.set_text_callback(gui_text_callback)
                except AttributeError:
                    # Fallback if the GUI interface doesn't have set_text_callback
                    self.audio_interface.on_input_change = gui_text_callback

            self.tasks = tasks

            try:
                # Use asyncio.gather to run tasks concurrently
                await asyncio.gather(*self.tasks)
            except Exception as e:
                logger.error("[CLIENT] One of the tasks raised an exception: %s", e)
            finally:
                # Cancel all tasks
                for task in self.tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*self.tasks, return_exceptions=True)

            if not self.running:
                return

            logger.info("[CLIENT] Connection lost, will retry...")
            retry_count += 1
            if retry_count < max_retries:
                logger.info(f"Retrying in {delay_seconds} seconds...")
                await asyncio.sleep(delay_seconds)

    async def initialize_audio(self):
        """Initialize microphone and speaker."""
        try:
            await audio_output.initialize()
            stream, device_info, rate, needs_resampling = self.audio_core.init_audio_device()
            if None in (stream, device_info, rate, needs_resampling):
                logger.error("[CLIENT] Error initializing microphone.")
                return False

            if self.audio_interface and self.audio_interface.has_gui:
                self.audio_interface.input_device_queue.put(device_info['name'])

            logger.info(f"[CLIENT] Audio initialized. Device: {device_info['name']}")
            return True
        except Exception as e:
            logger.error("[CLIENT] Audio init error: %s", e)
            return False

    async def async_main(self):
        """Main async loop for the client."""
        try:
            if not await self.initialize_audio():
                return
            await self.handle_server_connection()
        except Exception as e:
            logger.error("[CLIENT] Error in async main: %s", e)
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
            logger.error("[CLIENT] Error during cleanup: %s", e)

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
                logger.error("[CLIENT] Error stopping: %s", e)

    def run(self):
        """Thread run: create an event loop and run async_main."""
        try:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self.async_main())
        except asyncio.CancelledError:
            logger.info("[CLIENT] Async ops cancelled.")
        except Exception as e:
            logger.error("[CLIENT] Exception in async thread: %s", e)
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
    spawns the AsyncThread, and runs until termination.
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
                logger.info("[CLIENT] Window close -> stopping.")
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
        logger.error("[CLIENT] Error: %s", e, exc_info=True)
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
