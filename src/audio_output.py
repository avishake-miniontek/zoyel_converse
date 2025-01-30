import sounddevice as sd
import numpy as np
import asyncio
import platform
import struct
import time
import threading
from collections import deque
from scipy import signal

# Frame format:
#   Magic bytes (4 bytes): 0x4D495241 ("MIRA")
#   Frame type (1 byte):
#       0x01 = Audio data
#       0x02 = End of utterance
#   Frame length (4 bytes, big-endian): length of the payload in bytes
#   Payload: Audio data (for type=0x01) or empty (for type=0x02)


class AudioOutput:
    MAGIC_BYTES = b'MIRA'
    FRAME_TYPE_AUDIO = 0x01
    FRAME_TYPE_END = 0x02
    HEADER_SIZE = 9  # 4 bytes magic + 1 byte type + 4 bytes length

    def __init__(self):
        self.input_rate = 24000  # TTS output rate in Hz
        self.device_rate = None  # Will detect from the actual device
        self.stream = None
        self.audio_queue = None  # Will be initialized as asyncio.Queue
        self.playing = False
        self.playback_task = None
        self.current_device = None
        self.resampler = None
        self.current_utterance = []  # Buffer for current TTS utterance
        self.partial_frame = b''  # Buffer for incomplete frames
        self.loop = None  # Will store the event loop
        print("Audio output initialized")

    def initialize_sync(self):
        """Initialize audio output synchronously."""
        if self.stream and self.stream.active:
            return

        # Create a new event loop for synchronous initialization
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self.loop = loop
        
        try:
            device_idx, device_info = self._find_output_device()
            self.device_rate = int(device_info['default_samplerate'])
            print(f"[AUDIO] Device rate: {self.device_rate} Hz, Input rate: {self.input_rate} Hz")
            
            # Run async initialization in the event loop
            loop.run_until_complete(self._open_stream(device_idx))
            self._init_resampler()
            
        finally:
            loop.close()
            self.loop = None

    async def initialize(self):
        """Initialize audio output asynchronously."""
        if self.stream and self.stream.active:
            return

        device_idx, device_info = self._find_output_device()
        self.device_rate = int(device_info['default_samplerate'])
        print(f"[AUDIO] Device rate: {self.device_rate} Hz, Input rate: {self.input_rate} Hz")
        await self._open_stream(device_idx)
        self._init_resampler()

    async def _play_audio_loop(self):
        """Async audio playback loop."""
        print("[AUDIO] Playback loop started")
        last_write_time = time.time()
        
        while self.playing:
            try:
                # Wait for data with a timeout
                try:
                    audio_data = await asyncio.wait_for(self.audio_queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue
                
                if audio_data is not None and self.stream and self.stream.active:
                    now = time.time()
                    time_since_last = (now - last_write_time) * 1000
                    try:
                        # Use run_in_executor for non-blocking stream.write
                        await self.loop.run_in_executor(None, self.stream.write, audio_data)
                        last_write_time = time.time()
                    except Exception as e:
                        print(f"[AUDIO] Error writing to stream: {e}")
                        import traceback
                        print(traceback.format_exc())
                else:
                    if audio_data is None:
                        print("[AUDIO] Skipping None audio data")
                    if not self.stream or not self.stream.active:
                        print("[AUDIO] Stream not active")
                        # Don't try to recover here - let start_stream handle it
                        await asyncio.sleep(0.1)

            except Exception as e:
                print(f"[AUDIO] Error in playback loop: {e}")
                import traceback
                print(traceback.format_exc())
                await asyncio.sleep(0.1)

        print("[AUDIO] Playback loop stopping")

    async def start_stream(self):
        """Start the audio stream and playback loop."""
        try:
            # Stop any existing playback task
            if self.playing:
                self.playing = False
                if self.playback_task:
                    try:
                        await self.playback_task
                    except asyncio.CancelledError:
                        pass

            # Clear existing queue if any
            if self.audio_queue:
                while not self.audio_queue.empty():
                    try:
                        self.audio_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break

            # Initialize stream if not already initialized
            if not self.stream or not self.stream.active:
                await self.initialize()

            # Create new queue and start playback
            self.loop = asyncio.get_running_loop()
            self.audio_queue = asyncio.Queue()
            self.playing = True
            self.playback_task = asyncio.create_task(self._play_audio_loop())
            print("Started playback loop")

        except Exception as e:
            print(f"Error starting stream: {e}")
            import traceback
            print(traceback.format_exc())

    async def pause(self):
        """Stop playback and clear the queue."""
        self.playing = False
        if self.playback_task:
            try:
                await self.playback_task
            except asyncio.CancelledError:
                pass
        if self.stream and self.stream.active:
            self.stream.stop()
        if self.audio_queue:
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

    async def close(self):
        """Clean up resources."""
        await self.pause()
        if self.stream:
            self.stream.close()
            self.stream = None

    def get_device_name(self):
        """Get the name of the current output device."""
        if self.current_device:
            return self.current_device['name']
        return "No device selected"

    async def set_device_by_name(self, device_name):
        """Change the output device by name."""
        print(f"\nChanging output device to: {device_name}")
        # Only pause playback, don't close stream yet
        await self.pause()
        
        # Find new device info
        device_idx, device_info = self._find_output_device(device_name)
        new_rate = int(device_info['default_samplerate'])
        
        # Only close and reopen stream if device actually changed
        if (not self.stream or 
            not self.stream.active or 
            self.device_rate != new_rate or 
            self.current_device['name'] != device_name):
            
            if self.stream:
                self.stream.close()
                self.stream = None
                await asyncio.sleep(0.1)  # Give time for cleanup
            
            self.device_rate = new_rate
            await self._open_stream(device_idx)
            self._init_resampler()  # Initialize resampler for new device rate
        else:
            print("Device unchanged, keeping existing stream")

    async def play_chunk(self, chunk):
        """Queue a chunk of audio data for playback."""
        try:
            # Append new data to any partial frame from previous chunk
            data = self.partial_frame + chunk
            self.partial_frame = b''
            
            while data:
                frame, data = self._parse_frame(data)
                if frame is None:
                    # Store remaining data for next chunk
                    self.partial_frame = data
                    break
                
                frame_type, frame_data = frame
                if frame_type == self.FRAME_TYPE_AUDIO:
                    self.current_utterance.append(frame_data)
                elif frame_type == self.FRAME_TYPE_END:
                    self._process_complete_utterance()
            
        except Exception as e:
            print(f"[AUDIO] Error in play_chunk: {e}")
            import traceback
            print(traceback.format_exc())

    def _parse_frame(self, data):
        """Parse a frame from the data buffer. Returns (frame, remaining_data)."""
        if len(data) < self.HEADER_SIZE:
            return None, data

        # Check magic bytes
        if data[:4] != self.MAGIC_BYTES:
            # Try to find next magic bytes
            next_magic = data[4:].find(self.MAGIC_BYTES)
            if next_magic == -1:
                return None, b''  # Discard invalid data
            data = data[next_magic:]
            if len(data) < self.HEADER_SIZE:
                return None, data

        frame_type = data[4]
        frame_length = struct.unpack('>I', data[5:9])[0]
        total_length = self.HEADER_SIZE + frame_length

        if len(data) < total_length:
            return None, data  # Need more data

        frame_data = data[self.HEADER_SIZE:total_length]
        remaining_data = data[total_length:]
        return (frame_type, frame_data), remaining_data

    def _process_complete_utterance(self):
        """Process and queue the complete utterance."""
        if not self.current_utterance:
            return

        try:
            # Concatenate the raw audio data
            complete_chunk = b''.join(self.current_utterance)
            
            # Process the complete utterance
            audio_data = self._process_audio_data(complete_chunk)
            if audio_data is not None and self.audio_queue is not None:
                asyncio.create_task(self.audio_queue.put(audio_data))
        except Exception as e:
            print(f"[AUDIO] Error processing complete utterance: {e}")
            import traceback
            print(traceback.format_exc())
        finally:
            self.current_utterance = []

    def _process_audio_data(self, audio_bytes):
        """Process raw audio bytes into playable audio data."""
        try:
            process_start = time.time()
            # Convert to float32 stereo
            audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            if audio_data.size == 0:
                return None

            # Convert to stereo
            audio_data = np.column_stack((audio_data, audio_data))

            # Resample if needed
            if self.device_rate != self.input_rate:
                ratio = self.device_rate / self.input_rate
                self._init_resampler()
                audio_data = self.resampler.process(audio_data, ratio)

            return audio_data
        except Exception as e:
            print(f"[AUDIO] Error processing audio data: {e}")
            import traceback
            print(traceback.format_exc())
            return None

    def _init_resampler(self):
        """Initialize and warm up the resampler."""
        if self.resampler is None and self.device_rate is not None and self.device_rate != self.input_rate:
            import samplerate
            self.resampler = samplerate.Resampler('sinc_best', channels=2)
            # Generate a proper warmup signal (1 second of shaped noise)
            samples = int(self.input_rate)  # 1 second worth of samples
            t = np.linspace(0, 1, samples)
            # Create a sweep from 20Hz to 20kHz to properly initialize filter states
            warmup_signal = np.sin(2 * np.pi * np.logspace(1.3, 4.3, samples) * t)
            warmup_data = np.column_stack((warmup_signal, warmup_signal)).astype(np.float32)
            ratio = self.device_rate / self.input_rate
            # Process the warmup data through the resampler
            self.resampler.process(warmup_data, ratio)
            print("[AUDIO] Resampler warmed up with frequency sweep")

    def _find_output_device(self, device_name=None):
        """Find a suitable audio output device across platforms."""
        print("\nAvailable output devices:")
        devices = sd.query_devices()
        output_devices = []

        for i, dev in enumerate(devices):
            if dev['max_output_channels'] > 0:
                print(f"Device {i}: {dev['name']} "
                      f"(outputs={dev['max_output_channels']}, "
                      f"rate={dev['default_samplerate']:.0f}Hz)")
                output_devices.append((i, dev))

        system = platform.system()

        try:
            # If a device name was specified
            if device_name:
                for i, dev in output_devices:
                    if dev['name'] == device_name:
                        print(f"\n[AUDIO] Selected output device: {dev['name']}")
                        self.current_device = dev
                        return i, dev

            # Otherwise pick best device by platform
            if system == 'Linux':
                # Prefer PipeWire
                for i, dev in output_devices:
                    if 'pipewire' in dev['name'].lower():
                        print(f"\n[AUDIO] Selected PipeWire device: {dev['name']}")
                        self.current_device = dev
                        return i, dev

            # Next, try system default
            default_idx = sd.default.device[1]  # index of default output
            if default_idx is not None and 0 <= default_idx < len(devices):
                device_info = devices[default_idx]
                print(f"\n[AUDIO] Selected default device: {device_info['name']}")
                self.current_device = device_info
                return default_idx, device_info

            # If nothing else, pick the first available device
            if output_devices:
                idx, dev = output_devices[0]
                print(f"\n[AUDIO] Selected first available device: {dev['name']}")
                self.current_device = dev
                return idx, dev

            raise RuntimeError("No output devices found")

        except Exception as e:
            print(f"[AUDIO] Error finding output device: {e}")
            raise

    async def _warmup_stream(self):
        """Play a short silent buffer to properly initialize the audio system."""
        if not self.stream or not self.stream.active:
            return

        try:
            # Create 500ms of silence (increased for better initialization)
            duration = 0.5  # seconds
            num_samples = int(self.device_rate * duration)
            silence = np.zeros((num_samples, 2), dtype=np.float32)
            
            # Write the silent buffer using run_in_executor and wait for it to complete
            if self.loop is None:
                self.loop = asyncio.get_running_loop()
            await self.loop.run_in_executor(None, self.stream.write, silence)
            await asyncio.sleep(duration)  # Ensure warmup completes before proceeding
            
            # Write another shorter buffer to ensure stability
            short_duration = 0.1
            short_samples = int(self.device_rate * short_duration)
            short_silence = np.zeros((short_samples, 2), dtype=np.float32)
            await self.loop.run_in_executor(None, self.stream.write, short_silence)
            await asyncio.sleep(short_duration)
            
            # Clear any existing audio in queue
            if self.audio_queue:
                while not self.audio_queue.empty():
                    try:
                        self.audio_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
            
            print("[AUDIO] Stream warmed up with silent buffer")
            
        except Exception as e:
            print(f"[AUDIO] Warmup error (non-fatal): {e}")

    async def _open_stream(self, device_idx):
        """Open and start the audio output stream."""
        try:
            stream_kwargs = {
                'device': device_idx,
                'samplerate': self.device_rate,
                'channels': 2,
                'dtype': np.float32,
                'latency': 'high',  # Use high latency for more stable playback
                'callback': None,   # No callback needed, we use write mode
                'finished_callback': None
            }

            print(f"Opening stream with settings: {stream_kwargs}")
            self.stream = sd.OutputStream(**stream_kwargs)
            self.stream.start()
            # Give extra time for Windows to initialize
            if platform.system() == 'Windows':
                await asyncio.sleep(0.5)  # Longer delay for Windows
            else:
                await asyncio.sleep(0.2)
            print(f"Stream started successfully")
            
            # Warmup the stream
            await self._warmup_stream()

        except Exception as e:
            print(f"Failed to open stream: {e}")
            import traceback
            print(traceback.format_exc())
            if self.stream:
                self.stream.close()
                self.stream = None
            raise RuntimeError(f"Failed to open stream: {e}")
