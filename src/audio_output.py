import sounddevice as sd
import numpy as np
import threading
import time
import platform
import struct
from collections import deque
from scipy import signal

# Frame format:
# Magic bytes (4 bytes): 0x4D495241 ("MIRA")
# Frame type (1 byte): 
#   0x01 = Audio data
#   0x02 = End of utterance
# Frame length (4 bytes): Length of payload in bytes
# Payload: Audio data (for type 0x01) or empty (for type 0x02)

class AudioOutput:
    MAGIC_BYTES = b'MIRA'
    FRAME_TYPE_AUDIO = 0x01
    FRAME_TYPE_END = 0x02
    HEADER_SIZE = 9  # 4 bytes magic + 1 byte type + 4 bytes length

    def __init__(self):
        self.input_rate = 24000  # TTS output rate in Hz
        self.device_rate = None  # Will detect from the actual device
        self.stream = None
        self.audio_queue = deque()
        self.playing = False
        self.play_thread = None
        self.current_device = None
        self.resampler = None
        self.current_utterance = []  # Buffer for current TTS utterance
        self.partial_frame = b''  # Buffer for incomplete frames
        print("Audio output initialized")

    def _init_resampler(self):
        """Initialize and warm up the resampler."""
        if self.resampler is None and self.device_rate is not None and self.device_rate != self.input_rate:
            import samplerate
            self.resampler = samplerate.Resampler('sinc_best', channels=2)
            # Warm up the resampler with a small buffer
            warmup_data = np.zeros((512, 2), dtype=np.float32)
            ratio = self.device_rate / self.input_rate
            self.resampler.process(warmup_data, ratio)

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
            # Try specified device first
            if device_name:
                for i, dev in output_devices:
                    if dev['name'] == device_name:
                        print(f"\nSelected output device: {dev['name']}")
                        self.current_device = dev
                        return i, dev

            # Linux-specific detection
            if system == 'Linux':
                # Try PipeWire first
                for i, dev in output_devices:
                    if 'pipewire' in dev['name'].lower():
                        print(f"\nSelected PipeWire device: {dev['name']}")
                        self.current_device = dev
                        return i, dev

                # Then try system default
                default_idx = sd.default.device[1]
                if default_idx is not None and 0 <= default_idx < len(devices):
                    device_info = devices[default_idx]
                    print(f"\nSelected default device: {device_info['name']}")
                    self.current_device = device_info
                    return default_idx, device_info

            # If no device found yet, use first available
            if output_devices:
                idx, dev = output_devices[0]
                print(f"\nSelected first available device: {dev['name']}")
                self.current_device = dev
                return idx, dev

            raise RuntimeError("No output devices found")
        except Exception as e:
            print(f"Error finding output device: {e}")
            raise

    def _open_stream(self, device_idx):
        """Open and start the audio output stream."""
        try:
            # Close any existing stream first
            if self.stream:
                self.stream.close()
                self.stream = None
                time.sleep(0.1)  # Give time for cleanup

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
            time.sleep(0.1)  # Give stream time to fully initialize
            print(f"Stream started successfully")

        except Exception as e:
            if self.stream:
                self.stream.close()
                self.stream = None
            raise RuntimeError(f"Failed to open stream: {e}")

    def initialize_sync(self):
        """Initialize audio output synchronously."""
        if self.stream and self.stream.active:
            return

        device_idx, device_info = self._find_output_device()
        self.device_rate = int(device_info['default_samplerate'])
        print(f"[AUDIO] Device rate: {self.device_rate} Hz, Input rate: {self.input_rate} Hz")
        self._open_stream(device_idx)
        self._init_resampler()

    async def initialize(self):
        """Initialize audio output asynchronously."""
        if self.stream and self.stream.active:
            return

        device_idx, device_info = self._find_output_device()
        self.device_rate = int(device_info['default_samplerate'])
        print(f"[AUDIO] Device rate: {self.device_rate} Hz, Input rate: {self.input_rate} Hz")
        self._open_stream(device_idx)
        self._init_resampler()

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

    def _play_audio_thread(self):
        """Audio playback thread."""
        print("[AUDIO] Playback thread started")
        last_write_time = time.time()
        
        while self.playing:
            try:
                if self.audio_queue:
                    audio_data = self.audio_queue.popleft()
                    if audio_data is not None and self.stream and self.stream.active:
                        now = time.time()
                        time_since_last = (now - last_write_time) * 1000
                        try:
                            self.stream.write(audio_data)
                            last_write_time = time.time()
                        except Exception as e:
                            print(f"[AUDIO] Error writing to stream: {e}")
                            import traceback
                            print(traceback.format_exc())
                    else:
                        if audio_data is None:
                            print("[AUDIO] Skipping None audio data")
                        if not self.stream or not self.stream.active:
                            print("[AUDIO] Stream not active, attempting recovery")
                            self._open_stream(self.current_device['index'])
                            if self.stream and self.stream.active and audio_data is not None:
                                self.stream.write(audio_data)
                else:
                    time.sleep(0.001)

            except Exception as e:
                print(f"[AUDIO] Error in playback loop: {e}")
                import traceback
                print(traceback.format_exc())
                time.sleep(0.1)

        print("[AUDIO] Playback thread stopping")

    def _process_complete_utterance(self):
        """Process and queue the complete utterance."""
        if not self.current_utterance:
            return

        try:
            # Concatenate the raw audio data
            complete_chunk = b''.join(self.current_utterance)
            
            # Process the complete utterance
            audio_data = self._process_audio_data(complete_chunk)
            if audio_data is not None:
                self.audio_queue.append(audio_data)
        except Exception as e:
            print(f"[AUDIO] Error processing complete utterance: {e}")
            import traceback
            print(traceback.format_exc())
        finally:
            self.current_utterance = []

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

    async def start_stream(self):
        """Start the audio stream and playback thread."""
        try:
            # Stop any existing playback
            if self.playing:
                self.playing = False
                if self.play_thread and self.play_thread.is_alive():
                    self.play_thread.join(timeout=1.0)
                self.audio_queue.clear()

            # Initialize fresh stream
            await self.initialize()

            # Start new playback thread
            self.playing = True
            self.play_thread = threading.Thread(target=self._play_audio_thread, daemon=True)
            self.play_thread.start()
            print("Started new playback thread")

        except Exception as e:
            print(f"Error starting stream: {e}")
            # Try to recover
            self.stream = None
            await self.initialize()
            self.playing = True
            self.play_thread = threading.Thread(target=self._play_audio_thread, daemon=True)
            self.play_thread.start()
            print("Recovered stream after error")

    def pause(self):
        """Stop playback and clear the queue."""
        self.playing = False
        if self.play_thread and self.play_thread.is_alive():
            self.play_thread.join(timeout=1.0)
        if self.stream and self.stream.active:
            self.stream.stop()
        self.audio_queue.clear()

    def close(self):
        """Clean up resources."""
        self.pause()
        if self.stream:
            self.stream.close()
            self.stream = None

    def get_device_name(self):
        """Get the name of the current output device."""
        if self.current_device:
            return self.current_device['name']
        return "No device selected"

    def set_device_by_name(self, device_name):
        """Change the output device by name."""
        print(f"\nChanging output device to: {device_name}")
        self.pause()
        if self.stream:
            self.stream.close()
            self.stream = None

        device_idx, device_info = self._find_output_device(device_name)
        self.device_rate = int(device_info['default_samplerate'])
        self._open_stream(device_idx)
        self._init_resampler()  # Initialize resampler for new device rate
