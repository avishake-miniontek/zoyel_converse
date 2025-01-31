import json
import sounddevice as sd
import numpy as np
from collections import deque
import platform
import struct
from scipy import signal
import time
import asyncio

class AudioOutput:
    """
    A simplified, callback-driven audio output class that:
      - Selects an output device, or uses default
      - Handles MIRA-framed TTS audio (int16 mono blocks)
      - Parses frames outside the callback
      - Resamples data to float32 stereo at device_rate
      - Enqueues each final block into a ring buffer (self.audio_queue)
      - The callback drains that queue for playback
    """

    MAGIC_BYTES = b'MIRA'
    FRAME_TYPE_AUDIO = 0x01
    FRAME_TYPE_END = 0x02
    HEADER_SIZE = 9  # 4 magic + 1 frame_type + 4 length

    def __init__(self):
        self.input_rate = 24000           # TTS audio sample rate
        self.device_rate = None           # Will set after device selection
        self.stream = None

        # A queue of final float32 stereo blocks, ready for playback
        # Each element is shape (samples, 2)
        self.audio_queue = deque()

        # For partial frame parsing
        self.partial_frame = b''          # leftover bytes from last chunk
        self.current_utterance = []       # accumulate AUDIO frames until FRAME_TYPE_END

        # Whether we're actively playing audio
        self.playing = False

        # Device info
        self.current_device = None

    ###########################################################################
    # DEVICE SELECTION & INITIALIZATION
    ###########################################################################

    def _find_output_device(self, device_name=None):
        """
        Find an appropriate device. If device_name is specified, look for it by name.
        Otherwise, use the system default or the first output device we find.
        """
        devices = sd.query_devices()
        output_devices = []

        print("\n[AUDIO] Available output devices:")
        for i, dev in enumerate(devices):
            if dev['max_output_channels'] > 0:
                print(f"  Device {i}: {dev['name']} "
                      f"(outputs={dev['max_output_channels']}, "
                      f"rate={dev['default_samplerate']:.0f}Hz)")
                output_devices.append((i, dev))

        # Check if audio_devices is present in config
        with open('config.json', 'r') as f:
            config = json.load(f)
        
        if 'audio_devices' in config and 'output_device' in config['audio_devices']:
            output_device = config['audio_devices']['output_device']
            # Handle both numeric index and name string
            if isinstance(output_device, (int, float)):
                device_idx = int(output_device)
                for i, dev in output_devices:
                    if i == device_idx:
                        print(f"\n[AUDIO] Selected output device: {dev['name']}")
                        self.current_device = dev
                        return i, dev
            else:
                # Try matching by name
                for i, dev in output_devices:
                    if dev['name'] == output_device:
                        print(f"\n[AUDIO] Selected output device: {dev['name']}")
                        self.current_device = dev
                        return i, dev
        
        # If no output_device specified or not found, use existing logic
        if device_name:
            # Look for exact match
            for i, dev in output_devices:
                if dev['name'] == device_name:
                    print(f"\n[AUDIO] Selected output device: {dev['name']}")
                    self.current_device = dev
                    return i, dev

        # Otherwise pick default or the first available
        default_idx = sd.default.device[1]
        if default_idx is not None and 0 <= default_idx < len(devices):
            device_info = devices[default_idx]
            print(f"\n[AUDIO] Selected default device: {device_info['name']}")
            self.current_device = device_info
            return default_idx, device_info

        # Fallback: first valid device
        if output_devices:
            i, dev = output_devices[0]
            print(f"\n[AUDIO] Selected first available device: {dev['name']}")
            self.current_device = dev
            return i, dev

        raise RuntimeError("[AUDIO] No suitable output devices found.")

    def get_device_name(self):
        if self.current_device:
            return self.current_device['name']
        return "No device selected"

    def set_device_by_name(self, device_name):
        """
        Switch to a new device by name, re-initializing the stream if needed.
        """
        print(f"[AUDIO] Switching to device: {device_name}")
        self.pause()
        if self.stream:
            self.stream.close()
            self.stream = None
        self.initialize_sync(device_name=device_name)
        if self.playing:
            self.start_stream()

    def initialize_sync(self, device_name=None):
        """
        Synchronous init if you prefer. Similar to async but blocking.
        """
        if self.stream and self.stream.active:
            return
        try:
            self._do_initialize(device_name=device_name)
        except Exception as e:
            print(f"[AUDIO] Error in initialize_sync: {e}")

    async def initialize(self, device_name=None):
        """
        Async init. If the stream is not active, pick device and open stream.
        """
        if self.stream and self.stream.active:
            return
        self._do_initialize(device_name=device_name)

    async def _open_stream(self, device_idx):
        """
        Internal method to open the audio stream with better error handling. 
        """
        try:
            stream_kwargs = {
                'device': device_idx,
                'samplerate': self.device_rate,
                'channels': 2,
                'dtype': np.float32,
                'latency': 'high',  # higher latency => more stable in many environments
                'callback': self._audio_callback
            }

            print(f"[AUDIO] Opening stream with settings: {stream_kwargs}")
            self.stream = sd.OutputStream(**stream_kwargs)
            self.stream.start()
            await asyncio.sleep(0.2)  # Give the stream a moment to stabilize
            print("[AUDIO] Stream started successfully.")

        except Exception as e:
            print(f"[AUDIO] Failed to open stream: {e}")
            if self.stream:
                self.stream.close()
            self.stream = None
            raise RuntimeError(f"[AUDIO] Could not open output stream: {e}")

    def _do_initialize(self, device_name=None):
        """
        The actual device init logic (shared by sync and async methods).
        """
        device_idx, dev_info = self._find_output_device(device_name=device_name)
        self.device_rate = int(dev_info['default_samplerate'])
        
        # Create an event loop for the async _open_stream call
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self._open_stream(device_idx))
            print(f"[AUDIO] Stream started on device: {dev_info['name']} at {self.device_rate} Hz")
        except Exception as e:
            print(f"[AUDIO] Error in _do_initialize: {e}")
            raise
        finally:
            loop.close()

    ###########################################################################
    # CALLBACK & QUEUE
    ###########################################################################

    def _audio_callback(self, outdata, frames, time_info, status):
        """
        sounddevice callback. Fill 'outdata' with up to 'frames' samples
        from self.audio_queue. If empty, fill with silence and log underflow.
        """
        if status and status.output_underflow:
            print("[AUDIO] Output underflow (no data).")

        outdata.fill(0.0)
        needed = frames
        offset = 0

        while needed > 0 and self.audio_queue:
            chunk = self.audio_queue[0]
            chunk_len = len(chunk)
            if chunk_len <= needed:
                # Copy the entire chunk
                outdata[offset : offset + chunk_len] = chunk
                offset += chunk_len
                needed -= chunk_len
                self.audio_queue.popleft()
            else:
                # Copy part of the chunk
                portion = chunk[:needed]
                outdata[offset : offset + needed] = portion
                self.audio_queue[0] = chunk[needed:]
                needed = 0

    ###########################################################################
    # PLAY CONTROL
    ###########################################################################

    async def start_stream(self):
        """
        Mark ourselves as playing. If not initialized, do so. 
        The callback automatically starts pulling data.
        """
        self.playing = True
        if not self.stream or not self.stream.active:
            await self.initialize()

    def pause(self):
        """
        Stop playback, flush the queue, but keep the stream open if you want to resume.
        """
        self.playing = False
        self.audio_queue.clear()
        if self.stream and self.stream.active:
            self.stream.stop()
            print("[AUDIO] Playback paused.")

    async def close(self):
        """
        Fully close the audio stream and reset everything.
        """
        print("[AUDIO] Closing audio output...")
        self.pause()
        if self.stream:
            self.stream.close()
            self.stream = None
        self.partial_frame = b''
        self.current_utterance.clear()
        self.audio_queue.clear()
        print("[AUDIO] Audio output closed.")

    ###########################################################################
    # FRAME PARSING & ENQUEUE
    ###########################################################################

    def parse_frame(self, data: bytes):
        """
        Try parsing one frame from 'data' (which may have leftover from previous chunk).
        Returns (frame_type, payload), remainder_data or (None, data) if incomplete.
        """
        if len(data) < self.HEADER_SIZE:
            return None, data

        # Check magic
        if data[:4] != self.MAGIC_BYTES:
            # Attempt to find next valid magic
            idx = data.find(self.MAGIC_BYTES, 1)
            if idx == -1:
                # No valid magic, discard everything
                return None, b''
            else:
                # Skip up to the next magic
                data = data[idx:]
                if len(data) < self.HEADER_SIZE:
                    return None, data

        frame_type = data[4]
        frame_length = struct.unpack('>I', data[5:9])[0]
        total_length = self.HEADER_SIZE + frame_length
        if len(data) < total_length:
            # Incomplete
            return None, data

        payload = data[self.HEADER_SIZE : total_length]
        remainder = data[total_length:]
        return (frame_type, payload), remainder

    def process_utterance_and_enqueue(self, utterance_data: bytes):
        """
        Called when an utterance is complete. Convert int16 mono -> float32 stereo,
        resample if needed, then push to self.audio_queue.
        """
        if not utterance_data:
            return

        # Convert to float32 from int16
        audio_int16 = np.frombuffer(utterance_data, dtype=np.int16)
        if audio_int16.size == 0:
            return
        float_mono = audio_int16.astype(np.float32) / 32768.0

        # Resample if device_rate != input_rate
        if self.device_rate and (self.device_rate != self.input_rate):
            gcd_val = np.gcd(self.device_rate, self.input_rate)
            up = self.device_rate // gcd_val
            down = self.input_rate // gcd_val
            float_mono = signal.resample_poly(float_mono, up, down)

        # Expand to stereo
        float_stereo = np.column_stack([float_mono, float_mono]).astype(np.float32)

        # Add to queue
        self.audio_queue.append(float_stereo)

    async def play_chunk(self, chunk: bytes):
        """
        Public method: parse raw data (which may contain zero, one, or multiple frames).
        Accumulate AUDIO frames in self.current_utterance until we see FRAME_TYPE_END,
        then process & enqueue.
        """
        # If we haven't started, the callback won't play yet,
        # but we can still parse and queue data.
        data = self.partial_frame + chunk
        self.partial_frame = b''

        while True:
            frame, data = self.parse_frame(data)
            if frame is None:
                # No more complete frames
                self.partial_frame = data
                break

            frame_type, payload = frame
            if frame_type == self.FRAME_TYPE_AUDIO:
                # Accumulate
                self.current_utterance.append(payload)
            elif frame_type == self.FRAME_TYPE_END:
                # We have a complete utterance
                if self.current_utterance:
                    combined = b''.join(self.current_utterance)
                    self.current_utterance.clear()
                    self.process_utterance_and_enqueue(combined)

        # Optionally start stream if not playing
        if not self.playing:
            await self.start_stream()
