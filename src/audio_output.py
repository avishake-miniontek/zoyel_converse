import json
import sounddevice as sd
import numpy as np
from collections import deque
import platform
import struct
from scipy import signal
import time
import sys
import asyncio
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
import logging

logger = logging.getLogger(__name__)

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

        # A queue of final float32 stereo blocks, ready for playback.
        # Each element is shape (samples, 2). Limit queue size to prevent memory issues.
        self.audio_queue = deque(maxlen=32)

        # For partial frame parsing.
        self.partial_frame = b''          # Leftover bytes from last chunk.
        self.current_utterance = []       # Accumulate AUDIO frames until FRAME_TYPE_END.

        # Whether we're actively playing audio.
        self.playing = False

        # Device info.
        self.current_device = None

        # NEW: Volume factor (1.0 means full volume). May be adjusted via CLI.
        self.volume = 1.0

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

        logger.info("\n[AUDIO] Available output devices:")
        for i, dev in enumerate(devices):
            if dev['max_output_channels'] > 0:
                logger.info(f"   {i} {dev['name']}, ALSA ({dev['max_input_channels']} in, {dev['max_output_channels']} out)")
                output_devices.append((i, dev))

        with open('config.json', 'r') as f:
            config = json.load(f)
        
        if 'audio_devices' in config and 'output_device' in config['audio_devices']:
            output_device = config['audio_devices']['output_device']
            if isinstance(output_device, (int, float)):
                device_idx = int(output_device)
                for i, dev in output_devices:
                    if i == device_idx:
                        logger.info(f"\n[AUDIO] Selected output device: {dev['name']}")
                        self.current_device = dev
                        return i, dev
        
        if device_name:
            for i, dev in output_devices:
                if dev['name'] == device_name:
                    logger.info(f"\n[AUDIO] Selected output device: {dev['name']}")
                    self.current_device = dev
                    return i, dev

        default_idx = sd.default.device[1]
        if default_idx is not None and 0 <= default_idx < len(devices):
            device_info = devices[default_idx]
            logger.info(f"\n[AUDIO] Selected default device: {device_info['name']}")
            self.current_device = device_info
            return default_idx, device_info

        if output_devices:
            i, dev = output_devices[0]
            logger.info(f"\n[AUDIO] Selected first available device: {dev['name']}")
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
        logger.info(f"[AUDIO] Switching to device: {device_name}")
        self.pause()
        if self.stream:
            self.stream.close()
            self.stream = None
        self.initialize_sync(device_name=device_name)
        if self.playing:
            asyncio.run(self.start_stream())

    def initialize_sync(self, device_name=None):
        """
        Synchronous init if you prefer. Similar to async but blocking.
        """
        if self.stream and self.stream.active:
            return
        try:
            self._do_initialize(device_name=device_name)
        except Exception as e:
            logger.error("Error in initialize_sync: %s", e)

    async def initialize(self, device_name=None):
        """
        Async init. If the stream is not active, pick device and open stream.
        """
        if self.stream and self.stream.active:
            return
        self._do_initialize(device_name=device_name)

    def get_platform_audio_config(self):
        """
        Get platform-specific audio configuration to better integrate with system audio.
        """
        system = platform.system().lower()
        if system == 'linux':
            # Try PulseAudio first, fall back to ALSA.
            try:
                return {'device': 'pulse'}
            except:
                return {'device': None}  # Let sounddevice choose default.
        elif system == 'darwin':  # macOS
            return {'device': 'coreaudio default'}
        elif system == 'windows':
            return {'device': 'wasapi'}
        return {'device': None}  # Default fallback.

    async def _open_stream(self, device_idx):
        """
        Internal method to open the audio stream with better error handling 
        and platform-specific configuration.
        """
        try:
            buffer_size = min(2048, self.device_rate // 20)
            
            # Get platform-specific audio config.
            platform_config = self.get_platform_audio_config()
            
            stream_kwargs = {
                'samplerate': self.device_rate,
                'channels': 2,
                'dtype': np.float32,
                'latency': 0.1,
                'blocksize': buffer_size,
                'callback': self._audio_callback
            }

            # If a specific device index was provided, use it.
            if device_idx is not None:
                stream_kwargs['device'] = device_idx
            # Otherwise use the platform-specific device.
            elif platform_config['device'] is not None:
                stream_kwargs['device'] = platform_config['device']

            logger.info(f"[AUDIO] Opening stream with settings: {stream_kwargs}")
            self.stream = sd.OutputStream(**stream_kwargs)
            self.stream.start()
            await asyncio.sleep(0.2)
            logger.info("[AUDIO] Stream started successfully.")

        except Exception as e:
            logger.error("Failed to open stream: %s", e)
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
        
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self._open_stream(device_idx))
            logger.info(f"[AUDIO] Stream started on device: {dev_info['name']} at {self.device_rate} Hz")
        except Exception as e:
            logger.error("Error in _do_initialize: %s", e)
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
            logger.warning("[AUDIO] Output underflow (no data).")

        outdata.fill(0.0)
        needed = frames
        offset = 0

        while needed > 0 and self.audio_queue:
            chunk = self.audio_queue[0]
            chunk_len = len(chunk)
            if chunk_len <= needed:
                outdata[offset : offset + chunk_len] = chunk
                offset += chunk_len
                needed -= chunk_len
                self.audio_queue.popleft()
            else:
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
            logger.info("[AUDIO] Playback paused.")

    async def close(self):
        """
        Fully close the audio stream and reset everything.
        """
        logger.info("[AUDIO] Closing audio output...")
        self.pause()
        if self.stream:
            self.stream.close()
            self.stream = None
        self.partial_frame = b''
        self.current_utterance.clear()
        self.audio_queue.clear()
        logger.info("[AUDIO] Audio output closed.")

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

        if data[:4] != self.MAGIC_BYTES:
            idx = data.find(self.MAGIC_BYTES, 1)
            if idx == -1:
                return None, b''
            else:
                data = data[idx:]
                if len(data) < self.HEADER_SIZE:
                    return None, data

        frame_type = data[4]
        frame_length = struct.unpack('>I', data[5:9])[0]
        total_length = self.HEADER_SIZE + frame_length
        if len(data) < total_length:
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

        audio_int16 = np.frombuffer(utterance_data, dtype=np.int16)
        if audio_int16.size == 0:
            return
        float_mono = audio_int16.astype(np.float32) / 32768.0

        if self.device_rate and (self.device_rate != self.input_rate):
            samples_out = int(len(float_mono) * self.device_rate / self.input_rate)
            float_mono = signal.resample(float_mono, samples_out)

        float_stereo = np.column_stack([float_mono, float_mono]).astype(np.float32)
        # Apply the volume factor.
        float_stereo *= self.volume
        self.audio_queue.append(float_stereo)

    async def play_chunk(self, chunk: bytes):
        """
        Public method: parse raw data (which may contain zero, one, or multiple frames).
        Accumulate AUDIO frames in self.current_utterance until we see FRAME_TYPE_END,
        then process & enqueue.
        """
        data = self.partial_frame + chunk
        self.partial_frame = b''

        while True:
            frame, data = self.parse_frame(data)
            if frame is None:
                self.partial_frame = data
                break

            frame_type, payload = frame
            if frame_type == self.FRAME_TYPE_AUDIO:
                self.current_utterance.append(payload)
            elif frame_type == self.FRAME_TYPE_END:
                if self.current_utterance:
                    combined = b''.join(self.current_utterance)
                    self.current_utterance.clear()
                    self.process_utterance_and_enqueue(combined)

        if not self.playing:
            await self.start_stream()
