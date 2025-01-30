import sounddevice as sd
import numpy as np
import threading
import time
import platform
from collections import deque
from scipy import signal

class AudioOutput:
    def __init__(self):
        # The sample rate at which TTS audio is received (24kHz)
        self.input_rate = 24000

        # We discover the actual device sample rate dynamically:
        self.device_rate = None

        # SoundDevice stream object
        self.stream = None

        # Queue of raw audio chunks (PCM int16)
        self.audio_queue = deque()

        # Background playback thread
        self.play_thread = None
        self.playing = False

        # Track current output device
        self.current_device = None

    def _find_output_device(self, device_name=None):
        """
        Find suitable output device with improved Linux compatibility.
        """
        print("\nAvailable output devices:")
        devices = sd.query_devices()
        output_devices = []

        # Gather all output devices
        for i, dev in enumerate(devices):
            if dev['max_output_channels'] > 0:
                print(f"Device {i}: {dev['name']} "
                      f"(outputs={dev['max_output_channels']}, "
                      f"rate={dev['default_samplerate']:.0f}Hz)")
                output_devices.append((i, dev))

        system = platform.system()

        try:
            # Linux: Check for PipeWire first, then device_name, then other systems
            if system == 'Linux':
                # Check for PipeWire first
                for i, dev in output_devices:
                    if 'pipewire' in dev['name'].lower():
                        print(f"\nSelected PipeWire device: {dev['name']}")
                        self.current_device = dev
                        return i, dev

                # Then try other Linux-related device names
                preferred_keywords = ['pulse', 'default', 'hw:', 'plughw:', 'dmix', 'surround']
                if device_name:
                    for i, dev in output_devices:
                        if dev['name'] == device_name:
                            print(f"\nSelected output device by name: {dev['name']}")
                            self.current_device = dev
                            return i, dev

                # Then try preferred keywords
                for keyword in preferred_keywords:
                    for i, dev in output_devices:
                        if keyword in dev['name'].lower():
                            print(f"\nSelected Linux output device: {dev['name']}")
                            self.current_device = dev
                            return i, dev

            # Windows-specific selection
            elif system == 'Windows':
                # If device_name specified, try that first
                if device_name:
                    for i, dev in output_devices:
                        if dev['name'] == device_name:
                            print(f"\nSelected output device by name: {dev['name']}")
                            self.current_device = dev
                            return i, dev

                # Then try Windows-specific keywords
                preferred_keywords = ['speakers', 'headphones', 'realtek', 'hdmi', 'wasapi', 'directsound']
                for keyword in preferred_keywords:
                    for i, dev in output_devices:
                        if keyword in dev['name'].lower():
                            print(f"\nSelected Windows device by keyword '{keyword}': {dev['name']}")
                            self.current_device = dev
                            return i, dev

            # Try system default
            default_idx = sd.default.device[1]
            if default_idx is not None and 0 <= default_idx < len(devices):
                device_info = devices[default_idx]
                if device_info['max_output_channels'] > 0:
                    print(f"\nSelected default output device: {device_info['name']}")
                    self.current_device = device_info
                    return default_idx, device_info

            # Last resort - first available output device
            if output_devices:
                idx, dev_info = output_devices[0]
                print(f"\nFalling back to first available output device: {dev_info['name']}")
                self.current_device = dev_info
                return idx, dev_info

            raise RuntimeError("No valid output devices found!")

        except Exception as e:
            raise RuntimeError(f"Error selecting device: {e}")

    def get_device_name(self):
        """Returns name of current output device."""
        if self.current_device:
            return self.current_device['name']
        return "No device selected"

    def _open_stream(self, device_idx):
        """Create and start the output stream with platform-specific optimizations."""
        try:
            # Base stream configuration
            stream_kwargs = {
                'device': device_idx,
                'samplerate': self.device_rate,
                'channels': 2,
                'dtype': np.float32,
            }

            # Platform-specific settings
            system = platform.system()
            if system == 'Linux':
                stream_kwargs['latency'] = 'high'  # More reliable on Linux
            else:
                stream_kwargs['latency'] = 'low'

            self.stream = sd.OutputStream(**stream_kwargs)
            self.stream.start()
            print(f"[TTS Output] Stream opened at {self.device_rate} Hz on device idx {device_idx}")

        except Exception as e:
            if self.stream:
                self.stream.close()
                self.stream = None
            raise RuntimeError(f"Failed to open output stream: {e}")

    def set_device_by_name(self, device_name):
        """Change to a new output device by name."""
        print(f"\n[AudioOutput] Changing output device to '{device_name}' ...")
        self.pause()  # stop playback, join thread
        if self.stream:
            self.stream.close()
            self.stream = None

        idx, info = self._find_output_device(device_name=device_name)
        self.device_rate = int(info['default_samplerate'])
        self._open_stream(idx)

    def initialize_sync(self):
        """Synchronous initialization."""
        if self.stream and self.stream.active:
            return

        print("[TTS Output] Initializing audio output (sync)...")
        device_idx, device_info = self._find_output_device()
        self.device_rate = int(device_info['default_samplerate'])
        self._open_stream(device_idx)

    async def initialize(self):
        """Asynchronous initialization."""
        if self.stream and self.stream.active:
            return

        print("[TTS Output] Initializing audio output (async)...")
        device_idx, device_info = self._find_output_device()
        self.device_rate = int(device_info['default_samplerate'])
        self._open_stream(device_idx)

    def _play_audio_thread(self):
        """Process and play audio chunks immediately as they arrive."""
        while self.playing:
            try:
                if self.audio_queue:
                    chunk = self.audio_queue.popleft()

                    # Remove TTS: prefix if present
                    if chunk.startswith(b'TTS:'):
                        chunk = chunk[4:]

                    # Convert PCM int16 -> float32 in [-1..1]
                    audio_data = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0

                    # Convert mono to stereo by stacking columns
                    audio_data = np.column_stack((audio_data, audio_data))

                    # If device rate is different from input_rate, resample
                    if self.device_rate != self.input_rate:
                        gcd_val = np.gcd(self.device_rate, self.input_rate)
                        up = self.device_rate // gcd_val
                        down = self.input_rate // gcd_val
                        audio_data = signal.resample_poly(
                            audio_data,
                            up=up,
                            down=down,
                            axis=0,
                            padtype='line'
                        )

                    # Write to stream if active
                    if self.stream and self.stream.active:
                        self.stream.write(audio_data)
                else:
                    time.sleep(0.001)

            except Exception as e:
                print(f"[TTS Output] Error in playback thread: {e}")
                self.playing = False
                break

    async def play_chunk(self, chunk: bytes):
        """Queue a raw audio chunk for immediate processing."""
        try:
            # Queue the raw chunk directly - processing happens in play thread
            self.audio_queue.append(chunk)
        except Exception as e:
            print(f"[TTS Output] Error queueing chunk: {e}")

    async def start_stream(self):
        """Ensure stream is open and start playback thread."""
        if not self.stream or not self.stream.active:
            await self.initialize()

        if not self.playing:
            self.playing = True
            self.play_thread = threading.Thread(target=self._play_audio_thread, daemon=True)
            self.play_thread.start()
            print("[TTS Output] Playback thread started")

    def pause(self):
        """Stop playback and clear the queue."""
        self.playing = False
        if self.play_thread and self.play_thread.is_alive():
            self.play_thread.join(timeout=1.0)

        if self.stream and self.stream.active:
            self.stream.stop()

        self.audio_queue.clear()

    def close(self):
        """Clean shutdown of playback thread and audio stream."""
        print("[TTS Output] AudioOutput.close() called")
        self.pause()
        if self.stream:
            try:
                self.stream.close()
            except Exception as e:
                print(f"[TTS Output] Error closing stream: {e}")
            self.stream = None
        print("[TTS Output] AudioOutput.close() finished")