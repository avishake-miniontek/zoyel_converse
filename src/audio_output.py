import sounddevice as sd
import numpy as np
import threading
import time
import platform
from collections import deque
from scipy import signal

class AudioOutput:
    def __init__(self):
        self.input_rate = 24000  # TTS output rate in Hz
        self.device_rate = None  # Will detect from the actual device
        self.stream = None
        self.audio_queue = deque()
        self.playing = False
        self.play_thread = None
        self.current_device = None

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
        self._open_stream(device_idx)

    async def initialize(self):
        """Initialize audio output asynchronously."""
        if self.stream and self.stream.active:
            return

        device_idx, device_info = self._find_output_device()
        self.device_rate = int(device_info['default_samplerate'])
        self._open_stream(device_idx)

    def _play_audio_thread(self):
        """Audio playback thread."""
        print("Playback thread started")
        while self.playing:
            try:
                if self.audio_queue:
                    chunk = self.audio_queue.popleft()

                    # Skip TTS_END markers
                    if chunk.strip() == b'TTS_END':
                        continue

                    # Handle TTS prefix
                    if chunk.startswith(b'TTS:'):
                        chunk = chunk[4:]

                    try:
                        # Convert to float32 stereo
                        audio_data = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
                        if audio_data.size == 0:
                            continue

                        # Convert to stereo
                        audio_data = np.column_stack((audio_data, audio_data))

                        # Resample if needed
                        if self.device_rate != self.input_rate:
                            gcd = np.gcd(self.device_rate, self.input_rate)
                            up = self.device_rate // gcd
                            down = self.input_rate // gcd
                            audio_data = signal.resample_poly(
                                audio_data,
                                up=up,
                                down=down,
                                axis=0
                            )

                        # Write to stream if it's active
                        if self.stream and self.stream.active:
                            self.stream.write(audio_data)
                        else:
                            # Try to recover the stream
                            print("Stream not active, attempting to recover...")
                            self._open_stream(self.current_device['index'])
                            if self.stream and self.stream.active:
                                self.stream.write(audio_data)

                    except Exception as e:
                        print(f"Error processing chunk: {e}")
                        # Continue processing next chunk
                        continue

                else:
                    time.sleep(0.001)

            except Exception as e:
                print(f"Error in playback loop: {e}")
                # Don't break, try to continue
                time.sleep(0.1)

        print("Playback thread stopping")

    async def play_chunk(self, chunk):
        """Queue a chunk of audio data for playback."""
        if chunk.strip() == b'TTS_END':
            return  # Ignore end markers
        self.audio_queue.append(chunk)

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
