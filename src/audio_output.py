import sounddevice as sd
import numpy as np
import threading
from collections import deque
import time
from scipy import signal
import platform

class AudioOutput:
    def __init__(self):
        self.stream = None
        # The rate at which the TTS audio is generated (or received) from the server
        self.input_rate = 24000  # TTS output rate in Hz

        self.device_rate = None  # Will detect from the actual device
        self.audio_queue = deque()  # Store audio chunks
        self.playing = False
        self.play_thread = None
        self.current_device = None

    def _find_output_device(self, device_name=None):
        """
        Find a suitable audio output device across platforms.

        Returns:
            device_index, device_info
        """
        try:
            print("\nAvailable output devices:")
            devices = sd.query_devices()
            output_devices = []

            # Collect all devices that support output
            for i, dev in enumerate(devices):
                if dev['max_output_channels'] > 0:
                    print(f"Device {i}: {dev['name']} "
                          f"(outputs: {dev['max_output_channels']}, "
                          f"rate: {dev['default_samplerate']}Hz)")
                    output_devices.append((i, dev))

            system = platform.system()

            # Windows-specific suggestions (try to pick a typical device if not forcing device_name)
            if system == 'Windows':
                preferred_keywords = [
                    'speakers', 'headphones', 'realtek', 'hdmi',
                    'wasapi', 'directsound'
                ]
                for keyword in preferred_keywords:
                    for i, dev in output_devices:
                        if keyword in dev['name'].lower():
                            print(f"\nSelected Windows output device: {dev['name']}")
                            self.current_device = dev
                            return i, dev

            # Linux-specific suggestions (PipeWire, Pulse, etc.)
            elif system == 'Linux':
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

                for keyword in preferred_keywords:
                    for i, dev in output_devices:
                        if keyword in dev['name'].lower():
                            print(f"\nSelected Linux output device: {dev['name']}")
                            self.current_device = dev
                            return i, dev

            # If device_name was explicitly requested, try to find and use it
            if device_name:
                for i, dev in output_devices:
                    if dev['name'] == device_name:
                        print(f"\nSelected output device: {dev['name']}")
                        self.current_device = dev
                        return i, dev

            # Otherwise fall back to the system default
            default_idx = sd.default.device[1]  # index for the default OUTPUT device
            if default_idx is not None and 0 <= default_idx < len(devices):
                device_info = devices[default_idx]
            else:
                # If default device index is invalid, pick the first available output device
                if not output_devices:
                    raise RuntimeError("No output devices found at all.")
                default_idx, device_info = output_devices[0]

            print(f"\nSelected default output device: {device_info['name']}")
            self.current_device = device_info
            return default_idx, device_info

        except Exception as e:
            print(f"Error finding output device: {e}")
            raise

    def get_device_name(self):
        """Return the name of the current output device, or a fallback string."""
        if self.current_device:
            return self.current_device['name']
        return "No device selected"

    def set_device_by_name(self, device_name):
        """
        Change the output device by name.
        """
        print(f"\nChanging output device to: {device_name}")
        try:
            self.pause()
            if self.stream:
                self.stream.close()
                self.stream = None

            # Find new device
            device_idx, device_info = self._find_output_device(device_name)
            self.device_rate = int(device_info['default_samplerate'])

            stream_kwargs = {
                'device': device_idx,
                'samplerate': self.device_rate,
                'channels': 2,      # stereo
                'dtype': np.float32,
                'latency': 'low'    # you can omit or adjust this if needed
            }

            self.stream = sd.OutputStream(**stream_kwargs)
            self.stream.start()

            print(f"Successfully switched to device: {device_name}")

        except Exception as e:
            print(f"Error changing output device: {e}")
            # Fallback to initialize default device
            self.initialize_sync()

    def initialize_sync(self):
        """
        Synchronous version of initialization:
        - Detect and store the device sample rate
        - Open the output stream
        """
        if self.stream and self.stream.active:
            return  # Already initialized

        try:
            print("[TTS Output] Initializing audio output...")

            device_idx, device_info = self._find_output_device()
            self.device_rate = int(device_info['default_samplerate'])

            stream_kwargs = {
                'device': device_idx,
                'samplerate': self.device_rate,
                'channels': 2,
                'dtype': np.float32,
                'latency': 'low'
            }

            self.stream = sd.OutputStream(**stream_kwargs)
            self.stream.start()

            print(f"[TTS Output] Successfully initialized audio at device rate {self.device_rate} Hz")

        except Exception as e:
            print(f"[TTS Output] Error initializing audio: {e}")
            if self.stream:
                self.stream.close()
                self.stream = None

    async def initialize(self):
        """Asynchronous version of initialization."""
        if self.stream and self.stream.active:
            return  # Already initialized

        try:
            print("[TTS Output] Initializing audio output...")

            device_idx, device_info = self._find_output_device()
            self.device_rate = int(device_info['default_samplerate'])

            stream_kwargs = {
                'device': device_idx,
                'samplerate': self.device_rate,
                'channels': 2,
                'dtype': np.float32,
                'latency': 'low'
            }

            self.stream = sd.OutputStream(**stream_kwargs)
            self.stream.start()

            print(f"[TTS Output] Successfully initialized audio at device rate {self.device_rate} Hz")

        except Exception as e:
            print(f"[TTS Output] Error initializing audio: {e}")
            if self.stream:
                self.stream.close()
                self.stream = None

    def _play_audio_thread(self):
        """
        Background thread for continuous playback:
        - Dequeues PCM int16 data from self.audio_queue
        - Converts to float32 in [-1..1]
        - Resamples from self.input_rate to self.device_rate if needed
        - Writes to the active sounddevice.OutputStream
        """
        while self.playing:
            try:
                if self.audio_queue:
                    chunk = self.audio_queue.popleft()

                    # Remove a "TTS:" prefix if present
                    if chunk.startswith(b'TTS:'):
                        chunk = chunk[4:]

                    # Convert PCM int16 -> float32 in [-1..1]
                    audio_data = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0

                    # Convert mono to stereo by stacking columns:
                    # shape: (N,) -> (N, 2)
                    audio_data = np.column_stack((audio_data, audio_data))

                    # If device rate is different from input_rate, resample
                    if self.device_rate != self.input_rate:
                        gcd_val = np.gcd(self.device_rate, self.input_rate)
                        up = self.device_rate // gcd_val
                        down = self.input_rate // gcd_val
                        # Use polyphase resampling to maintain pitch/duration:
                        audio_data = signal.resample_poly(
                            audio_data,
                            up=up,
                            down=down,
                            axis=0,
                            padtype='line'
                        )
                        # Now audio_data has shape (new_length, 2)

                    # If there's an active stream, write the float32 data
                    if self.stream and self.stream.active:
                        self.stream.write(audio_data)
                else:
                    # If queue is empty, sleep briefly
                    time.sleep(0.001)

            except Exception as e:
                print(f"[TTS Output] Error in playback thread: {e}")
                self.playing = False
                break  # exit thread on error

    async def play_chunk(self, chunk):
        """
        Public method to enqueue a chunk of audio data (bytes).
        The chunk is expected to be 16-bit PCM at self.input_rate.
        """
        try:
            self.audio_queue.append(chunk)
        except Exception as e:
            print(f"[TTS Output] Error queueing chunk: {e}")

    async def start_stream(self):
        """
        Ensure the output stream is running and start the background playback thread if needed.
        """
        try:
            if not self.stream or not self.stream.active:
                await self.initialize()

            # Start playback thread if not already running
            if not self.playing:
                self.playing = True
                self.play_thread = threading.Thread(target=self._play_audio_thread)
                self.play_thread.daemon = True
                self.play_thread.start()
                print("[TTS Output] Playback thread started")

        except Exception as e:
            print(f"[TTS Output] Error starting stream: {e}")

    def pause(self):
        """
        Pause/stop playback by stopping the background thread and clearing the queue.
        """
        self.playing = False
        if self.play_thread:
            self.play_thread.join(timeout=1.0)
        if self.stream and self.stream.active:
            self.stream.stop()
        # Clear any remaining audio
        self.audio_queue.clear()

    def close(self):
        """
        Clean up audio resources and ensure the playback thread terminates.
        """
        print("[TTS Output] AudioOutput.close() called")
        self.playing = False
        # Join the thread
        if self.play_thread and self.play_thread.is_alive():
            try:
                self.play_thread.join(timeout=1.0)
            except Exception as e:
                print(f"[TTS Output] Error joining playback thread: {e}")

        # Close the stream if needed
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                print(f"[TTS Output] Error closing stream: {e}")
            finally:
                self.stream = None
        print("[TTS Output] AudioOutput.close() finished")
