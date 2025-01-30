import sounddevice as sd
import numpy as np
import threading
import time
import platform
from collections import deque
from scipy import signal

class AudioOutput:
    """
    AudioOutput provides a flexible, cross-platform way to:
      - Find and open a suitable audio device
      - Dynamically handle TTS audio (24 kHz by default) and resample it to the device rate
      - Accumulate TTS data in chunks until 'TTS_END' for single-pass resampling
      - Play the resampled float32 stereo audio in real-time via a background thread
      - Switch output devices on the fly if needed

    This version reverts Linux device detection logic to match the old code exactly,
    while retaining the new single-pass TTS approach and background thread playback.
    It also removes padtype='line' to avoid issues on older SciPy installations.
    """

    def __init__(self):
        # The sample rate at which TTS audio is received from the server (e.g., 24000 Hz).
        self.input_rate = 24000

        # We discover the actual device sample rate dynamically:
        self.device_rate = None

        # SoundDevice stream object
        self.stream = None

        # A queue of *fully prepared* float32 stereo buffers to be played in the background thread.
        self.audio_queue = deque()

        # Background playback thread
        self.play_thread = None
        self.playing = False

        # Track the currently selected output device
        self.current_device = None

        # For single-pass TTS accumulation
        self._accumulator = bytearray()
        self._accumulating = False

    ##########################################################################
    # Device selection (Linux logic reverted to old code)
    ##########################################################################

    def _find_output_device(self, device_name=None):
        """
        Finds a suitable output device across platforms. For Linux, it uses the
        same logic from the old code. Windows logic largely remains similar
        (preferring 'speakers', 'headphones', etc.). If no device is found,
        it falls back to the system default or the first available output device.
        """
        print("\nAvailable output devices:")
        devices = sd.query_devices()
        output_devices = []

        # Gather all devices that support output
        for i, dev in enumerate(devices):
            if dev['max_output_channels'] > 0:
                print(f"Device {i}: {dev['name']} "
                      f"(outputs={dev['max_output_channels']}, "
                      f"rate={dev['default_samplerate']:.0f}Hz)")
                output_devices.append((i, dev))

        system = platform.system()

        try:
            # If user specified a device_name, try to match it (before OS heuristics)
            if device_name:
                for i, dev in output_devices:
                    if dev['name'] == device_name:
                        print(f"\nSelected output device by exact name: {dev['name']}")
                        self.current_device = dev
                        return i, dev

            # Windows-specific attempts
            if system == 'Windows':
                preferred_keywords = ['speakers', 'headphones', 'realtek',
                                      'hdmi', 'wasapi', 'directsound']
                for keyword in preferred_keywords:
                    for i, dev in output_devices:
                        if keyword in dev['name'].lower():
                            print(f"\nSelected Windows device by keyword '{keyword}': {dev['name']}")
                            self.current_device = dev
                            return i, dev

            # Linux-specific device detection (from old code)
            elif system == 'Linux':
                # First try PipeWire
                for i, dev in output_devices:
                    if 'pipewire' in dev['name'].lower():
                        print(f"\nSelected PipeWire device: {dev['name']}")
                        self.current_device = dev
                        return i, dev

                # Then try other Linux-specific devices
                preferred_keywords = [
                    'pulse', 'default',  # Modern audio servers
                    'hw:', 'plughw:',    # ALSA devices
                    'dmix', 'surround'   # ALSA plugins
                ]

                # If device_name was provided but not matched exactly above, see if it partially matches:
                if device_name:
                    for i, dev in output_devices:
                        if dev['max_output_channels'] > 0 and dev['name'] == device_name:
                            print(f"\nSelected output device: {dev['name']}")
                            self.current_device = dev
                            return i, dev

                # Otherwise try preferred devices in order
                for keyword in preferred_keywords:
                    for i, dev in output_devices:
                        if keyword in dev['name'].lower():
                            print(f"\nSelected Linux output device: {dev['name']}")
                            self.current_device = dev
                            return i, dev

            # If no match yet, fall back to system default
            default_output_idx = sd.default.device[1]  # default output device index
            if default_output_idx is not None and 0 <= default_output_idx < len(devices):
                device_info = devices[default_output_idx]
                print(f"\nFalling back to default output device: {device_info['name']}")
                self.current_device = device_info
                return default_output_idx, device_info

            # If default device is invalid, pick the first from output_devices
            if len(output_devices) > 0:
                idx, dev_info = output_devices[0]
                print(f"\nFalling back to first available output device: {dev_info['name']}")
                self.current_device = dev_info
                return idx, dev_info

            # If we reach here, no valid output devices found
            raise RuntimeError("No valid output devices found at all!")

        except Exception as e:
            raise RuntimeError(f"Error selecting device: {e}")

    def get_device_name(self):
        """
        Returns the name of the currently selected output device.
        """
        if self.current_device:
            return self.current_device['name']
        return "No device selected"

    ##########################################################################
    # Stream management
    ##########################################################################

    def _open_stream(self, device_idx):
        """
        Create and start the sounddevice.OutputStream with the given device index
        and self.device_rate. We use float32 stereo for playback.
        """
        try:
            print(f"[TTS Output] Opening playback stream on '{self.current_device['name']}' "
                  f"at {self.device_rate} Hz...")
            stream_kwargs = {
                'device': device_idx,
                'samplerate': self.device_rate,
                'channels': 2,
                'dtype': np.float32,
                'latency': 'low'
            }

            # If Windows, optionally apply WASAPI settings
            if platform.system() == 'Windows':
                # Additional optional config for WASAPI
                try:
                    import sounddevice as sd_internal
                    wasapi_info = sd_internal.query_devices(device_idx, 'output').get('hostapi')
                    # Only do this if the host API is WASAPI
                    if wasapi_info is not None:
                        stream_kwargs['extra_settings'] = sd.WasapiSettings(
                            exclusive=False,
                            # buffer size can be tweaked as needed
                        )
                except Exception:
                    pass  # If we can't import or configure WASAPI, we just skip

            self.stream = sd.OutputStream(**stream_kwargs)
            self.stream.start()
            print(f"[TTS Output] Stream opened successfully on device idx {device_idx}")

        except Exception as e:
            if self.stream:
                self.stream.close()
                self.stream = None
            raise RuntimeError(f"Failed to open output stream: {e}")

    def set_device_by_name(self, device_name):
        """
        Change to a new output device by name (or partial name).
        Closes the current stream and re-initializes using old Linux logic
        or the relevant OS logic from above.

        :param device_name: The exact name of the desired device
        """
        print(f"\n[AudioOutput] Changing output device to '{device_name}' ...")
        self.pause()  # stop playback, join thread
        if self.stream:
            self.stream.close()
            self.stream = None

        idx, info = self._find_output_device(device_name=device_name)
        self.device_rate = int(info['default_samplerate'])
        self._open_stream(idx)

    def initialize_sync(self):
        """
        Synchronous initialization. Select device, set self.device_rate, open the stream.
        """
        if self.stream and self.stream.active:
            return  # already initialized

        print("[TTS Output] Initializing audio output (sync)...")
        device_idx, device_info = self._find_output_device()
        self.device_rate = int(device_info['default_samplerate'])
        self._open_stream(device_idx)

    async def initialize(self):
        """
        Asynchronous version of initialization. Same logic, but for use in an async context.
        """
        if self.stream and self.stream.active:
            return  # already active

        print("[TTS Output] Initializing audio output (async)...")
        device_idx, device_info = self._find_output_device()
        self.device_rate = int(device_info['default_samplerate'])
        self._open_stream(device_idx)

    ##########################################################################
    # Playback thread
    ##########################################################################

    def _play_audio_thread(self):
        """
        Runs in a dedicated thread: continuously pops float32 stereo arrays
        from self.audio_queue and writes them to the output stream.
        """
        while self.playing:
            if self.audio_queue:
                audio_data = self.audio_queue.popleft()
                try:
                    if self.stream and self.stream.active:
                        self.stream.write(audio_data)
                except Exception as e:
                    print(f"[TTS Output] Error writing to stream: {e}")
                    self.playing = False
            else:
                time.sleep(0.001)

    async def start_stream(self):
        """
        Ensure the stream is open, then start the playback thread if not already running.
        """
        if not self.stream or not self.stream.active:
            await self.initialize()

        if not self.playing:
            self.playing = True
            self.play_thread = threading.Thread(target=self._play_audio_thread, daemon=True)
            self.play_thread.start()
            print("[TTS Output] Playback thread started")

    ##########################################################################
    # Single-pass TTS logic
    ##########################################################################

    async def play_chunk(self, chunk: bytes):
        """
        Main entrypoint for receiving TTS or raw audio bytes.

        - If chunk starts with b'TTS:', we accumulate that data (16-bit PCM) in _accumulator.
        - If chunk.strip() == b'TTS_END', we finalize the utterance (single-pass resample).
        - Otherwise, we can treat it as raw PCM or partial TTS. We append to _accumulator.
        """
        # Strip trailing whitespace, e.g. b'\r\n'
        stripped_chunk = chunk.strip()

        # (1) If it's TTS data
        if stripped_chunk.startswith(b'TTS:'):
            # TTS partial chunk (remove the 'TTS:' prefix)
            data = stripped_chunk[4:]
            self._accumulator.extend(data)
            self._accumulating = True

        # (2) If it's TTS_END
        elif stripped_chunk == b'TTS_END':
            if self._accumulating and len(self._accumulator) > 0:
                await self._finalize_tts_utterance()
            self._accumulating = False
            self._accumulator.clear()

        # (3) Otherwise, treat as raw partial chunk (if your server sends raw PCM)
        else:
            self._accumulator.extend(chunk)

    async def _finalize_tts_utterance(self):
        """
        When TTS_END arrives, do a single-pass resample of everything in _accumulator:
          1) Convert int16 -> float32 in [-1..1], mono
          2) Expand to stereo
          3) Resample from self.input_rate -> self.device_rate
          4) Enqueue final float32 stereo buffer for playback
        """
        try:
            if not self._accumulator:
                return  # Nothing to finalize

            raw_pcm = bytes(self._accumulator)

            # Convert to float32 mono
            mono_f32 = np.frombuffer(raw_pcm, dtype=np.int16).astype(np.float32) / 32768.0
            if mono_f32.size == 0:
                return

            # Expand to stereo
            stereo_f32 = np.column_stack((mono_f32, mono_f32))

            # Resample once if needed
            if self.device_rate != self.input_rate:
                gcd_val = np.gcd(self.device_rate, self.input_rate)
                up = self.device_rate // gcd_val
                down = self.input_rate // gcd_val
                # Use a more common padtype than 'line' to avoid compatibility issues on older SciPy
                stereo_f32 = signal.resample_poly(
                    stereo_f32,
                    up=up,
                    down=down,
                    axis=0,
                    padtype='constant'
                )

            # Now stereo_f32 is float32, at device_rate
            self.audio_queue.append(stereo_f32)

        except Exception as e:
            print(f"[TTS Output] Error finalizing TTS utterance: {e}")

    ##########################################################################
    # Control methods: pause and close
    ##########################################################################

    def pause(self):
        """
        Stops playback and clears the queue. Also stops the background thread.
        """
        self.playing = False
        if self.play_thread and self.play_thread.is_alive():
            self.play_thread.join(timeout=1.0)

        if self.stream and self.stream.active:
            self.stream.stop()

        self.audio_queue.clear()

    def close(self):
        """
        Cleanly shut down the playback thread and close the audio stream.
        """
        print("[TTS Output] AudioOutput.close() called")
        self.pause()
        if self.stream:
            try:
                self.stream.close()
            except Exception as e:
                print(f"[TTS Output] Error closing stream: {e}")
            self.stream = None
        print("[TTS Output] AudioOutput.close() finished")
