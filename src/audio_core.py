"""
Client-side audio core for capturing and preprocessing audio.
Handles device selection, audio capture, and basic audio level monitoring.
"""

import json
import numpy as np
import time
import sounddevice as sd
import platform
import warnings
from src import audio_utils

class AudioCore:
    def __init__(self):
        with open('config.json', 'r') as f:
            self.config = json.load(f)

        # Basic audio properties
        self.stream = None
        self.rate = None
        self.needs_resampling = None
        self.CHUNK = self.config['audio_processing']['chunk_size']
        self.CHANNELS = None  # Will be set based on device
        self.DESIRED_RATE = self.config['audio_processing']['desired_rate']

        # Audio level tracking for GUI
        self._noise_floor = -96.0
        self._min_floor   = -96.0
        self._max_floor   = -36.0
        self._rms_level   = -96.0
        self._peak_level  = -96.0
        self.last_update  = time.time()

        # This "calibration" is mostly for display logs
        self.calibrated_floor = None

    # ----------------------------------------------------------------
    # Audio level properties for GUI meter
    # ----------------------------------------------------------------
    @property
    def noise_floor(self):
        return self._noise_floor

    @noise_floor.setter
    def noise_floor(self, value):
        try:
            if value is not None:
                self._noise_floor = float(value)
            else:
                self._noise_floor = -96.0
        except:
            self._noise_floor = -96.0

    @property
    def min_floor(self):
        return self._min_floor

    @min_floor.setter
    def min_floor(self, value):
        self._min_floor = float(value)

    @property
    def max_floor(self):
        return self._max_floor

    @max_floor.setter
    def max_floor(self, value):
        self._max_floor = float(value)

    @property
    def rms_level(self):
        return self._rms_level

    @rms_level.setter
    def rms_level(self, value):
        self._rms_level = float(value)

    @property
    def peak_level(self):
        return self._peak_level

    @peak_level.setter
    def peak_level(self, value):
        self._peak_level = float(value)

    # ----------------------------------------------------------------
    # Audio Device Initialization & Calibration
    # ----------------------------------------------------------------
    def init_audio_device(self):
        """
        Initialize audio device and perform a short calibration for GUI floor display.
        """
        try:
            system = platform.system().lower()
            if system == 'linux':
                warnings.filterwarnings("ignore", category=RuntimeWarning, module="sounddevice")

            devices = sd.query_devices()
            print("\nListing audio devices:")
            for i, dev in enumerate(devices):
                if dev['max_input_channels'] > 0:
                    print(f"   {i} {dev['name']}, ALSA ({dev['max_input_channels']} in, {dev['max_output_channels']} out)")
            working_device = None
            device_info = None

            # Check if audio_devices is present in config
            if 'audio_devices' in self.config and 'input_device' in self.config['audio_devices']:
                input_device = self.config['audio_devices']['input_device']
                # Handle numeric index directly - don't try to match by name
                # since client.py no longer converts indices to names
                if isinstance(input_device, (int, float)):
                    device_idx = int(input_device)
                    if 0 <= device_idx < len(devices):
                        device_info = devices[device_idx]
                        working_device = device_idx
            else:
                # Use existing auto device selection logic
                for i, device in enumerate(devices):
                    if device['max_input_channels'] > 0:
                        # Example: prefer a name with 'microphone' on macOS
                        if system == 'darwin' and 'microphone' in device['name'].lower():
                            working_device = i
                            device_info = device
                            break
                        elif system == 'linux' and 'acp' in device['name'].lower():
                            working_device = i
                            device_info = device
                            break

            # fallback to default if none matched
            if working_device is None:
                default_idx = sd.default.device[0]  # Get default input device index
                if default_idx is not None:
                    device_info = sd.query_devices(default_idx)
                    working_device = default_idx
                    print(f"\nUsing default input device: {device_info['name']}")

            if working_device is None or device_info is None:
                raise RuntimeError("No suitable input device found.")

            rate = int(device_info['default_samplerate'])
            needs_resampling = (rate != self.DESIRED_RATE)

            print("\nSelected device details:")
            print(f"  Name: {device_info['name']}")
            print(f"  Input channels: {device_info['max_input_channels']}")
            print(f"  Default samplerate: {rate}")
            print(f"  Latency: {device_info['default_low_input_latency']}")

            sd.default.device = (working_device, None)

            # Short calibration for GUI floor display
            calibration_duration = 0.5  # 500ms is enough for floor detection
            frames_needed = int(rate * calibration_duration)
            print(f"\nCalibrating GUI floor for {calibration_duration}s...")
            # Use a smaller chunk size for calibration on resource-constrained devices
            audio_buffer = sd.rec(frames_needed, samplerate=rate,
                                channels=1, dtype='float32',
                                blocksize=1024)  # Smaller blocksize for better responsiveness
            sd.wait()  # This will be faster now due to shorter duration
            audio_buffer = audio_buffer.flatten()

            chunk_rms_list = []
            chunk_size = 1024
            for i in range(0, len(audio_buffer), chunk_size):
                block = audio_buffer[i : i + chunk_size]
                if len(block) > 0:
                    block_rms = np.sqrt(np.mean(block**2))
                    block_rms_db = 20.0 * np.log10(max(block_rms, 1e-10))
                    chunk_rms_list.append(block_rms_db)

            if chunk_rms_list:
                initial_floor = float(np.percentile(chunk_rms_list, 20))
                # clamp
                if initial_floor < -85.0:
                    initial_floor = -85.0
                if initial_floor > -20.0:
                    initial_floor = -20.0
                self.noise_floor = initial_floor
                self.min_floor   = initial_floor
                self.max_floor   = initial_floor + 60
                self.rms_level   = initial_floor
                self.peak_level  = initial_floor
                self.calibrated_floor = initial_floor
            else:
                self.calibrated_floor = -60.0

            print(f"  GUI floor set to: {self.calibrated_floor:.1f} dB")

            # Get the number of input channels from the device
            self.CHANNELS = device_info['max_input_channels']
            print(f"  Number of channels: {self.CHANNELS}")

            # Open the main stream for continuous capture
            print("\nOpening main input stream...")
            stream = sd.InputStream(
                device=working_device,
                channels=self.CHANNELS,  # Use actual channel count
                samplerate=rate,
                dtype=np.float32,
                blocksize=self.CHUNK
            )
            stream.start()

            self.stream = stream
            self.rate = rate
            self.needs_resampling = needs_resampling

            return stream, device_info, rate, needs_resampling

        except Exception as e:
            raise RuntimeError(f"Failed to initialize audio: {str(e)}")

    # ----------------------------------------------------------------
    # Audio format conversion
    # ----------------------------------------------------------------
    def process_audio(self, audio_data):
        """
        Process audio for client-side level monitoring.
        
        Args:
            audio_data: float32 array of audio samples (multi-channel or mono)
            
        Returns:
            dict: Contains processed audio data and level information
        """
        # Convert to mono if multi-channel
        audio_data = audio_utils.convert_to_mono(audio_data)
        
        # Calculate levels and update tracking
        instant_rms_db, instant_peak_db = audio_utils.calculate_audio_levels(audio_data)
        
        now = time.time()
        self.last_update = now

        # Smoothed level tracking with faster attack, slower release
        if instant_rms_db > self.rms_level:
            self.rms_level += 0.7 * (instant_rms_db - self.rms_level)
        else:
            self.rms_level += 0.3 * (instant_rms_db - self.rms_level)
        
        if instant_peak_db > self.peak_level:
            self.peak_level = instant_peak_db
        else:
            self.peak_level *= 0.95  # Slightly faster decay

        # Return audio metrics
        return {
            'db_level': self.rms_level,
            'noise_floor': self.noise_floor,
            'peak_level': self.peak_level,
            'audio': audio_data
        }

    # ----------------------------------------------------------------
    # Resource cleanup
    # ----------------------------------------------------------------
    def close(self):
        """Clean up audio resources."""
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
                self.stream = None
            except Exception as e:
                print(f"Error closing audio stream: {e}")
