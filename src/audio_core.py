"""
Audio core that uses Py-WebRTCVad for speech detection
instead of manual floor/ratio gating.

We keep:
- init_audio_device() for device selection & optional 'calibration'
- process_audio() for chunk-based reading

BUT:
- The actual "is_speech" gating is done by WebRTC VAD in 20ms frames.
- We keep a short "hangover" (end_silence_frames) so we don't cut speech abruptly.
- We maintain a preroll buffer to catch the start of speech.
"""

import json
import numpy as np
import time
from collections import deque
import sounddevice as sd
import webrtcvad  # <-- New dependency
import platform
import warnings

class AudioCore:
    def __init__(self):
        with open('config.json', 'r') as f:
            self.config = json.load(f)

        # Basic audio properties
        self.stream = None
        self.rate = None
        self.needs_resampling = None
        self.CHUNK = self.config['audio_processing']['chunk_size']
        self.CHANNELS = 1
        self.DESIRED_RATE = self.config['audio_processing']['desired_rate']

        # We'll still track some "floor" for your GUI, but it's not used by the VAD gating
        self._noise_floor = -96.0
        self._min_floor   = -96.0
        self._max_floor   = -36.0
        self._rms_level   = -96.0
        self._peak_level  = -96.0
        self.last_update  = time.time()
        self.debug_counter = 0

        # This "calibration" is mostly for display logs
        self.calibrated_floor = None

        # --------------- NEW: WebRTC VAD Setup ---------------
        # Create a WebRTC VAD instance. Mode can be 0-3 (3 = most aggressive).
        vad_mode = self.config['speech_detection']['vad_settings']['mode']
        self.vad = webrtcvad.Vad(mode=vad_mode)

        # We'll feed the VAD frames at 16kHz, 16-bit mono
        self.VAD_FRAME_MS = self.config['speech_detection']['vad_settings']['frame_duration_ms']
        self.VAD_FRAME_SIZE = int((self.VAD_FRAME_MS / 1000.0) * self.DESIRED_RATE)  # Convert ms to seconds
        self._vad_buffer = bytearray()  # Holds leftover PCM between calls

        # Preroll and speech detection state
        self.preroll_duration = self.config['speech_detection']['preroll_duration']
        self.preroll_samples = int(self.DESIRED_RATE * self.preroll_duration)
        self.preroll_buffer = deque(maxlen=self.preroll_samples)
        self.audio_buffer = []  # Accumulates audio during speech
        self.last_speech_end = 0  # timestamp of last speech end
        self.last_speech_level = -96.0  # Track speech level for better gating

        # State machine for speech detection
        self.is_speaking = False
        self.speech_frames = 0
        self.silence_frames = 0
        self.speech_start_time = None
        self.min_speech_duration = self.config['speech_detection']['min_speech_duration']
        self.was_speech = False  # Track if we were speaking in previous chunk

        # Speech detection thresholds
        self.start_speech_frames = 2  # must see 'speech' in 2+ consecutive frames
        self.end_silence_frames = int(
            self.config['speech_detection']['end_silence_duration'] / (self.VAD_FRAME_MS / 1000.0)
        )
        # e.g. if end_silence_duration=0.8, 0.8 / 0.02 = 40 frames

        # NEW: For real-time VAD visualization
        self.consecutive_vad_speech = 0
        self.vad_speech_threshold = 2  # Number of consecutive speech frames needed
        self._vad_visualization_buffer = bytearray()

    # ----------------------------------------------------------------
    # Optional: Keep your old floor logic for logging or GUI meter
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
    # Audio Device Initialization & Optional "Calibration"
    # ----------------------------------------------------------------
    def init_audio_device(self):
        """
        Initialize audio device. We might do a short "calibration" for your GUI's floor display,
        but we won't rely on it for gating. The gating is handled by WebRTC VAD.
        """
        try:
            print("\nListing audio devices:")
            print(sd.query_devices())

            system = platform.system().lower()
            if system == 'linux':
                warnings.filterwarnings("ignore", category=RuntimeWarning, module="sounddevice")

            devices = sd.query_devices()
            working_device = None
            device_info = None

            # Check if audio_devices is present in config
            if 'audio_devices' in self.config and 'input_device' in self.config['audio_devices']:
                input_device = self.config['audio_devices']['input_device']
                # Handle both numeric index and name string
                if isinstance(input_device, (int, float)):
                    device_idx = int(input_device)
                    if 0 <= device_idx < len(devices):
                        device_info = devices[device_idx]
                        working_device = device_idx
                else:
                    # Try matching by name
                    for i, device in enumerate(devices):
                        if device['name'] == input_device:
                            working_device = i
                            device_info = device
                            break
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

            # If you still want a "floor" for the GUI, do a short capture (2s)
            # We won't rely on it for gating.
            calibration_duration = 2.0
            frames_needed = int(rate * calibration_duration)
            print(f"\nCalibrating GUI floor for {calibration_duration}s... (not used by VAD)")
            audio_buffer = sd.rec(frames_needed, samplerate=rate,
                                  channels=1, dtype='float32')
            sd.wait()
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

            print(f"  GUI floor set to: {self.calibrated_floor:.1f} dB (not used by VAD)")

            # Open the main stream for continuous capture
            print("\nOpening main input stream...")
            stream = sd.InputStream(
                device=working_device,
                channels=1,
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
    # Convert from float -> PCM int16 for VAD, etc.
    # ----------------------------------------------------------------
    def bytes_to_float32_audio(self, audio_data, sample_rate=None):
        """
        Convert int16-encoded bytes to float32 samples in [-1..1].
        This is typically what your client is sending/receiving.
        """
        audio_int16 = np.frombuffer(audio_data, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        return audio_float32, (sample_rate if sample_rate is not None else self.DESIRED_RATE)

    # ----------------------------------------------------------------
    # NEW: Real-time VAD visualization function
    # ----------------------------------------------------------------
    def process_audio_vad(self, audio_data):
        """
        Process audio purely for VAD visualization without utterance logic.
        This function focuses on immediate speech detection for the GUI.
        
        Args:
            audio_data: float32 array of audio samples
            
        Returns:
            bool: True if speech is detected in this frame
        """
        # Convert float32 to int16 for WebRTC VAD
        int16_data = np.clip(audio_data * 32767.0, -32767, 32767).astype(np.int16)
        pcm_bytes = int16_data.tobytes()
        self._vad_visualization_buffer.extend(pcm_bytes)
        
        # Process VAD frames
        is_speech_frame = False
        
        while len(self._vad_visualization_buffer) >= (self.VAD_FRAME_SIZE * 2):
            frame = self._vad_visualization_buffer[:(self.VAD_FRAME_SIZE * 2)]
            self._vad_visualization_buffer = self._vad_visualization_buffer[(self.VAD_FRAME_SIZE * 2):]
            
            try:
                frame_is_speech = self.vad.is_speech(frame, sample_rate=16000)
                if frame_is_speech:
                    self.consecutive_vad_speech += 1
                    if self.consecutive_vad_speech >= self.vad_speech_threshold:
                        is_speech_frame = True
                else:
                    self.consecutive_vad_speech = max(0, self.consecutive_vad_speech - 1)
            except Exception as e:
                self.consecutive_vad_speech = 0
        
        # Return true if we saw enough consecutive speech frames
        return is_speech_frame or self.consecutive_vad_speech >= self.vad_speech_threshold

    # ----------------------------------------------------------------
    # The "process_audio" function:
    # 1) Takes a float32 buffer
    # 2) Convert to 16-bit PCM
    # 3) Break into 20ms frames for WebRTC VAD
    # 4) Update speech state
    # 5) Return 'is_speech' (the overall gate state)
    # ----------------------------------------------------------------
    def reset_preroll(self):
        """Reset the preroll buffer and speech state"""
        self.preroll_buffer.clear()
        self.audio_buffer = []
        self.speech_frames = 0
        self.silence_frames = 0
        self.is_speaking = False
        self.speech_start_time = None
        self._vad_buffer = bytearray()

    def get_preroll_audio(self):
        """Get pre-roll audio if we've had enough silence to reset."""
        now = time.time()
        if now - self.last_speech_end > self.preroll_duration:
            # Convert deque to list, excluding most recent chunk
            preroll = list(self.preroll_buffer)[:-int(self.DESIRED_RATE * 0.1)]  # Exclude last 100ms
            return preroll if len(preroll) > 0 else []
        return []

    def process_audio(self, audio_data):
        """
        Process audio through WebRTC VAD with improved preroll and state handling.
        Returns dict with audio data and speech detection results.
        """
        # Calculate RMS and update levels for GUI
        if len(audio_data) > 0:
            block_rms = np.sqrt(np.mean(audio_data**2))
            instant_rms_db = 20.0 * np.log10(max(block_rms, 1e-10))

            now = time.time()
            self.last_update = now

            # Smoothed level tracking with faster attack, slower release
            if instant_rms_db > self.rms_level:
                self.rms_level += 0.7 * (instant_rms_db - self.rms_level)
            else:
                self.rms_level += 0.3 * (instant_rms_db - self.rms_level)
            
            if instant_rms_db > self.peak_level:
                self.peak_level = instant_rms_db
            else:
                self.peak_level *= 0.95  # Slightly faster decay

        # Add to preroll buffer after level calculations
        for sample in audio_data:
            self.preroll_buffer.append(sample)

        # Convert float32 to int16 for WebRTC VAD
        int16_data = np.clip(audio_data * 32767.0, -32767, 32767).astype(np.int16)
        pcm_bytes = int16_data.tobytes()
        self._vad_buffer.extend(pcm_bytes)

        # Process VAD frames
        speech_detected = False
        consecutive_speech = 0
        frame_count = 0

        while len(self._vad_buffer) >= (self.VAD_FRAME_SIZE * 2):
            frame = self._vad_buffer[:(self.VAD_FRAME_SIZE * 2)]
            self._vad_buffer = self._vad_buffer[(self.VAD_FRAME_SIZE * 2):]
            frame_count += 1

            try:
                frame_is_speech = self.vad.is_speech(frame, sample_rate=16000)
            except Exception as e:
                frame_is_speech = False

            if frame_is_speech:
                consecutive_speech += 1
                self.speech_frames += 1
                self.silence_frames = 0
                if consecutive_speech >= 2:  # Need at least 2 consecutive speech frames
                    speech_detected = True
            else:
                consecutive_speech = 0
                self.silence_frames += 1
                self.speech_frames = max(0, self.speech_frames - 1)  # Gradual decrease

        # State machine update with hysteresis
        audio_ready = False  # Flag to indicate if we should run ASR
        final_audio = None

        if not self.is_speaking:
            if speech_detected and self.speech_frames >= self.start_speech_frames:
                # Speech start
                self.is_speaking = True
                self.speech_start_time = time.time()
                self.last_speech_level = self.rms_level
                
                # Get preroll if enough silence has passed
                preroll = self.get_preroll_audio()
                if preroll:
                    self.audio_buffer = preroll
                else:
                    self.audio_buffer = []
                
                # Add current chunk
                self.audio_buffer.extend(audio_data)
                
        else:  # Currently speaking
            if speech_detected or self.silence_frames < self.end_silence_frames:
                # Continue speech
                self.audio_buffer.extend(audio_data)
                if speech_detected:
                    self.last_speech_level = max(self.last_speech_level, self.rms_level)
            else:
                # Speech end - capture the complete utterance
                audio_ready = True
                final_audio = np.array(self.audio_buffer)  # Get the complete utterance
                speech_duration = time.time() - self.speech_start_time
                
                # Mark speech end but keep state until after ASR
                self.is_speaking = False
                self.last_speech_end = time.time()

        # Prepare base result
        result = {
            'db_level': self.rms_level,
            'noise_floor': self.noise_floor,
            'peak_level': self.peak_level,
            'last_speech_level': self.last_speech_level,
            'is_complete': False,
            'is_speech': False,
            'speech_duration': 0
        }

        if audio_ready and final_audio is not None and speech_duration > 0:
            # We have a complete utterance
            is_valid_speech = speech_duration >= self.min_speech_duration
            if is_valid_speech:
                result.update({
                    'audio': final_audio,
                    'is_speech': True,
                    'speech_duration': speech_duration,
                    'is_complete': True
                })
            
            # Reset state after capturing everything we need
            self.speech_start_time = None
            self.reset_preroll()
        else:
            # During speech or silence
            result.update({
                'audio': audio_data
            })

        return result

    # ----------------------------------------------------------------
    # If the server or client needs to close resources
    # ----------------------------------------------------------------
    def close(self):
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
                self.stream = None
            except Exception as e:
                print(f"Error closing audio stream: {e}")
