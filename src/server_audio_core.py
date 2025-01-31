"""
Server-side audio core that uses Silero VAD for speech detection.
This handles the VAD processing and speech detection on the server side.
"""

import json
import numpy as np
import time
from collections import deque
import torch
from src import audio_utils

class ServerAudioCore:
    def __init__(self):
        with open('config.json', 'r') as f:
            self.config = json.load(f)

        # Basic audio properties
        self.DESIRED_RATE = self.config['audio_processing']['desired_rate']

        # Level tracking for analysis
        self._noise_floor = -96.0
        self._min_floor   = -96.0
        self._max_floor   = -36.0
        self._rms_level   = -96.0
        self._peak_level  = -96.0
        self.last_update  = time.time()

        # --------------- Silero VAD Setup ---------------
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f"[VAD] Using device: {self.device}")
        
        # Load Silero VAD model
        model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                model='silero_vad',
                                force_reload=False)
        self.vad_model = model.to(self.device)
        
        # Silero VAD requires exactly 512 samples for 16kHz audio
        self.VAD_FRAME_SIZE = 512
        self._vad_buffer = []  # Store float32 samples

        # Speech detection state
        self.is_speaking = False
        self.speech_frames = 0
        self.silence_frames = 0
        self.speech_start_time = None
        self.min_speech_duration = self.config['speech_detection']['min_speech_duration']
        self.last_speech_end = 0
        self.last_speech_level = -96.0

        # Speech detection thresholds
        self.start_speech_frames = self.config['speech_detection']['vad_settings'].get('consecutive_threshold', 2)
        # Calculate end silence frames based on duration and sample rate
        # For 16kHz audio with 512 samples per frame, each frame is 32ms
        frame_duration_ms = (self.VAD_FRAME_SIZE / self.DESIRED_RATE) * 1000
        self.end_silence_frames = int(
            self.config['speech_detection']['end_silence_duration'] / (frame_duration_ms / 1000.0)
        )

        # For real-time VAD visualization
        self.consecutive_vad_speech = 0
        self.vad_speech_threshold = self.config['speech_detection']['vad_settings'].get('consecutive_threshold', 2)

    def process_audio(self, audio_bytes):
        """
        Process audio through Silero VAD.
        
        Args:
            audio_data: float32 array of audio samples (mono, 16kHz)
            
        Returns:
            dict: Contains processed audio data and speech detection results
        """
        # Convert incoming bytes to float32 audio
        audio_data, _ = audio_utils.bytes_to_float32_audio(audio_bytes, self.DESIRED_RATE)
        
        # Convert to mono if needed
        audio_data = audio_utils.convert_to_mono(audio_data)
        
        # Calculate levels and update tracking
        instant_rms_db, instant_peak_db = audio_utils.calculate_audio_levels(audio_data)
        
        now = time.time()
        self.last_update = now

        # Smoothed level tracking with faster attack, slower release
        if instant_rms_db > self._rms_level:
            self._rms_level += 0.7 * (instant_rms_db - self._rms_level)
        else:
            self._rms_level += 0.3 * (instant_rms_db - self._rms_level)
        
        if instant_peak_db > self._peak_level:
            self._peak_level = instant_peak_db
        else:
            self._peak_level *= 0.95

        # Process audio through Silero VAD
        self._vad_buffer.extend(audio_data.tolist())
        
        speech_detected = False
        consecutive_speech = 0
        frame_count = 0
        speech_frames = 0

        print(f"[VAD DEBUG] Processing buffer size: {len(self._vad_buffer)}")

        # Process complete VAD frames (512 samples each)
        while len(self._vad_buffer) >= self.VAD_FRAME_SIZE:
            # Extract exactly 512 samples
            frame = np.array(self._vad_buffer[:self.VAD_FRAME_SIZE])
            self._vad_buffer = self._vad_buffer[self.VAD_FRAME_SIZE:]
            frame_count += 1
            
            try:
                # Convert to tensor
                tensor = torch.tensor([frame], dtype=torch.float32).to(self.device)
                
                # Get speech probability
                with torch.no_grad():
                    speech_prob = self.vad_model(tensor, 16000).item()
                
                # Lower threshold to 0.3 to be more sensitive to speech
                frame_is_speech = speech_prob > 0.08
                print(f"[VAD DEBUG] Speech probability: {speech_prob:.3f}")
                
                if frame_is_speech:
                    speech_frames += 1
                    consecutive_speech += 1
                    self.speech_frames += 1
                    self.silence_frames = 0
                    if consecutive_speech >= self.vad_speech_threshold:
                        speech_detected = True
                else:
                    consecutive_speech = 0
                    self.silence_frames += 1
                    self.speech_frames = max(0, self.speech_frames - 1)
            except Exception as e:
                print(f"[VAD ERROR] Frame processing failed: {e}")
                frame_is_speech = False

        if frame_count > 0:
            print(f"[VAD DEBUG] Processed {frame_count} frames, {speech_frames} had speech")

        # State machine update with hysteresis
        audio_ready = False
        speech_duration = 0

        if not self.is_speaking:
            if speech_detected and self.speech_frames >= self.start_speech_frames:
                # Speech start
                self.is_speaking = True
                self.speech_start_time = time.time()
                self.last_speech_level = self._rms_level
        else:  # Currently speaking
            if speech_detected or self.silence_frames < self.end_silence_frames:
                # Continue speech
                if speech_detected:
                    self.last_speech_level = max(self.last_speech_level, self._rms_level)
            else:
                # Speech end
                audio_ready = True
                speech_duration = time.time() - self.speech_start_time
                self.is_speaking = False
                self.last_speech_end = time.time()

        # Prepare result
        result = {
            'db_level': self._rms_level,
            'noise_floor': self._noise_floor,
            'peak_level': self._peak_level,
            'last_speech_level': self.last_speech_level,
            'is_complete': audio_ready,
            'is_speech': self.is_speaking or speech_detected,
            'speech_duration': speech_duration,
            'audio': audio_data
        }

        return result

    def close(self):
        """Clean up resources."""
        self._vad_buffer = []
