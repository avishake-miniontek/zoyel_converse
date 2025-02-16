#!/usr/bin/env python3
"""
Server-side audio core that uses Silero VAD for speech detection.
This module uses time-based silence detection to decide when an utterance is complete.
It also reports real-time voice detection data to the client, decoupled from the
utterance finalization logic.
"""

import json
import numpy as np
import time
from collections import deque
import torch
from src import audio_utils
import logging

logger = logging.getLogger(__name__)

class ServerAudioCore:
    def __init__(self):
        # Load configuration from config.json
        with open('config.json', 'r') as f:
            self.config = json.load(f)
            
        self.DESIRED_RATE = self.config['audio_processing']['desired_rate']
        # Use the greater of 0.5 or the configured preroll duration
        self.preroll_duration = max(0.5, self.config['speech_detection'].get('preroll_duration', 0.5))
        self.preroll_samples = int(self.DESIRED_RATE * self.preroll_duration)
        print(f"[VAD] Preroll configured for {self.preroll_duration:.1f}s ({self.preroll_samples} samples)")
        self.preroll_buffer = deque(maxlen=self.preroll_samples)
        self.audio_buffer = np.array([], dtype=np.float32)
        
        # Audio level tracking
        self._noise_floor = -96.0
        self._min_floor   = -96.0
        self._max_floor   = -36.0
        self._rms_level   = -96.0
        self._peak_level  = -96.0
        self.last_update  = time.time()
        
        self.calibrated_floor = None
        
        # ---------------- Silero VAD Setup ----------------
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f"[VAD] Using device: {self.device}")
        model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False
        )
        self.vad_model = model.to(self.device)
        # Silero VAD requires frames of 512 samples for 16kHz audio
        self.VAD_FRAME_SIZE = 512
        self._vad_buffer = []
        
        # Speech detection state
        self.is_speaking = False
        self.speech_frames = 0
        self.speech_start_time = None
        self.last_speech_time = None
        self.min_speech_duration = self.config['speech_detection']['min_speech_duration']
        
        # Summary statistics (for logging)
        self.stats_start_time = time.time()
        self.total_speech_frames = 0
        self.total_frames = 0
        self.last_summary_time = time.time()
        self.summary_interval = 5.0
        # Number of consecutive speech frames needed to trigger speech start
        self.start_speech_frames = self.config['speech_detection']['vad_settings'].get('consecutive_threshold', 2)
        
    def reset_buffers(self):
        """Reset all buffers and speech state."""
        self.preroll_buffer.clear()
        self.audio_buffer = np.array([], dtype=np.float32)
        self.speech_frames = 0
        self.is_speaking = False
        self.speech_start_time = None
        self.last_speech_time = None
        self._vad_buffer = []
        
    def get_preroll_audio(self):
        """Return preroll audio from the rolling buffer."""
        if len(self.preroll_buffer) > 0:
            return np.array(list(self.preroll_buffer), dtype=np.float32)
        return np.array([], dtype=np.float32)
    
    def process_audio(self, audio_bytes):
        """
        Process incoming audio bytes through VAD and update internal state.
        Returns a dict containing current audio levels and a flag 'is_speech'
        for display purposes (updated immediately) and other data used for utterance
        finalization.
        """
        # Convert incoming bytes to float32 array at desired rate
        audio_data, _ = audio_utils.bytes_to_float32_audio(audio_bytes, self.DESIRED_RATE)
        audio_data = audio_utils.convert_to_mono(audio_data)
        
        # Update preroll buffer sample by sample
        for sample in audio_data:
            self.preroll_buffer.append(sample)
            
        # Ensure proper format and clip audio data
        if not isinstance(audio_data, np.ndarray):
            audio_data = np.array(audio_data)
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        audio_data = np.clip(audio_data, -1.0, 1.0)
        
        # Append audio data to VAD buffer (as list of samples)
        self._vad_buffer.extend(audio_data.tolist())
        
        speech_detected = False
        frame_count = 0
        speech_frames_in_chunk = 0
        # Process complete frames from the VAD buffer
        while len(self._vad_buffer) >= self.VAD_FRAME_SIZE:
            frame = np.array(self._vad_buffer[:self.VAD_FRAME_SIZE], dtype=np.float32)
            self._vad_buffer = self._vad_buffer[self.VAD_FRAME_SIZE:]
            frame_count += 1
            try:
                tensor = torch.tensor([frame], dtype=torch.float32).to(self.device)
                with torch.no_grad():
                    speech_prob = self.vad_model(tensor, 16000).item()
                speech_threshold = self.config['speech_detection']['vad_settings'].get('threshold', 0.3)
                if speech_prob > speech_threshold:
                    speech_frames_in_chunk += 1
                    self.speech_frames += 1
                    self.last_speech_time = time.time()
                    speech_detected = True
            except Exception as e:
                print(f"[VAD ERROR] Frame processing failed: {e}")
        
        # Update RMS and peak levels for display
        instant_rms = np.sqrt(np.mean(audio_data**2))
        instant_rms_db = 20.0 * np.log10(max(instant_rms, 1e-10))
        instant_peak_db = 20.0 * np.log10(max(np.max(np.abs(audio_data)), 1e-10))
        now = time.time()
        self.last_update = now
        if instant_rms_db > self._rms_level:
            self._rms_level += 0.7 * (instant_rms_db - self._rms_level)
        else:
            self._rms_level += 0.3 * (instant_rms_db - self._rms_level)
        if instant_peak_db > self._peak_level:
            self._peak_level = instant_peak_db
        else:
            self._peak_level *= 0.95
        
        # For display purposes, use the immediate result from this chunk.
        display_is_speech = speech_detected
        
        # If speech has just started (based on consecutive frames)
        if not self.is_speaking and speech_detected and self.speech_frames >= self.start_speech_frames:
            print("[VAD] Speech start detected")
            self.is_speaking = True
            self.speech_start_time = time.time()
            self.last_speech_time = time.time()
            preroll = self.get_preroll_audio()
            if len(preroll) > 0:
                print(f"[VAD] Adding {len(preroll)/self.DESIRED_RATE:.3f}s of preroll audio")
                self.audio_buffer = preroll
            else:
                print("[VAD] No preroll audio available")
                self.audio_buffer = np.array([], dtype=np.float32)
            self.audio_buffer = np.concatenate([self.audio_buffer, audio_data])
        elif self.is_speaking:
            self.audio_buffer = np.concatenate([self.audio_buffer, audio_data])
            # Check if silence has lasted long enough to mark utterance end
            if self.last_speech_time is not None:
                time_since_last_speech = time.time() - self.last_speech_time
                if time_since_last_speech >= self.config['speech_detection']['end_silence_duration']:
                    utterance_duration = time.time() - self.speech_start_time
                    if utterance_duration >= self.config['speech_detection']['min_speech_duration']:
                        print(f"[VAD] Speech end detected after {time_since_last_speech:.2f}s of silence")
                        # Trim trailing silence based on elapsed time
                        silence_samples = int(time_since_last_speech * self.DESIRED_RATE)
                        speech_end_idx = len(self.audio_buffer) - silence_samples
                        final_audio = self.audio_buffer[:speech_end_idx].copy()
                        self.reset_buffers()
                        return {
                            'db_level': self._rms_level,
                            'noise_floor': self._noise_floor,
                            'peak_level': self._peak_level,
                            'last_speech_level': self._peak_level,
                            'is_complete': True,
                            # For display, we report the immediate voice detection data
                            'is_speech': display_is_speech,
                            'speech_duration': utterance_duration,
                            'audio': final_audio
                        }
        # Return current data for display purposes even if utterance is not complete
        current_duration = time.time() - (self.speech_start_time if self.speech_start_time else time.time())
        return {
            'db_level': self._rms_level,
            'noise_floor': self._noise_floor,
            'peak_level': self._peak_level,
            'last_speech_level': self._peak_level,
            'is_complete': False,
            'is_speech': display_is_speech,
            'speech_duration': current_duration,
            'audio': audio_data
        }
    
    def close(self):
        """Clean up resources."""
        self._vad_buffer = []
        self.preroll_buffer.clear()
        self.audio_buffer = np.array([], dtype=np.float32)

if __name__ == "__main__":
    print("ServerAudioCore module loaded.")
