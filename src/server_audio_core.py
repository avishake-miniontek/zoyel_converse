#!/usr/bin/env python3
"""
Server-side audio core that uses Silero VAD for speech detection.
This module has been modified to use time-based silence detection per configuration.
It detects speech onset, ensures the utterance meets minimum duration, adds preroll,
and signals when the complete utterance (with trailing silence removed) is ready.
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

        # Preroll and audio buffering
        self.preroll_duration = max(0.5, self.config['speech_detection']['preroll_duration'])  # Ensure minimum of 0.5s
        self.preroll_samples = int(self.DESIRED_RATE * self.preroll_duration)
        print(f"[VAD] Preroll configured for {self.preroll_duration:.1f}s ({self.preroll_samples} samples)")
        self.preroll_buffer = deque(maxlen=self.preroll_samples)
        self.audio_buffer = np.array([], dtype=np.float32)  # Accumulates audio during speech

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
        model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False
        )
        self.vad_model = model.to(self.device)
        
        # Silero VAD requires exactly 512 samples for 16kHz audio
        self.VAD_FRAME_SIZE = 512
        self._vad_buffer = []  # Store float32 samples

        # Speech detection state
        self.is_speaking = False
        self.speech_frames = 0
        self.speech_start_time = None
        self.last_speech_time = None  # Timestamp of the last detected speech frame
        self.min_speech_duration = self.config['speech_detection']['min_speech_duration']
        
        # Summary statistics
        self.stats_start_time = time.time()
        self.total_speech_frames = 0
        self.total_frames = 0
        self.last_summary_time = time.time()
        self.summary_interval = 5.0  # Output summary every 5 seconds

        # Speech detection threshold (number of consecutive speech frames needed to trigger speech start)
        self.start_speech_frames = self.config['speech_detection']['vad_settings'].get('consecutive_threshold', 2)

    def reset_buffers(self):
        """Reset audio buffers and speech state."""
        self.preroll_buffer.clear()
        self.audio_buffer = np.array([], dtype=np.float32)
        self.speech_frames = 0
        self.is_speaking = False
        self.speech_start_time = None
        self.last_speech_time = None
        self._vad_buffer = []

    def get_preroll_audio(self):
        """Retrieve pre-roll audio from the rolling buffer."""
        if len(self.preroll_buffer) > 0:
            return np.array(list(self.preroll_buffer), dtype=np.float32)
        return np.array([], dtype=np.float32)

    def process_audio(self, audio_bytes):
        """
        Process audio through Silero VAD using a time-based silence detection approach.
        
        Args:
            audio_bytes: Raw audio bytes in int16 format.
        
        Returns:
            dict: Contains processed audio data and speech detection results.
                  If an utterance is complete, 'is_complete' is True and 'audio' contains
                  the complete utterance (with preroll, and trailing silence removed).
                  Otherwise, 'audio' contains the latest audio chunk.
        """
        # Convert incoming bytes to float32 audio
        audio_data, _ = audio_utils.bytes_to_float32_audio(audio_bytes, self.DESIRED_RATE)
        # Convert to mono if needed
        audio_data = audio_utils.convert_to_mono(audio_data)
        
        # Calculate instantaneous audio levels and update tracking
        instant_rms_db, instant_peak_db = audio_utils.calculate_audio_levels(audio_data)
        now = time.time()
        self.last_update = now

        # Smoothed level tracking (fast attack, slow release)
        if instant_rms_db > self._rms_level:
            self._rms_level += 0.7 * (instant_rms_db - self._rms_level)
        else:
            self._rms_level += 0.3 * (instant_rms_db - self._rms_level)
        if instant_peak_db > self._peak_level:
            self._peak_level = instant_peak_db
        else:
            self._peak_level *= 0.95

        # Update the preroll buffer with the current audio samples
        for sample in audio_data:
            self.preroll_buffer.append(sample)

        # Ensure audio_data is a proper numpy array in float32 format and in the [-1.0, 1.0] range
        if not isinstance(audio_data, np.ndarray):
            audio_data = np.array(audio_data)
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        audio_data = np.clip(audio_data, -1.0, 1.0)

        # Append the current audio chunk to the VAD buffer (as a list of samples)
        self._vad_buffer.extend(audio_data.tolist())
        
        speech_detected = False
        frame_count = 0
        speech_frames_in_chunk = 0

        # Process complete VAD frames (each of 512 samples)
        while len(self._vad_buffer) >= self.VAD_FRAME_SIZE:
            # Extract a frame of 512 samples
            frame = np.array(self._vad_buffer[:self.VAD_FRAME_SIZE], dtype=np.float32)
            self._vad_buffer = self._vad_buffer[self.VAD_FRAME_SIZE:]
            frame_count += 1
            try:
                # Convert frame to tensor and process with Silero VAD
                tensor = torch.tensor([frame], dtype=torch.float32).to(self.device)
                with torch.no_grad():
                    speech_prob = self.vad_model(tensor, 16000).item()
                # Get speech threshold from config, default to 0.3 for backwards compatibility
                speech_threshold = self.config['speech_detection']['vad_settings'].get('threshold', 0.3)
                # Determine if the frame contains speech
                frame_is_speech = speech_prob > speech_threshold
                if frame_is_speech:
                    speech_frames_in_chunk += 1
                    self.speech_frames += 1
                    # Update the timestamp for the last detected speech
                    self.last_speech_time = time.time()
                    speech_detected = True
            except Exception as e:
                print(f"[VAD ERROR] Frame processing failed: {e}")

        if frame_count > 0:
            self.total_frames += frame_count
            self.total_speech_frames += speech_frames_in_chunk
            # Output summary every few seconds
            now = time.time()
            if now - self.last_summary_time >= self.summary_interval:
                speech_percentage = (self.total_speech_frames / max(1, self.total_frames)) * 100
                print(f"[VAD SUMMARY] Past {self.summary_interval:.1f}s: "
                      f"Speech detected in {speech_percentage:.1f}% of frames, "
                      f"Current level: {self._rms_level:.1f}dB")
                # Reset summary counters
                self.total_frames = 0
                self.total_speech_frames = 0
                self.last_summary_time = now

        audio_ready = False
        speech_duration = 0
        final_audio = None

        # --- State Machine Using Time-Based Silence Detection ---
        if not self.is_speaking:
            # If speech is detected in this chunk and enough frames indicate a start
            if speech_detected and self.speech_frames >= self.start_speech_frames:
                print("[VAD] Speech start detected")
                self.is_speaking = True
                self.speech_start_time = time.time()
                self.last_speech_time = time.time()
                # Retrieve preroll audio and begin accumulating the utterance
                preroll = self.get_preroll_audio()
                if len(preroll) > 0:
                    print(f"[VAD] Adding {len(preroll)/self.DESIRED_RATE:.3f}s of preroll audio")
                    self.audio_buffer = preroll
                else:
                    print("[VAD] No preroll audio available")
                    self.audio_buffer = np.array([], dtype=np.float32)
                self.audio_buffer = np.concatenate([self.audio_buffer, audio_data])
        else:
            # Already speaking: continue accumulating audio
            self.audio_buffer = np.concatenate([self.audio_buffer, audio_data])
            # If no speech has been detected recently, check the time elapsed since the last speech frame
            if self.last_speech_time is not None:
                time_since_last_speech = time.time() - self.last_speech_time
                if time_since_last_speech >= self.config['speech_detection']['end_silence_duration']:
                    # Only complete the utterance if it meets the minimum speech duration requirement
                    utterance_duration = time.time() - self.speech_start_time
                    if utterance_duration >= self.config['speech_detection']['min_speech_duration']:
                        print(f"[VAD] Speech end detected after {time_since_last_speech:.2f}s of silence")
                        audio_ready = True
                        speech_duration = utterance_duration
                        # Remove trailing silence from the accumulated audio
                        silence_samples = int(time_since_last_speech * self.DESIRED_RATE)
                        speech_end_idx = len(self.audio_buffer) - silence_samples
                        final_audio = self.audio_buffer[:speech_end_idx].copy()
                        self.is_speaking = False
                        self.reset_buffers()

        # Prepare the result.
        # If an utterance is complete, return the full audio (with preroll and without trailing silence);
        # otherwise, return the current audio chunk.
        result = {
            'db_level': self._rms_level,
            'noise_floor': self._noise_floor,
            'peak_level': self._peak_level,
            'last_speech_level': self._peak_level,
            'is_complete': audio_ready,
            'is_speech': self.is_speaking,
            'speech_duration': speech_duration,
            'audio': final_audio if audio_ready else audio_data
        }

        return result

    def close(self):
        """Clean up resources."""
        self._vad_buffer = []
        self.preroll_buffer.clear()
        self.audio_buffer = np.array([], dtype=np.float32)

if __name__ == "__main__":
    # For testing purposes: you can run this module independently.
    print("ServerAudioCore module loaded.")
