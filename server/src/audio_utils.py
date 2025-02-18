"""
Shared audio processing utilities used by both client and server audio cores.
"""

import numpy as np

def bytes_to_float32_audio(audio_bytes, sample_rate=16000):
    """
    Convert int16-encoded bytes to float32 samples in [-1..1] range.
    Used for converting audio data between client and server.
    
    Args:
        audio_bytes: Raw audio bytes in int16 format
        sample_rate: Sample rate of the audio (default: 16000)
        
    Returns:
        tuple: (float32 numpy array, sample rate)
    """
    # Convert to int16
    audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
    
    # Convert to float32 and normalize to [-1.0, 1.0]
    audio_float32 = audio_int16.astype(np.float32) / 32768.0
    
    # Remove DC offset
    audio_float32 = audio_float32 - np.mean(audio_float32)
    
    return audio_float32, sample_rate

def convert_to_mono(audio_data):
    """
    Convert multi-channel audio to mono by averaging all channels.
    This provides better quality audio input for processing.
    
    Args:
        audio_data: numpy array of shape (frames, channels) or (frames,) for mono
        
    Returns:
        numpy array of shape (frames,) containing mono audio averaged from all channels
    """
    if len(audio_data.shape) == 2 and audio_data.shape[1] > 1:
        return np.mean(audio_data, axis=1)  # Average all channels
    return audio_data

def calculate_audio_levels(audio_data):
    """
    Calculate RMS and peak levels from audio data.
    
    Args:
        audio_data: float32 array of audio samples
        
    Returns:
        tuple: (rms_db, peak_db)
    """
    if len(audio_data) > 0:
        block_rms = np.sqrt(np.mean(audio_data**2))
        rms_db = 20.0 * np.log10(max(block_rms, 1e-10))
        peak_db = 20.0 * np.log10(max(abs(max(audio_data)), abs(min(audio_data)), 1e-10))
        return rms_db, peak_db
    return -96.0, -96.0  # Default values if no audio
