import sounddevice as sd
import numpy as np
import threading
import time
import platform
import struct
import asyncio
from collections import deque
from scipy import signal

# Frame format:
# Magic bytes (4 bytes): 0x4D495241 ("MIRA")
# Frame type (1 byte): 
#   0x01 = Audio data
#   0x02 = End of utterance
# Frame length (4 bytes): Length of payload in bytes
# Payload: Audio data (for type 0x01) or empty (for type 0x02)

class AudioOutput:
    MAGIC_BYTES = b'MIRA'
    FRAME_TYPE_AUDIO = 0x01
    FRAME_TYPE_END = 0x02
    HEADER_SIZE = 9  # 4 bytes magic + 1 byte type + 4 bytes length

    def __init__(self):
        self.input_rate = 24000  # TTS output rate in Hz
        self.device_rate = None  # Will detect from the actual device
        self.stream = None
        self.audio_queue = deque(maxlen=1000)  # Limit queue size to prevent memory issues
        self.playing = False
        self.play_thread = None
        self.current_device = None
        self.resampler = None
        self.current_utterance = bytearray()  # More efficient for concatenation
        self.partial_frame = bytearray()  # More efficient for concatenation
        self.frame_buffer = bytearray(32768)  # Pre-allocated buffer for frame processing
        self.underflow_count = 0  # Track underflow occurrences
        self.min_buffer_size = 0  # Will be set based on device settings
        self.prebuffer_ready = False  # Track if we have enough data to start playback
        print("Audio output initialized")

    def _init_resampler(self):
        """Initialize and warm up the resampler."""
        if self.resampler is None and self.device_rate is not None and self.device_rate != self.input_rate:
            import samplerate
            # Use a faster resampling method on Windows
            if platform.system() == 'Windows':
                resampler_quality = 'linear'  # Faster, lower CPU usage
            else:
                resampler_quality = 'sinc_best'  # Higher quality for other platforms
                
            self.resampler = samplerate.Resampler(resampler_quality, channels=2)
            # Warm up the resampler with a small buffer
            warmup_data = np.zeros((512, 2), dtype=np.float32)
            ratio = self.device_rate / self.input_rate
            self.resampler.process(warmup_data, ratio)

    def _parse_frame(self, data):
        """Parse a frame from the data buffer. Returns (frame, remaining_data)."""
        if len(data) < self.HEADER_SIZE:
            return None, data

        # Check magic bytes
        if data[:4] != self.MAGIC_BYTES:
            # Try to find next magic bytes
            next_magic = data[4:].find(self.MAGIC_BYTES)
            if next_magic == -1:
                return None, b''  # Discard invalid data
            data = data[next_magic:]
            if len(data) < self.HEADER_SIZE:
                return None, data

        frame_type = data[4]
        frame_length = struct.unpack('>I', data[5:9])[0]
        total_length = self.HEADER_SIZE + frame_length

        if len(data) < total_length:
            return None, data  # Need more data

        frame_data = data[self.HEADER_SIZE:total_length]
        remaining_data = data[total_length:]
        return (frame_type, frame_data), remaining_data

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

    def _audio_callback(self, outdata, frames, time, status):
        """Callback for audio output stream."""
        try:
            if status:
                if status.output_underflow:
                    self.underflow_count += 1
                    if self.underflow_count % 10 == 0:  # Log every 10th underflow
                        print(f'[AUDIO] Output underflow ({self.underflow_count} total)')
                else:
                    print(f'[AUDIO] Status: {status}')

            # Check if we have enough data buffered
            total_buffered = sum(len(chunk) for chunk in self.audio_queue)
            if not self.prebuffer_ready:
                if total_buffered >= self.min_buffer_size:
                    self.prebuffer_ready = True
                else:
                    outdata.fill(0)
                    return

            if self.audio_queue:
                # Get the next chunk of audio data
                audio_data = self.audio_queue[0]
                frames_to_write = min(len(audio_data), frames)
                
                if frames_to_write > 0:
                    # Copy data efficiently using numpy operations
                    outdata[:frames_to_write] = audio_data[:frames_to_write]
                    if frames_to_write < frames:
                        outdata[frames_to_write:].fill(0)
                    
                    # Update or remove the processed chunk
                    remaining = audio_data[frames_to_write:]
                    if len(remaining) > 0:
                        self.audio_queue[0] = remaining
                    else:
                        self.audio_queue.popleft()
                        # Reset prebuffer if queue is getting low
                        if len(self.audio_queue) < 2:
                            self.prebuffer_ready = False
                else:
                    outdata.fill(0)
            else:
                # No data available, output silence
                outdata.fill(0)
                self.prebuffer_ready = False
        except Exception as e:
            print(f'[AUDIO] Error in callback: {e}')
            outdata.fill(0)

    def _open_stream(self, device_idx):
        """Open and start the audio output stream."""
        try:
            # Close any existing stream first
            if self.stream:
                self.stream.close()
                self.stream = None
                time.sleep(0.1)  # Give time for cleanup

            # Platform-specific optimizations
            system = platform.system()
            if system == 'Windows':
                # Windows performs better with specific buffer size and lower latency
                suggested_latency = 0.1  # 100ms latency
                blocksize = int(self.device_rate * 0.05)  # 50ms buffer size
            else:
                # Linux/macOS - use slightly larger buffers to prevent underflow
                suggested_latency = 0.05  # 50ms latency
                blocksize = int(self.device_rate * 0.02)  # 20ms buffer size
                
                # PipeWire specific adjustments
                if system == 'Linux' and hasattr(self, 'current_device') and \
                   self.current_device and 'pipewire' in self.current_device['name'].lower():
                    suggested_latency = 0.08  # 80ms latency for PipeWire
                    blocksize = int(self.device_rate * 0.04)  # 40ms buffer size

            # Set minimum buffer size for prebuffering (2x blocksize)
            self.min_buffer_size = blocksize * 2 if blocksize else int(self.device_rate * 0.1)
            self.prebuffer_ready = False
            self.underflow_count = 0

            stream_kwargs = {
                'device': device_idx,
                'samplerate': self.device_rate,
                'channels': 2,
                'dtype': np.float32,
                'latency': suggested_latency,
                'blocksize': blocksize,
                'callback': self._audio_callback,
                'finished_callback': None
            }

            print(f"Opening stream with settings: {stream_kwargs}")
            self.stream = sd.OutputStream(**stream_kwargs)
            self.stream.start()
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
        print(f"[AUDIO] Device rate: {self.device_rate} Hz, Input rate: {self.input_rate} Hz")
        self._open_stream(device_idx)
        self._init_resampler()

    async def initialize(self):
        """Initialize audio output asynchronously."""
        if self.stream and self.stream.active:
            return

        device_idx, device_info = self._find_output_device()
        self.device_rate = int(device_info['default_samplerate'])
        print(f"[AUDIO] Device rate: {self.device_rate} Hz, Input rate: {self.input_rate} Hz")
        self._open_stream(device_idx)
        self._init_resampler()

    def _process_audio_data(self, audio_bytes):
        """Process raw audio bytes into playable audio data."""
        try:
            # Convert to float32 more efficiently using preallocated buffer
            audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
            if len(audio_data) == 0:
                return None

            # Process entire chunk at once for better performance
            # Convert to float32, normalize, and make stereo in one operation
            audio_data = np.repeat(
                audio_data.astype(np.float32).reshape(-1, 1) / 32768.0,
                2, axis=1
            )

            # Resample if needed
            if self.device_rate != self.input_rate:
                ratio = self.device_rate / self.input_rate
                self._init_resampler()
                
                # Calculate output size to preallocate buffer
                output_size = int(len(audio_data) * ratio)
                if output_size > 0:
                    try:
                        audio_data = self.resampler.process(
                            audio_data,
                            ratio,
                            end_of_input=False
                        )
                    except Exception as e:
                        print(f"[AUDIO] Resampling error: {e}")
                        return None
                else:
                    return None

            return audio_data
        except Exception as e:
            print(f"[AUDIO] Error processing audio data: {e}")
            import traceback
            print(traceback.format_exc())
            return None

    def _monitor_stream(self):
        """Monitor stream health and recover if needed."""
        while self.playing:
            try:
                if not self.stream or not self.stream.active:
                    print("[AUDIO] Stream inactive, attempting recovery")
                    self._open_stream(self.current_device['index'])
                time.sleep(0.1)  # Check every 100ms
            except Exception as e:
                print(f"[AUDIO] Error monitoring stream: {e}")
                time.sleep(1)  # Back off on error

    def _process_complete_utterance(self):
        """Process and queue the complete utterance."""
        if not len(self.current_utterance):
            return

        try:
            # Process the complete utterance
            audio_data = self._process_audio_data(self.current_utterance)
            if audio_data is not None:
                # Split into smaller chunks for better buffer management
                chunk_size = int(self.device_rate * 0.1)  # 100ms chunks
                num_chunks = len(audio_data) // chunk_size + (1 if len(audio_data) % chunk_size else 0)
                
                for i in range(num_chunks):
                    start = i * chunk_size
                    end = min(start + chunk_size, len(audio_data))
                    chunk = audio_data[start:end]
                    
                    # Check queue size before adding
                    if len(self.audio_queue) < self.audio_queue.maxlen:
                        self.audio_queue.append(chunk)
                    else:
                        print("[AUDIO] Queue full, dropping audio chunk")
                        break
        except Exception as e:
            print(f"[AUDIO] Error processing complete utterance: {e}")
            import traceback
            print(traceback.format_exc())
        finally:
            self.current_utterance = bytearray()

    async def play_chunk(self, chunk):
        """Queue a chunk of audio data for playback."""
        try:
            # Pre-check for magic bytes to avoid unnecessary processing
            if not chunk.startswith(self.MAGIC_BYTES) and self.partial_frame.startswith(self.MAGIC_BYTES):
                # If we already have a valid partial frame, process it first
                await self._process_frames()
                # Then start fresh with the new chunk
                self.partial_frame = bytearray(chunk)
            else:
                # Append new data to partial frame buffer
                self.partial_frame.extend(chunk)
            
            await self._process_frames()
            
        except Exception as e:
            print(f"[AUDIO] Error in play_chunk: {e}")
            import traceback
            print(traceback.format_exc())
            # Clear buffers on error
            self.partial_frame.clear()
            self.current_utterance.clear()

    async def _process_frames(self):
        """Process complete frames from the partial frame buffer."""
        try:
            while len(self.partial_frame) >= self.HEADER_SIZE:
                # Check magic bytes
                if self.partial_frame[:4] != self.MAGIC_BYTES:
                    # Find next magic bytes efficiently
                    try:
                        next_magic = self.partial_frame.index(self.MAGIC_BYTES, 4)
                        del self.partial_frame[:next_magic]
                        continue
                    except ValueError:
                        # No magic bytes found, keep only the last 3 bytes
                        # (in case we have a partial magic bytes sequence)
                        if len(self.partial_frame) > 3:
                            self.partial_frame = self.partial_frame[-3:]
                        break

                # Parse frame header
                frame_type = self.partial_frame[4]
                frame_length = int.from_bytes(self.partial_frame[5:9], 'big')
                total_length = self.HEADER_SIZE + frame_length

                # Validate frame length
                if frame_length > 1024 * 1024:  # Max 1MB per frame
                    print("[AUDIO] Invalid frame length, discarding data")
                    self.partial_frame.clear()
                    break

                # Check if we have a complete frame
                if len(self.partial_frame) < total_length:
                    break

                # Process frame
                if frame_type == self.FRAME_TYPE_AUDIO:
                    # Extend utterance buffer directly from partial_frame view
                    self.current_utterance.extend(
                        memoryview(self.partial_frame)[self.HEADER_SIZE:total_length]
                    )
                elif frame_type == self.FRAME_TYPE_END:
                    # Process complete utterance in background
                    await asyncio.get_event_loop().run_in_executor(
                        None, self._process_complete_utterance
                    )

                # Remove processed frame efficiently
                del self.partial_frame[:total_length]

        except Exception as e:
            print(f"[AUDIO] Error in play_chunk: {e}")
            import traceback
            print(traceback.format_exc())
            # Clear buffers on error
            self.partial_frame.clear()
            self.current_utterance.clear()

    async def start_stream(self):
        """Start the audio stream and monitoring thread."""
        try:
            # Stop any existing playback
            if self.playing:
                self.playing = False
                if self.play_thread and self.play_thread.is_alive():
                    self.play_thread.join(timeout=1.0)
                self.audio_queue.clear()

            # Initialize fresh stream
            await self.initialize()

            # Start monitoring thread
            self.playing = True
            self.play_thread = threading.Thread(target=self._monitor_stream, daemon=True)
            self.play_thread.start()
            print("Started stream monitoring")

        except Exception as e:
            print(f"Error starting stream: {e}")
            # Try to recover
            self.stream = None
            await self.initialize()
            self.playing = True
            self.play_thread = threading.Thread(target=self._monitor_stream, daemon=True)
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
        self._init_resampler()  # Initialize resampler for new device rate
