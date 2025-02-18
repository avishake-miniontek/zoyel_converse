"""
Headless version of the audio interface that implements the same interface as graphical_interface
but without any GUI components. This is useful for devices that don't have display capabilities
or when running in CLI-only mode.
"""

import queue
from typing import Optional, Callable

class HeadlessAudioInterface:
    def __init__(self,
                 input_device_name: Optional[str] = None,
                 output_device_name: Optional[str] = None,
                 on_input_change: Optional[Callable[[str], None]] = None,
                 on_output_change: Optional[Callable[[str], None]] = None):
        """
        Initialize the headless audio interface that mimics the GUI interface
        but operates without any visual components.
        """
        self.running = True
        self.has_gui = False
        
        # Store device names
        self.input_device_name = input_device_name or "No device selected"
        self.output_device_name = output_device_name or "No device selected"
        self.on_input_change = on_input_change
        self.on_output_change = on_output_change

        # Maintain queues for compatibility
        self.input_device_queue = queue.Queue()
        self.output_device_queue = queue.Queue()
        self.text_input_queue = queue.Queue()
        self.speech_queue = queue.Queue()

        print("\nHeadless audio interface initialized")

    def process_vad(self, is_speech: bool):
        """
        Process voice activity detection state.
        In headless mode, we just print the state change.
        """
        print(f"\rSpeech detected: {'Yes' if is_speech else 'No'}    ", end='', flush=True)

    def get_text_input(self):
        """
        Thread-safe method to get text input from queue.
        Returns None if queue is empty.
        """
        try:
            text = self.text_input_queue.get_nowait()
            self.text_input_queue.task_done()
            return text
        except queue.Empty:
            return None

    def update(self):
        """
        Update method for compatibility with GUI version.
        Does nothing in headless mode.
        """
        pass

    def close(self):
        """
        Clean up resources.
        """
        self.running = False
        print("\nHeadless audio interface closed")
