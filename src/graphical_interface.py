"""
Tkinter-based interface that displays a rolling "speech detection" visualization
instead of a microphone input level bar.

We assume:
 - The rest of your code calls audio_interface.process_vad(is_speech)
   whenever a new chunk is processed.
 - We keep the old device selection combo boxes if you still want them.
"""

import tkinter as tk
from tkinter import ttk
import queue
import sounddevice as sd
import numpy as np
import time
import json
from typing import Optional, List, Callable

# Load config
with open('config.json', 'r') as f:
    CONFIG = json.load(f)

class AudioInterface:
    def __init__(self,
                 input_device_name: Optional[str] = None,
                 output_device_name: Optional[str] = None,
                 on_input_change: Optional[Callable[[str], None]] = None,
                 on_output_change: Optional[Callable[[str], None]] = None):
        """
        Initialize the Tkinter-based audio interface for speech detection display.
        """
        # GUI run state
        self.running = True
        self.has_gui = True

        # Device names in combo boxes
        self.input_device_name = input_device_name or "No device selected"
        self.output_device_name = output_device_name or "No device selected"
        self.on_input_change = on_input_change
        self.on_output_change = on_output_change

        # Callback for text input submissions (used by client for processing text)
        self.on_text_change: Optional[Callable[[str], None]] = None

        # Queues for thread-safe updates (used if a callback is not set)
        self.input_device_queue = queue.Queue()
        self.output_device_queue = queue.Queue()

        # --------------------------------------------------------------------
        # NEW: We'll store the last N speech states in a list for a rolling
        #      timeline. For instance, 50 items ~ last 5 seconds if we update
        #      10 times/sec. Adjust as needed.
        # --------------------------------------------------------------------
        self.max_points = 50
        self.speech_history = [False] * self.max_points

        # We also store a queue for new speech states from the main thread.
        self.speech_queue = queue.Queue()

        # Attempt to build the GUI
        try:
            self._init_gui()
            print("\nSpeech detection interface opened successfully")
        except Exception as e:
            print(f"\nWarning: Could not create GUI window ({e})")
            self.running = False
            self.has_gui = False

    def set_text_callback(self, callback: Callable[[str], None]):
        """
        Register a callback to be invoked when text is submitted from the GUI.
        """
        self.on_text_change = callback

    def _init_gui(self):
        """Initialize the Tkinter GUI."""
        self.root = tk.Tk()
        self.root.title("Speech Detection")
        self.root.geometry("600x450")  # Increased width and height for larger text area
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(2, weight=1)  # Allow speech display frame to expand vertically

        # Create frames for device selection (optional) and the speech display
        self._create_input_device_frame()
        self._create_output_device_frame()
        self._create_speech_display_frame()
        self._create_text_input_frame()  # Add text input frame

        # Set up periodic updates
        self._schedule_updates()

        # Queue for text input processing (fallback if no callback is set)
        self.text_input_queue = queue.Queue()

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _create_input_device_frame(self):
        """Create frame for input device selection."""
        input_frame = ttk.LabelFrame(self.root, text="Input Device", padding=5)
        input_frame.grid(row=0, column=0, padx=10, pady=5, sticky="ew")

        # Gather list of input devices
        devices = self._get_input_devices()

        self.input_device_var = tk.StringVar(value=self.input_device_name)
        self.input_device_combo = ttk.Combobox(
            input_frame,
            textvariable=self.input_device_var,
            values=devices,
            state="readonly",
            width=40
        )
        self.input_device_combo.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.input_device_combo.bind('<<ComboboxSelected>>', self._on_input_device_change)

    def _create_output_device_frame(self):
        """Create frame for output device selection."""
        output_frame = ttk.LabelFrame(self.root, text="Output Device", padding=5)
        output_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")

        # Gather list of output devices
        devices = self._get_output_devices()

        self.output_device_var = tk.StringVar(value=self.output_device_name)
        self.output_device_combo = ttk.Combobox(
            output_frame,
            textvariable=self.output_device_var,
            values=devices,
            state="readonly",
            width=40
        )
        self.output_device_combo.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.output_device_combo.bind('<<ComboboxSelected>>', self._on_output_device_change)

    def _create_speech_display_frame(self):
        """
        Create frame for the rolling speech detection display.
        We'll use a Canvas to draw rectangles for "speech" or "silence."
        """
        display_frame = ttk.LabelFrame(self.root, text="Possible Speech Detection Graph", padding=5)
        display_frame.grid(row=2, column=0, padx=10, pady=5, sticky="nsew")
        display_frame.grid_columnconfigure(0, weight=1)

        # Create Canvas that fills the frame
        self.canvas_width = 560  # Window width (600) minus padding (2*10 + 2*5)
        self.canvas_height = 60  # Increased height for better visibility
        self.speech_canvas = tk.Canvas(display_frame,
                                       width=self.canvas_width,
                                       height=self.canvas_height,
                                       bg="#f0f0f0")  # Light gray background
        self.speech_canvas.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        # Optional label that says "SPEAKING" or "SILENT"
        self.status_label = ttk.Label(display_frame, text="SILENT", foreground="#4a90e2")  # Soft blue
        self.status_label.grid(row=1, column=0, padx=5, pady=5)

    def _create_text_input_frame(self):
        """Create frame for text input."""
        text_frame = ttk.LabelFrame(self.root, text="Text Input", padding=5)
        text_frame.grid(row=3, column=0, padx=10, pady=5, sticky="ew")
        text_frame.grid_columnconfigure(0, weight=1)  # Make text entry expand

        # Create text entry as a Text widget for multiple lines
        self.text_entry = tk.Text(text_frame, width=50, height=3, wrap=tk.WORD)  # Multi-line text with word wrap
        self.text_entry.grid(row=0, column=0, padx=(5, 2), pady=5, sticky="ew")

        # Create send button
        self.send_button = ttk.Button(
            text_frame,
            text="Send",
            command=self._handle_text_submit
        )
        self.send_button.grid(row=0, column=1, padx=(2, 5), pady=5)

        # Bind Enter key to submit (Shift+Enter for new line)
        self.text_entry.bind('<Return>', self._on_enter_key)

    def _on_enter_key(self, event):
        """Handle Enter key press in text widget."""
        if not event.state & 0x1:  # If Shift is not pressed
            self._handle_text_submit()
            return "break"  # Prevent default Enter behavior
        return None  # Allow default Enter behavior (new line) when Shift is pressed

    def _handle_text_submit(self):
        """Handle text submission from entry field."""
        text = self.text_entry.get("1.0", tk.END).strip()
        if text:
            print("\n[GUI] Submitting text:", text)
            # If a callback for text submission is registered, call it directly.
            if self.on_text_change is not None:
                self.on_text_change(text)
            else:
                # Fallback: put text in the queue for polling.
                try:
                    self.text_input_queue.put_nowait(text)
                except queue.Full:
                    print("\n[GUI] Warning: Text input queue is full")
            # Clear the entry field
            self.text_entry.delete("1.0", tk.END)

    def _on_input_device_change(self, event):
        """Handle input device selection change."""
        new_device = self.input_device_var.get()
        print(f"\nSelected input device: {new_device}")
        self.input_device_name = new_device
        if self.on_input_change:
            self.on_input_change(new_device)

    def _on_output_device_change(self, event):
        """Handle output device selection change."""
        new_device = self.output_device_var.get()
        print(f"\nSelected output device: {new_device}")
        self.output_device_name = new_device
        if self.on_output_change:
            self.on_output_change(new_device)

    def _schedule_updates(self):
        """Schedule periodic UI updates."""
        if self.running:
            try:
                # Process pending device changes
                self._process_queued_updates()
                # Process speech state updates
                self._process_speech_updates()

                # Redraw the speech detection graph
                self._redraw_speech_graph()

                # Update window
                self.root.update_idletasks()
                # Schedule next update (~10 Hz or 100ms)
                self.root.after(100, self._schedule_updates)
            except Exception as e:
                print(f"Error in window update: {e}")

    def _get_input_devices(self) -> List[str]:
        """Get list of available input devices."""
        devices = []
        try:
            for device in sd.query_devices():
                if device['max_input_channels'] > 0:
                    devices.append(device['name'])
        except Exception as e:
            print(f"Error getting input devices: {e}")
        return devices

    def _get_output_devices(self) -> List[str]:
        """Get list of available output devices."""
        devices = []
        try:
            for device in sd.query_devices():
                if device['max_output_channels'] > 0:
                    devices.append(device['name'])
        except Exception as e:
            print(f"Error getting output devices: {e}")
        return devices

    def _process_queued_updates(self):
        """Process any queued device updates (not used as much now)."""
        try:
            while True:
                device = self.input_device_queue.get_nowait()
                self.input_device_var.set(device)
                self.input_device_queue.task_done()
        except queue.Empty:
            pass

        try:
            while True:
                device = self.output_device_queue.get_nowait()
                self.output_device_var.set(device)
                self.output_device_queue.task_done()
        except queue.Empty:
            pass

    def _process_speech_updates(self):
        """Consume new is_speech states from the queue and update self.speech_history."""
        try:
            while True:
                is_speech = self.speech_queue.get_nowait()
                # Remove the oldest and append the newest state
                self.speech_history.pop(0)
                self.speech_history.append(is_speech)
                self.speech_queue.task_done()

                # Update the status label accordingly
                if is_speech:
                    self.status_label.config(text="SPEAKING", foreground="#2ecc71")  # Soft green
                else:
                    self.status_label.config(text="SILENT", foreground="#4a90e2")  # Soft blue

        except queue.Empty:
            pass

    def _redraw_speech_graph(self):
        """
        Draw a simple rolling timeline of speech states in the Canvas.
        Each bar is a small rectangle with a slight gap between bars.
        Using softer colors: green for speech, light blue for silence.
        """
        self.speech_canvas.delete("all")  # clear existing
        # Calculate bar width with a small gap between bars
        gap = 2  # 2 pixels between bars
        bar_width = (self.canvas_width - (gap * (self.max_points - 1))) / self.max_points
        height = self.canvas_height
        
        # Add a background rectangle for better visibility
        self.speech_canvas.create_rectangle(
            0, 0, self.canvas_width, height,
            fill="#f8f9fa",  # Light background
            outline=""
        )

        for i, val in enumerate(self.speech_history):
            x1 = i * (bar_width + gap)
            x2 = x1 + bar_width
            y1 = 2  # Small top margin
            y2 = height - 2  # Small bottom margin

            if val:
                color = "#2ecc71"  # Soft green for speech
                self.speech_canvas.create_rectangle(
                    x1, y1, x2, y2,
                    fill=color,
                    outline="#27ae60"  # Slightly darker outline
                )
            else:
                color = "#bdc3c7"  # Soft gray-blue for silence
                self.speech_canvas.create_rectangle(
                    x1, y1, x2, y2,
                    fill=color,
                    outline="#95a5a6"  # Slightly darker outline
                )

    def _on_closing(self):
        """Handle window close."""
        print("\nClosing speech detection window...")
        # First set flags to stop processing
        self.running = False
        self.has_gui = False
        
        try:
            # Give time for audio processing to stop
            self.root.after(100, self._finish_closing)
        except Exception as e:
            print(f"Error initiating window close: {e}")
            self._finish_closing()

    def _finish_closing(self):
        """Complete the window closing process after delay."""
        try:
            # Clear all queues
            while not self.input_device_queue.empty():
                self.input_device_queue.get_nowait()
            while not self.output_device_queue.empty():
                self.output_device_queue.get_nowait()
            while not self.speech_queue.empty():
                self.speech_queue.get_nowait()
            while not self.text_input_queue.empty():
                self.text_input_queue.get_nowait()
                
            # Stop any pending updates by canceling all scheduled callbacks
            if hasattr(self, 'root'):
                for after_id in self.root.tk.call('after', 'info'):
                    self.root.after_cancel(after_id)
                
                # Destroy window in the main thread
                self.root.after_idle(self._destroy_window)
        except Exception as e:
            print(f"Error during window cleanup: {e}")
            self._destroy_window()

    def _destroy_window(self):
        """Final step of window destruction."""
        try:
            self.root.quit()
            self.root.destroy()
        except Exception as e:
            print(f"Error destroying window: {e}")

    def process_vad(self, is_speech: bool):
        """
        Thread-safe method to queue the latest speech state.
        The main logic calls this every time we have a new chunk
        or a new VAD decision.
        """
        self.speech_queue.put(is_speech)

    def get_text_input(self):
        """
        Thread-safe method to get text input from queue.
        Returns None if queue is empty.
        """
        try:
            text = self.text_input_queue.get_nowait()
            print("\n[GUI] Retrieved text from queue:", text)
            self.text_input_queue.task_done()
            return text
        except queue.Empty:
            return None

    def update(self):
        """
        Update the GUI if needed (optional if you want manual control).
        """
        if self.running and self.has_gui:
            try:
                self.root.update_idletasks()
                self.root.update()
            except Exception as e:
                print(f"Error updating GUI: {e}")
                self.running = False
                self.has_gui = False

    def close(self):
        """
        Close the window and clean up resources.
        """
        print("[Speech Detection Interface] close() called - start")
        if self.running and self.has_gui:
            self._on_closing()  # Use the same cleanup sequence as window close
        print("[Speech Detection Interface] close() finished")
