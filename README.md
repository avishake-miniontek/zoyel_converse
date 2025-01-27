# MiraConverse

MiraConverse: Your AI-Powered Voice Interaction Companion

MiraConverse is a real-time voice interaction system that serves as your AI conversation partner. By integrating speech recognition and text-to-speech technologies, it delivers a responsive and immersive conversational experience.

## Key Features

- **Model Flexibility:** Supports any model with an OpenAI-compatible API, including local private models, giving you the freedom to choose the best fit for your needs.

- **Customizable Trigger Word:** Easily set your preferred activation word, with "Mira" as the default, ensuring intuitive and natural interactions.

- **Contextual Awareness:** Configurable settings maintain conversation context, enabling fluid and coherent dialogue without repetition.

## Features

- Real-time speech recognition using Whisper
- Natural language processing with LLM integration
- Text-to-speech synthesis using Kokoro
- Professional-grade audio processing with WebRTC VAD
- Graphical interface for audio monitoring and device selection
- Configurable voice trigger system
- Robust WebSocket-based client-server architecture

## System Requirements

### Server Requirements
- Python 3.8 or higher
- NVIDIA GPU with at least 4GB VRAM (required for running both Whisper and Kokoro models)
  - GPU acceleration is required for real-time performance
  - CUDA toolkit must be installed for GPU support
- Sufficient disk space for models (approximately 10GB total)

### Client Requirements
- Python 3.8 or higher
- Audio input device (microphone)
- Audio output device (speakers)
- Basic CPU for audio processing
- tkinter/tk>=8.6 (only for GUI mode)

## Installation

### Server Setup

1. Clone the repository:
```bash
git clone [repository-url]
cd audio_chat
```

2. Install server dependencies:
```bash
pip install -r server_requirements.txt
```

3. Create your configuration file:
```bash
cp default_config.json config.json
```

4. Set up the required models:
   - Download the Whisper speech-to-text model from [HuggingFace](https://huggingface.co/openai/whisper-large-v3-turbo)
   - Download the Kokoro text-to-speech model from [HuggingFace](https://huggingface.co/hexgrad/Kokoro-82M)
   - Set the downloaded model paths in your config.json

Note: The models are large files (several GB) and require sufficient disk space. Make sure to use the correct paths where you downloaded the models in your config.json file:
```json
"models": {
    "whisper": {
        "path": "/path/to/whisper-large-v3-turbo"
    },
    "kokoro": {
        "path": "/path/to/Kokoro-82M",
        "voice_name": "af"  // Choose your preferred voice pack
    }
}
```

### Client Setup

Optionally, you can set up a virtual environment to isolate the project dependencies. If you don't have the venv module installed, you can install it using your system's package manager:

- On Ubuntu/Debian: `sudo apt-get install python3-venv`
- On CentOS/RHEL: `sudo yum install python3-venv`
- On Windows: The venv module is included with Python 3.x

#### Linux Setup

The installation process depends on whether you want to use the GUI interface or run in headless mode.

##### Headless Mode (No GUI)
If you only need command-line operation without a graphical interface:

1. Create your configuration file:
```bash
cp default_config.json config.json
```

2. (Optional) Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install headless dependencies:
```bash
pip install -r client_requirements_no_gui.txt
```

4. Run in headless mode:
```bash
python client.py --no-gui
```

##### GUI Mode
If you want to use the graphical interface:

1. Create your configuration file:
```bash
cp default_config.json config.json
```

2. (Optional) Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install GUI dependencies:
```bash
pip install -r client_requirements.txt
```

4. Run with GUI (default):
```bash
python client.py
```

#### Windows Setup

The installation process depends on whether you want to use the GUI interface or run in headless mode.

##### Headless Mode (No GUI)
If you only need command-line operation without a graphical interface:

1. Create your configuration file:
```bash
copy default_config.json config.json
```

2. (Optional) Create and activate a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate
```

3. Install headless dependencies:
```bash
pip install -r client_requirements_no_gui.txt
```

4. Run in headless mode:
```bash
python client.py --no-gui
```

##### GUI Mode
If you want to use the graphical interface:

1. Create your configuration file:
```bash
copy default_config.json config.json
```

2. (Optional) Create and activate a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate
```

3. Install GUI dependencies:
```bash
pip install -r client_requirements.txt
```

4. Run with GUI (default):
```bash
python client.py
```

#### macOS Setup

The installation process on macOS depends on whether you want to use the GUI interface or run in headless mode.

##### Headless Mode (No GUI)
If you only need command-line operation without a graphical interface:

1. Create your configuration file:
```bash
cp default_config.json config.json
```

2. (Optional) Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install headless dependencies:
```bash
pip install -r client_requirements_no_gui.txt
```

4. Run in headless mode:
```bash
python client.py --no-gui
```

##### GUI Mode
If you want to use the graphical interface, you'll need additional dependencies:

1. Install Homebrew if you haven't already:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

2. Install Python with tkinter support:
```bash
brew install python-tk
```

3. Create your configuration file:
```bash
cp default_config.json config.json
```

4. (Optional) Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

5. Install GUI dependencies:
```bash
pip install -r client_requirements.txt
```

6. Run with GUI (default):
```bash
python client.py
```

Note: The headless mode is particularly useful for server environments or when you don't need visual feedback. It uses fewer system resources and doesn't require the tkinter dependency.

## Configuration

The system uses a configuration file to manage all settings. To get started:

1. Copy the default configuration file to create your local config:
```bash
cp default_config.json config.json
```

2. Edit `config.json` with your specific settings:
   - Set paths to your downloaded models
   - Configure server addresses and ports
   - Set API keys
   - Adjust audio processing parameters if needed


Here's a detailed explanation of each configuration section:

### Assistant Settings
```json
"assistant": {
    "name": "Mira"  // Trigger word to activate the assistant
}
```

### Server Configuration
```json
"server": {
    "websocket": {
        "host": "localhost",     // WebSocket server host
        "port": 8765,            // WebSocket server port
        "api_key": "your_secure_key_here"    // Authentication key for client-server communication
    },
    "gpu_device": "auto",      // GPU device selection (auto, cpu, or cuda:N)
    "models": {
        "whisper": {
            "path": "/path/to/whisper/model"  // Path to Whisper model
        },
        "kokoro": {
            "path": "/path/to/kokoro/model",  // Path to Kokoro model
            "voice_name": "af"          // Voice pack to use
        }
    }
}
```

### OpenAI API Compatible LLM Configuration
```json
"llm": {
    "model_name": "your_model_name",  // Name of the LLM model to use
    "api_base": "http://localhost:8000/v1",  // API endpoint
    "api_key": "your_api_key_here",  // API key for LLM service
    "conversation": {
        "context_timeout": 180,       // Seconds before conversation context expires
        "max_tokens": 8000,          // Maximum tokens for context window
        "temperature": 0.7,          // Response randomness (0.0-1.0)
        "response_max_tokens": 1024   // Maximum tokens per response
    }
}
```

### Audio Processing Settings
```json
"audio_processing": {
    "chunk_size": 2048,          // Audio processing chunk size
    "desired_rate": 16000,       // Target sample rate in Hz
    "noise_floor": {
        "initial": -50.0,        // Initial noise floor (dB)
        "min": -65.0,           // Minimum allowed noise floor (dB)
        "max": -20.0            // Maximum allowed noise floor (dB)
    }
}
```

### Speech Detection Settings
```json
"speech_detection": {
    "preroll_duration": 0.5,     // Audio capture before speech detection (seconds)
    "min_speech_duration": 0.5,  // Minimum duration to consider as speech (seconds)
    "end_silence_duration": 0.8, // Maximum silence before closing capture (seconds)
    "vad_settings": {
        "mode": 2,              // WebRTC VAD aggressiveness (0-3, 3 being most aggressive)
        "frame_duration_ms": 20  // VAD frame duration in milliseconds
    }
}
```

### Client Settings
```json
"client": {
    "retry": {
        "max_attempts": 3,       // Maximum number of connection retry attempts (both for initial connection and reconnection)
        "delay_seconds": 2       // Delay between retry attempts in seconds
    }
}
```

These retry settings are used for both initial server connection and automatic reconnection attempts when the connection is lost. The client will attempt to reconnect up to the specified number of attempts, waiting the specified delay between each attempt.

## Usage

1. Start the server:
```bash
python server.py
```

2. Start the client:
```bash
# With GUI (default):
python client.py

# Without GUI (headless mode):
python client.py --no-gui
```

The headless mode is particularly useful for:
- Running on devices without display capabilities (e.g., Raspberry Pi)
- Server environments where GUI is not needed
- Minimizing resource usage

3. The system will:
   - Initialize audio devices
   - Calibrate noise floor
   - Connect to the server
   - Open the audio control interface
   - Begin listening for the trigger word

4. Speak the trigger word (default: "Mira") followed by your query
   - Example: "Mira, tell me a joke"
   - The system will process your speech and respond through text-to-speech

## Troubleshooting

- If audio devices aren't detected, check your system's audio settings
- For GPU errors, verify CUDA installation and GPU availability
- Connection issues may require checking firewall settings and network connectivity
- For model loading errors, verify model paths in config.json
- If you experience audio issues, try adjusting the VAD mode in speech_detection settings

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Architecture

MiraConverse uses a robust client-server architecture designed for scalability and efficiency:

### Server
- Centralized server that handles intensive processing tasks:
  - Speech recognition using Whisper
  - Text-to-speech synthesis using Kokoro
  - WebSocket server supporting multiple concurrent client connections
  - GPU acceleration for real-time model inference
  - Client session management with unique client IDs
  - Noise floor calibration per client

### Client
- Lightweight client focused on audio handling and user interface:
  - Efficient audio capture and streaming
  - Real-time audio level monitoring
  - WebSocket communication with automatic reconnection
  - Graphical interface for audio monitoring
  - Local audio output management
  - Minimal resource requirements (basic CPU for audio processing)

### Key Benefits
- **Scalable**: Single server can support multiple concurrent clients
- **Resource Efficient**: Heavy processing (ML models) runs on server, clients remain lightweight
- **Flexible Deployment**: Server can be run locally or on a remote machine
- **Robust Communication**: WebSocket-based protocol with authentication and automatic reconnection
- **Independent Operation**: Each client maintains its own session state and audio calibration
