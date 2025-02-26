# MiraConverse

MiraConverse: Your AI-Powered Voice Interaction Assistant

MiraConverse is a real-time voice interaction system that serves as your AI conversation partner. By integrating speech recognition and text-to-speech technologies, it delivers a responsive and immersive conversational experience.

![MiraConverse GUI](docs/screenshots/mira_converse_gui.png)

*MiraConverse's graphical interface with Mic input, and output device (speakers) selection, speech detection graph and text injection prompt.*

[![MiraConverse Demo](https://img.youtube.com/vi/n9oD7IPIWVI/0.jpg)](https://www.youtube.com/watch?v=n9oD7IPIWVI)

*Click the image above to watch a demo of MiraConverse in action*

## Key Features

- **Model Flexibility:** Supports any model with an OpenAI-compatible API, including local private models, giving you the freedom to choose the best fit for your needs.

- **Customizable Trigger Word:** Easily set your preferred activation word, with "Mira" as the default, ensuring intuitive and natural interactions.

- **Contextual Awareness:** Configurable settings maintain conversation context, enabling fluid and coherent dialogue without repetition.

## Features

- Real-time speech recognition using Whisper
- Natural language processing with LLM integration
- Text-to-speech synthesis using Kokoro
- Graphical interface for audio monitoring and device selection
- Configurable voice trigger system
- Robust WebSocket-based client-server architecture

## Tool System

MiraConverse features an extensible tool system that allows the AI to perform external actions through a structured interface. Tools are Python classes that implement specific functionality and can be called during conversations. For detailed technical specifications, see the [Tool System Documentation](docs/technical/tool_system.md).

### Available Tools

#### Weather Tool
Get real-time weather information for any location by asking "What's the weather in [city]?". The weather tool provides:
- Current temperature
- Weather conditions
- Wind speed
- Daily high/low temperatures
- Precipitation probability

Example: "Mira, what's the weather in San Francisco?"

## System Requirements

### Server Requirements
- Python 3.8 or higher
- NVIDIA GPU with at least 4GB VRAM (required for running both Whisper and Kokoro models)
  - GPU acceleration is required for real-time performance
- Sufficient disk space for models (approximately 10GB total)
- PortAudio library for audio processing
  - Ubuntu/Debian: `sudo apt-get install libportaudio2 portaudio19-dev`
  - Fedora: `sudo dnf install portaudio portaudio-devel`
  - Arch Linux: `sudo pacman -S portaudio`
  - macOS: `brew install portaudio`
  - Windows: Download and install the [PortAudio binaries](http://www.portaudio.com/download.html)
- espeak-ng (optional) for better text-to-speech phonemization
  - Ubuntu/Debian: Usually pre-installed, if not: `sudo apt-get install espeak-ng`
  - Fedora: `sudo dnf install espeak-ng`
  - Arch Linux: `sudo pacman -S espeak-ng`
  - macOS: `brew install espeak-ng`
  - Windows: Download and install from [GitHub releases](https://github.com/espeak-ng/espeak-ng/releases)
- Language-specific dependencies (only needed for certain languages):
  - Japanese support: `pip install misaki[ja]`
  - Chinese support: `pip install misaki[zh]`

### Client Requirements
- Python 3.11 or higher
- Audio input device (microphone)
- Audio output device (speakers)
- Basic CPU for audio processing

### Overview of Installation

The project uses Poetry for dependency management, which handles installing all required packages in an isolated environment. Here's how to get started:

1. Install Poetry:
   ```bash
   # Using pip
   pip install poetry
   
   # Or using pipx for isolated installation (recommended for advanced users)
   python -m pip install --user pipx
   python -m pipx ensurepath
   pipx install poetry
   ```

2. Set up both server and client:
   ```bash
   # In server directory
   cd server/
   poetry install  # This installs all dependencies in an isolated environment
   cp .env.example .env
   cp default_config.json config.json
   
   # In client directory
   cd ../client/
   poetry install  # This installs all dependencies in an isolated environment 
   # if this hangs, like it does on the raspbery pi, try: export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring and then run it again
   cp .env.example .env
   cp default_config.json config.json
   ```

3. Configure your environment:
   - Edit both .env files with your specific settings (see Configuration section below)
   - Update both config.json files as needed

Note: After running `poetry install`, you can run the Python files directly since Poetry automatically activates the virtual environment in the project directory. Alternatively, you can use `poetry run` to explicitly run commands in the Poetry environment.

### Server Setup

The server component requires Python 3.8 or higher and an NVIDIA GPU with at least 4GB VRAM for real-time performance. 

#### Language-Specific Dependencies

If you need Japanese or Chinese language support, you'll need to install additional dependencies on the server:

```bash
# For Japanese support
pip install misaki[ja]

# For Chinese support
pip install misaki[zh]
```

These dependencies are required only on the server side and only if you plan to use these specific languages.

Choose your operating system below for specific setup instructions.

#### Linux Server Setup

1. Install system dependencies:
```bash
# Ubuntu/Debian
sudo apt-get install libportaudio2 portaudio19-dev
# Optional: Install espeak-ng for better text-to-speech phonemization
sudo apt-get install espeak-ng

# Fedora
sudo dnf install portaudio portaudio-devel
# Optional: Install espeak-ng for better text-to-speech phonemization
sudo dnf install espeak-ng

# Arch Linux
sudo pacman -S portaudio
# Optional: Install espeak-ng for better text-to-speech phonemization
sudo pacman -S espeak-ng
```

2. Clone and set up the repository:
```bash
git clone https://github.com/KartDriver/mira_converse.git
cd mira_converse/server/
poetry install
cp .env.example .env
cp default_config.json config.json
```

3. Set up the required models:
   - Download the Whisper speech-to-text model from [HuggingFace](https://huggingface.co/openai/whisper-large-v3-turbo)
   - Download the Kokoro text-to-speech model from [HuggingFace](https://huggingface.co/hexgrad/Kokoro-82M)
   - Set the downloaded model paths in your .env file

4. Run the server:
```bash
# Run directly
python server.py

# Or using Poetry explicitly
poetry run python server.py
```

#### Windows Server Setup

1. Install system dependencies:
   - Install Python 3.8 or higher from [python.org](https://www.python.org/downloads/)
   - Optional: Install espeak-ng from [GitHub releases](https://github.com/espeak-ng/espeak-ng/releases) for better text-to-speech phonemization

2. Clone and set up the repository:
```bash
git clone https://github.com/KartDriver/mira_converse.git
cd mira_converse/server/
poetry install
copy .env.example .env
copy default_config.json config.json
```

3. Set up the required models:
   - Download the Whisper speech-to-text model from [HuggingFace](https://huggingface.co/openai/whisper-large-v3-turbo)
   - Download the Kokoro text-to-speech model from [HuggingFace](https://huggingface.co/hexgrad/Kokoro-82M)
   - Set the downloaded model paths in your .env file

4. Run the server:
```bash
# Run directly
python server.py

# Or using Poetry explicitly
poetry run python server.py
```

#### macOS Server Setup

1. Install system dependencies:
```bash
# Install Homebrew if you haven't already
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install required packages
brew install portaudio
# Optional: Install espeak-ng for better text-to-speech phonemization
brew install espeak-ng
```

2. Clone and set up the repository:
```bash
git clone https://github.com/KartDriver/mira_converse.git
cd mira_converse/server/
poetry install
cp .env.example .env
cp default_config.json config.json
```

3. Set up the required models:
   - Download the Whisper speech-to-text model from [HuggingFace](https://huggingface.co/openai/whisper-large-v3-turbo)
   - Download the Kokoro text-to-speech model from [HuggingFace](https://huggingface.co/hexgrad/Kokoro-82M)
   - Set the downloaded model paths in your .env file

4. Run the server:
```bash
# Run directly
python server.py

# Or using Poetry explicitly
poetry run python server.py
```

### Client Setup

#### Linux Client Setup

1. Install system dependencies:
```bash
# Ubuntu/Debian
sudo apt-get install python3-tk libportaudio2 portaudio19-dev

# Fedora
sudo dnf install python3-tk portaudio portaudio-devel

# Arch Linux
sudo pacman -S tk portaudio
```

2. Clone and set up the repository:
```bash
git clone https://github.com/KartDriver/mira_converse.git
cd mira_converse/client/
poetry install
cp .env.example .env
cp default_config.json config.json
```

3. Run the client:
```bash
# Run directly
python client.py

# Or using Poetry explicitly
poetry run python client.py
```

#### Windows Client Setup

1. Install system dependencies:
   - Install Python 3.12 or higher from [python.org](https://www.python.org/downloads/)
   - Download and install [PortAudio binaries](http://www.portaudio.com/download.html)
   - Ensure the PortAudio DLL is in your system PATH

2. Clone and set up the repository:
```bash
git clone https://github.com/KartDriver/mira_converse.git
cd mira_converse/client/
poetry install
copy .env.example .env
copy default_config.json config.json
```

3. Run the client:
```bash
# Run directly
python client.py

# Or using Poetry explicitly
poetry run python client.py
```

#### macOS Client Setup

1. Install system dependencies:
```bash
# Install Homebrew if you haven't already
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install required packages
brew install python-tk portaudio
```

2. Clone and set up the repository:
```bash
git clone https://github.com/KartDriver/mira_converse.git
cd mira_converse/client/
poetry install
cp .env.example .env
cp default_config.json config.json
```

3. Run the client:
```bash
# Run directly
python client.py

# Or using Poetry explicitly
poetry run python client.py
```

## Configuration

The system uses a combination of environment variables (.env) and configuration files (config.json) to manage settings. This separation allows for better security by keeping sensitive information like API keys in the environment variables while maintaining other configuration in JSON files.

### Configuration Structure

- **Environment Variables (.env)**: Store sensitive information and connection details
- **Configuration Files (config.json)**: Store non-sensitive settings and parameters
- **Priority**: Environment variables take precedence over config file settings

To get started:

1. Copy the default configuration files:
```bash
# For server
cp server/default_config.json server/config.json
cp server/.env.example server/.env

# For client
cp client/default_config.json client/config.json
cp client/.env.example client/.env
```

2. Edit both .env files and config.json files as needed

### Environment Variables (.env)

#### Server Environment Variables (server/.env)
```bash
# WebSocket Configuration
WEBSOCKET_HOST=localhost
WEBSOCKET_PORT=8765
WEBSOCKET_API_SECRET_KEY=your_secure_key_here

# Model Paths
WHISPER_PATH=path_to_whisper_model
KOKORO_PATH=path_to_kokoro_model
KOKORO_VOICE_NAME=af_heart
```

#### Client Environment Variables (client/.env)
```bash
# Server Connection
WEBSOCKET_HOST=localhost
WEBSOCKET_PORT=8765
WEBSOCKET_API_SECRET_KEY=your_secure_key_here

# LLM Configuration
MODEL_NAME=gpt-3.5-turbo  # Can be an OpenAI model or path to local model
API_SECRET_KEY=your_api_key_here
API_BASE=https://api.openai.com/v1  # Or URL to local API server
```

### Configuration Files (config.json)

The config.json files contain non-sensitive settings such as:
- Audio processing parameters
- Speech detection settings
- Conversation parameters
- Language settings
- Client retry settings

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
            // Linux/macOS path example:
            "path": "/home/user/models/whisper-large-v3-turbo",
            // Windows path examples (use either format):
            // "path": "C:/Users/user/models/whisper-large-v3-turbo"
            // "path": "C:\\Users\\user\\models\\whisper-large-v3-turbo"
        },
        "kokoro": {
            // Linux/macOS path example:
            "path": "/home/user/models/Kokoro-82M",
            // Windows path examples (use either format):
            // "path": "C:/Users/user/models/Kokoro-82M"
            // "path": "C:\\Users\\user\\models\\Kokoro-82M",
            "voice_name": "af",          // Voice pack to use
            "language_code": "a"         // Language code for TTS (optional, defaults to 'a')
                                        // See "Supported Languages" section for details
        }
    }
}
```

## Supported Languages

MiraConverse supports multiple languages for both prompts and text-to-speech synthesis. The language is selected on the client side and passed to the server for processing.

### Dynamic Language Pipeline Management

The server now implements a dynamic language pipeline management system that:
- Loads language-specific TTS pipelines on demand
- Caches recently used language pipelines for better performance
- Automatically frees resources from unused language pipelines

### Supported Languages

The following languages are currently supported:

| Language | ISO Code | Kokoro Code | Additional Dependencies |
|----------|----------|-------------|-------------------------|
| American English | en | a | None |
| British English | en-gb | b | None |
| Spanish | es | e | None |
| French | fr | f | None |
| Hindi | hi | h | None |
| Italian | it | i | None |
| Japanese | ja | j | `pip install misaki[ja]` (server-side) |
| Brazilian Portuguese | pt | p | None |
| Mandarin Chinese | zh | z | `pip install misaki[zh]` (server-side) |

### Language Configuration

Language selection is configured on the client side in the client's `config.json` file:

```json
"llm": {
    "prompt": {
        "language": "en",  // Set this to the desired ISO language code (en, es, fr, etc.)
        "custom_path": null,
        "directory": "prompts"
    }
}
```

When the client connects to the server, it automatically:
1. Sends the configured language code as a connection parameter
2. The server extracts this language code from the connection URI
3. The server maps the ISO language code (e.g., "en") to Kokoro's internal code (e.g., "a")
4. The appropriate language pipeline is loaded or retrieved from cache

> **Note:** For Japanese and Chinese language support, additional dependencies must be installed on the server:
> - For Japanese: `pip install misaki[ja]`
> - For Chinese: `pip install misaki[zh]`

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
        "threshold": 0.3,        // Silero VAD speech detection threshold (0.0-1.0)
        "consecutive_threshold": 2  // Number of consecutive speech frames needed to trigger speech start
    }
}
```

### Client Retry Settings
```json
"client": {
    "retry": {
        "max_attempts": 3,       // Maximum number of connection retry attempts
        "delay_seconds": 2       // Delay between retry attempts in seconds
    }
}
```

These retry settings are used for:
- Initial server connection attempts
- Automatic reconnection when the connection is lost
- Recovery from transient network issues

The client will attempt to reconnect up to the specified number of attempts, waiting the specified delay between each attempt. This provides robust operation even in unstable network environments.

## Usage

1. Start the server (in server directory):
```bash
# Run directly
python server.py

# Or using Poetry explicitly
poetry run python server.py
```

2. Start the client (in client directory):
```bash
# Run directly
python client.py

# Or using Poetry explicitly
poetry run python client.py
```

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

- If audio devices aren't detected:
  - Windows: Check Sound settings in Control Panel > Sound > Recording/Playback
  - Linux/macOS: Check your system's audio settings
- For GPU errors:
  - Windows: Verify CUDA installation in Device Manager > Display Adapters
  - All systems: Verify CUDA installation and GPU availability
- Connection issues:
  - Windows: Check Windows Defender Firewall settings
  - All systems: Check firewall settings and network connectivity
- For model loading errors:
  - Windows: Ensure paths use either forward slashes (C:/path) or escaped backslashes (C:\\path)
  - All systems: Verify model paths in config.json are correct
- If you experience audio issues:
  - Windows: Check audio format settings in Sound Control Panel
  - All systems: Try adjusting the VAD mode in speech_detection settings

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

## Long Term Project Goals

Expand capabailties to become a fully fledged smart home AI assistant with tool calling support and integration with 
third party API's.

1. **Weather Services**
   - Integration with weather APIs for real-time weather information
   - Support for multiple weather data providers

2. **Smart Home Integration**
   - Smart lighting control through various provider APIs
   - Support for major platforms like Philips Hue, LIFX, and other smart lighting systems
   - Expandable framework for other smart home devices
   - Scene creation and management capabilities

3. **Music Service Integration**
   - Spotify API integration for music playback and control
   - Playlist management and music recommendations
   - Support for other music streaming services
   - Voice-controlled music playback features

4. **Advanced Voice Detection and Fingerprinting**
   - Voice fingerprinting for user identification
   - Personalized conversation handling without trigger words
   - Multi-user support with unique voice profiles
   - Dynamic context switching based on speaker identity
   - Continuous conversation mode for fingerprinted voices

5. **Timer and Alarm System**
   - Voice-controlled alarm timer creation and management
   - Multiple concurrent timer support
