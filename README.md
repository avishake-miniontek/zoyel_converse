# MiraConverse

MiraConverse: Your AI-Powered Voice Interaction Assistant

MiraConverse is a real-time voice interaction system that serves as your AI conversation partner. By integrating speech recognition and text-to-speech technologies, it delivers a responsive and immersive conversational experience.

![MiraConverse GUI](docs/screenshots/mira_converse_gui.png)

*MiraConverse's graphical interface with Mic input, and output device (speakers) selection, speech detection graph and text injection prompt.*

[![MiraConverse Demo](https://img.youtube.com/vi/n9oD7IPIWVI/0.jpg)](https://www.youtube.com/watch?v=n9oD7IPIWVI)

*Click the image above to watch a demo of MiraConverse in action*

## Table of Contents

- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [System Requirements](#system-requirements)
- [Installation](#installation)
  - [Option 1: Using Poetry (Recommended)](#option-1-using-poetry-recommended)
  - [Option 2: Using pip with requirements.txt](#option-2-using-pip-with-requirementstxt)
  - [Option 3: Docker Setup](#option-3-docker-setup)
- [Configuration](#configuration)
- [Usage](#usage)
- [Supported Languages](#supported-languages)
- [Tool System](#tool-system)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Long Term Project Goals](#long-term-project-goals)

## Key Features

- **Model Flexibility:** Supports any model with an OpenAI-compatible API, including local private models, giving you the freedom to choose the best fit for your needs.

- **Customizable Trigger Word:** Easily set your preferred activation word, with "Mira" as the default, ensuring intuitive and natural interactions.

- **Contextual Awareness:** Configurable settings maintain conversation context, enabling fluid and coherent dialogue without repetition.

- **Real-time Speech Recognition:** Uses Whisper for accurate speech-to-text conversion.

- **Natural Text-to-Speech:** Employs Kokoro for high-quality voice synthesis.

- **Graphical Interface:** Provides audio monitoring and device selection.

- **Extensible Tool System:** Allows the AI to perform external actions through a structured interface.

## System Architecture

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

## System Requirements

### Server Requirements
- Python 3.11 or higher
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
- Python 3.10 or higher
- Audio input device (microphone)
- Audio output device (speakers)
- Basic CPU for audio processing

## Installation

There are three ways to install and run MiraConverse:

### Option 1: Using Poetry (Recommended)

[Poetry](https://python-poetry.org/) is the recommended dependency management tool for MiraConverse as it handles installing all required packages in an isolated environment.

1. Install Poetry:
   ```bash
   # Using pip
   pip install poetry
   
   # Or using pipx for isolated installation (recommended for advanced users)
   python -m pip install --user pipx
   python -m pipx ensurepath
   pipx install poetry
   ```

2. Clone the repository:
   ```bash
   git clone https://github.com/KartDriver/mira_converse.git
   cd mira_converse
   ```

3. Set up the server:
   ```bash
   cd server/
   poetry install  # This installs all dependencies in an isolated environment
   cp .env.example .env
   cp default_config.json config.json
   ```

4. Set up the client:
   ```bash
   cd ../client/
   poetry install  # This installs all dependencies in an isolated environment 
   # if this hangs, like it does on the raspberry pi, try: export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring and then run it again
   cp .env.example .env
   cp default_config.json config.json
   ```

5. Configure your environment:
   - Edit both .env files with your specific settings (see Configuration section below)
   - Update both config.json files as needed

### Option 2: Using pip with requirements.txt

If you prefer not to use Poetry, you can install the dependencies directly using pip with the provided requirements files.

1. Clone the repository:
   ```bash
   git clone https://github.com/KartDriver/mira_converse.git
   cd mira_converse
   ```

2. Set up the server:
   ```bash
   cd server/
   # Create and activate a virtual environment (recommended)
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r server_requirements.txt
   
   # Copy configuration files
   cp .env.example .env
   cp default_config.json config.json
   ```

3. Set up the client:
   ```bash
   cd ../client/
   # Create and activate a virtual environment (recommended)
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r client_requirements.txt
   
   # Copy configuration files
   cp .env.example .env
   cp default_config.json config.json
   ```

4. Configure your environment:
   - Edit both .env files with your specific settings (see Configuration section below)
   - Update both config.json files as needed

### Option 3: Docker Setup

MiraConverse can also be run using Docker containers. This is particularly useful for deployment or if you want to avoid installing dependencies directly on your system.

1. Clone the repository:
   ```bash
   git clone https://github.com/KartDriver/mira_converse.git
   cd mira_converse
   ```

2. Create `.env` files for both server and client based on their templates:
   ```bash
   # For server
   cd server
   cp .env.example .env
   
   # For client
   cd ../client
   cp .env.example .env
   ```

3. Edit both `.env` files to set your configuration.

4. Build and run the server container:
   ```bash
   # In the server directory
   docker compose build
   docker compose up
   ```

5. In a separate terminal, build and run the client container:
   ```bash
   # In the client directory
   docker compose build
   docker compose up
   ```

For more detailed Docker setup instructions, see [README.docker.md](README.docker.md).

## Operating System-Specific Setup

### Linux Setup

1. Install system dependencies:
```bash
# Ubuntu/Debian
sudo apt-get install libportaudio2 portaudio19-dev python3-tk
# Optional: Install espeak-ng for better text-to-speech phonemization
sudo apt-get install espeak-ng

# Fedora
sudo dnf install portaudio portaudio-devel python3-tk
# Optional: Install espeak-ng for better text-to-speech phonemization
sudo dnf install espeak-ng

# Arch Linux
sudo pacman -S portaudio tk
# Optional: Install espeak-ng for better text-to-speech phonemization
sudo pacman -S espeak-ng
```

2. Set up the required models:
   - Download the Whisper speech-to-text model from [HuggingFace](https://huggingface.co/openai/whisper-large-v3-turbo)
   - Download the Kokoro text-to-speech model from [HuggingFace](https://huggingface.co/hexgrad/Kokoro-82M)
   - Set the downloaded model paths in your .env file

### Windows Setup

1. Install system dependencies:
   - Install Python 3.10 or higher from [python.org](https://www.python.org/downloads/)
   - Download and install [PortAudio binaries](http://www.portaudio.com/download.html)
   - Optional: Install espeak-ng from [GitHub releases](https://github.com/espeak-ng/espeak-ng/releases) for better text-to-speech phonemization

2. Set up the required models:
   - Download the Whisper speech-to-text model from [HuggingFace](https://huggingface.co/openai/whisper-large-v3-turbo)
   - Download the Kokoro text-to-speech model from [HuggingFace](https://huggingface.co/hexgrad/Kokoro-82M)
   - Set the downloaded model paths in your .env file

### macOS Setup

1. Install system dependencies:
```bash
# Install Homebrew if you haven't already
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install required packages
brew install portaudio python-tk
# Optional: Install espeak-ng for better text-to-speech phonemization
brew install espeak-ng
```

2. Set up the required models:
   - Download the Whisper speech-to-text model from [HuggingFace](https://huggingface.co/openai/whisper-large-v3-turbo)
   - Download the Kokoro text-to-speech model from [HuggingFace](https://huggingface.co/hexgrad/Kokoro-82M)
   - Set the downloaded model paths in your .env file

## Configuration

The system uses a combination of environment variables (.env) and configuration files (config.json) to manage settings. This separation allows for better security by keeping sensitive information like API keys in the environment variables while maintaining other configuration in JSON files.

### Configuration Structure

- **Environment Variables (.env)**: Store sensitive information and connection details
- **Configuration Files (config.json)**: Store non-sensitive settings and parameters
- **Priority**: Environment variables take precedence over config file settings

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

For detailed configuration options, see the comments in the default config.json files.

## Usage

1. Start the server (in server directory):
```bash
# If using Poetry:
poetry run python server.py
# Or if Poetry environment is already activated:
python server.py

# If using pip with requirements.txt:
python server.py
```

2. Start the client (in client directory):
```bash
# If using Poetry:
poetry run python client.py
# Or if Poetry environment is already activated:
python client.py

# If using pip with requirements.txt:
python client.py
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

## Supported Languages

MiraConverse supports multiple languages for both prompts and text-to-speech synthesis. The language is selected on the client side and passed to the server for processing.

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

> **Note:** For Japanese and Chinese language support, additional dependencies must be installed on the server:
> - For Japanese: `pip install misaki[ja]`
> - For Chinese: `pip install misaki[zh]`

## Tool System

MiraConverse features an extensible tool system that allows the AI to perform external actions through a structured interface. Tools are Python classes that implement specific functionality and can be called during conversations.

### Available Tools

#### Weather Tool
Get real-time weather information for any location by asking "What's the weather in [city]?". The weather tool provides:
- Current temperature
- Weather conditions
- Wind speed
- Daily high/low temperatures
- Precipitation probability

Example: "Mira, what's the weather in San Francisco?"

For detailed technical specifications, see the [Tool System Documentation](docs/technical/tool_system.md).

## Troubleshooting

### Audio Issues
- If audio devices aren't detected:
  - Windows: Check Sound settings in Control Panel > Sound > Recording/Playback
  - Linux/macOS: Check your system's audio settings
- For microphone not working:
  - Ensure the correct input device is selected in the GUI
  - Check system permissions for microphone access
  - Try adjusting the noise floor settings in config.json

### GPU Issues
- For GPU errors:
  - Windows: Verify CUDA installation in Device Manager > Display Adapters
  - All systems: Verify CUDA installation and GPU availability
  - Try running `nvidia-smi` to check GPU status and memory
  - Ensure your GPU has at least 4GB VRAM

### Connection Issues
- If client cannot connect to server:
  - Check that the server is running
  - Verify WebSocket host and port settings in both .env files
  - Check that the API keys match in both .env files
  - Windows: Check Windows Defender Firewall settings
  - All systems: Check firewall settings and network connectivity

### Model Loading Errors
- If models fail to load:
  - Windows: Ensure paths use either forward slashes (C:/path) or escaped backslashes (C:\\path)
  - All systems: Verify model paths in config.json are correct
  - Check that you have sufficient disk space and RAM
  - Verify that the model files are complete and not corrupted

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Long Term Project Goals

Expand capabilities to become a fully fledged smart home AI assistant with tool calling support and integration with 
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
