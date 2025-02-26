FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.7.1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    python3-venv \
    git \
    curl \
    libsndfile1 \
    portaudio19-dev \
    espeak-ng \
    libespeak-ng-dev \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry to PATH
ENV PATH="$POETRY_HOME/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy Poetry configuration files
COPY pyproject.toml poetry.lock* ./

# Install dependencies
RUN poetry install --no-root --no-dev

# Second stage for the final image
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    libsndfile1 \
    portaudio19-dev \
    espeak-ng \
    libespeak-ng-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy Python dependencies from builder stage
COPY --from=builder /app/.venv /app/.venv

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Copy application code
COPY . .

# Create directories for model mounting
RUN mkdir -p /models/whisper /models/kokoro

# Expose the WebSocket port
EXPOSE 8765

# Set default environment variables
ENV WEBSOCKET_HOST=0.0.0.0 \
    WEBSOCKET_PORT=8765 \
    WHISPER_PATH=/models/whisper \
    KOKORO_PATH=/models/kokoro \
    KOKORO_VOICE_NAME=af_heart

# Run the server
CMD ["python", "server.py"]
