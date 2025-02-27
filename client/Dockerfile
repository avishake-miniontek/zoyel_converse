FROM python:3.11 as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=2.1.1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    libsndfile1 \
    portaudio19-dev \
    alsa-utils \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry to PATH
ENV PATH="$POETRY_HOME/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy Poetry configuration files
COPY pyproject.toml poetry.lock* ./

# Update lock file and install dependencies
RUN poetry lock && poetry install --no-root --only main

# Second stage for the final image
FROM python:3.11

# Install system dependencies for audio
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    portaudio19-dev \
    alsa-utils \
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

# Set default environment variables
ENV WEBSOCKET_HOST=server \
    WEBSOCKET_PORT=8765 \
    MODEL_NAME=gpt-3.5-turbo \
    API_BASE=https://api.openai.com/v1

# Run the client
CMD ["python", "client.py", "--no-gui"]
