FROM python:3.10-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    HF_HOME=/cache/huggingface \
    TRANSFORMERS_CACHE=/cache/huggingface \
    TORCH_HOME=/cache/torch \
    XDG_CACHE_HOME=/cache \
    TF_CPP_MIN_LOG_LEVEL=2

# System deps for audio and ML stacks
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    flac \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r /app/requirements.txt

# App code
COPY . /app

# Create default runtime dirs (can be overridden by env)
RUN mkdir -p /cache/huggingface /cache/torch \
    && mkdir -p /app/translation /app/audio /app/model

# Expose the default Render port (actual port comes from $PORT)
EXPOSE 10000

# Streamlit launch: binds to 0.0.0.0 and Render's $PORT
CMD streamlit run filmception_gui.py --server.port $PORT --server.address 0.0.0.0 