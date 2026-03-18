FROM node:20-slim AS coach-builder

WORKDIR /build

COPY package.json package-lock.json ./
COPY scripts/build-coach.mjs ./scripts/build-coach.mjs
COPY frontend/js/coach-app.jsx ./frontend/js/coach-app.jsx

RUN npm ci
RUN npm run build:coach

FROM python:3.13-slim

# Weightless Intelligence: Environment configuration
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHON_JIT=on

# Install system dependencies for Audio (Whisper/VAD) and Vision (OpenCV)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libasound2-dev \
    libportaudio2 \
    portaudio19-dev \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip and install dependencies
RUN pip install --no-cache-dir --upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application logic and frontend assets
COPY ./app ./app
COPY ./frontend ./frontend
COPY --from=coach-builder /build/frontend/js/coach-app.bundle.js ./frontend/js/coach-app.bundle.js

# Metadata and Data volumes
VOLUME ["/app/data"]

EXPOSE 4444

# Start with JIT enabled
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "4444", "--reload"]
