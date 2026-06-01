FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV USE_GPU=true
ENV HF_HOME=/app/.cache/huggingface
ENV EASYOCR_MODULE_PATH=/app/.cache/easyocr

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    poppler-utils \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

# Install CUDA-enabled PyTorch first. Do not install the CPU torch pins from requirements.txt.
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

COPY requirements-docker-gpu.txt .
RUN pip install --no-cache-dir -r requirements-docker-gpu.txt

COPY . .

RUN mkdir -p data/raw data/processed data/ocr_output data/structured data/ner_output models

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
