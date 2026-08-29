FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

WORKDIR /workspace

ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# ADIM 1: Sadece sistem araçlarını kurar. (Burada hata verirse internet/apt sorunu vardır)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# ADIM 2: Sadece pip'i günceller
RUN python3 -m pip install --upgrade pip

# ADIM 3: Standart kütüphaneleri kurar (runpod dahil)
RUN python3 -m pip install --no-cache-dir \
    transformers \
    accelerate \
    safetensors \
    huggingface_hub \
    hf-transfer \
    pillow \
    runpod

# ADIM 4: En son ve bazen bağlantı sorunu yaratabilen diffusers'ı ayrı kurar
RUN python3 -m pip install --no-cache-dir "git+https://github.com/huggingface/diffusers"

COPY handler.py /workspace/handler.py

ENV HF_HOME=/runpod-volume
ENV TRANSFORMERS_CACHE=/runpod-volume
ENV HF_HUB_CACHE=/runpod
