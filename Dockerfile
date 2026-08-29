# 1. Sunucudaki CUDA 12.4 sürücüsüyle TAM UYUMLU kararlı temel imaj
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /workspace

# Ubuntu kısıtlamasını kaldır ve hızlı indirmeyi aç
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# Sistem araçlarını kur
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Pip'i güncelle
RUN python3 -m pip install --upgrade pip

# 2. KARARLI (STABLE) sürümleri kullan. GitHub ana dalı (git+) kırık olduğu için 
# resmi PyPI sürümlerini kullanıyoruz. Bu, import hatalarını %100 çözer.
RUN python3 -m pip install --no-cache-dir \
    runpod \
    pillow \
    safetensors \
    huggingface_hub \
    hf-transfer \
    accelerate \
    "transformers>=4.46.0" \
    "diffusers>=0.31.0"

COPY handler.py /workspace/handler.py

# Modeli kalıcı depolama alanına yönlendir
ENV HF_HOME=/runpod-volume
ENV TRANSFORMERS_CACHE=/runpod-volume
ENV HF_HUB_CACHE=/runpod-volume

CMD ["python3", "-u", "handler.py"]
