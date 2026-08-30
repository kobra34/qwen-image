FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /workspace

# Sistem paketleri
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf -rf /var/lib/apt/lists/*

# Ortam değişkenleri
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV HF_HOME=/runpod-volume
ENV TRANSFORMERS_CACHE=/runpod-volume
ENV HF_HUB_CACHE=/runpod-volume
ENV PYTHONWARNINGS="ignore::FutureWarning:huggingface_hub.constants"

# KRİTİK: Doğru sürüm kombinasyonu
# 1. Önce torch ve torchvision'u uyumlu sürümlerde kur
RUN python3 -m pip install --no-cache-dir --upgrade \
    torch==2.4.0 \
    torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu121

# 2. torchaudio'yu kaldır (gerek yok, çakışma yaratır)
RUN python3 -m pip uninstall -y torchaudio

# 3. transformers'ı Qwen2.5-VL desteği olan sürüme güncelle (>=4.45.0)
# 4. diffusers'ı Qwen-Image pipeline desteği olan sürüme güncelle (>=0.30.0)
RUN python3 -m pip install --no-cache-dir \
    "transformers>=4.45.0" \
    "diffusers>=0.30.0" \
    accelerate \
    xformers \
    safetensors \
    pillow \
    runpod \
    huggingface_hub

# Handler dosyasını kopyala
COPY handler.py /workspace/handler.py

CMD ["python3", "-u", "handler.py"]
