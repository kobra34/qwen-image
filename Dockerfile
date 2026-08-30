FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /workspace

# Sistem paketleri
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face ve RunPod ayarları
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV HF_HOME=/runpod-volume/models
ENV TRANSFORMERS_CACHE=/runpod-volume/models
ENV HF_HUB_CACHE=/runpod-volume/models

# Bağımlılıklar: diffusers'ı en son sürümden (main) çekiyoruz ki Qwen-Image'ı tanısın
# xformers ekliyoruz ki VRAM tasarrufu sağlansın
RUN python3 -m pip install --no-cache-dir --upgrade \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121 \
    && python3 -m pip uninstall -y torchaudio \
    && python3 -m pip install --no-cache-dir \
    "diffusers>=0.31.0" \
    transformers \
    accelerate \
    xformers \
    safetensors \
    pillow \
    runpod \
    huggingface_hub

# Handler dosyasını kopyala
COPY handler.py /workspace/handler.py

CMD ["python3", "-u", "handler.py"]
