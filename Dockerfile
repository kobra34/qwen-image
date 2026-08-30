# RunPod'un kararlı PyTorch taban imajı
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /workspace

# Sistem paketleri
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face ve RunPod ortam değişkenleri
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV HF_HOME=/runpod-volume
ENV TRANSFORMERS_CACHE=/runpod-volume
ENV HF_HUB_CACHE=/runpod-volume
ENV PYTHONWARNINGS="ignore::FutureWarning:huggingface_hub.constants"

# Bağımlılıklar: 
# - diffusers >= 0.31.0 (Qwen-Image desteği için şart)
# - xformers (VRAM tasarrufu ve hız için şart)
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

# Handler dosyasını imaja kopyala
COPY handler.py /workspace/handler.py

# Başlangıç komutu
CMD ["python3", "-u", "handler.py"]
