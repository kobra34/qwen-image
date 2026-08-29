# Docker Hub'da kesinlikle var olan, kararlı RunPod imajı
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /workspace

ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# Sistem araçlarını kur
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

# Qwen-Image-2512 uyumluluğu için Hugging Face kütüphanelerini 
# doğrudan en güncel GitHub kaynaklarından çekiyoruz (PyPI sürümleri yetersiz kalıyor)
RUN python3 -m pip install --no-cache-dir \
    runpod \
    pillow \
    safetensors \
    git+https://github.com/huggingface/huggingface_hub.git \
    git+https://github.com/huggingface/accelerate.git \
    git+https://github.com/huggingface/transformers.git \
    git+https://github.com/huggingface/diffusers.git

COPY handler.py /workspace/handler.py

ENV HF_HOME=/runpod-volume
ENV TRANSFORMERS_CACHE=/runpod-volume
ENV HF_HUB_CACHE=/runpod-volume

CMD ["python3", "-u", "handler.py"]
