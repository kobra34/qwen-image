# CUDA 11.8 tabanlı, RunPod sürücüleriyle tam uyumlu kararlı imaj
FROM runpod/pytorch:2.2.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /workspace

ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV HF_XET_HIGH_PERFORMANCE=1

# RunPod ve Hugging Face uyarı loglarını temizlemek için değişkenler
ENV SCALING_THRESHOLD_BUFFER_MS=120000
ENV SCALING_MIN_QUEUE_TIME_MS=40000
ENV PYTHONWARNINGS="ignore::FutureWarning:huggingface_hub.constants"

# Sistem araçlarını kur
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Temiz ve hatasız kütüphane kurulumu
RUN python3 -m pip install --no-cache-dir runpod pillow safetensors diffusers[torch]

COPY handler.py /workspace/handler.py

ENV HF_HOME=/runpod-volume
ENV TRANSFORMERS_CACHE=/runpod-volume
ENV HF_HUB_CACHE=/runpod-volume

CMD ["python3", "-u", "handler.py"]
