FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /workspace

ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV HF_HOME=/runpod-volume
ENV TRANSFORMERS_CACHE=/runpod-volume
ENV HF_HUB_CACHE=/runpod-volume

# 1. torchvision ve torchaudio'yu TAMAMEN KALDIR
#    Bu, circular import hatasını kökünden çözer
RUN pip uninstall -y torchvision torchaudio 2>/dev/null || true

# 2. Sadece gerekli kütüphaneleri kur
#    xformers YOK (versiyon çakışması yaratıyor)
#    torchvision YOK (circular import yaratıyor)
RUN pip install --no-cache-dir \
    "transformers>=4.48.0" \
    "diffusers>=0.31.0" \
    accelerate \
    safetensors \
    pillow \
    runpod \
    huggingface_hub \
    qwen-vl-utils

COPY handler.py /workspace/handler.py

CMD ["python3", "-u", "handler.py"]
