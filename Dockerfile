FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

WORKDIR /workspace

# Ubuntu 24.04'ün pip kısıtlamasını kesin olarak kaldır
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# Sistem araçlarını kur
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

# TÜM kütüphaneleri (RUNPOD DAHİL) tek seferde ve önbelleksiz kur
# Bu sefer hata vermemesi için en temel ve güvenli formatta yazıldı
RUN python3 -m pip install --no-cache-dir \
    runpod \
    pillow \
    safetensors \
    huggingface_hub \
    hf-transfer \
    accelerate \
    transformers

# Diffusers en sona (GitHub bağlantısı bazen ilk denemede yavaş olabilir)
RUN python3 -m pip install --no-cache-dir "git+https://github.com/huggingface/diffusers"

COPY handler.py /workspace/handler.py

ENV HF_HOME=/runpod-volume
ENV TRANSFORMERS_CACHE=/runpod-volume
ENV HF_HUB_CACHE=/runpod-volume

CMD ["python3", "-u", "handler.py"]
