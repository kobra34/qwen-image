FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

WORKDIR /workspace

# 1. Ubuntu 24.04 kısıtlamasını kaldır
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# 2. Sistem araçlarını kur
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 3. Pip'i en son sürüme yükselt
RUN python3 -m pip install --upgrade pip

# 4. RUNPOD'u ZORLAYARAK kur (Mevcut çakışmaları yoksay)
RUN python3 -m pip install --no-cache-dir --force-reinstall --ignore-installed runpod

# 5. Diğer ML paketlerini ZORLAYARAK kur
RUN python3 -m pip install --no-cache-dir --force-reinstall --ignore-installed \
    pillow \
    safetensors \
    huggingface_hub \
    hf-transfer \
    accelerate \
    transformers

# 6. Diffusers'ı en sona bırak
RUN python3 -m pip install --no-cache-dir --force-reinstall --ignore-installed "git+https://github.com/huggingface/diffusers"

COPY handler.py /workspace/handler.py

ENV HF_HOME=/runpod-volume
ENV TRANSFORMERS_CACHE=/runpod-volume
ENV HF_HUB_CACHE=/runpod-volume

CMD ["python3", "-u", "handler.py"]
