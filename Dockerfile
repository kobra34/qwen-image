FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

WORKDIR /workspace

# Ubuntu 24.04 kısıtlamasını kaldır ve hızlı indirmeyi aç
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# Sistem araçlarını kur
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

# runpod kütüphanesi bu temel imajda ZATEN YÜKLÜDÜR. 
# Tekrar kurmaya çalışmak çakışma yarattığı için bu satır kaldırıldı.

RUN python3 -m pip install --no-cache-dir pillow
RUN python3 -m pip install --no-cache-dir safetensors
RUN python3 -m pip install --no-cache-dir huggingface_hub
RUN python3 -m pip install --no-cache-dir hf-transfer
RUN python3 -m pip install --no-cache-dir accelerate
RUN python3 -m pip install --no-cache-dir transformers

# Diffusers en son kalsın
RUN python3 -m pip install --no-cache-dir "git+https://github.com/huggingface/diffusers"

COPY handler.py /workspace/handler.py

ENV HF_HOME=/runpod-volume
ENV TRANSFORMERS_CACHE=/runpod-volume
ENV HF_HUB_CACHE=/runpod-volume

CMD ["python3", "-u", "handler.py"]
