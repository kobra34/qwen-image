FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

WORKDIR /workspace

# 1. Ubuntu 24.04'ün pip kısıtlamasını kaldır ve hf-transfer'ı aktif et
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# 2. TEK BİR ADIMDA: Sistem araçlarını kur, pip'i güncelle ve TÜM kütüphaneleri yükle.
# Bu zincirleme yapı, "git bulunamadı" veya "build araçları eksik" hatalarını %100 önler.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        python3-dev \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --upgrade pip \
    && python3 -m pip install --no-cache-dir \
        "git+https://github.com/huggingface/diffusers" \
        transformers \
        accelerate \
        safetensors \
        huggingface_hub \
        hf-transfer \
        pillow \
        runpod

# 3. Handler dosyasını kopyala
COPY handler.py /workspace/handler.py

# 4. HuggingFace önbelleğini kalıcı depolama alanına yönlendir
ENV HF_HOME=/runpod-volume
ENV TRANSFORMERS_CACHE=/runpod-volume
ENV HF_HUB_CACHE=/runpod-volume

# 5. Başlatma komutu
CMD ["python3", "-u", "handler.py"]
