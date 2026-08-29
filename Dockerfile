FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

WORKDIR /workspace

# 1. Ubuntu 24.04'ün "externally-managed-environment" hatasını kesin olarak devre dışı bırakır
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# 2. Sistem araçlarını kur ve pip'i en son sürüme yükselt
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --upgrade pip

# 3. TÜM kütüphaneleri TEK BİR komutla ve önbellek kullanmadan kurar
# (Bu, satır satır hata vermeyi ve bağımlılık çakışmalarını önler)
RUN python3 -m pip install --no-cache-dir \
    "git+https://github.com/huggingface/diffusers" \
    transformers \
    accelerate \
    safetensors \
    hf-transfer \
    pillow \
    runpod

# 4. Handler dosyasını kopyala
COPY handler.py /workspace/handler.py

# 5. HuggingFace önbelleğini kalıcı depolama alanına yönlendir
ENV HF_HOME=/runpod-volume
ENV TRANSFORMERS_CACHE=/runpod-volume
ENV HF_HUB_CACHE=/runpod-volume

# 6. Başlatma komutu
CMD ["python3", "-u", "handler.py"]
