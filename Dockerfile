FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04
WORKDIR /workspace
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV HF_XET_HIGH_PERFORMANCE=1
ENV SCALING_THRESHOLD_BUFFER_MS=120000
ENV SCALING_MIN_QUEUE_TIME_MS=40000
ENV PYTHONWARNINGS="ignore::FutureWarning:huggingface_hub.constants"

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu121

RUN python3 -m pip uninstall -y torchaudio

# RUN python3 -m pip install --no-cache-dir runpod pillow safetensors diffusers transformers accelerate
RUN python3 -m pip install --no-cache-dir runpod pillow safetensors transformers accelerate
RUN python3 -m pip install --no-cache-dir git+https://github.com/huggingface/diffusers.git

COPY handler.py /workspace/handler.py

CMD ["python3", "-u", "handler.py"]
