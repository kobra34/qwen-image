FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04
WORKDIR /workspace
RUN python3 -m pip install --no-cache-dir runpod
COPY handler.py /workspace/handler.py
CMD ["python3", "-u", "handler.py"]
