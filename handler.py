import sys
import types
import torch

# ============================================
# KRİTİK: diffusers import edilmeden ÖNCE
# torch.xpu mock'u uygulanmalı (NVIDIA GPU'da yok)
# ============================================
if not hasattr(torch, "xpu"):
    xpu = types.ModuleType("torch.xpu")
    xpu.is_available = lambda: False
    xpu.device_count = lambda: 0
    xpu.empty_cache = lambda: None
    xpu.manual_seed = lambda seed: None
    xpu.manual_seed_all = lambda seed: None
    xpu.current_device = lambda: 0
    xpu.set_device = lambda device: None
    xpu.get_device_name = lambda device=None: ""
    xpu.synchronize = lambda device=None: None
    torch.xpu = xpu
    sys.modules["torch.xpu"] = xpu
    print("🟢 torch.xpu mock uygulandı (NVIDIA GPU için)")

# ŞİMDİ diffusers'ı güvenle import edebiliriz
import runpod
import base64
import traceback
from io import BytesIO

# Qwen 2512 için model ID'sini buradan değiştir
MODEL_ID = "Qwen/Qwen-Image"

pipe = None

def load_model():
    global pipe
    if pipe is None:
        print(f"🟢 Model yükleniyor: {MODEL_ID}")
        from diffusers import QwenImagePipeline

        pipe = QwenImagePipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            cache_dir="/runpod-volume",
        )
        pipe.enable_model_cpu_offload()
        pipe.to("cuda")
        print("🟢 Model GPU'ya yüklendi ✅")
    return pipe

def handler(job):
    try:
        model = load_model()
        input_data = job.get('input', {})
        prompt = input_data.get('prompt', 'a highly detailed cat, 4k')
        negative_prompt = input_data.get('negative_prompt', 'blurry, low quality')
        width = int(input_data.get('width', 1024))
        height = int(input_data.get('height', 1024))
        steps = int(input_data.get('steps', 30))

        print(f"🟢 Görsel üretiliyor: {width}x{height}")

        with torch.inference_mode():
            result = model(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=5.0
            )

        # result.images bir listedir -> [0] ile ilk elemanı al
        buffered = BytesIO()
        result.images[0].save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {"status": "success", "image_base64": img_b64}

    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    print("🟢 RunPod Serverless başlatılıyor...")
    runpod.serverless.start({"handler": handler})
