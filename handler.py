import sys
import types
import torch
import traceback

class FakeXPU(types.ModuleType):
    def __getattr__(self, name):
        if name in ("is_available",):
            return lambda *a, **kw: False
        if name in ("device_count",):
            return lambda *a, **kw: 0
        return lambda *a, **kw: None

if not hasattr(torch, "xpu"):
    fake = FakeXPU("xpu")
    torch.xpu = fake
    sys.modules["torch.xpu"] = fake

import runpod
import os

MODEL_PATH = "/runpod-volume/qwen-image-2512"

pipe = None

def load_model():
    global pipe
    if pipe is None:
        print(f"🟢 Model yolu: {MODEL_PATH}")
        try:
            from diffusers import QwenImagePipeline
            pipe = QwenImagePipeline.from_pretrained(
                MODEL_PATH,
                torch_dtype=torch.bfloat16
            ).to("cuda")
            print("🟢 Model başarıyla yüklendi! ✅")
        except Exception as e:
            print("🔴 TAM HATA:")
            print(traceback.format_exc())
            raise e
    return pipe

def handler(job):
    print("🟢 İstek alındı...")
    try:
        model = load_model()
        input_data = job['input']
        prompt = input_data.get('prompt', 'a highly detailed cat, 4k, masterpiece')
        result = model(
            prompt=prompt,
            negative_prompt=input_data.get('negative_prompt', ''),
            width=input_data.get('width', 1024),
            height=input_data.get('height', 1024),
            num_inference_steps=input_data.get('steps', 50),
            true_cfg_scale=4.0
        )
        output_path = "/tmp/output.png"
        if hasattr(result, "images") and isinstance(result.images, list):
            result.images[0].save(output_path)
        else:
            result.images.save(output_path)
        print("🟢 Görsel kaydedildi.")
        return {"status": "success", "output_path": output_path}
    except Exception as e:
        print(f"🔴 HATA: {str(e)}")
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
