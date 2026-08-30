import runpod
import torch
import os
import base64
import traceback
from io import BytesIO

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen-Image")
pipe = None

def load_model():
    global pipe
    if pipe is None:
        print(f"🟢 Model yükleniyor: {MODEL_ID}")
        try:
            from diffusers import QwenImagePipeline
            
            pipe = QwenImagePipeline.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.bfloat16,
                cache_dir="/runpod-volume",
            )
            
            # CPU offload - 48GB GPU'da bile güvenli
            pipe.enable_model_cpu_offload()
            
            print("🟢 Model yüklendi ✅")
        except Exception as e:
            print(f"🔴 HATA: {e}")
            traceback.print_exc()
            raise
    return pipe

def handler(job):
    try:
        model = load_model()
        input_data = job.get('input', {})
        prompt = input_data.get('prompt', 'a cat')
        negative_prompt = input_data.get('negative_prompt', '')
        width = int(input_data.get('width', 1024))
        height = int(input_data.get('height', 1024))
        steps = int(input_data.get('steps', 30))

        with torch.inference_mode():
            result = model(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=5.0
            )

        buffered = BytesIO()
        result.images[0].save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {"status": "success", "image_base64": img_b64}

    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
