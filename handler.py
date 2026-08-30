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
        print(f"🟢 [Lazy Load] {MODEL_ID} modeli yükleniyor...")
        try:
            # Önce import test et, hata varsa erken fail et
            from diffusers import QwenImagePipeline
            print("🟢 QwenImagePipeline başarıyla import edildi ✅")
            
            pipe = QwenImagePipeline.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.bfloat16,
                cache_dir="/runpod-volume",
                variant="fp16"
            )
            
            # VRAM optimizasyonları
            pipe.enable_xformers_memory_efficient_attention()
            pipe.enable_model_cpu_offload()
            pipe.to("cuda")
            
            print("🟢 Model başarıyla GPU'ya yüklendi! ✅")
            
        except ImportError as e:
            print(f"🔴 IMPORT HATASI: {e}")
            print("🔴 Muhtemelen transformers veya diffusers sürümü çok eski!")
            raise
        except Exception as e:
            print(f"🔴 Model yüklenirken hata: {e}")
            raise
    return pipe

def handler(job):
    print("🟢 İstek alındı...")
    try:
        model = load_model()
        
        input_data = job.get('input', {})
        prompt = input_data.get('prompt', 'a highly detailed cat, 4k, masterpiece')
        negative_prompt = input_data.get('negative_prompt', 'blurry, low quality')
        width = int(input_data.get('width', 1024))
        height = int(input_data.get('height', 1024))
        steps = int(input_data.get('steps', 30))
        
        print(f"🟢 Görsel üretiliyor: {width}x{height}, {steps} adım")
        
        with torch.inference_mode():
            result = model(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=5.0
            )
        
        # result.images bir listedir, ilk elemanı al
        output_path = "/tmp/output.png"
        result.images[0].save(output_path)
        print(f"🟢 Görsel kaydedildi: {output_path}")
        
        # Base64'e çevir
        buffered = BytesIO()
        result.images[0].save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return {
            "status": "success",
            "output_path": output_path,
            "image_base64": img_base64
        }
        
    except Exception as e:
        print(f"🔴 Handler hatası: {str(e)}")
        traceback.print_exc()
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    print("🟢 RunPod Serverless başlatılıyor...")
    runpod.serverless.start({"handler": handler})
