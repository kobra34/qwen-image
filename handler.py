import runpod
import torch
import os
import base64
import traceback
from io import BytesIO

# Model ID'si: Resmi Qwen-Image veya topluluk versiyonu (örn: "Qwen/Qwen-Image")
# Not: Eğer özel bir "2512" repo'su kullanıyorsanız, HuggingFace'deki tam adını buraya yazın.
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen-Image")

pipe = None

def load_model():
    global pipe
    if pipe is None:
        print(f"🟢 [Lazy Load] {MODEL_ID} modeli yükleniyor... (İlk seferde indirme uzun sürebilir)")
        try:
            from diffusers import QwenImagePipeline
            
            pipe = QwenImagePipeline.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.bfloat16,
                cache_dir="/runpod-volume",
                variant="fp16" # Varsa daha hafif versiyonu indirir
            )
            
            # --- KRİTİK VRAM OPTİMİZASYONLARI ---
            # 1. xformers ile bellek verimli attention (Hız artırır, VRAM'i %30-40 düşürür)
            pipe.enable_xformers_memory_efficient_attention()
            
            # 2. CPU Offload: Modeli parça parça GPU'ya taşır. 24GB-40GB VRAM'de çalışmasını sağlar.
            pipe.enable_model_cpu_offload()
            # ------------------------------------
            
            pipe.to("cuda")
            print("🟢 Model başarıyla belleğe ve GPU'ya yüklendi! ✅")
            
        except Exception as e:
            print(f"🔴 HATA: Model yüklenirken çöktü! Detay: {e}")
            raise e
    return pipe

def handler(job):
    print("🟢 İstek alındı, model durumu kontrol ediliyor...")
    try:
        model = load_model()
        
        # RunPod job yapısını güvenli şekilde parse et
        input_data = job.get('input', {})
        prompt = input_data.get('prompt', 'a highly detailed cat, 4k, masterpiece, photorealistic')
        negative_prompt = input_data.get('negative_prompt', 'blurry, low quality, deformed, ugly')
        width = int(input_data.get('width', 1024))
        height = int(input_data.get('height', 1024))
        steps = int(input_data.get('steps', 30))
        guidance_scale = float(input_data.get('guidance_scale', 5.0))
        
        print(f"🟢 Görsel üretiliyor: {width}x{height}, {steps} adım. Prompt: {prompt[:50]}...")
        
        # Resim üretimi (Inference mode ile bellek sızıntısını önle)
        with torch.inference_mode():
            result = model(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance_scale
            )
        
        # 🔴 DÜZELTME: result.images bir listedir. İlk elemanı alınmalı.
        output_path = "/tmp/output.png"
        result.images[0].save(output_path)
        print(f"🟢 Görsel başarıyla kaydedildi: {output_path}")
        
        # RunPod API yanıtı için base64 formatına çevir
        buffered = BytesIO()
        result.images[0].save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return {
            "status": "success", 
            "output_path": output_path,
            "image_base64": img_base64,
            "message": "Görsel başarıyla oluşturuldu."
        }
        
    except Exception as e:
        print(f"🔴 Handler içinde kritik hata oluştu: {str(e)}")
        traceback.print_exc() # Hatanın tam yerini RunPod loglarında göster
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    print("🟢 RunPod Serverless başlatılıyor...")
    runpod.serverless.start({"handler": handler})
