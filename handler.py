import runpod
import torch
import os

# Qwen-Image için özel olarak geliştirilmiş pipeline sınıfını kullanıyoruz
try:
    from diffusers import QwenImagePipeline
except ImportError:
    # Eğer sürüm çok yeniyse ve doğrudan import edilemiyorsa fallback
    from diffusers import DiffusionPipeline as QwenImagePipeline

MODEL_ID = "Qwen/Qwen-Image"

print(f"🚀 {MODEL_ID} modeli yükleniyor (bu işlem ilk seferde 2-5 dakika sürebilir)...")

try:
    # Qwen-Image resmi dokümantasyonuna göre bfloat16 kullanmalıdır
    pipe = QwenImagePipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16, 
        cache_dir="/runpod-volume",
        trust_remote_code=True # Özel model mimarisi için zorunlu
    ).to("cuda")
    print("✅ Model başarıyla yüklendi ve GPU'ya taşındı!")
    
except Exception as e:
    print(f"❌ Model yüklenirken KRİTİK hata: {str(e)}")
    raise e

def handler(job):
    try:
        job_input = job.get("input", {})
        prompt = job_input.get("prompt", "A beautiful landscape, highly detailed, 4k resolution")
        
        # Qwen-Image için resmi dokümantasyonda önerilen negative prompt tek bir boşluktur
        negative_prompt = " "
        
        # Qwen-Image için önerilen "sihirli" eklenti (kaliteyi artırır)
        enhanced_prompt = prompt + ", Ultra HD, 4K, cinematic composition."
        
        print(f"🎨 Görsel oluşturuluyor. Prompt: {enhanced_prompt}")
        
        # Qwen-Image'e özgü parametrelerle görsel üretimi
        result = pipe(
            prompt=enhanced_prompt,
            negative_prompt=negative_prompt,
            width=1024,
            height=1024,
            num_inference_steps=30,      # Test için 30 adım yeterli ve hızlıdır
            true_cfg_scale=4.0,          # Qwen-Image için önerilen değer
            generator=torch.Generator(device="cuda").manual_seed(42)
        )
        
        image = result.images[0]
        output_path = "/tmp/output.png"
        image.save(output_path)
        print(f"✅ Görsel başarıyla kaydedildi: {output_path}")
        
        return {
            "status": "success",
            "output_path": output_path,
            "prompt_used": prompt
        }
        
    except Exception as e:
        print(f"❌ Görsel oluşturma sırasında hata: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
