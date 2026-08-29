import runpod
import torch
import os
from diffusers import DiffusionPipeline

# 1. MODEL BİLGİLERİ
# ÖNEMLİ: Buraya Hugging Face'deki modelin TAM ve DOĞRU adını yazmalısın.
# Eğer model adı yanlışsa, aşağıdaki hata tam olarak bu yüzden terjadi.
# Örnek: "Qwen/Qwen-Image" veya "black-forest-labs/FLUX.1-schnell" vb.
MODEL_ID = "Qwen/Qwen-Image"  # <-- BURAYI KONTROL ET VE DOĞRU MODEL ADINI YAZ

print(f"🚀 {MODEL_ID} modeli yükleniyor, lütfen bekleyin (bu işlem ilk seferde uzun sürebilir)...")

try:
    # trust_remote_code=True, özel model mimarilerinin çalışması için ZORUNLUDUR
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        cache_dir="/runpod-volume",
        use_safetensors=True,
        trust_remote_code=True  # <-- KRİTİK EKLEME
    ).to("cuda")
    print("✅ Model başarıyla yüklendi ve GPU'ya taşındı!")
    
except Exception as e:
    print(f"❌ Model yüklenirken KRİTİK hata oluştu: {str(e)}")
    raise e

# 2. RUNPOD HANDLER FONKSİYONU
def handler(job):
    try:
        job_input = job.get("input", {})
        prompt = job_input.get("prompt", "a beautiful landscape, highly detailed, 4k resolution")
        print(f"🎨 Görsel oluşturuluyor. Prompt: {prompt}")
        
        # Görsel oluşturma
        result = pipe(
            prompt=prompt,
            num_inference_steps=20, # Gerekirse ayarlanabilir
            guidance_scale=7.5      # Gerekirse ayarlanabilir
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
        print(f"❌ Görsel oluşturma (handler) sırasında hata: {str(e)}")
        return {"error": str(e)}

# 3. SUNUCUYU BAŞLAT
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
