import runpod
import torch
import sys
import os

print("🟢 1. ADIM: Script başladı.")

# Global pipeline değişkeni (İlk istek geldiğinde doldurulacak)
pipe = None
MODEL_ID = "Qwen/Qwen-Image-2512"

def load_model():
    global pipe
    if pipe is None:
        print(f"🟢 [Lazy Load] {MODEL_ID} modeli yükleniyor... Bu işlem ilk istekte birkaç dakika sürebilir.")
        try:
            # Qwen-Image için özelleştirilmiş Pipeline import ediliyor
            from diffusers import QwenImagePipeline
            
            pipe = QwenImagePipeline.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.bfloat16,
                cache_dir="/runpod-volume",
                device_map="auto" # VRAM optimizasyonu için otomatik yerleşim
            )
            print("🟢 Model başarıyla belleğe ve GPU'ya yüklendi! ✅")
        except Exception as e:
            print(f"🔴 HATA: Model yüklenirken çöktü! Detay: {e}")
            raise e
    return pipe

def handler(job):
    print("🟢 İstek alındı, model durumu kontrol ediliyor...")
    try:
        # Modeli güvenli bir şekilde fonksiyon içinde ayağa kaldırıyoruz
        model = load_model()
        
        input_data = job['input']
        prompt = input_data.get('prompt', 'a highly detailed cat, 4k, masterpiece')
        
        print(f"🟢 Görsel üretiliyor. Prompt: {prompt}")
        
        # Qwen-Image-2512 için resmi önerilen parametreler
        result = model(
            prompt=prompt,
            negative_prompt=input_data.get('negative_prompt', ''),
            width=input_data.get('width', 1024),
            height=input_data.get('height', 1024),
            num_inference_steps=input_data.get('steps', 50), # Resmi önerilen adım: 50
            true_cfg_scale=4.0
        )
        
        output_path = "/tmp/output.png"
        result.images[0].save(output_path)
        
        print("🟢 Görsel başarıyla kaydedildi.")
        return {"status": "success", "output_path": output_path}
        
    except Exception as e:
        print(f"🔴 Handler içinde hata oluştu: {str(e)}")
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    print("🟢 Runpod serverless başlatılıyor... (Model ilk istek geldiğinde arka planda güvenle indirilecek)")
    # Sistem anında ayağa kalkacak, RunPod 60 saniye aşımından dolayı kapatmayacak
    runpod.serverless.start({"handler": handler})
