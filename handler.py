import runpod
import torch
import sys

print("🟢 1. ADIM: Script başladı.")

try:
    from diffusers import DiffusionPipeline
    print("🟢 2. ADIM: Diffusers başarıyla import edildi.")
except Exception as e:
    print(f"🔴 HATA: Diffusers import edilemedi! {e}")
    sys.exit(1)

# DOĞRU VE EN GÜNCEL MODEL KİMLİĞİ
MODEL_ID = "Qwen/Qwen-Image-2512"
print(f"🟢 3. ADIM: {MODEL_ID} modeli indiriliyor/yükleniyor (Bu 3-5 dakika sürebilir, lütfen bekleyin...)")

try:
    # Qwen-Image-2512 resmi dokümantasyonuna göre bfloat16 ve DiffusionPipeline kullanılır
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        cache_dir="/runpod-volume",
        trust_remote_code=True
    ).to("cuda")
    print("🟢 4. ADIM: Model başarıyla GPU'ya yüklendi! ✅ Artık istek kabul edebilir.")
except Exception as e:
    print(f"🔴 HATA: Model yüklenirken çöktü! {e}")
    sys.exit(1)

def handler(job):
    print(f"🟢 5. ADIM: İstek alındı. Prompt: {job['input'].get('prompt', 'test')}")
    try:
        result = pipe(
            prompt=job['input'].get('prompt', 'a highly detailed cat, 4k, masterpiece'),
            negative_prompt=" ",  # Qwen için tek boşluk karakteri önerilir
            width=1024,
            height=1024,
            num_inference_steps=30,
            true_cfg_scale=4.0,
            generator=torch.Generator(device="cuda").manual_seed(42)
        )
        result.images[0].save("/tmp/output.png")
        print("🟢 6. ADIM: Görsel başarıyla oluşturuldu ve kaydedildi.")
        return {"status": "success", "output_path": "/tmp/output.png"}
    except Exception as e:
        print(f"🔴 HATA: Görsel oluşturma sırasında hata: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    print("🟢 7. ADIM: Runpod serverless başlatılıyor...")
    runpod.serverless.start({"handler": handler})
