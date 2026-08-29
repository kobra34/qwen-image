import runpod
import torch
import sys

print("🟢 1. ADIM: Script başladı.")

try:
    from diffusers import QwenImagePipeline
    print("🟢 2. ADIM: Diffusers başarıyla import edildi.")
except Exception as e:
    print(f"🔴 HATA: Diffusers import edilemedi! {e}")
    sys.exit(1)

MODEL_ID = "Qwen/Qwen-Image"
print(f"🟢 3. ADIM: {MODEL_ID} modeli indiriliyor/yükleniyor (Bu 3-5 dakika sürebilir, lütfen bekleyin...)")

try:
    pipe = QwenImagePipeline.from_pretrained(
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
            prompt=job['input'].get('prompt', 'a cat'),
            negative_prompt=" ",
            width=1024,
            height=1024,
            num_inference_steps=20, # Hızlı test için 20
            true_cfg_scale=4.0,
            generator=torch.Generator(device="cuda").manual_seed(42)
        )
        result.images[0].save("/tmp/output.png")
        return {"status": "success", "output": "/tmp/output.png"}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    print("🟢 6. ADIM: Runpod serverless başlatılıyor...")
    runpod.serverless.start({"handler": handler})
