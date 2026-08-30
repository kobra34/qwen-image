import sys
import types
import torch

# --- Gelişmiş XPU Yaması (Bütün alt metotları taklit eder) ---
if not hasattr(torch, "xpu"):
    class MockXPU:
        def __init__(self):
            self.__name__ = "torch.xpu"
        def is_available(self): return False
        def device_count(self): return 0
        def manual_seed(self, seed): pass
        def manual_seed_all(self, seed): pass
        def empty_cache(self): pass
        def current_device(self): return 0
        def get_device_name(self, device=None): return ""
        def set_device(self, device): pass

    xpu_mock = MockXPU()
    torch.xpu = xpu_mock
    sys.modules["torch.xpu"] = xpu_mock
# -------------------------------------------------------------

import runpod
print("🟢 1. ADIM: Script başladı ve Gelişmiş XPU yaması uygulandı.")

pipe = None
MODEL_ID = "Qwen/Qwen-Image-2512"

def load_model():
    global pipe
    if pipe is None:
        print(f"🟢 [Lazy Load] {MODEL_ID} modeli yükleniyor...")
        try:
            from diffusers import QwenImagePipeline
            
            pipe = QwenImagePipeline.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.bfloat16,
                cache_dir="/runpod-volume"
            ).to("cuda")
            
            print("🟢 Model başarıyla belleğe ve GPU'ya yüklendi! ✅")
        except Exception as e:
            print(f"🔴 HATA: Model yüklenirken çöktü! Detay: {e}")
            raise e
    return pipe

def handler(job):
    print("🟢 İstek alındı, model durumu kontrol ediliyor...")
    try:
        model = load_model()
        input_data = job['input']
        prompt = input_data.get('prompt', 'a highly detailed cat, 4k, masterpiece')
        
        print(f"🟢 Görsel üretiliyor. Prompt: {prompt}")
        
        result = model(
            prompt=prompt,
            negative_prompt=input_data.get('negative_prompt', ''),
            width=input_data.get('width', 1024),
            height=input_data.get('height', 1024),
            num_inference_steps=input_data.get('steps', 50),
            true_cfg_scale=4.0
        )
        
        output_path = "/tmp/output.png"
        result.images.save(output_path)
        
        print("🟢 Görsel başarıyla kaydedildi.")
        return {"status": "success", "output_path": output_path}
        
    except Exception as e:
        print(f"🔴 Handler içinde hata oluştu: {str(e)}")
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    print("🟢 Runpod serverless başlatılıyor...")
    runpod.serverless.start({"handler": handler})
