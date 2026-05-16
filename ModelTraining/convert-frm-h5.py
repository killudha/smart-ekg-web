import os
import sys
import types
import numpy as np

# ============================================================
# 1. PATCH NUMPY & BLOKIR HUB
# ============================================================
if not hasattr(np, 'object'): np.object = object
if not hasattr(np, 'bool'): np.bool = bool

# Blokir tensorflow_hub agar tidak mencari estimator yang hilang
class DummyKerasLayer: pass
fake_hub = types.ModuleType('tensorflow_hub')
fake_hub.KerasLayer = DummyKerasLayer
sys.modules['tensorflow_hub'] = fake_hub

print("✅ Patch Berhasil: tensorflow_hub diblokir!")

import tensorflowjs as tfjs

# ============================================================
# 2. PATH LANGSUNG KE FOLDER WEB KAMU
# ============================================================
# Kita gunakan SavedModel untuk mem-bypass error Keras 3
saved_model_path = r"C:\Users\Dhanny\Documents\CODING\ekg\ekg by megumi\saved_model_ekg"
output_dir = r"C:\Users\Dhanny\Documents\CODING\smart-ekg-web\tfjs_model"

print("\n--- Memulai Konversi dari SavedModel ke TFJS (GraphModel) ---")

try:
    if not os.path.exists(saved_model_path):
        print(f"❌ ERROR: Folder SavedModel tidak ditemukan di: {saved_model_path}")
    else:
        print("📦 Memuat TensorFlow Graph (Bypass Keras 3 Check)...")
        
        # Gunakan convert_tf_saved_model (GraphModel), bukan KerasModel
        tfjs.converters.convert_tf_saved_model(
            saved_model_path, 
            output_dir
        )
        
        print("\n✅ ALHAMDULILLAH BERHASIL!")
        print(f"Model GraphModel siap dipakai di:\n{output_dir}")
        print("Silakan langsung refresh (Ctrl+F5) web dashboard kamu!")

except Exception as e:
    print(f"\n❌ GAGAL KONVERSI: {e}")

print("\n=== PROSES SELESAI ===")