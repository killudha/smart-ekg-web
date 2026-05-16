import os
import wfdb
import numpy as np
import tensorflow as tf
from sklearn.utils import resample
from scipy.signal import resample as scipy_resample, butter, filtfilt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# ============================================================
# KONFIGURASI
# ============================================================
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
data_path = os.path.join(os.path.expanduser('~'), 'Documents', 'CODING', 'ekg',
                         'mit-bih-arrhythmia-database-1.0.0')

train_records = ['101','106','108','109','112','114','115','116','118','119',
                 '122','124','201','203','205','207','208','209','215','220','223','230']
test_records  = ['100','103','104','105','111','113','117','121','123','200','202','210',
                 '212','213','214','219','221','222','228','231','232','233','234']

TARGET_FS   = 125
WINDOW_SIZE = 375
HALF_WINDOW = WINDOW_SIZE // 2

# DEFINISI 4 KELAS
CLASS_NAMES = ['NORMAL', 'TAKIKARDIA', 'BRADIKARDIA', 'ARITMIA']
NUM_CLASSES = 4

def apply_filter(signal):
    nyq  = 0.5 * TARGET_FS
    b, a = butter(4, [0.5 / nyq, 40.0 / nyq], btype='band')
    return filtfilt(b, a, signal)

# ============================================================
# EKSTRAKSI DATA DENGAN LOGIKA BPM LOKAL
# ============================================================
def extract_data(record_list, name="Data"):
    X_list, y_list = [], []
    print(f"\nEkstraksi {name}...")
    
    for rec in record_list:
        fp = os.path.join(data_path, rec)
        try:
            record  = wfdb.rdrecord(fp)
            ann     = wfdb.rdann(fp, 'atr')
            signals = record.p_signal[:, 0]

            if record.fs != TARGET_FS:
                n = int(len(signals) * TARGET_FS / record.fs)
                signals = scipy_resample(signals, n)
                samples = np.round(ann.sample * TARGET_FS / record.fs).astype(int)
            else:
                samples = ann.sample.copy()

            count = 0
            # Mulai dari indeks 1 agar bisa menghitung jarak R-R (kecepatan)
            for i in range(1, len(ann.symbol)):
                symbol = ann.symbol[i]
                
                # 1. HITUNG BPM LOKAL (Temporal Feature)
                rr_interval_samples = samples[i] - samples[i-1]
                if rr_interval_samples <= 0: continue
                rr_interval_sec = rr_interval_samples / TARGET_FS
                bpm = 60 / rr_interval_sec

                # 2. PENENTUAN LABEL (Mapping 4 Kelas)
                label = -1
                if symbol in ['V', 'E']:
                    label = 3 # ARITMIA (PVC)
                elif symbol in ['N', 'L', 'R', 'B', 'e', 'j', 'A', 'a', 'J', 'S']:
                    if bpm > 100:
                        label = 1 # TAKIKARDIA
                    elif bpm < 60:
                        label = 2 # BRADIKARDIA
                    else:
                        label = 0 # NORMAL
                
                if label == -1: continue

                # 3. POTONG SINYAL (Spatial Feature)
                pos = samples[i]
                if pos <= HALF_WINDOW or pos + HALF_WINDOW + 1 > len(signals):
                    continue
                
                patch = signals[pos - HALF_WINDOW : pos + HALF_WINDOW + 1]
                if len(patch) != WINDOW_SIZE: continue
                
                # Preprocessing
                patch_f = apply_filter(patch)
                p_min, p_max = patch_f.min(), patch_f.max()
                if p_max == p_min: continue
                patch_norm = (patch_f - p_min) / (p_max - p_min)
                
                X_list.append(patch_norm)
                y_list.append(label)
                count += 1
            print(f"  Record {rec}: {count} sampel.")
        except Exception as e:
            print(f"  [SKIP] Record {rec}: {e}")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    unique, counts = np.unique(y, return_counts=True)
    print(f"  Hasil Ekstraksi: { {CLASS_NAMES[u]: int(c) for u, c in zip(unique, counts)} }")
    return X, y

# ============================================================
# BALANCING (Target 10.000 per kelas)
# ============================================================
def balance_data(X, y, target=10000):
    X_bal, y_bal = [], []
    for i in range(NUM_CLASSES):
        idx = np.where(y == i)[0]
        if len(idx) == 0: continue
        
        # Oversampling/Undersampling
        actual = target
        replace = actual > len(idx)
        chosen = resample(idx, replace=replace, n_samples=actual, random_state=42)
        
        for sid in chosen:
            sig = X[sid].copy()
            # Augmentasi sedikit jika oversampling
            if replace and np.random.rand() > 0.5:
                sig += np.random.normal(0, 0.01, sig.shape)
            X_bal.append(sig)
            y_bal.append(i)
    
    X_out = np.array(X_bal).reshape(-1, WINDOW_SIZE, 1)
    y_out = tf.keras.utils.to_categorical(y_bal, NUM_CLASSES)
    return X_out, y_out

# Proses Ekstraksi
X_tr_raw, y_tr_raw = extract_data(train_records, "Training")
X_te_raw, y_te_raw = extract_data(test_records, "Testing")

X_train, y_train = balance_data(X_tr_raw, y_tr_raw)
X_test = X_te_raw.reshape(-1, WINDOW_SIZE, 1)
y_test = tf.keras.utils.to_categorical(y_te_raw, NUM_CLASSES)

# ============================================================
# ARSITEKTUR 1D-CNN (Sesuai draf PNJ)
# ============================================================
def build_model():
    inp = tf.keras.Input(shape=(WINDOW_SIZE, 1))
    
    x = tf.keras.layers.Conv1D(64, 11, padding='same', activation='relu')(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    
    x = tf.keras.layers.Conv1D(128, 7, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    
    x = tf.keras.layers.Conv1D(256, 5, padding='same', activation='relu')(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # OUTPUT: 4 Kelas Softmax
    out = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
    return tf.keras.Model(inp, out)

model = build_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training
base_dir = os.path.dirname(os.path.abspath(__file__))
model.fit(X_train, y_train, epochs=50, batch_size=128, validation_data=(X_test, y_test),
          callbacks=[EarlyStopping(patience=10, restore_best_weights=True),
                     ModelCheckpoint(os.path.join(base_dir, "best_model.keras"), save_best_only=True)])

# ============================================================
# EXPORT ALL FORMAT
# ============================================================

import shutil
import sys
import types

# ============================================================
# 1. SAVE .KERAS
# ============================================================

keras_path = os.path.join(
    base_dir,
    "best_model.keras"
)

model.save(keras_path)

print("\n✅ best_model.keras DISIMPAN")

# ============================================================
# 2. SAVE .H5
# ============================================================

h5_path = os.path.join(
    base_dir,
    "best_model.h5"
)

model.save(h5_path)

print("✅ best_model.h5 DISIMPAN")

# ============================================================
# 3. EXPORT TFLITE
# ============================================================

print("\n🔄 Exporting TFLite...")

converter = tf.lite.TFLiteConverter.from_keras_model(
    model
)

converter.optimizations = [
    tf.lite.Optimize.DEFAULT
]

tflite_model = converter.convert()

tflite_path = os.path.join(
    base_dir,
    "best_model.tflite"
)

with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print("✅ best_model.tflite DISIMPAN")

# ============================================================
# 4. EXPORT SAVEDMODEL
# ============================================================

saved_model_dir = os.path.join(
    base_dir,
    "saved_model_ekg"
)

if os.path.exists(saved_model_dir):
    shutil.rmtree(saved_model_dir)

print("\n🔄 Exporting SavedModel...")

tf.saved_model.save(
    model,
    saved_model_dir
)

print("✅ saved_model_ekg/ DISIMPAN")

# ============================================================
# 5. PATCH TFJS
# ============================================================

import numpy as np

if not hasattr(np, 'object'):
    np.object = object

if not hasattr(np, 'bool'):
    np.bool = bool

fake_hub = types.ModuleType(
    "tensorflow_hub"
)

sys.modules[
    "tensorflow_hub"
] = fake_hub

# ============================================================
# 6. EXPORT TFJS
# ============================================================

print("\n🔄 Exporting TFJS...")

import tensorflowjs as tfjs

tfjs_output_dir = os.path.join(
    base_dir,
    "tfjs_model"
)

if os.path.exists(tfjs_output_dir):
    shutil.rmtree(tfjs_output_dir)

tfjs.converters.convert_tf_saved_model(
    saved_model_dir,
    tfjs_output_dir
)

print("✅ tfjs_model/ DISIMPAN")

# ============================================================
# DONE
# ============================================================

print("\n===================================")
print("✅ EXPORT SEMUA FORMAT BERHASIL")
print("===================================")

print("\nOUTPUT:")

print("1. best_model.keras")
print("2. best_model.h5")
print("3. best_model.tflite")
print("4. saved_model_ekg/")
print("5. tfjs_model/")