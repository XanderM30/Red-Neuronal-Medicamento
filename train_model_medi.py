import os
import sys
import pickle
import hashlib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, GRU, Dense, LSTM
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelBinarizer
import firebase_admin
from firebase_admin import credentials, firestore
import json
import matplotlib.pyplot as plt

# -------------------------
# CONFIGURACIÓN FIREBASE
# -------------------------
cred_path = os.path.join(os.getcwd(), "firebase_config/serviceAccountKey.json")
if not os.path.exists(cred_path):
    print("❌ No se encontró el archivo de credenciales de Firebase.")
    sys.exit(1)

if not firebase_admin._apps:
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred)

db = firestore.client()

# -------------------------
# FUNCIONES
# -------------------------
def obtener_datos(text_field='descripcion'):
    """Obtiene los datos de Firestore"""
    docs = db.collection("medicamentos").stream()
    X_raw = []
    Y_raw = {}

    for doc in docs:
        data = doc.to_dict()
        if text_field not in data:
            continue
        X_raw.append(str(data[text_field]))
        for key, value in data.items():
            if key != text_field:
                Y_raw.setdefault(key, []).append(value)

    if not X_raw:
        print("⚠️ No se encontraron datos en Firebase.")
        sys.exit(0)

    return X_raw, Y_raw


def hash_datos(X_raw, Y_raw):
    """Genera un hash para detectar cambios en los datos"""
    m = hashlib.md5()
    for x in X_raw:
        m.update(x.encode('utf-8'))
    for key in sorted(Y_raw.keys()):
        for y in Y_raw[key]:
            m.update(str(y).encode('utf-8'))
    return m.hexdigest()


def generar_tflite(model):
    """Convierte el modelo Keras actual a formato TFLite y lo guarda."""
    print("⚙️ Generando modelo TFLite...")

    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        # ✅ Permitir tanto operaciones estándar como Select TF Ops (necesario para LSTM)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        converter._experimental_lower_tensor_list_ops = False
        converter.experimental_new_converter = True

        # ✅ Optimización para móviles
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        tflite_model = converter.convert()

        with open(TFLITE_FILE, "wb") as f:
            f.write(tflite_model)

        print("✅ Modelo TFLite generado correctamente (con soporte SELECT_TF_OPS).")
        print("⚙️ Asegúrate de incluir 'tensorflow-lite-select-tf-ops' en tu app Flutter o Android.")

    except Exception as e:
        print(f"❌ Error al convertir a TFLite: {e}")
        sys.exit(1)


def cargar_pickle_seguro(path):
    """Carga un archivo pickle de forma segura, detectando corrupción o vacío."""
    if not os.path.exists(path):
        print(f"⚠️ No se encontró {path}, será regenerado.")
        return None
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if obj is None:
            raise ValueError("Archivo vacío.")
        return obj
    except Exception as e:
        print(f"⚠️ Error al cargar {path}: {e}. Será regenerado.")
        return None


# -------------------------
# ARCHIVOS LOCALES
# -------------------------
HASH_FILE = "data_hash.txt"
MODEL_FILE = "modelo_medicamentos_dinamico.h5"
TFLITE_FILE = "modelo_medicamentos_dinamico.tflite"
TOKENIZER_FILE = "tokenizer.pkl"
ENCODERS_FILE = "label_encoders.pkl"
VERSION_FILE = "version.txt"
TOKENIZER_JSON = "tokenizer.json"
ENCODERS_JSON = "label_encoders.json"
REPORT_FILE = "training_report.json"

# -------------------------
# OBTENER Y PROCESAR DATOS
# -------------------------
X_raw, Y_raw = obtener_datos()
current_hash = hash_datos(X_raw, Y_raw)
previous_hash = ""
if os.path.exists(HASH_FILE):
    with open(HASH_FILE, "r") as f:
        previous_hash = f.read().strip()

# -------------------------
# ENTRENAMIENTO O CARGA
# -------------------------
if current_hash == previous_hash and os.path.exists(MODEL_FILE):
    print("✅ No hay cambios en Firebase. Cargando modelo existente...")
    model = tf.keras.models.load_model(MODEL_FILE)
    tokenizer = cargar_pickle_seguro(TOKENIZER_FILE)
if tokenizer is None:
    print("🔁 Regenerando tokenizer...")
    tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token="<UNK>")
    tokenizer.fit_on_texts(X_raw)

label_encoders = cargar_pickle_seguro(ENCODERS_FILE)
if label_encoders is None:
    label_encoders = {}

else:
    print("⚡ Cambios detectados. Entrenando modelo...")

    # TOKENIZACIÓN
    tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token="<UNK>")
    tokenizer.fit_on_texts(X_raw)
    X_seq = tokenizer.texts_to_sequences(X_raw)
    max_len = max(len(seq) for seq in X_seq)
    X_pad = pad_sequences(X_seq, maxlen=max_len, padding='post')

    # CODIFICACIÓN DE SALIDAS
    label_encoders = {}
    Y_enc = []
    for key, values in Y_raw.items():
        lb = LabelBinarizer()
        encoded = lb.fit_transform(values)
        if encoded.ndim == 1:
            encoded = np.expand_dims(encoded, axis=-1)
        Y_enc.append(encoded)
        label_encoders[key] = lb

    # MODELO (se reemplaza GRU → LSTM por compatibilidad TFLite)
    vocab_size = len(tokenizer.word_index) + 1
    embedding_dim = 64

    input_layer = Input(shape=(max_len,))
    x = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(input_layer)
    x = LSTM(128, return_sequences=False)(x)
    x = Dense(64, activation='relu')(x)

    outputs = []
    for key, y in zip(Y_raw.keys(), Y_enc):
        activation = 'softmax' if y.shape[1] > 1 else 'sigmoid'
        outputs.append(Dense(y.shape[1], activation=activation, name=key)(x))

    model = Model(inputs=input_layer, outputs=outputs)
    losses = [
        'categorical_crossentropy' if y.shape[1] > 1 else 'binary_crossentropy'
        for y in Y_enc
    ]
    model.compile(optimizer='adam', loss=losses, metrics=['accuracy'] * len(Y_enc))

    # ENTRENAMIENTO CON VALIDACIÓN
    history = model.fit(
        X_pad, Y_enc,
        validation_split=0.2,
        epochs=15,
        batch_size=16,
        verbose=1
    )

    # GUARDAR MODELO Y OBJETOS
    model.save(MODEL_FILE)
    with open(TOKENIZER_FILE, "wb") as f:
        pickle.dump(tokenizer, f)
    with open(ENCODERS_FILE, "wb") as f:
        pickle.dump(label_encoders, f)
    with open(HASH_FILE, "w") as f:
        f.write(current_hash)

    print("✅ Modelo entrenado y guardado correctamente.")

    # 📊 GUARDAR MÉTRICAS
    report = {
        "epochs": len(history.history["loss"]),
        "batch_size": 16,
        "vocab_size": vocab_size,
        "max_len": int(max_len),
        "loss": [float(x) for x in history.history["loss"]],
        "val_loss": [float(x) for x in history.history["val_loss"]],
        "accuracy": [float(x) for x in history.history["accuracy"]],
        "val_accuracy": [float(x) for x in history.history["val_accuracy"]],
    }
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"📈 Reporte de entrenamiento guardado en {REPORT_FILE}")

    # GRAFICAR Y GUARDAR PNG
    plt.figure(figsize=(10, 5))
    plt.plot(history.history["accuracy"], label="Entrenamiento", color='blue')
    plt.plot(history.history["val_accuracy"], label="Validación", color='orange')
    plt.title("Evolución de la Precisión")
    plt.xlabel("Épocas")
    plt.ylabel("Precisión")
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_accuracy.png")
    plt.close()
    print("📊 Gráfica de entrenamiento guardada en training_accuracy.png")

# -------------------------
# VALIDAR O GENERAR TFLITE
# -------------------------
print("🔍 Verificando modelo TFLite...")
try:
    interpreter = tf.lite.Interpreter(model_path=TFLITE_FILE)
    interpreter.allocate_tensors()
    print("✅ Modelo TFLite válido.")
except Exception as e:
    print(f"⚠️ Modelo TFLite inválido: {e}")
    generar_tflite(model)

# -------------------------
# VERSIONADO
# -------------------------
version_number = 0
if os.path.exists(VERSION_FILE):
    with open(VERSION_FILE, "r") as f:
        try:
            version_number = int(f.read().strip())
        except ValueError:
            version_number = 0
version_number += 1
with open(VERSION_FILE, "w") as f:
    f.write(str(version_number))
print(f"🧾 Nueva versión generada: {version_number}")

## -------------------------
# EXPORTAR A JSON
# -------------------------
if tokenizer is not None:
    with open(TOKENIZER_JSON, "w", encoding="utf-8") as f:
        json.dump(tokenizer.word_index, f, ensure_ascii=False, indent=2)
    print(f"✅ Tokenizer exportado a {TOKENIZER_JSON}")
else:
    print("⚠️ No se pudo exportar el tokenizer (no disponible o dañado).")

if label_encoders:
    encoders_export = {key: list(lb.classes_) for key, lb in label_encoders.items()}
    with open(ENCODERS_JSON, "w", encoding="utf-8") as f:
        json.dump(encoders_export, f, ensure_ascii=False, indent=2)
    print(f"✅ Label encoders exportados a {ENCODERS_JSON}")
else:
    print("⚠️ No se pudieron exportar los label encoders.")




# -------------------------
# AVISO FINAL
# -------------------------
print("\n🚀 El modelo está listo para subir a GitHub.")
print(f"   - {MODEL_FILE}")
print(f"   - {TFLITE_FILE}")
print(f"   - {TOKENIZER_JSON}")
print(f"   - {ENCODERS_JSON}")
print(f"   - {REPORT_FILE}")
print(f"   - Versión: {version_number}")
print("\n🎯 Entrenamiento y métricas completadas con éxito.")
