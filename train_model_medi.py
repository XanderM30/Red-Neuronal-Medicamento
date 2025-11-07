import os
import sys
import pickle
import hashlib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, GRU, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelBinarizer
import firebase_admin
from firebase_admin import credentials, firestore
import json

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

        # ✅ Soporte extendido (para GRU/LSTM y TensorList)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        converter._experimental_lower_tensor_list_ops = False
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        tflite_model = converter.convert()

        with open(TFLITE_FILE, "wb") as f:
            f.write(tflite_model)

        print("✅ Modelo TFLite generado correctamente y compatible con Flutter.")

    except Exception as e:
        print(f"❌ Error al convertir a TFLite: {e}")
        sys.exit(1)

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

# -------------------------
# OBTENER Y PROCESAR DATOS
# -------------------------
X_raw, Y_raw = obtener_datos()
current_hash = hash_datos(X_raw, Y_raw)

# -------------------------
# VERIFICAR CAMBIOS
# -------------------------
if os.path.exists(HASH_FILE):
    with open(HASH_FILE, "r") as f:
        previous_hash = f.read().strip()
else:
    previous_hash = ""

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
# DECISIÓN: USAR MODELO EXISTENTE O ENTRENAR NUEVO
# -------------------------
if current_hash == previous_hash and os.path.exists(MODEL_FILE):
    print("✅ No hay cambios en Firebase. Intentando cargar modelo existente...")

    model = None
    tokenizer = None
    label_encoders = None

    try:
        model = tf.keras.models.load_model(MODEL_FILE)
    except Exception as e:
        print(f"⚠️ Error al cargar modelo H5: {e}. Será reentrenado.")

    tokenizer = cargar_pickle_seguro(TOKENIZER_FILE)
    label_encoders = cargar_pickle_seguro(ENCODERS_FILE)

    # Si alguno falla, reentrenamos
    if model is None or tokenizer is None or label_encoders is None:
        print("🔁 Archivos incompletos o corruptos. Se reentrenará el modelo.")
        previous_hash = ""  # Forzamos reentrenamiento
else:
    print("⚡ Cambios detectados en Firebase. Entrenando modelo...")


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

    # MODELO GRU
    vocab_size = len(tokenizer.word_index) + 1
    embedding_dim = 64
    input_layer = Input(shape=(max_len,))
    x = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(input_layer)
    x = GRU(128, return_sequences=False)(x)
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

    # ENTRENAMIENTO
    model.fit(X_pad, Y_enc, epochs=15, batch_size=16, verbose=1)

    # GUARDAR MODELO Y OBJETOS
    model.save(MODEL_FILE)
    with open(TOKENIZER_FILE, "wb") as f:
        pickle.dump(tokenizer, f)
    with open(ENCODERS_FILE, "wb") as f:
        pickle.dump(label_encoders, f)
    with open(HASH_FILE, "w") as f:
        f.write(current_hash)

    print("✅ Modelo entrenado y guardado correctamente.")

# -------------------------
# GENERAR O VALIDAR TFLITE
# -------------------------
print("🔍 Verificando o generando modelo TFLite...")

if not os.path.exists(TFLITE_FILE):
    generar_tflite(model)
else:
    try:
        interpreter = tf.lite.Interpreter(model_path=TFLITE_FILE)
        interpreter.allocate_tensors()
        print("✅ Modelo TFLite actual válido.")
    except Exception as e:
        print(f"⚠️ Modelo TFLite inválido: {e}")
        generar_tflite(model)

# -------------------------
# CONTROL DE VERSIONES
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

# Guardar tokenizer en JSON (para Flutter)
tokenizer_json_file = "tokenizer.json"

# -------------------------
# VERIFICAR EXISTENCIA DE TOKENIZER Y ENCODERS
# -------------------------
if 'tokenizer' not in locals() or tokenizer is None:
    print("⚠️ Tokenizer no encontrado en memoria. Regenerando a partir del modelo existente...")
    from tensorflow.keras.preprocessing.text import Tokenizer
    tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token="<UNK>")
    tokenizer.fit_on_texts(X_raw)
    print("✅ Tokenizer regenerado correctamente.")

if 'label_encoders' not in locals() or label_encoders is None:
    print("⚠️ Label encoders no encontrados en memoria. Regenerando...")
    from sklearn.preprocessing import LabelBinarizer
    label_encoders = {}
    for key, values in Y_raw.items():
        lb = LabelBinarizer()
        lb.fit(values)
        label_encoders[key] = lb
    print("✅ Label encoders regenerados correctamente.")

# -------------------------
# EXPORTAR TOKENIZER Y LABELS A JSON
# -------------------------
# Tokenizer (para Flutter)
with open(TOKENIZER_JSON, "w", encoding="utf-8") as f:
    json.dump(tokenizer.word_index, f, ensure_ascii=False, indent=2)
print(f"✅ Tokenizer exportado a {TOKENIZER_JSON}")

# Label encoders (para Flutter)
encoders_export = {key: list(lb.classes_) for key, lb in label_encoders.items()}
with open(ENCODERS_JSON, "w", encoding="utf-8") as f:
    json.dump(encoders_export, f, ensure_ascii=False, indent=2)
print(f"✅ Label encoders exportados a {ENCODERS_JSON}")

# -------------------------
# AVISO FINAL
# -------------------------
print("\n🚀 El modelo está listo para subir a GitHub.")
print(f"   - {MODEL_FILE}")
print(f"   - {TFLITE_FILE}")
print(f"   - {TOKENIZER_JSON}")
print(f"   - {ENCODERS_JSON}")
print(f"   - Versión: {version_number}")
print("\n👉 Ejecuta manualmente:")
print("   git add .")
print(f'   git commit -m "Modelo actualizado versión {version_number}"')
print("   git push origin main\n")
print("🎯 Entrenamiento, validación y generación completados con éxito.")
