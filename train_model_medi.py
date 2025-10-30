import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelBinarizer
import firebase_admin
from firebase_admin import credentials, firestore
import pickle
import hashlib

# -------------------------
# Configuración Firebase
# -------------------------
cred_path = os.path.join(os.getcwd(), "firebase_config/serviceAccountKey.json")
cred = credentials.Certificate(cred_path)
firebase_admin.initialize_app(cred)
db = firestore.client()

# -------------------------
# Función para obtener datos
# -------------------------
def obtener_datos(text_field='descripcion'):
    docs = db.collection("medicamentos").stream()
    X_raw = []
    Y_raw = {}

    for doc in docs:
        data = doc.to_dict()
        X_raw.append(data[text_field])
        for key, value in data.items():
            if key != text_field:
                if key not in Y_raw:
                    Y_raw[key] = []
                Y_raw[key].append(value)
    return X_raw, Y_raw

# -------------------------
# Función para detectar cambios en los datos
# -------------------------
def hash_datos(X_raw, Y_raw):
    m = hashlib.md5()
    for x in X_raw:
        m.update(x.encode('utf-8'))
    for key in sorted(Y_raw.keys()):
        for y in Y_raw[key]:
            m.update(str(y).encode('utf-8'))
    return m.hexdigest()

# -------------------------
# Archivos para guardar estado
# -------------------------
HASH_FILE = "data_hash.txt"
MODEL_FILE = "modelo_medicamentos_dinamico.h5"
TOKENIZER_FILE = "tokenizer.pkl"
ENCODERS_FILE = "label_encoders.pkl"

# -------------------------
# Obtener datos y hash actual
# -------------------------
X_raw, Y_raw = obtener_datos()
current_hash = hash_datos(X_raw, Y_raw)

# Verificar hash previo
if os.path.exists(HASH_FILE):
    with open(HASH_FILE, "r") as f:
        previous_hash = f.read().strip()
else:
    previous_hash = ""

if current_hash == previous_hash and os.path.exists(MODEL_FILE):
    print("✅ No hay cambios en Firebase. Se mantiene la red anterior.")
else:
    print("⚡ Cambios detectados en Firebase. Se actualizará y entrenará la red neuronal...")

    # -------------------------
    # Tokenización
    # -------------------------
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(X_raw)
    X_seq = tokenizer.texts_to_sequences(X_raw)
    max_len = max(len(seq) for seq in X_seq)
    X_pad = pad_sequences(X_seq, maxlen=max_len, padding='post')

    # -------------------------
    # Codificación automática de salidas
    # -------------------------
    label_encoders = {}
    Y_enc = []
    for key, values in Y_raw.items():
        lb = LabelBinarizer()
        Y_enc.append(lb.fit_transform(values))
        label_encoders[key] = lb

    # -------------------------
    # Crear modelo
    # -------------------------
    vocab_size = len(tokenizer.word_index) + 1
    embedding_dim = 50
    input_layer = Input(shape=(max_len,))
    x = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(input_layer)
    x = LSTM(64)(x)

    outputs = []
    for key, y in zip(Y_raw.keys(), Y_enc):
        activation = 'softmax' if y.shape[1] > 1 else 'sigmoid'
        outputs.append(Dense(y.shape[1], activation=activation, name=key)(x))

    model = Model(inputs=input_layer, outputs=outputs)

    # -------------------------
    # Compilar modelo
    # -------------------------
    model.compile(
        optimizer='adam',
        loss=['categorical_crossentropy' if y.shape[1] > 1 else 'binary_crossentropy' for y in Y_enc],
        metrics=['accuracy'] * len(Y_enc)
    )

    # -------------------------
    # Entrenamiento
    # -------------------------
    model.fit(
        X_pad,
        Y_enc,
        epochs=10,
        batch_size=16
    )

    # -------------------------
    # Guardar modelo y estado
    # -------------------------
    model.save(MODEL_FILE)
    with open(TOKENIZER_FILE, "wb") as f:
        pickle.dump(tokenizer, f)
    with open(ENCODERS_FILE, "wb") as f:
        pickle.dump(label_encoders, f)
    with open(HASH_FILE, "w") as f:
        f.write(current_hash)

    print("✅ Modelo entrenado y guardado.")

    # -------------------------
    # Convertir a TFLite
    # -------------------------
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Para ops no soportadas:
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                           tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_model = converter.convert()
    with open("modelo_medicamentos_dinamico.tflite", "wb") as f:
        f.write(tflite_model)

    print("✅ Modelo convertido a TFLite (con select TF ops si es necesario).")
