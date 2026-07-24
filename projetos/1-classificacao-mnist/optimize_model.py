import tensorflow as tf
import os

# ---------------------------------------------------------------------------
# Projeto 1 — Otimização do Modelo (MNIST)
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o modelo treinado em "model.h5"
#   2. Converter para TensorFlow Lite usando tf.lite.TFLiteConverter
#   3. Aplicar uma técnica de otimização (ex: Dynamic Range Quantization,
#      via converter.optimizations = [tf.lite.Optimize.DEFAULT])
#   4. Salvar o resultado como "model.tflite"
# ---------------------------------------------------------------------------

import os
import tensorflow as tf

script_dir = os.path.dirname(os.path.abspath(__file__))
h5_path = os.path.join(script_dir, "model.h5")
tflite_path = os.path.join(script_dir, "model.tflite")

# Carregar o modelo treinado
model = tf.keras.models.load_model(h5_path)

# Instanciar o conversor
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Opcional: habilitar otimização
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Converter o modelo
tflite_model = converter.convert()

# Salvar o modelo
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print(f"Modelo convertido para TensorFlow Lite e salvo como {tflite_path}")

