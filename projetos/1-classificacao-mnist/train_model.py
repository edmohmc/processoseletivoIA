import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

script_dir = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Projeto 1 — Classificação MNIST
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o dataset MNIST via tf.keras.datasets.mnist
#   2. Normalizar as imagens para [0, 1] e ajustar o shape para (28, 28, 1)
#   3. Separar um conjunto de validação (ex: validation_split ou split manual)
#   4. Construir uma CNN com 3-4 blocos Conv2D + BatchNormalization + MaxPooling2D,
#      seguida de Dropout antes da camada de saída (10 classes, softmax)
#   5. Treinar com EarlyStopping monitorando a perda de validação
#   6. Exibir a acurácia de validação final no terminal
#   7. Salvar o modelo treinado como "model.h5"
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Projeto 1 — Classificação MNIST
# ---------------------------------------------------------------------------

# ==========================
# Carregamento do dataset
# ==========================

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalização para [0,1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Ajuste do formato para (28,28,1)
x_train = x_train[..., None]
x_test = x_test[..., None]

# ==========================
# Construção da CNN
# ==========================

model = keras.Sequential([

    layers.Input(shape=(28, 28, 1)),

    # Bloco 1
    layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # Bloco 2
    layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # Bloco 3
    layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),

    # Regularização
    layers.Dropout(0.5),

    # Saída
    layers.Dense(10, activation="softmax")
])

# ==========================
# Compilação
# ==========================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ==========================
# Early Stopping
# ==========================

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

# ==========================
# Treinamento
# ==========================

history = model.fit(
    x_train,
    y_train,
    epochs=15,
    batch_size=64,
    validation_split=0.10,
    callbacks=[early_stopping],
    verbose=1
)

# ==========================
# Resultado da validação
# ==========================

best_val_acc = max(history.history["val_accuracy"])

print(f"\nAcurácia final de validação: {best_val_acc:.4f}")

# ==========================
# Salvamento do modelo
# ==========================

model_path = os.path.join(script_dir, "model.h5")
model.save(model_path)

print(f"\nModelo salvo como '{model_path}'")
