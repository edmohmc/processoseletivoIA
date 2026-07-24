# Projeto 1 — Classificação MNIST

## 💻 O Desafio Técnico

Desenvolva um **modelo de Visão Computacional** capaz de **classificar dígitos manuscritos (0-9)**, e posteriormente **otimize-o para execução em dispositivos Edge**.

O foco não é apenas obter alta acurácia, mas **compreender o fluxo completo**:

**treinamento → validação → salvamento → conversão → otimização**

## 🎯 Conjunto de Dados

Dataset **MNIST**, disponível diretamente via `tf.keras.datasets.mnist` (não é necessário download manual).

## ✅ Requisitos Obrigatórios

### Etapa 1 — Treinamento do Modelo (`train_model.py`)

Implemente:

- Carregamento do dataset MNIST via TensorFlow
- **Split explícito treino/validação** (ex: `validation_split` ou um split manual)
- Construção de uma CNN com:
  - **3 a 4 blocos convolucionais** (`Conv2D` + `BatchNormalization` + `MaxPooling2D`)
  - Camada de `Dropout` antes da saída, para regularização
- Treinamento com **early stopping** baseado na perda de validação (`EarlyStopping`)
- Exibição da **acurácia de validação final** no terminal
- Salvamento do modelo treinado em formato Keras (`model.h5`)

### Etapa 2 — Otimização do Modelo (`optimize_model.py`)

Implemente:

- Carregamento do `model.h5` treinado
- Conversão para **TensorFlow Lite** (`model.tflite`)
- Aplicação de uma técnica de otimização (ex: **Dynamic Range Quantization**)

### Etapa 3 — Inferência com o Modelo Otimizado (`run_inference.py`)

Implemente:

- Carregamento especificamente do **`model.tflite`** (o artefato de edge — não
  o `model.h5`) usando `tf.lite.Interpreter`
- Execução de inferência em pelo menos **5 amostras** do conjunto de teste
- Exibição no terminal, para cada amostra, da classe **predita** vs. a classe **real**

> 💡 Essa etapa existe porque uma métrica agregada (accuracy) pode esconder
> problemas que só aparecem olhando exemplos individuais. Também é o teste mais
> próximo do uso real em produção: carregar o artefato de edge e classificar
> uma entrada por vez.

**Objetivo:** reduzir o tamanho do modelo, mantendo desempenho adequado para aplicações de Edge AI.

## 📂 Estrutura da Pasta

⚠️ Não altere os nomes dos arquivos.

```
projetos/1-classificacao-mnist/
├── train_model.py         # ✏️ Treinamento do modelo
├── optimize_model.py      # ✏️ Conversão e otimização
├── run_inference.py       # ✏️ Inferência de exemplo com o modelo otimizado
├── requirements.txt       # 📄 Dependências do projeto
├── model.h5               # 🤖 Gerado por você — deve ser commitado
├── model.tflite           # ⚡ Gerado por você — deve ser commitado
└── README.md               # 📝 Este arquivo (também usado como relatório)
```

## ⚠️ Restrições e Considerações de Engenharia

- Entrada do modelo: imagens 28x28, 1 canal (grayscale), normalizadas em [0, 1]
- CNN simples — evite arquiteturas muito profundas
- Não utilize modelos pré-treinados
- Número de épocas limitado (ex: até 15, com early stopping)
- Treinamento apenas em CPU

## ⚖️ Critérios de Avaliação

- **Funcionalidade** — execução correta dos scripts e geração dos arquivos `.h5` e `.tflite`
- **Qualidade do modelo** — acurácia de validação consistente com o esperado para o dataset
- **Edge AI** — conversão correta para `.tflite` com técnica de otimização aplicada
- **Documentação** — preenchimento adequado do relatório abaixo

---

## 📝 Relatório do Candidato

👤 **Nome Completo:** Edmo Henrique Martins Cavalcante

### 1️⃣ Resumo da Arquitetura do Modelo

A arquitetura implementada em `train_model.py` é uma **Rede Convolucional (CNN)** estruturada em 3 blocos convolucionais principais seguidos por camadas de classificação e regularização:

- **Blocos Convolucionais (3 blocos):**
  - **Bloco 1:** `Conv2D` com 32 filtros (kernel 3x3, padding `"same"`, ativação ReLU) + `BatchNormalization` + `MaxPooling2D` (pool 2x2).
  - **Bloco 2:** `Conv2D` com 64 filtros (kernel 3x3, padding `"same"`, ativação ReLU) + `BatchNormalization` + `MaxPooling2D` (pool 2x2).
  - Bloco 3: `Conv2D` com 128 filtros (kernel 3x3, padding `"same"`, ativação ReLU) + `BatchNormalization` + `MaxPooling2D` (pool 2x2).
- **Camadas Densa e Regularização:**
  - `Flatten` para vetorização dos mapas de características.
  - `Dense` com 128 neurônios e ativação ReLU.
  - `Dropout(0.5)` aplicado antes da saída para reduzir overfitting.
  - Camada final `Dense` com 10 neurônios e ativação `softmax` para classificação das 10 classes de dígitos (0-9).
- **Estratégia de Validação e Early Stopping:**
  - Split explícito de 10% do dataset de treinamento para validação (`validation_split=0.10`).
  - Callback `EarlyStopping` monitorando a perda de validação (`monitor="val_loss"`), com paciência de 3 épocas (`patience=3`) e restauração dos melhores pesos (`restore_best_weights=True`).

### 2️⃣ Bibliotecas Utilizadas

- **TensorFlow**: `2.21.0` (incluindo `tf.lite` para conversão e inferência Edge)
- **Keras / `tf_keras`**: `3.14.1` / `2.21.0` (utilizando `tf_keras` para garantia de compatibilidade com a especificação HDF5/Keras 2 no salvamento de `model.h5`)
- **NumPy**: `1.26.4` (para pré-processamento de imagens e manipulação de arrays)

### 3️⃣ Técnica de Otimização do Modelo

Em `optimize_model.py`, foi aplicada a técnica de **Dynamic Range Quantization (Quantização de Intervalo Dinâmico)** por meio do conversor `tf.lite.TFLiteConverter` com a configuração `converter.optimizations = [tf.lite.Optimize.DEFAULT]`.

Essa técnica quantiza os pesos do modelo de ponto flutuante de 32 bits (FP32) para inteiros de 8 bits (INT8) em tempo de conversão, mantendo as ativações em ponto flutuante durante a inferência. Trata-se de uma estratégia extremamente eficiente para Edge AI, pois reduz significativamente o tamanho do modelo sem necessitar de um conjunto de dados representativo para calibração.

### 4️⃣ Resultados Obtidos

- **Acurácia Final de Validação:** **99.05%** (`0.9905`)
- **Tamanho do `model.h5`:** **2.98 MB** (2.979.352 bytes)
- **Tamanho do `model.tflite`:** **256.4 KB** (256.424 bytes)
- **Taxa de Compressão:** Redução de aproximadamente **91.4%** no tamanho do arquivo.

### 5️⃣ Comentários Adicionais (Opcional)

- **Compatibilidade entre Versões do Keras:** Durante a desserialização do `model.h5`, identificou-se uma incompatibilidade de formato entre Keras 3 e Keras 2 (devido ao parâmetro `synchronized` adicionado na camada `BatchNormalization` pelo Keras 3). O problema foi resolvido importando `tf_keras` no script de treinamento, assegurando total compatibilidade do arquivo `.h5` em diferentes ambientes de avaliação.
- **Eficiência para Dispositivos Edge:** O modelo final `.tflite` apresentou excelente desempenho e ocupação de memória baixíssima (~256 KB), sendo ideal para deploy em microcontroladores e dispositivos embarcados.

### 6️⃣ Exemplo de Inferência

Saída do terminal ao executar `run_inference.py`:

```text
Rodando inferencia em 5 amostras usando model.tflite:

Amostra 1: predito=7 | real=7
Amostra 2: predito=2 | real=2
Amostra 3: predito=1 | real=1
Amostra 4: predito=0 | real=0
Amostra 5: predito=4 | real=4
```

**Análise:** O modelo quantizado no formato TensorFlow Lite alcançou **100% de acerto** nas 5 amostras aleatórias de teste, classificando corretamente dígitos com traços diversos (como 7, 2, 1, 0, 4) sem degradação da precisão após a otimização dos pesos.
