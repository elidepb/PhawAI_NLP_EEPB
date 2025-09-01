# 📘 Análisis de Clasificación de Emociones en Tweets en Español

---

## 📌 Índice del Notebook

1. [Introducción y Objetivos](#libro-introducción)
2. [Configuración Inicial y Librerías](#libro-configuración)
3. [Carga y Exploración del Dataset](#libro-dataset)
4. [Preprocesamiento de Texto](#libro-preprocesamiento)
5. [Modelo 1: BiLSTM + Atención + Características Adicionales](#lapiz-modelo-1)
6. [Modelo 2: Fine-tuning de DeBERTa-v3](#lapiz-modelo-2)
7. [Modelo 3: Instruction-Tuning con Flan-T5](#lapiz-modelo-3)
8. [Evaluación y Comparativa de Modelos](#libro-evaluación)
9. [Conclusiones y Recomendaciones](#libro-conclusiones)

---

## 📖 Introducción

Este notebook tiene como objetivo clasificar emociones en tweets en español utilizando tres modelos diferentes:

1. **BiLSTM con Atención y Características Adicionales**
2. **DeBERTa-v3 con Fine-tuning**
3. **Flan-T5 con Instruction-Tuning**

Cada modelo aborda el problema desde una perspectiva arquitectónica y técnica distinta, permitiendo una comparación exhaustiva del rendimiento.

---

## 🧮 Configuración Inicial

Se utilizan librerías como:
- `pandas`, `numpy` para manipulación de datos
- `matplotlib`, `seaborn` para visualización
- `nltk` para preprocesamiento de texto
- `tensorflow` y `transformers` para modelado
- `scikit-learn` para métricas y validación

---

## 📊 Dataset: EmoEvent

- Dataset en español con tweets etiquetados con emociones.
- Contiene 8,193 tweets después de la limpieza.
- 7 emociones: `anger`, `disgust`, `fear`, `joy`, `others`, `sadness`, `surprise`.
- Incluye información adicional como evento y si es ofensivo.

---

## 🛠 Preprocesamiento de Texto

- Limpieza de URLs, menciones, hashtags, puntuación y números.
- Tokenización y eliminación de stopwords (excepto negaciones).
- Longitud máxima de secuencia: 50 tokens (95% de los tweets tienen ≤44 palabras).

---

## ✏️ Modelo 1: BiLSTM + Atención + Características Adicionales

### 🧠 Técnicas Utilizadas:
- **Embeddings FastText** preentrenados en español (300 dimensiones).
- **Capa Bidirectional LSTM** para capturar contexto bidirectional.
- **Mecanismo de Atención** para enfocarse en palabras relevantes.
- **Características adicionales**: evento y si es ofensivo (one-hot encoding).
- **Pesos de clase** para manejar desbalanceo.
- **Pérdida Focal** para enfocarse en ejemplos difíciles.
- **Validación cruzada estratificada** (5 folds).

---

## ✏️ Modelo 2: Fine-tuning de DeBERTa-v3

### 🧠 Técnicas Utilizadas:
- Modelo preentrenado `microsoft/deberta-v3-base`.
- Tokenización con máximo de 50 tokens.
- Fine-tuning con 4 épocas, learning rate 2e-5.
- Batch size reducido (16) y gradient accumulation (4) por limitaciones de memoria.
- Métricas: F1, precisión, recall ponderados.
  
---

## ✏️ Modelo 3: Instruction-Tuning con Flan-T5

### 🧠 Técnicas Utilizadas:
- Modelo `google/flan-t5-base`.
- Formato instruccional: `"Clasificar emoción: [texto] | opciones: ..."`
- Tokenización separada de entrada y salida.
- Entrenamiento con 4 épocas, learning rate 3e-5.
- Decodificación con beam search (4 beams).

---

## 📚 Recomendaciones

- Utilizar aumentación de datos para clases minoritarias.
- Explorar técnicas de few-shot learning con LLMs.

---
