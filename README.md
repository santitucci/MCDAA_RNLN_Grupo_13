# MCDAA_RNLN_Grupo_13

Laboratorios de RNLN.

Este repositorio reúne el trabajo del Grupo 13. Este README documenta el **Laboratorio 1: detección de clickbait en titulares en español**, desde los conceptos básicos hasta las decisiones de implementación, entrenamiento y evaluación.

El objetivo es comparar un MLP basado en el promedio de embeddings con una LSTM que procesa las palabras en orden, realizar una búsqueda automatizada de hiperparámetros y evaluar el modelo seleccionado sobre un conjunto de test reservado.

**Resultado principal:** la LSTM obtuvo el mayor macro-F1 en dev (**0,7738**), prácticamente empatada con el MLP optimizado (**0,7732**). Se seleccionó la LSTM y obtuvo **0,7289 de macro-F1 y 0,7800 de accuracy en test**.

Los resultados corresponden a las corridas documentadas del laboratorio. Los ejemplos de código de este README explican o inspeccionan el trabajo; el notebook contiene la implementación que debe ejecutarse en orden.

## Índice

1. [El problema](#1-el-problema)
2. [Preparación y ejecución](#2-preparación-y-ejecución)
3. [Datos y protocolo experimental](#3-datos-y-protocolo-experimental)
4. [Preprocesamiento y embeddings](#4-preprocesamiento-y-embeddings)
5. [Parte 1: MLP](#5-parte-1-mlp)
6. [Parte 2: LSTM](#6-parte-2-lstm)
7. [Cómo se entrenan los modelos](#7-cómo-se-entrenan-los-modelos)
8. [Parte 3: búsqueda de hiperparámetros](#8-parte-3-búsqueda-de-hiperparámetros)
9. [Métricas de evaluación](#9-métricas-de-evaluación)
10. [Parte 4: selección y test](#10-parte-4-selección-y-test)
11. [Matriz de confusión](#11-matriz-de-confusión)
12. [Dificultades y cambios realizados](#12-dificultades-y-cambios-realizados)
13. [Interpretación avanzada y mejoras](#13-interpretación-avanzada-y-mejoras)
14. [Guía para explicar el laboratorio](#14-guía-para-explicar-el-laboratorio)

## 1. El problema

Clickbait es un estilo de titular que busca despertar curiosidad para atraer un clic, por ejemplo, ocultando información o usando expresiones llamativas. El modelo aprende a reconocer patrones a partir de titulares etiquetados; no se programa una lista fija de frases que determinen la clase.

La tarea es una **clasificación binaria supervisada**:

| Elemento | Significado |
|---|---|
| Entrada | Texto de un titular |
| Clase 0: `No` | Titular etiquetado como no clickbait |
| Clase 1: `Clickbait` | Titular etiquetado como clickbait; clase positiva |
| Salida | Una predicción por titular |

Detectar clickbait no equivale a comprobar si una noticia es verdadera o falsa. Se aprende la categoría definida por las etiquetas del corpus.

La comparación central es entre dos formas de representar un texto: resumir sus palabras en un único vector o procesarlas como una secuencia.

## 2. Preparación y ejecución

### 2.1. Entorno

El laboratorio se desarrolla en Python con notebooks y PyTorch. Puede ejecutarse en Google Colab o en un entorno local con Jupyter y las dependencias instaladas. Una GPU puede acelerar la LSTM, aunque la implementación contempla el uso de CPU.

Las dependencias utilizadas son:

| Biblioteca | Uso |
|---|---|
| PyTorch | Redes, tensores, optimización y batches |
| NumPy | Operaciones sobre vectores |
| pandas | Lectura del corpus y tabla de experimentos |
| Gensim | Almacenamiento y consulta de embeddings |
| scikit-learn | Métricas y matriz de confusión |
| Matplotlib | Gráficas |
| NLTK | Stop words para la opción configurable de filtrado |

La celda de instalación de la versión revisada utiliza:

```python
%pip install "gensim>=4.4,<5" nltk numpy pandas scikit-learn matplotlib
```

Esta instrucción se ejecuta dentro del notebook. PyTorch debe estar disponible en el entorno. La celda revisada conserva la instalación de PyTorch de Colab; en un entorno local sin PyTorch, se debe completar primero su instalación según el hardware.

Si la instalación modifica una biblioteca ya importada, reiniciar el entorno y ejecutar desde el principio. Guardar las versiones que imprime el notebook: la instalación anterior no fija todas las dependencias y, por sí sola, no reconstruye exactamente una corrida histórica.

### 2.2. Orden de ejecución

1. Abrir el notebook del laboratorio.
2. Ejecutar la instalación y los imports.
3. Descargar y cargar los archivos de train, dev y test.
4. Preparar el texto y descargar los embeddings.
5. Ejecutar la parte 1: representación por centroides, entrenamiento del MLP y evaluación en dev.
6. Ejecutar la parte 2: preparación de secuencias, entrenamiento de la LSTM y evaluación en dev.
7. Ejecutar la parte 3: búsqueda de las 18 configuraciones del MLP.
8. Comparar los macro-F1 en dev y fijar el modelo elegido.
9. Ejecutar la parte 4: restaurar el checkpoint elegido, evaluar test y mostrar la matriz de confusión.

Las partes comparten variables y funciones. Ejecutar una celda final en un entorno recién iniciado puede producir errores por objetos inexistentes. Además, después de cambiar el preprocesamiento hay que regenerar las representaciones y volver a entrenar los modelos afectados.

### 2.3. Archivos y descargas

El notebook descarga los siguientes archivos del corpus:

- `TA1C_dataset_detection_train.csv`
- `TA1C_dataset_detection_dev_gold.csv`
- `TA1C_dataset_detection_test_gold.csv`

Los vectores se guardan en `embeddings/wiki.es.vec`. La descarga es grande; el código reutiliza el archivo si ya existe. También se puede colocar manualmente en esa ubicación antes de ejecutar la celda.

La descarga revisada utiliza un archivo temporal `.part` y lo renombra al terminar. Así se evita que una descarga interrumpida quede guardada con el nombre del archivo completo.

## 3. Datos y protocolo experimental

Se leen las columnas `Teaser Text` y `Tag Value`, renombradas como `text` y `label`.

| Conjunto | Función | ¿Actualiza los pesos? |
|---|---|---|
| Train | Aprender los parámetros de la red y definir el vocabulario del experimento | Sí |
| Dev | Seleccionar hiperparámetros y checkpoint; comparar modelos | No |
| Test | Medir el desempeño final del modelo ya seleccionado | No |

**Dev participa indirectamente en las decisiones del experimento:** aunque no se use para calcular gradientes, influye en qué modelo se conserva. Por eso hace falta un conjunto separado para la evaluación final.

Cargar el archivo de test no implica utilizarlo para entrenar. Lo esencial es no usar sus resultados para decidir vocabulario, hiperparámetros, épocas o modelo ganador.

En dev hay **700 titulares: 497 de la clase No y 203 de Clickbait**. Como existe desbalance, predecir siempre `No` produce una accuracy de **0,7100**, pero recall y F1 de Clickbait iguales a cero. Su macro-F1 es **0,4152**. Este baseline permite comprobar que una accuracy relativamente alta no garantiza detectar la clase positiva.

## 4. Preprocesamiento y embeddings

### 4.1. Del texto a los tokens

El preprocesamiento compartido:

1. Convierte el texto a minúsculas.
2. Normaliza Unicode a NFC.
3. Conserva tildes y ñ.
4. Extrae tokens mediante `re.findall(r"\w+", texto)`.
5. Conserva las stop words.

Por ejemplo, `¡Mirá qué pasó!` se transforma en `['mirá', 'qué', 'pasó']`.

Conservar tildes y ñ evita mezclar formas como `qué/que` o `año/ano`. Mantener palabras frecuentes permite conservar posibles señales del estilo de un titular. No significa que cada stop word sea útil: es una decisión que puede contrastarse experimentalmente en dev.

La expresión regular utilizada descarta la puntuación y conserva caracteres de palabra, incluidos números y guiones bajos. Por lo tanto, signos de interrogación o exclamación no llegan como tokens a los modelos de esta versión.

### 4.2. Qué es un embedding

Un embedding representa una palabra mediante un vector de números. En este laboratorio, cada palabra conocida se representa con **300 componentes** de fastText preentrenado en Wikipedia en español.

Los embeddings aportan una representación aprendida previamente. Permanecen **congelados**: el laboratorio entrena el clasificador, sin modificar estos vectores.

Se conservan en memoria únicamente los vectores de palabras presentes en train. El vocabulario del experimento es, por tanto, la intersección entre las palabras de train y las disponibles en el archivo preentrenado.

Aunque fastText permite trabajar con subpalabras, aquí se carga un archivo **`.vec` de vectores estáticos**. Esta implementación no genera embeddings nuevos para palabras desconocidas.

La cobertura observada de tokens fue de **94,88 % en train** y **83,37 % en dev**. No hubo titulares sin vectores conocidos en esos conjuntos. La cobertura mide la proporción de apariciones de tokens representadas, no la accuracy del clasificador.

### 4.3. Palabras desconocidas

| Situación | MLP | LSTM |
|---|---|---|
| Palabra conocida | Usa su vector | Usa su vector |
| Palabra fuera del vocabulario | La ignora en el promedio | La sustituye por `UNK` |
| Titular sin palabras conocidas | Utiliza un vector cero | Utiliza los `UNK` correspondientes |
| Texto sin tokens | Utiliza un vector cero | Utiliza un único `UNK` |

En la LSTM, `UNK` tiene un vector fijo calculado como el promedio de los embeddings conocidos de train. No representa específicamente el significado de la palabra sustituida: es una representación compartida para lo desconocido.

## 5. Parte 1: MLP

### 5.1. Idea básica

El MLP recibe un vector por titular. Para obtenerlo se promedian los embeddings de sus palabras conocidas: ese promedio es el **centroide** del texto.

Para un titular con N palabras conocidas y vectores e₁, …, eₙ:

```text
centroide = (e₁ + e₂ + ... + eₙ) / N
```

El resultado siempre tiene 300 componentes, independientemente de la longitud del titular. Si no hay palabras conocidas, se utiliza un vector cero.

Este método es simple, pero pierde el orden: dos textos con las mismas palabras y frecuencias tienen el mismo centroide aunque estén ordenadas de manera diferente.

### 5.2. Arquitectura y entrenamiento

| Componente | Configuración |
|---|---|
| Entrada | Centroide de 300 dimensiones |
| Primera capa oculta | 256 unidades + ReLU + dropout 0,5 |
| Segunda capa oculta | 64 unidades + ReLU + dropout 0,5 |
| Salida | Capa lineal de 2 logits |
| Optimizador | Adam |
| Learning rate | 0,00005 |
| Weight decay | 0,0001 |
| Batch size | 256 |
| Épocas | 150 |
| Pérdida | CrossEntropyLoss, sin ponderación de clases |
| Semilla | 42 |
| Selección | Mayor macro-F1 en dev; época 150 en la corrida registrada |

ReLU introduce no linealidad: sin funciones de activación entre las capas, varias transformaciones lineales seguirían siendo equivalentes a una sola transformación lineal.

### 5.3. Resultados en dev

| Métrica | Valor |
|---|---:|
| Accuracy | 0,8143 |
| Macro-precision | 0,7976 |
| Macro-recall | 0,7250 |
| Macro-F1 | 0,7465 |
| Precision de Clickbait | 0,7704 |
| Recall de Clickbait | 0,5123 |
| F1 de Clickbait | 0,6154 |

El modelo detectó **104 de los 203 clickbaits** y dejó 99 sin detectar. Superó al baseline, pero su recall positivo indica que todavía perdió casi la mitad de los casos positivos.

Como antecedente, el MLP original obtuvo 0,7286 de accuracy y 0,4877 de macro-F1 en dev. La versión revisada mejoró tras cambiar embeddings, preprocesamiento y otras decisiones. No corresponde atribuir toda la mejora únicamente al cambio de idioma de los embeddings.

## 6. Parte 2: LSTM

### 6.1. Idea básica

La LSTM recibe una secuencia de vectores y procesa las palabras en orden. Mantiene un estado interno que combina información previa con la palabra actual. El clasificador utiliza el estado oculto final para predecir la clase del titular completo.

Esto permite representar relaciones secuenciales que el promedio del MLP elimina. No garantiza un mejor resultado: la utilidad depende del corpus, los hiperparámetros y el entrenamiento.

### 6.2. Arquitectura

| Componente | Configuración |
|---|---|
| Embeddings | 300 dimensiones, congelados; vocabulario con PAD y UNK |
| LSTM | Una capa, unidireccional, entrada 300 y estado oculto 128 |
| Dropout | 0,3 sobre el estado oculto final |
| Salida | Capa lineal de 128 a 2 logits |
| Activaciones internas | Sigmoid en compuertas; tanh en candidato y transformación del estado |

Las compuertas regulan qué información conservar, incorporar y exponer. La LSTM tiene un estado de celda y un estado oculto; el clasificador utiliza este último.

Los parámetros entrenables son los de la LSTM y la capa lineal. Para esta configuración de PyTorch:

```text
LSTM:   4 × 128 × 300 + 4 × 128 × 128 + 2 × 4 × 128 = 220.160
Salida: 128 × 2 + 2 = 258
Total entrenable: 220.418
```

Las cuatro partes corresponden a tres compuertas y un candidato de actualización. PyTorch incluye dos vectores de sesgo en esta configuración. Los parámetros de la tabla de embeddings existen, pero no se cuentan como entrenables.

Para inspeccionar el modelo real de cada ejecución:

```python
print(model_lstm)
print("Parámetros entrenables:", sum(
    p.numel() for p in model_lstm.parameters() if p.requires_grad
))
```

El tamaño de la tabla de embeddings depende del vocabulario construido; la dimensión del vector y el tamaño oculto son hiperparámetros fijados.

### 6.3. Longitudes y padding

Los titulares tienen distinta cantidad de palabras. Para agruparlos en un batch se agrega `PAD` hasta igualar las longitudes dentro de ese batch.

Con `batch_first=True`, antes del empaquetado las formas son:

| Tensor | Forma |
|---|---|
| Índices de tokens | `(B, L)` |
| Embeddings | `(B, L, 300)` |
| Estado final usado por el clasificador | `(B, 128)` |
| Logits | `(B, 2)` |

B es la cantidad de ejemplos y L la longitud máxima del batch. La implementación utiliza `pack_padded_sequence` con las longitudes reales para excluir el padding del procesamiento recurrente. Así, el estado final corresponde al último token real y no a posiciones de relleno.

### 6.4. Entrenamiento y resultado

| Hiperparámetro | Valor |
|---|---|
| Optimizador | Adam |
| Learning rate | 0,001 |
| Weight decay | 0,0001 |
| Batch size | 32 |
| Máximo de épocas | 30 |
| Early stopping | Paciencia de 5 épocas sin mejora del macro-F1 de dev |
| Recorte del gradiente | Norma máxima de 1,0 |
| Pérdida | CrossEntropyLoss, sin ponderación de clases |
| Semilla | 42 |

El entrenamiento finalizó en la **época 14** y se restauró el checkpoint de la **época 9**. Ese modelo obtuvo en dev **0,8043 de accuracy** y **0,7738 de macro-F1**.

Durante las últimas épocas bajó la pérdida de train mientras subió la de dev, una señal de sobreajuste. Conservar el mejor checkpoint evita quedarse automáticamente con los pesos de la última época.

## 7. Cómo se entrenan los modelos

### 7.1. Una actualización, paso a paso

1. Seleccionar un batch de train.
2. Ejecutar el modelo para obtener dos logits por titular.
3. Compararlos con las etiquetas mediante CrossEntropyLoss.
4. Limpiar gradientes anteriores con `optimizer.zero_grad()`.
5. Calcular gradientes con `loss.backward()`.
6. En la LSTM, recortar la norma del gradiente.
7. Actualizar los parámetros con `optimizer.step()`.

Una **época** es un recorrido completo por train. Un **batch** es el grupo de ejemplos utilizado en una actualización. El **learning rate** controla la escala de las actualizaciones del optimizador.

### 7.2. Logits y pérdida

Los logits son puntuaciones sin normalizar. `CrossEntropyLoss` recibe esos logits y las etiquetas enteras; no se aplica softmax antes de la pérdida.

Para predecir la clase basta con:

```python
predicciones = logits.argmax(dim=1)
```

Si se necesitan probabilidades, se puede aplicar softmax durante la inferencia. Sus valores no deben interpretarse automáticamente como probabilidades perfectamente calibradas.

### 7.3. Regularización y evaluación

- **Dropout:** anula aleatoriamente componentes durante el entrenamiento para reducir dependencias excesivas entre unidades.
- **Weight decay:** penaliza pesos grandes; con el Adam utilizado funciona como penalización L2 acoplada a la actualización. No es el mismo mecanismo que el weight decay desacoplado de AdamW.
- **Gradient clipping:** limita la norma del gradiente en la LSTM. Ayuda frente a gradientes excesivos; no limita directamente la norma de los pesos.
- **Early stopping:** detiene el entrenamiento tras un número de épocas sin mejora de la métrica seleccionada.
- **Checkpoint:** copia independiente de los pesos que se desea conservar.

Para evaluar se usan `model.eval()` y `torch.no_grad()`. El primero cambia el comportamiento de módulos como dropout; el segundo desactiva el seguimiento de gradientes. Cumplen funciones distintas.

Las métricas de train y dev se calculan con pesos fijos y dropout desactivado, lo que hace más comparable su evaluación. El entrenamiento optimiza la entropía cruzada, mientras que la selección usa macro-F1: una pérdida menor no garantiza una mejora inmediata del macro-F1.

## 8. Parte 3: búsqueda de hiperparámetros

### 8.1. Diseño

Se eligió el MLP para explorar varias configuraciones con menor costo de entrenamiento. Se mantuvieron los centroides y el preprocesamiento de la parte 1.

La grilla contiene **3 × 2 × 3 = 18 combinaciones**:

| Hiperparámetro | Valores |
|---|---|
| Capas ocultas | `(64,)`, `(128,)`, `(128, 64)` |
| Activación | ReLU, Tanh |
| Learning rate | 0,0001; 0,0005; 0,001 |

Se fijaron Adam, dropout 0,3, weight decay 0,0001, batch size 256, máximo de 150 épocas y paciencia de 12. Cada configuración comenzó desde cero con semilla 42 y un generador independiente para mantener el mismo orden de batches.

Se guardó el checkpoint de mayor macro-F1 en dev dentro de cada configuración y luego se compararon esos checkpoints. La grilla es completa para los valores indicados, pero no explora todas las arquitecturas o hiperparámetros posibles.

### 8.2. Resultados de la grilla

| Capas ocultas | Activación | Learning rate | Mejor época | Macro-F1 dev |
|---|---|---:|---:|---:|
| 64 | ReLU | 0,0001 | 4 | 0,4152 |
| 64 | ReLU | 0,0005 | 105 | 0,7683 |
| 64 | ReLU | 0,001 | 88 | 0,7681 |
| 64 | Tanh | 0,0001 | 2 | 0,4152 |
| 64 | Tanh | 0,0005 | 76 | 0,7547 |
| 64 | Tanh | 0,001 | 23 | 0,7462 |
| 128 | ReLU | 0,0001 | 1 | 0,4152 |
| 128 | ReLU | 0,0005 | 34 | 0,7383 |
| 128 | ReLU | 0,001 | 23 | 0,7469 |
| 128 | Tanh | 0,0001 | 1 | 0,4152 |
| 128 | Tanh | 0,0005 | 74 | 0,7586 |
| 128 | Tanh | 0,001 | 83 | 0,7600 |
| 128, 64 | ReLU | 0,0001 | 1 | 0,4152 |
| 128, 64 | ReLU | 0,0005 | 54 | 0,7703 |
| **128, 64** | **ReLU** | **0,001** | **46** | **0,7732** |
| 128, 64 | Tanh | 0,0001 | 1 | 0,4152 |
| 128, 64 | Tanh | 0,0005 | 45 | 0,7594 |
| 128, 64 | Tanh | 0,001 | 27 | 0,7603 |

El mejor MLP tiene entrada de 300 componentes, capas ocultas de **128 y 64 unidades**, ReLU y dropout 0,3 después de cada capa oculta, y salida de dos logits. Se restauró la época **46** después de detener el entrenamiento en la **58**.

| Métrica en dev | Mejor MLP de la búsqueda |
|---|---:|
| Accuracy | 0,8200 |
| Macro-precision | 0,7856 |
| Macro-recall | 0,7640 |
| Macro-F1 | 0,7732 |
| Precision de Clickbait | 0,7151 |
| Recall de Clickbait | 0,6305 |
| F1 de Clickbait | 0,6702 |

La búsqueda elevó el macro-F1 del MLP de 0,7465 a 0,7732. Detectó **128 de los 203 clickbaits**, frente a 104 del MLP anterior, aunque bajó la precision positiva.

### 8.3. Limitación de la parada temprana

Todas las configuraciones con learning rate 0,0001 terminaron con checkpoints seleccionados que predecían únicamente la clase mayoritaria.

Esto no demuestra que ese learning rate sea inadecuado. Durante un aprendizaje lento, la pérdida puede disminuir sin cambiar las clases predichas y, por tanto, sin mejorar el macro-F1. Una paciencia de 12 épocas puede detener el experimento antes de que aparezca esa mejora.

Esta limitación quedó pendiente. Una comparación futura podría incluir un mínimo de épocas antes de activar early stopping o una paciencia mayor, siempre usando train/dev para esas decisiones.

## 9. Métricas de evaluación

Tomando Clickbait como clase positiva:

| Símbolo | Significado |
|---|---|
| TP | Clickbait predicho correctamente como Clickbait |
| FP | No clickbait predicho como Clickbait |
| FN | Clickbait predicho como No |
| TN | No clickbait predicho correctamente como No |

| Métrica | Fórmula | Pregunta que responde |
|---|---|---|
| Accuracy | `(TP + TN) / total` | ¿Qué proporción de todos los titulares se clasificó correctamente? |
| Precision positiva | `TP / (TP + FP)` | De los marcados como Clickbait, ¿cuántos lo eran? |
| Recall positivo | `TP / (TP + FN)` | De los clickbaits reales, ¿cuántos se detectaron? |
| F1 positivo | `2 × precision × recall / (precision + recall)` | ¿Cómo se combinan precision y recall de Clickbait? |

Las métricas macro se calculan por clase y luego se promedian con igual peso:

```text
Macro-precision = (precision_No + precision_Clickbait) / 2
Macro-recall    = (recall_No + recall_Clickbait) / 2
Macro-F1        = (F1_No + F1_Clickbait) / 2
```

**Macro-F1 no es el F1 calculado a partir de macro-precision y macro-recall.** Es el promedio de los F1 individuales. En el laboratorio se utiliza como criterio de selección para dar el mismo peso a ambas clases.

Si una clase no recibe predicciones, su precision puede quedar indefinida matemáticamente. El tratamiento computacional debe ser consistente; los resultados registrados para los modelos que solo predicen No asignan cero a las métricas positivas afectadas.

## 10. Parte 4: selección y test

### 10.1. Modelo elegido con dev

| Modelo | Accuracy dev | Macro-F1 dev |
|---|---:|---:|
| Baseline: siempre No | 0,7100 | 0,4152 |
| MLP de la parte 1 | 0,8143 | 0,7465 |
| LSTM de la parte 2 | 0,8043 | **0,7738** |
| MLP optimizado de la parte 3 | **0,8200** | 0,7732 |

Se eligió la **LSTM** porque obtuvo el mayor macro-F1 en dev, que es el criterio de la consigna. La diferencia con el MLP optimizado es aproximadamente **0,0006**, equivalente a **0,06 puntos porcentuales**. No permite afirmar una superioridad general de la arquitectura LSTM.

### 10.2. Procedimiento de test

Se restauró `best_state_lstm` en `model_lstm`, se aplicaron el mismo preprocesamiento y vocabulario, y se evaluó el modelo sin reentrenarlo. Las palabras nuevas se representaron mediante UNK.

La salida se guardó en `results_test_lstm`, incluyendo métricas, etiquetas reales y predicciones. El resultado de test no se utilizó para cambiar el modelo ganador.

### 10.3. Resultados finales

| Métrica en test | LSTM seleccionada |
|---|---:|
| Accuracy | **0,7800** |
| Macro-precision | 0,7154 |
| Macro-recall | 0,7610 |
| Macro-F1 | **0,7289** |
| Precision de Clickbait | 0,5284 |
| Recall de Clickbait | 0,7246 |
| F1 de Clickbait | 0,6111 |

El macro-F1 bajó de 0,7738 en dev a 0,7289 en test: una diferencia de **0,0449**, o **4,49 puntos porcentuales**. La accuracy bajó de 0,8043 a 0,7800.

El modelo detectó aproximadamente el **72,46 % de los clickbaits reales**, pero solo el **52,84 % de los titulares marcados como clickbait** pertenecía a esa clase. Dicho de otro modo, aproximadamente el 47,16 % de sus predicciones positivas fueron falsas alarmas. Este último porcentaje no es la tasa de falsos positivos sobre todos los negativos reales: usa otro denominador.

El desempeño inferior en test muestra una brecha entre conjuntos, pero por sí solo no demuestra sobreajuste. Puede reflejar variación muestral, diferencias entre conjuntos y el hecho de haber seleccionado el modelo usando dev. Las curvas de pérdida de la LSTM sí aportan una señal adicional de sobreajuste durante el entrenamiento.

Solo se presentan resultados de test para el modelo seleccionado. No se puede concluir que la LSTM supera al MLP en test porque no se realizó aquí esa comparación.

## 11. Matriz de confusión

La matriz se obtiene de las predicciones guardadas, con **filas como clases reales** y **columnas como clases predichas**:

| Real / Predicha | No | Clickbait |
|---|---|---|
| No | TN | FP |
| Clickbait | FN | TP |

Después de evaluar test:

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm_test = confusion_matrix(
    results_test_lstm["targets"],
    results_test_lstm["predictions"],
    labels=[0, 1],
)

ConfusionMatrixDisplay(
    confusion_matrix=cm_test,
    display_labels=["No", "Clickbait"],
).plot(cmap="Blues", values_format="d")

plt.title("Matriz de confusión — LSTM sobre TEST")
plt.xlabel("Clase predicha")
plt.ylabel("Clase real")
plt.grid(False)
plt.show()

tn, fp, fn, tp = cm_test.ravel()
print(f"TN: {tn} | FP: {fp} | FN: {fn} | TP: {tp}")
```

Los conteos exactos deben tomarse de esta salida. No se reconstruyen aquí a partir de métricas redondeadas. Inspeccionar los titulares de FP y FN permite estudiar qué errores comete el clasificador, sin convertir esa inspección en una nueva etapa de ajuste sobre test.

## 12. Dificultades y cambios realizados

| Dificultad | Cambio o tratamiento | Motivo y estado |
|---|---|---|
| Embeddings originales en inglés | Sustitución por fastText en español | Representación más adecuada al idioma; no se aisló su contribución mediante una ablación |
| Eliminación de tildes y stop words | Conservación de tildes, ñ y palabras frecuentes | Preservar información potencialmente útil |
| Carga de embeddings grande | Lectura del archivo conservando solo vectores de train | Reduce memoria; no evita descargar y recorrer el archivo |
| Desbalance de clases | Baseline mayoritario y selección por macro-F1 | Evitar depender únicamente de accuracy; persisten errores en Clickbait |
| Secuencias de distinta longitud | PAD, longitudes reales y empaquetado | Evitar utilizar el relleno como final real del titular |
| Palabras desconocidas | Omisión en MLP y UNK en LSTM | Mantener fijo el vocabulario de train |
| Sobreajuste de la LSTM | Dropout, weight decay, checkpoint y early stopping | Conservar el mejor estado observado en dev; no garantiza eliminar el sobreajuste |
| Evaluaciones poco comparables | `eval()` y `no_grad()` al medir train/dev | Desactivar dropout y evaluar con pesos fijos |
| Riesgo de reutilizar test para seleccionar | Reserva de test para la parte 4 | Mantener una evaluación final separada |
| Conflicto de versiones con torchvision | Celda revisada sin actualización indiscriminada de torch | Evitar alterar la instalación compatible; verificar que se esté usando esa celda y reiniciar si corresponde |
| Parada temprana con learning rate bajo | Limitación identificada | Pendiente de una búsqueda con mínimo de épocas o mayor paciencia |

Los comentarios `CAMBIO` y `POR QUÉ` del código revisado explican las modificaciones. Las opciones `STRIP_ACCENTS` y `REMOVE_STOPWORDS` permiten ensayar variantes del preprocesamiento, pero deben establecerse antes de volver a generar representaciones y entrenar. No se deben elegir sus valores a partir de test.

## 13. Interpretación avanzada y mejoras

### 13.1. Reproducibilidad

La semilla 42 controla parte de la aleatoriedad. En la grilla también se fija el generador de batches. Esto facilita comparar experimentos, pero no garantiza igualdad exacta entre hardware, versiones o algoritmos de GPU.

Para documentar una nueva corrida conviene conservar:

- Versión del notebook y archivos de datos utilizados.
- Versiones de Python y bibliotecas.
- Configuración de preprocesamiento y embeddings.
- Semillas, hiperparámetros y dispositivo.
- Tabla de búsqueda, curvas y métricas.
- Checkpoint seleccionado y vocabulario asociado.

Un `state_dict` solo no describe todo el sistema: para usar los pesos después se necesita reconstruir la misma arquitectura, el mapeo de tokens, las etiquetas y el preprocesamiento. Una copia del checkpoint en memoria se pierde al reiniciar el entorno si no se guarda en disco.

### 13.2. Qué permite concluir el experimento

La búsqueda mejoró el MLP y lo dejó casi empatado con la LSTM en macro-F1 de dev. Esto muestra que un modelo basado en centroides puede ser competitivo en este corpus.

El experimento no aísla exclusivamente el efecto del orden de palabras: MLP y LSTM también difieren en tratamiento de desconocidas, arquitectura, batches y configuración de entrenamiento. Para atribuir una mejora a un componente concreto se necesitarían comparaciones controladas.

La elección de la LSTM es correcta según la consigna. En una aplicación real también habría que considerar costos de inferencia y consecuencias de FP y FN. El MLP sería un candidato razonable por su resultado cercano y su representación más simple, pero aquí no se midieron tiempos ni costos.

### 13.3. Mejoras propuestas, no realizadas

1. Repetir los mejores experimentos con varias semillas y reportar media y dispersión.
2. Dar más tiempo a las configuraciones de learning rate bajo antes de activar early stopping.
3. Realizar ablaciones del preprocesamiento, manteniendo el resto constante.
4. Estudiar la cobertura y el efecto de las palabras desconocidas usando train/dev.
5. Probar puntuación como tokens para conservar señales del estilo del titular.
6. Evaluar ponderación de clases o cambios del umbral de decisión, elegidos exclusivamente en dev.
7. Comparar una representación que aproveche subpalabras para palabras desconocidas.

Estas propuestas no forman parte de los resultados informados. Una nueva etapa de desarrollo posterior a la inspección de test debería contar con una evaluación final independiente si se busca conservar una estimación no condicionada por esas decisiones.

## 14. Guía para explicar el laboratorio

**¿Qué hicimos?** Entrenamos clasificadores para detectar clickbait en titulares en español, comparamos sus resultados en dev y evaluamos el elegido en test.

**¿Por qué embeddings?** Las redes reciben números. Los embeddings convierten palabras en vectores y aportan información aprendida de un corpus externo.

**¿Cuál es la diferencia entre MLP y LSTM?** El MLP usa un promedio y pierde el orden. La LSTM procesa la secuencia y resume la información en su estado final.

**¿Qué aprendió la red?** Los pesos del clasificador. Los embeddings preentrenados permanecieron fijos.

**¿Por qué macro-F1?** Hay desbalance de clases y se quiere dar igual peso al F1 de No y Clickbait. Accuracy por sí sola puede ocultar que no se detecta la clase minoritaria.

**¿Qué modelo ganó?** La LSTM, con macro-F1 de dev 0,7738 frente a 0,7732 del MLP optimizado. La diferencia es muy pequeña y no demuestra superioridad general.

**¿Cómo rindió en test?** Accuracy 0,7800 y macro-F1 0,7289. Detectó el 72,46 % de los clickbaits, con precision positiva de 52,84 %.

**¿Qué limitación destaca?** Una proporción considerable de las predicciones positivas son falsos positivos. Además, la grilla pudo detener demasiado pronto las configuraciones que aprendían lentamente.

**¿Qué no debemos afirmar?** Que test se usó para elegir el ganador, que fastText generó vectores nuevos mediante subpalabras en esta implementación, o que una sola corrida prueba que la LSTM siempre es mejor que un MLP.
