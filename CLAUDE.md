# CLAUDE.md - Guía Metodológica para Generador de Texto con TensorFlow

## Arquitectura Mínima del Modelo

### Diseño de Red Neuronal LSTM

La arquitectura propuesta balancea simplicidad educacional con capacidad expresiva suficiente para generación coherente de texto:

```
Input Layer (Variable Length Sequences)
    ↓
Embedding Layer (vocab_size → 128 dimensions)
    ↓
LSTM Layer 1 (256 units, return_sequences=True)
    ↓
Dropout (0.3) - Regularización
    ↓
LSTM Layer 2 (256 units, return_sequences=True)
    ↓
Dropout (0.3)
    ↓
Dense Layer (vocab_size units)
    ↓
Softmax Activation
```

**Justificación de Decisiones Arquitectónicas:**

La dimensión de embedding de 128 representa un compromiso entre capacidad representacional y eficiencia computacional. Estudios empíricos muestran rendimientos marginales decrecientes más allá de 256 dimensiones para vocabularios < 50k tokens.

Las capas LSTM de 256 unidades proporcionan suficiente capacidad de memoria para capturar dependencias de 50-100 tokens, adecuado para coherencia a nivel de párrafo. El uso de dos capas permite aprendizaje jerárquico: la primera captura patrones sintácticos locales, la segunda relaciones semánticas más abstractas.

El dropout de 0.3 previene overfitting en datasets pequeños (< 10MB), crítico cuando el modelo tiene ~800k parámetros pero el corpus de entrenamiento puede ser limitado.

### Variante Transformer Básica (Opcional Avanzada)

Para estudiantes interesados en arquitecturas modernas, una implementación mínima de self-attention:

```
Input → Positional Encoding → 
Multi-Head Attention (4 heads, 32 dim/head) →
Feed Forward (512 units) →
Output Projection
```

Esta variante requiere 2x memoria pero ofrece paralelización superior y captura de dependencias long-range.

## Mejores Prácticas de TensorFlow para NLP

### 1. Gestión Eficiente de Memoria

**Uso de tf.data.Dataset para Pipeline Optimizado:**

El pipeline debe construirse con operaciones lazy-evaluated para minimizar memoria:

```python
# BUENA PRÁCTICA
dataset = tf.data.Dataset.from_tensor_slices(sequences)
dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)

# EVITAR
all_batches = [sequences[i:i+32] for i in range(0, len(sequences), 32)]
```

**Mixed Precision Training para RTX 2000 Ada:**

La arquitectura Ada Lovelace soporta FP16 con Tensor Cores:

```python
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
```

Esto reduce uso de memoria 40% y acelera entrenamiento 1.5-2x sin pérdida significativa de precisión.

### 2. Tokenización y Vocabulario

**Estrategia de Vocabulario Limitado:**

Para 8GB VRAM, limitar vocabulario a 10,000 tokens más frecuentes + tokens especiales:
- `<PAD>`: padding para batches uniformes
- `<UNK>`: palabras fuera de vocabulario  
- `<START>`: inicio de secuencia
- `<END>`: fin de secuencia

**Subword Tokenization (BPE Simplificado):**

Para mejor generalización con vocabulario limitado:
1. Comenzar con caracteres individuales
2. Iterativamente fusionar pares frecuentes
3. Detener en 10k subwords

Esto maneja mejor palabras raras y neologismos que tokenización por palabras completas.

### 3. Funciones de Pérdida y Métricas

**Sparse Categorical Crossentropy con Label Smoothing:**

```python
loss = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True,
    label_smoothing=0.1  # Previene overconfidence
)
```

Label smoothing mejora generalización distribuyendo 10% de probabilidad uniformemente, evitando que el modelo sea excesivamente confiado en predicciones.

**Perplexity como Métrica Interpretable:**

Perplexity = exp(cross_entropy_loss)

Interpretación práctica:
- Perplexity < 20: Excelente para generación
- Perplexity 20-50: Buena coherencia
- Perplexity 50-100: Aceptable para aplicaciones casuales
- Perplexity > 100: Requiere más entrenamiento

## Flujo de Trabajo Académico Paso a Paso

### Fase 1: Análisis Exploratorio del Corpus (30 min)

1. **Estadísticas Básicas:**
   - Conteo de tokens únicos
   - Distribución de longitudes de oración
   - Frecuencia de n-gramas (unigrams, bigrams, trigrams)

2. **Identificación de Patrones:**
   - Estructuras sintácticas recurrentes
   - Vocabulario especializado del dominio
   - Ratio type/token para medir diversidad léxica

3. **Decisiones de Preprocesamiento:**
   - Preservar/eliminar puntuación
   - Manejo de mayúsculas (lowercase vs. truecase)
   - Segmentación de oraciones

### Fase 2: Preparación de Datos (1 hora)

1. **Limpieza de Texto:**
   - Normalización Unicode (NFC/NFD)
   - Eliminación de caracteres no imprimibles
   - Manejo consistente de espacios/tabs

2. **Creación de Secuencias de Entrenamiento:**
   - Ventana deslizante con stride configurable
   - Longitud de secuencia = 100 tokens (balance memoria/contexto)
   - Overlap de 50% entre secuencias para más ejemplos

3. **División Train/Validation/Test:**
   - 80% entrenamiento
   - 10% validación (early stopping)
   - 10% test (evaluación final)

### Fase 3: Construcción del Modelo (1 hora)

1. **Implementación Incremental:**
   - Comenzar con modelo mínimo (1 LSTM, 128 units)
   - Verificar forward pass con batch sintético
   - Agregar complejidad gradualmente

2. **Debugging de Dimensiones:**
   - Print shapes en cada capa durante construcción
   - Verificar compatibilidad input/output
   - Test con batch_size=1 y batch_size=32

3. **Inicialización de Pesos:**
   - Glorot uniform para Dense layers
   - Orthogonal para LSTM (mejor gradient flow)
   - Embeddings con normal(0, 0.01)

### Fase 4: Entrenamiento Supervisado (2 horas)

1. **Configuración de Hiperparámetros Iniciales:**
   - Learning rate: 0.001 (Adam optimizer)
   - Batch size: 32 (máximo para 8GB VRAM)
   - Epochs: 10 (con early stopping patience=3)

2. **Monitoreo en Tiempo Real:**
   - TensorBoard para visualización
   - Track loss, perplexity, learning rate
   - Guardar checkpoints cada época

3. **Estrategias de Optimización:**
   - Learning rate scheduling (reduce on plateau)
   - Gradient clipping (norm=1.0) para estabilidad
   - Early stopping basado en validation perplexity

### Fase 5: Generación y Evaluación (30 min)

1. **Implementación de Sampling Strategies:**
   - **Greedy**: Selección del token más probable
   - **Temperature**: Ajuste de "creatividad" (T=0.5-1.5)
   - **Top-k**: Muestreo de k tokens más probables
   - **Nucleus (Top-p)**: Muestreo de probabilidad acumulada

2. **Métricas de Evaluación:**
   - **BLEU Score**: Para comparación con referencias
   - **Diversidad**: Ratio de n-gramas únicos
   - **Coherencia**: Análisis manual de muestras

3. **Análisis de Errores:**
   - Identificación de repeticiones
   - Detección de incoherencias gramaticales
   - Evaluación de mantención de contexto

## Conceptos Clave Explicados

### Embeddings: Representación Densa de Tokens

Los embeddings transforman tokens discretos (índices) en vectores continuos de dimensión fija. Conceptualmente, mapean palabras similares a regiones cercanas en el espacio vectorial.

**Intuición Geométrica:** Si "rey", "reina", "príncipe" tienen embeddings e_r, e_q, e_p, entonces:
- dist(e_r, e_q) < dist(e_r, "manzana")
- e_r - e_q ≈ e_p - "princesa" (analogías)

**Aprendizaje de Embeddings:** Se entrenan junto con el modelo mediante backpropagation. Inicialmente aleatorios, convergen a representaciones semánticamente significativas.

### Attention Mechanisms: Foco Selectivo

Aunque este curso usa LSTM, entender attention es crucial para NLP moderno:

**Self-Attention Simplificado:**
1. Para cada token, calcular relevancia con todos los demás
2. Ponderar representaciones según relevancia
3. Combinar para representación contextualizada

**Ventaja sobre LSTM:** Captura dependencias largas sin degradación de gradiente, paralelizable (LSTM es secuencial).

### Loss Functions: Guiando el Aprendizaje

**Categorical Crossentropy:** Mide divergencia entre distribución predicha y real:

H(p,q) = -Σ p(x)log(q(x))

Donde p es distribución real (one-hot) y q es predicción del modelo (softmax).

**Interpretación Práctica:**
- Loss = 0: Predicción perfecta
- Loss = log(vocab_size): Random guessing
- Loss < 2.0: Modelo útil para generación

### Temperature Sampling: Control de Creatividad

Temperature modifica la distribución de probabilidades antes del sampling:

P'(w) = exp(logit(w)/T) / Σ exp(logit(w')/T)

- **T = 0.5**: Más conservador, favorece tokens probables
- **T = 1.0**: Distribución original del modelo
- **T = 1.5**: Más creativo, explora tokens improbables

**Trade-off:** Menor temperature → mayor coherencia pero menos diversidad. Mayor temperature → más creatividad pero riesgo de incoherencia.

## Optimizaciones Específicas para RTX 2000 Ada

### Aprovechamiento de Tensor Cores

La arquitectura Ada incluye Tensor Cores de 3ra generación que aceleran operaciones matriciales:

1. **Alineación de Dimensiones:** Usar múltiplos de 8 para dimensiones de matrices (256, 512) para activación óptima de Tensor Cores.

2. **Mixed Precision:** FP16 para cómputo, FP32 para acumulación mantiene precisión con 2x throughput.

3. **XLA Compilation:** Habilitar con `@tf.function(jit_compile=True)` para fusión de operaciones y optimización de kernels.

### Gestión de Memoria VRAM

Con 8GB VRAM, estrategias críticas:

1. **Gradient Accumulation:** Si batch_size=32 causa OOM, usar batch_size=8 con accumulation_steps=4 para simular batch grande.

2. **Gradient Checkpointing:** Recomputar activaciones durante backward pass, trade-off 20% tiempo por 30% menos memoria.

3. **Dynamic Memory Growth:** 
```python
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
```

## Troubleshooting Común

### Problema: Loss NaN después de pocas épocas
**Causa:** Gradient explosion
**Solución:** Gradient clipping + reduce learning rate

### Problema: Modelo genera repeticiones
**Causa:** Overfitting a patrones locales
**Solución:** Aumentar dropout, reducir modelo, más datos

### Problema: OOM durante entrenamiento
**Causa:** Batch size o modelo muy grande
**Solución:** Reducir batch_size, usar gradient accumulation, mixed precision

### Problema: Generación incoherente
**Causa:** Modelo undertrained o temperatura muy alta
**Solución:** Más épocas, ajustar temperature sampling

## Referencias y Recursos Adicionales

### Papers Fundamentales
- Hochreiter & Schmidhuber (1997): "Long Short-Term Memory" - Paper original LSTM
- Vaswani et al. (2017): "Attention Is All You Need" - Introducción de Transformers
- Holtzman et al. (2019): "The Curious Case of Neural Text Degeneration" - Nucleus sampling

### Documentación Oficial
- TensorFlow Text Generation Guide: tensorflow.org/text/tutorials/text_generation
- NVIDIA Mixed Precision Training: developer.nvidia.com/automatic-mixed-precision

### Herramientas de Análisis
- TensorBoard: Visualización de métricas y arquitectura
- NVIDIA Nsight: Profiling de GPU para optimización
- Weights & Biases: Tracking de experimentos y hiperparámetros