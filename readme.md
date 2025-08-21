# robo-poet - Generador de Texto con TensorFlow

Implementación educacional de un generador de texto basado en LSTM usando TensorFlow 2.x, optimizado para GPUs NVIDIA consumer con arquitectura Ada Lovelace.

## Requisitos del Sistema

### Hardware Mínimo
- **GPU**: NVIDIA RTX 2000 Ada o superior (8GB VRAM)
- **RAM**: 16GB DDR4/DDR5
- **Almacenamiento**: 10GB espacio libre (modelo + datasets)
- **CPU**: Intel i5-8400 / AMD Ryzen 5 2600 o superior

### Software Requerido
- **OS**: Windows 11 con WSL2 Ubuntu 22.04 LTS
- **Python**: 3.8.10 - 3.10.x (3.11+ no soportado por TensorFlow 2.15)
- **CUDA Toolkit**: 11.8.0
- **cuDNN**: 8.6.0
- **NVIDIA Driver**: 525.60.13 o superior

### Dependencias Python
```
tensorflow==2.15.0
numpy==1.24.3
tqdm==4.66.1
matplotlib==3.7.2
tensorboard==2.15.0
```

## Configuración para WSL2 + NVIDIA RTX 2000 Ada

### Paso 1: Instalación de WSL2

```powershell
# PowerShell como Administrador
wsl --install -d Ubuntu-22.04
wsl --set-default-version 2
wsl --update
```

Reiniciar sistema después de instalación.

### Paso 2: Configuración de NVIDIA Driver en Windows

1. Descargar driver NVIDIA para Windows desde: https://www.nvidia.com/Download/index.aspx
   - Seleccionar: RTX 2000 Ada Generation
   - Versión mínima: 525.60.13

2. Instalar driver con opciones predeterminadas
3. Verificar instalación:
```powershell
nvidia-smi
```

### Paso 3: Setup CUDA en WSL2

```bash
# Dentro de WSL2 Ubuntu
# Actualizar sistema
sudo apt update && sudo apt upgrade -y

# Instalar CUDA Toolkit 11.8
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-11-8

# Configurar PATH
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verificar CUDA
nvcc --version  # Debe mostrar 11.8.0
```

### Paso 4: Instalación de cuDNN

```bash
# Descargar cuDNN 8.6.0 para CUDA 11.x desde NVIDIA Developer
# Requiere cuenta gratuita en developer.nvidia.com

# Después de descargar cudnn-linux-x86_64-8.6.0.163_cuda11-archive.tar.xz
tar -xvf cudnn-linux-x86_64-8.6.0.163_cuda11-archive.tar.xz
sudo cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda-11.8/include
sudo cp cudnn-*-archive/lib/libcudnn* /usr/local/cuda-11.8/lib64
sudo chmod a+r /usr/local/cuda-11.8/include/cudnn*.h /usr/local/cuda-11.8/lib64/libcudnn*
```

### Paso 5: Entorno Virtual Python

```bash
# Instalar Python 3.10 y venv
sudo apt install python3.10 python3.10-venv python3.10-dev -y

# Crear entorno virtual
python3.10 -m venv robo-poet-env
source robo-poet-env/bin/activate

# Actualizar pip
pip install --upgrade pip setuptools wheel
```

### Paso 6: Instalación de TensorFlow con GPU

```bash
# Instalar TensorFlow con soporte GPU
pip install tensorflow[and-cuda]==2.15.0

# Instalar dependencias adicionales
pip install numpy==1.24.3 tqdm==4.66.1 matplotlib==3.7.2 tensorboard==2.15.0

# Verificar GPU es detectada
python -c "import tensorflow as tf; print(f'GPUs disponibles: {len(tf.config.list_physical_devices(\"GPU\"))}')"
# Debe mostrar: GPUs disponibles: 1
```

## Verificación de Instalación

### Script de Verificación Completa

```python
# verify_setup.py
import tensorflow as tf
import sys

print("=" * 50)
print("VERIFICACIÓN DE CONFIGURACIÓN ROBO-POET")
print("=" * 50)

# Verificar versiones
print(f"Python: {sys.version}")
print(f"TensorFlow: {tf.__version__}")

# Verificar GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✓ GPU detectada: {gpus[0].name}")
    
    # Obtener detalles de GPU
    from tensorflow.python.client import device_lib
    local_devices = device_lib.list_local_devices()
    for device in local_devices:
        if device.device_type == 'GPU':
            print(f"  Compute Capability: {device.physical_device_desc}")
else:
    print("✗ No se detectó GPU")
    sys.exit(1)

# Test de operación en GPU
with tf.device('/GPU:0'):
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
    c = tf.matmul(a, b)
    print(f"✓ Operación matricial en GPU exitosa: {c.shape}")

# Verificar mixed precision
try:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print("✓ Mixed precision (FP16) disponible")
except:
    print("✗ Mixed precision no disponible")

print("=" * 50)
print("Configuración verificada exitosamente!")
```

Ejecutar con:
```bash
python verify_setup.py
```

## Uso Básico

### Estructura del Proyecto

```
robo-poet/
├── data/
│   ├── raw/           # Textos originales (.txt)
│   ├── processed/     # Datos preprocesados
│   └── vocab/         # Vocabulario y tokenizer
├── models/
│   ├── checkpoints/   # Checkpoints durante entrenamiento
│   └── final/         # Modelo final entrenado
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   ├── train.py
│   └── generate.py
├── logs/              # TensorBoard logs
└── robo_poet.py      # CLI principal
```

### Comandos CLI

#### Entrenamiento

```bash
# Entrenamiento básico con configuración por defecto
python robo_poet.py train --data data/raw/mi_texto.txt

# Entrenamiento con parámetros personalizados
python robo_poet.py train \
    --data data/raw/mi_texto.txt \
    --epochs 10 \
    --batch-size 32 \
    --learning-rate 0.001 \
    --lstm-units 256 \
    --embedding-dim 128 \
    --dropout 0.3 \
    --vocab-size 10000 \
    --sequence-length 100 \
    --checkpoint-dir models/checkpoints \
    --log-dir logs
```

#### Generación de Texto

```bash
# Generación interactiva
python robo_poet.py generate \
    --model models/final/model.h5 \
    --seed "En un lugar de la Mancha" \
    --length 200 \
    --temperature 0.8

# Generación con diferentes estrategias de sampling
python robo_poet.py generate \
    --model models/final/model.h5 \
    --seed "El futuro de la tecnología" \
    --length 500 \
    --strategy nucleus \
    --top-p 0.95

# Modo interactivo (REPL)
python robo_poet.py generate --model models/final/model.h5 --interactive
```

#### Evaluación

```bash
# Evaluar en conjunto de test
python robo_poet.py evaluate \
    --model models/final/model.h5 \
    --test-data data/raw/test.txt \
    --metrics perplexity,bleu,diversity

# Análisis detallado con visualizaciones
python robo_poet.py evaluate \
    --model models/final/model.h5 \
    --test-data data/raw/test.txt \
    --detailed \
    --output-dir results/
```

### Monitoreo con TensorBoard

```bash
# Durante entrenamiento, en terminal separada:
tensorboard --logdir logs --port 6006

# Acceder desde navegador:
# http://localhost:6006
```

Métricas visualizadas:
- Loss (train/validation)
- Perplexity
- Learning rate schedule
- Gradient norms
- Embedding projector (t-SNE/PCA)

## Preparación de Datos

### Formato de Entrada

El sistema acepta archivos de texto plano (.txt) con codificación UTF-8:

```python
# Ejemplo de preprocesamiento personalizado
def prepare_custom_dataset(input_file, output_dir):
    """
    Preprocesa texto para entrenamiento óptimo
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Limpieza básica
    text = text.replace('\r\n', '\n')  # Normalizar line endings
    text = re.sub(r'\n{3,}', '\n\n', text)  # Máximo 2 líneas vacías
    text = text.strip()
    
    # Guardar versión procesada
    processed_file = os.path.join(output_dir, 'processed.txt')
    with open(processed_file, 'w', encoding='utf-8') as f:
        f.write(text)
    
    return processed_file
```

### Datasets Recomendados para Práctica

1. **Literatura (Project Gutenberg)**
   ```bash
   wget https://www.gutenberg.org/files/2000/2000-0.txt -O data/raw/quijote.txt
   ```

2. **Código Python (GitHub)**
   ```bash
   # Clonar repositorio popular
   git clone https://github.com/tensorflow/models.git temp/
   find temp/ -name "*.py" -exec cat {} \; > data/raw/python_code.txt
   rm -rf temp/
   ```

3. **Wikipedia en Español**
   ```bash
   # Usar Wikipedia dump
   wget https://dumps.wikimedia.org/eswiki/latest/eswiki-latest-pages-articles.xml.bz2
   # Requiere procesamiento adicional con wikiextractor
   ```

## Optimización de Rendimiento

### Configuración para Máximo Rendimiento

```python
# config.py
import tensorflow as tf

def configure_gpu():
    """Configuración óptima para RTX 2000 Ada"""
    
    # Habilitar memory growth para evitar OOM
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"Error configurando GPU: {e}")
    
    # Mixed precision para Tensor Cores
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    
    # XLA compilation
    tf.config.optimizer.set_jit(True)
    
    # Thread tuning
    tf.config.threading.set_inter_op_parallelism_threads(4)
    tf.config.threading.set_intra_op_parallelism_threads(8)
```

### Benchmarks Esperados

Con configuración óptima en RTX 2000 Ada:

| Métrica | Valor |
|---------|-------|
| Tokens/segundo (training) | 15,000-20,000 |
| Batch size máximo | 32 (seq_len=100) |
| Tiempo/época (10MB dataset) | 8-12 minutos |
| Memoria GPU utilizada | 6.5-7.5 GB |
| Speedup con mixed precision | 1.5-1.8x |

## Troubleshooting

### Error: CUDA out of memory

```bash
# Solución 1: Reducir batch size
python robo_poet.py train --batch-size 16

# Solución 2: Reducir longitud de secuencia
python robo_poet.py train --sequence-length 50

# Solución 3: Usar gradient accumulation
python robo_poet.py train --batch-size 8 --accumulation-steps 4
```

### Error: Could not load dynamic library 'libcudnn.so.8'

```bash
# Verificar instalación de cuDNN
find /usr/local/cuda-11.8 -name "libcudnn*"

# Si no encuentra archivos, reinstalar cuDNN
# Si encuentra pero no carga, actualizar LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
```

### Error: No OpKernel was registered to support Op 'CudnnRNN'

```bash
# Reinstalar TensorFlow con soporte GPU correcto
pip uninstall tensorflow
pip install tensorflow[and-cuda]==2.15.0 --no-cache-dir
```

### Entrenamiento muy lento

```python
# Verificar que está usando GPU
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

# Verificar mixed precision activado
print(tf.keras.mixed_precision.global_policy())

# Profiling detallado
tf.profiler.experimental.start('logs/profile')
# ... código de entrenamiento ...
tf.profiler.experimental.stop()
```

## Contribuciones y Licencia

Este proyecto es con fines educacionales. Código base bajo MIT License.

### Estructura para Contribuciones

```bash
# Fork y clone
git clone https://github.com/tu-usuario/robo-poet.git
cd robo-poet

# Branch para feature
git checkout -b feature/mi-mejora

# Tests antes de PR
python -m pytest tests/
```

## Recursos Adicionales

- [TensorFlow Text Generation Tutorial](https://www.tensorflow.org/text/tutorials/text_generation)
- [NVIDIA Deep Learning Documentation](https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html)
- [Understanding LSTMs - Colah's Blog](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

## Contacto y Soporte

Para dudas sobre el curso o implementación:
- Issues: github.com/tu-repo/robo-poet/issues
- Documentación extendida: Ver CLAUDE.md
- Especificaciones técnicas: Ver SPECS.md