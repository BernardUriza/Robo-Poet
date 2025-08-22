# 🚀 Especificaciones Técnicas y Estrategias de Mejora - Robo-Poet v3.0
## Documento de Referencia para Desarrollo

### 📋 RESUMEN EJECUTIVO DE AUDITORÍA

**Fecha de auditoría:** 22 de Agosto, 2025  
**Versión analizada:** 2.1  
**Estado del proyecto:** Funcional con necesidades críticas de corrección

#### Hallazgos Críticos Identificados
| Prioridad | Problema | Impacto | Solución Requerida |
|-----------|----------|---------|-------------------|
| 🔴 **CRÍTICO** | Arquitectura LSTM incorrecta (1 capa vs 2 especificadas) | Alto | Implementar 2 capas LSTM según CLAUDE.md |
| 🔴 **CRÍTICO** | Política GPU confusa con fallback a CPU | Alto | Eliminar fallback, GPU obligatoria |
| 🟡 **ALTA** | Código duplicado (robo_poet_original_backup.py) | Medio | Eliminar archivo legacy |
| 🟡 **ALTA** | 60+ checkpoints ocupando 101MB | Medio | Implementar rotación automática |
| 🟡 **ALTA** | Configuración GPU distribuida | Medio | Centralizar en un módulo |

---

## 🔴 ESTRATEGIA 0: CORRECCIONES CRÍTICAS INMEDIATAS

### 0.1 Corregir Arquitectura LSTM (PRIORIDAD MÁXIMA)

#### Problema Actual
```python
# INCORRECTO - Implementación actual en src/model.py
lstm_out = layers.LSTM(self.lstm_units, dropout=self.dropout_rate, return_sequences=False)
```

#### Implementación Correcta Requerida
```python
# CORRECTO - Según especificación en CLAUDE.md
class CorrectedLSTMModel(tf.keras.Model):
    """
    Arquitectura LSTM corregida según especificaciones académicas
    """
    def __init__(self, vocab_size, embedding_dim=128, lstm_units=256):
        super().__init__()
        
        # Embedding Layer
        self.embedding = layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,  # 128 dimensions según CLAUDE.md
            mask_zero=True
        )
        
        # LSTM Layer 1 - 256 units con return_sequences=True
        self.lstm_1 = layers.LSTM(
            units=256,  # NO 128, debe ser 256
            return_sequences=True,  # CRÍTICO: debe retornar secuencias
            dropout=0.3,
            recurrent_dropout=0.3,
            name='lstm_layer_1'
        )
        
        # Dropout después de LSTM 1
        self.dropout_1 = layers.Dropout(0.3)
        
        # LSTM Layer 2 - 256 units con return_sequences=True
        self.lstm_2 = layers.LSTM(
            units=256,  # Mantener 256 units
            return_sequences=True,  # Para generación secuencial
            dropout=0.3,
            recurrent_dropout=0.3,
            name='lstm_layer_2'
        )
        
        # Dropout después de LSTM 2
        self.dropout_2 = layers.Dropout(0.3)
        
        # Dense Output Layer
        self.dense_output = layers.Dense(
            units=vocab_size,
            activation='softmax'
        )
    
    def call(self, inputs, training=False):
        # Forward pass según arquitectura especificada
        x = self.embedding(inputs)
        x = self.lstm_1(x, training=training)
        x = self.dropout_1(x, training=training)
        x = self.lstm_2(x, training=training)
        x = self.dropout_2(x, training=training)
        outputs = self.dense_output(x)
        return outputs
    
    def get_config(self):
        return {
            'vocab_size': self.vocab_size,
            'embedding_dim': 128,
            'lstm_units': 256,
            'architecture': '2-layer-lstm-256-units'
        }
```

#### Archivos a Modificar
1. `src/model.py` - Líneas 45-120 (build_model method)
2. `src/config.py` - Actualizar lstm_units de 128 a 256

### 0.2 Eliminar Fallback a CPU - GPU Obligatoria

#### Problema Actual
```python
# INCORRECTO - src/config.py línea 89
force_gpu: bool = True  # GPU is MANDATORY
# Pero luego...
return '/CPU:0'  # Fallback contradictorio
```

#### Implementación Correcta
```python
# src/config.py - Nueva implementación sin fallback
def get_device_string(self) -> str:
    """
    Obtiene string del dispositivo GPU.
    Sistema termina si GPU no está disponible.
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    
    if not gpus:
        # Intentar detección WSL2
        try:
            with tf.device('/GPU:0'):
                test = tf.constant([1.0])
                _ = tf.reduce_sum(test)
            print("✅ GPU detectada via workaround WSL2")
            return '/GPU:0'
        except Exception as e:
            # TERMINAR SISTEMA - NO HAY FALLBACK
            print("\n" + "="*60)
            print("🔴 ERROR CRÍTICO: GPU NO DISPONIBLE")
            print("="*60)
            print("\nEste proyecto REQUIERE GPU para cumplir requisitos académicos.")
            print("\nSoluciones:")
            print("1. Verificar driver NVIDIA: nvidia-smi")
            print("2. Activar entorno conda: conda activate robo-poet-gpu")
            print("3. Verificar CUDA: python -c 'import tensorflow as tf; print(tf.config.list_physical_devices(\"GPU\"))'")
            print("\nSi estás en WSL2, asegúrate de:")
            print("- Tener Windows 11 o Windows 10 build 21H2+")
            print("- Driver NVIDIA actualizado en Windows (no en WSL2)")
            print("- nvidia-smi funciona desde WSL2")
            print("\n" + "="*60)
            
            # TERMINAR EJECUCIÓN
            import sys
            sys.exit(1)
    
    # GPU detectada exitosamente
    return '/GPU:0'
```

#### Actualización en Orchestrator
```python
# src/orchestrator.py - Eliminar toda lógica de CPU fallback
def initialize_system(self):
    """Inicializa sistema con GPU obligatoria"""
    if not self.gpu_manager.setup_gpu():
        print("\n🔴 SISTEMA TERMINADO: GPU es obligatoria para este proyecto académico")
        sys.exit(1)
    
    # NO hay código de fallback a CPU
    print("✅ Sistema iniciado con GPU exitosamente")
```

---

## 🟡 MEJORAS PRIORITARIAS DE AUDITORÍA

### Prioridad 1: Eliminar Código Duplicado

#### Acción Requerida
```bash
# Archivar código legacy
mkdir -p archive/legacy
mv robo_poet_original_backup.py archive/legacy/
echo "# Archivo movido a archive/legacy/" > robo_poet_original_backup.py.MOVED

# Actualizar .gitignore
echo "archive/legacy/" >> .gitignore
```

### Prioridad 2: Limpiar Directorio de Modelos

#### Script de Limpieza Automática
```python
# scripts/cleanup_models.py
import os
from pathlib import Path
from datetime import datetime, timedelta

def cleanup_old_checkpoints(models_dir='models/', keep_last=5, max_age_days=7):
    """
    Limpia checkpoints antiguos manteniendo solo los más recientes
    """
    model_files = []
    
    # Obtener todos los archivos de modelo
    for file in Path(models_dir).glob('*.h5'):
        stat = file.stat()
        model_files.append({
            'path': file,
            'modified': datetime.fromtimestamp(stat.st_mtime),
            'size': stat.st_size
        })
    
    # Ordenar por fecha de modificación
    model_files.sort(key=lambda x: x['modified'], reverse=True)
    
    # Mantener los últimos N modelos
    to_keep = model_files[:keep_last]
    to_delete = model_files[keep_last:]
    
    # Filtrar por edad máxima
    cutoff_date = datetime.now() - timedelta(days=max_age_days)
    to_delete = [f for f in to_delete if f['modified'] < cutoff_date]
    
    # Eliminar archivos
    total_freed = 0
    for file_info in to_delete:
        print(f"Eliminando: {file_info['path'].name} ({file_info['size']/1024/1024:.1f}MB)")
        file_info['path'].unlink()
        
        # Eliminar metadata JSON asociada
        json_path = file_info['path'].with_suffix('.json')
        if json_path.exists():
            json_path.unlink()
        
        total_freed += file_info['size']
    
    print(f"\n✅ Limpieza completada")
    print(f"   Archivos eliminados: {len(to_delete)}")
    print(f"   Espacio liberado: {total_freed/1024/1024:.1f}MB")
    print(f"   Modelos conservados: {len(to_keep)}")

if __name__ == "__main__":
    cleanup_old_checkpoints()
```

### Prioridad 3: Centralizar Configuración GPU

#### Nuevo Módulo Unificado
```python
# src/gpu_manager.py - Módulo centralizado para GPU
import os
import sys
import tensorflow as tf
from typing import Optional, Dict, Any

class GPUManager:
    """
    Gestor centralizado de configuración GPU
    Único punto de verdad para toda configuración GPU
    """
    
    def __init__(self):
        self.gpu_available = False
        self.gpu_name = None
        self.vram_gb = None
        self.device_string = None
        
    def detect_and_configure(self) -> bool:
        """
        Detecta y configura GPU. Termina si no hay GPU.
        """
        # Configurar variables de entorno
        self._setup_environment()
        
        # Detectar GPU
        if not self._detect_gpu():
            self._handle_no_gpu()
            return False
        
        # Configurar GPU
        self._configure_gpu()
        
        # Validar configuración
        if not self._validate_gpu():
            self._handle_no_gpu()
            return False
        
        return True
    
    def _setup_environment(self):
        """Configura variables de entorno para GPU"""
        conda_prefix = os.environ.get('CONDA_PREFIX', '')
        if conda_prefix:
            os.environ['CUDA_HOME'] = conda_prefix
            os.environ['LD_LIBRARY_PATH'] = f"{conda_prefix}/lib:{conda_prefix}/lib64"
        
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    def _detect_gpu(self) -> bool:
        """Detecta GPU disponible"""
        gpus = tf.config.list_physical_devices('GPU')
        
        if not gpus:
            # Intentar workaround WSL2
            try:
                with tf.device('/GPU:0'):
                    test = tf.constant([1.0])
                    _ = tf.reduce_sum(test)
                self.gpu_available = True
                self.device_string = '/GPU:0'
                self.gpu_name = "GPU (WSL2 mode)"
                return True
            except:
                return False
        
        self.gpu_available = True
        self.device_string = '/GPU:0'
        self.gpu_name = gpus[0].name
        return True
    
    def _configure_gpu(self):
        """Configura GPU para uso óptimo"""
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Memory growth
                tf.config.experimental.set_memory_growth(gpus[0], True)
                
                # Obtener info de GPU
                import subprocess
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    info = result.stdout.strip().split(',')
                    self.gpu_name = info[0].strip()
                    self.vram_gb = float(info[1].strip().replace('MiB', '')) / 1024
            except Exception as e:
                print(f"Advertencia configurando GPU: {e}")
    
    def _validate_gpu(self) -> bool:
        """Valida que GPU funciona correctamente"""
        try:
            with tf.device(self.device_string):
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
                c = tf.matmul(a, b)
                return True
        except:
            return False
    
    def _handle_no_gpu(self):
        """Maneja caso de GPU no disponible - TERMINA EL SISTEMA"""
        print("\n" + "="*70)
        print("🔴 ERROR CRÍTICO: GPU NO DISPONIBLE - SISTEMA TERMINADO")
        print("="*70)
        print("\nEste proyecto REQUIERE GPU (NVIDIA) para funcionar.")
        print("\n📋 Lista de verificación:")
        print("  1. ¿nvidia-smi funciona? → Verifica driver NVIDIA")
        print("  2. ¿Conda activado? → conda activate robo-poet-gpu")
        print("  3. ¿CUDA instalado? → conda install -c conda-forge cudatoolkit")
        print("\n💡 Si estás en WSL2:")
        print("  - Windows 11 o Windows 10 21H2+ requerido")
        print("  - Driver NVIDIA debe estar en Windows (no WSL2)")
        print("  - Prueba: nvidia-smi desde terminal WSL2")
        print("="*70)
        sys.exit(1)
    
    def get_config(self) -> Dict[str, Any]:
        """Retorna configuración actual de GPU"""
        return {
            'gpu_available': self.gpu_available,
            'gpu_name': self.gpu_name,
            'vram_gb': self.vram_gb,
            'device_string': self.device_string,
            'mixed_precision': False,  # Por estabilidad
            'memory_growth': True
        }

# Singleton para uso global
gpu_manager = GPUManager()
```

### Prioridad 4: Estandarizar Logging

```python
# src/logger.py - Sistema de logging unificado
import logging
import sys
from pathlib import Path
from datetime import datetime

class RoboPoetLogger:
    """Sistema de logging centralizado"""
    
    @staticmethod
    def setup_logger(name='robo-poet', level=logging.INFO):
        """Configura logger unificado"""
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # Formato consistente
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(module)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(
            log_dir / f"robo_poet_{datetime.now():%Y%m%d}.log"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger

# Logger global
logger = RoboPoetLogger.setup_logger()
```

### Prioridad 5: Consistencia de Idioma

```python
# src/i18n.py - Mensajes en español consistente
class Messages:
    """Mensajes centralizados en español"""
    
    # Sistema
    WELCOME = "🎓 Bienvenido a Robo-Poet v2.1 - Generador Académico de Texto"
    GPU_REQUIRED = "🔴 ERROR: GPU obligatoria para este proyecto"
    GPU_DETECTED = "✅ GPU detectada: {gpu_name}"
    
    # Entrenamiento
    TRAINING_START = "🚀 Iniciando entrenamiento - Fase 1"
    TRAINING_PROGRESS = "📊 Época {epoch}/{total} - Loss: {loss:.3f}"
    TRAINING_COMPLETE = "✅ Entrenamiento completado exitosamente"
    
    # Generación
    GENERATION_START = "🎨 Iniciando generación de texto"
    GENERATION_MODE = "Modo seleccionado: {mode}"
    GENERATION_COMPLETE = "✅ Texto generado exitosamente"
    
    # Errores
    ERROR_FILE_NOT_FOUND = "❌ Archivo no encontrado: {file}"
    ERROR_INVALID_INPUT = "❌ Entrada inválida: {input}"
    ERROR_MODEL_LOAD = "❌ Error cargando modelo: {error}"
```

---

## 📊 ESTRATEGIAS AVANZADAS (Del specs.md original)

### Análisis de Resultados de Fase 2 - Diagnóstico Actualizado

#### Métricas Actuales del Modelo
| Métrica | Valor Actual | Estado |
|---------|--------------|--------|
| **Loss** | 2.1 | ⚠️ Necesita mejora (objetivo: <1.5) |
| **Vocabulario** | 44 caracteres | ❌ Muy limitado |
| **Temperature Óptima** | 0.6 | ✅ Sweet spot identificado |
| **Longitud Óptima** | 150 chars | ✅ Balance calidad/coherencia |
| **Velocidad Generación** | 16-31 chars/s | ✅ Aceptable |
| **Coherencia Máxima** | 77.1% @ T=0.6 | ⚠️ Mejorable |

[NOTA: El resto del contenido original de specs.md sobre estrategias 1-6 se mantiene igual, 
comenzando desde "## 🔥 ESTRATEGIA 1: Reentrenamiento Intensivo Optimizado" hasta el final]

---

## 🎯 PLAN DE IMPLEMENTACIÓN PARA DESARROLLO

### Fase 1: Correcciones Críticas (INMEDIATO)
1. [ ] Implementar arquitectura LSTM de 2 capas con 256 units
2. [ ] Eliminar todo código de fallback a CPU
3. [ ] Probar que sistema termina sin GPU

### Fase 2: Mejoras de Alta Prioridad (Semana 1)
4. [ ] Eliminar robo_poet_original_backup.py
5. [ ] Implementar limpieza automática de checkpoints
6. [ ] Centralizar configuración GPU en gpu_manager.py
7. [ ] Reemplazar print() con logging estructurado
8. [ ] Estandarizar mensajes en español

### Fase 3: Optimizaciones (Semana 2)
9. [ ] Expandir vocabulario a 5000 tokens
10. [ ] Implementar Weight-Dropped LSTM
11. [ ] Agregar tests unitarios básicos
12. [ ] Documentar APIs públicas

### Fase 4: Refinamiento (Semana 3-4)
13. [ ] Implementar métricas de evaluación continua
14. [ ] Optimizar batch sizes dinámicamente
15. [ ] Agregar benchmarking automatizado
16. [ ] Crear documentación de usuario final

---

## 📝 NOTAS PARA EL DESARROLLADOR

### Archivos Críticos a Modificar
1. **src/model.py** - Líneas 45-120 (arquitectura LSTM)
2. **src/config.py** - Línea 89 (eliminar CPU fallback)
3. **src/orchestrator.py** - Eliminar toda lógica de CPU
4. **robo_poet.py** - Actualizar para usar gpu_manager

### Comandos de Verificación Post-Cambios
```bash
# Verificar arquitectura LSTM
python -c "from src.model import RoboPoetModel; m = RoboPoetModel(); m.summary()"

# Verificar GPU obligatoria (debe fallar sin GPU)
CUDA_VISIBLE_DEVICES="" python robo_poet.py --help

# Verificar limpieza de modelos
python scripts/cleanup_models.py --dry-run

# Verificar logging unificado
python -c "from src.logger import logger; logger.info('Test')"
```

### Criterios de Aceptación
- [ ] Modelo usa exactamente 2 capas LSTM de 256 units cada una
- [ ] Sistema termina inmediatamente si no hay GPU disponible
- [ ] No existe código de fallback a CPU en ningún archivo
- [ ] Directorio models/ tiene máximo 5 checkpoints recientes
- [ ] Todos los mensajes están en español consistentemente
- [ ] Logging estructurado reemplaza todos los print()
- [ ] Tests pasan sin warnings ni errores

---

*Documento actualizado con correcciones críticas de auditoría y mejoras prioritarias*
*Preparado para implementación inmediata por equipo de desarrollo*