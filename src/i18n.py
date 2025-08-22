"""
Sistema de internacionalización (i18n) para Robo-Poet.

Centraliza todos los mensajes en español consistente, eliminando la mezcla
de idiomas que existía en el proyecto. Parte de mejoras Fase 2.
"""

from typing import Dict, Any


class Messages:
    """
    Mensajes centralizados en español consistente.
    Organizado por categorías para fácil mantenimiento.
    """
    
    # ===== MENSAJES DEL SISTEMA =====
    WELCOME = "🎓 Bienvenido a Robo-Poet v2.1 - Generador Académico de Texto"
    VERSION = "Robo-Poet v2.1 - Interfaz Académica para Generación de Texto"
    SYSTEM_READY = "✅ Sistema listo para usar"
    SYSTEM_SHUTDOWN = "👋 Sistema finalizado correctamente"
    
    # ===== MENSAJES DE GPU =====
    GPU_DETECTION_START = "🔍 Iniciando detección y configuración GPU..."
    GPU_DETECTED_STANDARD = "✅ GPU detectada (método estándar): {gpu_name}"
    GPU_DETECTED_WSL2 = "✅ GPU detectada via workaround WSL2"
    GPU_WSL2_TRYING = "⚠️ Detección estándar falló, intentando workaround WSL2..."
    GPU_WSL2_FAILED = "❌ Workaround WSL2 falló: {error}"
    GPU_FORCED_DISABLED = "🚫 CUDA_VISIBLE_DEVICES=\"\" - GPU forzadamente deshabilitada"
    GPU_MEMORY_GROWTH = "✅ Memory growth configurado"
    GPU_PRECISION_SET = "✅ Precisión configurada: {precision}"
    GPU_VALIDATION_OK = "✅ Validación GPU exitosa - operaciones básicas funcionando"
    GPU_VALIDATION_FAILED = "❌ Validación GPU falló: {error}"
    GPU_INFO_NVIDIA_SMI = "✅ Información GPU obtenida via nvidia-smi"
    GPU_INFO_UNAVAILABLE = "⚠️ nvidia-smi no disponible o falló"
    GPU_INFO_ERROR = "⚠️ Error obteniendo información GPU: {error}"
    GPU_MEMORY_ERROR = "⚠️ Error obteniendo memoria GPU: {error}"
    
    # Error crítico GPU
    GPU_CRITICAL_ERROR_HEADER = "🔴 ERROR CRÍTICO: GPU NO DISPONIBLE - SISTEMA TERMINADO"
    GPU_CRITICAL_ERROR_MSG = "Este proyecto REQUIERE GPU (NVIDIA) para funcionar."
    GPU_CRITICAL_ERROR_CHECKLIST = "📋 Lista de verificación:"
    GPU_CRITICAL_ERROR_CHECK1 = "  1. ¿nvidia-smi funciona? → Verifica driver NVIDIA"
    GPU_CRITICAL_ERROR_CHECK2 = "  2. ¿Conda activado? → conda activate robo-poet-gpu"
    GPU_CRITICAL_ERROR_CHECK3 = "  3. ¿CUDA instalado? → conda install -c conda-forge cudatoolkit"
    GPU_CRITICAL_ERROR_WSL2 = "💡 Si estás en WSL2:"
    GPU_CRITICAL_ERROR_WSL2_1 = "  - Windows 11 o Windows 10 21H2+ requerido"
    GPU_CRITICAL_ERROR_WSL2_2 = "  - Driver NVIDIA debe estar en Windows (no WSL2)"
    GPU_CRITICAL_ERROR_WSL2_3 = "  - Prueba: nvidia-smi desde terminal WSL2"
    GPU_CRITICAL_ERROR_CUDA_DISABLED = "🔧 Si CUDA_VISIBLE_DEVICES=\"\":"
    GPU_CRITICAL_ERROR_CUDA_DISABLED_1 = "  - Esto deshabilita la GPU forzadamente"
    GPU_CRITICAL_ERROR_CUDA_DISABLED_2 = "  - Para habilitar: export CUDA_VISIBLE_DEVICES=0"
    
    # Configuración GPU completada
    GPU_CONFIG_COMPLETE_HEADER = "🎯 CONFIGURACIÓN GPU COMPLETADA"
    GPU_CONFIG_GPU = "GPU: {gpu_name}"
    GPU_CONFIG_DRIVER = "Driver: {driver_version}"
    GPU_CONFIG_VRAM = "VRAM: {vram_gb:.1f}GB"
    GPU_CONFIG_DEVICE = "Device: {device_string}"
    GPU_CONFIG_PRECISION = "Mixed Precision: {precision}"
    GPU_CONFIG_MEMORY_GROWTH = "Memory Growth: {enabled}"
    
    # ===== MENSAJES DE MODELO =====
    MODEL_BUILDING_START = "🧠 Construyendo modelo LSTM corregido de 2 capas..."
    MODEL_BUILDING_ARCH = "Arquitectura: {layers} capas LSTM de {units} units cada una"
    MODEL_BUILDING_VOCAB = "Tamaño de vocabulario: {vocab_size}"
    MODEL_BUILDING_SEQ_LEN = "Longitud de secuencia: {seq_length}"
    MODEL_BUILDING_UNITS = "Unidades LSTM: {units} (CORREGIDO desde {old_units})"
    MODEL_BUILDING_DROPOUT = "Dropout: {dropout} (regularización mejorada)"
    MODEL_BUILDING_JIT = "Compilación JIT: {status}"
    MODEL_BUILT_SUCCESS = "✅ Modelo construido: {param_count:,} parámetros"
    MODEL_ARCHITECTURE_HEADER = "🧠 ARQUITECTURA DEL MODELO"
    MODEL_ARCHITECTURE_PARAMS = "Parámetros totales: {param_count:,}"
    
    # ===== MENSAJES DE ENTRENAMIENTO =====
    TRAINING_START = "🚀 Iniciando entrenamiento - Fase 1"
    TRAINING_START_INTENSIVE = "🔥 Iniciando entrenamiento intensivo de {epochs} épocas"
    TRAINING_PROGRESS = "🚀 Entrenamiento [{epoch:3d}/{total:3d}] - Loss: {loss:.4f}"
    TRAINING_PROGRESS_VAL = "🚀 Entrenamiento [{epoch:3d}/{total:3d}] - Loss: {loss:.4f} - Val Loss: {val_loss:.4f}"
    TRAINING_PROGRESS_ACC = "🚀 Entrenamiento [{epoch:3d}/{total:3d}] - Loss: {loss:.4f} - Accuracy: {accuracy:.3f}"
    TRAINING_COMPLETE = "✅ Entrenamiento completado exitosamente"
    TRAINING_DURATION = "⏱️ Duración del entrenamiento: {duration}"
    TRAINING_SAVED = "💾 Modelo guardado: {model_path}"
    TRAINING_METADATA_SAVED = "📝 Metadata guardada: {metadata_path}"
    TRAINING_EARLY_STOP = "⏹️ Parada temprana activada - épocas sin mejora: {patience}"
    TRAINING_BEST_LOSS = "🏆 Mejor loss alcanzado: {best_loss:.4f} en época {epoch}"
    
    # ===== MENSAJES DE GENERACIÓN =====
    GENERATION_START = "🎨 Iniciando generación de texto"
    GENERATION_MODE = "Modo seleccionado: {mode}"
    GENERATION_SEED = "Texto semilla: '{seed}'"
    GENERATION_PARAMS = "Parámetros: Temperature={temp} | Longitud={length}"
    GENERATION_COMPLETE = "✅ Texto generado exitosamente"
    GENERATION_RESULT_HEADER = "🎨 RESULTADO DE GENERACIÓN"
    GENERATION_TIME = "⏱️ Tiempo de generación: {duration:.2f}s"
    GENERATION_SPEED = "🚀 Velocidad: {speed:.1f} caracteres/segundo"
    GENERATION_QUALITY = "📊 Calidad estimada: {quality:.1%}"
    
    # Modos de generación
    GENERATION_MODE_QUICK = "Generación Rápida"
    GENERATION_MODE_LABORATORY = "Laboratorio Creativo"
    GENERATION_MODE_INTERACTIVE = "Sesión Interactiva"
    GENERATION_MODE_BATCH = "Experimentos en Lote"
    
    # ===== MENSAJES DE ARCHIVOS =====
    FILE_READING = "📖 Leyendo archivo: {filename}"
    FILE_READ_SUCCESS = "✅ Archivo leído exitosamente: {size:,} caracteres"
    FILE_NOT_FOUND = "❌ Archivo no encontrado: {filename}"
    FILE_PROCESSING = "⚙️ Procesando corpus de texto..."
    FILE_PROCESSING_COMPLETE = "✅ Procesamiento de texto completado"
    FILE_SAVE_SUCCESS = "💾 Archivo guardado: {filename}"
    FILE_SAVE_ERROR = "❌ Error guardando archivo: {error}"
    
    # ===== MENSAJES DE INTERFAZ =====
    MENU_MAIN_TITLE = "🎓 Robo-Poet v2.1 - Menú Principal"
    MENU_PHASE1_TITLE = "🔥 FASE 1: Entrenamiento Intensivo"
    MENU_PHASE2_TITLE = "🎨 FASE 2: Generación de Texto"
    MENU_MODELS_TITLE = "📊 Gestión de Modelos"
    MENU_CONFIG_TITLE = "⚙️ Configuración del Sistema"
    MENU_OPTION_INVALID = "❌ Opción inválida. Por favor, selecciona una opción válida."
    MENU_RETURNING_MAIN = "🔙 Volviendo al menú principal..."
    MENU_EXITING = "👋 Saliendo del programa..."
    
    # Entrada de usuario
    INPUT_PROMPT_GENERIC = "➤ Selecciona una opción: "
    INPUT_PROMPT_FILENAME = "➤ Nombre del archivo de texto: "
    INPUT_PROMPT_EPOCHS = "➤ Número de épocas de entrenamiento: "
    INPUT_PROMPT_SEED = "➤ Texto semilla para generación: "
    INPUT_PROMPT_LENGTH = "➤ Longitud del texto a generar: "
    INPUT_PROMPT_TEMPERATURE = "➤ Temperature (0.1-2.0): "
    INPUT_INVALID = "❌ Entrada inválida: {input_value}"
    INPUT_VALIDATION_ERROR = "❌ Error de validación: {error}"
    
    # ===== MENSAJES DE ERROR =====
    ERROR_GENERIC = "❌ Error: {error}"
    ERROR_UNEXPECTED = "❌ Error inesperado: {error}"
    ERROR_GPU_MANDATORY = "🔴 ERROR: GPU obligatoria para este proyecto académico"
    ERROR_MODEL_LOAD = "❌ Error cargando modelo: {error}"
    ERROR_MODEL_SAVE = "❌ Error guardando modelo: {error}"
    ERROR_TEXT_PROCESSING = "❌ Error procesando texto: {error}"
    ERROR_GENERATION_FAILED = "❌ Generación de texto falló: {error}"
    ERROR_CONFIG_INVALID = "❌ Configuración inválida: {config_item}"
    
    # ===== MENSAJES DE LIMPIEZA =====
    CLEANUP_START = "🧹 Iniciando limpieza automática de checkpoints..."
    CLEANUP_HEADER = "🧹 LIMPIEZA AUTOMÁTICA DE CHECKPOINTS"
    CLEANUP_STATUS_INITIAL = "📊 Estado inicial del directorio {directory}:"
    CLEANUP_FILES_FOUND = "   Total archivos encontrados: {count}"
    CLEANUP_FILES_H5 = "   Archivos .h5: {count}"
    CLEANUP_FILES_KERAS = "   Archivos .keras: {count}"
    CLEANUP_SIZE_TOTAL = "   Tamaño total: {size_mb:.1f}MB"
    CLEANUP_STRATEGY = "📋 Estrategia de limpieza:"
    CLEANUP_KEEP_LAST = "   Conservar últimos {count} archivos"
    CLEANUP_MAX_AGE = "   Eliminar archivos con más de {days} días"
    CLEANUP_TO_KEEP = "   Archivos a conservar: {count}"
    CLEANUP_TO_DELETE = "   Archivos a eliminar: {count}"
    CLEANUP_NO_FILES = "✅ No hay archivos para eliminar"
    CLEANUP_MARKED_FOR_DELETION = "🗑️ Archivos marcados para eliminación:"
    CLEANUP_FILE_ENTRY = "   📄 {filename} ({size_mb:.1f}MB, {age_days}d)"
    CLEANUP_IMPACT = "📈 Impacto de limpieza:"
    CLEANUP_SPACE_TO_FREE = "   Espacio a liberar: {size_mb:.1f}MB"
    CLEANUP_REDUCTION_PERCENT = "   Reducción: {percent:.1f}%"
    CLEANUP_DRY_RUN = "🧪 DRY RUN - No se eliminaron archivos"
    CLEANUP_EXECUTING = "🚀 Ejecutando limpieza..."
    CLEANUP_COMPLETE = "✅ Limpieza completada"
    CLEANUP_FILES_DELETED = "   Archivos eliminados: {count}"
    CLEANUP_SPACE_FREED = "   Espacio liberado: {size_mb:.1f}MB"
    CLEANUP_FILES_KEPT = "   Modelos conservados: {count}"
    CLEANUP_JSON_DELETED = "   🗑️ También eliminado: {filename}"
    CLEANUP_ERROR_DELETING = "   ❌ Error eliminando {filename}: {error}"
    CLEANUP_SUMMARY_HEADER = "📊 RESUMEN FINAL:"
    CLEANUP_SIMULATION_NOTE = "   (Simulación - ejecutar sin --dry-run para aplicar cambios)"
    
    # ===== MENSAJES DE FASE =====
    PHASE_START = "🚀 INICIANDO {phase_name}"
    PHASE_DESCRIPTION = "Descripción: {description}"
    PHASE_COMPLETE = "✅ {phase_name} COMPLETADA"
    PHASE_DURATION = "Duración: {minutes}m {seconds}s"
    
    # ===== MENSAJES DE PROGRESO =====
    PROGRESS_PROCESSING = "⚙️ Procesando... {percent:.1f}% completado"
    PROGRESS_STEP = "📋 Paso {current}/{total}: {description}"
    PROGRESS_ESTIMATED_TIME = "⏱️ Tiempo estimado restante: {time}"
    
    # ===== MENSAJES DE CONFIGURACIÓN =====
    CONFIG_LOADING = "📋 Cargando configuración del sistema..."
    CONFIG_LOADED = "✅ Configuración cargada exitosamente"
    CONFIG_SAVED = "💾 Configuración guardada"
    CONFIG_RESET = "🔄 Configuración restablecida a valores por defecto"
    CONFIG_CUDA_HOME = "CUDA_HOME: {path}"
    CONFIG_LD_LIBRARY_PATH = "LD_LIBRARY_PATH configurado"
    CONFIG_ENV_VARS = "Variables de entorno configuradas para GPU"
    
    @staticmethod
    def get_separator(length: int = 60, char: str = "=") -> str:
        """
        Genera un separador visual.
        
        Args:
            length: Longitud del separador
            char: Carácter para el separador
            
        Returns:
            String separador
        """
        return char * length
    
    @staticmethod
    def format_error_block(title: str, message: str, suggestions: list = None) -> str:
        """
        Formatea un bloque de error consistente.
        
        Args:
            title: Título del error
            message: Mensaje principal
            suggestions: Lista de sugerencias (opcional)
            
        Returns:
            String con error formateado
        """
        separator = Messages.get_separator(70)
        lines = [
            "",
            separator,
            title,
            separator,
            f"\n{message}",
        ]
        
        if suggestions:
            lines.append("\n💡 Soluciones sugeridas:")
            for i, suggestion in enumerate(suggestions, 1):
                lines.append(f"  {i}. {suggestion}")
        
        lines.append(f"\n{separator}")
        
        return "\n".join(lines)
    
    @staticmethod
    def format_success_block(title: str, details: Dict[str, Any] = None) -> str:
        """
        Formatea un bloque de éxito consistente.
        
        Args:
            title: Título del éxito
            details: Detalles adicionales (opcional)
            
        Returns:
            String con éxito formateado
        """
        separator = Messages.get_separator(50)
        lines = [
            separator,
            title,
            separator,
        ]
        
        if details:
            for key, value in details.items():
                lines.append(f"{key}: {value}")
        
        lines.append(separator)
        
        return "\n".join(lines)


# Alias para facilitar el uso
Msg = Messages


if __name__ == "__main__":
    """Test del sistema de mensajes."""
    print("🧪 PROBANDO SISTEMA DE MENSAJES")
    print("="*40)
    
    # Test de mensajes básicos
    print(Msg.WELCOME)
    print(Msg.GPU_DETECTION_START)
    print(Msg.GPU_DETECTED_STANDARD.format(gpu_name="RTX 2000 Ada"))
    print(Msg.MODEL_BUILT_SUCCESS.format(param_count=1234567))
    print(Msg.TRAINING_PROGRESS.format(epoch=1, total=50, loss=2.1234))
    print(Msg.GENERATION_COMPLETE)
    
    # Test de formateo de bloques
    print("\n" + Msg.format_error_block(
        "ERROR DE PRUEBA",
        "Este es un mensaje de error de ejemplo",
        ["Solución 1: Hacer algo", "Solución 2: Hacer otra cosa"]
    ))
    
    print(Msg.format_success_block(
        "✅ PRUEBA EXITOSA",
        {
            "Archivos procesados": "15",
            "Tiempo transcurrido": "2.5s",
            "Estado": "Completado"
        }
    ))
    
    print(f"\n📝 Total de mensajes definidos: {len([attr for attr in dir(Msg) if not attr.startswith('_') and attr.isupper()])}")