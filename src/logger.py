"""
Sistema de logging estructurado para Robo-Poet.

Centraliza todo el logging del proyecto, reemplazando print() statements
dispersos con logging estructurado con niveles, timestamps y formato consistente.
Parte de mejoras Fase 2.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


class RoboPoetLogger:
    """Sistema de logging centralizado para Robo-Poet."""
    
    _instances = {}  # Singleton instances por nombre
    
    @staticmethod
    def get_logger(name: str = 'robo-poet', level: int = logging.INFO) -> logging.Logger:
        """
        Obtiene o crea un logger con configuración unificada.
        
        Args:
            name: Nombre del logger (por módulo)
            level: Nivel de logging
            
        Returns:
            Logger configurado
        """
        if name in RoboPoetLogger._instances:
            return RoboPoetLogger._instances[name]
        
        # Crear nuevo logger
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # Evitar handlers duplicados
        if logger.handlers:
            logger.handlers.clear()
        
        # Formato consistente con emojis para mejor UX
        formatter = logging.Formatter(
            '%(asctime)s | %(name)-12s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler - siempre a stdout
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
        
        # File handler - logs persistentes
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        # Archivo de log diario
        log_filename = f"robo_poet_{datetime.now():%Y%m%d}.log"
        file_handler = logging.FileHandler(log_dir / log_filename, encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)  # Archivo guarda todo
        logger.addHandler(file_handler)
        
        # Error handler separado para errores críticos
        error_filename = f"robo_poet_errors_{datetime.now():%Y%m%d}.log"
        error_handler = logging.FileHandler(log_dir / error_filename, encoding='utf-8')
        error_handler.setFormatter(formatter)
        error_handler.setLevel(logging.ERROR)
        logger.addHandler(error_handler)
        
        # Guardar instancia
        RoboPoetLogger._instances[name] = logger
        
        return logger


# Pre-configurar loggers para módulos principales
main_logger = RoboPoetLogger.get_logger('main')
gpu_logger = RoboPoetLogger.get_logger('gpu', logging.INFO)
model_logger = RoboPoetLogger.get_logger('model', logging.INFO)
training_logger = RoboPoetLogger.get_logger('training', logging.INFO)
generation_logger = RoboPoetLogger.get_logger('generation', logging.INFO)
config_logger = RoboPoetLogger.get_logger('config', logging.INFO)


class LoggingMixin:
    """
    Mixin para agregar logging estructurado a cualquier clase.
    """
    
    @property
    def logger(self) -> logging.Logger:
        """
        Obtiene logger específico para la clase.
        
        Returns:
            Logger configurado para esta clase
        """
        if not hasattr(self, '_logger'):
            class_name = self.__class__.__name__.lower()
            self._logger = RoboPoetLogger.get_logger(class_name)
        return self._logger


def log_gpu_info(gpu_config: dict) -> None:
    """
    Log estructurado de información GPU.
    
    Args:
        gpu_config: Configuración GPU del gpu_manager
    """
    gpu_logger.info("=" * 50)
    gpu_logger.info("🎯 CONFIGURACIÓN GPU COMPLETADA")
    gpu_logger.info("=" * 50)
    gpu_logger.info(f"GPU: {gpu_config.get('gpu_name', 'Desconocida')}")
    
    if gpu_config.get('driver_version'):
        gpu_logger.info(f"Driver: {gpu_config['driver_version']}")
    
    if gpu_config.get('vram_gb'):
        gpu_logger.info(f"VRAM: {gpu_config['vram_gb']:.1f}GB")
    
    gpu_logger.info(f"Device: {gpu_config.get('device_string', '/GPU:0')}")
    gpu_logger.info(f"Mixed Precision: {gpu_config.get('mixed_precision', 'float32')}")
    gpu_logger.info(f"Memory Growth: {gpu_config.get('memory_growth', True)}")
    gpu_logger.info("=" * 50)


def log_model_architecture(model_summary: str, param_count: int) -> None:
    """
    Log estructurado de arquitectura de modelo.
    
    Args:
        model_summary: Resumen del modelo de Keras
        param_count: Número de parámetros
    """
    model_logger.info("🧠 ARQUITECTURA DEL MODELO")
    model_logger.info("=" * 40)
    model_logger.info(f"Parámetros totales: {param_count:,}")
    model_logger.debug("Arquitectura completa:")
    model_logger.debug(model_summary)
    model_logger.info("=" * 40)


def log_training_progress(epoch: int, total_epochs: int, loss: float, 
                         val_loss: Optional[float] = None, 
                         accuracy: Optional[float] = None) -> None:
    """
    Log estructurado de progreso de entrenamiento.
    
    Args:
        epoch: Época actual
        total_epochs: Total de épocas
        loss: Loss de entrenamiento
        val_loss: Loss de validación (opcional)
        accuracy: Accuracy (opcional)
    """
    progress = f"[{epoch:3d}/{total_epochs:3d}]"
    msg = f"🚀 Entrenamiento {progress} - Loss: {loss:.4f}"
    
    if val_loss is not None:
        msg += f" - Val Loss: {val_loss:.4f}"
    
    if accuracy is not None:
        msg += f" - Accuracy: {accuracy:.3f}"
    
    training_logger.info(msg)


def log_generation_result(seed_text: str, generated_text: str, 
                         temperature: float, length: int) -> None:
    """
    Log estructurado de resultado de generación.
    
    Args:
        seed_text: Texto semilla
        generated_text: Texto generado
        temperature: Temperature usada
        length: Longitud generada
    """
    generation_logger.info("🎨 GENERACIÓN COMPLETADA")
    generation_logger.info(f"Seed: '{seed_text[:30]}{'...' if len(seed_text) > 30 else ''}'")
    generation_logger.info(f"Temperature: {temperature} | Longitud: {length}")
    generation_logger.debug(f"Texto generado: {generated_text[:100]}...")


def log_error(error: Exception, context: str = "") -> None:
    """
    Log estructurado de errores.
    
    Args:
        error: Excepción ocurrida
        context: Contexto adicional del error
    """
    error_msg = f"❌ ERROR: {type(error).__name__}: {str(error)}"
    if context:
        error_msg += f" | Contexto: {context}"
    
    # Log tanto en main como en archivo de errores
    main_logger.error(error_msg, exc_info=True)


def log_phase_start(phase: str, description: str) -> None:
    """
    Log estructurado de inicio de fase.
    
    Args:
        phase: Nombre de la fase
        description: Descripción de la fase
    """
    main_logger.info("=" * 60)
    main_logger.info(f"🚀 INICIANDO {phase.upper()}")
    main_logger.info(f"Descripción: {description}")
    main_logger.info("=" * 60)


def log_phase_complete(phase: str, duration_seconds: float) -> None:
    """
    Log estructurado de finalización de fase.
    
    Args:
        phase: Nombre de la fase
        duration_seconds: Duración en segundos
    """
    minutes = int(duration_seconds // 60)
    seconds = int(duration_seconds % 60)
    
    main_logger.info("=" * 60)
    main_logger.info(f"✅ {phase.upper()} COMPLETADA")
    main_logger.info(f"Duración: {minutes}m {seconds}s")
    main_logger.info("=" * 60)


def log_system_info() -> None:
    """Log de información del sistema al inicio."""
    import platform
    import os
    
    main_logger.info("🎓 ROBO-POET v2.1 - Sistema de Generación de Texto Académico")
    main_logger.info("=" * 60)
    main_logger.info(f"Sistema: {platform.system()} {platform.release()}")
    main_logger.info(f"Python: {platform.python_version()}")
    main_logger.info(f"Directorio: {os.getcwd()}")
    main_logger.info(f"Usuario: {os.environ.get('USER', 'Desconocido')}")
    main_logger.info("=" * 60)


def setup_logging(level: int = logging.INFO) -> None:
    """
    Configuración inicial del sistema de logging.
    
    Args:
        level: Nivel global de logging
    """
    # Configurar logging root
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Silenciar logs verbosos de TensorFlow y otras librerías
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.getLogger('absl').setLevel(logging.ERROR)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    # Log inicial del sistema
    log_system_info()
    main_logger.info("📝 Sistema de logging iniciado correctamente")


if __name__ == "__main__":
    """Test del sistema de logging."""
    print("🧪 PROBANDO SISTEMA DE LOGGING")
    print("="*40)
    
    # Setup inicial
    setup_logging()
    
    # Test de diferentes niveles
    main_logger.debug("🔍 Mensaje de debug")
    main_logger.info("ℹ️ Mensaje informativo")
    main_logger.warning("⚠️ Mensaje de advertencia")
    main_logger.error("❌ Mensaje de error")
    
    # Test de funciones especializadas
    log_phase_start("Prueba", "Testing del sistema de logging")
    
    # Test de diferentes loggers
    gpu_logger.info("✅ GPU configurada correctamente")
    model_logger.info("🧠 Modelo construido: 1,234,567 parámetros")
    training_logger.info("🚀 Entrenamiento [001/050] - Loss: 2.1234")
    generation_logger.info("🎨 Texto generado exitosamente")
    
    log_phase_complete("Prueba", 42.5)
    
    print(f"\n📁 Logs guardados en: logs/robo_poet_{datetime.now():%Y%m%d}.log")