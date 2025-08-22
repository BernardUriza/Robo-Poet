"""
Gestor centralizado de configuración GPU para Robo-Poet.

Este módulo unifica toda la lógica de detección, configuración y validación GPU,
eliminando la configuración distribuida que existía en config.py, gpu_detection.py 
y orchestrator.py. Parte de mejoras Fase 2.
"""

import os
import sys
import subprocess
from typing import Optional, Dict, Any, Tuple
import tensorflow as tf


class GPUManager:
    """
    Gestor centralizado de configuración GPU.
    Único punto de verdad para toda configuración GPU del proyecto.
    """
    
    def __init__(self):
        """Inicializa el gestor GPU."""
        self.gpu_available: bool = False
        self.gpu_name: Optional[str] = None
        self.vram_gb: Optional[float] = None
        self.device_string: str = '/GPU:0'
        self.driver_version: Optional[str] = None
        self.cuda_version: Optional[str] = None
        self._initialized: bool = False
        
    def detect_and_configure(self) -> bool:
        """
        Detecta y configura GPU. Termina sistema si no hay GPU disponible.
        
        Returns:
            bool: True si GPU detectada y configurada exitosamente
        """
        if self._initialized:
            return self.gpu_available
            
        print("🔍 Iniciando detección y configuración GPU...")
        
        # 1. Configurar variables de entorno
        self._setup_environment()
        
        # 2. Detectar GPU
        if not self._detect_gpu():
            self._handle_no_gpu()
            return False
        
        # 3. Configurar GPU para uso óptimo
        self._configure_gpu()
        
        # 4. Validar configuración
        if not self._validate_gpu():
            self._handle_no_gpu()
            return False
        
        # 5. Obtener información detallada
        self._get_gpu_info()
        
        self._initialized = True
        self._print_gpu_summary()
        
        return True
    
    def _setup_environment(self) -> None:
        """Configura variables de entorno optimizadas para GPU."""
        conda_prefix = os.environ.get('CONDA_PREFIX', '')
        
        if conda_prefix:
            os.environ['CUDA_HOME'] = conda_prefix
            # Limpiar y configurar LD_LIBRARY_PATH
            lib_paths = [f"{conda_prefix}/lib", f"{conda_prefix}/lib64"]
            existing_ld = os.environ.get('LD_LIBRARY_PATH', '')
            if existing_ld:
                lib_paths.append(existing_ld)
            os.environ['LD_LIBRARY_PATH'] = ':'.join(lib_paths)
        
        # Configuraciones de TensorFlow optimizadas
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reducir verbosidad
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Memory growth
        os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'  # Threading optimizado
        
        # Asegurar GPU 0 visible (a menos que esté explícitamente deshabilitada)
        if 'CUDA_VISIBLE_DEVICES' not in os.environ:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    def _detect_gpu(self) -> bool:
        """
        Detecta GPU disponible usando múltiples estrategias.
        
        Returns:
            bool: True si GPU detectada
        """
        # Verificar si GPU fue forzadamente deshabilitada
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if cuda_visible == '':
            print("🚫 CUDA_VISIBLE_DEVICES=\"\" - GPU forzadamente deshabilitada")
            return False
        
        # Estrategia 1: Detección estándar TensorFlow
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            self.gpu_available = True
            self.gpu_name = gpus[0].name
            print(f"✅ GPU detectada (método estándar): {self.gpu_name}")
            return True
        
        # Estrategia 2: WSL2 Workaround - Test directo
        print("⚠️ Detección estándar falló, intentando workaround WSL2...")
        try:
            with tf.device('/GPU:0'):
                test_tensor = tf.constant([1.0, 2.0, 3.0])
                result = tf.reduce_sum(test_tensor)
            
            self.gpu_available = True
            self.gpu_name = "GPU (WSL2 workaround mode)"
            print("✅ GPU detectada via workaround WSL2")
            return True
            
        except Exception as e:
            print(f"❌ Workaround WSL2 falló: {e}")
            return False
    
    def _configure_gpu(self) -> None:
        """Configura GPU para uso óptimo."""
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            try:
                # Configurar memory growth para evitar OOM
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("✅ Memory growth configurado")
                
            except RuntimeError as e:
                print(f"⚠️ Error configurando memory growth: {e}")
        
        # Configurar precisión (float32 por estabilidad)
        policy = tf.keras.mixed_precision.Policy('float32')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("✅ Precisión configurada: float32")
    
    def _validate_gpu(self) -> bool:
        """
        Valida que GPU funciona correctamente con operaciones básicas.
        
        Returns:
            bool: True si GPU funcional
        """
        try:
            with tf.device(self.device_string):
                # Test de operación matricial
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
                c = tf.matmul(a, b)
                
                # Test de reducción
                d = tf.reduce_sum(c)
                
            print("✅ Validación GPU exitosa - operaciones básicas funcionando")
            return True
            
        except Exception as e:
            print(f"❌ Validación GPU falló: {e}")
            return False
    
    def _get_gpu_info(self) -> None:
        """Obtiene información detallada de la GPU via nvidia-smi."""
        try:
            # Obtener información de GPU
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=name,driver_version,memory.total,memory.free',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                info = result.stdout.strip().split(',')
                if len(info) >= 4:
                    self.gpu_name = info[0].strip()
                    self.driver_version = info[1].strip()
                    vram_total = float(info[2].strip())
                    vram_free = float(info[3].strip())
                    self.vram_gb = vram_total / 1024  # Convert MB to GB
                    
                    print(f"✅ Información GPU obtenida via nvidia-smi")
                else:
                    print("⚠️ Información nvidia-smi incompleta")
            else:
                print("⚠️ nvidia-smi no disponible o falló")
                
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            print(f"⚠️ Error obteniendo información GPU: {e}")
    
    def _handle_no_gpu(self) -> None:
        """
        Maneja caso de GPU no disponible - TERMINA EL SISTEMA.
        No hay fallback a CPU.
        """
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
        print("\n🔧 Si CUDA_VISIBLE_DEVICES=\"\":")
        print("  - Esto deshabilita la GPU forzadamente")
        print("  - Para habilitar: export CUDA_VISIBLE_DEVICES=0")
        print("="*70)
        
        sys.exit(1)
    
    def _print_gpu_summary(self) -> None:
        """Imprime resumen de configuración GPU."""
        print("\n" + "="*50)
        print("🎯 CONFIGURACIÓN GPU COMPLETADA")
        print("="*50)
        print(f"GPU: {self.gpu_name}")
        if self.driver_version:
            print(f"Driver: {self.driver_version}")
        if self.vram_gb:
            print(f"VRAM: {self.vram_gb:.1f}GB")
        print(f"Device: {self.device_string}")
        print(f"Mixed Precision: float32 (estabilidad)")
        print(f"Memory Growth: Habilitado")
        print("="*50)
    
    def get_device_string(self) -> str:
        """
        Obtiene string del dispositivo GPU.
        
        Returns:
            str: String del dispositivo GPU
        """
        if not self._initialized:
            if not self.detect_and_configure():
                # Ya maneja el sys.exit(1) internamente
                pass
        
        return self.device_string
    
    def get_config(self) -> Dict[str, Any]:
        """
        Retorna configuración actual completa de GPU.
        
        Returns:
            Dict con toda la configuración GPU
        """
        if not self._initialized:
            self.detect_and_configure()
            
        return {
            'gpu_available': self.gpu_available,
            'gpu_name': self.gpu_name,
            'driver_version': self.driver_version,
            'vram_gb': self.vram_gb,
            'device_string': self.device_string,
            'mixed_precision': 'float32',
            'memory_growth': True,
            'cuda_home': os.environ.get('CUDA_HOME'),
            'ld_library_path': os.environ.get('LD_LIBRARY_PATH'),
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'initialized': self._initialized
        }
    
    def get_memory_info(self) -> Dict[str, float]:
        """
        Obtiene información actual de memoria GPU.
        
        Returns:
            Dict con información de memoria
        """
        try:
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=memory.total,memory.used,memory.free',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                info = result.stdout.strip().split(',')
                if len(info) >= 3:
                    return {
                        'total_mb': float(info[0].strip()),
                        'used_mb': float(info[1].strip()),
                        'free_mb': float(info[2].strip()),
                        'total_gb': float(info[0].strip()) / 1024,
                        'used_gb': float(info[1].strip()) / 1024,
                        'free_gb': float(info[2].strip()) / 1024,
                        'usage_percent': (float(info[1].strip()) / float(info[0].strip())) * 100
                    }
        except Exception as e:
            print(f"⚠️ Error obteniendo memoria GPU: {e}")
            
        return {
            'error': 'No se pudo obtener información de memoria GPU'
        }
    
    def is_available(self) -> bool:
        """
        Verifica si GPU está disponible.
        
        Returns:
            bool: True si GPU disponible
        """
        if not self._initialized:
            return self.detect_and_configure()
        return self.gpu_available


# Singleton global para uso en todo el proyecto
gpu_manager = GPUManager()


def get_gpu_device() -> str:
    """
    Función de conveniencia para obtener device string.
    
    Returns:
        str: Device string GPU
    """
    return gpu_manager.get_device_string()


def ensure_gpu() -> None:
    """
    Función de conveniencia para asegurar que GPU esté disponible.
    Termina el sistema si no hay GPU.
    """
    if not gpu_manager.detect_and_configure():
        # Ya maneja sys.exit(1) internamente
        pass


def get_gpu_config() -> Dict[str, Any]:
    """
    Función de conveniencia para obtener configuración GPU completa.
    
    Returns:
        Dict: Configuración GPU
    """
    return gpu_manager.get_config()


if __name__ == "__main__":
    """Test del GPU manager."""
    print("🧪 PROBANDO GPU MANAGER")
    print("="*40)
    
    # Test de detección
    available = gpu_manager.detect_and_configure()
    print(f"GPU disponible: {available}")
    
    if available:
        # Test de configuración
        config = gpu_manager.get_config()
        print(f"\n📊 Configuración:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        
        # Test de memoria
        memory = gpu_manager.get_memory_info()
        print(f"\n💾 Memoria:")
        for key, value in memory.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.1f}")
            else:
                print(f"  {key}: {value}")