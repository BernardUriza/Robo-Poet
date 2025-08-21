#!/usr/bin/env python3
"""
Script de verificación para robo-poet
Verifica que TensorFlow detecte la GPU correctamente
"""

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
    
    # Configurar memory growth
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("✓ Memory growth configurado")
    except RuntimeError as e:
        print(f"Advertencia: {e}")
        
else:
    print("✗ No se detectó GPU")
    sys.exit(1)

# Test básico en GPU
print("\nProbando operación en GPU...")
with tf.device('/GPU:0'):
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
    c = tf.matmul(a, b)
    print(f"✓ Multiplicación matricial exitosa: {c.numpy()}")

# Verificar mixed precision
try:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print("✓ Mixed precision (FP16) disponible")
except Exception as e:
    print(f"⚠ Mixed precision no disponible: {e}")

print("=" * 50)
print("¡Configuración lista para entrenar!")
print("=" * 50)