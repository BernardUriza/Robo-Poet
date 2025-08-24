# 🚀 SISTEMA MULTI-CORPUS ROBO-POET

## 🎯 ¿Qué cambió?

**ANTES:** El sistema requería especificar UN solo archivo de texto para entrenar
```bash
python robo_poet.py --text archivo.txt --model mi_modelo --epochs 20
```

**AHORA:** El sistema automáticamente usa TODOS los archivos .txt de la carpeta `corpus/`
```bash
python robo_poet.py --model mi_modelo --epochs 20  # ¡Usa todo el corpus automáticamente!
```

## 📚 Corpus Actual

Tu sistema tiene listo un corpus combinado con:

- **4 archivos .txt** en la carpeta `corpus/`
- **651.7 KB** de texto literario total
- **Alice in Wonderland** (2 versiones)
- **Hamlet de Shakespeare** (2 versiones)

## ⚡ Ventajas del Multi-Corpus

### 🎭 Diversidad de Estilos
- Combina prosa (Alice) y teatro (Hamlet)
- Mezcla narrativa y diálogos
- Vocabulario más rico y variado

### 🧠 Modelos Más Inteligentes
- Mayor generalización
- Capacidad de cambiar de estilo dentro del mismo texto
- Generaciones más impredecibles y creativas

### 📈 Facilidad de Uso
- No más decisiones sobre "¿cuál texto usar?"
- Agrega cualquier .txt a `corpus/` y se incluye automáticamente
- Un solo comando para entrenar con todo

## 🔧 Cómo Usar

### 1. Agregar Nuevos Textos
```bash
# Simplemente copia archivos .txt a corpus/
cp mi_novela.txt corpus/
cp poemas.txt corpus/
cp ensayos.txt corpus/
```

### 2. Entrenar Modelos
```bash
# Entrenamiento básico (usa TODOS los textos en corpus/)
python robo_poet.py --model literatura_completa --epochs 20

# Entrenamiento más intensivo
python robo_poet.py --model mega_modelo --epochs 50
```

### 3. Analizar y Generar
```bash
# Análisis usando multi-corpus
python robo_poet.py --analyze mi_modelo.keras
python robo_poet.py --minima mi_modelo.keras --config deep

# Generación
python robo_poet.py --generate mi_modelo.keras --seed "Once upon" --temp 0.9
```

## 🧪 Casos de Uso Académicos

### Investigación Literaria
- Entrenar con corpus de múltiples autores
- Comparar estilos generativos entre épocas
- Análisis de transferencia de estilo

### Experimentos de NLP
- Impacto del tamaño del corpus en la calidad
- Diversidad del vocabulario vs. coherencia
- Estudios de ablación con diferentes combinaciones

### Proyectos Estudiantiles
- Fácil agregación de textos nuevos
- Un solo modelo que "conoce" múltiples obras
- Generación más interessante para demos

## 📊 Archivos Modificados

### Core del Sistema
- `src/data_processor.py`: Nuevo método `load_text()` multi-corpus
- `src/orchestrator.py`: Comando `--model` usa corpus automáticamente
- Todos los analizadores actualizados para multi-corpus

### Archivos Movidos
- `alice_wonderland.txt` → `corpus/alice_wonderland.txt`
- `alice_raw.txt` → `corpus/alice_raw.txt`
- `hamlet_shakespeare.txt` → `corpus/hamlet_shakespeare.txt`
- `hamlet_raw.txt` → `corpus/hamlet_raw.txt`

## 🎓 Ejemplo Completo

```bash
# 1. Ver el corpus actual
python demo_multi_corpus.py

# 2. Entrenar con todo el corpus
python robo_poet.py --model shakespeare_alice --epochs 15

# 3. Generar texto mezclando estilos
python robo_poet.py --generate shakespeare_alice.keras --seed "To be or not to be" --temp 1.1

# 4. Analizar el modelo híbrido
python robo_poet.py --analyze shakespeare_alice.keras --batches 50
```

## 🚀 Impacto en el Proyecto

### Para el Usuario
- ✅ Más simple de usar (no más decisión de archivos)
- ✅ Modelos más ricos automáticamente
- ✅ Fácil experimentación con nuevos textos

### Para el Sistema
- ✅ Pipeline unificado de datos
- ✅ Aprovecha toda la literatura disponible
- ✅ Escalable (solo agrega archivos a corpus/)

### Para la Academia
- ✅ Experiments más controlados
- ✅ Corpus reproducibles
- ✅ Análisis comparativos más sólidos

---

**🦁 Sistema multi-corpus implementado por Aslan**  
**🧉 Todo funcionando como un buen mate compartido entre múltiples literaturas**