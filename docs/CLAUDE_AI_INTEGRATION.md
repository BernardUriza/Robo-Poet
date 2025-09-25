# 🧠 Claude AI Integration - Ciclo de Entrenamiento Inteligente

## ¿Qué es el Ciclo Inteligente?

El **Ciclo Inteligente** es una nueva característica que utiliza Claude AI para mejorar dinámicamente el dataset durante el entrenamiento, creando un bucle de retroalimentación inteligente:

```
🔄 CICLO: Entrenar → Evaluar → Consultar Claude → Mejorar Dataset → Repetir
```

## Instalación Rápida

```bash
# 1. Instalar dependencias de Claude
python setup_claude_integration.py

# 2. O manualmente:
pip install -r requirements_claude.txt

# 3. Configurar API key
export CLAUDE_API_KEY=tu_api_key_aqui
```

## Cómo Usar

### 1. Obtener API Key de Claude
- Ve a https://console.anthropic.com/
- Crea una cuenta y genera una API key
- La key debe empezar con `sk-`

### 2. Ejecutar el Ciclo Inteligente
```bash
python robo_poet.py
# Selecciona opción: 3. 🧠 FASE 3: Ciclo Inteligente con Claude AI
```

### 3. Configurar el Ciclo
- **Nombre del modelo**: Identificador único para tu modelo
- **Dataset**: Selecciona de los datasets disponibles (Alicia, Shakespeare, etc.)
- **Número de ciclos**: Cuántas iteraciones de mejora (recomendado: 3-5)

## Cómo Funciona

### Fase 1: Entrenamiento Corto
- Entrena el modelo por 3 épocas
- Recopila métricas (loss, learning rate)

### Fase 2: Evaluación
- Genera texto de muestra
- Calcula perplejidad y métricas de calidad
- Evalúa coherencia del texto generado

### Fase 3: Consulta Inteligente
- Envía métricas y muestra de texto a Claude
- Claude analiza el rendimiento del modelo
- Recibe sugerencias específicas para mejorar el dataset

### Fase 4: Mejora del Dataset
- Aplica sugerencias de Claude al dataset:
  - ➕ **Agregar texto**: Nuevo contenido relevante
  - ✏️ **Modificar texto**: Limpieza y preprocesamiento
  - ⚖️ **Ajustar pesos**: Balance entre diferentes fuentes
  - 🗑️ **Remover texto**: Eliminar contenido problemático

## Configuración Avanzada

### Variables de Entorno (.env)
```bash
# Claude AI Configuration
CLAUDE_API_KEY=tu_api_key
CLAUDE_MODEL=claude-3-haiku-20240307
CLAUDE_MAX_TOKENS=2000
CLAUDE_TIMEOUT_SECONDS=30

# Intelligent Training
INTELLIGENT_TRAINING_MAX_CYCLES=5
INTELLIGENT_TRAINING_IMPROVEMENT_THRESHOLD=0.1
INTELLIGENT_TRAINING_MIN_CONFIDENCE=0.6
```

### Modelos de Claude Disponibles
- `claude-3-haiku-20240307` (rápido, económico) - **Recomendado**
- `claude-3-sonnet-20240229` (balance)
- `claude-3-opus-20240229` (más potente, más caro)

## Ventajas del Ciclo Inteligente

### 🎯 Optimización Automática
- **Sin intervención manual**: Claude decide qué mejorar
- **Basado en datos reales**: Análisis de métricas de entrenamiento
- **Mejora continua**: Cada ciclo construye sobre el anterior

### 📊 Análisis Inteligente
- **Detección de overfitting**: Identifica cuando val_loss > train_loss
- **Calidad del texto**: Evalúa coherencia y estilo literario
- **Convergencia**: Monitora la estabilidad del entrenamiento

### 🔄 Adaptación Dinámica
- **Dataset vivo**: Se mejora durante el entrenamiento
- **Respuesta a problemas**: Ajustes automáticos basados en rendimiento
- **Preservación automática**: Backups antes de cada modificación

## Resultados Esperados

### Métricas de Mejora
- **Reducción en validation loss**: 10-30% típicamente
- **Mejor coherencia**: Texto más fluido y consistente
- **Convergencia más rápida**: Menos épocas necesarias
- **Menor overfitting**: Mejor generalización

### Tipos de Mejoras que Claude Sugiere
1. **Más variedad textual**: Cuando detecta overfitting
2. **Limpieza de datos**: Cuando la perplejidad es muy alta
3. **Balance de géneros**: Cuando falta diversidad estilística
4. **Corrección de formato**: Cuando hay inconsistencias

## Troubleshooting

### Error: "Claude API key not found"
```bash
# Configura la key:
export CLAUDE_API_KEY=tu_key_aqui

# O agrega a .env:
echo "CLAUDE_API_KEY=tu_key_aqui" >> .env
```

### Error: "Connection failed"
- Verifica tu conexión a internet
- Confirma que la API key es válida
- Revisa que tienes créditos en tu cuenta Anthropic

### Error: "Low confidence suggestions"
- Es normal - Claude es conservador con sugerencias arriesgadas
- Las sugerencias con confidence < 0.6 se ignoran automáticamente
- Puedes bajar el threshold en configuración avanzada

## Costos Estimados

### Claude Haiku (Recomendado)
- **Por consulta**: ~$0.01-0.05 USD
- **Ciclo completo (5 iteraciones)**: ~$0.25 USD
- **Muy económico** para uso académico/experimental

### Optimización de Costos
- Usa `claude-3-haiku` para experimentos
- Limita ciclos a 3-5 iteraciones
- El sistema incluye fallbacks para cuando la API no está disponible

## Arquitectura Técnica

```
src/intelligence/
├── claude_integration.py      # Cliente API de Claude
├── __init__.py               # Exportaciones del módulo

src/interface/
├── phase3_intelligent_cycle.py   # Interfaz principal del ciclo

Configuración:
├── .env.example              # Variables de entorno
├── requirements_claude.txt   # Dependencias
├── setup_claude_integration.py   # Script de instalación
```

## Próximas Mejoras

- 🔍 **Análisis de atención**: Usar patterns de atención para mejoras
- 📈 **Visualización de mejoras**: Gráficos de progreso del ciclo
- 🎯 **Hiperparámetro tuning**: Claude sugiere ajustes de learning rate
- 🤖 **Multi-modelo**: Comparar diferentes arquitecturas
- 📚 **Dataset expansion**: Claude sugiere fuentes de datos adicionales

---

**¿Problemas o preguntas?** El sistema incluye fallbacks y funciona sin Claude AI si es necesario.