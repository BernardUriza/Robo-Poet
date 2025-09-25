#!/usr/bin/env python3
"""
Claude AI Integration for Real-time Dataset Enhancement

Integrates with Claude API to dynamically improve training datasets based on
model performance and training metrics.

Author: Bernard Uriza Orozco
Version: 1.0 - Intelligent Training Loop
"""

import os
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import requests
import time

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Training metrics for Claude analysis."""
    epoch: int
    train_loss: float
    val_loss: float
    perplexity: float
    learning_rate: float
    grad_norm: Optional[float] = None
    generated_sample: Optional[str] = None


@dataclass
class DatasetSuggestion:
    """Dataset enhancement suggestion from Claude."""
    action: str  # 'add_text', 'remove_text', 'modify_text', 'adjust_weights'
    content: Optional[str] = None
    reasoning: str = ""
    confidence: float = 0.0


class ClaudeIntegration:
    """Integration with Claude API for intelligent dataset enhancement."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Claude integration."""
        self.api_key = api_key or os.getenv('CLAUDE_API_KEY')
        if not self.api_key:
            raise ValueError("Claude API key is required. Set CLAUDE_API_KEY environment variable.")

        self.base_url = "https://api.anthropic.com/v1/messages"
        self.headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }

        self.conversation_history: List[Dict] = []

    def _create_training_analysis_prompt(self, metrics: TrainingMetrics,
                                       dataset_info: Dict) -> str:
        """Create prompt for Claude to analyze training progress."""

        prompt = f"""
Estás ayudando a optimizar un modelo GPT para generación de texto literario.

## Métricas de Entrenamiento Actual:
- Época: {metrics.epoch}
- Loss de Entrenamiento: {metrics.train_loss:.4f}
- Loss de Validación: {metrics.val_loss:.4f}
- Perplejidad: {metrics.perplexity:.2f}
- Learning Rate: {metrics.learning_rate:.6f}

## Información del Dataset:
- Archivos: {dataset_info.get('files', [])}
- Tamaño total: {dataset_info.get('total_size', 'N/A')} tokens
- Vocabulario: {dataset_info.get('vocab_size', 'N/A')} tokens únicos

## Muestra de Texto Generado:
```
{metrics.generated_sample or 'No disponible'}
```

## Análisis Requerido:
Basándote en estas métricas, sugiere mejoras específicas al dataset:

1. **Calidad del texto generado**: ¿Es coherente? ¿Mantiene el estilo literario?
2. **Convergencia**: ¿El loss está convergiendo adecuadamente?
3. **Overfitting**: ¿Hay signos de sobreajuste?
4. **Dataset balance**: ¿El dataset necesita más variedad o diferentes tipos de texto?

## Sugerencias específicas:
- ¿Agregar más texto de cierto tipo/autor?
- ¿Remover secciones problemáticas?
- ¿Ajustar pesos de diferentes fuentes?
- ¿Modificar la estrategia de entrenamiento?

Responde en JSON con el siguiente formato:
{{
    "analysis": "Tu análisis detallado aquí",
    "suggestions": [
        {{
            "action": "add_text|remove_text|modify_text|adjust_weights",
            "content": "contenido específico o descripción",
            "reasoning": "razón para esta sugerencia",
            "confidence": 0.0-1.0
        }}
    ],
    "next_steps": "Qué hacer en la siguiente iteración"
}}
"""
        return prompt

    def analyze_training_progress(self, metrics: TrainingMetrics,
                                dataset_info: Dict) -> Dict[str, Any]:
        """Analyze training progress and get dataset enhancement suggestions."""

        prompt = self._create_training_analysis_prompt(metrics, dataset_info)

        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json={
                    "model": "claude-3-haiku-20240307",  # Usando Haiku para velocidad
                    "max_tokens": 2000,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                },
                timeout=30
            )

            if response.status_code != 200:
                logger.error(f"Claude API error: {response.status_code} - {response.text}")
                return self._fallback_analysis(metrics)

            response_data = response.json()
            content = response_data['content'][0]['text']

            # Try to parse JSON response
            try:
                analysis = json.loads(content)
                logger.info("[OK] Análisis de Claude recibido exitosamente")
                return analysis
            except json.JSONDecodeError:
                logger.warning("WARNING: Respuesta de Claude no es JSON válido, usando contenido directo")
                return {
                    "analysis": content,
                    "suggestions": [],
                    "next_steps": "Continuar entrenamiento y monitorear"
                }

        except requests.RequestException as e:
            logger.error(f"Error connecting to Claude API: {e}")
            return self._fallback_analysis(metrics)

    def _fallback_analysis(self, metrics: TrainingMetrics) -> Dict[str, Any]:
        """Fallback analysis when Claude API is unavailable."""

        suggestions = []

        # Simple heuristic-based suggestions
        if metrics.val_loss > metrics.train_loss * 1.5:
            suggestions.append({
                "action": "add_text",
                "content": "Más variedad de texto para reducir overfitting",
                "reasoning": "Diferencia significativa entre train y validation loss",
                "confidence": 0.7
            })

        if metrics.perplexity > 100:
            suggestions.append({
                "action": "modify_text",
                "content": "Limpiar y preprocesar mejor el dataset",
                "reasoning": "Perplejidad muy alta indica texto inconsistente",
                "confidence": 0.8
            })

        return {
            "analysis": f"Análisis básico - Época {metrics.epoch}, Loss: {metrics.train_loss:.4f}",
            "suggestions": suggestions,
            "next_steps": "Continuar entrenamiento con monitoreo"
        }

    def suggest_new_training_text(self, topic: str, style: str = "literario") -> str:
        """Suggest new training text based on current needs."""

        prompt = f"""
Genera un fragmento de texto {style} sobre el tema '{topic}' que sea apropiado para entrenar un modelo de generación de texto.

Requisitos:
- Longitud: 200-400 palabras
- Estilo coherente con literatura clásica
- Vocabulario rico pero no excesivamente complejo
- Narrativa engaging que mantenga la atención

El texto debe ser original e inspirado en el estilo de autores como Lewis Carroll o Shakespeare, pero completamente nuevo.
"""

        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json={
                    "model": "claude-3-haiku-20240307",
                    "max_tokens": 1000,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                },
                timeout=30
            )

            if response.status_code == 200:
                response_data = response.json()
                return response_data['content'][0]['text']
            else:
                logger.error(f"Error generating text: {response.status_code}")
                return ""

        except requests.RequestException as e:
            logger.error(f"Error connecting to Claude API: {e}")
            return ""

    def validate_api_connection(self) -> bool:
        """Test Claude API connection."""
        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json={
                    "model": "claude-3-haiku-20240307",
                    "max_tokens": 100,
                    "messages": [
                        {
                            "role": "user",
                            "content": "Hello, can you respond with 'API connection successful'?"
                        }
                    ]
                },
                timeout=10
            )

            return response.status_code == 200

        except requests.RequestException:
            return False


class IntelligentDatasetManager:
    """Manages dataset modifications based on Claude suggestions."""

    def __init__(self, dataset_path: Path):
        """Initialize dataset manager."""
        self.dataset_path = Path(dataset_path)
        self.backup_path = self.dataset_path.parent / "dataset_backups"
        self.backup_path.mkdir(exist_ok=True)

    def apply_suggestions(self, suggestions: List[Dict]) -> bool:
        """Apply Claude's suggestions to the dataset."""

        # Create backup
        timestamp = int(time.time())
        backup_file = self.backup_path / f"dataset_backup_{timestamp}.txt"

        try:
            if self.dataset_path.exists():
                import shutil
                shutil.copy2(self.dataset_path, backup_file)
                logger.info(f" Backup creado: {backup_file}")

            # Apply each suggestion
            for suggestion in suggestions:
                action = suggestion.get('action', '')
                confidence = suggestion.get('confidence', 0.0)

                if confidence < 0.6:  # Skip low-confidence suggestions
                    logger.info(f"⏭  Saltando sugerencia de baja confianza: {action}")
                    continue

                if action == 'add_text' and suggestion.get('content'):
                    self._add_text_to_dataset(suggestion['content'])
                    logger.info(f" Texto agregado al dataset")

                elif action == 'modify_text':
                    # For now, just log - more complex modifications can be added
                    logger.info(f"  Modificación sugerida: {suggestion.get('reasoning', '')}")

            return True

        except Exception as e:
            logger.error(f"Error applying suggestions: {e}")
            return False

    def _add_text_to_dataset(self, new_text: str):
        """Add new text to the dataset."""
        try:
            with open(self.dataset_path, 'a', encoding='utf-8') as f:
                f.write(f"\n\n{new_text}\n")
        except Exception as e:
            logger.error(f"Error adding text to dataset: {e}")

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about current dataset."""
        try:
            if not self.dataset_path.exists():
                return {"error": "Dataset file not found"}

            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                content = f.read()

            words = content.split()
            unique_words = set(words)

            return {
                "file_path": str(self.dataset_path),
                "total_size": len(words),
                "unique_words": len(unique_words),
                "file_size_mb": self.dataset_path.stat().st_size / 1024 / 1024,
                "vocab_size": len(unique_words)
            }

        except Exception as e:
            logger.error(f"Error getting dataset info: {e}")
            return {"error": str(e)}


def test_claude_integration():
    """Test function for Claude integration."""

    # Test API key
    api_key = os.getenv('CLAUDE_API_KEY')
    if not api_key:
        print("[X] CLAUDE_API_KEY no encontrada en variables de entorno")
        return False

    try:
        claude = ClaudeIntegration(api_key)

        # Test connection
        print("[CYCLE] Probando conexión con Claude API...")
        if claude.validate_api_connection():
            print("[OK] Conexión con Claude API exitosa")
            return True
        else:
            print("[X] Error en conexión con Claude API")
            return False

    except Exception as e:
        print(f"[X] Error: {e}")
        return False


if __name__ == "__main__":
    test_claude_integration()