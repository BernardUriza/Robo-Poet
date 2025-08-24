#!/usr/bin/env python3
"""
Sharp vs Flat Minima Analyzer - Task 2.1
Creado por Bernard Orozco

Implementación del análisis de paisaje de pérdida basado en Li et al. 2018
"Visualizing the Loss Landscape of Neural Networks" para detectar mínimos 
agudos vs planos en modelos LSTM.

Teoría:
- Mínimos planos → mejor generalización (Hochreiter & Schmidhuber 1997)
- Mínimos agudos → overfitting, pobre generalización
- Sharpness = curvatura de la superficie de pérdida
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import os
from pathlib import Path

class LossLandscapeAnalyzer:
    """
    Analizador del paisaje de pérdida usando perturbaciones aleatorias
    y análisis de curvatura Hessiana simplificada.
    """
    
    def __init__(self, model: tf.keras.Model, test_data, config: Dict = None):
        """
        Inicializar analizador del paisaje de pérdida.
        
        Args:
            model: Modelo LSTM entrenado
            test_data: Dataset de prueba (X, y)
            config: Configuración de análisis
        """
        self.model = model
        self.test_data = test_data
        self.config = config or self._default_config()
        
        # Resultados del análisis
        self.results = {
            'sharpness_metrics': {},
            'perturbation_analysis': {},
            'curvature_analysis': {},
            'visualization_path': None
        }
        
    def _default_config(self) -> Dict:
        """Configuración por defecto del análisis."""
        return {
            'perturbation_scale': 0.01,      # Escala de perturbaciones
            'num_directions': 50,            # Direcciones aleatorias
            'num_samples': 100,              # Muestras para evaluación
            'hessian_samples': 20,           # Muestras para Hessiano
            'visualization_resolution': 51,  # Resolución de visualización
            'save_plots': True
        }
    
    def analyze_sharpness(self) -> Dict:
        """
        Análisis completo de sharpness del mínimo actual.
        
        Returns:
            Dict con métricas de sharpness y curvatura
        """
        print("🔬 Iniciando análisis de sharpness del paisaje de pérdida...")
        
        # 1. Análisis por perturbaciones aleatorias
        perturbation_metrics = self._perturbation_analysis()
        
        # 2. Estimación de curvatura Hessiana
        curvature_metrics = self._curvature_analysis()
        
        # 3. Análisis direccional
        directional_metrics = self._directional_analysis()
        
        # 4. Métricas consolidadas
        consolidated_metrics = self._consolidate_metrics(
            perturbation_metrics, curvature_metrics, directional_metrics
        )
        
        # 5. Generar visualización
        if self.config['save_plots']:
            viz_path = self._generate_visualization()
            consolidated_metrics['visualization_path'] = viz_path
        
        self.results = consolidated_metrics
        return consolidated_metrics
    
    def _perturbation_analysis(self) -> Dict:
        """
        Análisis de sharpness usando perturbaciones aleatorias.
        Implementa el método de Li et al. 2018.
        """
        print("   📊 Analizando perturbaciones aleatorias...")
        
        # Loss baseline sin perturbación
        baseline_loss = self._evaluate_loss()
        
        # Generar perturbaciones en múltiples escalas
        scales = [0.001, 0.005, 0.01, 0.05, 0.1]
        perturbation_results = {}
        
        for scale in scales:
            losses = []
            
            for i in range(self.config['num_directions']):
                # Generar perturbación aleatoria normalizada
                perturbation = self._generate_random_perturbation(scale)
                
                # Aplicar perturbación
                self._apply_perturbation(perturbation)
                
                # Evaluar loss
                perturbed_loss = self._evaluate_loss()
                losses.append(perturbed_loss)
                
                # Restaurar pesos originales
                self._restore_weights(perturbation)
            
            # Calcular métricas para esta escala
            mean_loss = np.mean(losses)
            std_loss = np.std(losses)
            max_loss = np.max(losses)
            
            perturbation_results[f'scale_{scale}'] = {
                'mean_loss': float(mean_loss),
                'std_loss': float(std_loss),
                'max_loss': float(max_loss),
                'loss_increase': float(mean_loss - baseline_loss),
                'sharpness': float((mean_loss - baseline_loss) / scale)
            }
        
        return {
            'baseline_loss': float(baseline_loss),
            'perturbation_results': perturbation_results,
            'overall_sharpness': float(np.mean([
                r['sharpness'] for r in perturbation_results.values()
            ]))
        }
    
    def _curvature_analysis(self) -> Dict:
        """
        Análisis de curvatura usando aproximación del Hessiano.
        """
        print("   📈 Calculando curvatura del Hessiano...")
        
        # Aproximación del Hessiano por diferencias finitas
        eigenvalues = self._estimate_hessian_eigenvalues()
        
        # Métricas de curvatura
        max_eigenvalue = np.max(eigenvalues)
        mean_eigenvalue = np.mean(eigenvalues)
        condition_number = np.max(eigenvalues) / (np.min(eigenvalues) + 1e-10)
        
        return {
            'max_eigenvalue': float(max_eigenvalue),
            'mean_eigenvalue': float(mean_eigenvalue),
            'condition_number': float(condition_number),
            'eigenvalue_spectrum': eigenvalues.tolist(),
            'curvature_sharpness': float(max_eigenvalue)
        }
    
    def _directional_analysis(self) -> Dict:
        """
        Análisis direccional del paisaje de pérdida.
        """
        print("   🧭 Analizando direcciones críticas...")
        
        # Direcciones basadas en gradientes
        gradient_direction = self._compute_gradient_direction()
        
        # Análisis en dirección del gradiente
        gradient_sharpness = self._analyze_direction(gradient_direction)
        
        # Direcciones aleatorias ortogonales
        random_directions = self._generate_orthogonal_directions(5)
        random_sharpness = []
        
        for direction in random_directions:
            sharpness = self._analyze_direction(direction)
            random_sharpness.append(sharpness)
        
        return {
            'gradient_direction_sharpness': float(gradient_sharpness),
            'random_directions_sharpness': {
                'mean': float(np.mean(random_sharpness)),
                'std': float(np.std(random_sharpness)),
                'individual': [float(s) for s in random_sharpness]
            }
        }
    
    def _generate_random_perturbation(self, scale: float) -> List[np.ndarray]:
        """Generar perturbación aleatoria normalizada."""
        perturbations = []
        
        for layer in self.model.trainable_variables:
            # Perturbación Gaussiana normalizada
            shape = layer.shape
            perturbation = np.random.normal(0, 1, shape)
            
            # Normalizar por la norma de Frobenius
            norm = np.linalg.norm(perturbation)
            if norm > 0:
                perturbation = perturbation / norm * scale
            
            perturbations.append(perturbation.astype(np.float32))
        
        return perturbations
    
    def _apply_perturbation(self, perturbations: List[np.ndarray]):
        """Aplicar perturbación a los pesos del modelo."""
        for var, perturbation in zip(self.model.trainable_variables, perturbations):
            var.assign_add(perturbation)
    
    def _restore_weights(self, perturbations: List[np.ndarray]):
        """Restaurar pesos originales."""
        for var, perturbation in zip(self.model.trainable_variables, perturbations):
            var.assign_sub(perturbation)
    
    def _evaluate_loss(self) -> float:
        """Evaluar loss en el dataset de prueba."""
        X_test, y_test = self.test_data
        
        # Tomar muestra aleatoria si el dataset es muy grande
        if len(X_test) > self.config['num_samples']:
            indices = np.random.choice(len(X_test), self.config['num_samples'], replace=False)
            X_sample = X_test[indices]
            y_sample = y_test[indices]
        else:
            X_sample, y_sample = X_test, y_test
        
        # Calcular loss
        predictions = self.model(X_sample, training=False)
        loss = tf.keras.losses.sparse_categorical_crossentropy(
            y_sample, predictions, from_logits=True
        )
        
        return float(tf.reduce_mean(loss))
    
    def _estimate_hessian_eigenvalues(self) -> np.ndarray:
        """
        Estimar eigenvalores del Hessiano usando diferencias finitas.
        """
        epsilon = 1e-4
        baseline_loss = self._evaluate_loss()
        eigenvalues = []
        
        # Aproximar eigenvalues para las capas principales
        for layer_idx, layer in enumerate(self.model.trainable_variables):
            if len(layer.shape) < 2:  # Saltar biases
                continue
                
            # Tomar muestra de direcciones
            num_directions = min(10, np.prod(layer.shape))
            
            for _ in range(num_directions):
                # Dirección aleatoria unitaria
                direction = np.random.normal(0, 1, layer.shape)
                direction = direction / np.linalg.norm(direction)
                
                # Calcular segunda derivada
                # f(x + h) + f(x - h) - 2f(x) ≈ h² * f''(x)
                
                # Perturbación positiva
                layer.assign_add(epsilon * direction)
                loss_plus = self._evaluate_loss()
                layer.assign_sub(epsilon * direction)
                
                # Perturbación negativa  
                layer.assign_sub(epsilon * direction)
                loss_minus = self._evaluate_loss()
                layer.assign_add(epsilon * direction)
                
                # Segunda derivada
                second_derivative = (loss_plus + loss_minus - 2 * baseline_loss) / (epsilon ** 2)
                eigenvalues.append(second_derivative)
        
        return np.array(eigenvalues)
    
    def _compute_gradient_direction(self) -> List[np.ndarray]:
        """Computar dirección del gradiente."""
        X_test, y_test = self.test_data
        
        # Tomar muestra pequeña
        indices = np.random.choice(len(X_test), min(50, len(X_test)), replace=False)
        X_sample = X_test[indices]
        y_sample = y_test[indices]
        
        with tf.GradientTape() as tape:
            predictions = self.model(X_sample, training=False)
            loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    y_sample, predictions, from_logits=True
                )
            )
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        
        # Normalizar gradientes
        normalized_gradients = []
        for grad in gradients:
            if grad is not None:
                norm = tf.norm(grad)
                normalized_grad = grad / (norm + 1e-10)
                normalized_gradients.append(normalized_grad.numpy())
            else:
                normalized_gradients.append(np.zeros_like(var.numpy()))
        
        return normalized_gradients
    
    def _generate_orthogonal_directions(self, num_directions: int) -> List[List[np.ndarray]]:
        """Generar direcciones aleatorias ortogonales."""
        directions = []
        
        for _ in range(num_directions):
            direction = []
            for layer in self.model.trainable_variables:
                random_dir = np.random.normal(0, 1, layer.shape)
                norm = np.linalg.norm(random_dir)
                if norm > 0:
                    random_dir = random_dir / norm
                direction.append(random_dir.astype(np.float32))
            directions.append(direction)
        
        return directions
    
    def _analyze_direction(self, direction: List[np.ndarray]) -> float:
        """Analizar sharpness en una dirección específica."""
        baseline_loss = self._evaluate_loss()
        
        # Probar diferentes escalas
        scales = [0.001, 0.01, 0.1]
        sharpness_values = []
        
        for scale in scales:
            # Aplicar perturbación
            for var, dir_component in zip(self.model.trainable_variables, direction):
                var.assign_add(scale * dir_component)
            
            # Evaluar
            perturbed_loss = self._evaluate_loss()
            sharpness = (perturbed_loss - baseline_loss) / scale
            sharpness_values.append(sharpness)
            
            # Restaurar
            for var, dir_component in zip(self.model.trainable_variables, direction):
                var.assign_sub(scale * dir_component)
        
        return np.mean(sharpness_values)
    
    def _consolidate_metrics(self, perturbation_metrics: Dict, 
                           curvature_metrics: Dict, directional_metrics: Dict) -> Dict:
        """Consolidar todas las métricas de sharpness."""
        
        # Clasificación de sharpness
        overall_sharpness = perturbation_metrics['overall_sharpness']
        
        if overall_sharpness < 0.1:
            sharpness_category = "FLAT_MINIMUM"
            interpretation = "Mínimo plano - Excelente generalización esperada"
        elif overall_sharpness < 0.5:
            sharpness_category = "MODERATE_SHARPNESS" 
            interpretation = "Sharpness moderado - Generalización aceptable"
        elif overall_sharpness < 2.0:
            sharpness_category = "SHARP_MINIMUM"
            interpretation = "Mínimo agudo - Riesgo de overfitting"
        else:
            sharpness_category = "VERY_SHARP"
            interpretation = "Mínimo muy agudo - Overfitting probable"
        
        return {
            'analysis_metadata': {
                'timestamp': datetime.now().isoformat(),
                'model_summary': {
                    'total_params': self.model.count_params(),
                    'trainable_params': sum(tf.size(var) for var in self.model.trainable_variables)
                }
            },
            'sharpness_classification': {
                'category': sharpness_category,
                'overall_sharpness': float(overall_sharpness),
                'interpretation': interpretation
            },
            'perturbation_analysis': perturbation_metrics,
            'curvature_analysis': curvature_metrics,
            'directional_analysis': directional_metrics,
            'recommendations': self._generate_recommendations(overall_sharpness, curvature_metrics)
        }
    
    def _generate_recommendations(self, sharpness: float, curvature: Dict) -> List[str]:
        """Generar recomendaciones basadas en el análisis."""
        recommendations = []
        
        if sharpness > 1.0:
            recommendations.append("CRÍTICO: Aumentar regularización (dropout, weight decay)")
            recommendations.append("Considerar early stopping más agresivo")
            recommendations.append("Reducir learning rate para entrenamiento más suave")
        
        if curvature['condition_number'] > 100:
            recommendations.append("Alto condition number - considerar batch normalization")
            recommendations.append("Gradient clipping recomendado")
        
        if sharpness < 0.1:
            recommendations.append("Excelente: Mínimo plano detectado - modelo bien regularizado")
        
        return recommendations
    
    def _generate_visualization(self) -> str:
        """Generar visualización del paisaje de pérdida."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"loss_landscape_{timestamp}.png"
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Sharpness por escala
        perturbation_data = self.results['perturbation_analysis']['perturbation_results']
        scales = [float(k.split('_')[1]) for k in perturbation_data.keys()]
        sharpness_values = [perturbation_data[k]['sharpness'] for k in perturbation_data.keys()]
        
        ax1.loglog(scales, sharpness_values, 'o-', linewidth=2, markersize=8)
        ax1.set_xlabel('Perturbation Scale')
        ax1.set_ylabel('Sharpness')
        ax1.set_title('Sharpness vs Perturbation Scale')
        ax1.grid(True, alpha=0.3)
        
        # 2. Distribución de eigenvalues
        eigenvalues = self.results['curvature_analysis']['eigenvalue_spectrum']
        ax2.hist(eigenvalues, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_xlabel('Eigenvalue')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Hessian Eigenvalue Distribution')
        ax2.axvline(np.mean(eigenvalues), color='red', linestyle='--', label=f'Mean: {np.mean(eigenvalues):.4f}')
        ax2.legend()
        
        # 3. Comparación direccional
        gradient_sharpness = self.results['directional_analysis']['gradient_direction_sharpness']
        random_sharpness = self.results['directional_analysis']['random_directions_sharpness']['individual']
        
        ax3.bar(['Gradient Direction'], [gradient_sharpness], color='red', alpha=0.7, label='Gradient')
        ax3.bar([f'Random {i+1}' for i in range(len(random_sharpness))], 
                random_sharpness, color='blue', alpha=0.7, label='Random')
        ax3.set_ylabel('Sharpness')
        ax3.set_title('Directional Sharpness Comparison')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Loss increase por escala
        loss_increases = [perturbation_data[k]['loss_increase'] for k in perturbation_data.keys()]
        ax4.semilogx(scales, loss_increases, 's-', linewidth=2, markersize=8, color='green')
        ax4.set_xlabel('Perturbation Scale')
        ax4.set_ylabel('Average Loss Increase')
        ax4.set_title('Loss Sensitivity to Perturbations')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📊 Visualización guardada: {filename}")
        return filename
    
    def save_results(self, filepath: str = None) -> str:
        """Guardar resultados del análisis."""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"minima_analysis_{timestamp}.json"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"   💾 Análisis guardado: {filepath}")
        return filepath


def analyze_model_sharpness(model_path: str, test_data_path: str = None, 
                          config: Dict = None) -> Dict:
    """
    Función principal para analizar sharpness de un modelo.
    
    Args:
        model_path: Ruta al modelo entrenado
        test_data_path: Ruta a datos de prueba (opcional)
        config: Configuración de análisis
    
    Returns:
        Dict con resultados del análisis
    """
    print("🔬 INICIANDO ANÁLISIS DE SHARP VS FLAT MINIMA")
    print("=" * 60)
    
    # Cargar modelo
    print(f"📂 Cargando modelo: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    # Generar datos de prueba sintéticos si no se proporcionan
    if test_data_path is None:
        print("⚠️ Generando datos de prueba sintéticos...")
        vocab_size = model.layers[-1].output_shape[-1]
        sequence_length = model.input_shape[1] if model.input_shape[1] else 100
        
        X_test = np.random.randint(0, vocab_size, (100, sequence_length))
        y_test = np.random.randint(0, vocab_size, (100, sequence_length))
        test_data = (X_test, y_test)
    else:
        # Cargar datos reales
        print(f"📂 Cargando datos de prueba: {test_data_path}")
        # Implementar carga según formato específico
        pass
    
    # Inicializar analizador
    analyzer = LossLandscapeAnalyzer(model, test_data, config)
    
    # Realizar análisis
    results = analyzer.analyze_sharpness()
    
    # Guardar resultados
    results_path = analyzer.save_results()
    
    print("=" * 60)
    print("✅ ANÁLISIS COMPLETADO")
    print(f"📊 Clasificación: {results['sharpness_classification']['category']}")
    print(f"📈 Sharpness general: {results['sharpness_classification']['overall_sharpness']:.4f}")
    print(f"💡 {results['sharpness_classification']['interpretation']}")
    
    return results


if __name__ == "__main__":
    # Ejemplo de uso
    import sys
    
    if len(sys.argv) < 2:
        print("Uso: python minima_analyzer.py <model_path>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    results = analyze_model_sharpness(model_path)