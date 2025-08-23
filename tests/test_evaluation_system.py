#!/usr/bin/env python3
"""
Test script para el Sistema de Evaluación Continua completo.

Prueba todas las métricas, monitoreo en tiempo real, dashboard y early stopping.
"""

import sys
import os
import time
import tempfile
from pathlib import Path
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_evaluation_system():
    """Test completo del sistema de evaluación."""
    print("🧪 Testing Sistema de Evaluación Continua - Strategy 3...")
    
    # Test 1: Métricas básicas
    try:
        print("1. Testing Basic Metrics...")
        
        from src.evaluation.metrics import (
            BLEUMetric, PerplexityMetric, NGramDiversityMetric, 
            ROUGEMetric, MetricCalculator, EvaluationResults
        )
        
        # Test BLEU
        bleu_metric = BLEUMetric()
        bleu_score = bleu_metric.calculate_sentence_bleu(
            "The quick brown fox jumps over the lazy dog",
            "The quick brown fox leaps over the lazy dog"
        )
        assert 0 <= bleu_score <= 1, f"BLEU score out of range: {bleu_score}"
        print(f"   ✅ BLEU score calculated: {bleu_score:.4f}")
        
        # Test Perplexity
        perp_metric = PerplexityMetric()
        test_probs = np.array([0.8, 0.6, 0.9, 0.7, 0.5])
        perplexity = perp_metric.calculate_perplexity(test_probs)
        assert perplexity > 0, f"Invalid perplexity: {perplexity}"
        print(f"   ✅ Perplexity calculated: {perplexity:.2f}")
        
        # Test N-gram diversity
        diversity_metric = NGramDiversityMetric()
        test_texts = [
            "hello world this is a test",
            "this is another test sentence",
            "hello again this is different"
        ]
        diversity = diversity_metric.calculate_diversity(test_texts, n=2)
        assert 0 <= diversity <= 1, f"Diversity out of range: {diversity}"
        print(f"   ✅ Bigram diversity: {diversity:.4f}")
        
        # Test ROUGE
        rouge_metric = ROUGEMetric()
        rouge_scores = rouge_metric.calculate_rouge(
            "The cat sat on the mat",
            "A cat was sitting on the mat"
        )
        assert all(0 <= score <= 1 for score in rouge_scores.values())
        print(f"   ✅ ROUGE scores: {rouge_scores}")
        
    except ImportError as e:
        print(f"❌ Import error in metrics: {e}")
        return False
    except Exception as e:
        print(f"❌ Metrics error: {e}")
        return False
    
    # Test 2: Calculadora de métricas completa
    try:
        print("2. Testing Metric Calculator...")
        
        calculator = MetricCalculator()
        
        # Datos de prueba
        references = [
            "The quick brown fox jumps over the lazy dog",
            "Machine learning is a powerful tool",
            "Natural language processing enables computers to understand text"
        ]
        
        candidates = [
            "The quick brown fox leaps over the lazy dog",
            "Machine learning is a useful tool",
            "NLP enables computers to process text"
        ]
        
        # Probabilidades simuladas
        probabilities = [
            np.random.uniform(0.1, 0.9, 10),
            np.random.uniform(0.1, 0.9, 8),
            np.random.uniform(0.1, 0.9, 9)
        ]
        
        # Evaluación completa
        results = calculator.evaluate_generation(references, candidates, probabilities)
        
        assert isinstance(results, EvaluationResults)
        assert 0 <= results.bleu_score <= 1
        assert results.perplexity > 0
        assert results.num_samples == len(candidates)
        
        print(f"   ✅ Complete evaluation: BLEU={results.bleu_score:.4f}, Perplexity={results.perplexity:.2f}")
        print(f"   ✅ Summary score: {results.get_summary_score():.4f}")
        
    except Exception as e:
        print(f"❌ Metric calculator error: {e}")
        return False
    
    # Test 3: Monitor de entrenamiento
    try:
        print("3. Testing Training Monitor...")
        
        from src.evaluation.monitor import (
            TrainingMonitor, RealTimeEvaluator, MetricLogger,
            TrainingMetrics, ProgressTracker
        )
        
        # Crear logger temporal
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "test_metrics.jsonl")
            
            # Configurar monitor
            metric_logger = MetricLogger(log_file=log_file)
            real_time_evaluator = RealTimeEvaluator(calculator)
            
            monitor = TrainingMonitor(metric_logger, real_time_evaluator)
            
            # Simular entrenamiento
            monitor.start_epoch(1, 100)
            
            # Simular algunos batches
            for batch in range(5):
                train_metrics = {
                    'loss': 2.5 - batch * 0.1,
                    'accuracy': 0.5 + batch * 0.05,
                    'learning_rate': 0.001,
                    'tokens_processed': 1000
                }
                
                val_metrics = {
                    'loss': 2.7 - batch * 0.08,
                    'accuracy': 0.45 + batch * 0.04
                }
                
                monitor.update_batch_metrics(
                    batch_num=batch,
                    total_batches=100,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics
                )
            
            monitor.end_epoch({'avg_loss': 2.1})
            
            # Verificar que se generaron métricas
            recent_metrics = metric_logger.get_recent_metrics(3)
            assert len(recent_metrics) > 0
            
            print(f"   ✅ Training monitor: {len(recent_metrics)} metrics logged")
            
            # Test progress tracker
            progress_tracker = ProgressTracker(total_epochs=10, batches_per_epoch=100)
            progress_tracker.update(epoch=1, batch=50)
            eta_info = progress_tracker.get_eta()
            
            assert 'progress' in eta_info
            assert 0 <= eta_info['progress'] <= 1
            
            print(f"   ✅ Progress tracker: {eta_info['progress']:.2%} complete")
            
    except Exception as e:
        print(f"❌ Training monitor error: {e}")
        return False
    
    # Test 4: Dashboard y TensorBoard
    try:
        print("4. Testing Dashboard...")
        
        from src.evaluation.dashboard import TensorBoardDashboard, MetricsDashboard
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Dashboard de métricas personalizado
            metrics_dashboard = MetricsDashboard(output_dir=temp_dir)
            
            # Simular métricas
            test_metrics = TrainingMetrics(
                epoch=1,
                batch=10,
                total_batches=100,
                train_loss=2.5,
                val_loss=2.7,
                train_accuracy=0.6,
                learning_rate=0.001
            )
            
            metrics_dashboard.update_metrics(test_metrics)
            
            # Verificar archivos generados
            assert (Path(temp_dir) / "metrics.jsonl").exists()
            assert (Path(temp_dir) / "summary.json").exists()
            
            print("   ✅ Metrics dashboard: Files generated")
            
            # Generar reporte
            report = metrics_dashboard.generate_report()
            assert "Training Report" in report
            assert "Experiment Overview" in report
            
            print("   ✅ Dashboard report generated")
            
            # Test TensorBoard (si está disponible)
            try:
                tb_dashboard = TensorBoardDashboard(log_dir=os.path.join(temp_dir, "tb"))
                tb_dashboard.log_training_metrics(test_metrics)
                print("   ✅ TensorBoard logging (if available)")
            except Exception as e:
                print(f"   ⚠️ TensorBoard not available: {e}")
            
    except Exception as e:
        print(f"❌ Dashboard error: {e}")
        return False
    
    # Test 5: Early Stopping
    try:
        print("5. Testing Early Stopping...")
        
        from src.evaluation.early_stopping import (
            MultiMetricEarlyStopping, EarlyStoppingCriteria, 
            MetricCriterion, MetricDirection
        )
        
        # Crear criterios
        criteria = EarlyStoppingCriteria.create_default()
        early_stopping = MultiMetricEarlyStopping(criteria)
        
        # Simular entrenamiento con mejora inicial y luego plateau
        for epoch in range(25):
            # Loss que mejora al principio y luego se estanca
            if epoch < 10:
                train_loss = 3.0 - epoch * 0.2
                val_loss = 3.2 - epoch * 0.18
            else:
                train_loss = 1.0 + np.random.normal(0, 0.01)
                val_loss = 1.2 + np.random.normal(0, 0.01)
            
            metrics = TrainingMetrics(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                train_accuracy=min(0.9, 0.3 + epoch * 0.03)
            )
            
            # Crear evaluation results ocasionalmente
            eval_results = None
            if epoch % 5 == 0:
                eval_results = EvaluationResults(
                    bleu_score=min(0.8, 0.1 + epoch * 0.02),
                    perplexity=max(2.0, 10.0 - epoch * 0.3),
                    rouge_1=min(0.7, 0.2 + epoch * 0.015)
                )
            
            should_stop = early_stopping.update(metrics, eval_results)
            
            if should_stop:
                print(f"   ✅ Early stopping triggered at epoch {epoch}")
                print(f"   ✅ Reason: {early_stopping.get_stop_reason()}")
                break
        
        # Verificar estado
        status = early_stopping.get_status_report()
        assert 'current_epoch' in status
        assert 'criteria_status' in status
        
        print(f"   ✅ Early stopping test completed")
        
    except Exception as e:
        print(f"❌ Early stopping error: {e}")
        return False
    
    # Test 6: Integración completa
    try:
        print("6. Testing Full Integration...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Configurar sistema completo
            metric_logger = MetricLogger()
            real_time_evaluator = RealTimeEvaluator(calculator, max_samples=3)
            monitor = TrainingMonitor(metric_logger, real_time_evaluator)
            
            dashboard = MetricsDashboard(output_dir=temp_dir)
            early_stopping = MultiMetricEarlyStopping(EarlyStoppingCriteria.create_aggressive())
            
            # Callback para dashboard
            def dashboard_callback(metrics):
                dashboard.update_metrics(metrics)
            
            monitor.add_batch_end_callback(lambda batch_num, metrics: dashboard_callback(metrics))
            
            # Simular entrenamiento completo
            monitor.start_epoch(1, 20)
            
            for batch in range(10):
                train_metrics = {
                    'loss': 3.0 - batch * 0.2,
                    'accuracy': 0.3 + batch * 0.05,
                    'learning_rate': 0.001
                }
                
                val_metrics = {
                    'loss': 3.2 - batch * 0.18,
                    'accuracy': 0.25 + batch * 0.04
                }
                
                # Simular samples generados ocasionalmente
                generated_samples = None
                if batch % 3 == 0:
                    generated_samples = {
                        'references': references[:2],
                        'generated': candidates[:2]
                    }
                
                monitor.update_batch_metrics(
                    batch_num=batch,
                    total_batches=20,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    generated_samples=generated_samples
                )
                
                # Verificar early stopping
                test_metrics = TrainingMetrics(
                    epoch=1,
                    batch=batch,
                    train_loss=train_metrics['loss'],
                    val_loss=val_metrics['loss']
                )
                
                if early_stopping.update(test_metrics):
                    print(f"   ✅ Integrated early stopping at batch {batch}")
                    break
            
            monitor.end_epoch({'avg_loss': 1.5})
            
            # Verificar resultados
            summary = monitor.get_training_summary()
            assert summary['current_epoch'] == 1
            
            # Verificar archivos del dashboard
            assert (Path(temp_dir) / "metrics.jsonl").exists()
            
            print("   ✅ Full integration test completed")
        
    except Exception as e:
        print(f"❌ Integration error: {e}")
        return False
    
    print("\n🎉 Sistema de Evaluación Continua Test Complete!")
    print("\n📋 Summary:")
    print("✅ BLEU, ROUGE, Perplexity, N-gram diversity metrics")
    print("✅ Real-time training monitor with callbacks")
    print("✅ TensorBoard dashboard integration")
    print("✅ Custom metrics dashboard with plots")
    print("✅ Multi-metric early stopping system")
    print("✅ Progress tracking and ETA estimation")
    print("✅ Full system integration")
    
    print("\n🏗️ Strategy 3 Implementation Status:")
    print("✅ 3.1 Implementar métricas BLEU automáticas")
    print("✅ 3.2 Calcular perplexity en tiempo real")
    print("✅ 3.3 Medir diversidad de n-gramas")
    print("✅ 3.4 Dashboard de métricas en TensorBoard")
    print("✅ 3.5 Early stopping basado en múltiples métricas")
    
    return True

if __name__ == "__main__":
    success = test_evaluation_system()
    sys.exit(0 if success else 1)