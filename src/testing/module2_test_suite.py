#!/usr/bin/env python3
"""
Módulo 2 - Test Suite Completo
Creado por Bernard Orozco

Suite de pruebas integral para validar todas las funcionalidades del Módulo 2:
- Entrenamiento rápido de modelo de prueba
- Análisis de gradientes completo  
- Análisis de paisaje de pérdida
- Experimentos de ablación
- Cirugía de emergencia
- Generación de reportes consolidados

Propósito: Demostración académica y validación automatizada del sistema.
"""

import os
import sys
import json
import time
import tempfile
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging
import subprocess

# Setup logging
def setup_test_logging() -> logging.Logger:
    """Configurar logging específico para tests."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"module2_test_suite_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"=== INICIANDO TEST SUITE MÓDULO 2 ===")
    logger.info(f"Log guardado en: {log_filename}")
    
    return logger, log_filename

class Module2TestSuite:
    """Suite completa de tests para el Módulo 2."""
    
    def __init__(self):
        """Inicializar suite de tests."""
        self.logger, self.log_filename = setup_test_logging()
        self.test_results = {}
        self.temp_model_path = None
        self.test_data_path = None
        self.start_time = time.time()
        
        # Configuración de test
        self.test_config = {
            'epochs': 3,
            'vocab_size': 1000,
            'sequence_length': 50,
            'batch_size': 16,
            'samples': 500
        }
        
        self.logger.info("Suite de tests inicializada")
        self.logger.info(f"Configuración: {self.test_config}")
    
    def run_full_test_suite(self, test_selection: List[str] = None) -> Dict:
        """
        Ejecutar suite completa de tests.
        
        Args:
            test_selection: Lista de tests a ejecutar, None para todos
        
        Returns:
            Dict con resultados de todos los tests
        """
        available_tests = [
            'training',
            'gradient_analysis', 
            'minima_analysis',
            'ablation_experiments',
            'emergency_surgery',
            'report_generation'
        ]
        
        if test_selection is None:
            test_selection = available_tests
            
        self.logger.info(f"Ejecutando tests: {test_selection}")
        
        try:
            # Test 1: Entrenamiento rápido
            if 'training' in test_selection:
                self._test_quick_training()
            
            # Test 2: Análisis de gradientes  
            if 'gradient_analysis' in test_selection:
                self._test_gradient_analysis()
            
            # Test 3: Análisis de minima
            if 'minima_analysis' in test_selection:
                self._test_minima_analysis()
                
            # Test 4: Experimentos de ablación
            if 'ablation_experiments' in test_selection:
                self._test_ablation_experiments()
            
            # Test 5: Cirugía de emergencia  
            if 'emergency_surgery' in test_selection:
                self._test_emergency_surgery()
            
            # Test 6: Generación de reportes
            if 'report_generation' in test_selection:
                self._test_report_generation()
            
            # Generar reporte final
            final_report = self._generate_final_report()
            
            self.logger.info("=== SUITE DE TESTS COMPLETADA ===")
            return final_report
            
        except Exception as e:
            self.logger.error(f"Error crítico en suite de tests: {e}")
            return {'error': str(e), 'partial_results': self.test_results}
        finally:
            self._cleanup()
    
    def _test_quick_training(self) -> Dict:
        """Test 1: Entrenar modelo rápido para pruebas."""
        self.logger.info("\n" + "="*60)
        self.logger.info("TEST 1: ENTRENAMIENTO RÁPIDO DE MODELO")
        self.logger.info("="*60)
        
        test_start = time.time()
        
        try:
            # Crear datos sintéticos de prueba
            self._create_test_data()
            
            # Entrenar modelo usando el orchestrator
            model_name = f"test_model_{datetime.now().strftime('%H%M%S')}"
            
            # Usar subprocess para ejecutar entrenamiento
            cmd = [
                sys.executable, "robo_poet.py",
                "--text", self.test_data_path,
                "--epochs", str(self.test_config['epochs']),
                "--model", model_name
            ]
            
            self.logger.info(f"Ejecutando comando: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=300  # 5 minutos máximo
            )
            
            if result.returncode == 0:
                # Buscar el modelo generado
                models_dir = Path("models")
                model_files = list(models_dir.glob(f"*{model_name}*.keras"))
                
                if model_files:
                    self.temp_model_path = str(model_files[0])
                    self.logger.info(f"✅ Modelo entrenado exitosamente: {self.temp_model_path}")
                    
                    # Verificar que el modelo es válido
                    import tensorflow as tf
                    model = tf.keras.models.load_model(self.temp_model_path)
                    param_count = model.count_params()
                    
                    test_result = {
                        'status': 'SUCCESS',
                        'model_path': self.temp_model_path,
                        'parameters': param_count,
                        'epochs_trained': self.test_config['epochs'],
                        'training_time': time.time() - test_start,
                        'stdout': result.stdout[-500:],  # Últimas líneas
                    }
                    
                else:
                    raise Exception("Modelo no encontrado después del entrenamiento")
            else:
                raise Exception(f"Entrenamiento falló: {result.stderr}")
                
        except Exception as e:
            self.logger.error(f"❌ Test de entrenamiento falló: {e}")
            test_result = {
                'status': 'FAILED',
                'error': str(e),
                'training_time': time.time() - test_start
            }
        
        self.test_results['training'] = test_result
        self.logger.info(f"Test 1 completado en {time.time() - test_start:.2f}s")
        return test_result
    
    def _test_gradient_analysis(self) -> Dict:
        """Test 2: Análisis de gradientes."""
        self.logger.info("\n" + "="*60)
        self.logger.info("TEST 2: ANÁLISIS DE GRADIENTES")
        self.logger.info("="*60)
        
        test_start = time.time()
        
        try:
            if not self.temp_model_path:
                raise Exception("Modelo no disponible para análisis")
            
            # Ejecutar análisis de gradientes
            from analysis.gradient_analyzer_lite import GradientAnalyzerLite
            
            analyzer = GradientAnalyzerLite(self.temp_model_path)
            results = analyzer.run_complete_analysis(num_batches=10)
            
            if results:
                collapse_info = results.get('collapse_analysis', {})
                pascanu_info = results.get('pascanu_analysis', {})
                
                test_result = {
                    'status': 'SUCCESS',
                    'analysis_duration': results['analysis_metadata']['duration_minutes'],
                    'batches_analyzed': results['analysis_metadata']['batches_analyzed'],
                    'has_collapse': len(collapse_info.get('layers_with_collapse', [])) > 0,
                    'has_vanishing': pascanu_info.get('has_vanishing', False),
                    'has_exploding': pascanu_info.get('has_exploding', False),
                    'visualization_created': bool(results.get('visualization_path')),
                    'total_time': time.time() - test_start
                }
                
                self.logger.info(f"✅ Análisis de gradientes completado")
                self.logger.info(f"   Vanishing: {test_result['has_vanishing']}")
                self.logger.info(f"   Exploding: {test_result['has_exploding']}")
                
            else:
                raise Exception("Análisis de gradientes retornó resultados vacíos")
                
        except Exception as e:
            self.logger.error(f"❌ Test de análisis de gradientes falló: {e}")
            test_result = {
                'status': 'FAILED',
                'error': str(e),
                'total_time': time.time() - test_start
            }
        
        self.test_results['gradient_analysis'] = test_result
        self.logger.info(f"Test 2 completado en {time.time() - test_start:.2f}s")
        return test_result
    
    def _test_minima_analysis(self) -> Dict:
        """Test 3: Análisis de paisaje de pérdida."""
        self.logger.info("\n" + "="*60)
        self.logger.info("TEST 3: ANÁLISIS DE PAISAJE DE PÉRDIDA")
        self.logger.info("="*60)
        
        test_start = time.time()
        
        try:
            if not self.temp_model_path:
                raise Exception("Modelo no disponible para análisis")
            
            # Configuración rápida para test
            config = {
                'num_directions': 10,
                'num_samples': 50,
                'hessian_samples': 5,
                'save_plots': True
            }
            
            from analysis.minima_analyzer import analyze_model_sharpness
            
            self.logger.info("Ejecutando análisis de sharpness...")
            results = analyze_model_sharpness(self.temp_model_path, config=config)
            
            if results:
                classification = results.get('sharpness_classification', {})
                
                test_result = {
                    'status': 'SUCCESS',
                    'sharpness_category': classification.get('category'),
                    'overall_sharpness': classification.get('overall_sharpness'),
                    'interpretation': classification.get('interpretation'),
                    'visualization_created': bool(results.get('visualization_path')),
                    'recommendations_count': len(results.get('recommendations', [])),
                    'total_time': time.time() - test_start
                }
                
                self.logger.info(f"✅ Análisis de minima completado")
                self.logger.info(f"   Categoría: {test_result['sharpness_category']}")
                self.logger.info(f"   Sharpness: {test_result['overall_sharpness']:.4f}")
                
            else:
                raise Exception("Análisis de minima retornó resultados vacíos")
                
        except Exception as e:
            self.logger.error(f"❌ Test de análisis de minima falló: {e}")
            test_result = {
                'status': 'FAILED',
                'error': str(e),
                'total_time': time.time() - test_start
            }
        
        self.test_results['minima_analysis'] = test_result
        self.logger.info(f"Test 3 completado en {time.time() - test_start:.2f}s")
        return test_result
    
    def _test_ablation_experiments(self) -> Dict:
        """Test 4: Experimentos de ablación."""
        self.logger.info("\n" + "="*60)
        self.logger.info("TEST 4: EXPERIMENTOS DE ABLACIÓN")
        self.logger.info("="*60)
        
        test_start = time.time()
        
        try:
            if not self.temp_model_path:
                raise Exception("Modelo no disponible para análisis")
            
            from analysis.ablation_analyzer import AblationExperimentRunner
            
            runner = AblationExperimentRunner(self.temp_model_path)
            
            # Experimentos reducidos para test rápido
            results = runner.run_ablation_study(
                experiment_types=['lstm_units'],  # Solo un tipo para rapidez
                epochs=2,
                quick_mode=True
            )
            
            if results and not results.get('error'):
                comparative = results.get('comparative_analysis', {})
                best_overall = comparative.get('best_overall', {})
                
                test_result = {
                    'status': 'SUCCESS',
                    'experiments_completed': len([k for k, v in results.items() 
                                                if isinstance(v, dict) and any(
                                                    variant.get('success', False) 
                                                    for variant in v.values()
                                                    if isinstance(variant, dict)
                                                )]),
                    'best_configuration': best_overall.get('by_perplexity', {}).get('variant'),
                    'best_perplexity': best_overall.get('by_perplexity', {}).get('metrics', {}).get('perplexity'),
                    'visualization_created': bool(results.get('visualization_path')),
                    'total_time': time.time() - test_start
                }
                
                self.logger.info(f"✅ Experimentos de ablación completados")
                self.logger.info(f"   Mejor configuración: {test_result['best_configuration']}")
                
            else:
                error = results.get('error', 'Resultados vacíos')
                raise Exception(f"Experimentos de ablación fallaron: {error}")
                
        except Exception as e:
            self.logger.error(f"❌ Test de ablación falló: {e}")
            test_result = {
                'status': 'FAILED',
                'error': str(e),
                'total_time': time.time() - test_start
            }
        
        self.test_results['ablation_experiments'] = test_result
        self.logger.info(f"Test 4 completado en {time.time() - test_start:.2f}s")
        return test_result
    
    def _test_emergency_surgery(self) -> Dict:
        """Test 5: Cirugía de emergencia."""
        self.logger.info("\n" + "="*60)
        self.logger.info("TEST 5: CIRUGÍA DE EMERGENCIA")
        self.logger.info("="*60)
        
        test_start = time.time()
        
        try:
            if not self.temp_model_path:
                raise Exception("Modelo no disponible para cirugía")
            
            from hospital.emergency_gate_surgery import quick_surgery
            
            self.logger.info("Ejecutando cirugía de gates...")
            operated_model_path, surgery_report = quick_surgery(self.temp_model_path)
            
            if operated_model_path and surgery_report:
                test_result = {
                    'status': 'SUCCESS',
                    'operated_model_path': operated_model_path,
                    'original_gates_input': surgery_report.get('pre_surgery', {}).get('input_gate_mean'),
                    'operated_gates_input': surgery_report.get('post_surgery', {}).get('input_gate_mean'),
                    'improvement_factor': surgery_report.get('improvement_analysis', {}).get('input_gate_improvement'),
                    'surgery_successful': surgery_report.get('surgery_assessment', {}).get('overall_success'),
                    'total_time': time.time() - test_start
                }
                
                self.logger.info(f"✅ Cirugía de emergencia completada")
                self.logger.info(f"   Modelo operado: {operated_model_path}")
                self.logger.info(f"   Éxito: {test_result['surgery_successful']}")
                
                # Actualizar modelo para siguientes tests si la cirugía fue exitosa
                if test_result['surgery_successful']:
                    self.temp_model_path = operated_model_path
                
            else:
                raise Exception("Cirugía no retornó modelo operado válido")
                
        except Exception as e:
            self.logger.error(f"❌ Test de cirugía falló: {e}")
            test_result = {
                'status': 'FAILED',
                'error': str(e),
                'total_time': time.time() - test_start
            }
        
        self.test_results['emergency_surgery'] = test_result
        self.logger.info(f"Test 5 completado en {time.time() - test_start:.2f}s")
        return test_result
    
    def _test_report_generation(self) -> Dict:
        """Test 6: Generación de reportes."""
        self.logger.info("\n" + "="*60)
        self.logger.info("TEST 6: GENERACIÓN DE REPORTES")
        self.logger.info("="*60)
        
        test_start = time.time()
        
        try:
            # Generar reporte consolidado de todos los tests
            report_data = {
                'test_suite_metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'total_execution_time': time.time() - self.start_time,
                    'model_used': self.temp_model_path,
                    'test_config': self.test_config
                },
                'individual_test_results': self.test_results,
                'summary_statistics': self._calculate_summary_stats()
            }
            
            # Guardar reporte en JSON
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"module2_test_report_{timestamp}.json"
            
            with open(report_filename, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            # Generar también reporte de texto
            text_report = self._generate_text_report(report_data)
            text_filename = f"module2_test_report_{timestamp}.txt"
            
            with open(text_filename, 'w', encoding='utf-8') as f:
                f.write(text_report)
            
            test_result = {
                'status': 'SUCCESS',
                'json_report': report_filename,
                'text_report': text_filename,
                'report_size_bytes': os.path.getsize(report_filename),
                'total_time': time.time() - test_start
            }
            
            self.logger.info(f"✅ Reportes generados")
            self.logger.info(f"   JSON: {report_filename}")
            self.logger.info(f"   Text: {text_filename}")
            
        except Exception as e:
            self.logger.error(f"❌ Test de reportes falló: {e}")
            test_result = {
                'status': 'FAILED',
                'error': str(e),
                'total_time': time.time() - test_start
            }
        
        self.test_results['report_generation'] = test_result
        self.logger.info(f"Test 6 completado en {time.time() - test_start:.2f}s")
        return test_result
    
    def _create_test_data(self) -> str:
        """Crear archivo de datos sintéticos para entrenamiento."""
        # Generar texto sintético simple pero coherente
        words = [
            "the", "power", "of", "artificial", "intelligence", "machine", "learning",
            "neural", "networks", "deep", "models", "training", "data", "analysis",
            "algorithms", "optimization", "gradient", "descent", "backpropagation",
            "layers", "neurons", "activation", "functions", "loss", "accuracy"
        ]
        
        # Generar párrafos coherentes
        paragraphs = []
        for _ in range(50):  # 50 párrafos
            paragraph_words = np.random.choice(words, size=20, replace=True)
            paragraph = " ".join(paragraph_words) + "."
            paragraphs.append(paragraph)
        
        text_content = "\n\n".join(paragraphs)
        
        # Guardar en archivo temporal
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.test_data_path = f"test_data_{timestamp}.txt"
        
        with open(self.test_data_path, 'w', encoding='utf-8') as f:
            f.write(text_content)
        
        self.logger.info(f"Datos de test creados: {self.test_data_path}")
        return self.test_data_path
    
    def _calculate_summary_stats(self) -> Dict:
        """Calcular estadísticas resumen de los tests."""
        stats = {
            'total_tests': len(self.test_results),
            'successful_tests': 0,
            'failed_tests': 0,
            'total_execution_time': time.time() - self.start_time,
            'average_test_time': 0
        }
        
        test_times = []
        
        for test_name, result in self.test_results.items():
            if result.get('status') == 'SUCCESS':
                stats['successful_tests'] += 1
            else:
                stats['failed_tests'] += 1
                
            if 'total_time' in result:
                test_times.append(result['total_time'])
        
        if test_times:
            stats['average_test_time'] = np.mean(test_times)
        
        stats['success_rate'] = stats['successful_tests'] / stats['total_tests'] if stats['total_tests'] > 0 else 0
        
        return stats
    
    def _generate_text_report(self, report_data: Dict) -> str:
        """Generar reporte en formato texto."""
        report = []
        report.append("=" * 80)
        report.append("REPORTE DE TESTS - MÓDULO 2")
        report.append("Creado por Bernard Orozco")
        report.append("=" * 80)
        report.append("")
        
        # Metadata
        metadata = report_data['test_suite_metadata']
        report.append(f"Timestamp: {metadata['timestamp']}")
        report.append(f"Tiempo total: {metadata['total_execution_time']:.2f} segundos")
        report.append(f"Modelo usado: {metadata['model_used']}")
        report.append("")
        
        # Estadísticas resumen
        stats = report_data['summary_statistics']
        report.append("RESUMEN DE RESULTADOS:")
        report.append("-" * 40)
        report.append(f"Tests totales: {stats['total_tests']}")
        report.append(f"Tests exitosos: {stats['successful_tests']}")
        report.append(f"Tests fallidos: {stats['failed_tests']}")
        report.append(f"Tasa de éxito: {stats['success_rate']:.1%}")
        report.append(f"Tiempo promedio por test: {stats['average_test_time']:.2f}s")
        report.append("")
        
        # Resultados individuales
        report.append("RESULTADOS DETALLADOS:")
        report.append("-" * 40)
        
        for test_name, result in report_data['individual_test_results'].items():
            status_emoji = "✅" if result.get('status') == 'SUCCESS' else "❌"
            report.append(f"{status_emoji} {test_name.upper().replace('_', ' ')}")
            
            if result.get('status') == 'SUCCESS':
                if test_name == 'training':
                    report.append(f"    Modelo: {result.get('model_path', 'N/A')}")
                    report.append(f"    Parámetros: {result.get('parameters', 'N/A'):,}")
                elif test_name == 'gradient_analysis':
                    report.append(f"    Vanishing: {result.get('has_vanishing', 'N/A')}")
                    report.append(f"    Exploding: {result.get('has_exploding', 'N/A')}")
                elif test_name == 'minima_analysis':
                    report.append(f"    Categoría: {result.get('sharpness_category', 'N/A')}")
                    report.append(f"    Sharpness: {result.get('overall_sharpness', 'N/A')}")
                elif test_name == 'emergency_surgery':
                    report.append(f"    Éxito: {result.get('surgery_successful', 'N/A')}")
            else:
                report.append(f"    Error: {result.get('error', 'N/A')}")
            
            report.append(f"    Tiempo: {result.get('total_time', 0):.2f}s")
            report.append("")
        
        report.append("=" * 80)
        report.append("Fin del reporte")
        
        return "\n".join(report)
    
    def _generate_final_report(self) -> Dict:
        """Generar reporte final consolidado."""
        return {
            'test_suite_metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_execution_time': time.time() - self.start_time,
                'model_used': self.temp_model_path,
                'test_config': self.test_config,
                'log_file': self.log_filename
            },
            'individual_test_results': self.test_results,
            'summary_statistics': self._calculate_summary_stats(),
            'success': all(result.get('status') == 'SUCCESS' for result in self.test_results.values())
        }
    
    def _cleanup(self):
        """Limpiar archivos temporales."""
        try:
            # Limpiar datos de test
            if self.test_data_path and os.path.exists(self.test_data_path):
                os.remove(self.test_data_path)
                self.logger.info(f"Archivo de datos de test limpiado: {self.test_data_path}")
        except Exception as e:
            self.logger.warning(f"Error limpiando archivos temporales: {e}")


def run_selected_tests(test_selection: List[str] = None) -> Dict:
    """
    Función principal para ejecutar tests seleccionados.
    
    Args:
        test_selection: Lista de tests a ejecutar
    
    Returns:
        Dict con resultados de los tests
    """
    suite = Module2TestSuite()
    return suite.run_full_test_suite(test_selection)


def run_quick_demo() -> Dict:
    """Ejecutar demo rápido con tests básicos."""
    return run_selected_tests(['training', 'gradient_analysis', 'emergency_surgery'])


def run_full_validation() -> Dict:
    """Ejecutar validación completa de todos los componentes."""
    return run_selected_tests()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Suite de Tests Módulo 2")
    parser.add_argument('--tests', nargs='+', 
                       choices=['training', 'gradient_analysis', 'minima_analysis', 
                               'ablation_experiments', 'emergency_surgery', 'report_generation'],
                       help='Tests específicos a ejecutar')
    parser.add_argument('--quick', action='store_true', help='Ejecutar demo rápido')
    
    args = parser.parse_args()
    
    if args.quick:
        print("🚀 Ejecutando demo rápido...")
        results = run_quick_demo()
    else:
        print("🧪 Ejecutando suite de tests...")
        results = run_selected_tests(args.tests)
    
    print(f"\n{'✅' if results.get('success') else '❌'} Tests completados")
    if results.get('success'):
        stats = results.get('summary_statistics', {})
        print(f"Éxito: {stats.get('successful_tests', 0)}/{stats.get('total_tests', 0)} tests")
    else:
        print("Revisa los logs para detalles de errores")