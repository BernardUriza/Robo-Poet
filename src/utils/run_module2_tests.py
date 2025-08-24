#!/usr/bin/env python3
"""
DEPRECATED: Usa python robo_poet.py --test quick en su lugar

Este script se mantiene por compatibilidad pero la funcionalidad
está integrada en el sistema principal.
"""

import sys
from pathlib import Path

# Add src directory to Python path  
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

def main():
    """Punto de entrada principal para tests del Módulo 2."""
    
    parser = argparse.ArgumentParser(
        description="🧪 Suite de Tests - Módulo 2: Deep Gradient Flow Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python run_module2_tests.py --quick                                # Demo rápido (5 min)
  python run_module2_tests.py --full                                 # Validación completa (20 min)
  python run_module2_tests.py --tests training gradient_analysis     # Tests específicos
  python run_module2_tests.py --block core                          # Bloque de funcionalidades
        """
    )
    
    # Modes of operation
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--quick', action='store_true',
                           help='Demo rápido: tests básicos (~5 min)')
    mode_group.add_argument('--full', action='store_true',
                           help='Validación completa: todos los tests (~20 min)')
    mode_group.add_argument('--tests', nargs='+',
                           choices=['training', 'gradient_analysis', 'minima_analysis', 
                                   'ablation_experiments', 'emergency_surgery', 'report_generation'],
                           help='Tests específicos a ejecutar')
    mode_group.add_argument('--block', choices=['core', 'analysis', 'advanced'],
                           help='Ejecutar bloque de tests por categoría')
    
    # Optional parameters
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Output verbose con detalles adicionales')
    parser.add_argument('--no-cleanup', action='store_true',
                       help='No limpiar archivos temporales después de tests')
    
    args = parser.parse_args()
    
    # Import test suite
    try:
        from testing.module2_test_suite import run_selected_tests, run_quick_demo, run_full_validation
    except ImportError as e:
        print(f"❌ Error importando suite de tests: {e}")
        print("Asegúrate de ejecutar desde el directorio raíz del proyecto")
        return 1
    
    print("🧪 INICIANDO TESTS MÓDULO 2")
    print("=" * 60)
    
    # Execute based on selected mode
    results = None
    
    try:
        if args.quick:
            print("🚀 Ejecutando DEMO RÁPIDO...")
            results = run_quick_demo()
            
        elif args.full:
            print("🔬 Ejecutando VALIDACIÓN COMPLETA...")
            results = run_full_validation()
            
        elif args.tests:
            print(f"🎯 Ejecutando TESTS ESPECÍFICOS: {args.tests}")
            results = run_selected_tests(args.tests)
            
        elif args.block:
            # Define test blocks
            test_blocks = {
                'core': ['training', 'emergency_surgery'],
                'analysis': ['gradient_analysis', 'minima_analysis'], 
                'advanced': ['ablation_experiments', 'report_generation']
            }
            
            selected_tests = test_blocks[args.block]
            print(f"📊 Ejecutando BLOQUE {args.block.upper()}: {selected_tests}")
            results = run_selected_tests(selected_tests)
        
        # Show results
        if results:
            print("\n" + "=" * 60)
            show_final_results(results, args.verbose)
            
            if results.get('success'):
                print("🎉 TODOS LOS TESTS EXITOSOS")
                return 0
            else:
                print("❌ ALGUNOS TESTS FALLARON")
                return 1
        else:
            print("❌ No se obtuvieron resultados de los tests")
            return 1
            
    except KeyboardInterrupt:
        print("\n\n🛑 Tests interrumpidos por usuario")
        return 0
    except Exception as e:
        print(f"\n❌ Error ejecutando tests: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

def show_final_results(results, verbose=False):
    """Mostrar resultados finales de los tests."""
    
    stats = results.get('summary_statistics', {})
    metadata = results.get('test_suite_metadata', {})
    individual_results = results.get('individual_test_results', {})
    
    print("📊 RESUMEN DE RESULTADOS:")
    print("-" * 40)
    print(f"✅ Tests exitosos: {stats.get('successful_tests', 0)}")
    print(f"❌ Tests fallidos: {stats.get('failed_tests', 0)}")
    print(f"📈 Tasa de éxito: {stats.get('success_rate', 0):.1%}")
    print(f"⏱️ Tiempo total: {stats.get('total_execution_time', 0):.2f} segundos")
    
    if verbose:
        print(f"\n📋 DETALLES POR TEST:")
        print("-" * 40)
        
        for test_name, result in individual_results.items():
            status_emoji = "✅" if result.get('status') == 'SUCCESS' else "❌"
            test_display = test_name.replace('_', ' ').title()
            
            print(f"{status_emoji} {test_display}")
            
            if result.get('status') == 'SUCCESS':
                if 'total_time' in result:
                    print(f"   ⏱️ Tiempo: {result['total_time']:.2f}s")
                    
                # Mostrar detalles específicos por tipo de test
                if test_name == 'training':
                    if 'parameters' in result:
                        print(f"   🧠 Parámetros: {result['parameters']:,}")
                elif test_name == 'gradient_analysis':
                    print(f"   📉 Vanishing: {'Sí' if result.get('has_vanishing') else 'No'}")
                    print(f"   📈 Exploding: {'Sí' if result.get('has_exploding') else 'No'}")
                elif test_name == 'minima_analysis':
                    print(f"   🏔️ Categoría: {result.get('sharpness_category', 'N/A')}")
                elif test_name == 'emergency_surgery':
                    success = 'Exitosa' if result.get('surgery_successful') else 'Fallida'
                    print(f"   🏥 Cirugía: {success}")
            else:
                print(f"   ❌ Error: {result.get('error', 'Error desconocido')}")
            
            print()
    
    # Archivos generados
    log_file = metadata.get('log_file')
    if log_file:
        print(f"📁 ARCHIVOS GENERADOS:")
        print(f"   📝 Log detallado: {log_file}")
        print(f"   📊 Reportes en directorio actual")
        
        if verbose:
            print(f"   🎯 Modelo usado: {metadata.get('model_used', 'N/A')}")
            print(f"   🕒 Timestamp: {metadata.get('timestamp', 'N/A')}")

if __name__ == "__main__":
    sys.exit(main())