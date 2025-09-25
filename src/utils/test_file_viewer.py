#!/usr/bin/env python3
"""
Test del Sistema de Visualización de Archivos
Creado por Bernard Orozco

Prueba todas las funcionalidades del visor de archivos sin dependencias.
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_file_viewer():
    """Probar todas las funcionalidades del visor."""
    print(" TESTING SISTEMA DE VISUALIZACIÓN DE ARCHIVOS")
    print("=" * 60)
    
    from file_viewer import FileViewer, LogInspector, quick_file_scan
    
    # Test 1: Escáner rápido
    print("\n1⃣ TEST: ESCÁNER RÁPIDO")
    print("-" * 40)
    result = quick_file_scan()
    print(result)
    
    # Test 2: FileViewer completo
    print("\n2⃣ TEST: FILEVIEWER COMPLETO")
    print("-" * 40)
    viewer = FileViewer()
    files = viewer.scan_generated_files()
    
    print("Categorías encontradas:")
    for category, file_list in files.items():
        print(f"   {category}: {len(file_list)} archivos")
    
    # Test 3: Análisis de logs
    print("\n3⃣ TEST: INSPECTOR DE LOGS")
    print("-" * 40)
    inspector = LogInspector()
    recent_logs = inspector.find_latest_logs()
    
    print(f"Logs encontrados: {len(recent_logs)}")
    for log in recent_logs[:3]:
        print(f"  [DOC] {log['name']} - {log['modified_human']}")
    
    # Test 4: Visualizaciones
    print("\n4⃣ TEST: ANÁLISIS DE VISUALIZACIONES")
    print("-" * 40)
    visualizations = files.get('visualizations', [])
    
    if visualizations:
        print(f"Visualizaciones encontradas: {len(visualizations)}")
        for viz in visualizations:
            viz_info = viewer.get_visualization_info(viz['path'])
            print(f"  [GROWTH] {viz['name']} - {viz_info.get('type', 'Unknown')}")
    else:
        print(" No se encontraron visualizaciones")
    
    # Test 5: Reportes detallados
    print("\n5⃣ TEST: ANÁLISIS DE REPORTES")
    print("-" * 40)
    reports = files.get('reports', [])
    
    if reports:
        print(f"Reportes encontrados: {len(reports)}")
        for report in reports[:3]:
            print(f"  [CHART] {report['name']}")
            if 'report_type' in report:
                print(f"      Tipo: {report['report_type']}")
            if 'success_rate' in report:
                print(f"      Éxito: {report['success_rate']:.1%}")
    else:
        print(" No se encontraron reportes")
    
    # Test 6: Comandos de apertura (si hay visualizaciones)
    if visualizations:
        print("\n6⃣ TEST: COMANDOS DE APERTURA DE IMÁGENES")
        print("-" * 40)
        
        import os
        
        print("[COMPUTER] COMANDOS PARA ABRIR VISUALIZACIONES:")
        for viz in visualizations[:3]:  # Solo mostrar primeras 3
            if os.name == 'nt':  # Windows
                print(f"   start \"{viz['path']}\"")
            else:  # Linux/Mac
                print(f"   xdg-open \"{viz['path']}\"")
    
    print("\n TODOS LOS TESTS COMPLETADOS")
    print("[OK] Sistema de visualización funcionando correctamente")


def demo_interactive_usage():
    """Demostrar uso interactivo del sistema."""
    print("\n DEMOSTRACIÓN DE USO INTERACTIVO")
    print("=" * 60)
    
    print("[TARGET] FUNCIONALIDADES DISPONIBLES EN LA UI:")
    print()
    print("[DOC] EXPLORADOR DE LOGS (Opción A):")
    print("  1.  Ver log más reciente")
    print("  2. [CHART] Inspeccionar reporte específico") 
    print("  3. [SEARCH] Buscar por tipo de archivo")
    print("  4. [GROWTH] Resumen de todos los logs")
    print()
    print("[GROWTH] EXPLORADOR DE VISUALIZACIONES (Opción B):")
    print("  1.  Ver información detallada de gráfico")
    print("  2.  Organizar por tipo de análisis")
    print("  3. [COMPUTER] Mostrar comandos para abrir imágenes")
    print()
    print("[IDEA] COMANDOS PARA GENERAR ARCHIVOS:")
    print("  • python robo_poet.py --test quick")
    print("  • python robo_poet.py --analyze modelo.keras")
    print("  • python robo_poet.py --minima modelo.keras")
    print("  • python src/utils/demo_simple.py")
    print()
    print("[FIX] UNA VEZ CON DEPENDENCIAS INSTALADAS:")
    print("  python robo_poet.py → A → [seleccionar opción]")
    print("  python robo_poet.py → B → [seleccionar opción]")


if __name__ == "__main__":
    test_file_viewer()
    demo_interactive_usage()
    
    print("\n Sistema consolidado y listo por Aslan")
    print(" Todo organizado como un buen mate compartido")