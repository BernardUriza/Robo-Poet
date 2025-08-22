#!/usr/bin/env python3
"""
Script de limpieza automática de checkpoints antiguos.

Limpia checkpoints manteniendo solo los más recientes y los mejores modelos.
Parte de las mejoras de Fase 2 - elimina 60+ archivos de checkpoint.
"""

import os
from pathlib import Path
from datetime import datetime, timedelta
import argparse
from typing import List, Dict, Any


def cleanup_old_checkpoints(
    models_dir: str = 'models/', 
    keep_last: int = 5, 
    max_age_days: int = 7,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Limpia checkpoints antiguos manteniendo solo los más recientes.
    
    Args:
        models_dir: Directorio de modelos
        keep_last: Número de modelos más recientes a conservar
        max_age_days: Edad máxima en días para conservar
        dry_run: Si True, solo muestra qué se eliminaría
        
    Returns:
        Dict con estadísticas de limpieza
    """
    models_path = Path(models_dir)
    if not models_path.exists():
        return {
            'error': f'Directorio {models_dir} no existe',
            'files_deleted': 0,
            'space_freed_mb': 0
        }
    
    # Obtener todos los archivos de modelo (tanto .h5 como .keras)
    model_files = []
    
    for pattern in ['*.h5', '*.keras']:
        for file in models_path.glob(pattern):
            stat = file.stat()
            model_files.append({
                'path': file,
                'name': file.name,
                'modified': datetime.fromtimestamp(stat.st_mtime),
                'size': stat.st_size,
                'extension': file.suffix
            })
    
    print(f"📊 Estado inicial del directorio {models_dir}:")
    print(f"   Total archivos encontrados: {len(model_files)}")
    
    if not model_files:
        return {
            'files_deleted': 0,
            'space_freed_mb': 0,
            'kept_files': 0
        }
    
    # Estadísticas por tipo
    h5_files = [f for f in model_files if f['extension'] == '.h5']
    keras_files = [f for f in model_files if f['extension'] == '.keras']
    
    print(f"   Archivos .h5: {len(h5_files)}")
    print(f"   Archivos .keras: {len(keras_files)}")
    
    total_size_mb = sum(f['size'] for f in model_files) / 1024 / 1024
    print(f"   Tamaño total: {total_size_mb:.1f}MB")
    
    # Ordenar por fecha de modificación (más reciente primero)
    model_files.sort(key=lambda x: x['modified'], reverse=True)
    
    # Estrategia de conservación:
    # 1. Conservar los últimos N archivos
    # 2. Entre los restantes, eliminar los más antiguos que max_age_days
    
    to_keep = model_files[:keep_last]
    candidates_for_deletion = model_files[keep_last:]
    
    # Filtrar por edad máxima
    cutoff_date = datetime.now() - timedelta(days=max_age_days)
    to_delete = [
        f for f in candidates_for_deletion 
        if f['modified'] < cutoff_date
    ]
    
    print(f"\n📋 Estrategia de limpieza:")
    print(f"   Conservar últimos {keep_last} archivos")
    print(f"   Eliminar archivos con más de {max_age_days} días")
    print(f"   Archivos a conservar: {len(to_keep)}")
    print(f"   Archivos a eliminar: {len(to_delete)}")
    
    if not to_delete:
        print("\n✅ No hay archivos para eliminar")
        return {
            'files_deleted': 0,
            'space_freed_mb': 0,
            'kept_files': len(model_files)
        }
    
    # Mostrar detalles de eliminación
    print(f"\n🗑️ Archivos marcados para eliminación:")
    total_freed = 0
    
    for file_info in to_delete:
        size_mb = file_info['size'] / 1024 / 1024
        age_days = (datetime.now() - file_info['modified']).days
        print(f"   📄 {file_info['name']} ({size_mb:.1f}MB, {age_days}d)")
        total_freed += file_info['size']
    
    print(f"\n📈 Impacto de limpieza:")
    print(f"   Espacio a liberar: {total_freed/1024/1024:.1f}MB")
    print(f"   Reducción: {(total_freed/sum(f['size'] for f in model_files)*100):.1f}%")
    
    if dry_run:
        print(f"\n🧪 DRY RUN - No se eliminaron archivos")
        return {
            'files_deleted': len(to_delete),
            'space_freed_mb': total_freed/1024/1024,
            'kept_files': len(model_files) - len(to_delete),
            'dry_run': True
        }
    
    # Eliminar archivos
    print(f"\n🚀 Ejecutando limpieza...")
    deleted_count = 0
    
    for file_info in to_delete:
        try:
            file_info['path'].unlink()
            
            # Eliminar metadata JSON asociada si existe
            json_path = file_info['path'].with_suffix('.json')
            if json_path.exists():
                json_path.unlink()
                print(f"   🗑️ También eliminado: {json_path.name}")
            
            deleted_count += 1
            
        except Exception as e:
            print(f"   ❌ Error eliminando {file_info['name']}: {e}")
    
    print(f"\n✅ Limpieza completada")
    print(f"   Archivos eliminados: {deleted_count}")
    print(f"   Espacio liberado: {total_freed/1024/1024:.1f}MB")
    print(f"   Modelos conservados: {len(model_files) - deleted_count}")
    
    return {
        'files_deleted': deleted_count,
        'space_freed_mb': total_freed/1024/1024,
        'kept_files': len(model_files) - deleted_count,
        'dry_run': False
    }


def main():
    """Función principal con CLI."""
    parser = argparse.ArgumentParser(
        description="Limpia checkpoints antiguos del directorio models/"
    )
    parser.add_argument(
        '--models-dir', 
        default='models/', 
        help='Directorio de modelos (default: models/)'
    )
    parser.add_argument(
        '--keep-last', 
        type=int, 
        default=5, 
        help='Número de modelos más recientes a conservar (default: 5)'
    )
    parser.add_argument(
        '--max-age-days', 
        type=int, 
        default=7, 
        help='Edad máxima en días para conservar modelos (default: 7)'
    )
    parser.add_argument(
        '--dry-run', 
        action='store_true', 
        help='Solo mostrar qué se eliminaría, no eliminar realmente'
    )
    
    args = parser.parse_args()
    
    print("🧹 LIMPIEZA AUTOMÁTICA DE CHECKPOINTS")
    print("=" * 50)
    
    result = cleanup_old_checkpoints(
        models_dir=args.models_dir,
        keep_last=args.keep_last,
        max_age_days=args.max_age_days,
        dry_run=args.dry_run
    )
    
    if 'error' in result:
        print(f"❌ Error: {result['error']}")
        return 1
    
    print(f"\n📊 RESUMEN FINAL:")
    print(f"   Archivos eliminados: {result['files_deleted']}")
    print(f"   Espacio liberado: {result['space_freed_mb']:.1f}MB")
    print(f"   Archivos conservados: {result['kept_files']}")
    
    if result.get('dry_run'):
        print(f"   (Simulación - ejecutar sin --dry-run para aplicar cambios)")
    
    return 0


if __name__ == "__main__":
    exit(main())