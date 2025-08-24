#!/usr/bin/env python3
"""
File Viewer and Log Inspector for Robo-Poet Framework
Creado por Bernard Orozco

Sistema de visualización y gestión de archivos generados por el framework:
- Logs detallados de ejecución  
- Gráficos PNG de análisis
- Reportes JSON/TXT
- Modelos entrenados

Integrado con la interfaz académica para fácil acceso.
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import re

class FileViewer:
    """Visor y gestor de archivos del framework."""
    
    def __init__(self, base_dir: str = "."):
        """
        Inicializar visor de archivos.
        
        Args:
            base_dir: Directorio base del proyecto
        """
        self.base_dir = Path(base_dir)
        
        # Definir patrones de archivos importantes
        self.file_patterns = {
            'logs': [
                'module2_test_suite_*.log',
                'training_*.log', 
                'robo_poet_*.log'
            ],
            'reports': [
                'module2_test_report_*.json',
                'module2_test_report_*.txt',
                'gradient_analysis_lite_*.json',
                'minima_analysis_*.json',
                'ablation_study_*.json'
            ],
            'visualizations': [
                'gradient_analysis_*.png',
                'loss_landscape_*.png',
                'ablation_visualization_*.png'
            ],
            'models': [
                'models/*.keras',
                'models/*.h5',
                'checkpoints/*'
            ]
        }
    
    def scan_generated_files(self) -> Dict[str, List[Dict]]:
        """
        Escanear todos los archivos generados por el framework.
        
        Returns:
            Dict organizado por categoría con metadatos de archivos
        """
        results = {}
        
        for category, patterns in self.file_patterns.items():
            files = []
            
            for pattern in patterns:
                matching_files = list(self.base_dir.glob(pattern))
                
                for file_path in matching_files:
                    if file_path.is_file():
                        file_info = self._get_file_info(file_path, category)
                        files.append(file_info)
            
            # Ordenar por fecha de modificación (más reciente primero)
            files.sort(key=lambda x: x['modified_time'], reverse=True)
            results[category] = files
        
        return results
    
    def _get_file_info(self, file_path: Path, category: str) -> Dict:
        """Obtener información detallada de un archivo."""
        stat = file_path.stat()
        
        info = {
            'name': file_path.name,
            'path': str(file_path),
            'size': stat.st_size,
            'size_human': self._human_size(stat.st_size),
            'modified_time': stat.st_mtime,
            'modified_human': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
            'category': category,
            'extension': file_path.suffix
        }
        
        # Agregar información específica por tipo
        if category == 'logs':
            info.update(self._analyze_log_file(file_path))
        elif category == 'reports':
            info.update(self._analyze_report_file(file_path))
        elif category == 'visualizations':
            info.update(self._analyze_image_file(file_path))
        elif category == 'models':
            info.update(self._analyze_model_file(file_path))
        
        return info
    
    def _analyze_log_file(self, file_path: Path) -> Dict:
        """Analizar archivo de log."""
        info = {'type': 'log'}
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
                
                info.update({
                    'lines': len(lines),
                    'errors': len([l for l in lines if 'ERROR' in l or '❌' in l]),
                    'warnings': len([l for l in lines if 'WARNING' in l or '⚠️' in l]),
                    'success_markers': len([l for l in lines if '✅' in l or 'SUCCESS' in l])
                })
                
                # Detectar tipo de log específico
                if 'test_suite' in file_path.name:
                    info['log_type'] = 'Test Suite'
                elif 'training' in file_path.name:
                    info['log_type'] = 'Training'
                else:
                    info['log_type'] = 'General'
                    
        except Exception as e:
            info['error'] = str(e)
        
        return info
    
    def _analyze_report_file(self, file_path: Path) -> Dict:
        """Analizar archivo de reporte."""
        info = {'type': 'report'}
        
        if file_path.suffix == '.json':
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    info.update({
                        'format': 'JSON',
                        'keys': len(data.keys()) if isinstance(data, dict) else 0
                    })
                    
                    # Detectar tipo específico
                    if 'test_suite_metadata' in data:
                        info['report_type'] = 'Test Suite'
                        if 'summary_statistics' in data:
                            stats = data['summary_statistics']
                            info['success_rate'] = stats.get('success_rate', 0)
                    elif 'sharpness_classification' in data:
                        info['report_type'] = 'Minima Analysis'
                        info['sharpness'] = data['sharpness_classification'].get('overall_sharpness', 0)
                    elif 'collapse_analysis' in data:
                        info['report_type'] = 'Gradient Analysis'
                    else:
                        info['report_type'] = 'Unknown'
                        
            except Exception as e:
                info['error'] = str(e)
        else:
            info['format'] = 'Text'
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    info['lines'] = len(content.split('\n'))
            except:
                pass
        
        return info
    
    def _analyze_image_file(self, file_path: Path) -> Dict:
        """Analizar archivo de imagen."""
        info = {'type': 'image'}
        
        # Detectar tipo de visualización
        if 'gradient_analysis' in file_path.name:
            info['viz_type'] = 'Gradient Flow Analysis'
        elif 'loss_landscape' in file_path.name:
            info['viz_type'] = 'Loss Landscape'
        elif 'ablation' in file_path.name:
            info['viz_type'] = 'Ablation Study'
        else:
            info['viz_type'] = 'Unknown'
        
        return info
    
    def _analyze_model_file(self, file_path: Path) -> Dict:
        """Analizar archivo de modelo."""
        info = {'type': 'model'}
        
        # Detectar tipo de modelo
        if 'operated' in file_path.name:
            info['model_type'] = 'Operated (Post-Surgery)'
        elif 'checkpoint' in file_path.name:
            info['model_type'] = 'Checkpoint'
        else:
            info['model_type'] = 'Trained Model'
        
        return info
    
    def _human_size(self, size_bytes: int) -> str:
        """Convertir bytes a formato legible."""
        if size_bytes == 0:
            return "0 B"
        
        size_names = ["B", "KB", "MB", "GB"]
        i = 0
        
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        
        return f"{size_bytes:.1f} {size_names[i]}"
    
    def display_file_summary(self) -> str:
        """Generar resumen textual de archivos."""
        files = self.scan_generated_files()
        
        summary = []
        summary.append("📁 ARCHIVOS GENERADOS POR EL FRAMEWORK")
        summary.append("=" * 60)
        
        total_files = 0
        total_size = 0
        
        for category, file_list in files.items():
            if not file_list:
                continue
                
            category_display = {
                'logs': '📝 LOGS',
                'reports': '📊 REPORTES', 
                'visualizations': '📈 VISUALIZACIONES',
                'models': '🧠 MODELOS'
            }.get(category, category.upper())
            
            summary.append(f"\n{category_display} ({len(file_list)} archivos):")
            summary.append("-" * 40)
            
            for file_info in file_list[:5]:  # Mostrar máximo 5 por categoría
                name = file_info['name']
                size = file_info['size_human']
                date = file_info['modified_human']
                
                # Información específica por tipo
                extra_info = ""
                if category == 'logs':
                    if 'errors' in file_info:
                        extra_info = f" (❌ {file_info['errors']} errores)"
                elif category == 'reports':
                    if 'success_rate' in file_info:
                        extra_info = f" (✅ {file_info['success_rate']:.1%} éxito)"
                
                summary.append(f"  • {name} - {size} - {date}{extra_info}")
                total_files += 1
                total_size += file_info['size']
            
            if len(file_list) > 5:
                summary.append(f"  ... y {len(file_list) - 5} archivos más")
        
        summary.append(f"\n📊 RESUMEN TOTAL:")
        summary.append(f"  Total archivos: {total_files}")
        summary.append(f"  Tamaño total: {self._human_size(total_size)}")
        
        return "\n".join(summary)
    
    def read_log_file(self, file_path: str, tail_lines: int = 100) -> Dict:
        """
        Leer archivo de log con opciones de filtrado.
        
        Args:
            file_path: Ruta al archivo de log
            tail_lines: Número de líneas finales a mostrar
        
        Returns:
            Dict con contenido del log y estadísticas
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            # Obtener líneas finales si se especifica
            if tail_lines and len(lines) > tail_lines:
                displayed_lines = lines[-tail_lines:]
                truncated = True
            else:
                displayed_lines = lines
                truncated = False
            
            # Analizar contenido
            analysis = {
                'total_lines': len(lines),
                'displayed_lines': len(displayed_lines),
                'truncated': truncated,
                'errors': [],
                'warnings': [],
                'success': []
            }
            
            for i, line in enumerate(lines):
                line_num = i + 1
                if any(marker in line for marker in ['ERROR', '❌', 'FAILED']):
                    analysis['errors'].append((line_num, line.strip()))
                elif any(marker in line for marker in ['WARNING', '⚠️', 'WARN']):
                    analysis['warnings'].append((line_num, line.strip()))
                elif any(marker in line for marker in ['✅', 'SUCCESS', 'COMPLETED']):
                    analysis['success'].append((line_num, line.strip()))
            
            return {
                'content': ''.join(displayed_lines),
                'analysis': analysis,
                'success': True
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'success': False
            }
    
    def get_visualization_info(self, image_path: str) -> Dict:
        """
        Obtener información sobre una visualización generada.
        
        Args:
            image_path: Ruta al archivo PNG
        
        Returns:
            Dict con información de la imagen
        """
        path = Path(image_path)
        
        if not path.exists():
            return {'error': 'Archivo no encontrado', 'success': False}
        
        info = {
            'name': path.name,
            'path': str(path),
            'size': self._human_size(path.stat().st_size),
            'created': datetime.fromtimestamp(path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
            'success': True
        }
        
        # Detectar tipo de análisis
        if 'gradient_analysis' in path.name:
            info.update({
                'type': 'Gradient Analysis',
                'description': 'Visualización del flujo de gradientes por capas',
                'contains': ['Gradient norms', 'Layer analysis', 'Vanishing/Exploding detection']
            })
        elif 'loss_landscape' in path.name:
            info.update({
                'type': 'Loss Landscape',  
                'description': 'Análisis del paisaje de pérdida (sharp vs flat)',
                'contains': ['Sharpness analysis', 'Hessian curvature', 'Perturbation results']
            })
        elif 'ablation' in path.name:
            info.update({
                'type': 'Ablation Study',
                'description': 'Comparación de componentes del modelo',
                'contains': ['Component comparison', 'Performance metrics', 'Parameter efficiency']
            })
        
        return info


class LogInspector:
    """Inspector especializado para logs del framework."""
    
    def __init__(self):
        self.viewer = FileViewer()
    
    def find_latest_logs(self, log_type: str = None) -> List[Dict]:
        """
        Encontrar logs más recientes.
        
        Args:
            log_type: Tipo específico ('test_suite', 'training', None para todos)
        
        Returns:
            Lista de logs ordenados por fecha
        """
        files = self.viewer.scan_generated_files()
        logs = files.get('logs', [])
        
        if log_type:
            logs = [log for log in logs if log_type in log['name']]
        
        return logs[:10]  # Máximo 10 más recientes
    
    def get_log_summary(self, log_path: str) -> Dict:
        """Obtener resumen ejecutivo de un log."""
        log_data = self.viewer.read_log_file(log_path)
        
        if not log_data['success']:
            return log_data
        
        analysis = log_data['analysis']
        
        # Generar resumen ejecutivo
        summary = {
            'total_events': analysis['total_lines'],
            'errors_count': len(analysis['errors']),
            'warnings_count': len(analysis['warnings']),
            'success_count': len(analysis['success']),
            'status': 'SUCCESS' if len(analysis['errors']) == 0 else 'WITH_ERRORS',
            'recent_errors': analysis['errors'][-5:] if analysis['errors'] else [],
            'recent_success': analysis['success'][-5:] if analysis['success'] else []
        }
        
        return {
            'summary': summary,
            'success': True
        }


def quick_file_scan() -> str:
    """Función rápida para escanear archivos desde CLI."""
    viewer = FileViewer()
    return viewer.display_file_summary()


if __name__ == "__main__":
    print("📁 VISOR DE ARCHIVOS ROBO-POET")
    print("=" * 50)
    print(quick_file_scan())