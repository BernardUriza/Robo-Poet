#!/usr/bin/env python3
"""
File Management Utilities for Robo-Poet Framework

Handles model files, generations, experiments and academic file organization.

Author: ML Academic Framework
Version: 2.1
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime


class FileManager:
    """Manages all file operations for the academic framework."""
    
    def __init__(self):
        """Initialize file manager with standard directories."""
        self.models_dir = Path("models")
        self.generations_dir = Path("generations")
        self.experiments_dir = Path("experiments")
        self.logs_dir = Path("logs")
        
        # Ensure directories exist
        for directory in [self.models_dir, self.generations_dir, self.experiments_dir, self.logs_dir]:
            directory.mkdir(exist_ok=True)
    
    def list_available_models(self) -> List[str]:
        """Get list of available trained models."""
        if not self.models_dir.exists():
            return []
        
        # Find ALL .keras and .h5 models (not just robo_poet_model_*)
        keras_models = list(self.models_dir.glob("*.keras"))
        h5_models = list(self.models_dir.glob("*.h5"))
        
        # Exclude checkpoint files
        all_models = [m for m in keras_models + h5_models 
                     if not m.name.startswith('checkpoint_')]
        
        # Sort by modification time (newest first)
        return sorted([str(model) for model in all_models], 
                     key=lambda x: Path(x).stat().st_mtime, reverse=True)
    
    def list_available_models_enhanced(self) -> List[Dict]:
        """Get enhanced list of models with metadata."""
        models = self.list_available_models()
        enhanced_models = []
        
        for model_path in models:
            model_info = {
                'path': model_path,
                'name': Path(model_path).name,
                'size_mb': Path(model_path).stat().st_size / (1024 * 1024),
                'created': datetime.fromtimestamp(Path(model_path).stat().st_mtime),
                'metadata': None,
                'quality_rating': 'Unknown'
            }
            
            # Try to load metadata
            metadata_path = Path(model_path).parent / (Path(model_path).stem + '_metadata.json')
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    model_info['metadata'] = metadata
                    
                    # Calculate quality rating based on loss
                    final_loss = metadata.get('final_loss', float('inf'))
                    if isinstance(final_loss, (int, float)):
                        if final_loss < 1.0:
                            model_info['quality_rating'] = '🌟 Excelente'
                        elif final_loss < 1.5:
                            model_info['quality_rating'] = '⭐ Bueno'
                        elif final_loss < 2.0:
                            model_info['quality_rating'] = '📊 Aceptable'
                        else:
                            model_info['quality_rating'] = '⚠️ Requiere mejora'
                
                except Exception as e:
                    print(f"⚠️ Error loading metadata for {model_path}: {e}")
            
            enhanced_models.append(model_info)
        
        return enhanced_models
    
    def clean_all_models(self) -> Dict[str, int]:
        """Clean all models and return cleanup statistics."""
        if not self.models_dir.exists():
            return {'models': 0, 'metadata': 0, 'checkpoints': 0, 'total_mb': 0}
        
        # Count existing files
        model_files = list(self.models_dir.glob("*.keras")) + list(self.models_dir.glob("*.h5"))
        metadata_files = list(self.models_dir.glob("*_metadata.json"))
        checkpoint_files = list(self.models_dir.glob("checkpoint_*.keras"))
        
        # Calculate total size
        total_size = 0
        all_files = model_files + metadata_files + checkpoint_files
        for file_path in all_files:
            if file_path.exists():
                total_size += file_path.stat().st_size
        
        # Delete files
        deleted_counts = {'models': 0, 'metadata': 0, 'checkpoints': 0}
        
        for model_file in model_files:
            if model_file.exists():
                model_file.unlink()
                deleted_counts['models'] += 1
        
        for metadata_file in metadata_files:
            if metadata_file.exists():
                metadata_file.unlink()
                deleted_counts['metadata'] += 1
        
        for checkpoint_file in checkpoint_files:
            if checkpoint_file.exists():
                checkpoint_file.unlink()
                deleted_counts['checkpoints'] += 1
        
        deleted_counts['total_mb'] = total_size / (1024 * 1024)
        return deleted_counts
    
    def save_generation_to_file(self, result: str, seed: str, temperature: float, 
                               length: int, model_name: str = "unknown") -> str:
        """Save generation result to file with metadata."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"generation_{timestamp}.txt"
        filepath = self.generations_dir / filename
        
        # Create generation metadata
        metadata = {
            'timestamp': timestamp,
            'model_name': model_name,
            'seed': seed,
            'temperature': temperature,
            'length': length,
            'actual_length': len(result),
            'generation_time': datetime.now().isoformat()
        }
        
        # Save generation with header
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("🎨 ROBO-POET: GENERACIÓN DE TEXTO\n")
            f.write("=" * 60 + "\n")
            f.write(f"📅 Fecha: {metadata['generation_time']}\n")
            f.write(f"🤖 Modelo: {model_name}\n")
            f.write(f"🌱 Seed: '{seed}'\n")
            f.write(f"🌡️ Temperature: {temperature}\n")
            f.write(f"📏 Longitud: {length} (real: {len(result)})\n")
            f.write("=" * 60 + "\n\n")
            f.write(result)
            f.write("\n\n" + "=" * 60 + "\n")
            f.write("🎓 Generado con Robo-Poet Academic Framework\n")
            f.write("=" * 60 + "\n")
        
        # Save metadata
        metadata_filepath = self.generations_dir / f"generation_{timestamp}_metadata.json"
        with open(metadata_filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        return str(filepath)
    
    def save_experiment_results(self, experiment_type: str, results: List[Dict], 
                              model_name: str = "unknown") -> str:
        """Save batch experiment results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"experiment_{experiment_type}_{timestamp}.json"
        filepath = self.experiments_dir / filename
        
        experiment_data = {
            'experiment_type': experiment_type,
            'timestamp': timestamp,
            'model_name': model_name,
            'results': results,
            'total_generations': len(results),
            'creation_time': datetime.now().isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(experiment_data, f, indent=2, ensure_ascii=False)
        
        return str(filepath)
    
    def get_text_files(self) -> List[str]:
        """Get list of available text files for training."""
        text_extensions = ['*.txt', '*.text']
        text_files = []
        
        # Search in current directory and common locations
        search_paths = [Path('.'), Path('data'), Path('corpus')]
        
        for search_path in search_paths:
            if search_path.exists():
                for extension in text_extensions:
                    text_files.extend(search_path.glob(extension))
        
        return sorted([str(f) for f in text_files if f.stat().st_size > 1000])  # Min 1KB
    
    def validate_text_file(self, filepath: str) -> Tuple[bool, str]:
        """Validate text file for training."""
        try:
            path = Path(filepath)
            if not path.exists():
                return False, f"Archivo no encontrado: {filepath}"
            
            if path.stat().st_size < 1000:
                return False, f"Archivo demasiado pequeño (mínimo 1KB): {path.stat().st_size} bytes"
            
            # Try to read and validate encoding
            with open(path, 'r', encoding='utf-8') as f:
                sample = f.read(1000)
                if len(sample.strip()) == 0:
                    return False, "Archivo vacío o solo espacios en blanco"
            
            return True, f"Archivo válido: {path.stat().st_size / 1024:.1f} KB"
        
        except UnicodeDecodeError:
            return False, "Error de codificación: el archivo debe estar en UTF-8"
        except Exception as e:
            return False, f"Error leyendo archivo: {e}"
    
    def get_available_generations(self) -> List[Dict]:
        """Get list of previously saved generations."""
        if not self.generations_dir.exists():
            return []
        
        generation_files = list(self.generations_dir.glob("generation_*.txt"))
        generations = []
        
        for gen_file in generation_files:
            # Try to load corresponding metadata
            timestamp = gen_file.stem.replace('generation_', '')
            metadata_file = self.generations_dir / f"generation_{timestamp}_metadata.json"
            
            gen_info = {
                'file': str(gen_file),
                'timestamp': timestamp,
                'size_kb': gen_file.stat().st_size / 1024,
                'created': datetime.fromtimestamp(gen_file.stat().st_mtime),
                'metadata': None
            }
            
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        gen_info['metadata'] = json.load(f)
                except:
                    pass
            
            generations.append(gen_info)
        
        return sorted(generations, key=lambda x: x['created'], reverse=True)