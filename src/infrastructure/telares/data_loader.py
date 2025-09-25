"""
Telares Data Loader Infrastructure
Handles loading and preprocessing of datasets for training
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
from pathlib import Path


class TelaresDataLoader:
    """
    Infrastructure component for loading telares datasets
    Handles CSV files, text corpus, and data preprocessing
    """
    
    def __init__(self):
        self.tactic_names = [
            "control_emocional",
            "presion_social", 
            "lenguaje_espiritual",
            "logica_circular",
            "urgencia_artificial",
            "testimonio_fabricado",
            "promesa_irrealista"
        ]
    
    def load_telares_dataset(self, dataset_path: str = None) -> Tuple[List[str], np.ndarray, Dict]:
        """
        Load telares dataset from CSV file
        
        Args:
            dataset_path: Path to CSV file, uses default if None
            
        Returns:
            Tuple of (messages, labels, metadata)
        """
        if dataset_path is None:
            dataset_path = "src/data/telares_dataset_135.csv"
            # Fallback to root directory if src path doesn't exist
            if not Path(dataset_path).exists():
                dataset_path = "telares_dataset_135.csv"
        
        try:
            dataset_file = Path(dataset_path)
            if not dataset_file.exists():
                raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
            
            # Load CSV
            df = pd.read_csv(dataset_file)
            print(f"[CHART] Dataset cargado: {len(df)} filas")
            
            # Extract messages
            if 'mensaje' not in df.columns:
                raise ValueError("Column 'mensaje' not found in dataset")
            
            messages = df['mensaje'].dropna().tolist()
            
            # Extract labels for each tactic
            labels_list = []
            available_tactics = []
            
            for tactic in self.tactic_names:
                if tactic in df.columns:
                    # Convert to binary labels
                    tactic_labels = df[tactic].fillna(0).astype(int).tolist()
                    labels_list.append(tactic_labels)
                    available_tactics.append(tactic)
                else:
                    print(f"WARNING: Táctica '{tactic}' no encontrada en dataset")
                    # Add zero labels for missing tactic
                    labels_list.append([0] * len(messages))
                    available_tactics.append(tactic)
            
            # Convert to numpy array (n_samples, n_tactics)
            labels_array = np.array(labels_list).T
            
            # Remove rows where message is empty
            valid_indices = [i for i, msg in enumerate(messages) if msg and len(msg.strip()) > 10]
            messages = [messages[i] for i in valid_indices]
            labels_array = labels_array[valid_indices]
            
            print(f"[OK] Mensajes válidos: {len(messages)}")
            print(f" Tácticas disponibles: {len(available_tactics)}")
            
            # Calculate label statistics
            label_stats = {}
            for i, tactic in enumerate(available_tactics):
                positive_count = labels_array[:, i].sum()
                percentage = (positive_count / len(messages)) * 100 if len(messages) > 0 else 0
                label_stats[tactic] = {
                    "positive_samples": int(positive_count),
                    "percentage": percentage
                }
            
            metadata = {
                "dataset_path": dataset_path,
                "total_messages": len(messages),
                "tactic_names": available_tactics,
                "label_statistics": label_stats,
                "dataset_columns": list(df.columns)
            }
            
            return messages, labels_array, metadata
            
        except Exception as e:
            print(f"[X] Error cargando dataset: {e}")
            return [], np.array([]), {}
    
    def load_poetic_corpus(self, corpus_dir: str = "corpus") -> List[str]:
        """
        Load poetic corpus as negative control samples
        
        Args:
            corpus_dir: Directory containing text files
            
        Returns:
            List of text fragments
        """
        corpus_path = Path(corpus_dir)
        
        if not corpus_path.exists():
            print(f"WARNING: Directorio de corpus no encontrado: {corpus_dir}")
            return []
        
        text_fragments = []
        text_files = list(corpus_path.glob("*.txt"))
        
        print(f"[BOOKS] Procesando {len(text_files)} archivos de corpus...")
        
        for txt_file in text_files:
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                
                # Split into fragments similar to WhatsApp messages (100-200 chars)
                fragments = self._split_into_fragments(content, min_length=50, max_length=200)
                
                # Limit fragments per file to avoid overwhelming the dataset
                selected_fragments = fragments[:25]  # Max 25 fragments per file
                text_fragments.extend(selected_fragments)
                
                print(f"    {txt_file.name}: {len(selected_fragments)} fragmentos")
                
            except Exception as e:
                print(f"WARNING: Error leyendo {txt_file}: {e}")
        
        print(f"[OK] Corpus poético procesado: {len(text_fragments)} fragmentos")
        return text_fragments
    
    def create_hybrid_dataset(self, 
                             telares_messages: List[str], 
                             telares_labels: np.ndarray,
                             poetic_fragments: List[str]) -> Tuple[List[str], np.ndarray]:
        """
        Create hybrid dataset combining telares and poetic corpus
        
        Args:
            telares_messages: Pyramid scheme messages
            telares_labels: Labels for telares messages
            poetic_fragments: Poetic text fragments (negative controls)
            
        Returns:
            Combined messages and labels
        """
        # Create zero labels for poetic fragments (no manipulation)
        poetic_labels = np.zeros((len(poetic_fragments), telares_labels.shape[1]))
        
        # Combine datasets
        combined_messages = telares_messages + poetic_fragments
        combined_labels = np.vstack([telares_labels, poetic_labels])
        
        print(f"[SCIENCE] Dataset híbrido creado:")
        print(f"    Mensajes telares: {len(telares_messages)}")
        print(f"   [BOOKS] Fragmentos poéticos: {len(poetic_fragments)}")
        print(f"   [CHART] Total combinado: {len(combined_messages)}")
        
        return combined_messages, combined_labels
    
    def validate_dataset(self, messages: List[str], labels: np.ndarray) -> Dict:
        """
        Validate dataset quality and statistics
        
        Args:
            messages: Text messages
            labels: Label array
            
        Returns:
            Validation report
        """
        report = {
            "total_messages": len(messages),
            "total_tactics": labels.shape[1] if labels.ndim > 1 else 1,
            "valid": True,
            "issues": []
        }
        
        # Check for empty messages
        empty_messages = sum(1 for msg in messages if not msg or len(msg.strip()) < 10)
        if empty_messages > 0:
            report["issues"].append(f"{empty_messages} mensajes vacíos o muy cortos")
        
        # Check label distribution
        if labels.ndim > 1:
            for i in range(labels.shape[1]):
                positive_count = labels[:, i].sum()
                if positive_count == 0:
                    report["issues"].append(f"Táctica {i} sin ejemplos positivos")
                elif positive_count == len(messages):
                    report["issues"].append(f"Táctica {i} sin ejemplos negativos")
        
        # Check message length distribution
        msg_lengths = [len(msg) for msg in messages]
        avg_length = np.mean(msg_lengths)
        report["average_message_length"] = avg_length
        
        if avg_length < 20:
            report["issues"].append("Mensajes muy cortos en promedio")
        elif avg_length > 500:
            report["issues"].append("Mensajes muy largos en promedio")
        
        report["valid"] = len(report["issues"]) == 0
        
        return report
    
    def _split_into_fragments(self, text: str, min_length: int = 50, max_length: int = 200) -> List[str]:
        """
        Split long text into fragments similar to messaging app messages
        
        Args:
            text: Original text
            min_length: Minimum fragment length
            max_length: Maximum fragment length
            
        Returns:
            List of text fragments
        """
        # First try to split by sentences
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        fragments = []
        
        current_fragment = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # If adding this sentence would exceed max_length, save current and start new
            if current_fragment and len(current_fragment + " " + sentence) > max_length:
                if len(current_fragment) >= min_length:
                    fragments.append(current_fragment)
                current_fragment = sentence
            else:
                current_fragment = current_fragment + " " + sentence if current_fragment else sentence
        
        # Add remaining fragment if it meets minimum length
        if current_fragment and len(current_fragment) >= min_length:
            fragments.append(current_fragment)
        
        # If no good sentence splits, use character-based splitting
        if not fragments:
            for i in range(0, len(text), max_length):
                fragment = text[i:i + max_length]
                if len(fragment) >= min_length:
                    fragments.append(fragment)
        
        return fragments
    
    def get_dataset_info(self) -> Dict:
        """Get information about available datasets"""
        info = {
            "supported_formats": ["CSV", "TXT"],
            "required_columns": ["mensaje"] + self.tactic_names,
            "tactic_names": self.tactic_names,
            "available_datasets": []
        }
        
        # Check for telares dataset
        telares_paths = ["src/data/telares_dataset_135.csv", "telares_dataset_135.csv"]
        for path in telares_paths:
            if Path(path).exists():
                info["available_datasets"].append({
                    "name": "Telares Dataset",
                    "path": path,
                    "type": "CSV",
                    "description": "Real pyramid scheme messages with manipulation labels"
                })
                break
        
        # Check for poetic corpus
        corpus_dir = Path("corpus")
        if corpus_dir.exists():
            txt_files = list(corpus_dir.glob("*.txt"))
            if txt_files:
                info["available_datasets"].append({
                    "name": "Poetic Corpus",
                    "path": str(corpus_dir),
                    "type": "TXT",
                    "description": f"Poetry texts for negative control ({len(txt_files)} files)"
                })
        
        return info