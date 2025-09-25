"""
Constructor de vocabulario avanzado para Robo-Poet.

Implementa técnicas modernas de tokenización incluyendo BPE (Byte-Pair Encoding)
simplificado para expandir el vocabulario de 44 caracteres a 5000+ tokens.
Parte de optimizaciones Fase 3.
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass
import unicodedata

from src.logger import RoboPoetLogger
from src.i18n import Messages as Msg

# Logger específico para vocabulario
logger = RoboPoetLogger.get_logger('vocabulary')


@dataclass
class VocabularyStats:
    """Estadísticas de construcción de vocabulario."""
    original_chars: int
    expanded_tokens: int
    coverage_percent: float
    most_frequent_tokens: List[Tuple[str, int]]
    rare_tokens: List[Tuple[str, int]]
    total_text_length: int
    unique_tokens_used: int


class EnhancedVocabularyBuilder:
    """
    Constructor de vocabulario mejorado con técnicas BPE simplificadas.
    
    Expande vocabulario de caracteres a subwords/tokens para mejor
    expresividad y manejo de palabras raras.
    """
    
    def __init__(self, min_freq: int = 5, max_vocab_size: int = 5000):
        """
        Inicializa constructor de vocabulario.
        
        Args:
            min_freq: Frecuencia mínima para incluir token
            max_vocab_size: Tamaño máximo del vocabulario
        """
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        self.special_tokens = ['<PAD>', '<UNK>', '<START>', '<END>']
        self.vocab: Dict[str, int] = {}
        self.reverse_vocab: Dict[int, str] = {}
        self.token_frequencies: Counter = Counter()
        
        logger.info(Msg.CONFIG_LOADING)
        logger.info(f"Configuración vocabulario: max_size={max_vocab_size}, min_freq={min_freq}")
    
    def build_character_vocab(self, text: str) -> Dict[str, int]:
        """
        Construye vocabulario básico a nivel de caracteres expandido.
        
        Args:
            text: Texto para análisis
            
        Returns:
            Dict: Mapeo carácter -> índice
        """
        logger.info("[BOOKS] Construyendo vocabulario de caracteres expandido...")
        
        # Caracteres básicos del texto
        chars = set(text)
        logger.info(f"Caracteres únicos encontrados: {len(chars)}")
        
        # Agregar caracteres especiales importantes para mejor expresividad
        special_chars = {
            # Whitespace y control
            '\n', '\t', ' ', '\r',
            # Puntuación básica
            '.', ',', '!', '?', ';', ':',
            # Quotes y guiones
            '"', "'", '-', '—', '–',
            # Paréntesis y brackets
            '(', ')', '[', ']', '{', '}',
            # Números
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            # Símbolos comunes
            '@', '#', '$', '%', '&', '*', '+', '=', '|', '\\', '/', '^', '~'
        }
        
        # Combinar caracteres del texto con especiales
        all_chars = sorted(list(chars | special_chars))
        logger.info(f"Total caracteres (incluyendo especiales): {len(all_chars)}")
        
        # Construir vocabulario con tokens especiales primero
        vocab = {}
        
        # Tokens especiales tienen precedencia
        for i, token in enumerate(self.special_tokens):
            vocab[token] = i
        
        # Agregar caracteres
        for i, char in enumerate(all_chars):
            if char not in vocab:  # Evitar duplicados con especiales
                vocab[char] = len(vocab)
        
        logger.info(f"[OK] Vocabulario de caracteres construido: {len(vocab)} tokens")
        return vocab
    
    def build_subword_vocab(self, text: str, num_merges: int = 1000) -> Dict[str, int]:
        """
        Construye vocabulario de subpalabras usando BPE simplificado.
        
        Args:
            text: Texto para análisis
            num_merges: Número de operaciones de merge BPE
            
        Returns:
            Dict: Vocabulario expandido carácter+subword -> índice
        """
        logger.info("[FIX] Iniciando construcción BPE de vocabulario...")
        logger.info(f"Texto de entrada: {len(text):,} caracteres")
        
        # Paso 1: Inicializar con vocabulario de caracteres
        vocab = self.build_character_vocab(text)
        logger.info(f"Vocabulario inicial: {len(vocab)} tokens")
        
        # Paso 2: Obtener frecuencias de palabras
        word_freq = self._get_word_frequencies(text)
        logger.info(f"Palabras únicas procesadas: {len(word_freq):,}")
        
        # Paso 3: Aplicar BPE merges iterativamente
        logger.info(f"Aplicando {num_merges} operaciones BPE...")
        
        for merge_num in range(num_merges):
            if len(vocab) >= self.max_vocab_size:
                logger.info(f"WARNING: Límite de vocabulario alcanzado: {len(vocab)}")
                break
            
            # Obtener pares de símbolos más frecuentes
            pairs = self._get_pairs(word_freq)
            if not pairs:
                logger.info(f"No hay más pares para fusionar. Deteniendo en merge {merge_num}")
                break
            
            # Encontrar el par más frecuente
            best_pair = max(pairs, key=pairs.get)
            
            # Solo proceder si el par es suficientemente frecuente
            if pairs[best_pair] < self.min_freq:
                logger.info(f"Frecuencia demasiado baja ({pairs[best_pair]}). Deteniendo BPE")
                break
            
            # Fusionar en vocabulario
            new_token = ''.join(best_pair)
            vocab[new_token] = len(vocab)
            
            # Actualizar frecuencias de palabras
            word_freq = self._merge_pair(best_pair, word_freq)
            
            # Log progreso cada 100 merges
            if merge_num % 100 == 0 and merge_num > 0:
                logger.info(f"Merge {merge_num:3d}: '{new_token}' (freq: {pairs[best_pair]})")
        
        logger.info(f"[OK] BPE completado: {len(vocab)} tokens totales")
        
        # Guardar estadísticas
        self.vocab = vocab
        self.reverse_vocab = {v: k for k, v in vocab.items()}
        
        return vocab
    
    def _get_word_frequencies(self, text: str) -> Dict[str, int]:
        """
        Obtiene frecuencias de palabras preparadas para BPE.
        
        Args:
            text: Texto de entrada
            
        Returns:
            Dict: Palabra segmentada -> frecuencia
        """
        # Normalizar texto
        text = self._normalize_text(text)
        
        # Dividir en palabras (considerando puntuación)
        words = re.findall(r'\b\w+\b|\S', text)
        word_counts = Counter(words)
        
        logger.info(f"Palabras únicas antes de segmentación: {len(word_counts)}")
        
        # Preparar para BPE: cada palabra como secuencia de caracteres + marcador de fin
        word_freq = {}
        for word, count in word_counts.items():
            # Separar caracteres y agregar marcador de fin de palabra
            segmented = ' '.join(list(word)) + ' </w>'
            word_freq[segmented] = count
        
        return word_freq
    
    def _normalize_text(self, text: str) -> str:
        """
        Normaliza texto para procesamiento uniforme.
        
        Args:
            text: Texto crudo
            
        Returns:
            str: Texto normalizado
        """
        # Normalización Unicode (NFC)
        text = unicodedata.normalize('NFC', text)
        
        # Reemplazar caracteres problemáticos
        replacements = {
            '"': '"', '"': '"',  # Quotes inteligentes
            ''': "'", ''': "'",  # Apostrofes inteligentes
            '…': '...',          # Elipsis
            '–': '-', '—': '--', # Guiones
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Normalizar espacios múltiples
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _get_pairs(self, word_freq: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        """
        Obtiene pares de símbolos consecutivos y sus frecuencias.
        
        Args:
            word_freq: Frecuencias de palabras segmentadas
            
        Returns:
            Dict: Par de símbolos -> frecuencia
        """
        pairs = defaultdict(int)
        
        for word, freq in word_freq.items():
            symbols = word.split()
            
            # Obtener todos los pares consecutivos
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pairs[pair] += freq
        
        return dict(pairs)
    
    def _merge_pair(self, pair: Tuple[str, str], word_freq: Dict[str, int]) -> Dict[str, int]:
        """
        Fusiona un par específico en todas las palabras donde aparece.
        
        Args:
            pair: Par de símbolos a fusionar
            word_freq: Frecuencias de palabras actuales
            
        Returns:
            Dict: Frecuencias actualizadas después del merge
        """
        new_word_freq = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        
        for word, freq in word_freq.items():
            # Reemplazar el bigram por su fusión
            new_word = word.replace(bigram, replacement)
            new_word_freq[new_word] = freq
        
        return new_word_freq
    
    def tokenize_text(self, text: str, vocab: Optional[Dict[str, int]] = None) -> List[int]:
        """
        Tokeniza texto usando el vocabulario construido.
        
        Args:
            text: Texto a tokenizar
            vocab: Vocabulario a usar (usa self.vocab si None)
            
        Returns:
            List[int]: Lista de IDs de tokens
        """
        if vocab is None:
            vocab = self.vocab
        
        if not vocab:
            raise ValueError("Vocabulario no construido. Llama build_subword_vocab() primero")
        
        # Normalizar texto
        text = self._normalize_text(text)
        
        tokens = []
        unk_id = vocab.get('<UNK>', 1)  # ID del token desconocido
        
        # Tokenización simple: intentar encontrar la secuencia más larga posible
        i = 0
        while i < len(text):
            # Intentar encontrar el token más largo que coincida
            found_token = None
            max_len = 0
            
            # Buscar desde la secuencia más larga hasta caracteres individuales
            for length in range(min(50, len(text) - i), 0, -1):
                candidate = text[i:i+length]
                if candidate in vocab:
                    found_token = candidate
                    max_len = length
                    break
            
            if found_token:
                tokens.append(vocab[found_token])
                i += max_len
            else:
                # Si no se encuentra, usar UNK para carácter individual
                tokens.append(unk_id)
                i += 1
        
        return tokens
    
    def detokenize(self, token_ids: List[int]) -> str:
        """
        Convierte lista de IDs de tokens de vuelta a texto.
        
        Args:
            token_ids: Lista de IDs de tokens
            
        Returns:
            str: Texto reconstruido
        """
        if not self.reverse_vocab:
            raise ValueError("Vocabulario reverso no disponible")
        
        tokens = []
        for token_id in token_ids:
            token = self.reverse_vocab.get(token_id, '<UNK>')
            tokens.append(token)
        
        # Reconstruir texto
        text = ''.join(tokens)
        
        # Limpiar marcadores de fin de palabra
        text = text.replace('</w>', ' ')
        
        # Normalizar espacios
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def get_vocabulary_stats(self, text: str) -> VocabularyStats:
        """
        Genera estadísticas del vocabulario construido.
        
        Args:
            text: Texto de referencia para estadísticas
            
        Returns:
            VocabularyStats: Estadísticas detalladas
        """
        if not self.vocab:
            raise ValueError("Vocabulario no construido")
        
        # Tokenizar texto para análisis
        tokens = self.tokenize_text(text)
        token_counts = Counter(tokens)
        
        # Calcular cobertura
        total_tokens = len(tokens)
        unique_tokens = len(set(tokens))
        coverage = (unique_tokens / len(self.vocab)) * 100 if self.vocab else 0
        
        # Obtener tokens más y menos frecuentes
        most_frequent = []
        rare_tokens = []
        
        # Convertir conteos de IDs a tokens
        token_name_counts = []
        for token_id, count in token_counts.items():
            token_name = self.reverse_vocab.get(token_id, '<UNK>')
            token_name_counts.append((token_name, count))
        
        # Ordenar por frecuencia
        token_name_counts.sort(key=lambda x: x[1], reverse=True)
        most_frequent = token_name_counts[:10]
        rare_tokens = token_name_counts[-10:] if len(token_name_counts) > 10 else []
        
        return VocabularyStats(
            original_chars=len(set(text)),
            expanded_tokens=len(self.vocab),
            coverage_percent=coverage,
            most_frequent_tokens=most_frequent,
            rare_tokens=rare_tokens,
            total_text_length=len(text),
            unique_tokens_used=unique_tokens
        )
    
    def save_vocabulary(self, filepath: Path) -> None:
        """
        Guarda vocabulario a archivo JSON.
        
        Args:
            filepath: Ruta donde guardar el vocabulario
        """
        vocab_data = {
            'vocab': self.vocab,
            'reverse_vocab': {str(k): v for k, v in self.reverse_vocab.items()},
            'config': {
                'min_freq': self.min_freq,
                'max_vocab_size': self.max_vocab_size,
                'special_tokens': self.special_tokens
            },
            'stats': {
                'vocab_size': len(self.vocab),
                'created': str(Path.cwd()),
            }
        }
        
        filepath.parent.mkdir(exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"[SAVE] Vocabulario guardado: {filepath}")
    
    def load_vocabulary(self, filepath: Path) -> None:
        """
        Carga vocabulario desde archivo JSON.
        
        Args:
            filepath: Ruta del archivo de vocabulario
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.vocab = vocab_data['vocab']
        self.reverse_vocab = {int(k): v for k, v in vocab_data['reverse_vocab'].items()}
        
        config = vocab_data.get('config', {})
        self.min_freq = config.get('min_freq', self.min_freq)
        self.max_vocab_size = config.get('max_vocab_size', self.max_vocab_size)
        self.special_tokens = config.get('special_tokens', self.special_tokens)
        
        logger.info(f" Vocabulario cargado: {filepath} ({len(self.vocab)} tokens)")


def expand_vocabulary_for_model(text_file: str, output_vocab: str = "vocab_5000.json") -> VocabularyStats:
    """
    Función de conveniencia para expandir vocabulario del modelo principal.
    
    Args:
        text_file: Archivo de texto para construir vocabulario
        output_vocab: Archivo de salida del vocabulario
        
    Returns:
        VocabularyStats: Estadísticas de expansión
    """
    logger.info("[LAUNCH] Iniciando expansión de vocabulario para modelo principal")
    
    # Leer archivo de texto
    text_path = Path(text_file)
    if not text_path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {text_file}")
    
    logger.info(f" Leyendo corpus: {text_file}")
    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    logger.info(f"Corpus cargado: {len(text):,} caracteres")
    
    # Construir vocabulario expandido
    builder = EnhancedVocabularyBuilder(min_freq=3, max_vocab_size=5000)
    vocab = builder.build_subword_vocab(text, num_merges=2000)
    
    # Generar estadísticas
    stats = builder.get_vocabulary_stats(text)
    
    # Guardar vocabulario
    output_path = Path(output_vocab)
    builder.save_vocabulary(output_path)
    
    # Log estadísticas finales
    logger.info("=" * 50)
    logger.info("[CHART] EXPANSIÓN DE VOCABULARIO COMPLETADA")
    logger.info("=" * 50)
    logger.info(f"Vocabulario original: {stats.original_chars} caracteres")
    logger.info(f"Vocabulario expandido: {stats.expanded_tokens} tokens")
    logger.info(f"Expansión: {stats.expanded_tokens / stats.original_chars:.1f}x")
    logger.info(f"Cobertura: {stats.coverage_percent:.1f}%")
    logger.info(f"Tokens únicos usados: {stats.unique_tokens_used}")
    logger.info("=" * 50)
    
    return stats


if __name__ == "__main__":
    """Test del constructor de vocabulario."""
    import sys
    
    print(" PROBANDO CONSTRUCTOR DE VOCABULARIO")
    print("="*50)
    
    # Texto de prueba
    test_text = """
    The power of language and the language of power are fundamental concepts
    in understanding human communication. Words have the ability to transform
    ideas, influence minds, and shape reality. This is particularly true in
    academic and scientific discourse, where precise vocabulary is essential.
    """
    
    # Construir vocabulario
    builder = EnhancedVocabularyBuilder(min_freq=1, max_vocab_size=100)
    
    # Test vocabulario de caracteres
    char_vocab = builder.build_character_vocab(test_text)
    print(f"[BOOKS] Vocabulario de caracteres: {len(char_vocab)} tokens")
    print(f"Tokens especiales: {builder.special_tokens}")
    
    # Test BPE
    subword_vocab = builder.build_subword_vocab(test_text, num_merges=20)
    print(f"[FIX] Vocabulario BPE: {len(subword_vocab)} tokens")
    
    # Test tokenización
    sample = "The power of language"
    tokens = builder.tokenize_text(sample)
    detokenized = builder.detokenize(tokens)
    print(f"[DOC] Original: '{sample}'")
    print(f" Tokens: {tokens}")
    print(f"[DOC] Detokenizado: '{detokenized}'")
    
    # Estadísticas
    stats = builder.get_vocabulary_stats(test_text)
    print(f"\n[CHART] Estadísticas:")
    print(f"   Caracteres únicos: {stats.original_chars}")
    print(f"   Tokens expandidos: {stats.expanded_tokens}")
    print(f"   Cobertura: {stats.coverage_percent:.1f}%")
    print(f"   Tokens más frecuentes: {stats.most_frequent_tokens[:3]}")
    
    # Test con archivo real si está disponible
    test_file = "The+48+Laws+Of+Power_texto.txt"
    if Path(test_file).exists():
        print(f"\n[LAUNCH] Probando con archivo real: {test_file}")
        try:
            stats = expand_vocabulary_for_model(test_file, "test_vocab.json")
            print(f"[OK] Expansión exitosa: {stats.expanded_tokens} tokens")
        except Exception as e:
            print(f"[X] Error: {e}")
    else:
        print(f"\nWARNING: Archivo de prueba no encontrado: {test_file}")
        print("   (Esto es normal si no estás en el directorio del proyecto)")