"""
Métricas de evaluación para generación de texto.

Implementa BLEU, Perplexity, diversidad de n-gramas y otras métricas
para evaluación automática de calidad del modelo.
"""

import math
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import Counter, defaultdict
import numpy as np

# Importaciones opcionales con fallbacks
try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize
    from nltk.util import ngrams
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from sacrebleu import sentence_bleu as sacre_sentence_bleu
    SACREBLEU_AVAILABLE = True
except ImportError:
    SACREBLEU_AVAILABLE = False

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False


logger = logging.getLogger(__name__)


@dataclass
class EvaluationResults:
    """Resultados de evaluación con todas las métricas."""
    
    # Métricas principales
    bleu_score: float = 0.0
    perplexity: float = float('inf')
    rouge_1: float = 0.0
    rouge_2: float = 0.0
    rouge_l: float = 0.0
    
    # Diversidad de n-gramas
    unigram_diversity: float = 0.0
    bigram_diversity: float = 0.0
    trigram_diversity: float = 0.0
    
    # Métricas adicionales
    repetition_ratio: float = 0.0
    avg_sentence_length: float = 0.0
    vocabulary_usage: float = 0.0
    
    # Metadatos
    num_samples: int = 0
    evaluation_time: float = 0.0
    timestamp: Optional[str] = None
    
    # Métricas detalladas
    detailed_scores: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario para serialización."""
        return {
            'bleu_score': self.bleu_score,
            'perplexity': self.perplexity,
            'rouge_1': self.rouge_1,
            'rouge_2': self.rouge_2,
            'rouge_l': self.rouge_l,
            'unigram_diversity': self.unigram_diversity,
            'bigram_diversity': self.bigram_diversity,
            'trigram_diversity': self.trigram_diversity,
            'repetition_ratio': self.repetition_ratio,
            'avg_sentence_length': self.avg_sentence_length,
            'vocabulary_usage': self.vocabulary_usage,
            'num_samples': self.num_samples,
            'evaluation_time': self.evaluation_time,
            'timestamp': self.timestamp,
            'detailed_scores': self.detailed_scores
        }
    
    def get_summary_score(self) -> float:
        """Calcula un score resumen combinando múltiples métricas."""
        # Normalizar perplexity (menor es mejor)
        normalized_perplexity = max(0, 1 - (self.perplexity - 1) / 100)
        
        # Combinar métricas con pesos
        weights = {
            'bleu': 0.3,
            'rouge': 0.2,
            'perplexity': 0.3,
            'diversity': 0.2
        }
        
        rouge_avg = (self.rouge_1 + self.rouge_2 + self.rouge_l) / 3
        diversity_avg = (self.unigram_diversity + self.bigram_diversity + self.trigram_diversity) / 3
        
        summary = (
            weights['bleu'] * self.bleu_score +
            weights['rouge'] * rouge_avg +
            weights['perplexity'] * normalized_perplexity +
            weights['diversity'] * diversity_avg
        )
        
        return max(0, min(1, summary))


class BLEUMetric:
    """Implementación de métrica BLEU para evaluación de texto generado."""
    
    def __init__(self, max_n: int = 4, smoothing: bool = True):
        self.max_n = max_n
        self.smoothing = smoothing
        
        if NLTK_AVAILABLE:
            # Descargar recursos de NLTK si no están disponibles
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                logger.info("Downloading NLTK punkt tokenizer...")
                nltk.download('punkt', quiet=True)
        
        if not NLTK_AVAILABLE and not SACREBLEU_AVAILABLE:
            logger.warning("Neither NLTK nor SacreBLEU available. BLEU scores will be approximated.")
    
    def calculate_sentence_bleu(self, reference: str, candidate: str) -> float:
        """Calcula BLEU score para una oración."""
        if SACREBLEU_AVAILABLE:
            return self._sacrebleu_sentence(reference, candidate)
        elif NLTK_AVAILABLE:
            return self._nltk_sentence_bleu(reference, candidate)
        else:
            return self._approximate_bleu(reference, candidate)
    
    def _sacrebleu_sentence(self, reference: str, candidate: str) -> float:
        """Implementación usando SacreBLEU."""
        try:
            score = sacre_sentence_bleu(candidate, [reference])
            return score.score / 100.0  # SacreBLEU devuelve 0-100
        except Exception as e:
            logger.warning(f"SacreBLEU error: {e}")
            return 0.0
    
    def _nltk_sentence_bleu(self, reference: str, candidate: str) -> float:
        """Implementación usando NLTK."""
        try:
            ref_tokens = word_tokenize(reference.lower())
            cand_tokens = word_tokenize(candidate.lower())
            
            # Usar suavizado si está habilitado
            smoothing_function = SmoothingFunction().method1 if self.smoothing else None
            
            weights = tuple([1.0/self.max_n] * self.max_n)
            
            score = sentence_bleu(
                [ref_tokens], 
                cand_tokens, 
                weights=weights,
                smoothing_function=smoothing_function
            )
            
            return score
            
        except Exception as e:
            logger.warning(f"NLTK BLEU error: {e}")
            return 0.0
    
    def _approximate_bleu(self, reference: str, candidate: str) -> float:
        """Aproximación simple de BLEU sin dependencias externas."""
        ref_words = reference.lower().split()
        cand_words = candidate.lower().split()
        
        if not cand_words:
            return 0.0
        
        # Calcular precisión de n-gramas simple
        scores = []
        for n in range(1, min(self.max_n + 1, len(cand_words) + 1)):
            ref_ngrams = Counter([' '.join(ref_words[i:i+n]) for i in range(len(ref_words)-n+1)])
            cand_ngrams = Counter([' '.join(cand_words[i:i+n]) for i in range(len(cand_words)-n+1)])
            
            if not cand_ngrams:
                scores.append(0.0)
                continue
            
            matches = sum((ref_ngrams & cand_ngrams).values())
            total = sum(cand_ngrams.values())
            precision = matches / total if total > 0 else 0.0
            scores.append(precision)
        
        if not scores:
            return 0.0
        
        # Promedio geométrico
        geometric_mean = np.exp(np.mean(np.log(np.maximum(scores, 1e-10))))
        
        # Penalización por brevedad simple
        brevity_penalty = min(1.0, len(cand_words) / max(len(ref_words), 1))
        
        return geometric_mean * brevity_penalty
    
    def calculate_corpus_bleu(self, references: List[str], candidates: List[str]) -> float:
        """Calcula BLEU score para un corpus completo."""
        if len(references) != len(candidates):
            raise ValueError("References and candidates must have same length")
        
        scores = []
        for ref, cand in zip(references, candidates):
            score = self.calculate_sentence_bleu(ref, cand)
            scores.append(score)
        
        return np.mean(scores) if scores else 0.0


class PerplexityMetric:
    """Implementación de métrica de perplejidad."""
    
    def __init__(self):
        self.epsilon = 1e-10  # Para evitar log(0)
    
    def calculate_perplexity(self, probabilities: np.ndarray) -> float:
        """
        Calcula perplejidad a partir de probabilidades del modelo.
        
        Args:
            probabilities: Array de probabilidades por token
            
        Returns:
            Valor de perplejidad
        """
        if len(probabilities) == 0:
            return float('inf')
        
        # Asegurar que las probabilidades están en rango válido
        probabilities = np.clip(probabilities, self.epsilon, 1.0)
        
        # Calcular log-probabilidad promedio
        log_prob = np.mean(np.log2(probabilities))
        
        # Perplejidad = 2^(-log_prob)
        perplexity = 2 ** (-log_prob)
        
        return float(perplexity)
    
    def calculate_cross_entropy_loss(self, probabilities: np.ndarray) -> float:
        """Calcula cross-entropy loss."""
        probabilities = np.clip(probabilities, self.epsilon, 1.0)
        return -np.mean(np.log(probabilities))
    
    def calculate_from_loss(self, cross_entropy_loss: float) -> float:
        """Calcula perplejidad a partir de cross-entropy loss."""
        return math.exp(cross_entropy_loss)


class NGramDiversityMetric:
    """Métrica de diversidad de n-gramas para medir repetitividad."""
    
    def __init__(self, max_n: int = 3):
        self.max_n = max_n
    
    def calculate_diversity(self, texts: List[str], n: int = 2) -> float:
        """
        Calcula diversidad de n-gramas.
        
        Args:
            texts: Lista de textos generados
            n: Tamaño del n-grama
            
        Returns:
            Ratio de n-gramas únicos vs total
        """
        if not texts:
            return 0.0
        
        all_ngrams = []
        
        for text in texts:
            tokens = text.lower().split()
            if len(tokens) >= n:
                text_ngrams = [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
                all_ngrams.extend(text_ngrams)
        
        if not all_ngrams:
            return 0.0
        
        unique_ngrams = len(set(all_ngrams))
        total_ngrams = len(all_ngrams)
        
        return unique_ngrams / total_ngrams
    
    def calculate_all_diversities(self, texts: List[str]) -> Dict[str, float]:
        """Calcula diversidad para todos los n-gramas hasta max_n."""
        diversities = {}
        
        for n in range(1, self.max_n + 1):
            diversity = self.calculate_diversity(texts, n)
            diversities[f'{n}gram_diversity'] = diversity
        
        return diversities
    
    def calculate_repetition_ratio(self, text: str, n: int = 2) -> float:
        """Calcula ratio de repetición de n-gramas en un texto."""
        tokens = text.lower().split()
        
        if len(tokens) < n:
            return 0.0
        
        ngrams = [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        
        if not ngrams:
            return 0.0
        
        ngram_counts = Counter(ngrams)
        repeated_ngrams = sum(count - 1 for count in ngram_counts.values() if count > 1)
        
        return repeated_ngrams / len(ngrams)


class ROUGEMetric:
    """Implementación de métrica ROUGE."""
    
    def __init__(self):
        self.available = ROUGE_AVAILABLE
        if ROUGE_AVAILABLE:
            self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        else:
            logger.warning("ROUGE scorer not available. Will use approximation.")
    
    def calculate_rouge(self, reference: str, candidate: str) -> Dict[str, float]:
        """Calcula scores ROUGE-1, ROUGE-2 y ROUGE-L."""
        if self.available:
            return self._calculate_rouge_official(reference, candidate)
        else:
            return self._calculate_rouge_approximate(reference, candidate)
    
    def _calculate_rouge_official(self, reference: str, candidate: str) -> Dict[str, float]:
        """Implementación oficial usando rouge-score."""
        try:
            scores = self.scorer.score(reference, candidate)
            return {
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure
            }
        except Exception as e:
            logger.warning(f"ROUGE calculation error: {e}")
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    def _calculate_rouge_approximate(self, reference: str, candidate: str) -> Dict[str, float]:
        """Aproximación simple de ROUGE."""
        ref_words = reference.lower().split()
        cand_words = candidate.lower().split()
        
        if not cand_words or not ref_words:
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
        
        # ROUGE-1 (overlap de unigramas)
        ref_unigrams = set(ref_words)
        cand_unigrams = set(cand_words)
        rouge1 = len(ref_unigrams & cand_unigrams) / len(ref_unigrams | cand_unigrams)
        
        # ROUGE-2 (overlap de bigramas)
        ref_bigrams = set([' '.join(ref_words[i:i+2]) for i in range(len(ref_words)-1)])
        cand_bigrams = set([' '.join(cand_words[i:i+2]) for i in range(len(cand_words)-1)])
        rouge2 = len(ref_bigrams & cand_bigrams) / max(len(ref_bigrams | cand_bigrams), 1)
        
        # ROUGE-L (LCS aproximado)
        rougeL = self._approximate_lcs(ref_words, cand_words)
        
        return {'rouge1': rouge1, 'rouge2': rouge2, 'rougeL': rougeL}
    
    def _approximate_lcs(self, ref_words: List[str], cand_words: List[str]) -> float:
        """Aproximación de Longest Common Subsequence."""
        if not ref_words or not cand_words:
            return 0.0
        
        # Matriz DP para LCS
        m, n = len(ref_words), len(cand_words)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_words[i-1] == cand_words[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        lcs_length = dp[m][n]
        
        # F-measure basado en precisión y recall
        precision = lcs_length / n if n > 0 else 0
        recall = lcs_length / m if m > 0 else 0
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * precision * recall / (precision + recall)


class MetricCalculator:
    """Calculadora principal que combina todas las métricas."""
    
    def __init__(self):
        self.bleu_metric = BLEUMetric()
        self.perplexity_metric = PerplexityMetric()
        self.diversity_metric = NGramDiversityMetric()
        self.rouge_metric = ROUGEMetric()
    
    def evaluate_generation(
        self,
        references: List[str],
        candidates: List[str],
        probabilities: Optional[List[np.ndarray]] = None
    ) -> EvaluationResults:
        """
        Evaluación completa de generación de texto.
        
        Args:
            references: Textos de referencia
            candidates: Textos generados por el modelo
            probabilities: Probabilidades del modelo (opcional)
            
        Returns:
            Resultados de evaluación completos
        """
        if len(references) != len(candidates):
            raise ValueError("References and candidates must have same length")
        
        results = EvaluationResults()
        results.num_samples = len(candidates)
        
        # BLEU Score
        try:
            results.bleu_score = self.bleu_metric.calculate_corpus_bleu(references, candidates)
        except Exception as e:
            logger.warning(f"BLEU calculation failed: {e}")
            results.bleu_score = 0.0
        
        # ROUGE Scores
        try:
            rouge_scores = []
            for ref, cand in zip(references, candidates):
                rouge = self.rouge_metric.calculate_rouge(ref, cand)
                rouge_scores.append(rouge)
            
            if rouge_scores:
                results.rouge_1 = np.mean([s['rouge1'] for s in rouge_scores])
                results.rouge_2 = np.mean([s['rouge2'] for s in rouge_scores])
                results.rouge_l = np.mean([s['rougeL'] for s in rouge_scores])
        except Exception as e:
            logger.warning(f"ROUGE calculation failed: {e}")
        
        # Diversidad de n-gramas
        try:
            diversities = self.diversity_metric.calculate_all_diversities(candidates)
            results.unigram_diversity = diversities.get('1gram_diversity', 0.0)
            results.bigram_diversity = diversities.get('2gram_diversity', 0.0)
            results.trigram_diversity = diversities.get('3gram_diversity', 0.0)
        except Exception as e:
            logger.warning(f"Diversity calculation failed: {e}")
        
        # Perplejidad (si se proporcionan probabilidades)
        if probabilities:
            try:
                all_probs = np.concatenate(probabilities)
                results.perplexity = self.perplexity_metric.calculate_perplexity(all_probs)
            except Exception as e:
                logger.warning(f"Perplexity calculation failed: {e}")
        
        # Métricas adicionales
        try:
            # Ratio de repetición promedio
            repetition_ratios = [
                self.diversity_metric.calculate_repetition_ratio(cand) 
                for cand in candidates
            ]
            results.repetition_ratio = np.mean(repetition_ratios)
            
            # Longitud promedio de oración
            sentence_lengths = [len(cand.split()) for cand in candidates]
            results.avg_sentence_length = np.mean(sentence_lengths)
            
            # Uso de vocabulario (palabras únicas / total)
            all_words = ' '.join(candidates).split()
            if all_words:
                results.vocabulary_usage = len(set(all_words)) / len(all_words)
            
        except Exception as e:
            logger.warning(f"Additional metrics calculation failed: {e}")
        
        # Timestamp
        from datetime import datetime
        results.timestamp = datetime.now().isoformat()
        
        return results
    
    def quick_evaluate(self, reference: str, candidate: str) -> Dict[str, float]:
        """Evaluación rápida para una sola oración."""
        bleu = self.bleu_metric.calculate_sentence_bleu(reference, candidate)
        rouge = self.rouge_metric.calculate_rouge(reference, candidate)
        
        return {
            'bleu': bleu,
            'rouge1': rouge['rouge1'],
            'rouge2': rouge['rouge2'],
            'rougeL': rouge['rougeL']
        }