"""
Telares Detection Service
Main business logic for analyzing messages and detecting pyramid schemes
"""

import time
from typing import List, Optional, Tuple
from pathlib import Path

from src.domain.telares.entities import TelaresMessage, ManipulationTactics
from src.domain.telares.value_objects import DetectionResult, TacticScore, RiskLevel
from src.infrastructure.telares.ml_detector import TelaresMLDetector


class TelaresDetectionService:
    """
    Application service for pyramid scheme detection
    Orchestrates detection workflow using ML infrastructure
    """
    
    def __init__(self, ml_detector: Optional[TelaresMLDetector] = None):
        self.ml_detector = ml_detector or TelaresMLDetector()
        self.tactic_names = [
            "control_emocional",
            "presion_social", 
            "lenguaje_espiritual",
            "logica_circular",
            "urgencia_artificial", 
            "testimonio_fabricado",
            "promesa_irrealista"
        ]
    
    def analyze_message(self, message_content: str, platform: str = None) -> DetectionResult:
        """
        Analyze a message for pyramid scheme manipulation tactics
        
        Args:
            message_content: Text content to analyze
            platform: Platform where message was found (WhatsApp, Telegram, etc.)
            
        Returns:
            DetectionResult with detected tactics and risk assessment
        """
        start_time = time.time()
        
        # Create domain entity
        message = TelaresMessage(
            content=message_content,
            platform=platform
        )
        
        # Use ML detector to get predictions
        predictions = self.ml_detector.predict([message_content])
        prediction_scores = predictions[0] if predictions else [0.0] * 7
        
        # Convert to TacticScore value objects
        tactic_scores = {}
        for i, tactic_name in enumerate(self.tactic_names):
            score = prediction_scores[i] if i < len(prediction_scores) else 0.0
            evidence_keywords = self._extract_evidence_keywords(message_content, tactic_name)
            
            tactic_scores[tactic_name] = TacticScore(
                tactic_name=tactic_name,
                confidence=float(score),
                evidence_keywords=evidence_keywords
            )
        
        # Calculate overall risk
        total_score = sum(score.confidence for score in tactic_scores.values())
        overall_confidence = min(total_score / 7.0, 1.0)
        risk_level = self._calculate_risk_level(total_score)
        
        processing_time = (time.time() - start_time) * 1000
        
        return DetectionResult(
            message=message_content,
            tactic_scores=tactic_scores,
            overall_risk=risk_level,
            confidence=overall_confidence,
            processing_time_ms=processing_time,
            model_version=self.ml_detector.model_version
        )
    
    def analyze_batch(self, messages: List[str]) -> List[DetectionResult]:
        """Analyze multiple messages efficiently"""
        results = []
        
        # Batch prediction for efficiency
        batch_predictions = self.ml_detector.predict(messages)
        
        for i, message_content in enumerate(messages):
            prediction_scores = batch_predictions[i] if i < len(batch_predictions) else [0.0] * 7
            
            # Create result for each message
            result = self._create_detection_result(message_content, prediction_scores)
            results.append(result)
        
        return results
    
    def get_system_status(self) -> dict:
        """Get detection system status"""
        return {
            "detector_loaded": self.ml_detector.is_loaded(),
            "model_version": self.ml_detector.model_version,
            "supported_tactics": self.tactic_names,
            "total_tactics": len(self.tactic_names),
            "ready_for_detection": self.ml_detector.is_ready()
        }
    
    def _extract_evidence_keywords(self, message: str, tactic_name: str) -> List[str]:
        """Extract keywords that indicate specific manipulation tactic"""
        keywords_by_tactic = {
            "control_emocional": ["amor", "familia", "sueños", "futuro", "felicidad", "éxito"],
            "presion_social": ["todos", "ahora", "rápido", "últimos", "cupos", "oportunidad"],
            "lenguaje_espiritual": ["dios", "bendición", "fe", "destino", "universo", "energía"],
            "logica_circular": ["porque", "funciona", "simple", "fácil", "automático"],
            "urgencia_artificial": ["hoy", "ya", "ahora", "último", "pronto", "inmediato"],
            "testimonio_fabricado": ["logré", "gané", "cambió", "vida", "gracias", "increíble"],
            "promesa_irrealista": ["millones", "rico", "fácil", "sin", "trabajo", "pasivo"]
        }
        
        tactic_keywords = keywords_by_tactic.get(tactic_name, [])
        message_lower = message.lower()
        
        found_keywords = [
            keyword for keyword in tactic_keywords 
            if keyword in message_lower
        ]
        
        return found_keywords[:5]  # Return max 5 keywords
    
    def _calculate_risk_level(self, total_score: float) -> RiskLevel:
        """Calculate risk level based on total manipulation score"""
        if total_score >= 5.0:
            return RiskLevel.EXTREMO
        elif total_score >= 3.0:
            return RiskLevel.ALTO  
        elif total_score >= 1.0:
            return RiskLevel.MODERADO
        elif total_score > 0.5:
            return RiskLevel.BAJO
        else:
            return RiskLevel.LIMPIO
    
    def _create_detection_result(self, message_content: str, prediction_scores: List[float]) -> DetectionResult:
        """Create DetectionResult from message and ML predictions"""
        start_time = time.time()
        
        # Convert to TacticScore value objects
        tactic_scores = {}
        for i, tactic_name in enumerate(self.tactic_names):
            score = prediction_scores[i] if i < len(prediction_scores) else 0.0
            evidence_keywords = self._extract_evidence_keywords(message_content, tactic_name)
            
            tactic_scores[tactic_name] = TacticScore(
                tactic_name=tactic_name,
                confidence=float(score),
                evidence_keywords=evidence_keywords
            )
        
        # Calculate overall metrics
        total_score = sum(score.confidence for score in tactic_scores.values())
        overall_confidence = min(total_score / 7.0, 1.0)
        risk_level = self._calculate_risk_level(total_score)
        
        processing_time = (time.time() - start_time) * 1000
        
        return DetectionResult(
            message=message_content,
            tactic_scores=tactic_scores,
            overall_risk=risk_level,
            confidence=overall_confidence,
            processing_time_ms=processing_time,
            model_version=self.ml_detector.model_version
        )