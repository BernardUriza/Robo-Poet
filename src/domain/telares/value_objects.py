"""
Telares Value Objects
Immutable objects representing detection results and scores
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum


class RiskLevel(Enum):
    """Risk levels for manipulation detection"""
    LIMPIO = "LIMPIO"
    BAJO = "BAJO" 
    MODERADO = "MODERADO"
    ALTO = "ALTO"
    EXTREMO = "EXTREMO"


@dataclass(frozen=True)
class TacticScore:
    """Immutable score for a specific manipulation tactic"""
    tactic_name: str
    confidence: float  # 0.0 to 1.0
    evidence_keywords: List[str]
    
    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")
        
        if not self.tactic_name:
            raise ValueError("Tactic name cannot be empty")
    
    @property
    def is_detected(self) -> bool:
        """True if tactic is detected with high confidence"""
        return self.confidence > 0.5
    
    @property
    def strength(self) -> str:
        """Strength of detection"""
        if self.confidence >= 0.8:
            return "FUERTE"
        elif self.confidence >= 0.6:
            return "MODERADO"
        elif self.confidence >= 0.4:
            return "DÉBIL"
        else:
            return "MÍNIMO"


@dataclass(frozen=True)
class DetectionResult:
    """Complete detection result for a message"""
    message: str
    tactic_scores: Dict[str, TacticScore]
    overall_risk: RiskLevel
    confidence: float
    processing_time_ms: float
    model_version: str = "1.0"
    
    @property
    def detected_tactics(self) -> List[str]:
        """List of detected manipulation tactics"""
        return [
            name for name, score in self.tactic_scores.items() 
            if score.is_detected
        ]
    
    @property
    def total_manipulation_score(self) -> float:
        """Sum of all tactic scores"""
        return sum(score.confidence for score in self.tactic_scores.values())
    
    @property
    def is_pyramid_scheme(self) -> bool:
        """True if message shows pyramid scheme characteristics"""
        return self.overall_risk in [RiskLevel.ALTO, RiskLevel.EXTREMO]
    
    @property
    def requires_human_review(self) -> bool:
        """True if detection needs human verification"""
        return (
            self.overall_risk == RiskLevel.MODERADO or
            len(self.detected_tactics) >= 3 or
            self.confidence < 0.7
        )
    
    def get_alert_message(self) -> str:
        """Generate alert message for suspicious content"""
        if not self.detected_tactics:
            return "[OK] Mensaje limpio - sin manipulación detectada"
        
        tactics_str = ", ".join(self.detected_tactics)
        return f" ALERTA: Detectadas tácticas de manipulación: {tactics_str} (Riesgo: {self.overall_risk.value})"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "message": self.message[:100] + "..." if len(self.message) > 100 else self.message,
            "detected_tactics": self.detected_tactics,
            "total_score": self.total_manipulation_score,
            "risk_level": self.overall_risk.value,
            "confidence": self.confidence,
            "is_pyramid_scheme": self.is_pyramid_scheme,
            "requires_review": self.requires_human_review,
            "processing_time_ms": self.processing_time_ms,
            "model_version": self.model_version
        }