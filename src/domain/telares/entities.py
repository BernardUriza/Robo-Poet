"""
Telares Domain Entities
Core business objects for pyramid scheme detection
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime


@dataclass
class ManipulationTactics:
    """7 manipulation tactics used in pyramid schemes"""
    control_emocional: float = 0.0
    presion_social: float = 0.0 
    lenguaje_espiritual: float = 0.0
    logica_circular: float = 0.0
    urgencia_artificial: float = 0.0
    testimonio_fabricado: float = 0.0
    promesa_irrealista: float = 0.0
    
    @property
    def total_score(self) -> float:
        """Total manipulation score (0-7)"""
        return sum([
            self.control_emocional,
            self.presion_social,
            self.lenguaje_espiritual, 
            self.logica_circular,
            self.urgencia_artificial,
            self.testimonio_fabricado,
            self.promesa_irrealista
        ])
    
    @property
    def is_manipulative(self) -> bool:
        """True if any manipulation tactic detected"""
        return self.total_score > 0.5
    
    @property
    def risk_level(self) -> str:
        """Risk assessment based on total score"""
        score = self.total_score
        if score >= 5.0:
            return "EXTREMO"
        elif score >= 3.0:
            return "ALTO"
        elif score >= 1.0:
            return "MODERADO"
        elif score > 0.0:
            return "BAJO"
        else:
            return "LIMPIO"


@dataclass
class TelaresMessage:
    """A message that can be analyzed for pyramid scheme manipulation"""
    content: str
    platform: Optional[str] = None
    region: Optional[str] = None
    fecha: Optional[datetime] = None
    manipulation_tactics: Optional[ManipulationTactics] = None
    
    def __post_init__(self):
        if not self.content or not self.content.strip():
            raise ValueError("Message content cannot be empty")
        
        if self.manipulation_tactics is None:
            self.manipulation_tactics = ManipulationTactics()
    
    @property
    def is_suspicious(self) -> bool:
        """Quick check if message shows manipulation signs"""
        return self.manipulation_tactics.is_manipulative
    
    @property
    def word_count(self) -> int:
        """Number of words in message"""
        return len(self.content.split())
    
    @property
    def char_count(self) -> int:
        """Number of characters in message"""
        return len(self.content)
    
    def get_detected_tactics(self) -> List[str]:
        """List of detected manipulation tactics"""
        tactics = []
        if self.manipulation_tactics.control_emocional > 0.5:
            tactics.append("Control Emocional")
        if self.manipulation_tactics.presion_social > 0.5:
            tactics.append("Presión Social")
        if self.manipulation_tactics.lenguaje_espiritual > 0.5:
            tactics.append("Lenguaje Espiritual")
        if self.manipulation_tactics.logica_circular > 0.5:
            tactics.append("Lógica Circular")
        if self.manipulation_tactics.urgencia_artificial > 0.5:
            tactics.append("Urgencia Artificial")
        if self.manipulation_tactics.testimonio_fabricado > 0.5:
            tactics.append("Testimonio Fabricado")
        if self.manipulation_tactics.promesa_irrealista > 0.5:
            tactics.append("Promesa Irrealista")
        return tactics