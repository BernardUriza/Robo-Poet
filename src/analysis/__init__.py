"""
Analysis Module - Gradient Flow and Model Diagnostics
Created by Bernard Orozco

Enterprise architecture module for deep analysis of model behavior.
"""

from .gradient_analyzer_lite import GradientAnalyzerLite

# Only import full analyzer if pandas available
try:
    from .gradient_flow_analyzer import GradientFlowAnalyzer
    __all__ = ['GradientAnalyzerLite', 'GradientFlowAnalyzer']
except ImportError:
    __all__ = ['GradientAnalyzerLite']