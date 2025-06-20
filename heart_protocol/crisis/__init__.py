"""
Crisis Escalation & Human Handoff System

Critical safety infrastructure that ensures users in crisis receive immediate
professional support while maintaining dignity, privacy, and autonomy.

Core Philosophy: "In crisis, every second of care matters."
"""

from .escalation_engine import EscalationEngine, CrisisLevel, EscalationType
from .human_handoff import HumanHandoffManager, HandoffType, HandoffStatus  
from .crisis_protocols import CrisisProtocolManager, ProtocolType, InterventionLevel
from .safety_monitoring import SafetyMonitor, SafetyAlert, MonitoringLevel

__all__ = [
    'EscalationEngine',
    'CrisisLevel', 
    'EscalationType',
    'HumanHandoffManager',
    'HandoffType',
    'HandoffStatus',
    'CrisisProtocolManager',
    'ProtocolType', 
    'InterventionLevel',
    'SafetyMonitor',
    'SafetyAlert',
    'MonitoringLevel'
]