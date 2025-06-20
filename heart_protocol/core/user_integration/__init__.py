"""
User Preference Integration System

Comprehensive integration system that connects all user preference and personalization
systems across the Heart Protocol, ensuring consistent and healing-focused experiences.

Core Philosophy: "Your preferences shape your healing journey."
"""

from .preference_integration_engine import PreferenceIntegrationEngine, IntegrationType, PreferenceScope
from .personalization_coordinator import PersonalizationCoordinator, PersonalizationLevel, AdaptationStrategy
from .user_experience_optimizer import UserExperienceOptimizer, ExperienceMetric, OptimizationGoal
from .cross_system_synchronizer import CrossSystemSynchronizer, SyncPolicy, ConflictResolution

__all__ = [
    'PreferenceIntegrationEngine',
    'IntegrationType',
    'PreferenceScope',
    'PersonalizationCoordinator',
    'PersonalizationLevel',
    'AdaptationStrategy',
    'UserExperienceOptimizer',
    'ExperienceMetric',
    'OptimizationGoal',
    'CrossSystemSynchronizer',
    'SyncPolicy',
    'ConflictResolution'
]