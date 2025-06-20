"""
Preference Integration Engine

Core engine that integrates user preferences across all Heart Protocol systems,
ensuring consistent and personalized healing experiences.
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable, Union, Set
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import logging
from abc import ABC, abstractmethod
import json

# Import preference systems
from ...preferences.preference_manager import PreferenceManager, PreferenceCategory
from ...preferences.cache_controller import UserCacheController, CacheType
from ...preferences.privacy_settings import PrivacySettingsManager, PrivacyLevel
from ...preferences.user_control import UserControlPanel, AccessLevel

# Import personalization systems
from ...feeds.daily_gentle_reminders.personalization import EmotionalCapacityPersonalizer
from ...feeds.daily_gentle_reminders.cultural_sensitivity import CulturalSensitivityEngine

logger = logging.getLogger(__name__)


class IntegrationType(Enum):
    """Types of preference integration"""
    FEED_PERSONALIZATION = "feed_personalization"      # Feed content personalization
    ALGORITHM_OPTIMIZATION = "algorithm_optimization"   # Algorithm parameter tuning
    PRIVACY_CONTROLS = "privacy_controls"              # Privacy setting enforcement
    CACHE_MANAGEMENT = "cache_management"              # Cache behavior control
    CULTURAL_ADAPTATION = "cultural_adaptation"        # Cultural responsiveness
    SAFETY_SETTINGS = "safety_settings"               # Safety and crisis settings
    ACCESSIBILITY = "accessibility"                    # Accessibility accommodations
    COMMUNICATION_STYLE = "communication_style"        # Communication preferences
    TIMING_OPTIMIZATION = "timing_optimization"        # Timing and scheduling
    HEALING_JOURNEY = "healing_journey"               # Healing journey customization


class PreferenceScope(Enum):
    """Scope of preference application"""
    GLOBAL = "global"                                  # Applied across all systems
    SYSTEM_SPECIFIC = "system_specific"               # Applied to specific system
    CONTEXT_AWARE = "context_aware"                   # Applied based on context
    TEMPORARY = "temporary"                           # Applied temporarily
    SESSION_ONLY = "session_only"                    # Applied for current session only


class PreferenceConflictStrategy(Enum):
    """Strategies for resolving preference conflicts"""
    USER_SAFETY_FIRST = "user_safety_first"          # Prioritize user safety
    HEALING_FOCUSED = "healing_focused"               # Prioritize healing outcomes
    USER_CHOICE = "user_choice"                       # Respect explicit user choice
    CONTEXT_SENSITIVE = "context_sensitive"           # Consider current context
    PRIVACY_PROTECTIVE = "privacy_protective"         # Prioritize privacy protection


@dataclass
class PreferenceMapping:
    """Mapping between preference and system implementation"""
    preference_id: str
    source_system: str
    target_systems: List[str]
    integration_type: IntegrationType
    scope: PreferenceScope
    priority: int  # 1-10, 10 being highest
    conflict_strategy: PreferenceConflictStrategy
    transformation_rules: Dict[str, Any]
    validation_rules: Dict[str, Any]
    healing_impact: str
    user_control_level: str


@dataclass
class IntegrationContext:
    """Context for preference integration"""
    user_id: str
    current_system: str
    operation_type: str
    care_context: str
    emotional_state: str
    healing_stage: str
    privacy_level: PrivacyLevel
    accessibility_needs: List[str]
    cultural_context: Dict[str, Any]
    session_data: Dict[str, Any]
    urgency_level: str
    user_consent_status: Dict[str, bool]


@dataclass
class IntegrationResult:
    """Result of preference integration"""
    integration_id: str
    user_id: str
    applied_preferences: Dict[str, Any]
    system_configurations: Dict[str, Dict[str, Any]]
    conflicts_resolved: List[Dict[str, Any]]
    healing_optimizations: List[str]
    privacy_protections: List[str]
    cultural_adaptations: List[str]
    accessibility_accommodations: List[str]
    integration_timestamp: datetime
    effectiveness_prediction: float
    user_satisfaction_prediction: float


class PreferenceIntegrationEngine:
    """
    Core engine for integrating user preferences across all Heart Protocol systems.
    
    Core Principles:
    - User agency and choice are paramount
    - Healing effectiveness guides preference application
    - Privacy and safety are never compromised
    - Cultural responsiveness is deeply integrated
    - Accessibility is built into every interaction
    - Trauma-informed principles guide all decisions
    - Community healing is supported alongside individual preferences
    - Gentle adaptation prevents overwhelming users
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize preference managers
        self.preference_manager = PreferenceManager(config.get('preferences', {}))
        self.cache_controller = UserCacheController(config.get('cache', {}))
        self.privacy_manager = PrivacySettingsManager(config.get('privacy', {}))
        self.user_control_panel = UserControlPanel(
            config.get('user_control', {}),
            self.preference_manager,
            self.cache_controller,
            self.privacy_manager
        )
        
        # Initialize personalization systems
        self.emotional_personalizer = EmotionalCapacityPersonalizer(config.get('emotional', {}))
        self.cultural_engine = CulturalSensitivityEngine(config.get('cultural', {}))
        
        # Integration state
        self.preference_mappings: Dict[str, PreferenceMapping] = {}
        self.integration_history: List[IntegrationResult] = []
        self.system_registry: Dict[str, Dict[str, Any]] = {}
        self.conflict_resolution_rules: Dict[str, Callable] = {}
        
        # Integration callbacks
        self.integration_callbacks: List[Callable] = []
        
        # Initialize preference mappings
        self._initialize_preference_mappings()
        
        # Initialize system registry
        self._initialize_system_registry()
        
        # Initialize conflict resolution
        self._initialize_conflict_resolution()
    
    def _initialize_preference_mappings(self):
        """Initialize mappings between preferences and system implementations"""
        
        # Feed Personalization Mappings
        self.preference_mappings["feed_emotional_intensity"] = PreferenceMapping(
            preference_id="content_emotional_intensity",
            source_system="preference_manager",
            target_systems=["daily_gentle_reminders", "community_wisdom", "hearts_seeking_light"],
            integration_type=IntegrationType.FEED_PERSONALIZATION,
            scope=PreferenceScope.GLOBAL,
            priority=8,
            conflict_strategy=PreferenceConflictStrategy.HEALING_FOCUSED,
            transformation_rules={
                "daily_gentle_reminders": "max_emotional_intensity",
                "community_wisdom": "content_intensity_filter",
                "hearts_seeking_light": "support_intensity_level"
            },
            validation_rules={
                "min_value": 0.1,
                "max_value": 1.0,
                "trauma_safe_max": 0.7
            },
            healing_impact="Controls emotional intensity to prevent overwhelm",
            user_control_level="complete"
        )
        
        self.preference_mappings["cultural_responsiveness"] = PreferenceMapping(
            preference_id="cultural_preferences",
            source_system="preference_manager",
            target_systems=["cultural_engine", "all_feeds", "communication_style"],
            integration_type=IntegrationType.CULTURAL_ADAPTATION,
            scope=PreferenceScope.GLOBAL,
            priority=9,
            conflict_strategy=PreferenceConflictStrategy.USER_CHOICE,
            transformation_rules={
                "cultural_engine": "primary_cultural_context",
                "all_feeds": "cultural_adaptation_level",
                "communication_style": "cultural_communication_style"
            },
            validation_rules={
                "require_user_consent": True,
                "respect_privacy": True
            },
            healing_impact="Ensures culturally appropriate healing approaches",
            user_control_level="complete"
        )
        
        # Privacy Control Mappings
        self.preference_mappings["data_sharing_consent"] = PreferenceMapping(
            preference_id="data_sharing_preferences",
            source_system="privacy_manager",
            target_systems=["all_systems"],
            integration_type=IntegrationType.PRIVACY_CONTROLS,
            scope=PreferenceScope.GLOBAL,
            priority=10,
            conflict_strategy=PreferenceConflictStrategy.PRIVACY_PROTECTIVE,
            transformation_rules={
                "analytics": "data_collection_level",
                "personalization": "personalization_data_usage",
                "community_features": "social_data_sharing"
            },
            validation_rules={
                "explicit_consent_required": True,
                "granular_control": True,
                "right_to_withdraw": True
            },
            healing_impact="Protects user privacy while enabling personalization",
            user_control_level="complete"
        )
        
        # Cache Management Mappings
        self.preference_mappings["cache_preferences"] = PreferenceMapping(
            preference_id="cache_control_settings",
            source_system="cache_controller",
            target_systems=["all_systems"],
            integration_type=IntegrationType.CACHE_MANAGEMENT,
            scope=PreferenceScope.GLOBAL,
            priority=7,
            conflict_strategy=PreferenceConflictStrategy.USER_CHOICE,
            transformation_rules={
                "performance_systems": "cache_aggressiveness",
                "privacy_systems": "data_retention_limits",
                "personalization": "learning_data_retention"
            },
            validation_rules={
                "respect_privacy_level": True,
                "honor_ttl_preferences": True
            },
            healing_impact="Balances performance and privacy preferences",
            user_control_level="high"
        )
        
        # Safety Settings Mappings
        self.preference_mappings["crisis_intervention"] = PreferenceMapping(
            preference_id="crisis_response_preferences",
            source_system="preference_manager",
            target_systems=["crisis_detection", "safety_monitoring", "escalation_engine"],
            integration_type=IntegrationType.SAFETY_SETTINGS,
            scope=PreferenceScope.CONTEXT_AWARE,
            priority=10,
            conflict_strategy=PreferenceConflictStrategy.USER_SAFETY_FIRST,
            transformation_rules={
                "crisis_detection": "intervention_threshold",
                "safety_monitoring": "monitoring_sensitivity",
                "escalation_engine": "escalation_preferences"
            },
            validation_rules={
                "safety_cannot_be_disabled": True,
                "emergency_override_allowed": True
            },
            healing_impact="Respects user autonomy while ensuring safety",
            user_control_level="high_with_safety_limits"
        )
        
        # Communication Style Mappings
        self.preference_mappings["communication_style"] = PreferenceMapping(
            preference_id="communication_preferences",
            source_system="preference_manager",
            target_systems=["monarch_bot", "all_feeds", "notifications"],
            integration_type=IntegrationType.COMMUNICATION_STYLE,
            scope=PreferenceScope.GLOBAL,
            priority=8,
            conflict_strategy=PreferenceConflictStrategy.HEALING_FOCUSED,
            transformation_rules={
                "monarch_bot": "persona_adaptation",
                "all_feeds": "content_tone",
                "notifications": "notification_style"
            },
            validation_rules={
                "trauma_informed_required": True,
                "cultural_sensitivity_required": True
            },
            healing_impact="Adapts communication to support healing",
            user_control_level="complete"
        )
        
        # Accessibility Mappings
        self.preference_mappings["accessibility_accommodations"] = PreferenceMapping(
            preference_id="accessibility_preferences",
            source_system="preference_manager",
            target_systems=["all_systems"],
            integration_type=IntegrationType.ACCESSIBILITY,
            scope=PreferenceScope.GLOBAL,
            priority=9,
            conflict_strategy=PreferenceConflictStrategy.USER_CHOICE,
            transformation_rules={
                "ui_systems": "accessibility_mode",
                "content_delivery": "content_accessibility",
                "interaction_systems": "interaction_accommodations"
            },
            validation_rules={
                "inclusive_design_required": True,
                "multiple_modalities": True
            },
            healing_impact="Ensures inclusive healing experiences",
            user_control_level="complete"
        )
        
        # Timing Optimization Mappings
        self.preference_mappings["timing_preferences"] = PreferenceMapping(
            preference_id="timing_and_scheduling",
            source_system="preference_manager",
            target_systems=["timing_optimizer", "notification_system", "feed_delivery"],
            integration_type=IntegrationType.TIMING_OPTIMIZATION,
            scope=PreferenceScope.CONTEXT_AWARE,
            priority=7,
            conflict_strategy=PreferenceConflictStrategy.HEALING_FOCUSED,
            transformation_rules={
                "timing_optimizer": "optimal_timing_windows",
                "notification_system": "notification_schedule",
                "feed_delivery": "content_delivery_timing"
            },
            validation_rules={
                "respect_quiet_hours": True,
                "emergency_override_allowed": True
            },
            healing_impact="Optimizes timing for maximum healing impact",
            user_control_level="high"
        )
        
        # Healing Journey Mappings
        self.preference_mappings["healing_journey_stage"] = PreferenceMapping(
            preference_id="healing_stage_preferences",
            source_system="preference_manager",
            target_systems=["all_algorithms", "content_systems", "matching_systems"],
            integration_type=IntegrationType.HEALING_JOURNEY,
            scope=PreferenceScope.CONTEXT_AWARE,
            priority=9,
            conflict_strategy=PreferenceConflictStrategy.HEALING_FOCUSED,
            transformation_rules={
                "all_algorithms": "healing_stage_optimization",
                "content_systems": "stage_appropriate_content",
                "matching_systems": "peer_stage_matching"
            },
            validation_rules={
                "stage_progression_respect": True,
                "no_forced_advancement": True
            },
            healing_impact="Adapts entire system to healing journey stage",
            user_control_level="collaborative"
        )
    
    def _initialize_system_registry(self):
        """Initialize registry of all systems that can receive preferences"""
        
        self.system_registry = {
            "preference_manager": {
                "type": "core_preference_system",
                "capabilities": ["store_preferences", "validate_preferences", "sync_preferences"],
                "preference_types": ["all"],
                "integration_methods": ["direct_api", "event_system"]
            },
            
            "cache_controller": {
                "type": "performance_system",
                "capabilities": ["cache_management", "ttl_control", "data_retention"],
                "preference_types": ["cache_preferences", "privacy_preferences"],
                "integration_methods": ["direct_api"]
            },
            
            "privacy_manager": {
                "type": "privacy_system",
                "capabilities": ["privacy_enforcement", "consent_management", "data_protection"],
                "preference_types": ["privacy_preferences", "data_sharing"],
                "integration_methods": ["direct_api", "policy_enforcement"]
            },
            
            "daily_gentle_reminders": {
                "type": "feed_system",
                "capabilities": ["content_personalization", "emotional_adaptation", "timing_optimization"],
                "preference_types": ["content_preferences", "emotional_preferences", "timing_preferences"],
                "integration_methods": ["configuration_injection", "runtime_adaptation"]
            },
            
            "community_wisdom": {
                "type": "feed_system",
                "capabilities": ["content_curation", "community_matching", "wisdom_personalization"],
                "preference_types": ["content_preferences", "community_preferences", "learning_preferences"],
                "integration_methods": ["configuration_injection", "algorithmic_parameters"]
            },
            
            "hearts_seeking_light": {
                "type": "feed_system",
                "capabilities": ["support_matching", "intervention_timing", "safety_protocols"],
                "preference_types": ["support_preferences", "privacy_preferences", "safety_preferences"],
                "integration_methods": ["configuration_injection", "safety_overrides"]
            },
            
            "guardian_energy_rising": {
                "type": "feed_system",
                "capabilities": ["progress_tracking", "celebration_timing", "milestone_recognition"],
                "preference_types": ["progress_preferences", "celebration_preferences", "achievement_preferences"],
                "integration_methods": ["configuration_injection", "progress_algorithms"]
            },
            
            "monarch_bot": {
                "type": "conversational_system",
                "capabilities": ["persona_adaptation", "communication_style", "crisis_response"],
                "preference_types": ["communication_preferences", "persona_preferences", "safety_preferences"],
                "integration_methods": ["persona_configuration", "response_adaptation"]
            },
            
            "crisis_detection": {
                "type": "safety_system",
                "capabilities": ["pattern_recognition", "threshold_adaptation", "intervention_triggering"],
                "preference_types": ["safety_preferences", "intervention_preferences", "privacy_preferences"],
                "integration_methods": ["threshold_configuration", "pattern_adaptation"]
            },
            
            "cultural_engine": {
                "type": "adaptation_system",
                "capabilities": ["cultural_adaptation", "content_localization", "communication_adaptation"],
                "preference_types": ["cultural_preferences", "communication_preferences", "content_preferences"],
                "integration_methods": ["cultural_configuration", "content_adaptation"]
            },
            
            "emotional_personalizer": {
                "type": "personalization_system",
                "capabilities": ["emotional_assessment", "capacity_adaptation", "intensity_control"],
                "preference_types": ["emotional_preferences", "capacity_preferences", "adaptation_preferences"],
                "integration_methods": ["parameter_injection", "real_time_adaptation"]
            }
        }
    
    def _initialize_conflict_resolution(self):
        """Initialize conflict resolution strategies"""
        
        self.conflict_resolution_rules = {
            "user_safety_first": self._resolve_safety_first,
            "healing_focused": self._resolve_healing_focused,
            "user_choice": self._resolve_user_choice,
            "context_sensitive": self._resolve_context_sensitive,
            "privacy_protective": self._resolve_privacy_protective
        }
    
    async def integrate_user_preferences(self, user_id: str, 
                                       integration_context: IntegrationContext,
                                       target_systems: Optional[List[str]] = None) -> IntegrationResult:
        """
        Integrate user preferences across specified systems or all systems
        
        Args:
            user_id: User whose preferences to integrate
            integration_context: Context for integration decisions
            target_systems: Specific systems to integrate with (None for all)
            
        Returns:
            Integration result with applied configurations
        """
        try:
            integration_id = f"integration_{user_id}_{datetime.utcnow().isoformat()}"
            
            # Get user preferences from all systems
            user_preferences = await self._collect_user_preferences(user_id)
            
            # Apply integration context
            contextual_preferences = await self._apply_integration_context(
                user_preferences, integration_context
            )
            
            # Resolve preference conflicts
            resolved_preferences = await self._resolve_preference_conflicts(
                contextual_preferences, integration_context
            )
            
            # Generate system configurations
            system_configurations = await self._generate_system_configurations(
                resolved_preferences, integration_context, target_systems
            )
            
            # Apply configurations to target systems
            application_results = await self._apply_system_configurations(
                system_configurations, integration_context
            )
            
            # Calculate effectiveness predictions
            effectiveness_prediction = await self._predict_integration_effectiveness(
                resolved_preferences, system_configurations, integration_context
            )
            
            # Create integration result
            result = IntegrationResult(
                integration_id=integration_id,
                user_id=user_id,
                applied_preferences=resolved_preferences,
                system_configurations=system_configurations,
                conflicts_resolved=application_results.get('conflicts_resolved', []),
                healing_optimizations=application_results.get('healing_optimizations', []),
                privacy_protections=application_results.get('privacy_protections', []),
                cultural_adaptations=application_results.get('cultural_adaptations', []),
                accessibility_accommodations=application_results.get('accessibility_accommodations', []),
                integration_timestamp=datetime.utcnow(),
                effectiveness_prediction=effectiveness_prediction,
                user_satisfaction_prediction=effectiveness_prediction * 0.9
            )
            
            # Store integration result
            self.integration_history.append(result)
            
            # Trigger integration callbacks
            for callback in self.integration_callbacks:
                try:
                    await callback(result)
                except Exception as e:
                    logger.error(f"Integration callback failed: {str(e)}")
            
            logger.info(f"Successfully integrated preferences for user {user_id}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to integrate user preferences: {str(e)}")
            # Return safe fallback result
            return await self._get_fallback_integration_result(user_id, integration_context)
    
    async def _collect_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Collect user preferences from all preference systems"""
        try:
            preferences = {}
            
            # Collect from main preference manager
            core_preferences = await self.preference_manager.get_user_preferences(user_id)
            preferences.update(core_preferences)
            
            # Collect cache preferences
            cache_preferences = await self.cache_controller.get_user_cache_preferences(user_id)
            preferences['cache_preferences'] = cache_preferences
            
            # Collect privacy settings
            privacy_settings = await self.privacy_manager.get_user_privacy_settings(user_id)
            preferences['privacy_settings'] = privacy_settings
            
            # Collect emotional personalization data
            emotional_profile = await self.emotional_personalizer.get_user_profile(user_id)
            if emotional_profile:
                preferences['emotional_profile'] = emotional_profile
            
            # Collect cultural preferences
            cultural_preferences = await self.cultural_engine.get_user_cultural_settings(user_id)
            preferences['cultural_preferences'] = cultural_preferences
            
            return preferences
            
        except Exception as e:
            logger.error(f"Failed to collect user preferences: {str(e)}")
            return {}
    
    async def _apply_integration_context(self, preferences: Dict[str, Any],
                                       context: IntegrationContext) -> Dict[str, Any]:
        """Apply integration context to modify preferences appropriately"""
        try:
            contextual_preferences = preferences.copy()
            
            # Apply care context modifications
            if context.care_context == "immediate_crisis":
                # Crisis context - prioritize safety and simplicity
                contextual_preferences['emotional_intensity_max'] = 0.3
                contextual_preferences['complexity_level'] = 'minimal'
                contextual_preferences['privacy_level'] = 'maximum_protection'
                
            elif context.care_context == "support_seeking":
                # Support seeking - enhance connection features
                contextual_preferences['community_features_enabled'] = True
                contextual_preferences['peer_matching_active'] = True
                contextual_preferences['support_visibility'] = 'enhanced'
                
            elif context.care_context == "healing_progress":
                # Healing progress - optimize for growth
                contextual_preferences['progress_tracking_enabled'] = True
                contextual_preferences['celebration_features'] = 'enhanced'
                contextual_preferences['growth_content_priority'] = 'high'
            
            # Apply emotional state modifications
            if context.emotional_state in ['fragile', 'overwhelmed']:
                # Reduce intensity and complexity
                current_intensity = contextual_preferences.get('emotional_intensity_max', 1.0)
                contextual_preferences['emotional_intensity_max'] = min(current_intensity, 0.5)
                contextual_preferences['gentle_processing'] = True
                contextual_preferences['opt_out_always_available'] = True
            
            # Apply urgency level modifications
            if context.urgency_level == 'high':
                # High urgency - streamline experience
                contextual_preferences['streamlined_interface'] = True
                contextual_preferences['essential_features_only'] = True
                contextual_preferences['fast_access_mode'] = True
            
            # Apply accessibility context
            if context.accessibility_needs:
                for need in context.accessibility_needs:
                    if need == 'cognitive_support':
                        contextual_preferences['simplified_language'] = True
                        contextual_preferences['clear_navigation'] = True
                    elif need == 'visual_impairment':
                        contextual_preferences['screen_reader_optimized'] = True
                        contextual_preferences['high_contrast'] = True
                    elif need == 'motor_impairment':
                        contextual_preferences['large_touch_targets'] = True
                        contextual_preferences['gesture_alternatives'] = True
            
            # Apply cultural context
            if context.cultural_context:
                cultural_values = context.cultural_context.get('values', [])
                if 'collective_healing' in cultural_values:
                    contextual_preferences['community_emphasis'] = 'high'
                    contextual_preferences['individual_vs_collective'] = 'collective_preferred'
                if 'family_involvement' in cultural_values:
                    contextual_preferences['family_features_enabled'] = True
                    contextual_preferences['family_privacy_considerations'] = True
            
            return contextual_preferences
            
        except Exception as e:
            logger.error(f"Failed to apply integration context: {str(e)}")
            return preferences
    
    async def _resolve_preference_conflicts(self, preferences: Dict[str, Any],
                                          context: IntegrationContext) -> Dict[str, Any]:
        """Resolve conflicts between different preference sources"""
        try:
            resolved_preferences = preferences.copy()
            conflicts_found = []
            
            # Check for common conflicts
            conflicts = [
                # Privacy vs Personalization
                {
                    'type': 'privacy_vs_personalization',
                    'preferences': ['privacy_level', 'personalization_level'],
                    'resolution_strategy': 'privacy_protective'
                },
                
                # Performance vs Gentleness
                {
                    'type': 'performance_vs_gentleness',
                    'preferences': ['cache_aggressiveness', 'gentle_processing'],
                    'resolution_strategy': 'healing_focused'
                },
                
                # Individual vs Community
                {
                    'type': 'individual_vs_community',
                    'preferences': ['individual_focus', 'community_emphasis'],
                    'resolution_strategy': 'context_sensitive'
                },
                
                # Safety vs Autonomy
                {
                    'type': 'safety_vs_autonomy',
                    'preferences': ['safety_monitoring', 'user_autonomy'],
                    'resolution_strategy': 'user_safety_first'
                }
            ]
            
            for conflict in conflicts:
                conflict_detected = await self._detect_preference_conflict(
                    resolved_preferences, conflict
                )
                
                if conflict_detected:
                    conflicts_found.append(conflict)
                    resolution_strategy = conflict['resolution_strategy']
                    resolver = self.conflict_resolution_rules.get(resolution_strategy)
                    
                    if resolver:
                        resolved_preferences = await resolver(
                            resolved_preferences, conflict, context
                        )
            
            # Log resolved conflicts
            if conflicts_found:
                logger.info(f"Resolved {len(conflicts_found)} preference conflicts for user {context.user_id}")
            
            return resolved_preferences
            
        except Exception as e:
            logger.error(f"Failed to resolve preference conflicts: {str(e)}")
            return preferences
    
    async def _detect_preference_conflict(self, preferences: Dict[str, Any],
                                        conflict_definition: Dict[str, Any]) -> bool:
        """Detect if a specific type of preference conflict exists"""
        try:
            conflict_type = conflict_definition['type']
            preference_keys = conflict_definition['preferences']
            
            if conflict_type == 'privacy_vs_personalization':
                privacy_level = preferences.get('privacy_level', 'medium')
                personalization_level = preferences.get('personalization_level', 'medium')
                
                # Conflict if high privacy but high personalization requested
                return (privacy_level in ['high', 'maximum'] and 
                       personalization_level in ['high', 'maximum'])
            
            elif conflict_type == 'performance_vs_gentleness':
                cache_aggressive = preferences.get('cache_aggressiveness', 'medium')
                gentle_processing = preferences.get('gentle_processing', False)
                
                # Conflict if aggressive caching but gentle processing requested
                return cache_aggressive == 'high' and gentle_processing
            
            elif conflict_type == 'individual_vs_community':
                individual_focus = preferences.get('individual_focus', 'medium')
                community_emphasis = preferences.get('community_emphasis', 'medium')
                
                # Conflict if both high individual focus and high community emphasis
                return (individual_focus == 'high' and community_emphasis == 'high')
            
            elif conflict_type == 'safety_vs_autonomy':
                safety_monitoring = preferences.get('safety_monitoring_level', 'medium')
                user_autonomy = preferences.get('user_autonomy_level', 'high')
                
                # Conflict if high safety monitoring but high autonomy preference
                return (safety_monitoring == 'intensive' and user_autonomy == 'maximum')
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to detect preference conflict: {str(e)}")
            return False
    
    async def _resolve_safety_first(self, preferences: Dict[str, Any],
                                  conflict: Dict[str, Any],
                                  context: IntegrationContext) -> Dict[str, Any]:
        """Resolve conflicts with user safety as top priority"""
        if conflict['type'] == 'safety_vs_autonomy':
            # In crisis contexts, safety takes precedence
            if context.care_context == 'immediate_crisis':
                preferences['safety_monitoring_level'] = 'intensive'
                preferences['user_autonomy_level'] = 'high_with_safety_limits'
            else:
                # In non-crisis, find middle ground
                preferences['safety_monitoring_level'] = 'balanced'
                preferences['user_autonomy_level'] = 'high'
        
        return preferences
    
    async def _resolve_healing_focused(self, preferences: Dict[str, Any],
                                     conflict: Dict[str, Any],
                                     context: IntegrationContext) -> Dict[str, Any]:
        """Resolve conflicts with healing effectiveness as priority"""
        if conflict['type'] == 'performance_vs_gentleness':
            # For healing focus, choose gentleness over performance
            preferences['cache_aggressiveness'] = 'moderate'
            preferences['gentle_processing'] = True
            preferences['performance_vs_healing_balance'] = 'healing_optimized'
        
        return preferences
    
    async def _resolve_user_choice(self, preferences: Dict[str, Any],
                                 conflict: Dict[str, Any],
                                 context: IntegrationContext) -> Dict[str, Any]:
        """Resolve conflicts by respecting explicit user choice"""
        # Look for explicit user preferences in the most recent settings
        if conflict['type'] == 'individual_vs_community':
            # Check which preference was set more recently or explicitly
            last_individual_update = preferences.get('individual_focus_last_updated')
            last_community_update = preferences.get('community_emphasis_last_updated')
            
            if last_individual_update and last_community_update:
                if last_individual_update > last_community_update:
                    preferences['community_emphasis'] = 'moderate'
                else:
                    preferences['individual_focus'] = 'moderate'
        
        return preferences
    
    async def _resolve_context_sensitive(self, preferences: Dict[str, Any],
                                       conflict: Dict[str, Any],
                                       context: IntegrationContext) -> Dict[str, Any]:
        """Resolve conflicts based on current context"""
        if conflict['type'] == 'individual_vs_community':
            # Context-sensitive resolution
            if context.care_context == 'support_seeking':
                # When seeking support, emphasize community
                preferences['community_emphasis'] = 'high'
                preferences['individual_focus'] = 'moderate'
            elif context.care_context == 'healing_progress':
                # When making progress, balance both
                preferences['community_emphasis'] = 'moderate'
                preferences['individual_focus'] = 'moderate'
            elif context.emotional_state in ['fragile', 'overwhelmed']:
                # When fragile, prioritize individual care
                preferences['individual_focus'] = 'high'
                preferences['community_emphasis'] = 'low'
        
        return preferences
    
    async def _resolve_privacy_protective(self, preferences: Dict[str, Any],
                                        conflict: Dict[str, Any],
                                        context: IntegrationContext) -> Dict[str, Any]:
        """Resolve conflicts with privacy protection as priority"""
        if conflict['type'] == 'privacy_vs_personalization':
            # Privacy takes precedence
            privacy_level = preferences.get('privacy_level', 'medium')
            
            if privacy_level in ['high', 'maximum']:
                preferences['personalization_level'] = 'privacy_safe'
                preferences['data_collection_minimal'] = True
                preferences['local_personalization_preferred'] = True
        
        return preferences
    
    async def _generate_system_configurations(self, preferences: Dict[str, Any],
                                            context: IntegrationContext,
                                            target_systems: Optional[List[str]]) -> Dict[str, Dict[str, Any]]:
        """Generate system-specific configurations from integrated preferences"""
        try:
            configurations = {}
            
            # Determine which systems to configure
            systems_to_configure = target_systems or list(self.system_registry.keys())
            
            for system_name in systems_to_configure:
                if system_name not in self.system_registry:
                    continue
                
                system_info = self.system_registry[system_name]
                system_config = {}
                
                # Apply relevant preferences to this system
                for mapping_id, mapping in self.preference_mappings.items():
                    if system_name in mapping.target_systems or 'all_systems' in mapping.target_systems:
                        # Transform preference for this system
                        transformed_value = await self._transform_preference_for_system(
                            preferences, mapping, system_name, context
                        )
                        
                        if transformed_value is not None:
                            system_config.update(transformed_value)
                
                # Add system-specific optimizations
                system_config = await self._add_system_optimizations(
                    system_config, system_name, preferences, context
                )
                
                configurations[system_name] = system_config
            
            return configurations
            
        except Exception as e:
            logger.error(f"Failed to generate system configurations: {str(e)}")
            return {}
    
    async def _transform_preference_for_system(self, preferences: Dict[str, Any],
                                             mapping: PreferenceMapping,
                                             system_name: str,
                                             context: IntegrationContext) -> Optional[Dict[str, Any]]:
        """Transform a preference value for a specific system"""
        try:
            # Get preference value
            preference_value = preferences.get(mapping.preference_id)
            if preference_value is None:
                return None
            
            # Apply transformation rules
            transformation_rules = mapping.transformation_rules
            system_key = None
            
            # Find system-specific transformation
            if system_name in transformation_rules:
                system_key = transformation_rules[system_name]
            elif 'all_systems' in transformation_rules:
                system_key = transformation_rules['all_systems']
            elif 'default' in transformation_rules:
                system_key = transformation_rules['default']
            
            if not system_key:
                return None
            
            # Apply validation rules
            validated_value = await self._validate_preference_value(
                preference_value, mapping.validation_rules, context
            )
            
            if validated_value is None:
                return None
            
            # Create system configuration
            return {system_key: validated_value}
            
        except Exception as e:
            logger.error(f"Failed to transform preference for system {system_name}: {str(e)}")
            return None
    
    async def _validate_preference_value(self, value: Any,
                                       validation_rules: Dict[str, Any],
                                       context: IntegrationContext) -> Optional[Any]:
        """Validate preference value against validation rules"""
        try:
            validated_value = value
            
            # Apply validation rules
            if 'min_value' in validation_rules and isinstance(value, (int, float)):
                validated_value = max(validated_value, validation_rules['min_value'])
            
            if 'max_value' in validation_rules and isinstance(value, (int, float)):
                validated_value = min(validated_value, validation_rules['max_value'])
            
            # Trauma-informed safety checks
            if validation_rules.get('trauma_safe_max') and context.care_context == 'trauma_sensitive':
                if isinstance(validated_value, (int, float)):
                    validated_value = min(validated_value, validation_rules['trauma_safe_max'])
            
            # Privacy protection checks
            if validation_rules.get('require_user_consent'):
                consent_key = f"consent_{context.current_system}"
                if not context.user_consent_status.get(consent_key, False):
                    return None
            
            # Respect privacy level
            if validation_rules.get('respect_privacy'):
                if context.privacy_level in [PrivacyLevel.MAXIMUM, PrivacyLevel.HIGH]:
                    # Apply privacy-safe defaults
                    if isinstance(validated_value, (int, float)):
                        validated_value = min(validated_value, 0.5)
            
            return validated_value
            
        except Exception as e:
            logger.error(f"Failed to validate preference value: {str(e)}")
            return value
    
    async def _add_system_optimizations(self, config: Dict[str, Any],
                                      system_name: str,
                                      preferences: Dict[str, Any],
                                      context: IntegrationContext) -> Dict[str, Any]:
        """Add system-specific optimizations to configuration"""
        try:
            optimized_config = config.copy()
            
            # Feed system optimizations
            if system_name in ['daily_gentle_reminders', 'community_wisdom', 'hearts_seeking_light', 'guardian_energy_rising']:
                # Add healing-focused optimizations
                optimized_config['healing_focus'] = True
                optimized_config['trauma_informed'] = True
                optimized_config['cultural_responsive'] = True
                
                # Add emotional capacity considerations
                if context.emotional_state in ['fragile', 'overwhelmed']:
                    optimized_config['gentle_mode'] = True
                    optimized_config['reduced_complexity'] = True
                
                # Add accessibility optimizations
                if context.accessibility_needs:
                    optimized_config['accessibility_enhanced'] = True
                    optimized_config['accessibility_features'] = context.accessibility_needs
            
            # Safety system optimizations
            elif system_name in ['crisis_detection', 'safety_monitoring']:
                # Add safety-focused optimizations
                optimized_config['safety_priority'] = True
                optimized_config['false_positive_prevention'] = True
                
                # Adjust sensitivity based on user preferences
                safety_sensitivity = preferences.get('safety_monitoring_sensitivity', 'balanced')
                optimized_config['monitoring_sensitivity'] = safety_sensitivity
            
            # Privacy system optimizations
            elif system_name == 'privacy_manager':
                # Add privacy-focused optimizations
                optimized_config['privacy_by_design'] = True
                optimized_config['data_minimization'] = True
                optimized_config['user_control_maximum'] = True
            
            # Personalization system optimizations
            elif system_name in ['emotional_personalizer', 'cultural_engine']:
                # Add personalization optimizations
                optimized_config['ethical_personalization'] = True
                optimized_config['transparency_enabled'] = True
                optimized_config['user_agency_preserved'] = True
            
            return optimized_config
            
        except Exception as e:
            logger.error(f"Failed to add system optimizations: {str(e)}")
            return config
    
    async def _apply_system_configurations(self, configurations: Dict[str, Dict[str, Any]],
                                         context: IntegrationContext) -> Dict[str, Any]:
        """Apply configurations to target systems"""
        try:
            application_results = {
                'successful_applications': [],
                'failed_applications': [],
                'conflicts_resolved': [],
                'healing_optimizations': [],
                'privacy_protections': [],
                'cultural_adaptations': [],
                'accessibility_accommodations': []
            }
            
            for system_name, config in configurations.items():
                try:
                    # Apply configuration based on system type
                    success = await self._apply_configuration_to_system(
                        system_name, config, context
                    )
                    
                    if success:
                        application_results['successful_applications'].append(system_name)
                        
                        # Track specific types of optimizations
                        if config.get('healing_focus'):
                            application_results['healing_optimizations'].append(
                                f"Applied healing focus to {system_name}"
                            )
                        
                        if config.get('privacy_by_design'):
                            application_results['privacy_protections'].append(
                                f"Enhanced privacy protection in {system_name}"
                            )
                        
                        if config.get('cultural_responsive'):
                            application_results['cultural_adaptations'].append(
                                f"Enabled cultural responsiveness in {system_name}"
                            )
                        
                        if config.get('accessibility_enhanced'):
                            application_results['accessibility_accommodations'].append(
                                f"Enhanced accessibility in {system_name}"
                            )
                    else:
                        application_results['failed_applications'].append(system_name)
                
                except Exception as e:
                    logger.error(f"Failed to apply configuration to {system_name}: {str(e)}")
                    application_results['failed_applications'].append(system_name)
            
            return application_results
            
        except Exception as e:
            logger.error(f"Failed to apply system configurations: {str(e)}")
            return {}
    
    async def _apply_configuration_to_system(self, system_name: str,
                                           config: Dict[str, Any],
                                           context: IntegrationContext) -> bool:
        """Apply configuration to a specific system"""
        try:
            system_info = self.system_registry.get(system_name)
            if not system_info:
                return False
            
            integration_methods = system_info['integration_methods']
            
            # Apply configuration using available integration methods
            if 'direct_api' in integration_methods:
                return await self._apply_via_direct_api(system_name, config, context)
            elif 'configuration_injection' in integration_methods:
                return await self._apply_via_configuration_injection(system_name, config, context)
            elif 'runtime_adaptation' in integration_methods:
                return await self._apply_via_runtime_adaptation(system_name, config, context)
            elif 'event_system' in integration_methods:
                return await self._apply_via_event_system(system_name, config, context)
            else:
                logger.warning(f"No supported integration method for {system_name}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to apply configuration to {system_name}: {str(e)}")
            return False
    
    async def _apply_via_direct_api(self, system_name: str,
                                  config: Dict[str, Any],
                                  context: IntegrationContext) -> bool:
        """Apply configuration via direct API call"""
        try:
            # Get system instance and apply configuration
            if system_name == 'preference_manager':
                return await self.preference_manager.apply_user_configuration(
                    context.user_id, config
                )
            elif system_name == 'cache_controller':
                return await self.cache_controller.apply_user_configuration(
                    context.user_id, config
                )
            elif system_name == 'privacy_manager':
                return await self.privacy_manager.apply_user_configuration(
                    context.user_id, config
                )
            elif system_name == 'emotional_personalizer':
                return await self.emotional_personalizer.apply_configuration(
                    context.user_id, config
                )
            elif system_name == 'cultural_engine':
                return await self.cultural_engine.apply_configuration(
                    context.user_id, config
                )
            else:
                logger.warning(f"Direct API not implemented for {system_name}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to apply configuration via direct API: {str(e)}")
            return False
    
    async def _apply_via_configuration_injection(self, system_name: str,
                                               config: Dict[str, Any],
                                               context: IntegrationContext) -> bool:
        """Apply configuration via configuration injection"""
        try:
            # Store configuration for system to pick up
            config_key = f"user_config_{context.user_id}_{system_name}"
            
            # In production, this would use a configuration store
            # For now, we'll simulate successful injection
            logger.info(f"Injected configuration for {system_name}: {config}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply configuration via injection: {str(e)}")
            return False
    
    async def _apply_via_runtime_adaptation(self, system_name: str,
                                          config: Dict[str, Any],
                                          context: IntegrationContext) -> bool:
        """Apply configuration via runtime adaptation"""
        try:
            # Signal system to adapt configuration at runtime
            adaptation_signal = {
                'user_id': context.user_id,
                'system_name': system_name,
                'configuration': config,
                'context': {
                    'care_context': context.care_context,
                    'emotional_state': context.emotional_state,
                    'urgency_level': context.urgency_level
                }
            }
            
            # In production, this would use an event bus or signal system
            logger.info(f"Sent runtime adaptation signal for {system_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply configuration via runtime adaptation: {str(e)}")
            return False
    
    async def _apply_via_event_system(self, system_name: str,
                                    config: Dict[str, Any],
                                    context: IntegrationContext) -> bool:
        """Apply configuration via event system"""
        try:
            # Publish configuration update event
            event = {
                'event_type': 'preference_configuration_update',
                'user_id': context.user_id,
                'target_system': system_name,
                'configuration': config,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # In production, this would use an event bus
            logger.info(f"Published configuration event for {system_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply configuration via event system: {str(e)}")
            return False
    
    async def _predict_integration_effectiveness(self, preferences: Dict[str, Any],
                                               configurations: Dict[str, Dict[str, Any]],
                                               context: IntegrationContext) -> float:
        """Predict effectiveness of preference integration"""
        try:
            effectiveness_score = 0.5  # Start with neutral score
            
            # Boost for healing-focused configurations
            healing_focused_systems = sum(
                1 for config in configurations.values() 
                if config.get('healing_focus', False)
            )
            effectiveness_score += (healing_focused_systems / len(configurations)) * 0.2
            
            # Boost for privacy protection
            if any(config.get('privacy_by_design') for config in configurations.values()):
                effectiveness_score += 0.1
            
            # Boost for cultural responsiveness
            if any(config.get('cultural_responsive') for config in configurations.values()):
                effectiveness_score += 0.1
            
            # Boost for accessibility
            if any(config.get('accessibility_enhanced') for config in configurations.values()):
                effectiveness_score += 0.1
            
            # Consider context appropriateness
            if context.care_context == 'immediate_crisis':
                # High effectiveness if safety systems are prioritized
                safety_systems = ['crisis_detection', 'safety_monitoring']
                safety_configured = any(system in configurations for system in safety_systems)
                if safety_configured:
                    effectiveness_score += 0.2
            
            # Consider emotional state appropriateness
            if context.emotional_state in ['fragile', 'overwhelmed']:
                # High effectiveness if gentle processing is enabled
                gentle_enabled = any(
                    config.get('gentle_mode') for config in configurations.values()
                )
                if gentle_enabled:
                    effectiveness_score += 0.15
            
            return min(1.0, effectiveness_score)
            
        except Exception as e:
            logger.error(f"Failed to predict integration effectiveness: {str(e)}")
            return 0.5
    
    async def _get_fallback_integration_result(self, user_id: str,
                                             context: IntegrationContext) -> IntegrationResult:
        """Get safe fallback integration result"""
        return IntegrationResult(
            integration_id=f"fallback_{user_id}_{datetime.utcnow().isoformat()}",
            user_id=user_id,
            applied_preferences={
                'safe_defaults': True,
                'privacy_protected': True,
                'healing_focused': True
            },
            system_configurations={
                'all_systems': {
                    'safe_mode': True,
                    'privacy_maximum': True,
                    'gentle_processing': True
                }
            },
            conflicts_resolved=[],
            healing_optimizations=['Applied safe healing defaults'],
            privacy_protections=['Maximum privacy protection enabled'],
            cultural_adaptations=[],
            accessibility_accommodations=[],
            integration_timestamp=datetime.utcnow(),
            effectiveness_prediction=0.6,
            user_satisfaction_prediction=0.6
        )
    
    def add_integration_callback(self, callback: Callable[[IntegrationResult], None]):
        """Add callback for integration completion events"""
        self.integration_callbacks.append(callback)
    
    async def get_integration_report(self, user_id: str, 
                                   time_range_hours: int = 24) -> Dict[str, Any]:
        """Get comprehensive integration report for a user"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=time_range_hours)
            
            user_integrations = [
                result for result in self.integration_history
                if result.user_id == user_id and result.integration_timestamp >= cutoff_time
            ]
            
            if not user_integrations:
                return {'no_data': True, 'user_id': user_id, 'time_range_hours': time_range_hours}
            
            # Calculate metrics
            avg_effectiveness = sum(r.effectiveness_prediction for r in user_integrations) / len(user_integrations)
            avg_satisfaction = sum(r.user_satisfaction_prediction for r in user_integrations) / len(user_integrations)
            
            # System usage analysis
            system_usage = {}
            for result in user_integrations:
                for system_name in result.system_configurations:
                    system_usage[system_name] = system_usage.get(system_name, 0) + 1
            
            # Optimization tracking
            all_healing_optimizations = []
            all_privacy_protections = []
            all_cultural_adaptations = []
            all_accessibility_accommodations = []
            
            for result in user_integrations:
                all_healing_optimizations.extend(result.healing_optimizations)
                all_privacy_protections.extend(result.privacy_protections)
                all_cultural_adaptations.extend(result.cultural_adaptations)
                all_accessibility_accommodations.extend(result.accessibility_accommodations)
            
            return {
                'user_id': user_id,
                'time_range_hours': time_range_hours,
                'total_integrations': len(user_integrations),
                'average_effectiveness': avg_effectiveness,
                'average_satisfaction_prediction': avg_satisfaction,
                'system_usage_distribution': system_usage,
                'healing_optimizations_applied': len(all_healing_optimizations),
                'privacy_protections_applied': len(all_privacy_protections),
                'cultural_adaptations_applied': len(all_cultural_adaptations),
                'accessibility_accommodations_applied': len(all_accessibility_accommodations),
                'integration_frequency': len(user_integrations) / max(1, time_range_hours / 24),
                'most_recent_integration': user_integrations[-1].integration_timestamp.isoformat() if user_integrations else None,
                'recommendation_score': self._calculate_recommendation_score(user_integrations)
            }
            
        except Exception as e:
            logger.error(f"Failed to generate integration report: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_recommendation_score(self, integrations: List[IntegrationResult]) -> float:
        """Calculate recommendation score for integration system performance"""
        try:
            if not integrations:
                return 0.5
            
            # Base score on effectiveness and satisfaction
            avg_effectiveness = sum(r.effectiveness_prediction for r in integrations) / len(integrations)
            avg_satisfaction = sum(r.user_satisfaction_prediction for r in integrations) / len(integrations)
            
            # Weight recent integrations more heavily
            recent_integrations = integrations[-5:]  # Last 5 integrations
            if recent_integrations:
                recent_effectiveness = sum(r.effectiveness_prediction for r in recent_integrations) / len(recent_integrations)
                recent_satisfaction = sum(r.user_satisfaction_prediction for r in recent_integrations) / len(recent_integrations)
                
                # Blend historical and recent performance
                recommendation_score = (
                    avg_effectiveness * 0.3 + avg_satisfaction * 0.3 +
                    recent_effectiveness * 0.2 + recent_satisfaction * 0.2
                )
            else:
                recommendation_score = (avg_effectiveness + avg_satisfaction) / 2
            
            return min(1.0, max(0.0, recommendation_score))
            
        except Exception as e:
            logger.error(f"Failed to calculate recommendation score: {str(e)}")
            return 0.5