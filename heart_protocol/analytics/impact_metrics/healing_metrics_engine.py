"""
Healing Metrics Engine

Core engine for measuring and tracking healing-focused outcomes rather than
traditional engagement metrics, prioritizing user wellbeing and authentic growth.
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import logging
import statistics
from collections import defaultdict, deque
import numpy as np

logger = logging.getLogger(__name__)


class HealingDimension(Enum):
    """Dimensions of healing to measure"""
    EMOTIONAL_WELLBEING = "emotional_wellbeing"         # Emotional state improvement
    PSYCHOLOGICAL_SAFETY = "psychological_safety"       # Feeling safe and secure
    SELF_COMPASSION = "self_compassion"                 # Self-kindness and acceptance
    RESILIENCE_BUILDING = "resilience_building"         # Capacity to cope and recover
    AUTHENTIC_CONNECTION = "authentic_connection"       # Meaningful relationships
    PURPOSE_MEANING = "purpose_meaning"                 # Sense of purpose and meaning
    TRAUMA_HEALING = "trauma_healing"                   # Recovery from trauma
    GROWTH_MINDSET = "growth_mindset"                   # Openness to learning and growth
    COMMUNITY_BELONGING = "community_belonging"         # Sense of belonging and acceptance
    HOPE_OPTIMISM = "hope_optimism"                     # Hope for the future
    SELF_EFFICACY = "self_efficacy"                     # Belief in one's abilities
    BOUNDARY_HEALTH = "boundary_health"                 # Healthy boundary setting


class ImpactLevel(Enum):
    """Levels of healing impact"""
    TRANSFORMATIONAL = "transformational"               # Life-changing impact
    SIGNIFICANT = "significant"                         # Major positive change
    MODERATE = "moderate"                               # Clear improvement
    SUBTLE = "subtle"                                   # Gentle positive shift
    MAINTAINING = "maintaining"                         # Maintaining current wellbeing
    CONCERNING = "concerning"                           # Concerning patterns
    CRISIS = "crisis"                                   # Crisis intervention needed


class MeasurementMethod(Enum):
    """Methods for measuring healing outcomes"""
    SELF_REPORTED = "self_reported"                     # User self-assessment
    BEHAVIORAL_INDICATORS = "behavioral_indicators"     # Observed behavior patterns
    INTERACTION_QUALITY = "interaction_quality"         # Quality of interactions
    CONTENT_ANALYSIS = "content_analysis"               # Content sentiment/themes
    LONGITUDINAL_TRACKING = "longitudinal_tracking"     # Changes over time
    PEER_FEEDBACK = "peer_feedback"                     # Community feedback
    PROFESSIONAL_ASSESSMENT = "professional_assessment" # Professional evaluation
    PHYSIOLOGICAL_MARKERS = "physiological_markers"     # Stress/wellness indicators


class TimeWindow(Enum):
    """Time windows for measurement"""
    IMMEDIATE = "immediate"                             # Real-time/immediate
    DAILY = "daily"                                     # Daily aggregation
    WEEKLY = "weekly"                                   # Weekly trends
    MONTHLY = "monthly"                                 # Monthly patterns
    QUARTERLY = "quarterly"                             # Quarterly assessments
    YEARLY = "yearly"                                   # Annual review
    LIFETIME = "lifetime"                               # Lifetime journey


@dataclass
class HealingMeasurement:
    """Individual healing measurement"""
    measurement_id: str
    user_id: str
    healing_dimension: HealingDimension
    measurement_method: MeasurementMethod
    value: float
    confidence_level: float
    context_factors: Dict[str, Any]
    cultural_considerations: List[str]
    trauma_informed_adjustments: List[str]
    measurement_timestamp: datetime
    baseline_comparison: Optional[float]
    trend_direction: str
    significance_level: float
    privacy_level: str
    consent_given: bool


@dataclass
class HealingProfile:
    """User's healing profile and journey"""
    user_id: str
    healing_dimensions: Dict[HealingDimension, float]
    baseline_measurements: Dict[HealingDimension, float]
    growth_trajectory: Dict[HealingDimension, List[float]]
    significant_milestones: List[Dict[str, Any]]
    crisis_events: List[Dict[str, Any]]
    intervention_responses: List[Dict[str, Any]]
    support_network_quality: float
    resilience_indicators: List[str]
    vulnerability_factors: List[str]
    cultural_context: List[str]
    trauma_considerations: List[str]
    healing_goals: List[str]
    progress_celebrations: List[Dict[str, Any]]
    last_updated: datetime
    consent_preferences: Dict[str, bool]


@dataclass
class ImpactReport:
    """Comprehensive impact assessment report"""
    report_id: str
    time_period: Tuple[datetime, datetime]
    user_population: int
    healing_dimension_summary: Dict[HealingDimension, Dict[str, float]]
    community_impact_metrics: Dict[str, float]
    crisis_intervention_outcomes: Dict[str, Any]
    connection_quality_metrics: Dict[str, float]
    cultural_inclusion_metrics: Dict[str, float]
    trauma_informed_effectiveness: Dict[str, float]
    significant_achievements: List[Dict[str, Any]]
    areas_for_improvement: List[Dict[str, Any]]
    methodology_notes: List[str]
    ethical_considerations: List[str]
    generated_at: datetime


class HealingMetricsEngine:
    """
    Core engine for measuring healing-focused outcomes and impact.
    
    Core Principles:
    - Healing outcomes matter more than engagement metrics
    - User wellbeing and authentic growth are the primary measures of success
    - Privacy and consent are paramount in all measurement
    - Cultural sensitivity guides measurement interpretation
    - Trauma-informed approaches prevent re-traumatization through measurement
    - Longitudinal tracking captures authentic healing journeys
    - Community healing is as important as individual healing
    - Holistic measurement includes emotional, psychological, and social dimensions
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Engine settings
        self.measurement_enabled = self.config.get('measurement_enabled', True)
        self.privacy_first_approach = self.config.get('privacy_first_approach', True)
        self.trauma_informed_measurement = self.config.get('trauma_informed_measurement', True)
        self.cultural_sensitivity_level = self.config.get('cultural_sensitivity_level', 'high')
        
        # Measurement framework
        self.measurement_methods = self._initialize_measurement_methods()
        self.healing_indicators = self._initialize_healing_indicators()
        self.baseline_calculator = self._initialize_baseline_calculator()
        self.trend_analyzer = self._initialize_trend_analyzer()
        
        # Data storage and tracking
        self.user_healing_profiles: Dict[str, HealingProfile] = {}
        self.measurement_history: List[HealingMeasurement] = []
        self.impact_reports: List[ImpactReport] = []
        self.community_metrics: Dict[str, Any] = {}
        
        # Processing queues
        self.measurement_queue: asyncio.Queue = asyncio.Queue()
        self.analysis_queue: asyncio.Queue = asyncio.Queue()
        
        # Performance tracking
        self.total_measurements_processed = 0
        self.healing_improvements_detected = 0
        self.crisis_interventions_measured = 0
        self.community_healing_events = 0
        
        # Privacy and safety systems
        self.privacy_protector = None           # Would integrate with privacy system
        self.consent_manager = None            # Would integrate with consent management
        self.cultural_interpreter = None      # Would integrate with cultural sensitivity
        self.trauma_safety_validator = None   # Would integrate with trauma safety
        
        # Callbacks and integrations
        self.healing_milestone_callbacks: List[Callable] = []
        self.crisis_detected_callbacks: List[Callable] = []
        self.community_achievement_callbacks: List[Callable] = []
        self.impact_report_callbacks: List[Callable] = []
        
        # Processing state
        self.processing_active = False
        self.processing_tasks: Set[asyncio.Task] = set()
    
    def _initialize_measurement_methods(self) -> Dict[MeasurementMethod, Dict[str, Any]]:
        """Initialize measurement methods and their configurations"""
        return {
            MeasurementMethod.SELF_REPORTED: {
                'weight': 0.8,
                'confidence_adjustment': 0.0,
                'trauma_considerations': ['self_awareness_level', 'current_capacity'],
                'cultural_adaptations': ['expression_styles', 'values_alignment'],
                'privacy_requirements': ['explicit_consent', 'data_minimization']
            },
            
            MeasurementMethod.BEHAVIORAL_INDICATORS: {
                'weight': 0.7,
                'confidence_adjustment': 0.1,
                'trauma_considerations': ['behavioral_changes', 'coping_mechanisms'],
                'cultural_adaptations': ['behavioral_norms', 'expression_differences'],
                'privacy_requirements': ['observational_consent', 'behavioral_privacy']
            },
            
            MeasurementMethod.INTERACTION_QUALITY: {
                'weight': 0.9,
                'confidence_adjustment': 0.2,
                'trauma_considerations': ['interaction_safety', 'boundary_respect'],
                'cultural_adaptations': ['communication_styles', 'relationship_norms'],
                'privacy_requirements': ['interaction_consent', 'relationship_privacy']
            },
            
            MeasurementMethod.CONTENT_ANALYSIS: {
                'weight': 0.6,
                'confidence_adjustment': -0.1,
                'trauma_considerations': ['content_safety', 'expression_freedom'],
                'cultural_adaptations': ['language_nuances', 'cultural_expressions'],
                'privacy_requirements': ['content_consent', 'expression_privacy']
            },
            
            MeasurementMethod.LONGITUDINAL_TRACKING: {
                'weight': 0.9,
                'confidence_adjustment': 0.3,
                'trauma_considerations': ['healing_pace', 'setback_sensitivity'],
                'cultural_adaptations': ['healing_timelines', 'progress_concepts'],
                'privacy_requirements': ['long_term_consent', 'journey_privacy']
            }
        }
    
    def _initialize_healing_indicators(self) -> Dict[HealingDimension, Dict[str, Any]]:
        """Initialize indicators for each healing dimension"""
        return {
            HealingDimension.EMOTIONAL_WELLBEING: {
                'positive_indicators': [
                    'emotional_stability', 'joy_expression', 'emotional_awareness',
                    'emotional_regulation', 'positive_mood_frequency', 'emotional_resilience'
                ],
                'concerning_indicators': [
                    'emotional_volatility', 'persistent_sadness', 'emotional_numbness',
                    'overwhelming_emotions', 'emotional_withdrawal'
                ],
                'measurement_approaches': [
                    'mood_tracking', 'emotional_expression_analysis', 'self_reported_feelings',
                    'interaction_emotional_tone', 'content_emotional_markers'
                ],
                'cultural_considerations': [
                    'emotional_expression_norms', 'cultural_emotional_vocabulary',
                    'family_emotional_patterns', 'cultural_healing_practices'
                ],
                'trauma_informed_factors': [
                    'emotional_safety_requirements', 'trigger_awareness',
                    'emotional_capacity_respect', 'healing_pace_honoring'
                ]
            },
            
            HealingDimension.PSYCHOLOGICAL_SAFETY: {
                'positive_indicators': [
                    'trust_building', 'vulnerability_comfort', 'boundary_respect',
                    'safety_feeling', 'predictability_comfort', 'control_sense'
                ],
                'concerning_indicators': [
                    'hypervigilance', 'trust_difficulties', 'safety_concerns',
                    'boundary_violations', 'control_loss_anxiety'
                ],
                'measurement_approaches': [
                    'safety_self_assessment', 'trust_relationship_quality',
                    'boundary_setting_success', 'comfort_level_tracking'
                ],
                'cultural_considerations': [
                    'safety_cultural_definitions', 'trust_building_norms',
                    'authority_relationship_patterns', 'community_safety_concepts'
                ],
                'trauma_informed_factors': [
                    'trauma_history_awareness', 'safety_planning_needs',
                    'trigger_management', 'control_restoration_focus'
                ]
            },
            
            HealingDimension.AUTHENTIC_CONNECTION: {
                'positive_indicators': [
                    'meaningful_relationships', 'authentic_expression', 'mutual_support',
                    'emotional_intimacy', 'shared_vulnerability', 'genuine_caring'
                ],
                'concerning_indicators': [
                    'isolation_patterns', 'superficial_connections', 'relationship_avoidance',
                    'people_pleasing', 'connection_anxiety'
                ],
                'measurement_approaches': [
                    'relationship_quality_assessment', 'connection_depth_analysis',
                    'mutual_support_tracking', 'authenticity_expression_measurement'
                ],
                'cultural_considerations': [
                    'relationship_cultural_norms', 'connection_expression_styles',
                    'family_relationship_patterns', 'community_connection_values'
                ],
                'trauma_informed_factors': [
                    'attachment_considerations', 'relationship_trauma_awareness',
                    'connection_safety_needs', 'intimacy_pacing_respect'
                ]
            },
            
            HealingDimension.RESILIENCE_BUILDING: {
                'positive_indicators': [
                    'coping_skill_development', 'adversity_recovery', 'adaptability',
                    'problem_solving_confidence', 'stress_management', 'bounce_back_ability'
                ],
                'concerning_indicators': [
                    'overwhelming_stress_response', 'coping_mechanism_absence',
                    'prolonged_recovery_periods', 'avoidance_patterns'
                ],
                'measurement_approaches': [
                    'stress_response_tracking', 'coping_strategy_effectiveness',
                    'recovery_time_measurement', 'adaptability_assessment'
                ],
                'cultural_considerations': [
                    'resilience_cultural_definitions', 'coping_cultural_methods',
                    'strength_cultural_expressions', 'adversity_cultural_meanings'
                ],
                'trauma_informed_factors': [
                    'trauma_impact_on_resilience', 'recovery_pace_respect',
                    'strength_based_approaches', 'trauma_informed_coping'
                ]
            },
            
            HealingDimension.HOPE_OPTIMISM: {
                'positive_indicators': [
                    'future_orientation', 'positive_expectations', 'goal_setting',
                    'possibility_mindset', 'growth_belief', 'recovery_confidence'
                ],
                'concerning_indicators': [
                    'hopelessness_expressions', 'future_pessimism', 'despair_indicators',
                    'futility_feelings', 'growth_disbelief'
                ],
                'measurement_approaches': [
                    'hope_scale_assessments', 'future_goal_tracking',
                    'optimism_expression_analysis', 'recovery_belief_measurement'
                ],
                'cultural_considerations': [
                    'hope_cultural_expressions', 'future_cultural_orientations',
                    'optimism_cultural_norms', 'spiritual_hope_connections'
                ],
                'trauma_informed_factors': [
                    'trauma_impact_on_hope', 'hope_restoration_process',
                    'realistic_hope_building', 'despair_sensitivity'
                ]
            }
        }
    
    def _initialize_baseline_calculator(self) -> Dict[str, Any]:
        """Initialize baseline calculation methods"""
        return {
            'initial_assessment_window': timedelta(days=14),
            'baseline_stabilization_period': timedelta(days=30),
            'minimum_data_points': 5,
            'confidence_threshold': 0.7,
            'cultural_adjustment_factors': {
                'expression_style_differences': 0.1,
                'cultural_norm_variations': 0.15,
                'language_expression_differences': 0.1
            },
            'trauma_informed_adjustments': {
                'trauma_impact_consideration': 0.2,
                'healing_capacity_adjustment': 0.15,
                'safety_prioritization': 0.1
            }
        }
    
    def _initialize_trend_analyzer(self) -> Dict[str, Any]:
        """Initialize trend analysis methods"""
        return {
            'short_term_window': timedelta(days=7),
            'medium_term_window': timedelta(days=30),
            'long_term_window': timedelta(days=90),
            'trend_significance_threshold': 0.1,
            'improvement_detection_sensitivity': 0.05,
            'decline_detection_sensitivity': 0.08,
            'plateau_detection_window': timedelta(days=21),
            'breakthrough_detection_threshold': 0.25
        }
    
    async def start_measurement_engine(self) -> bool:
        """Start the healing metrics measurement engine"""
        try:
            if self.processing_active:
                logger.warning("Healing metrics engine already active")
                return True
            
            self.processing_active = True
            
            # Start processing tasks
            await self._start_processing_tasks()
            
            logger.info("Healing metrics engine started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start healing metrics engine: {str(e)}")
            self.processing_active = False
            return False
    
    async def stop_measurement_engine(self) -> bool:
        """Stop the healing metrics measurement engine"""
        try:
            self.processing_active = False
            
            # Cancel processing tasks
            for task in self.processing_tasks:
                if not task.done():
                    task.cancel()
            
            self.processing_tasks.clear()
            
            logger.info("Healing metrics engine stopped")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop healing metrics engine: {str(e)}")
            return False
    
    async def _start_processing_tasks(self):
        """Start measurement processing tasks"""
        try:
            # Start measurement processing loop
            task = asyncio.create_task(self._measurement_processing_loop())
            self.processing_tasks.add(task)
            
            # Start analysis loop
            task = asyncio.create_task(self._analysis_processing_loop())
            self.processing_tasks.add(task)
            
            # Start trend analysis loop
            task = asyncio.create_task(self._trend_analysis_loop())
            self.processing_tasks.add(task)
            
            # Start impact reporting loop
            task = asyncio.create_task(self._impact_reporting_loop())
            self.processing_tasks.add(task)
            
        except Exception as e:
            logger.error(f"Failed to start processing tasks: {str(e)}")
    
    async def record_healing_measurement(self, user_id: str, dimension: HealingDimension,
                                       method: MeasurementMethod, value: float,
                                       context: Optional[Dict[str, Any]] = None) -> Optional[HealingMeasurement]:
        """Record a healing measurement for a user"""
        try:
            if not self.measurement_enabled:
                return None
            
            # Validate consent
            if not await self._validate_measurement_consent(user_id, dimension, method):
                logger.warning(f"Measurement consent not found for user {user_id}")
                return None
            
            # Create measurement
            measurement = await self._create_healing_measurement(
                user_id, dimension, method, value, context
            )
            
            if measurement:
                # Add to processing queue
                await self.measurement_queue.put(measurement)
                
                return measurement
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to record healing measurement: {str(e)}")
            return None
    
    async def _create_healing_measurement(self, user_id: str, dimension: HealingDimension,
                                        method: MeasurementMethod, value: float,
                                        context: Optional[Dict[str, Any]]) -> Optional[HealingMeasurement]:
        """Create a healing measurement with full analysis"""
        try:
            # Calculate confidence level
            confidence_level = await self._calculate_measurement_confidence(dimension, method, value, context)
            
            # Analyze cultural considerations
            cultural_considerations = await self._analyze_cultural_considerations(user_id, dimension, context)
            
            # Apply trauma-informed adjustments
            trauma_adjustments = await self._apply_trauma_informed_adjustments(user_id, dimension, value, context)
            
            # Calculate baseline comparison
            baseline_comparison = await self._calculate_baseline_comparison(user_id, dimension, value)
            
            # Determine trend direction
            trend_direction = await self._determine_trend_direction(user_id, dimension, value)
            
            # Calculate significance level
            significance_level = await self._calculate_significance_level(dimension, value, baseline_comparison)
            
            # Determine privacy level
            privacy_level = await self._determine_privacy_level(user_id, dimension, method)
            
            measurement = HealingMeasurement(
                measurement_id=f"healing_measurement_{datetime.utcnow().isoformat()}_{id(self)}",
                user_id=user_id,
                healing_dimension=dimension,
                measurement_method=method,
                value=value,
                confidence_level=confidence_level,
                context_factors=context or {},
                cultural_considerations=cultural_considerations,
                trauma_informed_adjustments=trauma_adjustments,
                measurement_timestamp=datetime.utcnow(),
                baseline_comparison=baseline_comparison,
                trend_direction=trend_direction,
                significance_level=significance_level,
                privacy_level=privacy_level,
                consent_given=True  # Already validated
            )
            
            return measurement
            
        except Exception as e:
            logger.error(f"Error creating healing measurement: {str(e)}")
            return None
    
    async def _validate_measurement_consent(self, user_id: str, dimension: HealingDimension,
                                          method: MeasurementMethod) -> bool:
        """Validate user consent for measurement"""
        try:
            # Check if user has healing profile with consent preferences
            if user_id in self.user_healing_profiles:
                profile = self.user_healing_profiles[user_id]
                
                # Check specific consent for dimension and method
                consent_key = f"{dimension.value}_{method.value}"
                if consent_key in profile.consent_preferences:
                    return profile.consent_preferences[consent_key]
                
                # Check general measurement consent
                if 'general_measurement' in profile.consent_preferences:
                    return profile.consent_preferences['general_measurement']
            
            # Default to requiring explicit consent
            return False
            
        except Exception as e:
            logger.error(f"Error validating measurement consent: {str(e)}")
            return False
    
    async def _calculate_measurement_confidence(self, dimension: HealingDimension,
                                              method: MeasurementMethod, value: float,
                                              context: Optional[Dict[str, Any]]) -> float:
        """Calculate confidence in the measurement"""
        try:
            # Base confidence from method
            method_config = self.measurement_methods.get(method, {})
            base_confidence = method_config.get('weight', 0.5)
            
            # Adjust for context factors
            context_adjustment = 0.0
            if context:
                # Higher confidence with more context
                context_richness = len(context) / 10  # Normalize by expected context size
                context_adjustment += min(0.2, context_richness * 0.1)
                
                # Specific context factors
                if context.get('multiple_indicators', False):
                    context_adjustment += 0.1
                
                if context.get('consistent_pattern', False):
                    context_adjustment += 0.1
                
                if context.get('external_validation', False):
                    context_adjustment += 0.15
            
            # Adjust for value reasonableness
            value_adjustment = 0.0
            if 0.0 <= value <= 1.0:  # Reasonable range
                value_adjustment += 0.1
            else:
                value_adjustment -= 0.1
            
            total_confidence = base_confidence + context_adjustment + value_adjustment
            
            return max(0.0, min(1.0, total_confidence))
            
        except Exception as e:
            logger.error(f"Error calculating measurement confidence: {str(e)}")
            return 0.5
    
    async def _analyze_cultural_considerations(self, user_id: str, dimension: HealingDimension,
                                             context: Optional[Dict[str, Any]]) -> List[str]:
        """Analyze cultural considerations for measurement interpretation"""
        try:
            considerations = []
            
            # Get user's cultural context
            if user_id in self.user_healing_profiles:
                user_culture = self.user_healing_profiles[user_id].cultural_context
                
                # Dimension-specific cultural considerations
                dimension_config = self.healing_indicators.get(dimension, {})
                cultural_factors = dimension_config.get('cultural_considerations', [])
                
                for factor in cultural_factors:
                    if any(culture in factor for culture in user_culture):
                        considerations.append(factor)
            
            # Context-based cultural considerations
            if context:
                if context.get('cultural_expression_style'):
                    considerations.append('cultural_expression_style_consideration')
                
                if context.get('language_differences'):
                    considerations.append('language_cultural_adaptation')
                
                if context.get('family_cultural_context'):
                    considerations.append('family_cultural_norms_consideration')
            
            return considerations
            
        except Exception as e:
            logger.error(f"Error analyzing cultural considerations: {str(e)}")
            return []
    
    async def _apply_trauma_informed_adjustments(self, user_id: str, dimension: HealingDimension,
                                                value: float, context: Optional[Dict[str, Any]]) -> List[str]:
        """Apply trauma-informed adjustments to measurement"""
        try:
            adjustments = []
            
            # Get user's trauma considerations
            if user_id in self.user_healing_profiles:
                trauma_factors = self.user_healing_profiles[user_id].trauma_considerations
                
                # Dimension-specific trauma considerations
                dimension_config = self.healing_indicators.get(dimension, {})
                trauma_informed_factors = dimension_config.get('trauma_informed_factors', [])
                
                for factor in trauma_informed_factors:
                    if any(trauma in factor for trauma in trauma_factors):
                        adjustments.append(factor)
            
            # Value-based trauma considerations
            if value < 0.3:  # Low wellbeing scores
                adjustments.append('low_wellbeing_trauma_sensitivity')
            
            # Context-based trauma considerations
            if context:
                if context.get('trauma_trigger_context'):
                    adjustments.append('trauma_trigger_measurement_adjustment')
                
                if context.get('crisis_context'):
                    adjustments.append('crisis_context_trauma_awareness')
            
            return adjustments
            
        except Exception as e:
            logger.error(f"Error applying trauma-informed adjustments: {str(e)}")
            return []
    
    async def _calculate_baseline_comparison(self, user_id: str, dimension: HealingDimension,
                                           value: float) -> Optional[float]:
        """Calculate comparison to user's baseline"""
        try:
            if user_id in self.user_healing_profiles:
                profile = self.user_healing_profiles[user_id]
                
                if dimension in profile.baseline_measurements:
                    baseline = profile.baseline_measurements[dimension]
                    return value - baseline
            
            return None
            
        except Exception as e:
            logger.error(f"Error calculating baseline comparison: {str(e)}")
            return None
    
    async def _determine_trend_direction(self, user_id: str, dimension: HealingDimension,
                                       value: float) -> str:
        """Determine trend direction for the measurement"""
        try:
            if user_id in self.user_healing_profiles:
                profile = self.user_healing_profiles[user_id]
                
                if dimension in profile.growth_trajectory:
                    trajectory = profile.growth_trajectory[dimension]
                    
                    if len(trajectory) >= 2:
                        recent_values = trajectory[-3:]  # Last 3 values
                        
                        if len(recent_values) >= 2:
                            recent_trend = recent_values[-1] - recent_values[-2]
                            
                            if recent_trend > 0.05:
                                return 'improving'
                            elif recent_trend < -0.05:
                                return 'declining'
                            else:
                                return 'stable'
            
            return 'initial'
            
        except Exception as e:
            logger.error(f"Error determining trend direction: {str(e)}")
            return 'unknown'
    
    async def _calculate_significance_level(self, dimension: HealingDimension, value: float,
                                          baseline_comparison: Optional[float]) -> float:
        """Calculate significance level of the measurement"""
        try:
            significance = 0.5  # Base significance
            
            # Significance based on absolute value
            if value > 0.8:
                significance += 0.2
            elif value < 0.3:
                significance += 0.3  # Concerning values are highly significant
            
            # Significance based on change from baseline
            if baseline_comparison is not None:
                change_magnitude = abs(baseline_comparison)
                
                if change_magnitude > 0.2:
                    significance += 0.3
                elif change_magnitude > 0.1:
                    significance += 0.2
                elif change_magnitude > 0.05:
                    significance += 0.1
            
            return min(1.0, significance)
            
        except Exception as e:
            logger.error(f"Error calculating significance level: {str(e)}")
            return 0.5
    
    async def _determine_privacy_level(self, user_id: str, dimension: HealingDimension,
                                     method: MeasurementMethod) -> str:
        """Determine privacy level for the measurement"""
        try:
            # Get method privacy requirements
            method_config = self.measurement_methods.get(method, {})
            privacy_requirements = method_config.get('privacy_requirements', [])
            
            # Determine privacy level based on dimension sensitivity
            sensitive_dimensions = [
                HealingDimension.TRAUMA_HEALING,
                HealingDimension.PSYCHOLOGICAL_SAFETY,
                HealingDimension.AUTHENTIC_CONNECTION
            ]
            
            if dimension in sensitive_dimensions:
                return 'high_privacy'
            elif 'explicit_consent' in privacy_requirements:
                return 'standard_privacy'
            else:
                return 'minimal_privacy'
                
        except Exception as e:
            logger.error(f"Error determining privacy level: {str(e)}")
            return 'standard_privacy'
    
    async def _measurement_processing_loop(self):
        """Main measurement processing loop"""
        try:
            while self.processing_active:
                try:
                    # Wait for measurement with timeout
                    measurement = await asyncio.wait_for(self.measurement_queue.get(), timeout=1.0)
                    
                    # Process the measurement
                    await self._process_healing_measurement(measurement)
                    
                    # Mark queue task as done
                    self.measurement_queue.task_done()
                    
                except asyncio.TimeoutError:
                    # No measurements in queue, continue waiting
                    continue
                except Exception as e:
                    logger.error(f"Error in measurement processing loop: {str(e)}")
                    await asyncio.sleep(1)
                    
        except Exception as e:
            logger.error(f"Fatal error in measurement processing loop: {str(e)}")
    
    async def _process_healing_measurement(self, measurement: HealingMeasurement):
        """Process a healing measurement"""
        try:
            # Store measurement
            self.measurement_history.append(measurement)
            self.total_measurements_processed += 1
            
            # Update user healing profile
            await self._update_user_healing_profile(measurement)
            
            # Check for significant events
            await self._check_for_significant_events(measurement)
            
            # Add to analysis queue for deeper analysis
            await self.analysis_queue.put(measurement)
            
        except Exception as e:
            logger.error(f"Error processing healing measurement: {str(e)}")
    
    async def _update_user_healing_profile(self, measurement: HealingMeasurement):
        """Update user's healing profile with new measurement"""
        try:
            user_id = measurement.user_id
            dimension = measurement.healing_dimension
            
            # Create profile if doesn't exist
            if user_id not in self.user_healing_profiles:
                self.user_healing_profiles[user_id] = await self._create_new_healing_profile(user_id)
            
            profile = self.user_healing_profiles[user_id]
            
            # Update current dimension value
            profile.healing_dimensions[dimension] = measurement.value
            
            # Update growth trajectory
            if dimension not in profile.growth_trajectory:
                profile.growth_trajectory[dimension] = []
            
            profile.growth_trajectory[dimension].append(measurement.value)
            
            # Keep trajectory manageable (last 100 measurements)
            if len(profile.growth_trajectory[dimension]) > 100:
                profile.growth_trajectory[dimension] = profile.growth_trajectory[dimension][-100:]
            
            # Update baseline if needed
            await self._update_baseline_if_needed(profile, dimension, measurement)
            
            # Check for milestones
            await self._check_for_milestones(profile, measurement)
            
            # Update timestamp
            profile.last_updated = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Error updating user healing profile: {str(e)}")
    
    async def _create_new_healing_profile(self, user_id: str) -> HealingProfile:
        """Create a new healing profile for a user"""
        try:
            return HealingProfile(
                user_id=user_id,
                healing_dimensions={},
                baseline_measurements={},
                growth_trajectory={},
                significant_milestones=[],
                crisis_events=[],
                intervention_responses=[],
                support_network_quality=0.5,
                resilience_indicators=[],
                vulnerability_factors=[],
                cultural_context=[],
                trauma_considerations=[],
                healing_goals=[],
                progress_celebrations=[],
                last_updated=datetime.utcnow(),
                consent_preferences={}
            )
            
        except Exception as e:
            logger.error(f"Error creating new healing profile: {str(e)}")
            raise
    
    async def _update_baseline_if_needed(self, profile: HealingProfile, dimension: HealingDimension,
                                       measurement: HealingMeasurement):
        """Update baseline measurement if needed"""
        try:
            # Check if baseline exists
            if dimension not in profile.baseline_measurements:
                # Need enough measurements to establish baseline
                if dimension in profile.growth_trajectory:
                    trajectory = profile.growth_trajectory[dimension]
                    
                    if len(trajectory) >= self.baseline_calculator['minimum_data_points']:
                        # Calculate baseline as median of early measurements
                        early_measurements = trajectory[:self.baseline_calculator['minimum_data_points']]
                        baseline = statistics.median(early_measurements)
                        profile.baseline_measurements[dimension] = baseline
            
        except Exception as e:
            logger.error(f"Error updating baseline: {str(e)}")
    
    async def _check_for_milestones(self, profile: HealingProfile, measurement: HealingMeasurement):
        """Check for healing milestones"""
        try:
            dimension = measurement.healing_dimension
            value = measurement.value
            
            # Check for significant improvements
            if measurement.baseline_comparison and measurement.baseline_comparison > 0.2:
                milestone = {
                    'type': 'significant_improvement',
                    'dimension': dimension.value,
                    'improvement_amount': measurement.baseline_comparison,
                    'timestamp': measurement.measurement_timestamp,
                    'significance': measurement.significance_level
                }
                
                profile.significant_milestones.append(milestone)
                self.healing_improvements_detected += 1
                
                # Trigger callbacks
                for callback in self.healing_milestone_callbacks:
                    try:
                        await callback(milestone)
                    except Exception as e:
                        logger.error(f"Healing milestone callback failed: {str(e)}")
            
            # Check for breakthrough moments (very high values)
            if value > 0.9 and dimension in profile.baseline_measurements:
                baseline = profile.baseline_measurements[dimension]
                if value - baseline > 0.3:
                    milestone = {
                        'type': 'breakthrough_moment',
                        'dimension': dimension.value,
                        'value': value,
                        'baseline': baseline,
                        'timestamp': measurement.measurement_timestamp
                    }
                    
                    profile.significant_milestones.append(milestone)
            
        except Exception as e:
            logger.error(f"Error checking for milestones: {str(e)}")
    
    async def _check_for_significant_events(self, measurement: HealingMeasurement):
        """Check for significant events requiring attention"""
        try:
            # Check for crisis indicators
            if measurement.value < 0.2 and measurement.significance_level > 0.7:
                crisis_event = {
                    'user_id': measurement.user_id,
                    'dimension': measurement.healing_dimension.value,
                    'value': measurement.value,
                    'timestamp': measurement.measurement_timestamp,
                    'urgency': 'high' if measurement.value < 0.1 else 'moderate'
                }
                
                self.crisis_interventions_measured += 1
                
                # Trigger callbacks
                for callback in self.crisis_detected_callbacks:
                    try:
                        await callback(crisis_event)
                    except Exception as e:
                        logger.error(f"Crisis detected callback failed: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error checking for significant events: {str(e)}")
    
    async def _analysis_processing_loop(self):
        """Analysis processing loop for deeper measurement analysis"""
        try:
            while self.processing_active:
                try:
                    # Wait for measurement with timeout
                    measurement = await asyncio.wait_for(self.analysis_queue.get(), timeout=1.0)
                    
                    # Perform deeper analysis
                    await self._perform_deep_analysis(measurement)
                    
                    # Mark queue task as done
                    self.analysis_queue.task_done()
                    
                except asyncio.TimeoutError:
                    # No measurements in queue, continue waiting
                    continue
                except Exception as e:
                    logger.error(f"Error in analysis processing loop: {str(e)}")
                    await asyncio.sleep(1)
                    
        except Exception as e:
            logger.error(f"Fatal error in analysis processing loop: {str(e)}")
    
    async def _perform_deep_analysis(self, measurement: HealingMeasurement):
        """Perform deep analysis on measurement"""
        try:
            # Analyze patterns across dimensions
            await self._analyze_cross_dimensional_patterns(measurement)
            
            # Analyze community impact
            await self._analyze_community_impact(measurement)
            
            # Analyze cultural patterns
            await self._analyze_cultural_patterns(measurement)
            
            # Update community metrics
            await self._update_community_metrics(measurement)
            
        except Exception as e:
            logger.error(f"Error performing deep analysis: {str(e)}")
    
    async def _analyze_cross_dimensional_patterns(self, measurement: HealingMeasurement):
        """Analyze patterns across healing dimensions"""
        try:
            user_id = measurement.user_id
            
            if user_id in self.user_healing_profiles:
                profile = self.user_healing_profiles[user_id]
                
                # Look for correlated improvements
                current_dimension = measurement.healing_dimension
                current_value = measurement.value
                
                for other_dimension, other_value in profile.healing_dimensions.items():
                    if other_dimension != current_dimension:
                        # Check for correlation
                        correlation = await self._calculate_dimension_correlation(
                            profile, current_dimension, other_dimension
                        )
                        
                        if correlation > 0.7:
                            logger.debug(f"High correlation detected between {current_dimension} and {other_dimension}")
            
        except Exception as e:
            logger.error(f"Error analyzing cross-dimensional patterns: {str(e)}")
    
    async def _calculate_dimension_correlation(self, profile: HealingProfile,
                                             dim1: HealingDimension, dim2: HealingDimension) -> float:
        """Calculate correlation between two healing dimensions"""
        try:
            if dim1 in profile.growth_trajectory and dim2 in profile.growth_trajectory:
                traj1 = profile.growth_trajectory[dim1]
                traj2 = profile.growth_trajectory[dim2]
                
                # Align trajectories to same length
                min_length = min(len(traj1), len(traj2))
                if min_length < 3:
                    return 0.0
                
                aligned_traj1 = traj1[-min_length:]
                aligned_traj2 = traj2[-min_length:]
                
                # Calculate correlation
                correlation = np.corrcoef(aligned_traj1, aligned_traj2)[0, 1]
                
                return abs(correlation) if not np.isnan(correlation) else 0.0
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating dimension correlation: {str(e)}")
            return 0.0
    
    async def _analyze_community_impact(self, measurement: HealingMeasurement):
        """Analyze community impact of individual healing"""
        try:
            # Individual healing contributes to community healing
            if measurement.value > 0.7 and measurement.baseline_comparison and measurement.baseline_comparison > 0.1:
                self.community_healing_events += 1
                
                # Track community metrics
                if 'individual_healing_contributions' not in self.community_metrics:
                    self.community_metrics['individual_healing_contributions'] = 0
                
                self.community_metrics['individual_healing_contributions'] += 1
            
        except Exception as e:
            logger.error(f"Error analyzing community impact: {str(e)}")
    
    async def _analyze_cultural_patterns(self, measurement: HealingMeasurement):
        """Analyze cultural patterns in healing"""
        try:
            # Track cultural healing patterns
            for cultural_consideration in measurement.cultural_considerations:
                if 'cultural_healing_patterns' not in self.community_metrics:
                    self.community_metrics['cultural_healing_patterns'] = defaultdict(int)
                
                self.community_metrics['cultural_healing_patterns'][cultural_consideration] += 1
            
        except Exception as e:
            logger.error(f"Error analyzing cultural patterns: {str(e)}")
    
    async def _update_community_metrics(self, measurement: HealingMeasurement):
        """Update community-level metrics"""
        try:
            # Update aggregate metrics
            dimension = measurement.healing_dimension
            
            if 'dimension_averages' not in self.community_metrics:
                self.community_metrics['dimension_averages'] = {}
            
            if dimension.value not in self.community_metrics['dimension_averages']:
                self.community_metrics['dimension_averages'][dimension.value] = []
            
            self.community_metrics['dimension_averages'][dimension.value].append(measurement.value)
            
            # Keep manageable size
            if len(self.community_metrics['dimension_averages'][dimension.value]) > 1000:
                self.community_metrics['dimension_averages'][dimension.value] = \
                    self.community_metrics['dimension_averages'][dimension.value][-1000:]
            
        except Exception as e:
            logger.error(f"Error updating community metrics: {str(e)}")
    
    async def _trend_analysis_loop(self):
        """Trend analysis loop"""
        try:
            while self.processing_active:
                try:
                    await asyncio.sleep(3600)  # Run every hour
                    
                    # Analyze trends for all users
                    await self._analyze_user_trends()
                    
                    # Analyze community trends
                    await self._analyze_community_trends()
                    
                except Exception as e:
                    logger.error(f"Error in trend analysis loop: {str(e)}")
                    await asyncio.sleep(3600)
                    
        except Exception as e:
            logger.error(f"Fatal error in trend analysis loop: {str(e)}")
    
    async def _analyze_user_trends(self):
        """Analyze trends for individual users"""
        try:
            for user_id, profile in self.user_healing_profiles.items():
                # Analyze each dimension's trend
                for dimension, trajectory in profile.growth_trajectory.items():
                    if len(trajectory) >= 5:  # Need minimum data points
                        trend_analysis = await self._perform_trend_analysis(trajectory)
                        
                        # Store trend analysis results
                        # In production, would store in user profile
                        
        except Exception as e:
            logger.error(f"Error analyzing user trends: {str(e)}")
    
    async def _perform_trend_analysis(self, trajectory: List[float]) -> Dict[str, Any]:
        """Perform trend analysis on a trajectory"""
        try:
            if len(trajectory) < 3:
                return {}
            
            # Calculate short-term trend (last 7 data points)
            short_term_data = trajectory[-7:] if len(trajectory) >= 7 else trajectory
            short_term_trend = await self._calculate_trend_slope(short_term_data)
            
            # Calculate long-term trend (all data)
            long_term_trend = await self._calculate_trend_slope(trajectory)
            
            # Detect patterns
            patterns = await self._detect_trajectory_patterns(trajectory)
            
            return {
                'short_term_trend': short_term_trend,
                'long_term_trend': long_term_trend,
                'patterns': patterns,
                'volatility': statistics.stdev(trajectory) if len(trajectory) > 1 else 0.0,
                'current_momentum': short_term_trend,
                'overall_direction': 'improving' if long_term_trend > 0.01 else 'declining' if long_term_trend < -0.01 else 'stable'
            }
            
        except Exception as e:
            logger.error(f"Error performing trend analysis: {str(e)}")
            return {}
    
    async def _calculate_trend_slope(self, data: List[float]) -> float:
        """Calculate trend slope using linear regression"""
        try:
            if len(data) < 2:
                return 0.0
            
            x = list(range(len(data)))
            y = data
            
            # Simple linear regression
            n = len(data)
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(x[i] * y[i] for i in range(n))
            sum_x2 = sum(x[i] ** 2 for i in range(n))
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
            
            return slope
            
        except Exception as e:
            logger.error(f"Error calculating trend slope: {str(e)}")
            return 0.0
    
    async def _detect_trajectory_patterns(self, trajectory: List[float]) -> List[str]:
        """Detect patterns in trajectory data"""
        try:
            patterns = []
            
            if len(trajectory) < 5:
                return patterns
            
            # Detect breakthrough patterns (sudden improvements)
            for i in range(1, len(trajectory)):
                improvement = trajectory[i] - trajectory[i-1]
                if improvement > 0.2:
                    patterns.append('breakthrough_moment')
                    break
            
            # Detect plateau patterns
            recent_values = trajectory[-5:]
            if len(recent_values) == 5:
                if max(recent_values) - min(recent_values) < 0.05:
                    patterns.append('plateau_pattern')
            
            # Detect recovery patterns (improvement after decline)
            if len(trajectory) >= 7:
                mid_point = len(trajectory) // 2
                first_half_avg = statistics.mean(trajectory[:mid_point])
                second_half_avg = statistics.mean(trajectory[mid_point:])
                
                if second_half_avg > first_half_avg + 0.1:
                    patterns.append('recovery_pattern')
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting trajectory patterns: {str(e)}")
            return []
    
    async def _analyze_community_trends(self):
        """Analyze community-wide trends"""
        try:
            # Analyze dimension averages
            if 'dimension_averages' in self.community_metrics:
                for dimension, values in self.community_metrics['dimension_averages'].items():
                    if len(values) >= 10:
                        avg_value = statistics.mean(values[-100:])  # Last 100 measurements
                        
                        # Store community trend
                        if 'community_trends' not in self.community_metrics:
                            self.community_metrics['community_trends'] = {}
                        
                        self.community_metrics['community_trends'][dimension] = avg_value
            
        except Exception as e:
            logger.error(f"Error analyzing community trends: {str(e)}")
    
    async def _impact_reporting_loop(self):
        """Impact reporting loop"""
        try:
            while self.processing_active:
                try:
                    await asyncio.sleep(86400)  # Run daily
                    
                    # Generate daily impact report
                    report = await self._generate_impact_report()
                    
                    if report:
                        self.impact_reports.append(report)
                        
                        # Trigger callbacks
                        for callback in self.impact_report_callbacks:
                            try:
                                await callback(report)
                            except Exception as e:
                                logger.error(f"Impact report callback failed: {str(e)}")
                    
                except Exception as e:
                    logger.error(f"Error in impact reporting loop: {str(e)}")
                    await asyncio.sleep(86400)
                    
        except Exception as e:
            logger.error(f"Fatal error in impact reporting loop: {str(e)}")
    
    async def _generate_impact_report(self) -> Optional[ImpactReport]:
        """Generate comprehensive impact report"""
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=1)  # Daily report
            
            # Filter measurements for time period
            period_measurements = [
                m for m in self.measurement_history
                if start_time <= m.measurement_timestamp <= end_time
            ]
            
            if not period_measurements:
                return None
            
            # Calculate user population
            user_population = len(set(m.user_id for m in period_measurements))
            
            # Calculate healing dimension summary
            dimension_summary = await self._calculate_dimension_summary(period_measurements)
            
            # Calculate community impact metrics
            community_impact = await self._calculate_community_impact_metrics(period_measurements)
            
            # Calculate crisis intervention outcomes
            crisis_outcomes = await self._calculate_crisis_outcomes(period_measurements)
            
            # Calculate connection quality metrics
            connection_metrics = await self._calculate_connection_metrics(period_measurements)
            
            # Calculate cultural inclusion metrics
            cultural_metrics = await self._calculate_cultural_metrics(period_measurements)
            
            # Calculate trauma-informed effectiveness
            trauma_effectiveness = await self._calculate_trauma_effectiveness(period_measurements)
            
            # Identify significant achievements
            achievements = await self._identify_significant_achievements(period_measurements)
            
            # Identify areas for improvement
            improvements = await self._identify_improvement_areas(period_measurements)
            
            report = ImpactReport(
                report_id=f"impact_report_{end_time.isoformat()}_{id(self)}",
                time_period=(start_time, end_time),
                user_population=user_population,
                healing_dimension_summary=dimension_summary,
                community_impact_metrics=community_impact,
                crisis_intervention_outcomes=crisis_outcomes,
                connection_quality_metrics=connection_metrics,
                cultural_inclusion_metrics=cultural_metrics,
                trauma_informed_effectiveness=trauma_effectiveness,
                significant_achievements=achievements,
                areas_for_improvement=improvements,
                methodology_notes=[
                    'Healing-focused metrics prioritize wellbeing over engagement',
                    'Privacy-preserving measurement with user consent',
                    'Trauma-informed approaches prevent re-traumatization',
                    'Cultural sensitivity guides interpretation'
                ],
                ethical_considerations=[
                    'User privacy and consent paramount',
                    'No measurement without explicit consent',
                    'Cultural sensitivity in interpretation',
                    'Trauma-informed measurement practices'
                ],
                generated_at=datetime.utcnow()
            )
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating impact report: {str(e)}")
            return None
    
    async def _calculate_dimension_summary(self, measurements: List[HealingMeasurement]) -> Dict[HealingDimension, Dict[str, float]]:
        """Calculate summary statistics for each healing dimension"""
        try:
            dimension_summary = {}
            
            # Group measurements by dimension
            by_dimension = defaultdict(list)
            for measurement in measurements:
                by_dimension[measurement.healing_dimension].append(measurement.value)
            
            # Calculate summary statistics
            for dimension, values in by_dimension.items():
                if values:
                    dimension_summary[dimension] = {
                        'average': statistics.mean(values),
                        'median': statistics.median(values),
                        'std_dev': statistics.stdev(values) if len(values) > 1 else 0.0,
                        'min': min(values),
                        'max': max(values),
                        'count': len(values),
                        'improvement_rate': sum(1 for v in values if v > 0.6) / len(values)
                    }
            
            return dimension_summary
            
        except Exception as e:
            logger.error(f"Error calculating dimension summary: {str(e)}")
            return {}
    
    async def _calculate_community_impact_metrics(self, measurements: List[HealingMeasurement]) -> Dict[str, float]:
        """Calculate community impact metrics"""
        try:
            if not measurements:
                return {}
            
            # Community healing score
            all_values = [m.value for m in measurements]
            community_healing_score = statistics.mean(all_values)
            
            # Improvement rate
            improvement_measurements = [m for m in measurements if m.baseline_comparison and m.baseline_comparison > 0]
            improvement_rate = len(improvement_measurements) / len(measurements)
            
            # Significant positive changes
            significant_improvements = [m for m in measurements if m.baseline_comparison and m.baseline_comparison > 0.2]
            significant_improvement_rate = len(significant_improvements) / len(measurements)
            
            # Crisis intervention effectiveness
            crisis_measurements = [m for m in measurements if m.value < 0.3]
            crisis_rate = len(crisis_measurements) / len(measurements)
            
            return {
                'community_healing_score': community_healing_score,
                'improvement_rate': improvement_rate,
                'significant_improvement_rate': significant_improvement_rate,
                'crisis_intervention_rate': crisis_rate,
                'measurement_consent_rate': sum(1 for m in measurements if m.consent_given) / len(measurements),
                'cultural_adaptation_rate': sum(1 for m in measurements if m.cultural_considerations) / len(measurements)
            }
            
        except Exception as e:
            logger.error(f"Error calculating community impact metrics: {str(e)}")
            return {}
    
    async def _calculate_crisis_outcomes(self, measurements: List[HealingMeasurement]) -> Dict[str, Any]:
        """Calculate crisis intervention outcomes"""
        try:
            crisis_measurements = [m for m in measurements if m.value < 0.3]
            
            if not crisis_measurements:
                return {'crisis_interventions': 0}
            
            # Recovery tracking
            recovery_cases = 0
            for crisis_measurement in crisis_measurements:
                # Check if user showed improvement after crisis
                user_id = crisis_measurement.user_id
                later_measurements = [
                    m for m in measurements
                    if m.user_id == user_id and 
                    m.measurement_timestamp > crisis_measurement.measurement_timestamp and
                    m.healing_dimension == crisis_measurement.healing_dimension
                ]
                
                if later_measurements and any(m.value > crisis_measurement.value + 0.2 for m in later_measurements):
                    recovery_cases += 1
            
            recovery_rate = recovery_cases / len(crisis_measurements) if crisis_measurements else 0.0
            
            return {
                'crisis_interventions': len(crisis_measurements),
                'recovery_rate': recovery_rate,
                'average_crisis_severity': statistics.mean([m.value for m in crisis_measurements]),
                'crisis_response_time_quality': 0.8  # Would be calculated from actual response times
            }
            
        except Exception as e:
            logger.error(f"Error calculating crisis outcomes: {str(e)}")
            return {}
    
    async def _calculate_connection_metrics(self, measurements: List[HealingMeasurement]) -> Dict[str, float]:
        """Calculate connection quality metrics"""
        try:
            connection_measurements = [
                m for m in measurements 
                if m.healing_dimension == HealingDimension.AUTHENTIC_CONNECTION
            ]
            
            if not connection_measurements:
                return {}
            
            values = [m.value for m in connection_measurements]
            
            return {
                'average_connection_quality': statistics.mean(values),
                'connection_improvement_rate': sum(1 for m in connection_measurements if m.baseline_comparison and m.baseline_comparison > 0) / len(connection_measurements),
                'deep_connection_rate': sum(1 for v in values if v > 0.8) / len(values)
            }
            
        except Exception as e:
            logger.error(f"Error calculating connection metrics: {str(e)}")
            return {}
    
    async def _calculate_cultural_metrics(self, measurements: List[HealingMeasurement]) -> Dict[str, float]:
        """Calculate cultural inclusion metrics"""
        try:
            cultural_measurements = [m for m in measurements if m.cultural_considerations]
            
            if not measurements:
                return {}
            
            cultural_adaptation_rate = len(cultural_measurements) / len(measurements)
            
            # Cultural healing effectiveness
            cultural_effectiveness = 0.0
            if cultural_measurements:
                cultural_values = [m.value for m in cultural_measurements]
                cultural_effectiveness = statistics.mean(cultural_values)
            
            return {
                'cultural_adaptation_rate': cultural_adaptation_rate,
                'cultural_healing_effectiveness': cultural_effectiveness,
                'cultural_sensitivity_score': 0.85  # Would be calculated from cultural feedback
            }
            
        except Exception as e:
            logger.error(f"Error calculating cultural metrics: {str(e)}")
            return {}
    
    async def _calculate_trauma_effectiveness(self, measurements: List[HealingMeasurement]) -> Dict[str, float]:
        """Calculate trauma-informed approach effectiveness"""
        try:
            trauma_measurements = [m for m in measurements if m.trauma_informed_adjustments]
            
            if not measurements:
                return {}
            
            trauma_informed_rate = len(trauma_measurements) / len(measurements)
            
            # Trauma healing effectiveness
            trauma_effectiveness = 0.0
            if trauma_measurements:
                trauma_values = [m.value for m in trauma_measurements]
                trauma_effectiveness = statistics.mean(trauma_values)
            
            return {
                'trauma_informed_rate': trauma_informed_rate,
                'trauma_healing_effectiveness': trauma_effectiveness,
                'trauma_safety_score': 0.9  # Would be calculated from safety feedback
            }
            
        except Exception as e:
            logger.error(f"Error calculating trauma effectiveness: {str(e)}")
            return {}
    
    async def _identify_significant_achievements(self, measurements: List[HealingMeasurement]) -> List[Dict[str, Any]]:
        """Identify significant achievements from measurements"""
        try:
            achievements = []
            
            # Major improvements
            major_improvements = [
                m for m in measurements 
                if m.baseline_comparison and m.baseline_comparison > 0.3
            ]
            
            if major_improvements:
                achievements.append({
                    'type': 'major_healing_improvements',
                    'count': len(major_improvements),
                    'description': f'{len(major_improvements)} users experienced major healing improvements'
                })
            
            # High wellbeing achievements
            high_wellbeing = [m for m in measurements if m.value > 0.9]
            if high_wellbeing:
                achievements.append({
                    'type': 'high_wellbeing_achievements',
                    'count': len(high_wellbeing),
                    'description': f'{len(high_wellbeing)} measurements showed exceptional wellbeing'
                })
            
            # Crisis recovery achievements
            crisis_recoveries = [
                m for m in measurements 
                if m.value > 0.6 and m.baseline_comparison and m.baseline_comparison > 0.4
            ]
            
            if crisis_recoveries:
                achievements.append({
                    'type': 'crisis_recovery_achievements',
                    'count': len(crisis_recoveries),
                    'description': f'{len(crisis_recoveries)} users showed remarkable recovery from crisis'
                })
            
            return achievements
            
        except Exception as e:
            logger.error(f"Error identifying significant achievements: {str(e)}")
            return []
    
    async def _identify_improvement_areas(self, measurements: List[HealingMeasurement]) -> List[Dict[str, Any]]:
        """Identify areas for improvement"""
        try:
            improvements = []
            
            # Check for dimensions with concerning trends
            by_dimension = defaultdict(list)
            for measurement in measurements:
                by_dimension[measurement.healing_dimension].append(measurement.value)
            
            for dimension, values in by_dimension.items():
                avg_value = statistics.mean(values)
                if avg_value < 0.5:
                    improvements.append({
                        'area': f'{dimension.value}_support',
                        'priority': 'high' if avg_value < 0.3 else 'medium',
                        'description': f'{dimension.value} shows concerning patterns requiring attention',
                        'average_score': avg_value
                    })
            
            # Check for low consent rates
            consent_rate = sum(1 for m in measurements if m.consent_given) / len(measurements) if measurements else 0
            if consent_rate < 0.8:
                improvements.append({
                    'area': 'consent_engagement',
                    'priority': 'high',
                    'description': 'Improve user consent engagement for measurement',
                    'current_rate': consent_rate
                })
            
            return improvements
            
        except Exception as e:
            logger.error(f"Error identifying improvement areas: {str(e)}")
            return []
    
    # Callback management
    def add_healing_milestone_callback(self, callback: Callable):
        """Add callback for healing milestones"""
        self.healing_milestone_callbacks.append(callback)
    
    def add_crisis_detected_callback(self, callback: Callable):
        """Add callback for crisis detection"""
        self.crisis_detected_callbacks.append(callback)
    
    def add_community_achievement_callback(self, callback: Callable):
        """Add callback for community achievements"""
        self.community_achievement_callbacks.append(callback)
    
    def add_impact_report_callback(self, callback: Callable):
        """Add callback for impact reports"""
        self.impact_report_callbacks.append(callback)
    
    # Analytics and reporting
    def get_healing_analytics(self) -> Dict[str, Any]:
        """Get comprehensive healing analytics"""
        try:
            if not self.measurement_history:
                return {
                    'total_measurements': 0,
                    'healing_improvements': 0,
                    'crisis_interventions': 0,
                    'community_healing_events': 0,
                    'active_users': 0
                }
            
            # Calculate overall metrics
            recent_measurements = [
                m for m in self.measurement_history
                if m.measurement_timestamp > datetime.utcnow() - timedelta(days=30)
            ]
            
            avg_healing_score = statistics.mean([m.value for m in recent_measurements]) if recent_measurements else 0.0
            
            improvement_rate = (
                sum(1 for m in recent_measurements if m.baseline_comparison and m.baseline_comparison > 0) / 
                len(recent_measurements) if recent_measurements else 0.0
            )
            
            return {
                'total_measurements': self.total_measurements_processed,
                'healing_improvements': self.healing_improvements_detected,
                'crisis_interventions': self.crisis_interventions_measured,
                'community_healing_events': self.community_healing_events,
                'active_users': len(self.user_healing_profiles),
                'average_healing_score': avg_healing_score,
                'improvement_rate': improvement_rate,
                'consent_rate': sum(1 for m in recent_measurements if m.consent_given) / len(recent_measurements) if recent_measurements else 0.0,
                'cultural_adaptation_rate': sum(1 for m in recent_measurements if m.cultural_considerations) / len(recent_measurements) if recent_measurements else 0.0,
                'trauma_informed_rate': sum(1 for m in recent_measurements if m.trauma_informed_adjustments) / len(recent_measurements) if recent_measurements else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error generating healing analytics: {str(e)}")
            return {}
    
    def get_user_healing_summary(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get healing summary for a specific user"""
        try:
            if user_id not in self.user_healing_profiles:
                return None
            
            profile = self.user_healing_profiles[user_id]
            
            # Calculate overall healing score
            if profile.healing_dimensions:
                overall_score = statistics.mean(profile.healing_dimensions.values())
            else:
                overall_score = 0.0
            
            # Calculate improvement trends
            improvements = 0
            for dimension, trajectory in profile.growth_trajectory.items():
                if len(trajectory) >= 2:
                    recent_trend = trajectory[-1] - trajectory[-2]
                    if recent_trend > 0.05:
                        improvements += 1
            
            return {
                'user_id': user_id,
                'overall_healing_score': overall_score,
                'dimensions_tracked': len(profile.healing_dimensions),
                'improving_dimensions': improvements,
                'significant_milestones': len(profile.significant_milestones),
                'crisis_events': len(profile.crisis_events),
                'last_updated': profile.last_updated.isoformat(),
                'healing_journey_length': (datetime.utcnow() - profile.last_updated).days if profile.last_updated else 0
            }
            
        except Exception as e:
            logger.error(f"Error generating user healing summary: {str(e)}")
            return None