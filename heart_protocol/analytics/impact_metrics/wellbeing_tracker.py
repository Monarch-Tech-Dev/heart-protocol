"""
Wellbeing Tracker

Real-time tracking and monitoring of user wellbeing indicators with
trauma-informed, culturally sensitive approaches to mental health measurement.
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


class WellbeingIndicator(Enum):
    """Key indicators of user wellbeing"""
    MOOD_STABILITY = "mood_stability"                   # Emotional stability over time
    STRESS_LEVELS = "stress_levels"                     # Stress and anxiety indicators
    SLEEP_QUALITY = "sleep_quality"                     # Sleep pattern health
    SOCIAL_CONNECTION = "social_connection"             # Quality of social interactions
    ENERGY_LEVELS = "energy_levels"                     # Physical and mental energy
    COPING_SKILLS = "coping_skills"                     # Adaptive coping mechanisms
    SELF_CARE_PRACTICES = "self_care_practices"         # Self-care and maintenance
    COGNITIVE_CLARITY = "cognitive_clarity"             # Mental clarity and focus
    EMOTIONAL_REGULATION = "emotional_regulation"       # Emotional management skills
    LIFE_SATISFACTION = "life_satisfaction"             # Overall satisfaction
    PURPOSE_FULFILLMENT = "purpose_fulfillment"         # Sense of meaning
    SAFETY_SECURITY = "safety_security"                 # Feeling safe and secure


class ProgressDirection(Enum):
    """Direction of wellbeing progress"""
    SIGNIFICANT_IMPROVEMENT = "significant_improvement"  # Major positive change
    GRADUAL_IMPROVEMENT = "gradual_improvement"          # Steady positive progress
    STABLE_MAINTAINING = "stable_maintaining"            # Maintaining current level
    MINOR_FLUCTUATION = "minor_fluctuation"              # Normal variation
    CONCERNING_DECLINE = "concerning_decline"            # Worrying decrease
    CRISIS_DETERIORATION = "crisis_deterioration"        # Rapid decline requiring intervention


class WellbeingRisk(Enum):
    """Risk levels for wellbeing concerns"""
    LOW_RISK = "low_risk"                               # Stable, low risk
    MODERATE_RISK = "moderate_risk"                     # Some concerning patterns
    HIGH_RISK = "high_risk"                             # Significant risk factors
    CRISIS_RISK = "crisis_risk"                         # Immediate intervention needed


class MeasurementContext(Enum):
    """Context for wellbeing measurements"""
    DAILY_CHECKIN = "daily_checkin"                     # Regular daily assessment
    TRIGGERED_ASSESSMENT = "triggered_assessment"        # Assessment triggered by events
    CRISIS_EVALUATION = "crisis_evaluation"             # Crisis-triggered evaluation
    MILESTONE_REVIEW = "milestone_review"               # Progress milestone review
    PROFESSIONAL_ASSESSMENT = "professional_assessment" # Professional evaluation
    PEER_FEEDBACK = "peer_feedback"                     # Community feedback
    SELF_REFLECTION = "self_reflection"                 # User self-assessment


@dataclass
class WellbeingMeasurement:
    """Individual wellbeing measurement"""
    measurement_id: str
    user_id: str
    indicator: WellbeingIndicator
    value: float  # 0-100 scale
    confidence_level: float
    measurement_context: MeasurementContext
    contextual_factors: Dict[str, Any]
    cultural_considerations: List[str]
    trauma_informed_adjustments: List[str]
    privacy_level: str
    consent_given: bool
    measurement_timestamp: datetime
    baseline_comparison: Optional[float]
    trend_direction: ProgressDirection
    risk_level: WellbeingRisk
    intervention_recommendations: List[str]
    follow_up_needed: bool


@dataclass
class WellbeingProfile:
    """Comprehensive wellbeing profile for a user"""
    user_id: str
    current_indicators: Dict[WellbeingIndicator, float]
    baseline_indicators: Dict[WellbeingIndicator, float]
    trend_history: Dict[WellbeingIndicator, List[float]]
    risk_factors: List[str]
    protective_factors: List[str]
    intervention_history: List[Dict[str, Any]]
    wellness_goals: List[Dict[str, Any]]
    support_preferences: Dict[str, Any]
    cultural_context: List[str]
    trauma_considerations: List[str]
    accessibility_needs: List[str]
    privacy_preferences: Dict[str, Any]
    last_updated: datetime
    overall_wellbeing_score: float
    progress_direction: ProgressDirection
    current_risk_level: WellbeingRisk


@dataclass
class WellbeingAlert:
    """Alert for concerning wellbeing patterns"""
    alert_id: str
    user_id: str
    alert_type: str
    severity_level: WellbeingRisk
    triggered_indicators: List[WellbeingIndicator]
    trend_analysis: Dict[str, Any]
    immediate_concerns: List[str]
    recommended_interventions: List[str]
    escalation_required: bool
    cultural_considerations: List[str]
    trauma_informed_response: List[str]
    privacy_requirements: List[str]
    created_at: datetime
    requires_human_review: bool


class WellbeingTracker:
    """
    Real-time wellbeing tracking system that monitors user mental health
    indicators with trauma-informed, culturally sensitive approaches.
    
    Core Principles:
    - User wellbeing and safety are the highest priorities
    - Trauma-informed measurement that doesn't re-traumatize
    - Cultural sensitivity in assessment and interpretation
    - Privacy and consent are paramount
    - Holistic view of wellbeing, not just absence of symptoms
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.user_profiles: Dict[str, WellbeingProfile] = {}
        self.measurement_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.active_alerts: Dict[str, List[WellbeingAlert]] = defaultdict(list)
        self.intervention_callbacks: List[Callable] = []
        self.cultural_adaptations: Dict[str, Any] = {}
        self.trauma_protocols: Dict[str, Any] = {}
        self.baseline_calculators: Dict[WellbeingIndicator, Callable] = {}
        self.trend_analyzers: Dict[WellbeingIndicator, Callable] = {}
        self.risk_assessors: List[Callable] = []
        
        # Initialize measurement protocols
        self._setup_measurement_protocols()
        self._setup_cultural_adaptations()
        self._setup_trauma_protocols()
        self._setup_risk_assessment()
    
    def _setup_measurement_protocols(self):
        """Setup measurement protocols for different indicators"""
        self.measurement_protocols = {
            WellbeingIndicator.MOOD_STABILITY: {
                'frequency': 'daily',
                'sensitivity': 'high',
                'trauma_considerations': ['mood_tracking_trauma', 'emotional_overwhelm'],
                'cultural_factors': ['emotional_expression_norms', 'collectivist_vs_individualist']
            },
            WellbeingIndicator.STRESS_LEVELS: {
                'frequency': 'real_time',
                'sensitivity': 'very_high',
                'trauma_considerations': ['hypervigilance', 'stress_response_triggers'],
                'cultural_factors': ['stress_stigma', 'coping_mechanisms']
            },
            WellbeingIndicator.SOCIAL_CONNECTION: {
                'frequency': 'weekly',
                'sensitivity': 'medium',
                'trauma_considerations': ['social_trauma', 'isolation_patterns'],
                'cultural_factors': ['family_structure', 'community_orientation']
            },
            WellbeingIndicator.SAFETY_SECURITY: {
                'frequency': 'real_time',
                'sensitivity': 'very_high',
                'trauma_considerations': ['safety_trauma', 'hypervigilance'],
                'cultural_factors': ['safety_perceptions', 'community_safety']
            }
        }
    
    def _setup_cultural_adaptations(self):
        """Setup cultural adaptations for wellbeing measurement"""
        self.cultural_adaptations = {
            'individualistic_cultures': {
                'measurement_focus': ['personal_achievement', 'individual_growth', 'self_reliance'],
                'indicator_weights': {
                    WellbeingIndicator.PURPOSE_FULFILLMENT: 1.2,
                    WellbeingIndicator.COGNITIVE_CLARITY: 1.1
                }
            },
            'collectivistic_cultures': {
                'measurement_focus': ['family_harmony', 'community_wellbeing', 'social_roles'],
                'indicator_weights': {
                    WellbeingIndicator.SOCIAL_CONNECTION: 1.3,
                    WellbeingIndicator.SAFETY_SECURITY: 1.2
                }
            },
            'high_context_cultures': {
                'measurement_approach': 'indirect_observation',
                'communication_style': 'implicit_understanding',
                'indicator_interpretation': 'contextual_meaning'
            },
            'trauma_informed_cultures': {
                'measurement_frequency': 'reduced',
                'consent_requirements': 'enhanced',
                'safety_prioritization': 'maximum'
            }
        }
    
    def _setup_trauma_protocols(self):
        """Setup trauma-informed measurement protocols"""
        self.trauma_protocols = {
            'consent_verification': {
                'frequency': 'before_each_measurement',
                'opt_out_always_available': True,
                'explanation_required': True,
                'anonymization_options': True
            },
            'trauma_triggers': {
                'identification_methods': ['content_analysis', 'behavioral_patterns'],
                'response_protocols': ['immediate_pause', 'gentle_transition', 'support_offer'],
                'escalation_criteria': ['distress_indicators', 'crisis_patterns']
            },
            'safety_prioritization': {
                'measurement_adjustments': 'trauma_sensitive',
                'timing_considerations': 'user_controlled',
                'content_filtering': 'protective'
            }
        }
    
    def _setup_risk_assessment(self):
        """Setup risk assessment algorithms"""
        self.risk_assessment_criteria = {
            WellbeingRisk.CRISIS_RISK: {
                'indicators': [
                    {'indicator': WellbeingIndicator.MOOD_STABILITY, 'threshold': 20, 'direction': 'below'},
                    {'indicator': WellbeingIndicator.SAFETY_SECURITY, 'threshold': 25, 'direction': 'below'},
                    {'indicator': WellbeingIndicator.STRESS_LEVELS, 'threshold': 85, 'direction': 'above'}
                ],
                'trend_patterns': ['rapid_decline', 'sustained_low', 'erratic_fluctuation'],
                'intervention_urgency': 'immediate'
            },
            WellbeingRisk.HIGH_RISK: {
                'indicators': [
                    {'indicator': WellbeingIndicator.MOOD_STABILITY, 'threshold': 35, 'direction': 'below'},
                    {'indicator': WellbeingIndicator.SOCIAL_CONNECTION, 'threshold': 30, 'direction': 'below'},
                    {'indicator': WellbeingIndicator.COPING_SKILLS, 'threshold': 40, 'direction': 'below'}
                ],
                'trend_patterns': ['gradual_decline', 'concerning_fluctuation'],
                'intervention_urgency': 'priority'
            }
        }
    
    async def track_wellbeing(self, user_id: str, indicator: WellbeingIndicator, 
                            value: float, context: MeasurementContext,
                            contextual_factors: Optional[Dict[str, Any]] = None) -> WellbeingMeasurement:
        """Track a wellbeing measurement with trauma-informed approaches"""
        try:
            # Verify consent and cultural considerations
            if not await self._verify_measurement_consent(user_id, indicator, context):
                logger.info(f"Measurement consent not verified for user {user_id}, indicator {indicator}")
                return None
            
            # Apply cultural adaptations
            adapted_value = await self._apply_cultural_adaptations(user_id, indicator, value)
            
            # Apply trauma-informed adjustments
            trauma_adjustments = await self._apply_trauma_considerations(user_id, indicator, adapted_value)
            
            # Create measurement
            measurement = WellbeingMeasurement(
                measurement_id=f"wb_{user_id}_{indicator.value}_{datetime.now().isoformat()}",
                user_id=user_id,
                indicator=indicator,
                value=adapted_value,
                confidence_level=await self._calculate_confidence(user_id, indicator, contextual_factors),
                measurement_context=context,
                contextual_factors=contextual_factors or {},
                cultural_considerations=await self._get_cultural_considerations(user_id),
                trauma_informed_adjustments=trauma_adjustments,
                privacy_level=await self._determine_privacy_level(user_id, indicator),
                consent_given=True,
                measurement_timestamp=datetime.now(),
                baseline_comparison=await self._calculate_baseline_comparison(user_id, indicator, adapted_value),
                trend_direction=await self._analyze_trend_direction(user_id, indicator, adapted_value),
                risk_level=await self._assess_risk_level(user_id, indicator, adapted_value),
                intervention_recommendations=await self._generate_intervention_recommendations(user_id, indicator, adapted_value),
                follow_up_needed=await self._determine_follow_up_need(user_id, indicator, adapted_value)
            )
            
            # Store measurement
            self.measurement_history[user_id].append(measurement)
            
            # Update user profile
            await self._update_wellbeing_profile(user_id, measurement)
            
            # Check for alerts
            await self._check_wellbeing_alerts(user_id, measurement)
            
            logger.info(f"Wellbeing measurement tracked: {user_id}, {indicator}, value: {adapted_value}")
            return measurement
            
        except Exception as e:
            logger.error(f"Error tracking wellbeing: {e}")
            return None
    
    async def _verify_measurement_consent(self, user_id: str, indicator: WellbeingIndicator, 
                                        context: MeasurementContext) -> bool:
        """Verify user consent for wellbeing measurement"""
        # Check if user has provided consent for this type of measurement
        profile = self.user_profiles.get(user_id)
        if not profile:
            return False
        
        # Check general consent
        if not profile.privacy_preferences.get('wellbeing_tracking_consent', False):
            return False
        
        # Check specific indicator consent
        indicator_consent = profile.privacy_preferences.get('indicator_consent', {})
        if not indicator_consent.get(indicator.value, True):
            return False
        
        # Check context-specific consent
        context_consent = profile.privacy_preferences.get('context_consent', {})
        if not context_consent.get(context.value, True):
            return False
        
        return True
    
    async def _apply_cultural_adaptations(self, user_id: str, indicator: WellbeingIndicator, 
                                        value: float) -> float:
        """Apply cultural adaptations to measurement values"""
        profile = self.user_profiles.get(user_id)
        if not profile:
            return value
        
        cultural_context = profile.cultural_context
        adapted_value = value
        
        # Apply cultural weightings
        for culture in cultural_context:
            if culture in self.cultural_adaptations:
                weights = self.cultural_adaptations[culture].get('indicator_weights', {})
                if indicator in weights:
                    adapted_value *= weights[indicator]
        
        # Ensure value stays within bounds
        return max(0, min(100, adapted_value))
    
    async def _apply_trauma_considerations(self, user_id: str, indicator: WellbeingIndicator, 
                                         value: float) -> List[str]:
        """Apply trauma-informed considerations to measurement"""
        profile = self.user_profiles.get(user_id)
        if not profile:
            return []
        
        trauma_considerations = profile.trauma_considerations
        adjustments = []
        
        # Check for trauma-related measurement sensitivities
        protocol = self.measurement_protocols.get(indicator, {})
        trauma_factors = protocol.get('trauma_considerations', [])
        
        for factor in trauma_factors:
            if factor in trauma_considerations:
                adjustments.append(f"trauma_adjusted_{factor}")
        
        return adjustments
    
    async def _calculate_confidence(self, user_id: str, indicator: WellbeingIndicator, 
                                  contextual_factors: Optional[Dict[str, Any]]) -> float:
        """Calculate confidence level for measurement"""
        base_confidence = 0.8
        
        # Adjust based on measurement context
        if contextual_factors:
            if contextual_factors.get('self_reported', False):
                base_confidence *= 0.9  # Self-reported can be subjective
            if contextual_factors.get('validated_by_peer', False):
                base_confidence *= 1.1  # Peer validation increases confidence
            if contextual_factors.get('professional_verified', False):
                base_confidence *= 1.2  # Professional verification increases confidence
        
        # Adjust based on user history
        profile = self.user_profiles.get(user_id)
        if profile:
            history_length = len(self.measurement_history.get(user_id, []))
            if history_length > 10:
                base_confidence *= 1.1  # More history increases confidence
        
        return max(0.1, min(1.0, base_confidence))
    
    async def _get_cultural_considerations(self, user_id: str) -> List[str]:
        """Get cultural considerations for user"""
        profile = self.user_profiles.get(user_id)
        if profile:
            return profile.cultural_context
        return []
    
    async def _determine_privacy_level(self, user_id: str, indicator: WellbeingIndicator) -> str:
        """Determine privacy level for measurement"""
        profile = self.user_profiles.get(user_id)
        if not profile:
            return "high_privacy"
        
        privacy_prefs = profile.privacy_preferences
        return privacy_prefs.get('measurement_privacy', 'medium_privacy')
    
    async def _calculate_baseline_comparison(self, user_id: str, indicator: WellbeingIndicator, 
                                          value: float) -> Optional[float]:
        """Calculate comparison to baseline value"""
        profile = self.user_profiles.get(user_id)
        if not profile:
            return None
        
        baseline = profile.baseline_indicators.get(indicator)
        if baseline is None:
            return None
        
        return value - baseline
    
    async def _analyze_trend_direction(self, user_id: str, indicator: WellbeingIndicator, 
                                     value: float) -> ProgressDirection:
        """Analyze trend direction for indicator"""
        history = self.measurement_history.get(user_id, deque())
        if len(history) < 3:
            return ProgressDirection.MINOR_FLUCTUATION
        
        # Get recent measurements for this indicator
        recent_values = [m.value for m in list(history)[-10:] if m.indicator == indicator]
        if len(recent_values) < 3:
            return ProgressDirection.MINOR_FLUCTUATION
        
        # Calculate trend
        x = list(range(len(recent_values)))
        slope = np.polyfit(x, recent_values, 1)[0]
        
        if slope > 5:
            return ProgressDirection.SIGNIFICANT_IMPROVEMENT
        elif slope > 1:
            return ProgressDirection.GRADUAL_IMPROVEMENT
        elif slope > -1:
            return ProgressDirection.STABLE_MAINTAINING
        elif slope > -5:
            return ProgressDirection.CONCERNING_DECLINE
        else:
            return ProgressDirection.CRISIS_DETERIORATION
    
    async def _assess_risk_level(self, user_id: str, indicator: WellbeingIndicator, 
                               value: float) -> WellbeingRisk:
        """Assess risk level based on measurement"""
        # Check crisis risk first
        crisis_criteria = self.risk_assessment_criteria[WellbeingRisk.CRISIS_RISK]
        for criterion in crisis_criteria['indicators']:
            if criterion['indicator'] == indicator:
                if criterion['direction'] == 'below' and value < criterion['threshold']:
                    return WellbeingRisk.CRISIS_RISK
                elif criterion['direction'] == 'above' and value > criterion['threshold']:
                    return WellbeingRisk.CRISIS_RISK
        
        # Check high risk
        high_risk_criteria = self.risk_assessment_criteria[WellbeingRisk.HIGH_RISK]
        for criterion in high_risk_criteria['indicators']:
            if criterion['indicator'] == indicator:
                if criterion['direction'] == 'below' and value < criterion['threshold']:
                    return WellbeingRisk.HIGH_RISK
                elif criterion['direction'] == 'above' and value > criterion['threshold']:
                    return WellbeingRisk.HIGH_RISK
        
        # Default risk assessment
        if value < 40:
            return WellbeingRisk.MODERATE_RISK
        else:
            return WellbeingRisk.LOW_RISK
    
    async def _generate_intervention_recommendations(self, user_id: str, 
                                                   indicator: WellbeingIndicator, 
                                                   value: float) -> List[str]:
        """Generate intervention recommendations based on measurement"""
        recommendations = []
        
        if value < 30:
            recommendations.extend([
                f"immediate_support_for_{indicator.value}",
                "professional_consultation_recommended",
                "crisis_prevention_protocols"
            ])
        elif value < 50:
            recommendations.extend([
                f"enhanced_support_for_{indicator.value}",
                "skill_building_resources",
                "peer_support_connection"
            ])
        elif value < 70:
            recommendations.extend([
                f"maintenance_support_for_{indicator.value}",
                "wellness_practices",
                "progress_monitoring"
            ])
        
        return recommendations
    
    async def _determine_follow_up_need(self, user_id: str, indicator: WellbeingIndicator, 
                                      value: float) -> bool:
        """Determine if follow-up is needed"""
        return value < 60  # Follow up needed for concerning values
    
    async def _update_wellbeing_profile(self, user_id: str, measurement: WellbeingMeasurement):
        """Update user's wellbeing profile with new measurement"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = WellbeingProfile(
                user_id=user_id,
                current_indicators={},
                baseline_indicators={},
                trend_history=defaultdict(list),
                risk_factors=[],
                protective_factors=[],
                intervention_history=[],
                wellness_goals=[],
                support_preferences={},
                cultural_context=[],
                trauma_considerations=[],
                accessibility_needs=[],
                privacy_preferences={},
                last_updated=datetime.now(),
                overall_wellbeing_score=50.0,
                progress_direction=ProgressDirection.STABLE_MAINTAINING,
                current_risk_level=WellbeingRisk.LOW_RISK
            )
        
        profile = self.user_profiles[user_id]
        
        # Update current indicators
        profile.current_indicators[measurement.indicator] = measurement.value
        
        # Update trend history
        profile.trend_history[measurement.indicator].append(measurement.value)
        if len(profile.trend_history[measurement.indicator]) > 100:
            profile.trend_history[measurement.indicator] = profile.trend_history[measurement.indicator][-100:]
        
        # Update overall wellbeing score
        profile.overall_wellbeing_score = statistics.mean(profile.current_indicators.values())
        
        # Update progress direction
        profile.progress_direction = measurement.trend_direction
        
        # Update risk level
        profile.current_risk_level = measurement.risk_level
        
        # Update timestamp
        profile.last_updated = datetime.now()
    
    async def _check_wellbeing_alerts(self, user_id: str, measurement: WellbeingMeasurement):
        """Check if wellbeing alerts should be triggered"""
        if measurement.risk_level in [WellbeingRisk.CRISIS_RISK, WellbeingRisk.HIGH_RISK]:
            alert = WellbeingAlert(
                alert_id=f"alert_{user_id}_{measurement.indicator.value}_{datetime.now().isoformat()}",
                user_id=user_id,
                alert_type=f"{measurement.indicator.value}_concern",
                severity_level=measurement.risk_level,
                triggered_indicators=[measurement.indicator],
                trend_analysis={'direction': measurement.trend_direction.value, 'value': measurement.value},
                immediate_concerns=measurement.intervention_recommendations,
                recommended_interventions=measurement.intervention_recommendations,
                escalation_required=measurement.risk_level == WellbeingRisk.CRISIS_RISK,
                cultural_considerations=measurement.cultural_considerations,
                trauma_informed_response=measurement.trauma_informed_adjustments,
                privacy_requirements=[measurement.privacy_level],
                created_at=datetime.now(),
                requires_human_review=measurement.risk_level == WellbeingRisk.CRISIS_RISK
            )
            
            self.active_alerts[user_id].append(alert)
            
            # Trigger intervention callbacks
            for callback in self.intervention_callbacks:
                try:
                    await callback(alert)
                except Exception as e:
                    logger.error(f"Error in intervention callback: {e}")
    
    async def get_wellbeing_profile(self, user_id: str) -> Optional[WellbeingProfile]:
        """Get user's current wellbeing profile"""
        return self.user_profiles.get(user_id)
    
    async def get_wellbeing_trends(self, user_id: str, 
                                 indicator: Optional[WellbeingIndicator] = None,
                                 time_range: Optional[timedelta] = None) -> Dict[str, Any]:
        """Get wellbeing trends for user"""
        history = self.measurement_history.get(user_id, deque())
        if not history:
            return {}
        
        # Filter by time range
        if time_range:
            cutoff = datetime.now() - time_range
            history = [m for m in history if m.measurement_timestamp >= cutoff]
        
        # Filter by indicator
        if indicator:
            history = [m for m in history if m.indicator == indicator]
        
        if not history:
            return {}
        
        # Calculate trends
        values = [m.value for m in history]
        timestamps = [m.measurement_timestamp for m in history]
        
        return {
            'values': values,
            'timestamps': timestamps,
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'std_dev': statistics.stdev(values) if len(values) > 1 else 0,
            'min': min(values),
            'max': max(values),
            'trend_direction': history[-1].trend_direction.value if history else None,
            'current_risk_level': history[-1].risk_level.value if history else None
        }
    
    async def add_intervention_callback(self, callback: Callable):
        """Add callback for intervention alerts"""
        self.intervention_callbacks.append(callback)
    
    async def generate_wellbeing_report(self, user_id: str) -> Dict[str, Any]:
        """Generate comprehensive wellbeing report"""
        profile = await self.get_wellbeing_profile(user_id)
        if not profile:
            return {}
        
        # Get trends for all indicators
        trends = {}
        for indicator in WellbeingIndicator:
            indicator_trends = await self.get_wellbeing_trends(user_id, indicator, timedelta(days=30))
            if indicator_trends:
                trends[indicator.value] = indicator_trends
        
        # Get active alerts
        alerts = self.active_alerts.get(user_id, [])
        
        return {
            'user_id': user_id,
            'report_generated': datetime.now().isoformat(),
            'overall_wellbeing_score': profile.overall_wellbeing_score,
            'progress_direction': profile.progress_direction.value,
            'current_risk_level': profile.current_risk_level.value,
            'current_indicators': {k.value: v for k, v in profile.current_indicators.items()},
            'baseline_comparison': {
                k.value: profile.current_indicators.get(k, 0) - profile.baseline_indicators.get(k, 0)
                for k in profile.baseline_indicators.keys()
            },
            'trends': trends,
            'active_alerts': len(alerts),
            'crisis_alerts': len([a for a in alerts if a.severity_level == WellbeingRisk.CRISIS_RISK]),
            'intervention_recommendations': profile.intervention_history[-5:] if profile.intervention_history else [],
            'cultural_considerations': profile.cultural_context,
            'trauma_considerations': profile.trauma_considerations,
            'privacy_level': profile.privacy_preferences.get('report_privacy', 'medium_privacy')
        }