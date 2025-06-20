"""
Community Health Monitor

Monitors and analyzes the overall health and wellbeing of communities
with focus on collective healing, resilience, and supportive environments.
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


class HealthDimension(Enum):
    """Dimensions of community health to monitor"""
    PSYCHOLOGICAL_SAFETY = "psychological_safety"           # Safe to be vulnerable
    COLLECTIVE_RESILIENCE = "collective_resilience"         # Community's ability to cope
    SOCIAL_COHESION = "social_cohesion"                     # Strength of social bonds
    SUPPORT_NETWORK_STRENGTH = "support_network_strength"   # Quality of support systems
    CONFLICT_RESOLUTION = "conflict_resolution"             # Healthy conflict handling
    INCLUSIVE_PARTICIPATION = "inclusive_participation"     # Equal participation opportunities
    SHARED_PURPOSE = "shared_purpose"                       # Common goals and values
    KNOWLEDGE_SHARING = "knowledge_sharing"                 # Wisdom and resource sharing
    HEALING_CULTURE = "healing_culture"                     # Culture that promotes healing
    CRISIS_RESPONSE = "crisis_response"                     # Community crisis response
    CELEBRATION_CULTURE = "celebration_culture"             # Celebrating successes together
    BOUNDARY_RESPECT = "boundary_respect"                   # Respecting individual boundaries


class CommunityVitality(Enum):
    """Levels of community vitality and health"""
    THRIVING = "thriving"                                   # Community is flourishing
    HEALTHY = "healthy"                                     # Community is doing well
    STABLE = "stable"                                       # Community is maintaining
    CONCERNING = "concerning"                               # Some worrying patterns
    STRUGGLING = "struggling"                               # Community facing challenges
    CRISIS = "crisis"                                       # Community in crisis state


class CommunitySize(Enum):
    """Categories of community size"""
    INTIMATE = "intimate"                                   # 2-15 members
    SMALL = "small"                                         # 16-50 members
    MEDIUM = "medium"                                       # 51-150 members
    LARGE = "large"                                         # 151-500 members
    VERY_LARGE = "very_large"                              # 500+ members


class InterventionScope(Enum):
    """Scope of community interventions"""
    INDIVIDUAL_SUPPORT = "individual_support"              # Supporting individual members
    INTERPERSONAL_MEDIATION = "interpersonal_mediation"    # Mediating between members
    GROUP_FACILITATION = "group_facilitation"              # Facilitating group processes
    STRUCTURAL_CHANGES = "structural_changes"              # Changing community structures
    CULTURAL_SHIFTS = "cultural_shifts"                    # Shifting community culture
    CRISIS_INTERVENTION = "crisis_intervention"            # Emergency community intervention


@dataclass
class CommunityHealthMetric:
    """Individual community health measurement"""
    metric_id: str
    community_id: str
    health_dimension: HealthDimension
    value: float  # 0-100 scale
    confidence_level: float
    measurement_method: str
    data_sources: List[str]
    contributing_factors: Dict[str, Any]
    cultural_context: List[str]
    community_characteristics: Dict[str, Any]
    timestamp: datetime
    baseline_comparison: Optional[float]
    trend_direction: str
    risk_indicators: List[str]
    protective_factors: List[str]
    intervention_recommendations: List[str]


@dataclass
class CommunityProfile:
    """Comprehensive profile of a community's health"""
    community_id: str
    community_name: str
    community_size: CommunitySize
    member_demographics: Dict[str, Any]
    health_dimensions: Dict[HealthDimension, float]
    baseline_metrics: Dict[HealthDimension, float]
    vitality_level: CommunityVitality
    cultural_characteristics: List[str]
    structural_features: Dict[str, Any]
    support_systems: List[Dict[str, Any]]
    leadership_model: str
    conflict_patterns: List[Dict[str, Any]]
    healing_practices: List[str]
    crisis_history: List[Dict[str, Any]]
    resilience_factors: List[str]
    vulnerability_factors: List[str]
    intervention_history: List[Dict[str, Any]]
    growth_areas: List[str]
    strengths: List[str]
    last_assessed: datetime
    next_assessment_due: datetime


@dataclass
class CommunityAlert:
    """Alert for concerning community health patterns"""
    alert_id: str
    community_id: str
    alert_type: str
    severity_level: CommunityVitality
    affected_dimensions: List[HealthDimension]
    warning_indicators: List[str]
    potential_impacts: List[str]
    immediate_risks: List[str]
    recommended_interventions: List[InterventionScope]
    escalation_timeline: str
    cultural_considerations: List[str]
    stakeholder_notifications: List[str]
    monitoring_requirements: Dict[str, Any]
    created_at: datetime
    requires_immediate_action: bool
    expert_consultation_needed: bool


class CommunityHealthMonitor:
    """
    Comprehensive community health monitoring system that tracks collective
    wellbeing, resilience, and healing capacity of communities.
    
    Core Principles:
    - Community wellbeing reflects individual wellbeing and vice versa
    - Healthy communities support individual healing and growth
    - Cultural context shapes community health patterns
    - Prevention is more effective than crisis intervention
    - Community resilience can be measured and strengthened
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.community_profiles: Dict[str, CommunityProfile] = {}
        self.health_metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.active_alerts: Dict[str, List[CommunityAlert]] = defaultdict(list)
        self.intervention_callbacks: List[Callable] = []
        self.cultural_frameworks: Dict[str, Any] = {}
        self.assessment_protocols: Dict[str, Any] = {}
        self.baseline_calculators: Dict[HealthDimension, Callable] = {}
        self.trend_analyzers: Dict[HealthDimension, Callable] = {}
        
        # Initialize monitoring systems
        self._setup_assessment_protocols()
        self._setup_cultural_frameworks()
        self._setup_health_indicators()
        self._setup_intervention_protocols()
    
    def _setup_assessment_protocols(self):
        """Setup assessment protocols for different community health dimensions"""
        self.assessment_protocols = {
            HealthDimension.PSYCHOLOGICAL_SAFETY: {
                'indicators': [
                    'vulnerability_sharing_frequency',
                    'mistake_response_patterns',
                    'feedback_quality',
                    'newcomer_integration_success'
                ],
                'measurement_methods': ['behavioral_observation', 'member_surveys', 'interaction_analysis'],
                'frequency': 'weekly',
                'sensitivity': 'high'
            },
            HealthDimension.COLLECTIVE_RESILIENCE: {
                'indicators': [
                    'crisis_recovery_time',
                    'adaptation_to_challenges',
                    'mutual_support_during_stress',
                    'learning_from_setbacks'
                ],
                'measurement_methods': ['longitudinal_tracking', 'crisis_response_analysis'],
                'frequency': 'monthly',
                'sensitivity': 'medium'
            },
            HealthDimension.SOCIAL_COHESION: {
                'indicators': [
                    'member_interaction_frequency',
                    'cross_subgroup_connections',
                    'shared_activity_participation',
                    'collective_decision_making'
                ],
                'measurement_methods': ['network_analysis', 'participation_tracking'],
                'frequency': 'weekly',
                'sensitivity': 'medium'
            },
            HealthDimension.HEALING_CULTURE: {
                'indicators': [
                    'healing_practice_adoption',
                    'trauma_informed_responses',
                    'growth_mindset_prevalence',
                    'compassion_demonstrations'
                ],
                'measurement_methods': ['cultural_analysis', 'practice_observation'],
                'frequency': 'monthly',
                'sensitivity': 'high'
            }
        }
    
    def _setup_cultural_frameworks(self):
        """Setup cultural frameworks for community assessment"""
        self.cultural_frameworks = {
            'individualistic_communities': {
                'health_emphasis': ['personal_growth', 'individual_achievement', 'self_reliance'],
                'assessment_adaptations': {
                    HealthDimension.PSYCHOLOGICAL_SAFETY: 'focus_on_personal_expression',
                    HealthDimension.SHARED_PURPOSE: 'emphasize_individual_contribution'
                }
            },
            'collectivistic_communities': {
                'health_emphasis': ['group_harmony', 'collective_success', 'interdependence'],
                'assessment_adaptations': {
                    HealthDimension.SOCIAL_COHESION: 'focus_on_group_unity',
                    HealthDimension.CONFLICT_RESOLUTION: 'emphasize_harmony_preservation'
                }
            },
            'hierarchical_communities': {
                'health_emphasis': ['clear_leadership', 'structured_roles', 'respect_for_authority'],
                'assessment_adaptations': {
                    HealthDimension.INCLUSIVE_PARTICIPATION: 'consider_role_based_participation',
                    HealthDimension.BOUNDARY_RESPECT: 'include_hierarchy_boundaries'
                }
            },
            'egalitarian_communities': {
                'health_emphasis': ['equal_participation', 'shared_leadership', 'consensus_building'],
                'assessment_adaptations': {
                    HealthDimension.INCLUSIVE_PARTICIPATION: 'focus_on_equal_voice',
                    HealthDimension.SHARED_PURPOSE: 'emphasize_collective_visioning'
                }
            }
        }
    
    def _setup_health_indicators(self):
        """Setup specific health indicators for each dimension"""
        self.health_indicators = {
            HealthDimension.PSYCHOLOGICAL_SAFETY: [
                'members_share_vulnerabilities_openly',
                'mistakes_met_with_support_not_blame',
                'diverse_perspectives_welcomed',
                'members_ask_for_help_without_fear',
                'feedback_given_constructively',
                'newcomers_feel_welcomed_quickly'
            ],
            HealthDimension.COLLECTIVE_RESILIENCE: [
                'community_adapts_to_challenges_effectively',
                'members_support_each_other_during_crises',
                'community_learns_from_setbacks',
                'stress_shared_rather_than_concentrated',
                'recovery_time_from_disruptions_decreasing',
                'preventive_measures_implemented_proactively'
            ],
            HealthDimension.SOCIAL_COHESION: [
                'members_interact_across_subgroups',
                'high_participation_in_shared_activities',
                'decisions_made_collectively',
                'members_feel_sense_of_belonging',
                'informal_support_networks_strong',
                'shared_stories_and_traditions_maintained'
            ],
            HealthDimension.HEALING_CULTURE: [
                'trauma_responses_handled_with_care',
                'growth_and_learning_celebrated',
                'compassion_practiced_regularly',
                'healing_practices_integrated_naturally',
                'space_for_emotional_expression_provided',
                'restorative_rather_than_punitive_approaches'
            ]
        }
    
    def _setup_intervention_protocols(self):
        """Setup intervention protocols for different community health issues"""
        self.intervention_protocols = {
            CommunityVitality.CRISIS: {
                'immediate_actions': [
                    'activate_crisis_response_team',
                    'ensure_member_safety',
                    'communicate_transparently',
                    'provide_immediate_support_resources'
                ],
                'timeline': 'immediate',
                'escalation_criteria': ['safety_threats', 'mass_exodus', 'leadership_breakdown']
            },
            CommunityVitality.STRUGGLING: {
                'immediate_actions': [
                    'conduct_community_health_assessment',
                    'facilitate_community_dialogue',
                    'identify_and_address_key_stressors',
                    'strengthen_support_systems'
                ],
                'timeline': '1-2_weeks',
                'escalation_criteria': ['deteriorating_metrics', 'member_distress', 'leadership_concerns']
            },
            CommunityVitality.CONCERNING: {
                'immediate_actions': [
                    'increase_monitoring_frequency',
                    'engage_community_leaders',
                    'implement_preventive_measures',
                    'offer_additional_resources'
                ],
                'timeline': '2-4_weeks',
                'escalation_criteria': ['trend_continuation', 'new_risk_factors']
            }
        }
    
    async def assess_community_health(self, community_id: str, 
                                    dimension: HealthDimension,
                                    assessment_data: Dict[str, Any]) -> CommunityHealthMetric:
        """Assess a specific dimension of community health"""
        try:
            # Get community profile
            profile = self.community_profiles.get(community_id)
            if not profile:
                logger.warning(f"No profile found for community {community_id}")
                return None
            
            # Calculate health metric value
            value = await self._calculate_health_dimension_value(community_id, dimension, assessment_data)
            
            # Apply cultural adaptations
            adapted_value = await self._apply_cultural_adaptations(profile, dimension, value)
            
            # Calculate confidence level
            confidence = await self._calculate_assessment_confidence(dimension, assessment_data)
            
            # Analyze trend
            trend_direction = await self._analyze_health_trend(community_id, dimension, adapted_value)
            
            # Identify risk and protective factors
            risk_indicators = await self._identify_risk_indicators(community_id, dimension, assessment_data)
            protective_factors = await self._identify_protective_factors(community_id, dimension, assessment_data)
            
            # Generate intervention recommendations
            interventions = await self._generate_intervention_recommendations(
                community_id, dimension, adapted_value, risk_indicators
            )
            
            # Create health metric
            metric = CommunityHealthMetric(
                metric_id=f"ch_{community_id}_{dimension.value}_{datetime.now().isoformat()}",
                community_id=community_id,
                health_dimension=dimension,
                value=adapted_value,
                confidence_level=confidence,
                measurement_method=assessment_data.get('method', 'comprehensive_assessment'),
                data_sources=assessment_data.get('sources', ['community_observation']),
                contributing_factors=assessment_data.get('factors', {}),
                cultural_context=profile.cultural_characteristics,
                community_characteristics={
                    'size': profile.community_size.value,
                    'leadership_model': profile.leadership_model,
                    'member_demographics': profile.member_demographics
                },
                timestamp=datetime.now(),
                baseline_comparison=await self._calculate_baseline_comparison(community_id, dimension, adapted_value),
                trend_direction=trend_direction,
                risk_indicators=risk_indicators,
                protective_factors=protective_factors,
                intervention_recommendations=interventions
            )
            
            # Store metric
            self.health_metrics_history[community_id].append(metric)
            
            # Update community profile
            await self._update_community_profile(community_id, metric)
            
            # Check for alerts
            await self._check_community_alerts(community_id, metric)
            
            logger.info(f"Community health assessed: {community_id}, {dimension}, value: {adapted_value}")
            return metric
            
        except Exception as e:
            logger.error(f"Error assessing community health: {e}")
            return None
    
    async def _calculate_health_dimension_value(self, community_id: str, 
                                              dimension: HealthDimension,
                                              assessment_data: Dict[str, Any]) -> float:
        """Calculate the value for a specific health dimension"""
        indicators = self.health_indicators.get(dimension, [])
        if not indicators:
            return 50.0  # Default middle value
        
        # Calculate based on indicator assessments
        indicator_scores = []
        for indicator in indicators:
            score = assessment_data.get('indicators', {}).get(indicator, 50.0)
            indicator_scores.append(score)
        
        if not indicator_scores:
            return 50.0
        
        # Weight indicators based on community characteristics
        profile = self.community_profiles.get(community_id)
        if profile:
            weighted_scores = await self._apply_indicator_weights(profile, dimension, indicator_scores)
            return statistics.mean(weighted_scores)
        
        return statistics.mean(indicator_scores)
    
    async def _apply_cultural_adaptations(self, profile: CommunityProfile, 
                                        dimension: HealthDimension, value: float) -> float:
        """Apply cultural adaptations to health dimension values"""
        cultural_characteristics = profile.cultural_characteristics
        adapted_value = value
        
        # Apply cultural framework adaptations
        for culture in cultural_characteristics:
            if culture in self.cultural_frameworks:
                framework = self.cultural_frameworks[culture]
                adaptations = framework.get('assessment_adaptations', {})
                
                if dimension in adaptations:
                    # Apply cultural adjustment
                    adjustment_factor = 1.0
                    if 'emphasize' in adaptations[dimension]:
                        adjustment_factor = 1.1
                    elif 'de_emphasize' in adaptations[dimension]:
                        adjustment_factor = 0.9
                    
                    adapted_value *= adjustment_factor
        
        return max(0, min(100, adapted_value))
    
    async def _apply_indicator_weights(self, profile: CommunityProfile, 
                                     dimension: HealthDimension, 
                                     scores: List[float]) -> List[float]:
        """Apply weights to indicators based on community characteristics"""
        # Weight based on community size
        size_weights = {
            CommunitySize.INTIMATE: {'interpersonal_focus': 1.2, 'structural_focus': 0.8},
            CommunitySize.SMALL: {'interpersonal_focus': 1.1, 'structural_focus': 0.9},
            CommunitySize.MEDIUM: {'interpersonal_focus': 1.0, 'structural_focus': 1.0},
            CommunitySize.LARGE: {'interpersonal_focus': 0.9, 'structural_focus': 1.1},
            CommunitySize.VERY_LARGE: {'interpersonal_focus': 0.8, 'structural_focus': 1.2}
        }
        
        weights = size_weights.get(profile.community_size, {'interpersonal_focus': 1.0, 'structural_focus': 1.0})
        
        # Apply dimension-specific weighting logic
        weighted_scores = []
        for i, score in enumerate(scores):
            weight = 1.0
            
            # Apply weights based on dimension characteristics
            if dimension in [HealthDimension.PSYCHOLOGICAL_SAFETY, HealthDimension.SOCIAL_COHESION]:
                weight = weights['interpersonal_focus']
            elif dimension in [HealthDimension.INCLUSIVE_PARTICIPATION, HealthDimension.CONFLICT_RESOLUTION]:
                weight = weights['structural_focus']
            
            weighted_scores.append(score * weight)
        
        return weighted_scores
    
    async def _calculate_assessment_confidence(self, dimension: HealthDimension, 
                                             assessment_data: Dict[str, Any]) -> float:
        """Calculate confidence level for the assessment"""
        base_confidence = 0.7
        
        # Adjust based on data sources
        sources = assessment_data.get('sources', [])
        if len(sources) > 2:
            base_confidence += 0.1
        if 'member_surveys' in sources:
            base_confidence += 0.1
        if 'behavioral_observation' in sources:
            base_confidence += 0.1
        if 'expert_assessment' in sources:
            base_confidence += 0.1
        
        # Adjust based on data quality
        sample_size = assessment_data.get('sample_size', 0)
        if sample_size > 50:
            base_confidence += 0.1
        elif sample_size > 20:
            base_confidence += 0.05
        
        return max(0.1, min(1.0, base_confidence))
    
    async def _analyze_health_trend(self, community_id: str, dimension: HealthDimension, 
                                  value: float) -> str:
        """Analyze trend direction for health dimension"""
        history = self.health_metrics_history.get(community_id, deque())
        if len(history) < 3:
            return "insufficient_data"
        
        # Get recent measurements for this dimension
        recent_values = [m.value for m in list(history)[-10:] if m.health_dimension == dimension]
        if len(recent_values) < 3:
            return "insufficient_data"
        
        # Calculate trend
        x = list(range(len(recent_values)))
        slope = np.polyfit(x, recent_values, 1)[0]
        
        if slope > 5:
            return "significant_improvement"
        elif slope > 1:
            return "gradual_improvement"
        elif slope > -1:
            return "stable"
        elif slope > -5:
            return "gradual_decline"
        else:
            return "significant_decline"
    
    async def _identify_risk_indicators(self, community_id: str, dimension: HealthDimension,
                                      assessment_data: Dict[str, Any]) -> List[str]:
        """Identify risk indicators from assessment data"""
        risk_indicators = []
        
        # Check for specific risk patterns
        indicators_data = assessment_data.get('indicators', {})
        
        for indicator, score in indicators_data.items():
            if score < 30:
                risk_indicators.append(f"low_{indicator}")
            elif score < 50:
                risk_indicators.append(f"concerning_{indicator}")
        
        # Check for dimension-specific risks
        if dimension == HealthDimension.PSYCHOLOGICAL_SAFETY:
            if indicators_data.get('mistakes_met_with_support_not_blame', 100) < 40:
                risk_indicators.append("blame_culture_present")
            if indicators_data.get('members_share_vulnerabilities_openly', 100) < 30:
                risk_indicators.append("vulnerability_suppression")
        
        elif dimension == HealthDimension.COLLECTIVE_RESILIENCE:
            if indicators_data.get('recovery_time_from_disruptions_decreasing', 100) < 30:
                risk_indicators.append("poor_crisis_recovery")
            if indicators_data.get('stress_shared_rather_than_concentrated', 100) < 40:
                risk_indicators.append("stress_concentration")
        
        return risk_indicators
    
    async def _identify_protective_factors(self, community_id: str, dimension: HealthDimension,
                                         assessment_data: Dict[str, Any]) -> List[str]:
        """Identify protective factors from assessment data"""
        protective_factors = []
        
        # Check for strong indicators
        indicators_data = assessment_data.get('indicators', {})
        
        for indicator, score in indicators_data.items():
            if score > 80:
                protective_factors.append(f"strong_{indicator}")
            elif score > 70:
                protective_factors.append(f"good_{indicator}")
        
        # Check for dimension-specific protective factors
        if dimension == HealthDimension.SOCIAL_COHESION:
            if indicators_data.get('members_interact_across_subgroups', 0) > 75:
                protective_factors.append("strong_cross_group_connections")
            if indicators_data.get('high_participation_in_shared_activities', 0) > 80:
                protective_factors.append("high_collective_engagement")
        
        return protective_factors
    
    async def _generate_intervention_recommendations(self, community_id: str, 
                                                   dimension: HealthDimension,
                                                   value: float, 
                                                   risk_indicators: List[str]) -> List[str]:
        """Generate intervention recommendations based on assessment"""
        recommendations = []
        
        # Base recommendations on value and dimension
        if value < 30:
            recommendations.extend([
                f"immediate_intervention_for_{dimension.value}",
                "expert_consultation_recommended",
                "crisis_prevention_protocols"
            ])
        elif value < 50:
            recommendations.extend([
                f"structured_improvement_for_{dimension.value}",
                "community_dialogue_facilitation",
                "targeted_skill_building"
            ])
        elif value < 70:
            recommendations.extend([
                f"enhancement_activities_for_{dimension.value}",
                "preventive_measures",
                "strength_building"
            ])
        
        # Add specific recommendations based on risk indicators
        for risk in risk_indicators:
            if "blame_culture" in risk:
                recommendations.append("implement_learning_culture_practices")
            elif "stress_concentration" in risk:
                recommendations.append("distribute_leadership_and_responsibilities")
            elif "poor_crisis_recovery" in risk:
                recommendations.append("develop_crisis_response_protocols")
        
        return list(set(recommendations))  # Remove duplicates
    
    async def _calculate_baseline_comparison(self, community_id: str, 
                                           dimension: HealthDimension, 
                                           value: float) -> Optional[float]:
        """Calculate comparison to baseline value"""
        profile = self.community_profiles.get(community_id)
        if not profile:
            return None
        
        baseline = profile.baseline_metrics.get(dimension)
        if baseline is None:
            return None
        
        return value - baseline
    
    async def _update_community_profile(self, community_id: str, metric: CommunityHealthMetric):
        """Update community profile with new health metric"""
        if community_id not in self.community_profiles:
            logger.warning(f"Cannot update profile for unknown community {community_id}")
            return
        
        profile = self.community_profiles[community_id]
        
        # Update health dimensions
        profile.health_dimensions[metric.health_dimension] = metric.value
        
        # Update overall vitality level
        profile.vitality_level = await self._calculate_overall_vitality(community_id)
        
        # Update assessment timestamp
        profile.last_assessed = datetime.now()
        profile.next_assessment_due = datetime.now() + timedelta(days=30)
        
        # Update intervention history if recommendations exist
        if metric.intervention_recommendations:
            profile.intervention_history.append({
                'timestamp': datetime.now(),
                'dimension': metric.health_dimension.value,
                'recommendations': metric.intervention_recommendations,
                'trigger_value': metric.value
            })
    
    async def _calculate_overall_vitality(self, community_id: str) -> CommunityVitality:
        """Calculate overall community vitality level"""
        profile = self.community_profiles.get(community_id)
        if not profile or not profile.health_dimensions:
            return CommunityVitality.STABLE
        
        # Calculate weighted average
        dimension_weights = {
            HealthDimension.PSYCHOLOGICAL_SAFETY: 1.2,
            HealthDimension.COLLECTIVE_RESILIENCE: 1.1,
            HealthDimension.SOCIAL_COHESION: 1.0,
            HealthDimension.HEALING_CULTURE: 1.1,
            HealthDimension.CRISIS_RESPONSE: 1.0
        }
        
        weighted_sum = 0
        total_weight = 0
        
        for dimension, value in profile.health_dimensions.items():
            weight = dimension_weights.get(dimension, 1.0)
            weighted_sum += value * weight
            total_weight += weight
        
        if total_weight == 0:
            return CommunityVitality.STABLE
        
        overall_score = weighted_sum / total_weight
        
        # Map score to vitality level
        if overall_score >= 85:
            return CommunityVitality.THRIVING
        elif overall_score >= 70:
            return CommunityVitality.HEALTHY
        elif overall_score >= 55:
            return CommunityVitality.STABLE
        elif overall_score >= 40:
            return CommunityVitality.CONCERNING
        elif overall_score >= 25:
            return CommunityVitality.STRUGGLING
        else:
            return CommunityVitality.CRISIS
    
    async def _check_community_alerts(self, community_id: str, metric: CommunityHealthMetric):
        """Check if community alerts should be triggered"""
        profile = self.community_profiles.get(community_id)
        if not profile:
            return
        
        # Check for crisis or concerning vitality levels
        if profile.vitality_level in [CommunityVitality.CRISIS, CommunityVitality.STRUGGLING]:
            alert = CommunityAlert(
                alert_id=f"alert_{community_id}_{metric.health_dimension.value}_{datetime.now().isoformat()}",
                community_id=community_id,
                alert_type=f"{metric.health_dimension.value}_concern",
                severity_level=profile.vitality_level,
                affected_dimensions=[metric.health_dimension],
                warning_indicators=metric.risk_indicators,
                potential_impacts=await self._assess_potential_impacts(profile.vitality_level),
                immediate_risks=metric.risk_indicators,
                recommended_interventions=[InterventionScope.CRISIS_INTERVENTION] if profile.vitality_level == CommunityVitality.CRISIS else [InterventionScope.GROUP_FACILITATION],
                escalation_timeline="immediate" if profile.vitality_level == CommunityVitality.CRISIS else "within_week",
                cultural_considerations=profile.cultural_characteristics,
                stakeholder_notifications=["community_leaders", "support_team"],
                monitoring_requirements={"frequency": "daily", "focus_areas": metric.risk_indicators},
                created_at=datetime.now(),
                requires_immediate_action=profile.vitality_level == CommunityVitality.CRISIS,
                expert_consultation_needed=profile.vitality_level in [CommunityVitality.CRISIS, CommunityVitality.STRUGGLING]
            )
            
            self.active_alerts[community_id].append(alert)
            
            # Trigger intervention callbacks
            for callback in self.intervention_callbacks:
                try:
                    await callback(alert)
                except Exception as e:
                    logger.error(f"Error in community intervention callback: {e}")
    
    async def _assess_potential_impacts(self, vitality_level: CommunityVitality) -> List[str]:
        """Assess potential impacts of current vitality level"""
        impact_map = {
            CommunityVitality.CRISIS: [
                "member_exodus_risk",
                "community_dissolution_risk",
                "individual_trauma_escalation",
                "support_system_collapse"
            ],
            CommunityVitality.STRUGGLING: [
                "reduced_member_engagement",
                "increased_conflict_likelihood",
                "weakened_support_networks",
                "cultural_erosion"
            ],
            CommunityVitality.CONCERNING: [
                "gradual_decline_risk",
                "reduced_resilience",
                "member_satisfaction_decrease"
            ]
        }
        
        return impact_map.get(vitality_level, [])
    
    async def register_community(self, community_profile: CommunityProfile):
        """Register a new community for monitoring"""
        self.community_profiles[community_profile.community_id] = community_profile
        logger.info(f"Community registered for monitoring: {community_profile.community_id}")
    
    async def get_community_health_report(self, community_id: str) -> Dict[str, Any]:
        """Generate comprehensive community health report"""
        profile = self.community_profiles.get(community_id)
        if not profile:
            return {}
        
        # Get recent metrics
        recent_metrics = list(self.health_metrics_history.get(community_id, deque()))[-30:]
        
        # Calculate trends
        trends = {}
        for dimension in HealthDimension:
            dimension_metrics = [m for m in recent_metrics if m.health_dimension == dimension]
            if dimension_metrics:
                values = [m.value for m in dimension_metrics]
                trends[dimension.value] = {
                    'current': values[-1] if values else 0,
                    'average': statistics.mean(values),
                    'trend': dimension_metrics[-1].trend_direction if dimension_metrics else 'stable'
                }
        
        # Get active alerts
        alerts = self.active_alerts.get(community_id, [])
        
        return {
            'community_id': community_id,
            'community_name': profile.community_name,
            'report_generated': datetime.now().isoformat(),
            'vitality_level': profile.vitality_level.value,
            'health_dimensions': {k.value: v for k, v in profile.health_dimensions.items()},
            'trends': trends,
            'strengths': profile.strengths,
            'growth_areas': profile.growth_areas,
            'active_alerts': len(alerts),
            'crisis_alerts': len([a for a in alerts if a.severity_level == CommunityVitality.CRISIS]),
            'intervention_recommendations': profile.intervention_history[-3:] if profile.intervention_history else [],
            'cultural_characteristics': profile.cultural_characteristics,
            'community_size': profile.community_size.value,
            'leadership_model': profile.leadership_model,
            'last_assessed': profile.last_assessed.isoformat(),
            'next_assessment_due': profile.next_assessment_due.isoformat()
        }
    
    async def add_intervention_callback(self, callback: Callable):
        """Add callback for community intervention alerts"""
        self.intervention_callbacks.append(callback)