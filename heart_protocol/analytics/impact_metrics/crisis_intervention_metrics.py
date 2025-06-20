"""
Crisis Intervention Metrics

Comprehensive measurement and tracking of crisis intervention effectiveness,
outcomes, and long-term impact on individuals and communities.
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


class InterventionOutcome(Enum):
    """Outcomes of crisis interventions"""
    CRISIS_RESOLVED = "crisis_resolved"                     # Crisis successfully resolved
    CRISIS_STABILIZED = "crisis_stabilized"                # Crisis stabilized but ongoing support needed
    ESCALATION_PREVENTED = "escalation_prevented"          # Prevented escalation to more severe crisis
    PROFESSIONAL_REFERRAL = "professional_referral"        # Successfully referred to professional help
    ONGOING_MONITORING = "ongoing_monitoring"               # Requires continued monitoring
    INTERVENTION_DECLINED = "intervention_declined"        # Person declined intervention
    LOST_CONTACT = "lost_contact"                          # Lost contact with person in crisis
    UNSUCCESSFUL = "unsuccessful"                           # Intervention was not successful


class LifeImpactLevel(Enum):
    """Levels of life impact from crisis intervention"""
    LIFE_SAVING = "life_saving"                           # Intervention likely saved a life
    MAJOR_HARM_PREVENTION = "major_harm_prevention"        # Prevented significant harm
    MODERATE_IMPROVEMENT = "moderate_improvement"          # Notable positive change
    STABILIZATION = "stabilization"                       # Helped stabilize situation
    MINOR_SUPPORT = "minor_support"                       # Provided minor but meaningful support
    NO_MEASURABLE_IMPACT = "no_measurable_impact"         # No clear impact detected
    POTENTIAL_HARM = "potential_harm"                     # May have caused unintended harm


class CrisisType(Enum):
    """Types of crises handled"""
    SUICIDAL_IDEATION = "suicidal_ideation"
    SUICIDE_ATTEMPT = "suicide_attempt"
    SELF_HARM = "self_harm"
    SEVERE_MENTAL_HEALTH = "severe_mental_health"
    DOMESTIC_VIOLENCE = "domestic_violence"
    SUBSTANCE_CRISIS = "substance_crisis"
    TRAUMA_RESPONSE = "trauma_response"
    PSYCHOTIC_EPISODE = "psychotic_episode"
    PANIC_DISORDER = "panic_disorder"
    EATING_DISORDER = "eating_disorder"
    CHILD_SAFETY = "child_safety"
    ELDER_ABUSE = "elder_abuse"


class InterventionType(Enum):
    """Types of interventions provided"""
    IMMEDIATE_SAFETY = "immediate_safety"                  # Immediate safety intervention
    ACTIVE_LISTENING = "active_listening"                  # Supportive listening
    RESOURCE_PROVISION = "resource_provision"              # Providing resources/information
    PROFESSIONAL_REFERRAL = "professional_referral"       # Referral to professionals
    PEER_SUPPORT = "peer_support"                         # Peer support connection
    CRISIS_HOTLINE = "crisis_hotline"                     # Crisis hotline referral
    EMERGENCY_SERVICES = "emergency_services"             # Emergency services contact
    FOLLOW_UP_SUPPORT = "follow_up_support"               # Ongoing follow-up
    FAMILY_NOTIFICATION = "family_notification"           # Family/friend notification
    SAFETY_PLANNING = "safety_planning"                   # Safety plan development


class ResponseSpeed(Enum):
    """Speed of crisis response"""
    IMMEDIATE = "immediate"                                # Within minutes
    RAPID = "rapid"                                       # Within 1 hour
    SAME_DAY = "same_day"                                 # Within 24 hours
    NEXT_DAY = "next_day"                                 # Within 48 hours
    DELAYED = "delayed"                                   # More than 48 hours


@dataclass
class CrisisInterventionRecord:
    """Record of a crisis intervention"""
    intervention_id: str
    crisis_event_id: str
    person_id: str  # Anonymous identifier
    crisis_type: CrisisType
    intervention_types: List[InterventionType]
    response_speed: ResponseSpeed
    intervention_start: datetime
    intervention_end: Optional[datetime]
    outcome: InterventionOutcome
    life_impact_level: LifeImpactLevel
    cultural_considerations: List[str]
    trauma_informed_approaches: List[str]
    barriers_encountered: List[str]
    success_factors: List[str]
    professional_handoff: bool
    follow_up_planned: bool
    follow_up_completed: bool
    safety_plan_created: bool
    resource_connections_made: List[str]
    intervention_notes: str  # Anonymized notes
    responder_feedback: Dict[str, Any]
    person_feedback: Optional[Dict[str, Any]]
    ethical_considerations: List[str]
    privacy_compliance: bool
    consent_status: str


@dataclass
class InterventionFollowUp:
    """Follow-up tracking for crisis interventions"""
    follow_up_id: str
    intervention_id: str
    person_id: str
    follow_up_date: datetime
    contact_method: str
    contact_successful: bool
    person_status: str
    improvement_indicators: List[str]
    ongoing_concerns: List[str]
    additional_resources_needed: List[str]
    satisfaction_rating: Optional[float]
    would_recommend: Optional[bool]
    safety_plan_adherence: Optional[str]
    professional_support_engagement: Optional[str]
    crisis_recurrence: bool
    long_term_stability: Optional[str]
    follow_up_notes: str


@dataclass
class InterventionEffectivenessAnalysis:
    """Analysis of intervention effectiveness"""
    analysis_id: str
    intervention_id: str
    time_period: str
    immediate_effectiveness: float  # 0-100 scale
    short_term_effectiveness: float  # 1-4 weeks
    medium_term_effectiveness: float  # 1-6 months
    long_term_effectiveness: float  # 6+ months
    life_impact_score: float
    quality_of_life_change: float
    crisis_recurrence_rate: float
    professional_engagement_success: float
    support_network_strengthening: float
    cultural_sensitivity_score: float
    trauma_informed_score: float
    ethical_compliance_score: float
    lessons_learned: List[str]
    improvement_recommendations: List[str]


class CrisisInterventionMetrics:
    """
    Comprehensive system for measuring the effectiveness and impact of
    crisis interventions, with focus on healing outcomes and ethical practices.
    
    Core Principles:
    - Life safety and wellbeing are the ultimate measures of success
    - Long-term healing matters more than immediate crisis resolution
    - Cultural sensitivity and trauma-informed care are essential
    - Privacy and consent must be maintained even in crisis situations
    - Continuous learning and improvement are vital
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.intervention_records: Dict[str, CrisisInterventionRecord] = {}
        self.follow_up_records: Dict[str, List[InterventionFollowUp]] = defaultdict(list)
        self.effectiveness_analyses: Dict[str, InterventionEffectivenessAnalysis] = {}
        self.aggregate_metrics: Dict[str, Any] = {}
        self.outcome_predictors: Dict[str, Any] = {}
        self.improvement_callbacks: List[Callable] = []
        
        # Initialize measurement systems
        self._setup_effectiveness_metrics()
        self._setup_outcome_predictors()
        self._setup_quality_indicators()
        self._setup_ethical_guidelines()
    
    def _setup_effectiveness_metrics(self):
        """Setup metrics for measuring intervention effectiveness"""
        self.effectiveness_metrics = {
            'immediate_safety': {
                'indicators': [
                    'crisis_de_escalation_success',
                    'immediate_harm_prevention',
                    'safety_plan_implementation',
                    'emergency_services_coordination'
                ],
                'weight': 0.4,  # Highest weight for immediate safety
                'measurement_timeframe': 'immediate'
            },
            'short_term_stabilization': {
                'indicators': [
                    'crisis_resolution_within_week',
                    'professional_engagement_success',
                    'support_system_activation',
                    'coping_strategy_implementation'
                ],
                'weight': 0.3,
                'measurement_timeframe': '1-4_weeks'
            },
            'medium_term_recovery': {
                'indicators': [
                    'sustained_improvement',
                    'crisis_recurrence_prevention',
                    'life_functioning_improvement',
                    'relationship_quality_enhancement'
                ],
                'weight': 0.2,
                'measurement_timeframe': '1-6_months'
            },
            'long_term_healing': {
                'indicators': [
                    'overall_life_satisfaction_improvement',
                    'resilience_building',
                    'meaning_making_progress',
                    'post_traumatic_growth'
                ],
                'weight': 0.1,
                'measurement_timeframe': '6+_months'
            }
        }
    
    def _setup_outcome_predictors(self):
        """Setup predictors for intervention outcomes"""
        self.outcome_predictors = {
            'positive_outcome_indicators': [
                'rapid_response_time',
                'person_engaged_willingly',
                'strong_support_network_present',
                'previous_positive_intervention_experience',
                'cultural_sensitivity_maintained',
                'trauma_informed_approach_used',
                'professional_resources_available',
                'safety_plan_collaboratively_developed'
            ],
            'risk_factors': [
                'delayed_response_time',
                'person_resistant_to_help',
                'isolation_from_support_networks',
                'multiple_concurrent_crises',
                'cultural_barriers_present',
                'trauma_triggers_activated',
                'limited_professional_resources',
                'safety_concerns_for_responders'
            ],
            'intervention_quality_factors': [
                'responder_training_level',
                'cultural_competency',
                'trauma_informed_training',
                'crisis_intervention_experience',
                'emotional_regulation_skills',
                'active_listening_proficiency',
                'safety_assessment_accuracy'
            ]
        }
    
    def _setup_quality_indicators(self):
        """Setup quality indicators for crisis interventions"""
        self.quality_indicators = {
            'person_centered_care': [
                'person_dignity_maintained',
                'choices_and_autonomy_respected',
                'cultural_values_honored',
                'individual_needs_addressed',
                'collaborative_approach_used'
            ],
            'trauma_informed_care': [
                'safety_prioritized',
                'trustworthiness_maintained',
                'choice_and_control_supported',
                'collaboration_emphasized',
                'empowerment_promoted'
            ],
            'cultural_sensitivity': [
                'cultural_background_considered',
                'language_preferences_accommodated',
                'cultural_practices_respected',
                'family_dynamics_understood',
                'community_resources_utilized'
            ],
            'ethical_practice': [
                'consent_obtained_when_possible',
                'confidentiality_maintained',
                'boundaries_respected',
                'dual_relationships_avoided',
                'cultural_humility_demonstrated'
            ]
        }
    
    def _setup_ethical_guidelines(self):
        """Setup ethical guidelines for crisis intervention metrics"""
        self.ethical_guidelines = {
            'data_collection': {
                'principles': [
                    'minimize_data_collection_during_crisis',
                    'prioritize_consent_when_capacity_allows',
                    'anonymize_data_immediately',
                    'secure_storage_protocols',
                    'limited_access_controls'
                ],
                'prohibited_practices': [
                    'data_collection_that_delays_intervention',
                    'sharing_identifiable_information',
                    'using_crisis_data_for_non_intervention_purposes',
                    'retaining_data_beyond_necessity'
                ]
            },
            'measurement_ethics': {
                'principles': [
                    'measurement_serves_improvement_not_judgment',
                    'focus_on_system_improvement_not_individual_blame',
                    'respect_for_persons_in_vulnerable_states',
                    'beneficence_guides_all_measurement_decisions'
                ]
            }
        }
    
    async def record_crisis_intervention(self, intervention_record: CrisisInterventionRecord) -> str:
        """Record a crisis intervention with full ethical compliance"""
        try:
            # Verify ethical compliance
            if not await self._verify_ethical_compliance(intervention_record):
                logger.error("Intervention record failed ethical compliance check")
                return None
            
            # Anonymize sensitive data
            anonymized_record = await self._anonymize_intervention_record(intervention_record)
            
            # Store record
            self.intervention_records[intervention_record.intervention_id] = anonymized_record
            
            # Update aggregate metrics
            await self._update_aggregate_metrics(anonymized_record)
            
            # Trigger improvement analysis if needed
            await self._check_improvement_triggers(anonymized_record)
            
            logger.info(f"Crisis intervention recorded: {intervention_record.intervention_id}")
            return intervention_record.intervention_id
            
        except Exception as e:
            logger.error(f"Error recording crisis intervention: {e}")
            return None
    
    async def _verify_ethical_compliance(self, record: CrisisInterventionRecord) -> bool:
        """Verify that the intervention record meets ethical standards"""
        # Check privacy compliance
        if not record.privacy_compliance:
            return False
        
        # Verify data minimization
        if len(record.intervention_notes) > 1000:  # Limit note length
            logger.warning("Intervention notes exceed recommended length for privacy")
        
        # Check for potentially identifying information
        if await self._contains_identifying_info(record.intervention_notes):
            logger.error("Intervention notes contain potentially identifying information")
            return False
        
        # Verify trauma-informed approaches were used
        if not record.trauma_informed_approaches:
            logger.warning("No trauma-informed approaches documented")
        
        return True
    
    async def _contains_identifying_info(self, text: str) -> bool:
        """Check if text contains potentially identifying information"""
        # Simple heuristic checks - in production would use more sophisticated NLP
        identifying_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
            r'\b\d{3}-\d{3}-\d{4}\b',  # Phone pattern
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email pattern
            r'\b\d{1,5}\s\w+\s(Street|St|Avenue|Ave|Road|Rd|Drive|Dr)\b'  # Address pattern
        ]
        
        import re
        for pattern in identifying_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    async def _anonymize_intervention_record(self, record: CrisisInterventionRecord) -> CrisisInterventionRecord:
        """Anonymize intervention record while preserving analytical value"""
        # Create anonymized copy
        anonymized = CrisisInterventionRecord(
            intervention_id=record.intervention_id,
            crisis_event_id=record.crisis_event_id,
            person_id=await self._anonymize_person_id(record.person_id),
            crisis_type=record.crisis_type,
            intervention_types=record.intervention_types,
            response_speed=record.response_speed,
            intervention_start=record.intervention_start,
            intervention_end=record.intervention_end,
            outcome=record.outcome,
            life_impact_level=record.life_impact_level,
            cultural_considerations=record.cultural_considerations,
            trauma_informed_approaches=record.trauma_informed_approaches,
            barriers_encountered=record.barriers_encountered,
            success_factors=record.success_factors,
            professional_handoff=record.professional_handoff,
            follow_up_planned=record.follow_up_planned,
            follow_up_completed=record.follow_up_completed,
            safety_plan_created=record.safety_plan_created,
            resource_connections_made=record.resource_connections_made,
            intervention_notes=await self._anonymize_notes(record.intervention_notes),
            responder_feedback=await self._anonymize_feedback(record.responder_feedback),
            person_feedback=await self._anonymize_feedback(record.person_feedback) if record.person_feedback else None,
            ethical_considerations=record.ethical_considerations,
            privacy_compliance=record.privacy_compliance,
            consent_status=record.consent_status
        )
        
        return anonymized
    
    async def _anonymize_person_id(self, person_id: str) -> str:
        """Create anonymous but consistent identifier"""
        import hashlib
        return hashlib.sha256(person_id.encode()).hexdigest()[:16]
    
    async def _anonymize_notes(self, notes: str) -> str:
        """Anonymize notes while preserving analytical content"""
        # Remove potentially identifying information
        # This is a simplified version - production would use more sophisticated NLP
        anonymized = notes
        
        # Remove names (simple pattern)
        import re
        anonymized = re.sub(r'\b[A-Z][a-z]+\s[A-Z][a-z]+\b', '[NAME]', anonymized)
        
        # Remove addresses
        anonymized = re.sub(r'\b\d{1,5}\s\w+\s(Street|St|Avenue|Ave|Road|Rd|Drive|Dr)\b', '[ADDRESS]', anonymized)
        
        # Remove phone numbers
        anonymized = re.sub(r'\b\d{3}-\d{3}-\d{4}\b', '[PHONE]', anonymized)
        
        # Limit length for privacy
        if len(anonymized) > 500:
            anonymized = anonymized[:500] + "[TRUNCATED_FOR_PRIVACY]"
        
        return anonymized
    
    async def _anonymize_feedback(self, feedback: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Anonymize feedback while preserving useful information"""
        if not feedback:
            return None
        
        anonymized = {}
        for key, value in feedback.items():
            if isinstance(value, str):
                anonymized[key] = await self._anonymize_notes(value)
            elif isinstance(value, (int, float, bool)):
                anonymized[key] = value
            else:
                anonymized[key] = str(value)  # Convert complex types to string
        
        return anonymized
    
    async def _update_aggregate_metrics(self, record: CrisisInterventionRecord):
        """Update aggregate metrics with new intervention record"""
        current_month = record.intervention_start.strftime("%Y-%m")
        
        if current_month not in self.aggregate_metrics:
            self.aggregate_metrics[current_month] = {
                'total_interventions': 0,
                'outcomes': defaultdict(int),
                'crisis_types': defaultdict(int),
                'response_speeds': defaultdict(int),
                'life_impact_levels': defaultdict(int),
                'average_effectiveness': 0.0,
                'cultural_considerations': defaultdict(int),
                'trauma_informed_usage': 0,
                'professional_handoffs': 0,
                'follow_up_completion_rate': 0.0
            }
        
        metrics = self.aggregate_metrics[current_month]
        metrics['total_interventions'] += 1
        metrics['outcomes'][record.outcome.value] += 1
        metrics['crisis_types'][record.crisis_type.value] += 1
        metrics['response_speeds'][record.response_speed.value] += 1
        metrics['life_impact_levels'][record.life_impact_level.value] += 1
        
        if record.trauma_informed_approaches:
            metrics['trauma_informed_usage'] += 1
        
        if record.professional_handoff:
            metrics['professional_handoffs'] += 1
        
        for consideration in record.cultural_considerations:
            metrics['cultural_considerations'][consideration] += 1
    
    async def _check_improvement_triggers(self, record: CrisisInterventionRecord):
        """Check if intervention patterns suggest need for system improvements"""
        # Check for concerning patterns
        if record.outcome in [InterventionOutcome.UNSUCCESSFUL, InterventionOutcome.LOST_CONTACT]:
            await self._analyze_unsuccessful_intervention(record)
        
        if record.response_speed == ResponseSpeed.DELAYED:
            await self._analyze_delayed_response(record)
        
        if record.life_impact_level == LifeImpactLevel.POTENTIAL_HARM:
            await self._analyze_potential_harm(record)
    
    async def _analyze_unsuccessful_intervention(self, record: CrisisInterventionRecord):
        """Analyze unsuccessful interventions to identify improvement opportunities"""
        analysis = {
            'intervention_id': record.intervention_id,
            'outcome': record.outcome.value,
            'barriers': record.barriers_encountered,
            'crisis_type': record.crisis_type.value,
            'response_speed': record.response_speed.value,
            'cultural_factors': record.cultural_considerations,
            'improvement_opportunities': []
        }
        
        # Identify improvement opportunities
        if 'language_barrier' in record.barriers_encountered:
            analysis['improvement_opportunities'].append('enhance_multilingual_support')
        
        if 'cultural_misunderstanding' in record.barriers_encountered:
            analysis['improvement_opportunities'].append('improve_cultural_competency_training')
        
        if record.response_speed in [ResponseSpeed.DELAYED, ResponseSpeed.NEXT_DAY]:
            analysis['improvement_opportunities'].append('improve_response_time_systems')
        
        # Trigger improvement callbacks
        for callback in self.improvement_callbacks:
            try:
                await callback(analysis)
            except Exception as e:
                logger.error(f"Error in improvement callback: {e}")
    
    async def record_follow_up(self, follow_up: InterventionFollowUp) -> str:
        """Record follow-up for crisis intervention"""
        try:
            # Anonymize follow-up data
            anonymized_follow_up = await self._anonymize_follow_up(follow_up)
            
            # Store follow-up
            self.follow_up_records[follow_up.intervention_id].append(anonymized_follow_up)
            
            # Update intervention record
            if follow_up.intervention_id in self.intervention_records:
                self.intervention_records[follow_up.intervention_id].follow_up_completed = True
            
            # Analyze effectiveness
            await self._analyze_intervention_effectiveness(follow_up.intervention_id)
            
            logger.info(f"Follow-up recorded: {follow_up.follow_up_id}")
            return follow_up.follow_up_id
            
        except Exception as e:
            logger.error(f"Error recording follow-up: {e}")
            return None
    
    async def _anonymize_follow_up(self, follow_up: InterventionFollowUp) -> InterventionFollowUp:
        """Anonymize follow-up record"""
        return InterventionFollowUp(
            follow_up_id=follow_up.follow_up_id,
            intervention_id=follow_up.intervention_id,
            person_id=await self._anonymize_person_id(follow_up.person_id),
            follow_up_date=follow_up.follow_up_date,
            contact_method=follow_up.contact_method,
            contact_successful=follow_up.contact_successful,
            person_status=follow_up.person_status,
            improvement_indicators=follow_up.improvement_indicators,
            ongoing_concerns=follow_up.ongoing_concerns,
            additional_resources_needed=follow_up.additional_resources_needed,
            satisfaction_rating=follow_up.satisfaction_rating,
            would_recommend=follow_up.would_recommend,
            safety_plan_adherence=follow_up.safety_plan_adherence,
            professional_support_engagement=follow_up.professional_support_engagement,
            crisis_recurrence=follow_up.crisis_recurrence,
            long_term_stability=follow_up.long_term_stability,
            follow_up_notes=await self._anonymize_notes(follow_up.follow_up_notes)
        )
    
    async def _analyze_intervention_effectiveness(self, intervention_id: str):
        """Analyze the effectiveness of a specific intervention"""
        if intervention_id not in self.intervention_records:
            return
        
        record = self.intervention_records[intervention_id]
        follow_ups = self.follow_up_records.get(intervention_id, [])
        
        if not follow_ups:
            return
        
        # Calculate effectiveness scores
        immediate_score = await self._calculate_immediate_effectiveness(record)
        short_term_score = await self._calculate_short_term_effectiveness(record, follow_ups)
        medium_term_score = await self._calculate_medium_term_effectiveness(follow_ups)
        long_term_score = await self._calculate_long_term_effectiveness(follow_ups)
        
        # Create effectiveness analysis
        analysis = InterventionEffectivenessAnalysis(
            analysis_id=f"eff_{intervention_id}_{datetime.now().isoformat()}",
            intervention_id=intervention_id,
            time_period=f"{record.intervention_start.date()}_to_{follow_ups[-1].follow_up_date.date()}",
            immediate_effectiveness=immediate_score,
            short_term_effectiveness=short_term_score,
            medium_term_effectiveness=medium_term_score,
            long_term_effectiveness=long_term_score,
            life_impact_score=await self._calculate_life_impact_score(record, follow_ups),
            quality_of_life_change=await self._calculate_quality_of_life_change(follow_ups),
            crisis_recurrence_rate=await self._calculate_crisis_recurrence_rate(follow_ups),
            professional_engagement_success=await self._calculate_professional_engagement_success(follow_ups),
            support_network_strengthening=await self._calculate_support_network_strengthening(follow_ups),
            cultural_sensitivity_score=await self._calculate_cultural_sensitivity_score(record),
            trauma_informed_score=await self._calculate_trauma_informed_score(record),
            ethical_compliance_score=await self._calculate_ethical_compliance_score(record),
            lessons_learned=await self._extract_lessons_learned(record, follow_ups),
            improvement_recommendations=await self._generate_improvement_recommendations(record, follow_ups)
        )
        
        self.effectiveness_analyses[intervention_id] = analysis
    
    async def _calculate_immediate_effectiveness(self, record: CrisisInterventionRecord) -> float:
        """Calculate immediate effectiveness score"""
        score = 50.0  # Base score
        
        # Outcome impact
        outcome_scores = {
            InterventionOutcome.CRISIS_RESOLVED: 100,
            InterventionOutcome.CRISIS_STABILIZED: 80,
            InterventionOutcome.ESCALATION_PREVENTED: 90,
            InterventionOutcome.PROFESSIONAL_REFERRAL: 75,
            InterventionOutcome.ONGOING_MONITORING: 60,
            InterventionOutcome.INTERVENTION_DECLINED: 30,
            InterventionOutcome.LOST_CONTACT: 20,
            InterventionOutcome.UNSUCCESSFUL: 10
        }
        score = outcome_scores.get(record.outcome, 50)
        
        # Life impact bonus
        impact_bonuses = {
            LifeImpactLevel.LIFE_SAVING: 20,
            LifeImpactLevel.MAJOR_HARM_PREVENTION: 15,
            LifeImpactLevel.MODERATE_IMPROVEMENT: 10,
            LifeImpactLevel.STABILIZATION: 5,
            LifeImpactLevel.MINOR_SUPPORT: 2,
            LifeImpactLevel.NO_MEASURABLE_IMPACT: 0,
            LifeImpactLevel.POTENTIAL_HARM: -20
        }
        score += impact_bonuses.get(record.life_impact_level, 0)
        
        # Response speed bonus
        speed_bonuses = {
            ResponseSpeed.IMMEDIATE: 10,
            ResponseSpeed.RAPID: 5,
            ResponseSpeed.SAME_DAY: 0,
            ResponseSpeed.NEXT_DAY: -5,
            ResponseSpeed.DELAYED: -10
        }
        score += speed_bonuses.get(record.response_speed, 0)
        
        return max(0, min(100, score))
    
    async def get_intervention_effectiveness_report(self, time_period: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive effectiveness report"""
        if time_period:
            analyses = {k: v for k, v in self.effectiveness_analyses.items() 
                       if time_period in v.time_period}
        else:
            analyses = self.effectiveness_analyses
        
        if not analyses:
            return {}
        
        # Calculate aggregate metrics
        all_immediate = [a.immediate_effectiveness for a in analyses.values()]
        all_short_term = [a.short_term_effectiveness for a in analyses.values()]
        all_life_impact = [a.life_impact_score for a in analyses.values()]
        
        return {
            'report_period': time_period or 'all_time',
            'total_interventions_analyzed': len(analyses),
            'average_immediate_effectiveness': statistics.mean(all_immediate),
            'average_short_term_effectiveness': statistics.mean(all_short_term),
            'average_life_impact_score': statistics.mean(all_life_impact),
            'outcome_distribution': await self._calculate_outcome_distribution(analyses),
            'most_effective_intervention_types': await self._identify_most_effective_interventions(analyses),
            'improvement_opportunities': await self._identify_improvement_opportunities(analyses),
            'cultural_sensitivity_average': statistics.mean([a.cultural_sensitivity_score for a in analyses.values()]),
            'trauma_informed_average': statistics.mean([a.trauma_informed_score for a in analyses.values()]),
            'ethical_compliance_average': statistics.mean([a.ethical_compliance_score for a in analyses.values()]),
            'report_generated': datetime.now().isoformat()
        }
    
    async def add_improvement_callback(self, callback: Callable):
        """Add callback for improvement opportunities"""
        self.improvement_callbacks.append(callback)