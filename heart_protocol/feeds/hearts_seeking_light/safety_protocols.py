"""
False Positive Prevention using Safety Protocols

Comprehensive safety system to prevent harmful mismatches and protect users
from inappropriate connections. Built on trauma-informed care principles.
"""

import asyncio
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
import logging
import re

from .support_detection import SupportSeeker, SupportOffer, SupportType, SupportUrgency
from .consent_system import ConsentManager, ConsentLevel, ConsentScope

logger = logging.getLogger(__name__)


class SafetyLevel(Enum):
    """Safety risk levels for potential connections"""
    SAFE = "safe"                    # Low risk, standard protocols
    CAUTIOUS = "cautious"           # Moderate risk, enhanced monitoring
    HIGH_RISK = "high_risk"         # High risk, requires review
    DANGEROUS = "dangerous"         # Dangerous, block connection
    REQUIRES_HUMAN = "requires_human"  # Needs human assessment


class SafetyFlag(Enum):
    """Types of safety concerns"""
    VULNERABILITY_MISMATCH = "vulnerability_mismatch"
    INEXPERIENCED_SUPPORTER = "inexperienced_supporter"
    BOUNDARY_VIOLATIONS = "boundary_violations" 
    INAPPROPRIATE_MOTIVATION = "inappropriate_motivation"
    SKILL_MISMATCH = "skill_mismatch"
    CULTURAL_INSENSITIVITY = "cultural_insensitivity"
    PREDATORY_PATTERNS = "predatory_patterns"
    EMOTIONAL_OVERWHELM = "emotional_overwhelm"
    UNQUALIFIED_ADVICE = "unqualified_advice"
    CRISIS_INAPPROPRIATE = "crisis_inappropriate"
    CONSENT_VIOLATIONS = "consent_violations"
    SPAM_PATTERNS = "spam_patterns"


@dataclass
class SafetyAssessment:
    """Assessment of safety for a potential connection"""
    safety_level: SafetyLevel
    safety_flags: List[SafetyFlag]
    risk_score: float  # 0.0 (safe) to 1.0 (dangerous)
    confidence: float  # 0.0 to 1.0
    reasoning: str
    recommendations: List[str]
    monitoring_required: bool
    human_review_required: bool
    allowed_interactions: Set[ConsentScope]
    blocked_interactions: Set[ConsentScope]
    safety_conditions: List[str]
    escalation_triggers: List[str]
    
    def is_connection_safe(self) -> bool:
        """Check if connection is safe to proceed"""
        return self.safety_level in [SafetyLevel.SAFE, SafetyLevel.CAUTIOUS]
    
    def requires_intervention(self) -> bool:
        """Check if safety intervention is required"""
        return self.safety_level in [SafetyLevel.HIGH_RISK, SafetyLevel.DANGEROUS, SafetyLevel.REQUIRES_HUMAN]


class SafetyProtocolEngine:
    """
    Comprehensive safety engine to prevent harmful connections and protect users.
    
    Core Safety Principles:
    - Trauma-informed care: Avoid retraumatization
    - Vulnerability protection: Protect those in crisis or distress
    - Boundary enforcement: Respect user limits and preferences
    - Qualified support: Match skills to needs appropriately
    - Human oversight: Escalate complex or risky situations
    - Continuous monitoring: Watch for emerging safety concerns
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Safety assessment rules and patterns
        self.vulnerability_patterns = self._initialize_vulnerability_patterns()
        self.predatory_patterns = self._initialize_predatory_patterns()
        self.qualification_requirements = self._initialize_qualification_requirements()
        self.boundary_rules = self._initialize_boundary_rules()
        self.monitoring_protocols = self._initialize_monitoring_protocols()
        
        # Safety tracking and analytics
        self.safety_incidents = []
        self.blocked_connections = {}  # user_id -> List[blocked_connection_info]
        self.safety_metrics = {
            'assessments_performed': 0,
            'connections_blocked': 0,
            'human_reviews_triggered': 0,
            'safety_incidents_detected': 0,
            'false_positives_reported': 0
        }
        
        logger.info("Safety Protocol Engine initialized")
    
    def _initialize_vulnerability_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize patterns that indicate user vulnerability"""
        
        return {
            'crisis_vulnerability': {
                'indicators': [
                    'suicidal ideation', 'self harm', 'crisis', 'emergency',
                    'cant take it', 'give up', 'end it all', 'no hope',
                    'want to die', 'hurt myself', 'not worth living'
                ],
                'risk_multiplier': 3.0,
                'special_protections': [
                    'crisis_qualified_supporters_only',
                    'immediate_human_oversight',
                    'professional_backup_required',
                    'continuous_monitoring'
                ]
            },
            
            'trauma_vulnerability': {
                'indicators': [
                    'trauma', 'ptsd', 'abuse', 'assault', 'violence',
                    'flashbacks', 'triggered', 'retraumatized',
                    'survivor', 'recovering from'
                ],
                'risk_multiplier': 2.0,
                'special_protections': [
                    'trauma_informed_supporters_only',
                    'gentle_approach_required',
                    'trigger_warning_awareness',
                    'professional_referral_ready'
                ]
            },
            
            'isolation_vulnerability': {
                'indicators': [
                    'alone', 'isolated', 'no friends', 'nobody cares',
                    'abandoned', 'rejected', 'outcasted', 'lonely',
                    'no support system', 'cut off from everyone'
                ],
                'risk_multiplier': 1.5,
                'special_protections': [
                    'gradual_trust_building',
                    'community_connection_focus',
                    'boundary_respect_critical'
                ]
            },
            
            'first_time_seeker': {
                'indicators': [
                    'first time', 'never asked for help', 'dont usually',
                    'hard to reach out', 'not used to this',
                    'new to seeking support'
                ],
                'risk_multiplier': 1.3,
                'special_protections': [
                    'experienced_supporters_preferred',
                    'extra_gentleness_required',
                    'clear_expectations_setting'
                ]
            }
        }
    
    def _initialize_predatory_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize patterns that indicate potentially predatory behavior"""
        
        return {
            'grooming_patterns': {
                'indicators': [
                    'special connection', 'understand you better', 'between us',
                    'dont tell others', 'our secret', 'trust me completely',
                    'only i can help', 'others dont understand', 'isolate from'
                ],
                'severity': 'high',
                'automatic_block': True
            },
            
            'boundary_pushing': {
                'indicators': [
                    'give me your number', 'meet in person', 'private photos',
                    'personal details', 'where do you live', 'relationship status',
                    'bypass platform', 'ignore boundaries', 'special exception'
                ],
                'severity': 'high',
                'automatic_block': False,  # Requires context
                'human_review': True
            },
            
            'inappropriate_motivation': {
                'indicators': [
                    'lonely too', 'looking for connection', 'need someone',
                    'relationship', 'dating', 'romantic', 'attracted',
                    'fix you', 'save you', 'rescue', 'only i can'
                ],
                'severity': 'medium',
                'automatic_block': False,
                'enhanced_monitoring': True
            },
            
            'qualification_inflation': {
                'indicators': [
                    'professional therapist', 'certified counselor', 'doctor',
                    'licensed', 'trained professional', 'expert in',
                    'years of experience', 'medical advice'
                ],
                'severity': 'medium',
                'verification_required': True
            },
            
            'urgency_exploitation': {
                'indicators': [
                    'act now', 'limited time', 'urgent decision',
                    'cant wait', 'immediate action', 'trust me now',
                    'no time to think', 'others will judge'
                ],
                'severity': 'medium',
                'automatic_flag': True
            }
        }
    
    def _initialize_qualification_requirements(self) -> Dict[SupportType, Dict[str, Any]]:
        """Initialize qualification requirements for different support types"""
        
        return {
            SupportType.CRISIS_INTERVENTION: {
                'required_qualifications': [
                    'crisis_intervention_training',
                    'suicide_prevention_certification',
                    'mental_health_first_aid'
                ],
                'minimum_experience': 'professional_or_extensive_peer',
                'human_oversight': 'required',
                'maximum_concurrent': 1,
                'response_time_requirement': timedelta(minutes=5)
            },
            
            SupportType.PROFESSIONAL_RESOURCES: {
                'required_qualifications': [
                    'knowledge_of_local_resources',
                    'understanding_of_referral_process'
                ],
                'minimum_experience': 'moderate_peer_or_professional',
                'verification_required': True
            },
            
            SupportType.EMOTIONAL_SUPPORT: {
                'required_qualifications': [
                    'active_listening_skills',
                    'empathy_and_compassion',
                    'boundary_awareness'
                ],
                'minimum_experience': 'basic_peer_or_lived_experience',
                'training_recommended': [
                    'peer_support_basics',
                    'trauma_informed_care'
                ]
            },
            
            SupportType.PRACTICAL_ADVICE: {
                'required_qualifications': [
                    'relevant_experience_in_area',
                    'problem_solving_skills'
                ],
                'minimum_experience': 'lived_experience_or_training',
                'disclaimer_required': True
            },
            
            SupportType.SHARED_EXPERIENCE: {
                'required_qualifications': [
                    'similar_life_experience',
                    'some_healing_progress'
                ],
                'minimum_experience': 'lived_experience',
                'recovery_stability_required': True
            }
        }
    
    def _initialize_boundary_rules(self) -> Dict[str, List[str]]:
        """Initialize boundary enforcement rules"""
        
        return {
            'communication_boundaries': [
                'respect_preferred_communication_methods',
                'honor_response_time_preferences',
                'respect_topic_boundaries',
                'maintain_appropriate_relationship_level'
            ],
            
            'privacy_boundaries': [
                'no_personal_information_requests',
                'respect_anonymity_preferences',
                'no_platform_circumvention',
                'secure_communication_only'
            ],
            
            'emotional_boundaries': [
                'respect_emotional_capacity',
                'no_emotional_dumping',
                'mutual_support_only_when_appropriate',
                'respect_healing_pace'
            ],
            
            'professional_boundaries': [
                'clear_role_definitions',
                'no_diagnosis_or_medical_advice',
                'appropriate_scope_of_support',
                'referral_when_needed'
            ]
        }
    
    def _initialize_monitoring_protocols(self) -> Dict[str, Dict[str, Any]]:
        """Initialize monitoring protocols for ongoing safety"""
        
        return {
            'high_risk_monitoring': {
                'frequency': timedelta(hours=2),
                'indicators_to_watch': [
                    'escalating_distress',
                    'boundary_violations',
                    'inappropriate_direction',
                    'supporter_overwhelm'
                ],
                'automatic_interventions': [
                    'human_reviewer_notification',
                    'support_check_in',
                    'resource_provision'
                ]
            },
            
            'crisis_monitoring': {
                'frequency': timedelta(minutes=30),
                'continuous_availability': True,
                'escalation_triggers': [
                    'no_response_to_check_in',
                    'escalating_crisis_language',
                    'safety_plan_breakdown'
                ],
                'immediate_actions': [
                    'crisis_team_notification',
                    'emergency_contact_protocols',
                    'professional_handoff'
                ]
            },
            
            'routine_monitoring': {
                'frequency': timedelta(days=1),
                'check_points': [
                    'support_relationship_health',
                    'boundary_respect',
                    'progress_indicators',
                    'satisfaction_levels'
                ]
            }
        }
    
    async def assess_connection_safety(self, seeker: SupportSeeker, 
                                     offer: SupportOffer,
                                     context: Dict[str, Any]) -> SafetyAssessment:
        """
        Comprehensive safety assessment for a potential support connection.
        
        Args:
            seeker: Person seeking support
            offer: Support offer being evaluated
            context: Additional context about the connection
        """
        try:
            safety_flags = []
            risk_score = 0.0
            reasoning_parts = []
            recommendations = []
            monitoring_required = False
            human_review_required = False
            
            # Assess seeker vulnerability
            vulnerability_assessment = await self._assess_seeker_vulnerability(seeker)
            risk_score += vulnerability_assessment['risk_contribution']
            safety_flags.extend(vulnerability_assessment['flags'])
            reasoning_parts.append(vulnerability_assessment['reasoning'])
            
            if vulnerability_assessment['special_protections']:
                monitoring_required = True
            
            # Assess supporter qualifications
            qualification_assessment = await self._assess_supporter_qualifications(offer, seeker)
            risk_score += qualification_assessment['risk_contribution']
            safety_flags.extend(qualification_assessment['flags'])
            reasoning_parts.append(qualification_assessment['reasoning'])
            
            if qualification_assessment['human_review_needed']:
                human_review_required = True
            
            # Check for predatory patterns
            predatory_assessment = await self._check_predatory_patterns(offer, context)
            risk_score += predatory_assessment['risk_contribution']
            safety_flags.extend(predatory_assessment['flags'])
            reasoning_parts.append(predatory_assessment['reasoning'])
            
            if predatory_assessment['automatic_block']:
                risk_score = 1.0  # Maximum risk
                human_review_required = True
            
            # Assess communication compatibility
            communication_assessment = await self._assess_communication_safety(seeker, offer)
            risk_score += communication_assessment['risk_contribution']
            safety_flags.extend(communication_assessment['flags'])
            reasoning_parts.append(communication_assessment['reasoning'])
            
            # Check for boundary violations
            boundary_assessment = await self._check_boundary_violations(seeker, offer, context)
            risk_score += boundary_assessment['risk_contribution']
            safety_flags.extend(boundary_assessment['flags'])
            reasoning_parts.append(boundary_assessment['reasoning'])
            
            # Determine safety level
            safety_level = self._calculate_safety_level(risk_score, safety_flags)
            
            # Generate recommendations
            recommendations = await self._generate_safety_recommendations(
                safety_level, safety_flags, seeker, offer
            )
            
            # Determine allowed/blocked interactions
            allowed_interactions, blocked_interactions = await self._determine_interaction_permissions(
                safety_level, safety_flags, seeker, offer
            )
            
            # Set up monitoring and escalation
            safety_conditions = await self._generate_safety_conditions(safety_level, safety_flags)
            escalation_triggers = await self._generate_escalation_triggers(safety_level, seeker)
            
            # Calculate confidence
            confidence = await self._calculate_assessment_confidence(
                vulnerability_assessment, qualification_assessment, predatory_assessment
            )
            
            assessment = SafetyAssessment(
                safety_level=safety_level,
                safety_flags=safety_flags,
                risk_score=min(1.0, risk_score),
                confidence=confidence,
                reasoning='; '.join(filter(None, reasoning_parts)),
                recommendations=recommendations,
                monitoring_required=monitoring_required or safety_level in [SafetyLevel.CAUTIOUS, SafetyLevel.HIGH_RISK],
                human_review_required=human_review_required or safety_level in [SafetyLevel.HIGH_RISK, SafetyLevel.DANGEROUS, SafetyLevel.REQUIRES_HUMAN],
                allowed_interactions=allowed_interactions,
                blocked_interactions=blocked_interactions,
                safety_conditions=safety_conditions,
                escalation_triggers=escalation_triggers
            )
            
            # Update metrics
            self.safety_metrics['assessments_performed'] += 1
            
            if safety_level in [SafetyLevel.HIGH_RISK, SafetyLevel.DANGEROUS]:
                self.safety_metrics['connections_blocked'] += 1
            
            if human_review_required:
                self.safety_metrics['human_reviews_triggered'] += 1
            
            logger.info(f"Safety assessment completed: {seeker.user_id[:8]}... -> {offer.user_id[:8]}... "
                       f"(Level: {safety_level.value}, Risk: {risk_score:.2f})")
            
            return assessment
            
        except Exception as e:
            logger.error(f"Error in safety assessment: {e}")
            return await self._get_fallback_safety_assessment(seeker, offer)
    
    async def _assess_seeker_vulnerability(self, seeker: SupportSeeker) -> Dict[str, Any]:
        """Assess vulnerability level of support seeker"""
        
        assessment = {
            'risk_contribution': 0.0,
            'flags': [],
            'reasoning': '',
            'special_protections': []
        }
        
        # Check for crisis indicators
        if seeker.urgency == SupportUrgency.CRISIS:
            assessment['risk_contribution'] += 0.4
            assessment['flags'].append(SafetyFlag.VULNERABILITY_MISMATCH)
            assessment['reasoning'] = 'Crisis situation requires enhanced safety protocols'
            assessment['special_protections'].extend(
                self.vulnerability_patterns['crisis_vulnerability']['special_protections']
            )
        
        # Check content for vulnerability patterns
        content_indicators = seeker.context_summary.lower() + ' '.join(seeker.specific_needs).lower()
        
        for pattern_name, pattern_config in self.vulnerability_patterns.items():
            indicators_found = [
                indicator for indicator in pattern_config['indicators']
                if indicator in content_indicators
            ]
            
            if indicators_found:
                risk_contribution = 0.1 * len(indicators_found) * pattern_config['risk_multiplier']
                assessment['risk_contribution'] += risk_contribution
                
                if risk_contribution > 0.2:
                    assessment['flags'].append(SafetyFlag.VULNERABILITY_MISMATCH)
                
                assessment['special_protections'].extend(
                    pattern_config.get('special_protections', [])
                )
        
        # Check trigger warnings
        if seeker.trigger_warnings:
            assessment['risk_contribution'] += 0.1 * len(seeker.trigger_warnings)
            assessment['flags'].append(SafetyFlag.EMOTIONAL_OVERWHELM)
        
        return assessment
    
    async def _assess_supporter_qualifications(self, offer: SupportOffer, 
                                             seeker: SupportSeeker) -> Dict[str, Any]:
        """Assess if supporter is qualified for the needed support types"""
        
        assessment = {
            'risk_contribution': 0.0,
            'flags': [],
            'reasoning': '',
            'human_review_needed': False
        }
        
        # Check qualifications for each support type
        for support_type in seeker.support_types_needed:
            if support_type not in offer.support_types_offered:
                continue  # Not offering this type of support
            
            requirements = self.qualification_requirements.get(support_type, {})
            
            # Check required qualifications
            required_quals = requirements.get('required_qualifications', [])
            supporter_quals = offer.credentials
            
            missing_quals = [
                qual for qual in required_quals
                if not any(qual_keyword in ' '.join(supporter_quals) 
                          for qual_keyword in qual.split('_'))
            ]
            
            if missing_quals:
                if support_type == SupportType.CRISIS_INTERVENTION:
                    assessment['risk_contribution'] += 0.6  # Very high risk
                    assessment['flags'].append(SafetyFlag.CRISIS_INAPPROPRIATE)
                    assessment['human_review_needed'] = True
                elif support_type == SupportType.PROFESSIONAL_RESOURCES:
                    assessment['risk_contribution'] += 0.3
                    assessment['flags'].append(SafetyFlag.UNQUALIFIED_ADVICE)
                else:
                    assessment['risk_contribution'] += 0.1
                    assessment['flags'].append(SafetyFlag.SKILL_MISMATCH)
            
            # Check experience level
            min_experience = requirements.get('minimum_experience', '')
            if 'professional' in min_experience and 'therapist' not in supporter_quals:
                assessment['risk_contribution'] += 0.2
                assessment['flags'].append(SafetyFlag.INEXPERIENCED_SUPPORTER)
        
        # Check supporter capacity
        if offer.current_support_count >= offer.max_concurrent_support:
            assessment['risk_contribution'] += 0.2
            assessment['flags'].append(SafetyFlag.EMOTIONAL_OVERWHELM)
        
        return assessment
    
    async def _check_predatory_patterns(self, offer: SupportOffer, 
                                      context: Dict[str, Any]) -> Dict[str, Any]:
        """Check for predatory behavior patterns"""
        
        assessment = {
            'risk_contribution': 0.0,
            'flags': [],
            'reasoning': '',
            'automatic_block': False
        }
        
        # Check offer content and history
        offer_content = getattr(offer, 'offer_text', '') + ' '.join(offer.experience_areas)
        
        for pattern_name, pattern_config in self.predatory_patterns.items():
            indicators_found = [
                indicator for indicator in pattern_config['indicators']
                if indicator in offer_content.lower()
            ]
            
            if indicators_found:
                severity = pattern_config['severity']
                
                if severity == 'high':
                    assessment['risk_contribution'] += 0.5
                    assessment['flags'].append(SafetyFlag.PREDATORY_PATTERNS)
                    
                    if pattern_config.get('automatic_block', False):
                        assessment['automatic_block'] = True
                        assessment['reasoning'] = f'Predatory pattern detected: {pattern_name}'
                
                elif severity == 'medium':
                    assessment['risk_contribution'] += 0.2
                    assessment['flags'].append(SafetyFlag.INAPPROPRIATE_MOTIVATION)
        
        # Check for qualification inflation
        if any('professional' in cred.lower() or 'licensed' in cred.lower() 
               for cred in offer.credentials):
            if not context.get('credentials_verified', False):
                assessment['risk_contribution'] += 0.3
                assessment['flags'].append(SafetyFlag.UNQUALIFIED_ADVICE)
        
        return assessment
    
    async def _assess_communication_safety(self, seeker: SupportSeeker, 
                                         offer: SupportOffer) -> Dict[str, Any]:
        """Assess safety of communication patterns"""
        
        assessment = {
            'risk_contribution': 0.0,
            'flags': [],
            'reasoning': ''
        }
        
        # Check communication preference compatibility
        seeker_prefs = set(seeker.communication_preferences)
        offer_prefs = set(offer.communication_preferences)
        
        # Warning if only private communication is possible for vulnerable users
        if (seeker.urgency in [SupportUrgency.CRISIS, SupportUrgency.HIGH] and
            seeker_prefs & offer_prefs == {'private_message'}):
            assessment['risk_contribution'] += 0.1
            assessment['flags'].append(SafetyFlag.BOUNDARY_VIOLATIONS)
        
        # Check timezone mismatches for crisis support
        if (seeker.urgency == SupportUrgency.CRISIS and
            seeker.timezone != offer.timezone):
            assessment['risk_contribution'] += 0.1
            assessment['reasoning'] = 'Timezone mismatch may delay crisis response'
        
        return assessment
    
    async def _check_boundary_violations(self, seeker: SupportSeeker, 
                                       offer: SupportOffer, 
                                       context: Dict[str, Any]) -> Dict[str, Any]:
        """Check for potential boundary violations"""
        
        assessment = {
            'risk_contribution': 0.0,
            'flags': [],
            'reasoning': ''
        }
        
        # Check for anonymity mismatches
        if (seeker.anonymity_preference == 'fully_anonymous' and
            'private_message' in offer.communication_preferences):
            assessment['risk_contribution'] += 0.1
            assessment['flags'].append(SafetyFlag.BOUNDARY_VIOLATIONS)
        
        # Check for demographic mismatches that could be problematic
        seeker_demo_prefs = seeker.preferred_demographics
        offer_demographics = offer.demographic_info
        
        if seeker_demo_prefs.get('gender_preference'):
            preferred_gender = seeker_demo_prefs['gender_preference']
            offer_gender = offer_demographics.get('gender')
            
            if offer_gender and preferred_gender != offer_gender:
                assessment['risk_contribution'] += 0.05
                assessment['reasoning'] = 'Gender preference mismatch'
        
        return assessment
    
    def _calculate_safety_level(self, risk_score: float, 
                               safety_flags: List[SafetyFlag]) -> SafetyLevel:
        """Calculate overall safety level based on risk score and flags"""
        
        # Automatic dangerous for predatory patterns
        if SafetyFlag.PREDATORY_PATTERNS in safety_flags:
            return SafetyLevel.DANGEROUS
        
        # Requires human review for crisis mismatches
        if SafetyFlag.CRISIS_INAPPROPRIATE in safety_flags:
            return SafetyLevel.REQUIRES_HUMAN
        
        # Risk score thresholds
        if risk_score >= 0.8:
            return SafetyLevel.DANGEROUS
        elif risk_score >= 0.6:
            return SafetyLevel.HIGH_RISK
        elif risk_score >= 0.3:
            return SafetyLevel.CAUTIOUS
        else:
            return SafetyLevel.SAFE
    
    async def _generate_safety_recommendations(self, safety_level: SafetyLevel,
                                             safety_flags: List[SafetyFlag],
                                             seeker: SupportSeeker,
                                             offer: SupportOffer) -> List[str]:
        """Generate safety recommendations based on assessment"""
        
        recommendations = []
        
        if safety_level == SafetyLevel.DANGEROUS:
            recommendations.append("Block this connection immediately")
            recommendations.append("Report to safety team for investigation")
            return recommendations
        
        if SafetyFlag.CRISIS_INAPPROPRIATE in safety_flags:
            recommendations.append("Refer crisis support to qualified crisis counselors")
            recommendations.append("Activate professional crisis response protocols")
        
        if SafetyFlag.VULNERABILITY_MISMATCH in safety_flags:
            recommendations.append("Provide additional supporter training before connection")
            recommendations.append("Implement enhanced monitoring protocols")
        
        if SafetyFlag.BOUNDARY_VIOLATIONS in safety_flags:
            recommendations.append("Establish clear communication boundaries")
            recommendations.append("Monitor for boundary respect")
        
        if SafetyFlag.INEXPERIENCED_SUPPORTER in safety_flags:
            recommendations.append("Pair with experienced mentor supporter")
            recommendations.append("Provide additional training resources")
        
        if safety_level == SafetyLevel.CAUTIOUS:
            recommendations.append("Proceed with enhanced monitoring")
            recommendations.append("Regular check-ins on connection health")
        
        return recommendations
    
    async def _determine_interaction_permissions(self, safety_level: SafetyLevel,
                                               safety_flags: List[SafetyFlag],
                                               seeker: SupportSeeker,
                                               offer: SupportOffer) -> Tuple[Set[ConsentScope], Set[ConsentScope]]:
        """Determine allowed and blocked interaction types"""
        
        # Start with all requested scopes
        all_scopes = {
            ConsentScope.PUBLIC_REPLY,
            ConsentScope.PRIVATE_MESSAGE,
            ConsentScope.RESOURCE_SHARING,
            ConsentScope.EMOTIONAL_SUPPORT,
            ConsentScope.PRACTICAL_ADVICE,
            ConsentScope.CRISIS_CONTACT,
            ConsentScope.ONGOING_CHECKIN,
            ConsentScope.COMMUNITY_INTRO,
            ConsentScope.PROFESSIONAL_REFERRAL
        }
        
        allowed_interactions = set()
        blocked_interactions = set()
        
        if safety_level == SafetyLevel.DANGEROUS:
            blocked_interactions = all_scopes
            return allowed_interactions, blocked_interactions
        
        # Safe interactions for all levels
        safe_interactions = {
            ConsentScope.PUBLIC_REPLY,
            ConsentScope.RESOURCE_SHARING,
            ConsentScope.COMMUNITY_INTRO
        }
        
        allowed_interactions.update(safe_interactions)
        
        # Additional permissions based on safety level
        if safety_level == SafetyLevel.SAFE:
            allowed_interactions.update(all_scopes)
        
        elif safety_level == SafetyLevel.CAUTIOUS:
            allowed_interactions.update({
                ConsentScope.EMOTIONAL_SUPPORT,
                ConsentScope.PRACTICAL_ADVICE,
                ConsentScope.PRIVATE_MESSAGE
            })
            
            # Block crisis contact unless qualified
            if SafetyFlag.CRISIS_INAPPROPRIATE in safety_flags:
                blocked_interactions.add(ConsentScope.CRISIS_CONTACT)
        
        elif safety_level in [SafetyLevel.HIGH_RISK, SafetyLevel.REQUIRES_HUMAN]:
            # Very limited interactions
            blocked_interactions.update({
                ConsentScope.PRIVATE_MESSAGE,
                ConsentScope.CRISIS_CONTACT,
                ConsentScope.ONGOING_CHECKIN
            })
        
        # Apply specific flag restrictions
        if SafetyFlag.BOUNDARY_VIOLATIONS in safety_flags:
            blocked_interactions.add(ConsentScope.PRIVATE_MESSAGE)
        
        if SafetyFlag.UNQUALIFIED_ADVICE in safety_flags:
            blocked_interactions.add(ConsentScope.PROFESSIONAL_REFERRAL)
        
        # Remove blocked from allowed
        allowed_interactions -= blocked_interactions
        
        return allowed_interactions, blocked_interactions
    
    async def _generate_safety_conditions(self, safety_level: SafetyLevel,
                                         safety_flags: List[SafetyFlag]) -> List[str]:
        """Generate safety conditions for the connection"""
        
        conditions = []
        
        if safety_level in [SafetyLevel.CAUTIOUS, SafetyLevel.HIGH_RISK]:
            conditions.append("Regular safety check-ins required")
            conditions.append("Boundary compliance monitoring active")
        
        if SafetyFlag.VULNERABILITY_MISMATCH in safety_flags:
            conditions.append("Enhanced trauma-informed approach required")
            conditions.append("Immediate escalation for any concerning behavior")
        
        if SafetyFlag.INEXPERIENCED_SUPPORTER in safety_flags:
            conditions.append("Supervisor oversight required")
            conditions.append("Additional training completion before unsupervised support")
        
        return conditions
    
    async def _generate_escalation_triggers(self, safety_level: SafetyLevel,
                                          seeker: SupportSeeker) -> List[str]:
        """Generate triggers that would escalate safety concerns"""
        
        triggers = [
            "Any indication of increased distress or crisis",
            "Boundary violations by either party",
            "Inappropriate relationship development",
            "Signs of exploitation or manipulation"
        ]
        
        if seeker.urgency == SupportUrgency.CRISIS:
            triggers.extend([
                "No response to safety check-ins",
                "Escalating suicidal ideation",
                "Breakdown of safety planning"
            ])
        
        if safety_level in [SafetyLevel.CAUTIOUS, SafetyLevel.HIGH_RISK]:
            triggers.extend([
                "Unusual communication patterns",
                "Requests for personal information",
                "Attempts to move communication off-platform"
            ])
        
        return triggers
    
    async def _calculate_assessment_confidence(self, *assessments) -> float:
        """Calculate confidence in the safety assessment"""
        
        # Base confidence
        confidence = 0.8
        
        # Reduce confidence for incomplete information
        for assessment in assessments:
            if assessment.get('incomplete_information', False):
                confidence -= 0.1
        
        # Increase confidence for clear patterns
        total_flags = sum(len(a.get('flags', [])) for a in assessments)
        if total_flags > 3:
            confidence += 0.1
        
        return max(0.3, min(1.0, confidence))
    
    async def _get_fallback_safety_assessment(self, seeker: SupportSeeker,
                                            offer: SupportOffer) -> SafetyAssessment:
        """Get conservative fallback assessment when main assessment fails"""
        
        return SafetyAssessment(
            safety_level=SafetyLevel.REQUIRES_HUMAN,
            safety_flags=[SafetyFlag.REQUIRES_HUMAN],
            risk_score=0.7,
            confidence=0.2,
            reasoning="Safety assessment failed - requires human review",
            recommendations=["Human safety review required before connection"],
            monitoring_required=True,
            human_review_required=True,
            allowed_interactions={ConsentScope.PUBLIC_REPLY, ConsentScope.RESOURCE_SHARING},
            blocked_interactions={ConsentScope.PRIVATE_MESSAGE, ConsentScope.CRISIS_CONTACT},
            safety_conditions=["Human review and approval required"],
            escalation_triggers=["Any concerning behavior or communication"]
        )
    
    async def report_safety_incident(self, incident_data: Dict[str, Any]) -> str:
        """Report and log a safety incident"""
        
        incident_id = f"incident_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{len(self.safety_incidents)}"
        
        incident_record = {
            'incident_id': incident_id,
            'timestamp': datetime.utcnow().isoformat(),
            'type': incident_data.get('type', 'unknown'),
            'severity': incident_data.get('severity', 'medium'),
            'users_involved': incident_data.get('users_involved', []),
            'description': incident_data.get('description', ''),
            'evidence': incident_data.get('evidence', {}),
            'immediate_actions_taken': incident_data.get('actions_taken', []),
            'investigation_required': incident_data.get('investigation_required', True)
        }
        
        self.safety_incidents.append(incident_record)
        self.safety_metrics['safety_incidents_detected'] += 1
        
        logger.error(f"Safety incident reported: {incident_id} - {incident_data.get('type', 'unknown')}")
        
        return incident_id
    
    def get_safety_metrics(self) -> Dict[str, Any]:
        """Get safety system metrics and analytics"""
        
        return {
            'assessments_performed': self.safety_metrics['assessments_performed'],
            'connections_blocked': self.safety_metrics['connections_blocked'],
            'human_reviews_triggered': self.safety_metrics['human_reviews_triggered'],
            'safety_incidents_detected': self.safety_metrics['safety_incidents_detected'],
            'false_positives_reported': self.safety_metrics['false_positives_reported'],
            'blocking_rate': (self.safety_metrics['connections_blocked'] / 
                             max(1, self.safety_metrics['assessments_performed'])) * 100,
            'human_review_rate': (self.safety_metrics['human_reviews_triggered'] / 
                                 max(1, self.safety_metrics['assessments_performed'])) * 100,
            'system_health': {
                'vulnerability_patterns_loaded': len(self.vulnerability_patterns) > 0,
                'predatory_patterns_loaded': len(self.predatory_patterns) > 0,
                'qualification_requirements_loaded': len(self.qualification_requirements) > 0,
                'monitoring_protocols_active': len(self.monitoring_protocols) > 0,
                'safety_incident_tracking_active': True
            },
            'recent_incidents': len([
                incident for incident in self.safety_incidents
                if datetime.fromisoformat(incident['timestamp']) > datetime.utcnow() - timedelta(days=7)
            ])
        }