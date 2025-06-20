"""
Crisis Detector

Specialized real-time crisis detection system with immediate intervention
capabilities and trauma-informed response protocols.
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import logging
import re
from collections import defaultdict, deque
import statistics

from .firehose_monitor import StreamEvent

logger = logging.getLogger(__name__)


class CrisisLevel(Enum):
    """Levels of crisis severity"""
    IMMEDIATE_DANGER = "immediate_danger"               # Life-threatening emergency
    SEVERE_CRISIS = "severe_crisis"                     # Severe mental health crisis
    MODERATE_CRISIS = "moderate_crisis"                 # Moderate crisis requiring intervention
    MILD_CRISIS = "mild_crisis"                        # Mild crisis, support needed
    ELEVATED_CONCERN = "elevated_concern"               # Concerning but not crisis
    WATCHFUL_MONITORING = "watchful_monitoring"         # Monitor for escalation


class InterventionUrgency(Enum):
    """Urgency levels for crisis intervention"""
    IMMEDIATE = "immediate"                             # Respond within seconds/minutes
    URGENT = "urgent"                                   # Respond within 15 minutes
    PRIORITY = "priority"                               # Respond within 1 hour
    SCHEDULED = "scheduled"                             # Respond within 4 hours
    ROUTINE = "routine"                                 # Respond within 24 hours


class CrisisType(Enum):
    """Types of crisis situations"""
    SUICIDAL_IDEATION = "suicidal_ideation"           # Thoughts of suicide
    SUICIDE_PLAN = "suicide_plan"                      # Active suicide planning
    SELF_HARM = "self_harm"                            # Self-harm behavior
    PSYCHOTIC_BREAK = "psychotic_break"                # Psychotic episode
    SEVERE_PANIC = "severe_panic"                      # Severe panic attack
    DOMESTIC_VIOLENCE = "domestic_violence"            # Domestic violence situation
    SUBSTANCE_CRISIS = "substance_crisis"              # Substance abuse crisis
    TRAUMA_RESPONSE = "trauma_response"                # Acute trauma response
    EATING_DISORDER_CRISIS = "eating_disorder_crisis" # Eating disorder emergency
    CHILD_SAFETY = "child_safety"                      # Child safety concern
    ELDER_ABUSE = "elder_abuse"                        # Elder abuse situation
    MEDICAL_EMERGENCY = "medical_emergency"            # Medical emergency


class ResponseChannel(Enum):
    """Channels for crisis response"""
    PLATFORM_DIRECT = "platform_direct"                # Direct platform response
    EMERGENCY_SERVICES = "emergency_services"          # Contact emergency services
    CRISIS_HOTLINE = "crisis_hotline"                  # Crisis hotline referral
    MENTAL_HEALTH_PROFESSIONAL = "mental_health_professional"  # Professional referral
    TRUSTED_CONTACT = "trusted_contact"                # Contact trusted person
    COMMUNITY_RESPONDER = "community_responder"        # Community crisis responder
    PEER_SUPPORT = "peer_support"                      # Peer crisis support
    FAMILY_NOTIFICATION = "family_notification"        # Notify family/friends


@dataclass
class CrisisAlert:
    """Crisis alert with detailed analysis"""
    alert_id: str
    original_event: StreamEvent
    crisis_level: CrisisLevel
    crisis_types: List[CrisisType]
    intervention_urgency: InterventionUrgency
    confidence_score: float
    risk_assessment: Dict[str, float]
    protective_factors: List[str]
    risk_factors: List[str]
    immediate_needs: List[str]
    cultural_considerations: List[str]
    trauma_informed_factors: List[str]
    privacy_sensitivity: str
    recommended_responses: List[ResponseChannel]
    intervention_plan: Dict[str, Any]
    safety_plan_elements: List[str]
    follow_up_requirements: Dict[str, Any]
    escalation_triggers: List[str]
    detected_at: datetime
    requires_human_review: bool
    legal_reporting_required: bool


@dataclass
class CrisisPattern:
    """Pattern for detecting crisis situations"""
    pattern_id: str
    crisis_type: CrisisType
    severity_indicators: Dict[str, List[str]]
    escalation_markers: List[str]
    protective_factor_indicators: List[str]
    cultural_variations: Dict[str, List[str]]
    false_positive_filters: List[str]
    confidence_weights: Dict[str, float]


@dataclass
class CrisisResponse:
    """Record of crisis response"""
    response_id: str
    alert_id: str
    response_channels: List[ResponseChannel]
    response_actions: List[str]
    professional_contacts: List[Dict[str, Any]]
    emergency_services_contacted: bool
    user_safety_status: str
    intervention_outcome: str
    follow_up_scheduled: bool
    follow_up_timing: Optional[datetime]
    response_completed_at: datetime
    effectiveness_score: float
    lessons_learned: List[str]


class CrisisDetector:
    """
    Specialized crisis detection system with immediate intervention capabilities.
    
    Core Principles:
    - Life safety is the absolute highest priority
    - Immediate response to severe crisis situations
    - Trauma-informed crisis intervention approaches
    - Cultural sensitivity in crisis response
    - Privacy protection while ensuring safety
    - Professional integration for serious situations
    - Family/support network engagement when appropriate
    - Continuous monitoring and follow-up
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Detection settings
        self.detection_enabled = self.config.get('detection_enabled', True)
        self.immediate_response_enabled = self.config.get('immediate_response_enabled', True)
        self.professional_integration_enabled = self.config.get('professional_integration_enabled', True)
        self.emergency_services_integration = self.config.get('emergency_services_integration', False)
        
        # Crisis patterns and detection
        self.crisis_patterns = self._initialize_crisis_patterns()
        self.risk_assessment_factors = self._initialize_risk_factors()
        self.protective_factors = self._initialize_protective_factors()
        
        # Crisis management
        self.active_alerts: Dict[str, CrisisAlert] = {}
        self.crisis_responses: List[CrisisResponse] = []
        self.monitoring_queue: asyncio.Queue = asyncio.Queue()
        
        # Performance tracking
        self.crises_detected = 0
        self.interventions_made = 0
        self.lives_potentially_saved = 0
        self.average_response_time_seconds = 0.0
        self.false_positive_rate = 0.0
        
        # Professional and emergency contacts
        self.crisis_professionals: List[Dict[str, Any]] = []
        self.emergency_contacts: Dict[str, Any] = {}
        self.crisis_hotlines: List[Dict[str, Any]] = []
        
        # Safety and integration systems
        self.emergency_service_integrator = None    # Would integrate with emergency services
        self.mental_health_professional_network = None  # Professional network integration
        self.crisis_hotline_connector = None        # Crisis hotline integration
        self.legal_reporting_system = None          # Legal reporting for mandatory situations
        
        # Callbacks and notifications
        self.crisis_detected_callbacks: List[Callable] = []
        self.intervention_executed_callbacks: List[Callable] = []
        self.professional_notification_callbacks: List[Callable] = []
        self.emergency_escalation_callbacks: List[Callable] = []
        
        # Crisis processing state
        self.detection_active = False
        self.processing_tasks: Set[asyncio.Task] = set()
    
    def _initialize_crisis_patterns(self) -> Dict[CrisisType, CrisisPattern]:
        """Initialize patterns for detecting different types of crises"""
        return {
            CrisisType.SUICIDAL_IDEATION: CrisisPattern(
                pattern_id="suicidal_ideation_v1",
                crisis_type=CrisisType.SUICIDAL_IDEATION,
                severity_indicators={
                    'mild': [
                        r'\b(sometimes think|wonder about|thought about)\s+(dying|death|not being here)\b',
                        r'\b(tired of living|life is hard|don\'t want to be here)\b'
                    ],
                    'moderate': [
                        r'\b(want to die|wish i was dead|better off dead)\b',
                        r'\b(thinking about suicide|suicidal thoughts|end it all)\b',
                        r'\b(don\'t want to wake up|permanent solution)\b'
                    ],
                    'severe': [
                        r'\b(going to kill myself|plan to die|ready to end)\b',
                        r'\b(goodbye|final message|last time)\b',
                        r'\b(have a plan|know how|when i die)\b'
                    ]
                },
                escalation_markers=[
                    'specific method mentioned', 'timeline indicated', 'final arrangements',
                    'saying goodbye', 'giving away possessions', 'sudden calmness'
                ],
                protective_factor_indicators=[
                    'reasons to live', 'support system', 'future plans', 'help seeking',
                    'therapy mentioned', 'medication compliance'
                ],
                cultural_variations={
                    'indirect_expression': ['burden', 'tired of fighting', 'rest', 'peace'],
                    'honor_cultures': ['shame', 'disgrace', 'family honor', 'reputation'],
                    'religious_contexts': ['sin', 'punishment', 'afterlife', 'suffering']
                },
                false_positive_filters=[
                    'fictional context', 'song lyrics', 'movie reference', 'academic discussion'
                ],
                confidence_weights={
                    'direct_statement': 0.9,
                    'method_mentioned': 0.8,
                    'timeline_given': 0.8,
                    'previous_attempts': 0.7,
                    'giving_away_items': 0.7
                }
            ),
            
            CrisisType.SELF_HARM: CrisisPattern(
                pattern_id="self_harm_v1",
                crisis_type=CrisisType.SELF_HARM,
                severity_indicators={
                    'mild': [
                        r'\b(hurt myself|self harm|cutting)\b',
                        r'\b(deserve pain|need to hurt|punish myself)\b'
                    ],
                    'moderate': [
                        r'\b(cutting again|relapsed|can\'t stop hurting)\b',
                        r'\b(razor|blade|sharp object)\b'
                    ],
                    'severe': [
                        r'\b(deep cuts|serious injury|medical attention)\b',
                        r'\b(overdose|pills|too many)\b'
                    ]
                },
                escalation_markers=[
                    'frequency increasing', 'severity increasing', 'method escalation',
                    'medical complications', 'infection risk'
                ],
                protective_factor_indicators=[
                    'reaching out', 'seeking help', 'distraction techniques',
                    'support person', 'therapy engagement'
                ],
                cultural_variations={
                    'shame_based': ['deserved', 'punishment', 'bad person'],
                    'control_based': ['only way', 'control pain', 'release']
                },
                false_positive_filters=[
                    'accidental injury', 'medical procedure', 'sports injury'
                ],
                confidence_weights={
                    'current_behavior': 0.9,
                    'escalation_pattern': 0.8,
                    'method_specificity': 0.7
                }
            ),
            
            CrisisType.DOMESTIC_VIOLENCE: CrisisPattern(
                pattern_id="domestic_violence_v1",
                crisis_type=CrisisType.DOMESTIC_VIOLENCE,
                severity_indicators={
                    'mild': [
                        r'\b(partner angry|relationship problems|fighting)\b',
                        r'\b(scared of|nervous around|walking on eggshells)\b'
                    ],
                    'moderate': [
                        r'\b(hit me|pushed me|threw something)\b',
                        r'\b(threatened|afraid|hiding)\b'
                    ],
                    'severe': [
                        r'\b(seriously hurt|hospital|police|weapon)\b',
                        r'\b(going to kill|death threats|escape)\b'
                    ]
                },
                escalation_markers=[
                    'violence increasing', 'threats escalating', 'isolation increasing',
                    'access to weapons', 'stalking behavior'
                ],
                protective_factor_indicators=[
                    'safety plan', 'support network', 'resources known',
                    'professional help', 'documentation'
                ],
                cultural_variations={
                    'family_shame': ['family honor', 'disgrace', 'cultural shame'],
                    'economic_dependence': ['nowhere to go', 'no money', 'children']
                },
                false_positive_filters=[
                    'fictional account', 'news discussion', 'advocacy post'
                ],
                confidence_weights={
                    'current_danger': 0.9,
                    'escalation_pattern': 0.8,
                    'immediate_threat': 0.9
                }
            ),
            
            CrisisType.PSYCHOTIC_BREAK: CrisisPattern(
                pattern_id="psychotic_break_v1",
                crisis_type=CrisisType.PSYCHOTIC_BREAK,
                severity_indicators={
                    'mild': [
                        r'\b(seeing things|hearing voices|not real)\b',
                        r'\b(confused|disoriented|mixed up)\b'
                    ],
                    'moderate': [
                        r'\b(can\'t tell|reality|losing it|going crazy)\b',
                        r'\b(paranoid|following me|watching)\b'
                    ],
                    'severe': [
                        r'\b(voices telling|command|must obey)\b',
                        r'\b(danger|threat|protect myself|they\'re coming)\b'
                    ]
                },
                escalation_markers=[
                    'command hallucinations', 'paranoid delusions', 'agitation',
                    'threat perception', 'disorganized thinking'
                ],
                protective_factor_indicators=[
                    'insight maintained', 'seeking help', 'medication compliance',
                    'support person present'
                ],
                cultural_variations={
                    'spiritual_context': ['spirits', 'ancestors', 'divine messages'],
                    'cultural_explanations': ['curse', 'evil eye', 'possession']
                },
                false_positive_filters=[
                    'sleep deprivation', 'substance use', 'spiritual experience'
                ],
                confidence_weights={
                    'reality_testing_loss': 0.8,
                    'command_hallucinations': 0.9,
                    'threat_perception': 0.8
                }
            ),
            
            CrisisType.CHILD_SAFETY: CrisisPattern(
                pattern_id="child_safety_v1",
                crisis_type=CrisisType.CHILD_SAFETY,
                severity_indicators={
                    'mild': [
                        r'\b(child upset|kid crying|behavior problems)\b',
                        r'\b(struggling with|difficult parenting)\b'
                    ],
                    'moderate': [
                        r'\b(child afraid|hiding|bruises|marks)\b',
                        r'\b(inappropriate touch|uncomfortable|secrets)\b'
                    ],
                    'severe': [
                        r'\b(child injured|hospital|abuse|neglect)\b',
                        r'\b(sexual abuse|molestation|immediate danger)\b'
                    ]
                },
                escalation_markers=[
                    'physical evidence', 'disclosure by child', 'behavioral changes',
                    'access to child by perpetrator'
                ],
                protective_factor_indicators=[
                    'safe adult present', 'authorities notified', 'child removed',
                    'professional intervention'
                ],
                cultural_variations={
                    'discipline_norms': ['cultural discipline', 'traditional methods'],
                    'family_privacy': ['family matter', 'private business']
                },
                false_positive_filters=[
                    'general parenting stress', 'normal discipline', 'developmental issues'
                ],
                confidence_weights={
                    'physical_evidence': 0.9,
                    'child_disclosure': 0.9,
                    'immediate_danger': 1.0
                }
            )
        }
    
    def _initialize_risk_factors(self) -> Dict[str, Dict[str, float]]:
        """Initialize risk assessment factors"""
        return {
            'suicide_risk': {
                'previous_attempts': 0.8,
                'mental_illness': 0.6,
                'substance_abuse': 0.7,
                'social_isolation': 0.6,
                'recent_loss': 0.5,
                'chronic_pain': 0.4,
                'financial_stress': 0.3,
                'relationship_problems': 0.4,
                'job_loss': 0.3,
                'access_to_means': 0.7
            },
            'violence_risk': {
                'history_of_violence': 0.8,
                'substance_abuse': 0.6,
                'mental_illness': 0.4,
                'access_to_weapons': 0.7,
                'escalating_threats': 0.8,
                'social_isolation': 0.5,
                'financial_stress': 0.3,
                'recent_separation': 0.6
            },
            'self_harm_risk': {
                'previous_self_harm': 0.8,
                'emotional_dysregulation': 0.7,
                'trauma_history': 0.6,
                'social_isolation': 0.5,
                'perfectionism': 0.4,
                'body_image_issues': 0.5,
                'bullying_victimization': 0.6
            }
        }
    
    def _initialize_protective_factors(self) -> Dict[str, List[str]]:
        """Initialize protective factors"""
        return {
            'social_support': [
                'strong relationships', 'family support', 'friend network',
                'romantic partner', 'community connections', 'professional support'
            ],
            'coping_skills': [
                'therapy participation', 'medication compliance', 'stress management',
                'problem solving', 'emotional regulation', 'mindfulness practice'
            ],
            'life_factors': [
                'stable housing', 'employment', 'financial security',
                'future goals', 'responsibilities', 'pets', 'children'
            ],
            'help_seeking': [
                'actively seeking help', 'therapy engagement', 'crisis plan',
                'support group participation', 'medication management'
            ],
            'spiritual_cultural': [
                'religious beliefs', 'cultural values', 'spiritual practices',
                'community belonging', 'cultural identity'
            ]
        }
    
    async def start_detection(self) -> bool:
        """Start crisis detection"""
        try:
            if self.detection_active:
                logger.warning("Crisis detection already active")
                return True
            
            self.detection_active = True
            
            # Start detection tasks
            await self._start_detection_tasks()
            
            logger.info("Crisis detection started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start crisis detection: {str(e)}")
            self.detection_active = False
            return False
    
    async def stop_detection(self) -> bool:
        """Stop crisis detection"""
        try:
            self.detection_active = False
            
            # Cancel detection tasks
            for task in self.processing_tasks:
                if not task.done():
                    task.cancel()
            
            self.processing_tasks.clear()
            
            logger.info("Crisis detection stopped")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop crisis detection: {str(e)}")
            return False
    
    async def _start_detection_tasks(self):
        """Start detection tasks"""
        try:
            # Start main detection loop
            task = asyncio.create_task(self._detection_loop())
            self.processing_tasks.add(task)
            
            # Start monitoring loop
            task = asyncio.create_task(self._monitoring_loop())
            self.processing_tasks.add(task)
            
            # Start response coordinator
            task = asyncio.create_task(self._response_coordinator())
            self.processing_tasks.add(task)
            
        except Exception as e:
            logger.error(f"Failed to start detection tasks: {str(e)}")
    
    async def detect_crisis(self, event: StreamEvent) -> Optional[CrisisAlert]:
        """Detect crisis situations in stream events"""
        try:
            if not self.detection_enabled:
                return None
            
            # Add to monitoring queue for processing
            await self.monitoring_queue.put(event)
            
            return None  # Processing happens asynchronously
            
        except Exception as e:
            logger.error(f"Failed to queue event for crisis detection: {str(e)}")
            return None
    
    async def _detection_loop(self):
        """Main crisis detection loop"""
        try:
            while self.detection_active:
                try:
                    # Wait for event with timeout
                    event = await asyncio.wait_for(self.monitoring_queue.get(), timeout=1.0)
                    
                    # Analyze for crisis
                    alert = await self._analyze_for_crisis(event)
                    
                    if alert:
                        # Store alert
                        self.active_alerts[alert.alert_id] = alert
                        self.crises_detected += 1
                        
                        # Trigger immediate response if needed
                        if alert.intervention_urgency == InterventionUrgency.IMMEDIATE:
                            await self._trigger_immediate_response(alert)
                        
                        # Trigger callbacks
                        for callback in self.crisis_detected_callbacks:
                            try:
                                await callback(alert)
                            except Exception as e:
                                logger.error(f"Crisis detected callback failed: {str(e)}")
                    
                    # Mark queue task as done
                    self.monitoring_queue.task_done()
                    
                except asyncio.TimeoutError:
                    # No events in queue, continue waiting
                    continue
                except Exception as e:
                    logger.error(f"Error in detection loop: {str(e)}")
                    await asyncio.sleep(1)
                    
        except Exception as e:
            logger.error(f"Fatal error in detection loop: {str(e)}")
    
    async def _analyze_for_crisis(self, event: StreamEvent) -> Optional[CrisisAlert]:
        """Analyze event for crisis indicators"""
        try:
            content = event.processed_content.lower()
            detected_crises = []
            crisis_scores = {}
            
            # Analyze against each crisis pattern
            for crisis_type, pattern in self.crisis_patterns.items():
                crisis_score = await self._calculate_crisis_score(content, pattern)
                
                if crisis_score > 0.3:  # Threshold for crisis detection
                    detected_crises.append(crisis_type)
                    crisis_scores[crisis_type] = crisis_score
            
            if not detected_crises:
                return None
            
            # Determine overall crisis level and urgency
            highest_score = max(crisis_scores.values())
            primary_crisis = max(crisis_scores.items(), key=lambda x: x[1])[0]
            
            crisis_level = await self._determine_crisis_level(highest_score, detected_crises)
            intervention_urgency = await self._determine_intervention_urgency(crisis_level, detected_crises)
            
            # Calculate overall confidence
            confidence_score = await self._calculate_overall_confidence(content, detected_crises, crisis_scores)
            
            # Perform risk assessment
            risk_assessment = await self._perform_risk_assessment(content, detected_crises)
            
            # Identify protective and risk factors
            protective_factors = await self._identify_protective_factors(content)
            risk_factors = await self._identify_risk_factors(content, detected_crises)
            
            # Analyze cultural and trauma factors
            cultural_considerations = await self._analyze_cultural_considerations(content)
            trauma_informed_factors = await self._analyze_trauma_factors(content)
            
            # Determine immediate needs and responses
            immediate_needs = await self._identify_immediate_needs(detected_crises, crisis_level)
            recommended_responses = await self._recommend_response_channels(
                detected_crises, crisis_level, cultural_considerations
            )
            
            # Create intervention plan
            intervention_plan = await self._create_intervention_plan(
                detected_crises, crisis_level, risk_assessment
            )
            
            # Create safety plan elements
            safety_plan_elements = await self._create_safety_plan_elements(detected_crises, protective_factors)
            
            # Determine follow-up requirements
            follow_up_requirements = await self._determine_follow_up_requirements(crisis_level, detected_crises)
            
            # Identify escalation triggers
            escalation_triggers = await self._identify_escalation_triggers(detected_crises)
            
            # Assess privacy sensitivity
            privacy_sensitivity = await self._assess_privacy_sensitivity(content, detected_crises)
            
            # Check if human review required
            requires_human_review = (
                crisis_level in [CrisisLevel.IMMEDIATE_DANGER, CrisisLevel.SEVERE_CRISIS] or
                confidence_score < 0.7
            )
            
            # Check if legal reporting required
            legal_reporting_required = await self._check_legal_reporting_requirements(detected_crises, content)
            
            alert = CrisisAlert(
                alert_id=f"crisis_alert_{datetime.utcnow().isoformat()}_{id(self)}",
                original_event=event,
                crisis_level=crisis_level,
                crisis_types=detected_crises,
                intervention_urgency=intervention_urgency,
                confidence_score=confidence_score,
                risk_assessment=risk_assessment,
                protective_factors=protective_factors,
                risk_factors=risk_factors,
                immediate_needs=immediate_needs,
                cultural_considerations=cultural_considerations,
                trauma_informed_factors=trauma_informed_factors,
                privacy_sensitivity=privacy_sensitivity,
                recommended_responses=recommended_responses,
                intervention_plan=intervention_plan,
                safety_plan_elements=safety_plan_elements,
                follow_up_requirements=follow_up_requirements,
                escalation_triggers=escalation_triggers,
                detected_at=datetime.utcnow(),
                requires_human_review=requires_human_review,
                legal_reporting_required=legal_reporting_required
            )
            
            return alert
            
        except Exception as e:
            logger.error(f"Error analyzing for crisis: {str(e)}")
            return None
    
    async def _calculate_crisis_score(self, content: str, pattern: CrisisPattern) -> float:
        """Calculate crisis score for a specific pattern"""
        try:
            total_score = 0.0
            
            # Check severity indicators
            for severity, indicators in pattern.severity_indicators.items():
                severity_weight = {'mild': 0.3, 'moderate': 0.6, 'severe': 0.9}.get(severity, 0.3)
                
                for indicator in indicators:
                    if re.search(indicator, content):
                        total_score = max(total_score, severity_weight)
            
            # Check escalation markers
            escalation_score = 0.0
            for marker in pattern.escalation_markers:
                marker_patterns = self._get_marker_patterns(marker)
                if any(re.search(p, content) for p in marker_patterns):
                    escalation_score += 0.1
            
            total_score += min(0.3, escalation_score)
            
            # Apply confidence weights
            confidence_adjustment = 0.0
            for factor, weight in pattern.confidence_weights.items():
                factor_patterns = self._get_confidence_factor_patterns(factor)
                if any(re.search(p, content) for p in factor_patterns):
                    confidence_adjustment += (weight - 0.5) * 0.1
            
            total_score += confidence_adjustment
            
            # Check for false positive filters
            false_positive_penalty = 0.0
            for filter_term in pattern.false_positive_filters:
                filter_patterns = self._get_false_positive_patterns(filter_term)
                if any(re.search(p, content) for p in filter_patterns):
                    false_positive_penalty += 0.2
            
            total_score -= false_positive_penalty
            
            return max(0.0, min(1.0, total_score))
            
        except Exception as e:
            logger.error(f"Error calculating crisis score: {str(e)}")
            return 0.0
    
    def _get_marker_patterns(self, marker: str) -> List[str]:
        """Get regex patterns for escalation markers"""
        marker_patterns = {
            'specific method mentioned': [r'\b(pills|rope|gun|bridge|knife|overdose)\b'],
            'timeline indicated': [r'\b(tonight|tomorrow|this week|soon|ready)\b'],
            'final arrangements': [r'\b(will|testament|goodbye|final|last)\b'],
            'saying goodbye': [r'\b(goodbye|farewell|see you never|final message)\b'],
            'giving away possessions': [r'\b(giving away|take my|keep this|have my)\b'],
            'sudden calmness': [r'\b(peaceful|calm|decided|clear|relief)\b'],
            'frequency increasing': [r'\b(more often|increasing|worse|daily|constantly)\b'],
            'severity increasing': [r'\b(deeper|worse|more serious|escalating)\b'],
            'violence increasing': [r'\b(getting worse|more violent|escalating|angrier)\b'],
            'threats escalating': [r'\b(serious threat|going to|will hurt|promised)\b']
        }
        return marker_patterns.get(marker, [])
    
    def _get_confidence_factor_patterns(self, factor: str) -> List[str]:
        """Get patterns for confidence factors"""
        factor_patterns = {
            'direct_statement': [r'\b(i will|i am going to|i plan to)\b'],
            'method_mentioned': [r'\b(pills|rope|gun|knife|razor|bridge)\b'],
            'timeline_given': [r'\b(tonight|tomorrow|next week|when)\b'],
            'previous_attempts': [r'\b(tried before|attempted|last time)\b'],
            'current_behavior': [r'\b(right now|currently|just did|am doing)\b'],
            'escalation_pattern': [r'\b(getting worse|increasing|more frequent)\b'],
            'immediate_danger': [r'\b(right now|immediate|urgent|emergency)\b']
        }
        return factor_patterns.get(factor, [])
    
    def _get_false_positive_patterns(self, filter_term: str) -> List[str]:
        """Get patterns for false positive filters"""
        filter_patterns = {
            'fictional context': [r'\b(movie|book|story|character|fiction)\b'],
            'song lyrics': [r'\b(song|lyrics|music|album|artist)\b'],
            'movie reference': [r'\b(film|movie|scene|actor|director)\b'],
            'academic discussion': [r'\b(study|research|academic|theory|discussion)\b'],
            'accidental injury': [r'\b(accident|accidental|mistake|unintentional)\b'],
            'medical procedure': [r'\b(surgery|procedure|medical|doctor|hospital)\b'],
            'news discussion': [r'\b(news|article|report|journalist|media)\b'],
            'advocacy post': [r'\b(awareness|advocate|campaign|prevention)\b']
        }
        return filter_patterns.get(filter_term, [])
    
    async def _determine_crisis_level(self, highest_score: float, crisis_types: List[CrisisType]) -> CrisisLevel:
        """Determine overall crisis level"""
        try:
            # Immediate danger situations
            immediate_danger_types = [
                CrisisType.SUICIDE_PLAN, CrisisType.IMMEDIATE_DANGER,
                CrisisType.CHILD_SAFETY, CrisisType.MEDICAL_EMERGENCY
            ]
            
            if any(ct in immediate_danger_types for ct in crisis_types) and highest_score > 0.8:
                return CrisisLevel.IMMEDIATE_DANGER
            
            # Severe crisis situations
            if highest_score > 0.7 or CrisisType.SUICIDAL_IDEATION in crisis_types:
                return CrisisLevel.SEVERE_CRISIS
            
            # Moderate crisis
            if highest_score > 0.5:
                return CrisisLevel.MODERATE_CRISIS
            
            # Mild crisis
            if highest_score > 0.4:
                return CrisisLevel.MILD_CRISIS
            
            # Elevated concern
            return CrisisLevel.ELEVATED_CONCERN
            
        except Exception as e:
            logger.error(f"Error determining crisis level: {str(e)}")
            return CrisisLevel.WATCHFUL_MONITORING
    
    async def _determine_intervention_urgency(self, crisis_level: CrisisLevel, 
                                            crisis_types: List[CrisisType]) -> InterventionUrgency:
        """Determine intervention urgency"""
        try:
            if crisis_level == CrisisLevel.IMMEDIATE_DANGER:
                return InterventionUrgency.IMMEDIATE
            elif crisis_level == CrisisLevel.SEVERE_CRISIS:
                return InterventionUrgency.URGENT
            elif crisis_level == CrisisLevel.MODERATE_CRISIS:
                return InterventionUrgency.PRIORITY
            elif crisis_level == CrisisLevel.MILD_CRISIS:
                return InterventionUrgency.SCHEDULED
            else:
                return InterventionUrgency.ROUTINE
                
        except Exception as e:
            logger.error(f"Error determining intervention urgency: {str(e)}")
            return InterventionUrgency.ROUTINE
    
    async def _calculate_overall_confidence(self, content: str, crisis_types: List[CrisisType],
                                          crisis_scores: Dict[CrisisType, float]) -> float:
        """Calculate overall confidence in crisis detection"""
        try:
            if not crisis_scores:
                return 0.0
            
            # Base confidence from highest scoring crisis
            base_confidence = max(crisis_scores.values())
            
            # Multiple crisis types increase confidence
            multiple_crisis_bonus = min(0.2, (len(crisis_types) - 1) * 0.1)
            
            # Content length consideration
            length_factor = min(0.1, len(content) / 1000)
            
            # Specificity bonus
            specificity_bonus = 0.0
            specific_indicators = ['tonight', 'plan', 'method', 'ready', 'goodbye']
            specificity_matches = sum(1 for indicator in specific_indicators if indicator in content)
            specificity_bonus = min(0.2, specificity_matches * 0.05)
            
            total_confidence = base_confidence + multiple_crisis_bonus + length_factor + specificity_bonus
            
            return min(1.0, total_confidence)
            
        except Exception as e:
            logger.error(f"Error calculating overall confidence: {str(e)}")
            return 0.5
    
    async def _perform_risk_assessment(self, content: str, crisis_types: List[CrisisType]) -> Dict[str, float]:
        """Perform comprehensive risk assessment"""
        try:
            risk_assessment = {}
            
            # Assess suicide risk
            if CrisisType.SUICIDAL_IDEATION in crisis_types or CrisisType.SUICIDE_PLAN in crisis_types:
                suicide_risk = await self._assess_suicide_risk(content)
                risk_assessment['suicide_risk'] = suicide_risk
            
            # Assess violence risk
            if CrisisType.DOMESTIC_VIOLENCE in crisis_types:
                violence_risk = await self._assess_violence_risk(content)
                risk_assessment['violence_risk'] = violence_risk
            
            # Assess self-harm risk
            if CrisisType.SELF_HARM in crisis_types:
                self_harm_risk = await self._assess_self_harm_risk(content)
                risk_assessment['self_harm_risk'] = self_harm_risk
            
            # Assess imminent danger
            imminent_danger = await self._assess_imminent_danger(content, crisis_types)
            risk_assessment['imminent_danger'] = imminent_danger
            
            return risk_assessment
            
        except Exception as e:
            logger.error(f"Error performing risk assessment: {str(e)}")
            return {}
    
    async def _assess_suicide_risk(self, content: str) -> float:
        """Assess suicide risk level"""
        try:
            risk_score = 0.0
            
            # Check for risk factors
            for factor, weight in self.risk_assessment_factors['suicide_risk'].items():
                factor_patterns = self._get_risk_factor_patterns(factor)
                if any(pattern in content for pattern in factor_patterns):
                    risk_score += weight
            
            # Normalize score
            return min(1.0, risk_score / 3.0)  # Normalize by dividing by expected max
            
        except Exception as e:
            logger.error(f"Error assessing suicide risk: {str(e)}")
            return 0.5
    
    async def _assess_violence_risk(self, content: str) -> float:
        """Assess violence risk level"""
        try:
            risk_score = 0.0
            
            for factor, weight in self.risk_assessment_factors['violence_risk'].items():
                factor_patterns = self._get_risk_factor_patterns(factor)
                if any(pattern in content for pattern in factor_patterns):
                    risk_score += weight
            
            return min(1.0, risk_score / 3.0)
            
        except Exception as e:
            logger.error(f"Error assessing violence risk: {str(e)}")
            return 0.5
    
    async def _assess_self_harm_risk(self, content: str) -> float:
        """Assess self-harm risk level"""
        try:
            risk_score = 0.0
            
            for factor, weight in self.risk_assessment_factors['self_harm_risk'].items():
                factor_patterns = self._get_risk_factor_patterns(factor)
                if any(pattern in content for pattern in factor_patterns):
                    risk_score += weight
            
            return min(1.0, risk_score / 3.0)
            
        except Exception as e:
            logger.error(f"Error assessing self-harm risk: {str(e)}")
            return 0.5
    
    async def _assess_imminent_danger(self, content: str, crisis_types: List[CrisisType]) -> float:
        """Assess imminent danger level"""
        try:
            danger_score = 0.0
            
            # Immediate danger indicators
            immediate_indicators = [
                'right now', 'tonight', 'ready', 'going to', 'about to',
                'emergency', 'urgent', 'immediate', 'can\'t wait'
            ]
            
            immediate_matches = sum(1 for indicator in immediate_indicators if indicator in content)
            danger_score += immediate_matches * 0.2
            
            # Crisis type danger levels
            danger_levels = {
                CrisisType.SUICIDE_PLAN: 0.9,
                CrisisType.MEDICAL_EMERGENCY: 1.0,
                CrisisType.CHILD_SAFETY: 0.8,
                CrisisType.DOMESTIC_VIOLENCE: 0.7,
                CrisisType.SUICIDAL_IDEATION: 0.6
            }
            
            for crisis_type in crisis_types:
                if crisis_type in danger_levels:
                    danger_score = max(danger_score, danger_levels[crisis_type])
            
            return min(1.0, danger_score)
            
        except Exception as e:
            logger.error(f"Error assessing imminent danger: {str(e)}")
            return 0.5
    
    def _get_risk_factor_patterns(self, factor: str) -> List[str]:
        """Get patterns for risk factors"""
        factor_patterns = {
            'previous_attempts': ['tried before', 'attempted suicide', 'last time'],
            'mental_illness': ['depression', 'bipolar', 'schizophrenia', 'ptsd'],
            'substance_abuse': ['drinking', 'drugs', 'alcohol', 'high', 'addiction'],
            'social_isolation': ['alone', 'no friends', 'isolated', 'lonely'],
            'recent_loss': ['died', 'lost', 'breakup', 'divorce', 'fired'],
            'chronic_pain': ['chronic pain', 'constant pain', 'hurts', 'suffering'],
            'financial_stress': ['money problems', 'debt', 'bankrupt', 'poor'],
            'relationship_problems': ['fighting', 'relationship issues', 'breakup'],
            'job_loss': ['fired', 'unemployed', 'lost job', 'laid off'],
            'access_to_means': ['gun', 'pills', 'rope', 'knife', 'bridge'],
            'history_of_violence': ['violent before', 'hit', 'hurt', 'aggressive'],
            'access_to_weapons': ['gun', 'knife', 'weapon', 'rifle'],
            'escalating_threats': ['going to hurt', 'will kill', 'threatened'],
            'recent_separation': ['separated', 'divorce', 'left me', 'breakup'],
            'emotional_dysregulation': ['can\'t control', 'overwhelming', 'intense'],
            'trauma_history': ['abuse', 'trauma', 'ptsd', 'assault'],
            'perfectionism': ['perfect', 'never good enough', 'failure'],
            'body_image_issues': ['fat', 'ugly', 'hate my body', 'appearance'],
            'bullying_victimization': ['bullied', 'picked on', 'harassment']
        }
        return factor_patterns.get(factor, [])
    
    async def _identify_protective_factors(self, content: str) -> List[str]:
        """Identify protective factors present"""
        try:
            protective_factors = []
            
            for category, factors in self.protective_factors.items():
                for factor in factors:
                    factor_patterns = self._get_protective_factor_patterns(factor)
                    if any(pattern in content for pattern in factor_patterns):
                        protective_factors.append(factor)
            
            return protective_factors
            
        except Exception as e:
            logger.error(f"Error identifying protective factors: {str(e)}")
            return []
    
    def _get_protective_factor_patterns(self, factor: str) -> List[str]:
        """Get patterns for protective factors"""
        factor_patterns = {
            'strong relationships': ['close friend', 'supportive family', 'loving relationship'],
            'family support': ['family helps', 'parents support', 'family there'],
            'friend network': ['good friends', 'friends support', 'close friends'],
            'therapy participation': ['therapy', 'counseling', 'therapist'],
            'medication compliance': ['taking medication', 'meds help', 'prescribed'],
            'future goals': ['plans for', 'goals', 'looking forward', 'excited about'],
            'responsibilities': ['kids depend', 'job to do', 'responsible for'],
            'pets': ['my dog', 'my cat', 'pet needs me'],
            'religious beliefs': ['faith', 'god', 'church', 'prayer'],
            'cultural values': ['culture important', 'tradition', 'heritage']
        }
        return factor_patterns.get(factor, [])
    
    async def _identify_risk_factors(self, content: str, crisis_types: List[CrisisType]) -> List[str]:
        """Identify risk factors present"""
        try:
            risk_factors = []
            
            # General risk factor identification
            all_risk_factors = {}
            for risk_category in self.risk_assessment_factors.values():
                all_risk_factors.update(risk_category)
            
            for factor in all_risk_factors:
                factor_patterns = self._get_risk_factor_patterns(factor)
                if any(pattern in content for pattern in factor_patterns):
                    risk_factors.append(factor)
            
            return risk_factors
            
        except Exception as e:
            logger.error(f"Error identifying risk factors: {str(e)}")
            return []
    
    async def _analyze_cultural_considerations(self, content: str) -> List[str]:
        """Analyze cultural considerations for crisis response"""
        try:
            considerations = []
            
            # Cultural markers
            cultural_markers = {
                'family_honor': ['family shame', 'disgrace', 'honor', 'reputation'],
                'religious_considerations': ['sin', 'punishment', 'afterlife', 'god'],
                'collectivist_values': ['family', 'community', 'tradition', 'elders'],
                'language_barriers': ['english difficult', 'language problem', 'translator'],
                'traditional_healing': ['traditional medicine', 'spiritual healing', 'ceremony']
            }
            
            for consideration, markers in cultural_markers.items():
                if any(marker in content for marker in markers):
                    considerations.append(consideration)
            
            return considerations
            
        except Exception as e:
            logger.error(f"Error analyzing cultural considerations: {str(e)}")
            return []
    
    async def _analyze_trauma_factors(self, content: str) -> List[str]:
        """Analyze trauma-informed factors"""
        try:
            factors = []
            
            trauma_indicators = {
                'trauma_history': ['abuse', 'assault', 'trauma', 'violence'],
                'ptsd_symptoms': ['flashbacks', 'nightmares', 'triggered', 'hypervigilant'],
                'trust_issues': ['hard to trust', 'don\'t trust', 'betrayed'],
                'hypervigilance': ['on edge', 'watching', 'alert', 'suspicious'],
                'dissociation': ['not real', 'detached', 'floating', 'numb']
            }
            
            for factor, indicators in trauma_indicators.items():
                if any(indicator in content for indicator in indicators):
                    factors.append(factor)
            
            return factors
            
        except Exception as e:
            logger.error(f"Error analyzing trauma factors: {str(e)}")
            return []
    
    async def _identify_immediate_needs(self, crisis_types: List[CrisisType], 
                                      crisis_level: CrisisLevel) -> List[str]:
        """Identify immediate needs for crisis response"""
        try:
            needs = []
            
            # Crisis type specific needs
            if CrisisType.SUICIDAL_IDEATION in crisis_types or CrisisType.SUICIDE_PLAN in crisis_types:
                needs.extend([
                    'suicide_prevention_resources',
                    'crisis_hotline_contact',
                    'safety_planning',
                    'professional_assessment'
                ])
            
            if CrisisType.SELF_HARM in crisis_types:
                needs.extend([
                    'medical_assessment',
                    'harm_reduction_resources',
                    'therapeutic_support'
                ])
            
            if CrisisType.DOMESTIC_VIOLENCE in crisis_types:
                needs.extend([
                    'safety_planning',
                    'emergency_shelter_information',
                    'legal_resources',
                    'protective_services'
                ])
            
            if CrisisType.CHILD_SAFETY in crisis_types:
                needs.extend([
                    'child_protective_services',
                    'mandatory_reporting',
                    'immediate_safety_assessment'
                ])
            
            # Crisis level specific needs
            if crisis_level == CrisisLevel.IMMEDIATE_DANGER:
                needs.extend([
                    'emergency_services_contact',
                    'immediate_professional_intervention',
                    'continuous_monitoring'
                ])
            
            return list(set(needs))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Error identifying immediate needs: {str(e)}")
            return []
    
    async def _recommend_response_channels(self, crisis_types: List[CrisisType],
                                         crisis_level: CrisisLevel,
                                         cultural_considerations: List[str]) -> List[ResponseChannel]:
        """Recommend appropriate response channels"""
        try:
            channels = []
            
            # Emergency situations
            if crisis_level == CrisisLevel.IMMEDIATE_DANGER:
                channels.extend([
                    ResponseChannel.EMERGENCY_SERVICES,
                    ResponseChannel.CRISIS_HOTLINE,
                    ResponseChannel.MENTAL_HEALTH_PROFESSIONAL
                ])
            
            # Severe crisis
            elif crisis_level == CrisisLevel.SEVERE_CRISIS:
                channels.extend([
                    ResponseChannel.CRISIS_HOTLINE,
                    ResponseChannel.MENTAL_HEALTH_PROFESSIONAL,
                    ResponseChannel.PLATFORM_DIRECT
                ])
            
            # Moderate crisis
            elif crisis_level == CrisisLevel.MODERATE_CRISIS:
                channels.extend([
                    ResponseChannel.PLATFORM_DIRECT,
                    ResponseChannel.CRISIS_HOTLINE,
                    ResponseChannel.PEER_SUPPORT
                ])
            
            # Crisis type specific channels
            if CrisisType.CHILD_SAFETY in crisis_types:
                channels.append(ResponseChannel.EMERGENCY_SERVICES)
            
            if CrisisType.DOMESTIC_VIOLENCE in crisis_types:
                channels.extend([
                    ResponseChannel.EMERGENCY_SERVICES,
                    ResponseChannel.TRUSTED_CONTACT
                ])
            
            # Cultural considerations
            if 'family_honor' in cultural_considerations:
                # Be careful about family notification
                if ResponseChannel.FAMILY_NOTIFICATION in channels:
                    channels.remove(ResponseChannel.FAMILY_NOTIFICATION)
            
            if 'collectivist_values' in cultural_considerations:
                channels.append(ResponseChannel.COMMUNITY_RESPONDER)
            
            return list(set(channels))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Error recommending response channels: {str(e)}")
            return [ResponseChannel.PLATFORM_DIRECT]
    
    async def _create_intervention_plan(self, crisis_types: List[CrisisType],
                                      crisis_level: CrisisLevel,
                                      risk_assessment: Dict[str, float]) -> Dict[str, Any]:
        """Create intervention plan"""
        try:
            plan = {
                'immediate_actions': [],
                'short_term_goals': [],
                'long_term_objectives': [],
                'resource_connections': [],
                'monitoring_requirements': [],
                'escalation_criteria': []
            }
            
            # Immediate actions based on crisis level
            if crisis_level == CrisisLevel.IMMEDIATE_DANGER:
                plan['immediate_actions'].extend([
                    'contact_emergency_services',
                    'provide_crisis_hotline_information',
                    'continuous_monitoring_initiate'
                ])
            
            # Crisis type specific interventions
            if CrisisType.SUICIDAL_IDEATION in crisis_types:
                plan['immediate_actions'].extend([
                    'suicide_assessment',
                    'safety_plan_development',
                    'means_restriction_discussion'
                ])
                
                plan['short_term_goals'].extend([
                    'professional_mental_health_evaluation',
                    'support_system_activation',
                    'regular_check_ins'
                ])
            
            if CrisisType.DOMESTIC_VIOLENCE in crisis_types:
                plan['immediate_actions'].extend([
                    'safety_assessment',
                    'escape_plan_if_needed',
                    'legal_resource_information'
                ])
            
            # Add monitoring based on risk level
            if max(risk_assessment.values()) > 0.7:
                plan['monitoring_requirements'].append('intensive_monitoring')
            elif max(risk_assessment.values()) > 0.5:
                plan['monitoring_requirements'].append('regular_monitoring')
            else:
                plan['monitoring_requirements'].append('periodic_check_in')
            
            return plan
            
        except Exception as e:
            logger.error(f"Error creating intervention plan: {str(e)}")
            return {}
    
    async def _create_safety_plan_elements(self, crisis_types: List[CrisisType],
                                         protective_factors: List[str]) -> List[str]:
        """Create safety plan elements"""
        try:
            elements = []
            
            # Universal safety plan elements
            elements.extend([
                'crisis_hotline_numbers',
                'trusted_contact_information',
                'safe_environment_creation',
                'coping_strategies_list'
            ])
            
            # Crisis type specific elements
            if CrisisType.SUICIDAL_IDEATION in crisis_types or CrisisType.SUICIDE_PLAN in crisis_types:
                elements.extend([
                    'means_restriction_plan',
                    'warning_signs_identification',
                    'reasons_for_living_list',
                    'professional_contact_information'
                ])
            
            if CrisisType.SELF_HARM in crisis_types:
                elements.extend([
                    'alternative_coping_strategies',
                    'harm_reduction_techniques',
                    'distraction_activities',
                    'support_person_contact'
                ])
            
            if CrisisType.DOMESTIC_VIOLENCE in crisis_types:
                elements.extend([
                    'escape_route_planning',
                    'important_documents_preparation',
                    'safe_place_identification',
                    'legal_protection_information'
                ])
            
            # Leverage protective factors
            if 'strong relationships' in protective_factors:
                elements.append('support_network_activation_plan')
            
            if 'therapy participation' in protective_factors:
                elements.append('therapist_emergency_contact')
            
            if 'religious beliefs' in protective_factors:
                elements.append('spiritual_support_resources')
            
            return list(set(elements))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Error creating safety plan elements: {str(e)}")
            return []
    
    async def _determine_follow_up_requirements(self, crisis_level: CrisisLevel,
                                              crisis_types: List[CrisisType]) -> Dict[str, Any]:
        """Determine follow-up requirements"""
        try:
            requirements = {
                'frequency': 'daily',
                'duration': '1_week',
                'methods': ['platform_check', 'crisis_hotline_followup'],
                'escalation_criteria': [],
                'professional_involvement': False
            }
            
            # Adjust based on crisis level
            if crisis_level == CrisisLevel.IMMEDIATE_DANGER:
                requirements.update({
                    'frequency': 'continuous_for_24h_then_hourly',
                    'duration': '2_weeks',
                    'professional_involvement': True
                })
            elif crisis_level == CrisisLevel.SEVERE_CRISIS:
                requirements.update({
                    'frequency': 'every_2_hours_for_24h_then_daily',
                    'duration': '1_week',
                    'professional_involvement': True
                })
            elif crisis_level == CrisisLevel.MODERATE_CRISIS:
                requirements.update({
                    'frequency': 'daily',
                    'duration': '3_days'
                })
            
            # Crisis type specific requirements
            if CrisisType.CHILD_SAFETY in crisis_types:
                requirements.update({
                    'professional_involvement': True,
                    'legal_reporting_followup': True
                })
            
            return requirements
            
        except Exception as e:
            logger.error(f"Error determining follow-up requirements: {str(e)}")
            return {}
    
    async def _identify_escalation_triggers(self, crisis_types: List[CrisisType]) -> List[str]:
        """Identify escalation triggers"""
        try:
            triggers = []
            
            # Universal escalation triggers
            triggers.extend([
                'increased_desperation',
                'social_withdrawal',
                'giving_away_possessions',
                'sudden_mood_improvement_after_depression',
                'access_to_means_of_harm'
            ])
            
            # Crisis type specific triggers
            if CrisisType.SUICIDAL_IDEATION in crisis_types:
                triggers.extend([
                    'development_of_specific_plan',
                    'rehearsal_behaviors',
                    'final_arrangements',
                    'saying_goodbye'
                ])
            
            if CrisisType.DOMESTIC_VIOLENCE in crisis_types:
                triggers.extend([
                    'threat_escalation',
                    'weapon_access',
                    'stalking_behavior',
                    'violation_of_protective_orders'
                ])
            
            if CrisisType.SELF_HARM in crisis_types:
                triggers.extend([
                    'frequency_increase',
                    'severity_increase',
                    'method_escalation',
                    'medical_complications'
                ])
            
            return triggers
            
        except Exception as e:
            logger.error(f"Error identifying escalation triggers: {str(e)}")
            return []
    
    async def _assess_privacy_sensitivity(self, content: str, crisis_types: List[CrisisType]) -> str:
        """Assess privacy sensitivity level"""
        try:
            # High sensitivity situations
            if CrisisType.CHILD_SAFETY in crisis_types or CrisisType.DOMESTIC_VIOLENCE in crisis_types:
                return 'maximum_privacy_protection'
            
            # Check for explicit privacy requests
            privacy_indicators = ['private', 'confidential', 'don\'t share', 'secret']
            if any(indicator in content for indicator in privacy_indicators):
                return 'high_privacy_protection'
            
            # Default for crisis situations
            return 'standard_crisis_privacy'
            
        except Exception as e:
            logger.error(f"Error assessing privacy sensitivity: {str(e)}")
            return 'standard_crisis_privacy'
    
    async def _check_legal_reporting_requirements(self, crisis_types: List[CrisisType],
                                                content: str) -> bool:
        """Check if legal reporting is required"""
        try:
            # Mandatory reporting situations
            mandatory_reporting_types = [
                CrisisType.CHILD_SAFETY,
                CrisisType.ELDER_ABUSE
            ]
            
            if any(ct in mandatory_reporting_types for ct in crisis_types):
                return True
            
            # Specific content indicators for mandatory reporting
            mandatory_indicators = [
                'child abuse', 'child neglect', 'elder abuse', 'minor in danger'
            ]
            
            if any(indicator in content for indicator in mandatory_indicators):
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking legal reporting requirements: {str(e)}")
            return False
    
    async def _trigger_immediate_response(self, alert: CrisisAlert):
        """Trigger immediate response for critical situations"""
        try:
            logger.critical(f"IMMEDIATE CRISIS RESPONSE TRIGGERED: {alert.alert_id}")
            
            # Execute immediate response actions
            response = await self._execute_crisis_response(alert)
            
            if response:
                self.crisis_responses.append(response)
                self.interventions_made += 1
                
                # Calculate response time
                response_time = (response.response_completed_at - alert.detected_at).total_seconds()
                self.average_response_time_seconds = (
                    (self.average_response_time_seconds * (self.interventions_made - 1) + response_time)
                    / self.interventions_made
                )
            
        except Exception as e:
            logger.error(f"Error triggering immediate response: {str(e)}")
    
    async def _execute_crisis_response(self, alert: CrisisAlert) -> Optional[CrisisResponse]:
        """Execute crisis response plan"""
        try:
            start_time = datetime.utcnow()
            
            response_actions = []
            professional_contacts = []
            emergency_services_contacted = False
            
            # Execute recommended response channels
            for channel in alert.recommended_responses:
                if channel == ResponseChannel.EMERGENCY_SERVICES:
                    # In production, would integrate with emergency services
                    response_actions.append('emergency_services_notification_sent')
                    emergency_services_contacted = True
                
                elif channel == ResponseChannel.CRISIS_HOTLINE:
                    response_actions.append('crisis_hotline_information_provided')
                
                elif channel == ResponseChannel.MENTAL_HEALTH_PROFESSIONAL:
                    response_actions.append('mental_health_professional_notification_sent')
                    professional_contacts.append({
                        'type': 'mental_health_professional',
                        'contacted_at': datetime.utcnow().isoformat()
                    })
                
                elif channel == ResponseChannel.PLATFORM_DIRECT:
                    response_actions.append('direct_platform_support_message_sent')
            
            # Execute intervention plan actions
            for action in alert.intervention_plan.get('immediate_actions', []):
                response_actions.append(f"intervention_action_{action}")
            
            # Simulate response effectiveness (in production, would be measured)
            effectiveness_score = await self._simulate_response_effectiveness(alert)
            
            response = CrisisResponse(
                response_id=f"crisis_response_{datetime.utcnow().isoformat()}_{id(self)}",
                alert_id=alert.alert_id,
                response_channels=alert.recommended_responses,
                response_actions=response_actions,
                professional_contacts=professional_contacts,
                emergency_services_contacted=emergency_services_contacted,
                user_safety_status='monitoring_initiated',
                intervention_outcome='response_executed',
                follow_up_scheduled=True,
                follow_up_timing=datetime.utcnow() + timedelta(hours=1),
                response_completed_at=datetime.utcnow(),
                effectiveness_score=effectiveness_score,
                lessons_learned=await self._extract_response_lessons(alert, response_actions)
            )
            
            # Trigger callbacks
            for callback in self.intervention_executed_callbacks:
                try:
                    await callback(response)
                except Exception as e:
                    logger.error(f"Intervention executed callback failed: {str(e)}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error executing crisis response: {str(e)}")
            return None
    
    async def _simulate_response_effectiveness(self, alert: CrisisAlert) -> float:
        """Simulate response effectiveness (placeholder for real measurement)"""
        try:
            # In production, would be based on actual outcomes
            base_effectiveness = 0.7
            
            # Higher effectiveness for immediate danger responses
            if alert.crisis_level == CrisisLevel.IMMEDIATE_DANGER:
                base_effectiveness += 0.2
            
            # Higher effectiveness for high confidence detections
            if alert.confidence_score > 0.8:
                base_effectiveness += 0.1
            
            # Cultural considerations improve effectiveness
            if alert.cultural_considerations:
                base_effectiveness += 0.1
            
            return min(1.0, base_effectiveness)
            
        except Exception as e:
            logger.error(f"Error simulating response effectiveness: {str(e)}")
            return 0.5
    
    async def _extract_response_lessons(self, alert: CrisisAlert, actions: List[str]) -> List[str]:
        """Extract lessons learned from crisis response"""
        try:
            lessons = []
            
            # Alert quality lessons
            if alert.confidence_score > 0.9:
                lessons.append("High confidence detection enabled rapid response")
            elif alert.confidence_score < 0.6:
                lessons.append("Low confidence required additional validation")
            
            # Response channel lessons
            if ResponseChannel.EMERGENCY_SERVICES in alert.recommended_responses:
                lessons.append("Emergency services integration critical for immediate danger")
            
            if alert.cultural_considerations:
                lessons.append("Cultural considerations important for appropriate response")
            
            # Crisis type lessons
            if CrisisType.SUICIDAL_IDEATION in alert.crisis_types:
                lessons.append("Suicide risk requires immediate professional involvement")
            
            return lessons
            
        except Exception as e:
            logger.error(f"Error extracting response lessons: {str(e)}")
            return []
    
    async def _monitoring_loop(self):
        """Monitor active crisis alerts"""
        try:
            while self.detection_active:
                try:
                    await asyncio.sleep(60)  # Check every minute
                    
                    # Monitor active alerts for escalation
                    current_time = datetime.utcnow()
                    
                    for alert_id, alert in list(self.active_alerts.items()):
                        # Check if alert is too old
                        alert_age = (current_time - alert.detected_at).total_seconds()
                        
                        if alert_age > 86400:  # 24 hours
                            # Move to resolved if no escalation
                            del self.active_alerts[alert_id]
                            logger.info(f"Crisis alert {alert_id} aged out after 24 hours")
                    
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {str(e)}")
                    await asyncio.sleep(60)
                    
        except Exception as e:
            logger.error(f"Fatal error in monitoring loop: {str(e)}")
    
    async def _response_coordinator(self):
        """Coordinate crisis responses"""
        try:
            while self.detection_active:
                try:
                    await asyncio.sleep(30)  # Check every 30 seconds
                    
                    # Check for responses needing follow-up
                    current_time = datetime.utcnow()
                    
                    for response in self.crisis_responses:
                        if (response.follow_up_scheduled and 
                            response.follow_up_timing and 
                            current_time >= response.follow_up_timing):
                            
                            await self._execute_follow_up_response(response)
                            response.follow_up_scheduled = False
                    
                except Exception as e:
                    logger.error(f"Error in response coordinator: {str(e)}")
                    await asyncio.sleep(30)
                    
        except Exception as e:
            logger.error(f"Fatal error in response coordinator: {str(e)}")
    
    async def _execute_follow_up_response(self, response: CrisisResponse):
        """Execute follow-up response"""
        try:
            logger.info(f"Executing follow-up for crisis response {response.response_id}")
            
            # In production, would implement actual follow-up logic
            # For now, just log the follow-up
            
        except Exception as e:
            logger.error(f"Error executing follow-up response: {str(e)}")
    
    # Callback management
    def add_crisis_detected_callback(self, callback: Callable):
        """Add callback for crisis detection"""
        self.crisis_detected_callbacks.append(callback)
    
    def add_intervention_executed_callback(self, callback: Callable):
        """Add callback for intervention execution"""
        self.intervention_executed_callbacks.append(callback)
    
    def add_professional_notification_callback(self, callback: Callable):
        """Add callback for professional notifications"""
        self.professional_notification_callbacks.append(callback)
    
    def add_emergency_escalation_callback(self, callback: Callable):
        """Add callback for emergency escalations"""
        self.emergency_escalation_callbacks.append(callback)
    
    # Analytics and reporting
    def get_crisis_analytics(self) -> Dict[str, Any]:
        """Get comprehensive crisis detection analytics"""
        try:
            if not self.crisis_responses:
                return {
                    'crises_detected': self.crises_detected,
                    'interventions_made': self.interventions_made,
                    'lives_potentially_saved': self.lives_potentially_saved,
                    'average_response_time_seconds': self.average_response_time_seconds,
                    'false_positive_rate': self.false_positive_rate,
                    'active_alerts': len(self.active_alerts)
                }
            
            # Calculate detailed analytics
            effectiveness_scores = [r.effectiveness_score for r in self.crisis_responses]
            
            return {
                'crises_detected': self.crises_detected,
                'interventions_made': self.interventions_made,
                'lives_potentially_saved': self.lives_potentially_saved,
                'average_response_time_seconds': self.average_response_time_seconds,
                'false_positive_rate': self.false_positive_rate,
                'active_alerts': len(self.active_alerts),
                'average_effectiveness': statistics.mean(effectiveness_scores),
                'emergency_services_contacted': sum(1 for r in self.crisis_responses if r.emergency_services_contacted),
                'professional_involvement_rate': sum(1 for r in self.crisis_responses if r.professional_contacts) / len(self.crisis_responses),
                'follow_up_completion_rate': sum(1 for r in self.crisis_responses if not r.follow_up_scheduled) / len(self.crisis_responses)
            }
            
        except Exception as e:
            logger.error(f"Error generating crisis analytics: {str(e)}")
            return {}