"""
Crisis Escalation Engine

Detects crisis situations and orchestrates appropriate escalation to human
support while maintaining user autonomy and dignity throughout the process.
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
import logging

from ..core.base import CareLevel, Post, CareAssessment
from ..bot.caring_intelligence import CaringAssessment, CareNeeds

logger = logging.getLogger(__name__)


class CrisisLevel(Enum):
    """Levels of crisis severity"""
    IMMINENT_DANGER = "imminent_danger"        # Immediate threat to life/safety
    SEVERE_DISTRESS = "severe_distress"        # High risk, urgent intervention needed  
    ACUTE_CRISIS = "acute_crisis"              # Crisis state, professional help recommended
    ELEVATED_CONCERN = "elevated_concern"      # Concerning signs, enhanced monitoring
    WATCHFUL_MONITORING = "watchful_monitoring" # Increased attention needed


class EscalationType(Enum):
    """Types of crisis escalation"""
    EMERGENCY_SERVICES = "emergency_services"    # 911/emergency services
    CRISIS_HOTLINE = "crisis_hotline"           # Crisis counseling hotline
    MENTAL_HEALTH_PROFESSIONAL = "mental_health_professional"  # Therapist/psychiatrist
    TRUSTED_CONTACT = "trusted_contact"         # User's emergency contact
    COMMUNITY_RESPONDER = "community_responder" # Trained community volunteer
    PEER_SUPPORT = "peer_support"               # Peer support specialist


@dataclass
class CrisisIndicators:
    """Indicators that suggest crisis state"""
    suicide_ideation: bool = False
    self_harm_intent: bool = False  
    immediate_danger: bool = False
    severe_impairment: bool = False
    substance_crisis: bool = False
    psychotic_symptoms: bool = False
    domestic_violence: bool = False
    child_safety_concern: bool = False
    elder_abuse_concern: bool = False
    medical_emergency: bool = False


@dataclass
class EscalationDecision:
    """Decision about crisis escalation"""
    escalation_needed: bool
    crisis_level: CrisisLevel
    escalation_types: List[EscalationType]
    urgency_minutes: int  # How urgently intervention is needed
    safety_plan_sufficient: bool
    user_consent_required: bool
    immediate_resources: List[str]
    reasoning: str
    confidence_score: float


class EscalationEngine:
    """
    Engine for detecting crisis situations and determining appropriate escalation.
    
    Principles:
    - User safety is paramount
    - Maintain dignity and autonomy
    - Least restrictive intervention
    - Cultural sensitivity
    - Trauma-informed approach
    - Privacy protection
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Crisis detection patterns
        self.crisis_keywords = self._initialize_crisis_keywords()
        self.escalation_thresholds = self._initialize_escalation_thresholds()
        self.intervention_protocols = self._initialize_intervention_protocols()
        
        # Crisis state tracking
        self.active_crisis_sessions = {}  # user_id -> crisis_session_data
        self.escalation_history = {}      # user_id -> List[escalation_events]
        self.safety_plans = {}            # user_id -> safety_plan_data
        
        # System integrations
        self.emergency_contacts = {}      # Available emergency resources
        self.crisis_hotlines = self._initialize_crisis_hotlines()
        self.professional_network = {}   # Available mental health professionals
        
        logger.info("Crisis Escalation Engine initialized")
    
    def _initialize_crisis_keywords(self) -> Dict[str, Dict[str, Any]]:
        """Initialize crisis detection keywords and patterns"""
        
        return {
            'suicide_indicators': {
                'explicit': [
                    'kill myself', 'end it all', 'suicide', 'want to die',
                    'better off dead', 'end my life', 'no point living'
                ],
                'implicit': [
                    'can\'t go on', 'nothing to live for', 'tired of living',
                    'escape this pain', 'permanent solution', 'final answer'
                ],
                'weight': 3.0  # High weight for suicide indicators
            },
            
            'self_harm_indicators': {
                'explicit': [
                    'hurt myself', 'cut myself', 'self harm', 'self injury',
                    'want to cut', 'need to hurt', 'punish myself'
                ],
                'implicit': [
                    'deserve pain', 'need to feel something', 'release the pressure',
                    'only way to cope', 'makes me feel better'
                ],
                'weight': 2.5
            },
            
            'immediate_danger_indicators': {
                'explicit': [
                    'emergency', 'urgent help', 'immediate danger', 'crisis',
                    'can\'t be safe', 'going to hurt', 'about to'
                ],
                'implicit': [
                    'losing control', 'can\'t stop myself', 'happening now',
                    'right now', 'very soon', 'tonight'
                ],
                'weight': 3.0
            },
            
            'severe_distress_indicators': {
                'explicit': [
                    'breakdown', 'falling apart', 'can\'t cope', 'overwhelmed',
                    'drowning', 'suffocating', 'crushing me'
                ],
                'implicit': [
                    'too much', 'can\'t handle', 'breaking point',
                    'lost control', 'spiraling', 'crashing down'
                ],
                'weight': 2.0
            },
            
            'substance_crisis_indicators': {
                'explicit': [
                    'overdose', 'too much drugs', 'alcohol poisoning',
                    'can\'t stop using', 'relapsed badly'
                ],
                'implicit': [
                    'using to cope', 'numbing the pain', 'only thing that helps',
                    'drinking too much', 'high all the time'
                ],
                'weight': 2.5
            }
        }
    
    def _initialize_escalation_thresholds(self) -> Dict[CrisisLevel, Dict[str, Any]]:
        """Initialize thresholds for different crisis levels"""
        
        return {
            CrisisLevel.IMMINENT_DANGER: {
                'crisis_score_threshold': 8.0,
                'keyword_matches_min': 2,
                'urgency_minutes': 5,
                'requires_immediate_intervention': True,
                'bypass_user_consent': True  # In extreme danger
            },
            
            CrisisLevel.SEVERE_DISTRESS: {
                'crisis_score_threshold': 6.5,
                'keyword_matches_min': 2,
                'urgency_minutes': 30,
                'requires_immediate_intervention': True,
                'bypass_user_consent': False
            },
            
            CrisisLevel.ACUTE_CRISIS: {
                'crisis_score_threshold': 5.0,
                'keyword_matches_min': 1,
                'urgency_minutes': 120,
                'requires_immediate_intervention': False,
                'bypass_user_consent': False
            },
            
            CrisisLevel.ELEVATED_CONCERN: {
                'crisis_score_threshold': 3.5,
                'keyword_matches_min': 1,
                'urgency_minutes': 360,  # 6 hours
                'requires_immediate_intervention': False,
                'bypass_user_consent': False
            },
            
            CrisisLevel.WATCHFUL_MONITORING: {
                'crisis_score_threshold': 2.0,
                'keyword_matches_min': 0,
                'urgency_minutes': 1440,  # 24 hours
                'requires_immediate_intervention': False,
                'bypass_user_consent': False
            }
        }
    
    def _initialize_intervention_protocols(self) -> Dict[CrisisLevel, List[EscalationType]]:
        """Initialize intervention protocols for each crisis level"""
        
        return {
            CrisisLevel.IMMINENT_DANGER: [
                EscalationType.EMERGENCY_SERVICES,
                EscalationType.CRISIS_HOTLINE,
                EscalationType.TRUSTED_CONTACT
            ],
            
            CrisisLevel.SEVERE_DISTRESS: [
                EscalationType.CRISIS_HOTLINE,
                EscalationType.MENTAL_HEALTH_PROFESSIONAL,
                EscalationType.TRUSTED_CONTACT
            ],
            
            CrisisLevel.ACUTE_CRISIS: [
                EscalationType.CRISIS_HOTLINE,
                EscalationType.MENTAL_HEALTH_PROFESSIONAL,
                EscalationType.COMMUNITY_RESPONDER
            ],
            
            CrisisLevel.ELEVATED_CONCERN: [
                EscalationType.MENTAL_HEALTH_PROFESSIONAL,
                EscalationType.PEER_SUPPORT,
                EscalationType.COMMUNITY_RESPONDER
            ],
            
            CrisisLevel.WATCHFUL_MONITORING: [
                EscalationType.PEER_SUPPORT,
                EscalationType.COMMUNITY_RESPONDER
            ]
        }
    
    def _initialize_crisis_hotlines(self) -> Dict[str, Dict[str, Any]]:
        """Initialize crisis hotline information"""
        
        return {
            'national_suicide_prevention': {
                'name': 'National Suicide Prevention Lifeline',
                'number': '988',
                'available': '24/7',
                'languages': ['English', 'Spanish'],
                'text_option': True,
                'chat_option': True,
                'specializations': ['suicide_prevention', 'crisis_counseling']
            },
            
            'crisis_text_line': {
                'name': 'Crisis Text Line',
                'number': 'Text HOME to 741741',
                'available': '24/7',
                'languages': ['English', 'Spanish'],
                'text_option': True,
                'chat_option': False,
                'specializations': ['crisis_counseling', 'youth_support']
            },
            
            'lgbt_hotline': {
                'name': 'LGBT National Hotline',
                'number': '1-888-843-4564',
                'available': 'Mon-Fri 1pm-9pm PST',
                'languages': ['English'],
                'text_option': False,
                'chat_option': True,
                'specializations': ['lgbt_support', 'identity_crisis']
            },
            
            'domestic_violence_hotline': {
                'name': 'National Domestic Violence Hotline',
                'number': '1-800-799-7233',
                'available': '24/7',
                'languages': ['English', 'Spanish', 'over 200 others'],
                'text_option': True,
                'chat_option': True,
                'specializations': ['domestic_violence', 'safety_planning']
            }
        }
    
    async def assess_crisis_level(self, user_input: str, 
                                 user_context: Dict[str, Any],
                                 caring_assessment: CaringAssessment) -> EscalationDecision:
        """
        Assess if user input indicates crisis and determine escalation needs.
        
        Args:
            user_input: What the user said/wrote
            user_context: User's context and history
            caring_assessment: Assessment from caring intelligence
        """
        try:
            # Detect crisis indicators
            crisis_indicators = await self._detect_crisis_indicators(user_input, user_context)
            
            # Calculate crisis score
            crisis_score = await self._calculate_crisis_score(
                user_input, user_context, crisis_indicators, caring_assessment
            )
            
            # Determine crisis level
            crisis_level = await self._determine_crisis_level(crisis_score, crisis_indicators)
            
            # Check for escalation needs
            escalation_needed = await self._should_escalate(crisis_level, user_context)
            
            # Determine escalation types
            escalation_types = await self._determine_escalation_types(
                crisis_level, crisis_indicators, user_context
            )
            
            # Calculate urgency
            urgency_minutes = await self._calculate_urgency_minutes(crisis_level, crisis_indicators)
            
            # Assess if safety plan is sufficient
            safety_plan_sufficient = await self._assess_safety_plan_sufficiency(
                crisis_level, user_context
            )
            
            # Determine if user consent is required
            user_consent_required = await self._assess_consent_requirements(
                crisis_level, crisis_indicators
            )
            
            # Generate immediate resources
            immediate_resources = await self._generate_immediate_resources(
                crisis_level, crisis_indicators, user_context
            )
            
            # Generate reasoning
            reasoning = await self._generate_escalation_reasoning(
                crisis_score, crisis_level, crisis_indicators
            )
            
            decision = EscalationDecision(
                escalation_needed=escalation_needed,
                crisis_level=crisis_level,
                escalation_types=escalation_types,
                urgency_minutes=urgency_minutes,
                safety_plan_sufficient=safety_plan_sufficient,
                user_consent_required=user_consent_required,
                immediate_resources=immediate_resources,
                reasoning=reasoning,
                confidence_score=min(crisis_score / 10.0, 1.0)
            )
            
            # Track crisis session if escalation needed
            if escalation_needed:
                await self._initiate_crisis_session(user_context.get('user_id'), decision)
            
            logger.debug(f"Crisis assessment: level={crisis_level.value}, "
                        f"score={crisis_score:.2f}, escalation={escalation_needed}")
            
            return decision
            
        except Exception as e:
            logger.error(f"Error assessing crisis level: {e}")
            return await self._get_default_escalation_decision()
    
    async def _detect_crisis_indicators(self, user_input: str, 
                                      user_context: Dict[str, Any]) -> CrisisIndicators:
        """Detect specific crisis indicators in user input"""
        
        user_input_lower = user_input.lower()
        indicators = CrisisIndicators()
        
        # Check for suicide indicators
        suicide_keywords = self.crisis_keywords['suicide_indicators']
        if any(keyword in user_input_lower for keyword in 
               suicide_keywords['explicit'] + suicide_keywords['implicit']):
            indicators.suicide_ideation = True
        
        # Check for self-harm indicators  
        self_harm_keywords = self.crisis_keywords['self_harm_indicators']
        if any(keyword in user_input_lower for keyword in
               self_harm_keywords['explicit'] + self_harm_keywords['implicit']):
            indicators.self_harm_intent = True
        
        # Check for immediate danger indicators
        danger_keywords = self.crisis_keywords['immediate_danger_indicators']
        if any(keyword in user_input_lower for keyword in
               danger_keywords['explicit'] + danger_keywords['implicit']):
            indicators.immediate_danger = True
        
        # Check for substance crisis indicators
        substance_keywords = self.crisis_keywords['substance_crisis_indicators']
        if any(keyword in user_input_lower for keyword in
               substance_keywords['explicit'] + substance_keywords['implicit']):
            indicators.substance_crisis = True
        
        # Check context for additional indicators
        if user_context.get('recent_trauma', False):
            indicators.severe_impairment = True
        
        if user_context.get('domestic_violence_risk', False):
            indicators.domestic_violence = True
        
        return indicators
    
    async def _calculate_crisis_score(self, user_input: str,
                                    user_context: Dict[str, Any],
                                    crisis_indicators: CrisisIndicators,
                                    caring_assessment: CaringAssessment) -> float:
        """Calculate numerical crisis score"""
        
        score = 0.0
        user_input_lower = user_input.lower()
        
        # Score based on keyword matches
        for category, keywords in self.crisis_keywords.items():
            weight = keywords['weight']
            matches = 0
            
            for keyword in keywords['explicit']:
                if keyword in user_input_lower:
                    matches += 1
            
            for keyword in keywords['implicit']:
                if keyword in user_input_lower:
                    matches += 0.7  # Lower weight for implicit
            
            score += matches * weight
        
        # Adjust based on crisis indicators
        if crisis_indicators.suicide_ideation:
            score += 4.0
        if crisis_indicators.self_harm_intent:
            score += 3.0
        if crisis_indicators.immediate_danger:
            score += 4.0
        if crisis_indicators.substance_crisis:
            score += 2.5
        
        # Adjust based on caring assessment
        if caring_assessment.safety_level == CareLevel.CRISIS:
            score += 3.0
        elif caring_assessment.safety_level == CareLevel.HIGH:
            score += 2.0
        
        if caring_assessment.urgency_level > 0.8:
            score += 2.0
        
        # Context adjustments
        if user_context.get('previous_crisis_episode', False):
            score += 1.0
        
        if user_context.get('social_isolation', False):
            score += 0.5
        
        if user_context.get('recent_loss', False):
            score += 1.0
        
        return score
    
    async def _determine_crisis_level(self, crisis_score: float,
                                    crisis_indicators: CrisisIndicators) -> CrisisLevel:
        """Determine crisis level based on score and indicators"""
        
        # Check for imminent danger indicators first
        if (crisis_indicators.immediate_danger and crisis_indicators.suicide_ideation) or \
           (crisis_indicators.immediate_danger and crisis_indicators.self_harm_intent):
            return CrisisLevel.IMMINENT_DANGER
        
        # Check thresholds
        for level, threshold_data in self.escalation_thresholds.items():
            if crisis_score >= threshold_data['crisis_score_threshold']:
                return level
        
        return CrisisLevel.WATCHFUL_MONITORING
    
    async def _should_escalate(self, crisis_level: CrisisLevel,
                             user_context: Dict[str, Any]) -> bool:
        """Determine if escalation is needed"""
        
        # Always escalate for severe levels
        if crisis_level in [CrisisLevel.IMMINENT_DANGER, CrisisLevel.SEVERE_DISTRESS]:
            return True
        
        # For moderate levels, consider user preferences and context
        if crisis_level == CrisisLevel.ACUTE_CRISIS:
            # Respect user's escalation preferences if available
            user_preference = user_context.get('escalation_preference', 'allow')
            if user_preference == 'never':
                return False
            return True
        
        # For lower levels, more selective escalation
        if crisis_level == CrisisLevel.ELEVATED_CONCERN:
            # Only escalate if user has opted in or shows concerning patterns
            if user_context.get('escalation_preference') == 'proactive':
                return True
            if user_context.get('concerning_pattern_detected', False):
                return True
            return False
        
        return False
    
    async def _determine_escalation_types(self, crisis_level: CrisisLevel,
                                        crisis_indicators: CrisisIndicators,
                                        user_context: Dict[str, Any]) -> List[EscalationType]:
        """Determine appropriate escalation types"""
        
        # Start with protocol defaults
        escalation_types = self.intervention_protocols.get(crisis_level, [])
        
        # Customize based on indicators and context
        if crisis_indicators.immediate_danger:
            if EscalationType.EMERGENCY_SERVICES not in escalation_types:
                escalation_types.insert(0, EscalationType.EMERGENCY_SERVICES)
        
        if crisis_indicators.domestic_violence:
            escalation_types.append(EscalationType.TRUSTED_CONTACT)
        
        # Consider user preferences
        preferred_escalation = user_context.get('preferred_escalation_type')
        if preferred_escalation and preferred_escalation in escalation_types:
            # Move preferred type to front
            escalation_types.remove(preferred_escalation)
            escalation_types.insert(0, preferred_escalation)
        
        # Remove types user has opted out of
        blocked_types = user_context.get('blocked_escalation_types', [])
        escalation_types = [t for t in escalation_types if t not in blocked_types]
        
        return escalation_types
    
    async def _calculate_urgency_minutes(self, crisis_level: CrisisLevel,
                                       crisis_indicators: CrisisIndicators) -> int:
        """Calculate how urgently intervention is needed"""
        
        base_urgency = self.escalation_thresholds[crisis_level]['urgency_minutes']
        
        # Adjust based on specific indicators
        if crisis_indicators.immediate_danger:
            base_urgency = min(base_urgency, 5)  # Maximum 5 minutes
        
        if crisis_indicators.suicide_ideation and crisis_indicators.self_harm_intent:
            base_urgency = min(base_urgency, 15)  # Maximum 15 minutes
        
        return base_urgency
    
    async def _assess_safety_plan_sufficiency(self, crisis_level: CrisisLevel,
                                            user_context: Dict[str, Any]) -> bool:
        """Assess if existing safety plan is sufficient"""
        
        user_id = user_context.get('user_id')
        if not user_id or user_id not in self.safety_plans:
            return False
        
        safety_plan = self.safety_plans[user_id]
        
        # Safety plan is insufficient for severe crisis levels
        if crisis_level in [CrisisLevel.IMMINENT_DANGER, CrisisLevel.SEVERE_DISTRESS]:
            return False
        
        # Check if plan is recent and comprehensive
        plan_age = datetime.utcnow() - safety_plan.get('created_at', datetime.min)
        if plan_age > timedelta(days=30):  # Plan is outdated
            return False
        
        required_elements = ['crisis_contacts', 'coping_strategies', 'warning_signs']
        if not all(element in safety_plan for element in required_elements):
            return False
        
        return True
    
    async def _assess_consent_requirements(self, crisis_level: CrisisLevel,
                                         crisis_indicators: CrisisIndicators) -> bool:
        """Determine if user consent is required for escalation"""
        
        # In imminent danger, bypass consent for safety
        if crisis_level == CrisisLevel.IMMINENT_DANGER:
            return False
        
        # For all other levels, require consent
        return True
    
    async def _generate_immediate_resources(self, crisis_level: CrisisLevel,
                                          crisis_indicators: CrisisIndicators,
                                          user_context: Dict[str, Any]) -> List[str]:
        """Generate immediate resources to provide to user"""
        
        resources = []
        
        # Always include primary crisis hotline
        resources.append("National Suicide Prevention Lifeline: 988 (24/7)")
        
        # Add specific resources based on indicators
        if crisis_indicators.suicide_ideation or crisis_indicators.self_harm_intent:
            resources.extend([
                "Crisis Text Line: Text HOME to 741741",
                "If you're in immediate danger, call 911"
            ])
        
        if crisis_indicators.domestic_violence:
            resources.append("National Domestic Violence Hotline: 1-800-799-7233")
        
        if crisis_indicators.substance_crisis:
            resources.append("SAMHSA National Helpline: 1-800-662-4357")
        
        # Add user-specific resources if available
        user_preferred_resources = user_context.get('preferred_crisis_resources', [])
        resources.extend(user_preferred_resources)
        
        return resources
    
    async def _generate_escalation_reasoning(self, crisis_score: float,
                                           crisis_level: CrisisLevel,
                                           crisis_indicators: CrisisIndicators) -> str:
        """Generate reasoning for escalation decision"""
        
        reasons = []
        
        if crisis_indicators.suicide_ideation:
            reasons.append("suicide ideation detected")
        
        if crisis_indicators.self_harm_intent:
            reasons.append("self-harm intent indicated")
        
        if crisis_indicators.immediate_danger:
            reasons.append("immediate danger signals present")
        
        if crisis_score > 8.0:
            reasons.append(f"high crisis score ({crisis_score:.1f})")
        
        if not reasons:
            reasons.append(f"crisis level assessment: {crisis_level.value}")
        
        return f"Escalation recommended due to: {', '.join(reasons)}"
    
    async def _initiate_crisis_session(self, user_id: str, 
                                     decision: EscalationDecision) -> None:
        """Initiate crisis session tracking"""
        
        if not user_id:
            return
        
        session_data = {
            'session_id': f"crisis_{user_id}_{int(datetime.utcnow().timestamp())}",
            'user_id': user_id,
            'crisis_level': decision.crisis_level,
            'escalation_types': decision.escalation_types,
            'initiated_at': datetime.utcnow(),
            'urgency_minutes': decision.urgency_minutes,
            'status': 'active',
            'escalation_attempts': [],
            'resolution_status': None
        }
        
        self.active_crisis_sessions[user_id] = session_data
        
        # Track in escalation history
        if user_id not in self.escalation_history:
            self.escalation_history[user_id] = []
        
        self.escalation_history[user_id].append({
            'timestamp': datetime.utcnow(),
            'crisis_level': decision.crisis_level.value,
            'escalation_types': [t.value for t in decision.escalation_types],
            'reasoning': decision.reasoning
        })
        
        logger.info(f"Initiated crisis session for user {user_id[:8]}... "
                   f"at level {decision.crisis_level.value}")
    
    async def _get_default_escalation_decision(self) -> EscalationDecision:
        """Get default escalation decision when assessment fails"""
        
        return EscalationDecision(
            escalation_needed=True,  # Err on side of caution
            crisis_level=CrisisLevel.ELEVATED_CONCERN,
            escalation_types=[EscalationType.CRISIS_HOTLINE],
            urgency_minutes=120,
            safety_plan_sufficient=False,
            user_consent_required=True,
            immediate_resources=["National Suicide Prevention Lifeline: 988"],
            reasoning="Default escalation due to assessment error - safety prioritized",
            confidence_score=0.5
        )
    
    async def update_crisis_session(self, user_id: str, 
                                  update_data: Dict[str, Any]) -> None:
        """Update active crisis session"""
        
        if user_id in self.active_crisis_sessions:
            self.active_crisis_sessions[user_id].update(update_data)
            self.active_crisis_sessions[user_id]['last_updated'] = datetime.utcnow()
    
    async def resolve_crisis_session(self, user_id: str, 
                                   resolution_status: str,
                                   resolution_notes: str = "") -> None:
        """Mark crisis session as resolved"""
        
        if user_id in self.active_crisis_sessions:
            session = self.active_crisis_sessions[user_id]
            session.update({
                'status': 'resolved',
                'resolution_status': resolution_status,
                'resolution_notes': resolution_notes,
                'resolved_at': datetime.utcnow()
            })
            
            # Move to history and remove from active
            # (In production, would move to database)
            del self.active_crisis_sessions[user_id]
            
            logger.info(f"Resolved crisis session for user {user_id[:8]}... "
                       f"with status: {resolution_status}")
    
    def get_crisis_session(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get active crisis session for user"""
        return self.active_crisis_sessions.get(user_id)
    
    def has_active_crisis_session(self, user_id: str) -> bool:
        """Check if user has active crisis session"""
        return user_id in self.active_crisis_sessions
    
    async def get_crisis_analytics(self) -> Dict[str, Any]:
        """Get crisis system analytics"""
        
        total_active_sessions = len(self.active_crisis_sessions)
        
        # Calculate crisis level distribution
        level_distribution = {}
        for session in self.active_crisis_sessions.values():
            level = session['crisis_level'].value
            level_distribution[level] = level_distribution.get(level, 0) + 1
        
        # Calculate historical escalation rate
        total_historical = sum(len(history) for history in self.escalation_history.values())
        
        return {
            'active_crisis_sessions': total_active_sessions,
            'crisis_level_distribution': level_distribution,
            'total_users_with_escalation_history': len(self.escalation_history),
            'total_historical_escalations': total_historical,
            'crisis_detection_active': True,
            'escalation_protocols_healthy': len(self.intervention_protocols) > 0,
            'generated_at': datetime.utcnow().isoformat()
        }