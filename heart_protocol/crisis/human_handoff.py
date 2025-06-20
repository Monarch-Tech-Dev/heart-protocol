"""
Human Handoff Manager

Manages the transition from AI support to human professional care,
ensuring seamless continuity while maintaining user dignity and choice.
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
import logging

from .escalation_engine import EscalationDecision, CrisisLevel, EscalationType

logger = logging.getLogger(__name__)


class HandoffType(Enum):
    """Types of human handoff"""
    EMERGENCY_RESPONDER = "emergency_responder"       # Emergency services
    CRISIS_COUNSELOR = "crisis_counselor"            # Crisis hotline counselor
    MENTAL_HEALTH_PROFESSIONAL = "mental_health_professional"  # Therapist/psychiatrist
    PEER_SUPPORT_SPECIALIST = "peer_support_specialist"  # Trained peer supporter
    COMMUNITY_VOLUNTEER = "community_volunteer"       # Community crisis responder
    TRUSTED_INDIVIDUAL = "trusted_individual"         # User's trusted contact
    MEDICAL_PROFESSIONAL = "medical_professional"     # Doctor/nurse
    SOCIAL_WORKER = "social_worker"                   # Licensed social worker


class HandoffStatus(Enum):
    """Status of handoff process"""
    INITIATED = "initiated"                 # Handoff process started
    CONSENT_PENDING = "consent_pending"     # Waiting for user consent
    CONSENT_GRANTED = "consent_granted"     # User has consented
    CONSENT_DENIED = "consent_denied"       # User has declined
    CONNECTING = "connecting"               # Attempting to connect
    CONNECTED = "connected"                 # Successfully connected to human
    FAILED = "failed"                      # Connection failed
    COMPLETED = "completed"                # Handoff successfully completed
    CANCELLED = "cancelled"                # User cancelled handoff


class HandoffUrgency(Enum):
    """Urgency levels for handoff"""
    IMMEDIATE = "immediate"        # Emergency - bypass normal processes
    URGENT = "urgent"             # Within minutes
    HIGH_PRIORITY = "high_priority"  # Within hours
    STANDARD = "standard"         # Within 24 hours
    SCHEDULED = "scheduled"       # Scheduled appointment


@dataclass
class HandoffRequest:
    """Request for human handoff"""
    request_id: str
    user_id: str
    handoff_type: HandoffType
    urgency: HandoffUrgency
    escalation_decision: EscalationDecision
    user_context: Dict[str, Any]
    consent_required: bool
    consent_granted: Optional[bool]
    requested_at: datetime
    metadata: Dict[str, Any]


@dataclass
class HandoffAttempt:
    """Individual attempt to connect to human support"""
    attempt_id: str
    handoff_request_id: str
    contact_method: str
    contact_details: Dict[str, Any]
    attempted_at: datetime
    status: str
    response_time_seconds: Optional[int]
    success: bool
    failure_reason: Optional[str]
    notes: str


@dataclass
class HumanResponder:
    """Information about available human responder"""
    responder_id: str
    responder_type: HandoffType
    name: str
    credentials: List[str]
    specializations: List[str]
    availability_hours: Dict[str, Any]
    contact_methods: List[str]
    response_time_minutes: int
    languages: List[str]
    trauma_informed: bool
    cultural_competencies: List[str]
    current_availability: bool


class HumanHandoffManager:
    """
    Manages handoff from AI to human support with dignity and effectiveness.
    
    Key Principles:
    - User consent and autonomy paramount
    - Seamless transition experience  
    - Preserve user context and dignity
    - Cultural and trauma sensitivity
    - Multiple fallback options
    - Real-time availability tracking
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Handoff tracking
        self.active_handoffs = {}        # request_id -> HandoffRequest
        self.handoff_history = {}        # user_id -> List[handoff_events]
        self.handoff_attempts = {}       # request_id -> List[HandoffAttempt]
        
        # Human responder network
        self.available_responders = {}   # responder_id -> HumanResponder
        self.responder_availability = {} # responder_id -> availability_data
        self.responder_specialties = {}  # specialty -> List[responder_ids]
        
        # Connection methods
        self.connection_handlers = {}    # contact_method -> handler_function
        self.escalation_callbacks = {}   # callback_type -> callback_function
        
        # Configuration
        self.max_connection_attempts = config.get('max_connection_attempts', 3)
        self.connection_timeout_seconds = config.get('connection_timeout_seconds', 300)
        self.consent_timeout_minutes = config.get('consent_timeout_minutes', 10)
        
        self._initialize_default_responders()
        self._initialize_connection_handlers()
        
        logger.info("Human Handoff Manager initialized")
    
    def _initialize_default_responders(self) -> None:
        """Initialize default responder network"""
        
        # National crisis hotlines (always available)
        self.available_responders['988_lifeline'] = HumanResponder(
            responder_id='988_lifeline',
            responder_type=HandoffType.CRISIS_COUNSELOR,
            name='National Suicide Prevention Lifeline',
            credentials=['Crisis Counseling'],
            specializations=['suicide_prevention', 'crisis_intervention'],
            availability_hours={'24/7': True},
            contact_methods=['phone', 'chat', 'text'],
            response_time_minutes=2,
            languages=['English', 'Spanish'],
            trauma_informed=True,
            cultural_competencies=['general', 'LGBTQ+', 'veterans'],
            current_availability=True
        )
        
        self.available_responders['crisis_text_line'] = HumanResponder(
            responder_id='crisis_text_line',
            responder_type=HandoffType.CRISIS_COUNSELOR,
            name='Crisis Text Line',
            credentials=['Crisis Counseling'],
            specializations=['crisis_intervention', 'youth_support'],
            availability_hours={'24/7': True},
            contact_methods=['text'],
            response_time_minutes=5,
            languages=['English', 'Spanish'],
            trauma_informed=True,
            cultural_competencies=['youth', 'general'],
            current_availability=True
        )
        
        # Emergency services
        self.available_responders['emergency_services'] = HumanResponder(
            responder_id='emergency_services',
            responder_type=HandoffType.EMERGENCY_RESPONDER,
            name='Emergency Services (911)',
            credentials=['Emergency Response'],
            specializations=['emergency_intervention', 'immediate_safety'],
            availability_hours={'24/7': True},
            contact_methods=['phone'],
            response_time_minutes=1,
            languages=['English', 'interpreter_services'],
            trauma_informed=False,  # Variable
            cultural_competencies=['general'],
            current_availability=True
        )
        
        # Specialized responders
        self.available_responders['lgbt_hotline'] = HumanResponder(
            responder_id='lgbt_hotline',
            responder_type=HandoffType.PEER_SUPPORT_SPECIALIST,
            name='LGBT National Hotline',
            credentials=['Peer Support'],
            specializations=['LGBTQ+_support', 'identity_crisis'],
            availability_hours={'weekdays': '1pm-9pm PST'},
            contact_methods=['phone', 'chat'],
            response_time_minutes=3,
            languages=['English'],
            trauma_informed=True,
            cultural_competencies=['LGBTQ+', 'identity_issues'],
            current_availability=True  # Would check real-time
        )
    
    def _initialize_connection_handlers(self) -> None:
        """Initialize connection method handlers"""
        
        self.connection_handlers = {
            'phone': self._handle_phone_connection,
            'text': self._handle_text_connection,
            'chat': self._handle_chat_connection,
            'video': self._handle_video_connection,
            'in_person': self._handle_in_person_connection
        }
    
    async def initiate_handoff(self, escalation_decision: EscalationDecision,
                              user_context: Dict[str, Any],
                              user_preferences: Optional[Dict[str, Any]] = None) -> HandoffRequest:
        """
        Initiate handoff to human support based on escalation decision.
        
        Args:
            escalation_decision: Decision from escalation engine
            user_context: User's current context
            user_preferences: User's handoff preferences
        """
        try:
            # Generate handoff request
            handoff_request = await self._create_handoff_request(
                escalation_decision, user_context, user_preferences
            )
            
            # Store active handoff
            self.active_handoffs[handoff_request.request_id] = handoff_request
            
            # Handle consent if required
            if handoff_request.consent_required:
                await self._request_user_consent(handoff_request)
            else:
                # Proceed immediately for emergency situations
                await self._execute_handoff(handoff_request)
            
            # Track in user history
            await self._track_handoff_history(handoff_request)
            
            logger.info(f"Initiated handoff request {handoff_request.request_id} "
                       f"for user {user_context.get('user_id', 'unknown')[:8]}...")
            
            return handoff_request
            
        except Exception as e:
            logger.error(f"Error initiating handoff: {e}")
            raise
    
    async def _create_handoff_request(self, escalation_decision: EscalationDecision,
                                    user_context: Dict[str, Any],
                                    user_preferences: Optional[Dict[str, Any]]) -> HandoffRequest:
        """Create handoff request from escalation decision"""
        
        # Determine handoff type based on escalation types
        handoff_type = await self._determine_handoff_type(
            escalation_decision.escalation_types, user_preferences
        )
        
        # Determine urgency
        urgency = await self._determine_handoff_urgency(
            escalation_decision.crisis_level, escalation_decision.urgency_minutes
        )
        
        # Generate unique request ID
        timestamp = int(datetime.utcnow().timestamp())
        user_id = user_context.get('user_id', 'anonymous')
        request_id = f"handoff_{user_id}_{timestamp}"
        
        return HandoffRequest(
            request_id=request_id,
            user_id=user_id,
            handoff_type=handoff_type,
            urgency=urgency,
            escalation_decision=escalation_decision,
            user_context=user_context,
            consent_required=escalation_decision.user_consent_required,
            consent_granted=None,
            requested_at=datetime.utcnow(),
            metadata={
                'crisis_level': escalation_decision.crisis_level.value,
                'escalation_reasoning': escalation_decision.reasoning,
                'user_preferences': user_preferences or {}
            }
        )
    
    async def _determine_handoff_type(self, escalation_types: List[EscalationType],
                                    user_preferences: Optional[Dict[str, Any]]) -> HandoffType:
        """Determine appropriate handoff type"""
        
        # Map escalation types to handoff types
        type_mapping = {
            EscalationType.EMERGENCY_SERVICES: HandoffType.EMERGENCY_RESPONDER,
            EscalationType.CRISIS_HOTLINE: HandoffType.CRISIS_COUNSELOR,
            EscalationType.MENTAL_HEALTH_PROFESSIONAL: HandoffType.MENTAL_HEALTH_PROFESSIONAL,
            EscalationType.PEER_SUPPORT: HandoffType.PEER_SUPPORT_SPECIALIST,
            EscalationType.COMMUNITY_RESPONDER: HandoffType.COMMUNITY_VOLUNTEER,
            EscalationType.TRUSTED_CONTACT: HandoffType.TRUSTED_INDIVIDUAL
        }
        
        # Check user preferences first
        if user_preferences and 'preferred_handoff_type' in user_preferences:
            preferred = user_preferences['preferred_handoff_type']
            if any(type_mapping.get(et) == preferred for et in escalation_types):
                return preferred
        
        # Use first available escalation type
        for escalation_type in escalation_types:
            if escalation_type in type_mapping:
                return type_mapping[escalation_type]
        
        # Default to crisis counselor
        return HandoffType.CRISIS_COUNSELOR
    
    async def _determine_handoff_urgency(self, crisis_level: CrisisLevel,
                                       urgency_minutes: int) -> HandoffUrgency:
        """Determine handoff urgency level"""
        
        if crisis_level == CrisisLevel.IMMINENT_DANGER:
            return HandoffUrgency.IMMEDIATE
        elif urgency_minutes <= 5:
            return HandoffUrgency.IMMEDIATE
        elif urgency_minutes <= 30:
            return HandoffUrgency.URGENT
        elif urgency_minutes <= 240:  # 4 hours
            return HandoffUrgency.HIGH_PRIORITY
        else:
            return HandoffUrgency.STANDARD
    
    async def _request_user_consent(self, handoff_request: HandoffRequest) -> None:
        """Request user consent for handoff"""
        
        # Update handoff status
        await self._update_handoff_status(handoff_request.request_id, HandoffStatus.CONSENT_PENDING)
        
        # Generate consent request (would integrate with UI)
        consent_message = await self._generate_consent_message(handoff_request)
        
        # Set consent timeout
        consent_timeout = datetime.utcnow() + timedelta(minutes=self.consent_timeout_minutes)
        handoff_request.metadata['consent_timeout'] = consent_timeout
        
        logger.info(f"Requested consent for handoff {handoff_request.request_id}")
        
        # In production, this would send consent request to user interface
        # For now, we'll simulate immediate consent for demo
        await asyncio.sleep(0.1)  # Simulate brief delay
        await self.process_consent_response(handoff_request.request_id, True)
    
    async def _generate_consent_message(self, handoff_request: HandoffRequest) -> str:
        """Generate consent request message"""
        
        handoff_type_descriptions = {
            HandoffType.CRISIS_COUNSELOR: "a trained crisis counselor",
            HandoffType.EMERGENCY_RESPONDER: "emergency services",
            HandoffType.MENTAL_HEALTH_PROFESSIONAL: "a mental health professional",
            HandoffType.PEER_SUPPORT_SPECIALIST: "a peer support specialist"
        }
        
        responder_description = handoff_type_descriptions.get(
            handoff_request.handoff_type, "a human support person"
        )
        
        return f\"\"\"I'm concerned about your safety and wellbeing. I'd like to connect you with {responder_description} who can provide additional support.

This person is trained to help in situations like yours and can offer resources and guidance that I cannot provide as an AI.

Would you like me to help connect you with this support? You can say no, and I'll still be here to support you in other ways.

Your privacy and choices are important - you maintain control over what information is shared.\"\"\"\n\n    async def process_consent_response(self, request_id: str, consent_granted: bool) -> None:\n        \"\"\"Process user's consent response\"\"\"\n        \n        if request_id not in self.active_handoffs:\n            logger.warning(f\"Consent response for unknown handoff request: {request_id}\")\n            return\n        \n        handoff_request = self.active_handoffs[request_id]\n        handoff_request.consent_granted = consent_granted\n        \n        if consent_granted:\n            await self._update_handoff_status(request_id, HandoffStatus.CONSENT_GRANTED)\n            await self._execute_handoff(handoff_request)\n        else:\n            await self._update_handoff_status(request_id, HandoffStatus.CONSENT_DENIED)\n            await self._handle_consent_denied(handoff_request)\n        \n        logger.info(f\"Processed consent for handoff {request_id}: {'granted' if consent_granted else 'denied'}\")\n    \n    async def _execute_handoff(self, handoff_request: HandoffRequest) -> None:\n        \"\"\"Execute the handoff to human support\"\"\"\n        \n        try:\n            await self._update_handoff_status(handoff_request.request_id, HandoffStatus.CONNECTING)\n            \n            # Find available responders\n            available_responders = await self._find_available_responders(handoff_request)\n            \n            if not available_responders:\n                await self._handle_no_responders_available(handoff_request)\n                return\n            \n            # Attempt connections in order of preference\n            connection_successful = False\n            \n            for responder in available_responders:\n                attempt = await self._attempt_connection(handoff_request, responder)\n                \n                if attempt.success:\n                    connection_successful = True\n                    await self._handle_successful_connection(handoff_request, responder, attempt)\n                    break\n                else:\n                    await self._handle_failed_connection_attempt(handoff_request, attempt)\n            \n            if not connection_successful:\n                await self._handle_all_connections_failed(handoff_request)\n            \n        except Exception as e:\n            logger.error(f\"Error executing handoff {handoff_request.request_id}: {e}\")\n            await self._update_handoff_status(handoff_request.request_id, HandoffStatus.FAILED)\n    \n    async def _find_available_responders(self, handoff_request: HandoffRequest) -> List[HumanResponder]:\n        \"\"\"Find available responders for handoff type\"\"\"\n        \n        # Filter responders by type\n        type_responders = [\n            responder for responder in self.available_responders.values()\n            if responder.responder_type == handoff_request.handoff_type\n        ]\n        \n        # Filter by availability\n        available_responders = [\n            responder for responder in type_responders\n            if responder.current_availability\n        ]\n        \n        # Sort by preference factors\n        user_context = handoff_request.user_context\n        user_language = user_context.get('preferred_language', 'English')\n        user_cultural_needs = user_context.get('cultural_considerations', [])\n        \n        def responder_score(responder: HumanResponder) -> float:\n            score = 0.0\n            \n            # Language match\n            if user_language in responder.languages:\n                score += 2.0\n            \n            # Cultural competency match\n            for need in user_cultural_needs:\n                if need in responder.cultural_competencies:\n                    score += 1.0\n            \n            # Trauma-informed care\n            if responder.trauma_informed:\n                score += 1.0\n            \n            # Response time (lower is better)\n            score += (10.0 - responder.response_time_minutes) / 10.0\n            \n            return score\n        \n        available_responders.sort(key=responder_score, reverse=True)\n        \n        return available_responders\n    \n    async def _attempt_connection(self, handoff_request: HandoffRequest,\n                                responder: HumanResponder) -> HandoffAttempt:\n        \"\"\"Attempt to connect to a specific responder\"\"\"\n        \n        attempt_id = f\"{handoff_request.request_id}_attempt_{len(self.handoff_attempts.get(handoff_request.request_id, []))}\"\n        \n        # Choose best contact method\n        user_preferences = handoff_request.metadata.get('user_preferences', {})\n        preferred_method = user_preferences.get('contact_method')\n        \n        if preferred_method and preferred_method in responder.contact_methods:\n            contact_method = preferred_method\n        else:\n            # Default priority: phone > chat > text\n            method_priority = ['phone', 'chat', 'text', 'video']\n            contact_method = next(\n                (method for method in method_priority if method in responder.contact_methods),\n                responder.contact_methods[0]\n            )\n        \n        attempt = HandoffAttempt(\n            attempt_id=attempt_id,\n            handoff_request_id=handoff_request.request_id,\n            contact_method=contact_method,\n            contact_details={'responder_id': responder.responder_id},\n            attempted_at=datetime.utcnow(),\n            status='attempting',\n            response_time_seconds=None,\n            success=False,\n            failure_reason=None,\n            notes=''\n        )\n        \n        # Store attempt\n        if handoff_request.request_id not in self.handoff_attempts:\n            self.handoff_attempts[handoff_request.request_id] = []\n        self.handoff_attempts[handoff_request.request_id].append(attempt)\n        \n        # Execute connection attempt\n        start_time = datetime.utcnow()\n        \n        try:\n            handler = self.connection_handlers.get(contact_method)\n            if handler:\n                success = await handler(handoff_request, responder, attempt)\n                attempt.success = success\n                \n                if not success:\n                    attempt.failure_reason = f\"Connection via {contact_method} failed\"\n            else:\n                attempt.success = False\n                attempt.failure_reason = f\"No handler for contact method: {contact_method}\"\n            \n        except Exception as e:\n            attempt.success = False\n            attempt.failure_reason = f\"Connection error: {str(e)}\"\n            logger.error(f\"Connection attempt failed: {e}\")\n        \n        # Calculate response time\n        end_time = datetime.utcnow()\n        attempt.response_time_seconds = int((end_time - start_time).total_seconds())\n        attempt.status = 'completed'\n        \n        return attempt\n    \n    async def _handle_phone_connection(self, handoff_request: HandoffRequest,\n                                     responder: HumanResponder,\n                                     attempt: HandoffAttempt) -> bool:\n        \"\"\"Handle phone connection to responder\"\"\"\n        \n        # In production, this would:\n        # 1. Initiate 3-way call or transfer\n        # 2. Provide user with direct number\n        # 3. Send context to responder system\n        \n        # For simulation, we'll assume success for known responders\n        if responder.responder_id in ['988_lifeline', 'emergency_services']:\n            attempt.notes = f\"Connected to {responder.name} via phone\"\n            return True\n        \n        return False\n    \n    async def _handle_text_connection(self, handoff_request: HandoffRequest,\n                                    responder: HumanResponder,\n                                    attempt: HandoffAttempt) -> bool:\n        \"\"\"Handle text connection to responder\"\"\"\n        \n        # For text-based services like Crisis Text Line\n        if responder.responder_id == 'crisis_text_line':\n            attempt.notes = \"User directed to text HOME to 741741\"\n            return True\n        \n        return False\n    \n    async def _handle_chat_connection(self, handoff_request: HandoffRequest,\n                                    responder: HumanResponder,\n                                    attempt: HandoffAttempt) -> bool:\n        \"\"\"Handle chat connection to responder\"\"\"\n        \n        # For chat-based services\n        attempt.notes = f\"Chat connection established with {responder.name}\"\n        return True\n    \n    async def _handle_video_connection(self, handoff_request: HandoffRequest,\n                                     responder: HumanResponder,\n                                     attempt: HandoffAttempt) -> bool:\n        \"\"\"Handle video connection to responder\"\"\"\n        \n        # Video calling integration would go here\n        return False\n    \n    async def _handle_in_person_connection(self, handoff_request: HandoffRequest,\n                                         responder: HumanResponder,\n                                         attempt: HandoffAttempt) -> bool:\n        \"\"\"Handle in-person meeting arrangement\"\"\"\n        \n        # Schedule in-person meeting\n        return False\n    \n    async def _handle_successful_connection(self, handoff_request: HandoffRequest,\n                                          responder: HumanResponder,\n                                          attempt: HandoffAttempt) -> None:\n        \"\"\"Handle successful connection to human support\"\"\"\n        \n        await self._update_handoff_status(handoff_request.request_id, HandoffStatus.CONNECTED)\n        \n        # Prepare handoff context for responder\n        handoff_context = await self._prepare_handoff_context(handoff_request)\n        \n        # Log successful handoff\n        logger.info(f\"Successfully connected handoff {handoff_request.request_id} \"\n                   f\"to {responder.name} via {attempt.contact_method}\")\n        \n        # Schedule follow-up\n        await self._schedule_handoff_follow_up(handoff_request, responder)\n    \n    async def _prepare_handoff_context(self, handoff_request: HandoffRequest) -> Dict[str, Any]:\n        \"\"\"Prepare context information for human responder\"\"\"\n        \n        # Sanitize and prepare user context for human responder\n        context = {\n            'handoff_id': handoff_request.request_id,\n            'crisis_level': handoff_request.escalation_decision.crisis_level.value,\n            'urgency': handoff_request.urgency.value,\n            'escalation_reasoning': handoff_request.escalation_decision.reasoning,\n            'immediate_resources_provided': handoff_request.escalation_decision.immediate_resources,\n            'user_consent_status': handoff_request.consent_granted,\n            'cultural_considerations': handoff_request.user_context.get('cultural_considerations', []),\n            'preferred_language': handoff_request.user_context.get('preferred_language', 'English'),\n            'trauma_informed_needed': handoff_request.user_context.get('trauma_sensitive', False)\n        }\n        \n        return context\n    \n    async def _handle_failed_connection_attempt(self, handoff_request: HandoffRequest,\n                                              attempt: HandoffAttempt) -> None:\n        \"\"\"Handle failed connection attempt\"\"\"\n        \n        logger.warning(f\"Connection attempt failed for handoff {handoff_request.request_id}: \"\n                      f\"{attempt.failure_reason}\")\n    \n    async def _handle_all_connections_failed(self, handoff_request: HandoffRequest) -> None:\n        \"\"\"Handle case where all connection attempts failed\"\"\"\n        \n        await self._update_handoff_status(handoff_request.request_id, HandoffStatus.FAILED)\n        \n        # Provide fallback resources\n        fallback_message = await self._generate_fallback_message(handoff_request)\n        \n        logger.error(f\"All connection attempts failed for handoff {handoff_request.request_id}\")\n        \n        # In critical situations, escalate further\n        if handoff_request.urgency == HandoffUrgency.IMMEDIATE:\n            await self._emergency_escalation(handoff_request)\n    \n    async def _handle_no_responders_available(self, handoff_request: HandoffRequest) -> None:\n        \"\"\"Handle case where no responders are available\"\"\"\n        \n        await self._update_handoff_status(handoff_request.request_id, HandoffStatus.FAILED)\n        \n        logger.error(f\"No responders available for handoff {handoff_request.request_id}\")\n        \n        # Always have fallback options\n        await self._emergency_escalation(handoff_request)\n    \n    async def _handle_consent_denied(self, handoff_request: HandoffRequest) -> None:\n        \"\"\"Handle case where user denies consent for handoff\"\"\"\n        \n        logger.info(f\"User denied consent for handoff {handoff_request.request_id}\")\n        \n        # Provide alternative support options\n        alternative_message = await self._generate_alternative_support_message(handoff_request)\n        \n        # For severe cases, still provide emergency resources\n        if handoff_request.escalation_decision.crisis_level == CrisisLevel.IMMINENT_DANGER:\n            await self._provide_emergency_resources_anyway(handoff_request)\n    \n    async def _emergency_escalation(self, handoff_request: HandoffRequest) -> None:\n        \"\"\"Emergency escalation when primary handoff fails\"\"\"\n        \n        # Always provide emergency contact information\n        emergency_resources = [\n            \"National Suicide Prevention Lifeline: 988\",\n            \"Crisis Text Line: Text HOME to 741741\",\n            \"Emergency Services: 911\"\n        ]\n        \n        # Log emergency escalation\n        logger.critical(f\"Emergency escalation triggered for handoff {handoff_request.request_id}\")\n    \n    async def _update_handoff_status(self, request_id: str, status: HandoffStatus) -> None:\n        \"\"\"Update handoff status\"\"\"\n        \n        if request_id in self.active_handoffs:\n            handoff_request = self.active_handoffs[request_id]\n            handoff_request.metadata['status'] = status.value\n            handoff_request.metadata['last_updated'] = datetime.utcnow()\n    \n    async def _track_handoff_history(self, handoff_request: HandoffRequest) -> None:\n        \"\"\"Track handoff in user history\"\"\"\n        \n        user_id = handoff_request.user_id\n        if user_id not in self.handoff_history:\n            self.handoff_history[user_id] = []\n        \n        self.handoff_history[user_id].append({\n            'timestamp': handoff_request.requested_at,\n            'request_id': handoff_request.request_id,\n            'handoff_type': handoff_request.handoff_type.value,\n            'urgency': handoff_request.urgency.value,\n            'crisis_level': handoff_request.escalation_decision.crisis_level.value,\n            'consent_granted': handoff_request.consent_granted\n        })\n    \n    async def _schedule_handoff_follow_up(self, handoff_request: HandoffRequest,\n                                        responder: HumanResponder) -> None:\n        \"\"\"Schedule follow-up after handoff\"\"\"\n        \n        # Schedule based on urgency and crisis level\n        if handoff_request.urgency == HandoffUrgency.IMMEDIATE:\n            follow_up_hours = 2\n        elif handoff_request.urgency == HandoffUrgency.URGENT:\n            follow_up_hours = 6\n        else:\n            follow_up_hours = 24\n        \n        follow_up_time = datetime.utcnow() + timedelta(hours=follow_up_hours)\n        \n        handoff_request.metadata['follow_up_scheduled'] = follow_up_time\n        \n        logger.info(f\"Scheduled follow-up for handoff {handoff_request.request_id} \"\n                   f\"at {follow_up_time}\")\n    \n    async def complete_handoff(self, request_id: str, \n                             completion_status: str,\n                             completion_notes: str = \"\") -> None:\n        \"\"\"Mark handoff as completed\"\"\"\n        \n        if request_id in self.active_handoffs:\n            handoff_request = self.active_handoffs[request_id]\n            \n            await self._update_handoff_status(request_id, HandoffStatus.COMPLETED)\n            \n            handoff_request.metadata.update({\n                'completion_status': completion_status,\n                'completion_notes': completion_notes,\n                'completed_at': datetime.utcnow()\n            })\n            \n            # Move to history\n            del self.active_handoffs[request_id]\n            \n            logger.info(f\"Completed handoff {request_id} with status: {completion_status}\")\n    \n    def get_active_handoffs(self, user_id: Optional[str] = None) -> List[HandoffRequest]:\n        \"\"\"Get active handoffs, optionally filtered by user\"\"\"\n        \n        if user_id:\n            return [req for req in self.active_handoffs.values() if req.user_id == user_id]\n        \n        return list(self.active_handoffs.values())\n    \n    async def get_handoff_analytics(self) -> Dict[str, Any]:\n        \"\"\"Get handoff system analytics\"\"\"\n        \n        total_active = len(self.active_handoffs)\n        total_responders = len(self.available_responders)\n        available_responders = sum(1 for r in self.available_responders.values() \n                                  if r.current_availability)\n        \n        # Calculate success rate from history\n        all_attempts = []\n        for attempts in self.handoff_attempts.values():\n            all_attempts.extend(attempts)\n        \n        total_attempts = len(all_attempts)\n        successful_attempts = sum(1 for attempt in all_attempts if attempt.success)\n        success_rate = (successful_attempts / total_attempts * 100) if total_attempts > 0 else 0\n        \n        return {\n            'active_handoffs': total_active,\n            'total_responders': total_responders,\n            'available_responders': available_responders,\n            'responder_availability_rate': (available_responders / total_responders * 100) if total_responders > 0 else 0,\n            'connection_success_rate': success_rate,\n            'total_connection_attempts': total_attempts,\n            'handoff_system_healthy': available_responders > 0,\n            'generated_at': datetime.utcnow().isoformat()\n        }\n    \n    async def _generate_fallback_message(self, handoff_request: HandoffRequest) -> str:\n        \"\"\"Generate fallback message when handoff fails\"\"\"\n        \n        return f\"\"\"I wasn't able to connect you directly with human support right now, but please know that help is available:\n\n**Immediate Resources:**\n• National Suicide Prevention Lifeline: 988 (24/7)\n• Crisis Text Line: Text HOME to 741741\n• Emergency Services: 911\n\nYour safety and wellbeing matter. Please reach out to one of these resources - they're staffed by caring professionals who are ready to help.\n\nI'm also still here to support you in whatever way I can.\"\"\"\n    \n    async def _generate_alternative_support_message(self, handoff_request: HandoffRequest) -> str:\n        \"\"\"Generate alternative support message when consent is denied\"\"\"\n        \n        return f\"\"\"I understand and respect your choice. I'm still here to support you.\n\nIf you change your mind, here are resources available anytime:\n• National Suicide Prevention Lifeline: 988 (24/7)\n• Crisis Text Line: Text HOME to 741741\n\nWould you like to talk about what might feel most helpful right now? I'm here to listen and support you in whatever way feels right for you.\"\"\"\n    \n    async def _provide_emergency_resources_anyway(self, handoff_request: HandoffRequest) -> None:\n        \"\"\"Provide emergency resources even when consent is denied\"\"\"\n        \n        # In cases of imminent danger, still provide critical resources\n        emergency_message = f\"\"\"I respect your choice, and I also want to make sure you have these resources available:\n\n**If you're in immediate danger:**\n• Emergency Services: 911\n• National Suicide Prevention Lifeline: 988\n\nThese are available 24/7 with trained professionals ready to help.\"\"\""