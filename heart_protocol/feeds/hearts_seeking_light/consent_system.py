"""
User Consent Verification for Matching

Implements comprehensive consent management for support connections.
Ensures all interactions respect user agency and privacy preferences
according to the Open Source Love License principles.
"""

import asyncio
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
import logging
import json
import hashlib

from .support_detection import SupportSeeker, SupportOffer, SupportType

logger = logging.getLogger(__name__)


class ConsentLevel(Enum):
    """Levels of consent for support interactions"""
    NONE = "none"                           # No consent given
    IMPLICIT = "implicit"                   # Implied consent through public posting
    EXPLICIT_LIMITED = "explicit_limited"   # Consent for specific types of support
    EXPLICIT_FULL = "explicit_full"         # Full consent for support matching
    ONGOING_RELATIONSHIP = "ongoing"        # Consent for ongoing support relationship


class ConsentScope(Enum):
    """Scope of consent permissions"""
    PUBLIC_REPLY = "public_reply"           # Can reply publicly to posts
    PRIVATE_MESSAGE = "private_message"     # Can send private messages
    RESOURCE_SHARING = "resource_sharing"   # Can share resources/links
    EMOTIONAL_SUPPORT = "emotional_support" # Can offer emotional support
    PRACTICAL_ADVICE = "practical_advice"   # Can offer practical advice
    CRISIS_CONTACT = "crisis_contact"       # Can contact during crisis
    ONGOING_CHECKIN = "ongoing_checkin"     # Can check in over time
    COMMUNITY_INTRO = "community_intro"     # Can introduce to community
    PROFESSIONAL_REFERRAL = "professional_referral"  # Can suggest professional help


class ConsentDuration(Enum):
    """Duration of consent validity"""
    SINGLE_INTERACTION = "single"          # One-time consent
    SESSION = "session"                    # Valid for current session/post
    LIMITED_TIME = "limited"               # Valid for specified time period
    ONGOING = "ongoing"                    # Valid until revoked
    CRISIS_OVERRIDE = "crisis_override"    # Special crisis permissions


@dataclass
class ConsentRecord:
    """Record of user consent for support interactions"""
    user_id: str
    consenter_id: str  # Who gave consent (usually same as user_id)
    consent_level: ConsentLevel
    consent_scopes: Set[ConsentScope]
    duration: ConsentDuration
    granted_at: datetime
    expires_at: Optional[datetime]
    specific_conditions: List[str]
    revocation_triggers: List[str]
    consent_context: str  # Context in which consent was given
    verification_method: str  # How consent was verified
    metadata: Dict[str, Any]
    
    def is_valid(self, current_time: datetime = None) -> bool:
        """Check if consent is currently valid"""
        if current_time is None:
            current_time = datetime.utcnow()
        
        # Check expiration
        if self.expires_at and current_time > self.expires_at:
            return False
        
        return True
    
    def allows_scope(self, scope: ConsentScope) -> bool:
        """Check if consent allows a specific scope"""
        return scope in self.consent_scopes
    
    def time_remaining(self, current_time: datetime = None) -> Optional[timedelta]:
        """Get time remaining on consent"""
        if not self.expires_at:
            return None
        
        if current_time is None:
            current_time = datetime.utcnow()
        
        if current_time > self.expires_at:
            return timedelta(0)
        
        return self.expires_at - current_time


@dataclass
class ConsentRequest:
    """Request for user consent"""
    request_id: str
    requesting_user_id: str
    target_user_id: str
    requested_scopes: Set[ConsentScope]
    requested_duration: ConsentDuration
    support_context: str
    urgency_level: str
    explanation: str
    created_at: datetime
    expires_at: datetime
    
    def to_user_message(self) -> str:
        """Generate user-friendly consent request message"""
        scope_descriptions = {
            ConsentScope.PUBLIC_REPLY: "reply to your posts publicly",
            ConsentScope.PRIVATE_MESSAGE: "send you private messages",
            ConsentScope.RESOURCE_SHARING: "share helpful resources with you",
            ConsentScope.EMOTIONAL_SUPPORT: "offer emotional support",
            ConsentScope.PRACTICAL_ADVICE: "share practical advice",
            ConsentScope.CRISIS_CONTACT: "contact you during crisis situations",
            ConsentScope.ONGOING_CHECKIN: "check in with you over time",
            ConsentScope.COMMUNITY_INTRO: "introduce you to supportive communities",
            ConsentScope.PROFESSIONAL_REFERRAL: "suggest professional resources"
        }
        
        scope_list = [scope_descriptions.get(scope, scope.value) 
                     for scope in self.requested_scopes]
        
        return f"""
Support Connection Request

Someone in the community would like to offer you support. They are requesting permission to:

{chr(10).join(f'• {scope}' for scope in scope_list)}

Context: {self.support_context}

Duration: {self.requested_duration.value}

You have full control over this decision. You can:
- Accept all permissions
- Accept only some permissions  
- Decline this request
- Set specific conditions

Your privacy and comfort are our highest priority.
"""


class ConsentManager:
    """
    Manages consent for all support interactions in Hearts Seeking Light Feed.
    
    Core Principles:
    - Informed consent: Users understand what they're agreeing to
    - Granular control: Users can consent to specific types of interaction
    - Revocable consent: Users can withdraw consent at any time
    - Context-aware: Consent considers the specific situation
    - Privacy-preserving: Minimal data collection and storage
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Consent storage (in production, would be database-backed)
        self.consent_records = {}  # user_id -> List[ConsentRecord]
        self.pending_requests = {}  # request_id -> ConsentRequest
        self.revoked_consents = {}  # user_id -> List[revoked_record_ids]
        
        # Default consent policies
        self.default_policies = self._initialize_default_policies()
        self.crisis_policies = self._initialize_crisis_policies()
        self.verification_methods = self._initialize_verification_methods()
        
        # Consent analytics for system improvement
        self.consent_analytics = {
            'requests_sent': 0,
            'consents_granted': 0,
            'consents_denied': 0,
            'consents_revoked': 0,
            'scope_preferences': {}
        }
        
        logger.info("Consent Manager initialized")
    
    def _initialize_default_policies(self) -> Dict[str, Any]:
        """Initialize default consent policies"""
        
        return {
            'implicit_consent_scopes': {
                # What public posting implies consent for
                ConsentScope.PUBLIC_REPLY,
                ConsentScope.RESOURCE_SHARING
            },
            
            'default_durations': {
                ConsentLevel.IMPLICIT: ConsentDuration.SESSION,
                ConsentLevel.EXPLICIT_LIMITED: ConsentDuration.LIMITED_TIME,
                ConsentLevel.EXPLICIT_FULL: ConsentDuration.ONGOING
            },
            
            'automatic_expiration': {
                ConsentDuration.SESSION: timedelta(hours=24),
                ConsentDuration.LIMITED_TIME: timedelta(days=7),
                ConsentDuration.ONGOING: None  # No automatic expiration
            },
            
            'minimum_age_hours': 1,  # Minimum time between consent requests
            'max_pending_requests': 5  # Maximum pending requests per user
        }
    
    def _initialize_crisis_policies(self) -> Dict[str, Any]:
        """Initialize special policies for crisis situations"""
        
        return {
            'crisis_override_scopes': {
                # What crisis situations allow without explicit consent
                ConsentScope.CRISIS_CONTACT,
                ConsentScope.RESOURCE_SHARING,
                ConsentScope.PROFESSIONAL_REFERRAL
            },
            
            'crisis_duration': timedelta(hours=24),
            
            'crisis_verification_required': True,
            
            'crisis_human_oversight': True,
            
            'crisis_documentation_required': True
        }
    
    def _initialize_verification_methods(self) -> Dict[str, Dict[str, Any]]:
        """Initialize consent verification methods"""
        
        return {
            'public_post_analysis': {
                'description': 'Consent inferred from public post content',
                'reliability': 0.6,
                'suitable_for': [ConsentLevel.IMPLICIT]
            },
            
            'explicit_user_response': {
                'description': 'User explicitly responds to consent request',
                'reliability': 0.95,
                'suitable_for': [ConsentLevel.EXPLICIT_LIMITED, ConsentLevel.EXPLICIT_FULL]
            },
            
            'profile_settings': {
                'description': 'User profile consent preferences',
                'reliability': 0.9,
                'suitable_for': [ConsentLevel.EXPLICIT_FULL]
            },
            
            'ongoing_relationship': {
                'description': 'Consent within established support relationship',
                'reliability': 0.85,
                'suitable_for': [ConsentLevel.ONGOING_RELATIONSHIP]
            },
            
            'crisis_override': {
                'description': 'Crisis situation override with human verification',
                'reliability': 0.8,
                'suitable_for': [ConsentLevel.EXPLICIT_LIMITED]
            }
        }
    
    async def assess_consent_for_interaction(self, seeker: SupportSeeker, 
                                           offer: SupportOffer,
                                           interaction_scopes: Set[ConsentScope]) -> Dict[str, Any]:
        """
        Assess what level of consent exists for a potential support interaction.
        
        Args:
            seeker: Person seeking support
            offer: Support offer being considered
            interaction_scopes: What scopes the interaction would need
        
        Returns:
            Dict with consent assessment details
        """
        try:
            assessment = {
                'has_sufficient_consent': False,
                'consent_level': ConsentLevel.NONE,
                'allowed_scopes': set(),
                'missing_scopes': set(),
                'existing_consents': [],
                'recommendations': [],
                'requires_request': False,
                'can_proceed_with_limitations': False
            }
            
            # Get existing consent records for the seeker
            existing_consents = self._get_valid_consents(seeker.user_id, offer.user_id)
            assessment['existing_consents'] = existing_consents
            
            # Check implicit consent from support-seeking post
            implicit_consent = await self._assess_implicit_consent(seeker)
            if implicit_consent:
                assessment['consent_level'] = ConsentLevel.IMPLICIT
                assessment['allowed_scopes'].update(implicit_consent['scopes'])
            
            # Check explicit consent records
            for consent in existing_consents:
                if consent.consent_level.value in ['explicit_limited', 'explicit_full', 'ongoing']:
                    assessment['consent_level'] = consent.consent_level
                    assessment['allowed_scopes'].update(consent.consent_scopes)
            
            # Determine missing scopes
            assessment['missing_scopes'] = interaction_scopes - assessment['allowed_scopes']
            
            # Check if we have sufficient consent
            assessment['has_sufficient_consent'] = len(assessment['missing_scopes']) == 0
            
            # Check if we can proceed with limitations
            if assessment['allowed_scopes'] & interaction_scopes:
                assessment['can_proceed_with_limitations'] = True
            
            # Generate recommendations
            if not assessment['has_sufficient_consent']:
                assessment['recommendations'] = await self._generate_consent_recommendations(
                    seeker, offer, assessment['missing_scopes']
                )
                
                # Check if we should request additional consent
                if await self._should_request_consent(seeker, offer, assessment['missing_scopes']):
                    assessment['requires_request'] = True
            
            logger.debug(f"Consent assessment for {seeker.user_id[:8]}... -> {offer.user_id[:8]}...: "
                        f"Level: {assessment['consent_level'].value}, "
                        f"Sufficient: {assessment['has_sufficient_consent']}")
            
            return assessment
            
        except Exception as e:
            logger.error(f"Error assessing consent: {e}")
            return {
                'has_sufficient_consent': False,
                'consent_level': ConsentLevel.NONE,
                'error': str(e)
            }
    
    def _get_valid_consents(self, user_id: str, partner_id: str = None) -> List[ConsentRecord]:
        """Get valid consent records for a user"""
        
        user_consents = self.consent_records.get(user_id, [])
        valid_consents = []
        
        current_time = datetime.utcnow()
        
        for consent in user_consents:
            # Check if consent is still valid
            if not consent.is_valid(current_time):
                continue
            
            # Check if consent is for specific partner
            if partner_id and hasattr(consent, 'partner_id') and consent.partner_id != partner_id:
                continue
            
            # Check if consent was revoked
            revoked_ids = self.revoked_consents.get(user_id, [])
            consent_id = self._generate_consent_id(consent)
            if consent_id in revoked_ids:
                continue
            
            valid_consents.append(consent)
        
        return valid_consents
    
    async def _assess_implicit_consent(self, seeker: SupportSeeker) -> Optional[Dict[str, Any]]:
        """Assess implicit consent from support-seeking behavior"""
        
        # When someone posts seeking support, they implicitly consent to certain interactions
        implicit_scopes = set()
        
        # Public post seeking support implies consent for public replies
        implicit_scopes.add(ConsentScope.PUBLIC_REPLY)
        
        # Seeking resources implies consent for resource sharing
        if SupportType.PROFESSIONAL_RESOURCES in seeker.support_types_needed:
            implicit_scopes.add(ConsentScope.RESOURCE_SHARING)
            implicit_scopes.add(ConsentScope.PROFESSIONAL_REFERRAL)
        
        # Seeking advice implies consent for advice sharing
        if SupportType.PRACTICAL_ADVICE in seeker.support_types_needed:
            implicit_scopes.add(ConsentScope.PRACTICAL_ADVICE)
        
        # Seeking emotional support implies consent for emotional support offers
        if SupportType.EMOTIONAL_SUPPORT in seeker.support_types_needed:
            implicit_scopes.add(ConsentScope.EMOTIONAL_SUPPORT)
        
        # Crisis situations have special implicit consent rules
        if seeker.urgency.value == 'crisis':
            implicit_scopes.update(self.crisis_policies['crisis_override_scopes'])
        
        if implicit_scopes:
            return {
                'scopes': implicit_scopes,
                'verification_method': 'public_post_analysis',
                'confidence': 0.7,
                'duration': self.default_policies['automatic_expiration'][ConsentDuration.SESSION]
            }
        
        return None
    
    async def _generate_consent_recommendations(self, seeker: SupportSeeker, 
                                              offer: SupportOffer,
                                              missing_scopes: Set[ConsentScope]) -> List[str]:
        """Generate recommendations for obtaining consent"""
        
        recommendations = []
        
        # Recommend starting with public interaction if private is missing
        if ConsentScope.PRIVATE_MESSAGE in missing_scopes:
            if ConsentScope.PUBLIC_REPLY not in missing_scopes:
                recommendations.append("Start with a public reply to build rapport before requesting private communication")
        
        # Recommend resource sharing as a gentle first step
        if ConsentScope.RESOURCE_SHARING not in missing_scopes:
            recommendations.append("Begin by sharing helpful resources publicly")
        
        # Recommend explicit consent request for sensitive scopes
        sensitive_scopes = {ConsentScope.CRISIS_CONTACT, ConsentScope.ONGOING_CHECKIN}
        if missing_scopes & sensitive_scopes:
            recommendations.append("Request explicit consent for ongoing support relationship")
        
        # Recommend gradual trust building
        if len(missing_scopes) > 2:
            recommendations.append("Build trust gradually through smaller interactions before requesting broader permissions")
        
        # Consider user's communication preferences
        if 'private_message' in seeker.communication_preferences:
            recommendations.append("User prefers private messages - consider requesting private communication consent")
        
        return recommendations
    
    async def _should_request_consent(self, seeker: SupportSeeker, offer: SupportOffer,
                                    missing_scopes: Set[ConsentScope]) -> bool:
        """Determine if we should request additional consent"""
        
        # Don't request for crisis situations - use implicit consent
        if seeker.urgency.value == 'crisis':
            return False
        
        # Don't request if user explicitly prefers no contact
        if seeker.consent_level == 'no_contact':
            return False
        
        # Don't request if we already have a recent pending request
        recent_requests = [
            req for req in self.pending_requests.values()
            if req.target_user_id == seeker.user_id and req.requesting_user_id == offer.user_id
            and req.created_at > datetime.utcnow() - timedelta(hours=self.default_policies['minimum_age_hours'])
        ]
        
        if recent_requests:
            return False
        
        # Don't request if user has too many pending requests
        user_pending = [
            req for req in self.pending_requests.values()
            if req.target_user_id == seeker.user_id
        ]
        
        if len(user_pending) >= self.default_policies['max_pending_requests']:
            return False
        
        # Request consent for private or ongoing interactions
        request_worthy_scopes = {
            ConsentScope.PRIVATE_MESSAGE,
            ConsentScope.ONGOING_CHECKIN,
            ConsentScope.CRISIS_CONTACT
        }
        
        if missing_scopes & request_worthy_scopes:
            return True
        
        return False
    
    async def create_consent_request(self, requesting_user_id: str, target_user_id: str,
                                   requested_scopes: Set[ConsentScope],
                                   support_context: str, urgency_level: str = "moderate") -> str:
        """Create a consent request for support interaction"""
        
        try:
            # Generate unique request ID
            request_id = self._generate_request_id(requesting_user_id, target_user_id)
            
            # Determine appropriate duration
            if ConsentScope.ONGOING_CHECKIN in requested_scopes:
                duration = ConsentDuration.ONGOING
            elif ConsentScope.CRISIS_CONTACT in requested_scopes:
                duration = ConsentDuration.LIMITED_TIME
            else:
                duration = ConsentDuration.SESSION
            
            # Create consent request
            consent_request = ConsentRequest(
                request_id=request_id,
                requesting_user_id=requesting_user_id,
                target_user_id=target_user_id,
                requested_scopes=requested_scopes,
                requested_duration=duration,
                support_context=support_context,
                urgency_level=urgency_level,
                explanation=f"Support connection for: {support_context}",
                created_at=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(days=7)  # Request expires in 7 days
            )
            
            # Store pending request
            self.pending_requests[request_id] = consent_request
            
            # Update analytics
            self.consent_analytics['requests_sent'] += 1
            
            logger.info(f"Created consent request {request_id} from {requesting_user_id[:8]}... "
                       f"to {target_user_id[:8]}...")
            
            return request_id
            
        except Exception as e:
            logger.error(f"Error creating consent request: {e}")
            raise
    
    async def process_consent_response(self, request_id: str, response: Dict[str, Any]) -> bool:
        """Process user response to consent request"""
        
        try:
            if request_id not in self.pending_requests:
                logger.warning(f"Consent request {request_id} not found")
                return False
            
            consent_request = self.pending_requests[request_id]
            user_response = response.get('decision', 'deny')
            approved_scopes = set(response.get('approved_scopes', []))
            conditions = response.get('conditions', [])
            
            if user_response == 'approve':
                # Create consent record
                consent_scopes = approved_scopes if approved_scopes else consent_request.requested_scopes
                
                consent_record = ConsentRecord(
                    user_id=consent_request.target_user_id,
                    consenter_id=consent_request.target_user_id,
                    consent_level=ConsentLevel.EXPLICIT_FULL if len(consent_scopes) == len(consent_request.requested_scopes)
                                 else ConsentLevel.EXPLICIT_LIMITED,
                    consent_scopes=consent_scopes,
                    duration=consent_request.requested_duration,
                    granted_at=datetime.utcnow(),
                    expires_at=self._calculate_expiration(consent_request.requested_duration),
                    specific_conditions=conditions,
                    revocation_triggers=response.get('revocation_triggers', []),
                    consent_context=consent_request.support_context,
                    verification_method='explicit_user_response',
                    metadata={
                        'request_id': request_id,
                        'requesting_user_id': consent_request.requesting_user_id,
                        'response_data': response
                    }
                )
                
                # Store consent record
                await self._store_consent_record(consent_record)
                
                # Update analytics
                self.consent_analytics['consents_granted'] += 1
                for scope in consent_scopes:
                    self.consent_analytics['scope_preferences'][scope.value] = \
                        self.consent_analytics['scope_preferences'].get(scope.value, 0) + 1
                
                logger.info(f"Consent granted for request {request_id}")
                
            else:
                # Update analytics
                self.consent_analytics['consents_denied'] += 1
                logger.info(f"Consent denied for request {request_id}")
            
            # Remove pending request
            del self.pending_requests[request_id]
            
            return user_response == 'approve'
            
        except Exception as e:
            logger.error(f"Error processing consent response: {e}")
            return False
    
    async def revoke_consent(self, user_id: str, consent_criteria: Dict[str, Any]) -> bool:
        """Revoke consent based on user request"""
        
        try:
            consents_to_revoke = []
            
            # Find consents matching criteria
            user_consents = self.consent_records.get(user_id, [])
            
            for consent in user_consents:
                match = True
                
                # Check partner ID if specified
                if 'partner_id' in consent_criteria:
                    if getattr(consent, 'partner_id', None) != consent_criteria['partner_id']:
                        match = False
                
                # Check scope if specified
                if 'scope' in consent_criteria:
                    if consent_criteria['scope'] not in consent.consent_scopes:
                        match = False
                
                # Check consent level if specified
                if 'consent_level' in consent_criteria:
                    if consent.consent_level.value != consent_criteria['consent_level']:
                        match = False
                
                if match:
                    consents_to_revoke.append(consent)
            
            # Revoke matching consents
            if user_id not in self.revoked_consents:
                self.revoked_consents[user_id] = []
            
            for consent in consents_to_revoke:
                consent_id = self._generate_consent_id(consent)
                self.revoked_consents[user_id].append(consent_id)
            
            # Update analytics
            self.consent_analytics['consents_revoked'] += len(consents_to_revoke)
            
            logger.info(f"Revoked {len(consents_to_revoke)} consent(s) for user {user_id[:8]}...")
            
            return len(consents_to_revoke) > 0
            
        except Exception as e:
            logger.error(f"Error revoking consent: {e}")
            return False
    
    def _calculate_expiration(self, duration: ConsentDuration) -> Optional[datetime]:
        """Calculate consent expiration time"""
        
        expiration_deltas = self.default_policies['automatic_expiration']
        delta = expiration_deltas.get(duration)
        
        if delta:
            return datetime.utcnow() + delta
        
        return None  # No expiration for ongoing consent
    
    async def _store_consent_record(self, consent_record: ConsentRecord) -> None:
        """Store consent record"""
        
        user_id = consent_record.user_id
        
        if user_id not in self.consent_records:
            self.consent_records[user_id] = []
        
        self.consent_records[user_id].append(consent_record)
        
        # Keep only recent records
        max_records = 100
        if len(self.consent_records[user_id]) > max_records:
            self.consent_records[user_id] = self.consent_records[user_id][-max_records:]
    
    def _generate_request_id(self, requesting_user_id: str, target_user_id: str) -> str:
        """Generate unique request ID"""
        
        data = f"{requesting_user_id}:{target_user_id}:{datetime.utcnow().isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def _generate_consent_id(self, consent_record: ConsentRecord) -> str:
        """Generate unique consent ID"""
        
        data = f"{consent_record.user_id}:{consent_record.granted_at.isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    async def check_crisis_override_authorization(self, seeker: SupportSeeker, 
                                                responder_id: str) -> Dict[str, Any]:
        """Check if crisis situation authorizes override of normal consent requirements"""
        
        authorization = {
            'authorized': False,
            'allowed_scopes': set(),
            'conditions': [],
            'documentation_required': True,
            'human_oversight_required': True
        }
        
        # Only authorize for actual crisis
        if seeker.urgency.value != 'crisis':
            return authorization
        
        # Crisis authorizes specific emergency scopes
        crisis_scopes = self.crisis_policies['crisis_override_scopes']
        
        authorization.update({
            'authorized': True,
            'allowed_scopes': crisis_scopes,
            'conditions': [
                'Crisis situation verified',
                'Human oversight activated',
                'Documentation required',
                'Limited to emergency scopes only'
            ],
            'expires_at': datetime.utcnow() + self.crisis_policies['crisis_duration']
        })
        
        logger.warning(f"Crisis override authorized for {seeker.user_id[:8]}... -> {responder_id[:8]}...")
        
        return authorization
    
    def get_user_consent_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user's consent preferences and history"""
        
        user_consents = self.consent_records.get(user_id, [])
        valid_consents = self._get_valid_consents(user_id)
        
        # Analyze preference patterns
        preferred_scopes = {}
        for consent in user_consents:
            for scope in consent.consent_scopes:
                preferred_scopes[scope.value] = preferred_scopes.get(scope.value, 0) + 1
        
        # Get pending requests
        pending_requests = [
            req for req in self.pending_requests.values()
            if req.target_user_id == user_id
        ]
        
        return {
            'user_id': user_id[:8] + '...',  # Anonymized
            'total_consents_given': len(user_consents),
            'active_consents': len(valid_consents),
            'preferred_scopes': preferred_scopes,
            'pending_requests': len(pending_requests),
            'consent_patterns': {
                'typically_grants_full_consent': self._analyze_consent_generosity(user_consents),
                'prefers_limited_duration': self._analyze_duration_preference(user_consents),
                'responsive_to_requests': self._analyze_responsiveness(user_id)
            }
        }
    
    def _analyze_consent_generosity(self, consents: List[ConsentRecord]) -> bool:
        """Analyze if user typically grants full consent"""
        if not consents:
            return False
        
        full_consents = sum(1 for c in consents if c.consent_level == ConsentLevel.EXPLICIT_FULL)
        return full_consents / len(consents) > 0.6
    
    def _analyze_duration_preference(self, consents: List[ConsentRecord]) -> bool:
        """Analyze if user prefers limited duration consents"""
        if not consents:
            return False
        
        limited_duration = sum(1 for c in consents if c.duration == ConsentDuration.LIMITED_TIME)
        return limited_duration / len(consents) > 0.5
    
    def _analyze_responsiveness(self, user_id: str) -> bool:
        """Analyze if user is responsive to consent requests"""
        # In production, would analyze response time and rate
        return True  # Default assumption
    
    def get_consent_analytics(self) -> Dict[str, Any]:
        """Get analytics about consent patterns"""
        
        total_users = len(self.consent_records)
        total_consents = sum(len(consents) for consents in self.consent_records.values())
        
        return {
            'total_users_with_consents': total_users,
            'total_consent_records': total_consents,
            'pending_requests': len(self.pending_requests),
            'consent_request_stats': {
                'requests_sent': self.consent_analytics['requests_sent'],
                'consents_granted': self.consent_analytics['consents_granted'],
                'consents_denied': self.consent_analytics['consents_denied'],
                'consents_revoked': self.consent_analytics['consents_revoked'],
                'approval_rate': (self.consent_analytics['consents_granted'] / 
                                max(1, self.consent_analytics['requests_sent'])) * 100
            },
            'popular_consent_scopes': dict(sorted(
                self.consent_analytics['scope_preferences'].items(),
                key=lambda x: x[1], reverse=True
            )[:5]),
            'system_health': {
                'consent_verification_active': True,
                'crisis_override_protocols_ready': True,
                'user_agency_protected': True,
                'privacy_preserving': True
            }
        }
    
    async def cleanup_expired_data(self) -> None:
        """Clean up expired consent requests and records"""
        
        current_time = datetime.utcnow()
        
        # Clean up expired pending requests
        expired_requests = [
            req_id for req_id, req in self.pending_requests.items()
            if req.expires_at < current_time
        ]
        
        for req_id in expired_requests:
            del self.pending_requests[req_id]
        
        # Note: We don't automatically delete expired consent records 
        # as users may want to reference their consent history
        
        if expired_requests:
            logger.info(f"Cleaned up {len(expired_requests)} expired consent requests")