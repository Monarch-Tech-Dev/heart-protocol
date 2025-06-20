"""
Support Detection Integration

Builds on the existing care detection engine to identify:
1. People seeking support (Hearts Seeking Light)
2. People offering support (Light Givers)
3. Context-appropriate matching opportunities

Maintains strict privacy and consent requirements from the Open Source Love License.
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
import logging
import re

from ...core.base import Post, CareAssessment, CareLevel
from ...core.care_detection import CareDetectionEngine

logger = logging.getLogger(__name__)


class SupportType(Enum):
    """Types of support that can be offered or needed"""
    EMOTIONAL_SUPPORT = "emotional_support"         # Listening, empathy, understanding
    PRACTICAL_ADVICE = "practical_advice"           # Specific guidance, tips, resources
    SHARED_EXPERIENCE = "shared_experience"         # "I've been there too" connection
    PROFESSIONAL_RESOURCES = "professional_resources" # Therapist referrals, crisis lines
    COMMUNITY_CONNECTION = "community_connection"   # Local groups, meetups, activities
    CRISIS_INTERVENTION = "crisis_intervention"     # Immediate safety support
    HEALING_JOURNEY = "healing_journey"             # Long-term recovery companionship
    SKILL_SHARING = "skill_sharing"                 # Teaching coping mechanisms
    ACCOUNTABILITY = "accountability"               # Gentle check-ins, progress support
    CELEBRATION = "celebration"                     # Acknowledging wins and milestones


class SupportUrgency(Enum):
    """Urgency levels for support needs"""
    CRISIS = "crisis"                 # Immediate intervention needed
    HIGH = "high"                     # Support needed within hours
    MODERATE = "moderate"             # Support helpful within days
    LOW = "low"                       # Support welcome when available
    ONGOING = "ongoing"               # Long-term support relationship


@dataclass
class SupportSeeker:
    """Represents someone seeking support"""
    user_id: str
    post_id: str
    support_types_needed: List[SupportType]
    urgency: SupportUrgency
    care_assessment: CareAssessment
    context_summary: str
    specific_needs: List[str]
    preferred_demographics: Dict[str, Any]  # age_range, gender_preference, etc.
    communication_preferences: List[str]    # direct_message, public_reply, etc.
    timezone: str
    available_times: List[str]
    consent_level: str  # Will be defined in consent_system.py
    anonymity_preference: str
    detected_at: datetime
    expires_at: datetime
    support_keywords: List[str]
    trigger_warnings: List[str]
    previous_support_received: int
    
    def __post_init__(self):
        if self.detected_at is None:
            self.detected_at = datetime.utcnow()
        if self.expires_at is None:
            # Support needs expire after 72 hours unless crisis
            hours = 24 if self.urgency == SupportUrgency.CRISIS else 72
            self.expires_at = self.detected_at + timedelta(hours=hours)


@dataclass
class SupportOffer:
    """Represents someone offering support"""
    user_id: str
    post_id: Optional[str]  # May be proactive offer, not response to specific post
    support_types_offered: List[SupportType]
    availability: str  # "immediate", "within_hours", "flexible"
    experience_areas: List[str]  # depression, anxiety, grief, etc.
    credentials: List[str]       # peer_counselor, therapist, lived_experience
    demographic_info: Dict[str, Any]
    communication_preferences: List[str]
    timezone: str
    available_times: List[str]
    max_concurrent_support: int  # How many people they can help at once
    current_support_count: int
    consent_level: str
    created_at: datetime
    expires_at: datetime
    success_stories: int         # Past successful support connections
    feedback_rating: float       # Average rating from those they've helped
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.expires_at is None:
            # Support offers expire after 7 days
            self.expires_at = self.created_at + timedelta(days=7)


class SupportDetectionEngine:
    """
    Extends the care detection engine to identify support-seeking and 
    support-offering posts with high precision and privacy protection.
    """
    
    def __init__(self, config: Dict[str, Any], care_detection_engine: CareDetectionEngine):
        self.config = config
        self.care_engine = care_detection_engine
        
        # Initialize support detection patterns
        self.support_seeking_patterns = self._initialize_support_seeking_patterns()
        self.support_offering_patterns = self._initialize_support_offering_patterns()
        self.false_positive_filters = self._initialize_false_positive_filters()
        
        # Active support seekers and offers
        self.active_seekers = {}  # user_id -> SupportSeeker
        self.active_offers = {}   # user_id -> List[SupportOffer]
        
        # Detection metrics for continuous improvement
        self.detection_metrics = {
            'seekers_detected': 0,
            'offers_detected': 0,
            'false_positives_filtered': 0,
            'successful_connections': 0
        }
        
        logger.info("Support Detection Engine initialized")
    
    def _initialize_support_seeking_patterns(self) -> Dict[SupportType, Dict[str, Any]]:
        """Initialize patterns for detecting support-seeking posts"""
        
        return {
            SupportType.EMOTIONAL_SUPPORT: {
                'keywords': [
                    'feeling alone', 'need someone to talk', 'struggling with', 
                    'overwhelmed', 'need support', 'going through', 'hard time',
                    'feeling lost', 'need help', 'anyone else', 'feeling isolated',
                    'need a friend', 'feeling down', 'bad day', 'rough patch'
                ],
                'phrases': [
                    'could really use someone to talk to',
                    'feeling like no one understands',
                    'need someone who gets it',
                    'feeling so alone right now',
                    'could use a listening ear',
                    'need some emotional support'
                ],
                'urgency_indicators': {
                    SupportUrgency.HIGH: ['really struggling', 'desperate', 'urgent'],
                    SupportUrgency.MODERATE: ['having a hard time', 'difficult', 'tough'],
                    SupportUrgency.LOW: ['could use', 'would appreciate', 'if anyone']
                }
            },
            
            SupportType.PRACTICAL_ADVICE: {
                'keywords': [
                    'how do i', 'what should i do', 'advice needed', 'suggestions',
                    'tips', 'guidance', 'help with', 'best way to', 'how to cope',
                    'strategies', 'what works', 'recommendations', 'resources'
                ],
                'phrases': [
                    'any advice would be helpful',
                    'what has worked for you',
                    'looking for practical tips',
                    'need guidance on how to',
                    'what would you recommend',
                    'how did you handle'
                ],
                'urgency_indicators': {
                    SupportUrgency.HIGH: ['need to know asap', 'urgent advice'],
                    SupportUrgency.MODERATE: ['advice needed', 'help figuring out'],
                    SupportUrgency.LOW: ['curious about', 'wondering if']
                }
            },
            
            SupportType.SHARED_EXPERIENCE: {
                'keywords': [
                    'anyone else', 'has anyone', 'others who', 'similar experience',
                    'been through', 'going through same', 'relate to', 'understand',
                    'in the same boat', 'feel alone in this', 'only one'
                ],
                'phrases': [
                    'is anyone else dealing with',
                    'has anyone been through',
                    'looking for others who understand',
                    'need to know i am not alone',
                    'others with similar experience',
                    'anyone who can relate'
                ],
                'urgency_indicators': {
                    SupportUrgency.MODERATE: ['really need to connect', 'feeling isolated'],
                    SupportUrgency.LOW: ['curious if others', 'wondering about']
                }
            },
            
            SupportType.CRISIS_INTERVENTION: {
                'keywords': [
                    'crisis', 'emergency', 'cant take it', 'end it all', 'give up',
                    'hopeless', 'no point', 'cant go on', 'want to die', 'hurt myself'
                ],
                'phrases': [
                    'thinking about ending it',
                    'not sure i can go on',
                    'feeling like giving up',
                    'in crisis right now',
                    'need immediate help',
                    'contemplating suicide'
                ],
                'urgency_indicators': {
                    SupportUrgency.CRISIS: ['all_crisis_keywords']  # All crisis keywords are urgent
                }
            },
            
            SupportType.PROFESSIONAL_RESOURCES: {
                'keywords': [
                    'need therapist', 'find counselor', 'therapy recommendations',
                    'mental health professional', 'crisis hotline', 'treatment options',
                    'professional help', 'psychiatric care', 'medication help'
                ],
                'phrases': [
                    'looking for a therapist',
                    'need professional help',
                    'therapy recommendations in',
                    'mental health resources',
                    'how to find help',
                    'need medical attention'
                ]
            }
        }
    
    def _initialize_support_offering_patterns(self) -> Dict[SupportType, Dict[str, Any]]:
        """Initialize patterns for detecting support-offering posts"""
        
        return {
            SupportType.EMOTIONAL_SUPPORT: {
                'keywords': [
                    'here to listen', 'dm me', 'reach out', 'always available',
                    'open to chat', 'inbox open', 'here for you', 'message me',
                    'need someone to talk', 'lending an ear', 'support anyone'
                ],
                'phrases': [
                    'if anyone needs to talk',
                    'here if you need support',
                    'always willing to listen',
                    'my dms are open',
                    'reach out if you need',
                    'here for anyone struggling'
                ]
            },
            
            SupportType.PRACTICAL_ADVICE: {
                'keywords': [
                    'been there', 'share my experience', 'what worked for me',
                    'advice available', 'tips that helped', 'strategies i used',
                    'resources that work', 'guidance based on', 'lessons learned'
                ],
                'phrases': [
                    'happy to share what worked',
                    'can offer some tips',
                    'learned some strategies',
                    'resources that helped me',
                    'experience might help',
                    'advice from my journey'
                ]
            },
            
            SupportType.SHARED_EXPERIENCE: {
                'keywords': [
                    'been through', 'survived', 'overcame', 'journey with',
                    'experience with', 'dealt with', 'lived through', 'understand',
                    'relate to', 'similar story', 'walked this path'
                ],
                'phrases': [
                    'been through similar',
                    'survived the same thing',
                    'understand what you are going through',
                    'walked a similar path',
                    'share my story if helpful',
                    'connect with others who'
                ]
            },
            
            SupportType.PROFESSIONAL_RESOURCES: {
                'keywords': [
                    'therapist', 'counselor', 'mental health professional',
                    'crisis line', 'hotline', 'treatment center', 'support group',
                    'peer counselor', 'trained in', 'certified', 'licensed'
                ],
                'phrases': [
                    'trained peer counselor',
                    'mental health professional',
                    'know good resources',
                    'connections to help',
                    'professional background in',
                    'can recommend services'
                ]
            }
        }
    
    def _initialize_false_positive_filters(self) -> Dict[str, List[str]]:
        """Initialize filters to prevent false positive matches"""
        
        return {
            'support_seeking_false_positives': [
                'just venting',
                'not looking for advice',
                'rhetorical question', 
                'dont need help',
                'figure it out myself',
                'just sharing',
                'not seeking support',
                'already have help',
                'just processing'
            ],
            
            'support_offering_false_positives': [
                'not qualified',
                'cant help right now',
                'too busy to',
                'not available',
                'dont have experience',
                'not the right person',
                'wish i could help but',
                'if only i could',
                'sorry cant'
            ],
            
            'general_false_positives': [
                'just kidding',
                'sarcasm',
                'joking',
                'not serious',
                'being dramatic',
                'exaggerating',
                'figure of speech',
                'hypothetically',
                'asking for a friend'
            ]
        }
    
    async def detect_support_seeking(self, post: Post, care_assessment: CareAssessment) -> Optional[SupportSeeker]:
        """
        Detect if a post represents someone seeking support.
        
        Builds on existing care assessment but looks specifically for 
        explicit or implicit requests for community support.
        """
        try:
            # First check if care assessment indicates need
            if care_assessment.care_level == CareLevel.NONE:
                return None
            
            # Check for false positives first
            if await self._is_false_positive_seeker(post):
                self.detection_metrics['false_positives_filtered'] += 1
                return None
            
            # Analyze post for support-seeking patterns
            support_types = await self._identify_needed_support_types(post)
            
            if not support_types:
                return None
            
            # Determine urgency based on care level and content
            urgency = await self._determine_support_urgency(post, care_assessment)
            
            # Extract specific needs and context
            specific_needs = await self._extract_specific_needs(post)
            context_summary = await self._generate_context_summary(post, care_assessment)
            
            # Get user preferences (if available)
            preferences = await self._get_user_support_preferences(post.author_id)
            
            # Create support seeker
            seeker = SupportSeeker(
                user_id=post.author_id,
                post_id=post.id,
                support_types_needed=support_types,
                urgency=urgency,
                care_assessment=care_assessment,
                context_summary=context_summary,
                specific_needs=specific_needs,
                preferred_demographics=preferences.get('demographics', {}),
                communication_preferences=preferences.get('communication', ['direct_message']),
                timezone=preferences.get('timezone', 'UTC'),
                available_times=preferences.get('available_times', ['anytime']),
                consent_level=preferences.get('consent_level', 'explicit_consent_required'),
                anonymity_preference=preferences.get('anonymity', 'partial_anonymous'),
                support_keywords=await self._extract_support_keywords(post),
                trigger_warnings=await self._identify_trigger_warnings(post),
                previous_support_received=preferences.get('previous_support_count', 0),
                detected_at=datetime.utcnow(),
                expires_at=None  # Will be set in __post_init__
            )
            
            # Store active seeker
            self.active_seekers[post.author_id] = seeker
            self.detection_metrics['seekers_detected'] += 1
            
            logger.info(f"Support seeker detected: {post.author_id[:8]}... "
                       f"(Types: {[t.value for t in support_types]}, Urgency: {urgency.value})")
            
            return seeker
            
        except Exception as e:
            logger.error(f"Error detecting support seeking: {e}")
            return None
    
    async def detect_support_offering(self, post: Post) -> Optional[SupportOffer]:
        """
        Detect if a post represents someone offering support to others.
        """
        try:
            # Check for false positives
            if await self._is_false_positive_offer(post):
                self.detection_metrics['false_positives_filtered'] += 1
                return None
            
            # Analyze post for support-offering patterns
            support_types = await self._identify_offered_support_types(post)
            
            if not support_types:
                return None
            
            # Extract offering details
            availability = await self._determine_availability(post)
            experience_areas = await self._extract_experience_areas(post)
            credentials = await self._identify_credentials(post)
            
            # Get user support profile
            support_profile = await self._get_user_support_profile(post.author_id)
            
            # Create support offer
            offer = SupportOffer(
                user_id=post.author_id,
                post_id=post.id,
                support_types_offered=support_types,
                availability=availability,
                experience_areas=experience_areas,
                credentials=credentials,
                demographic_info=support_profile.get('demographics', {}),
                communication_preferences=support_profile.get('communication', ['direct_message']),
                timezone=support_profile.get('timezone', 'UTC'),
                available_times=support_profile.get('available_times', ['anytime']),
                max_concurrent_support=support_profile.get('max_concurrent', 3),
                current_support_count=support_profile.get('current_count', 0),
                consent_level=support_profile.get('consent_level', 'explicit_consent_required'),
                success_stories=support_profile.get('success_count', 0),
                feedback_rating=support_profile.get('rating', 0.0),
                created_at=datetime.utcnow(),
                expires_at=None  # Will be set in __post_init__
            )
            
            # Store active offer
            if post.author_id not in self.active_offers:
                self.active_offers[post.author_id] = []
            self.active_offers[post.author_id].append(offer)
            
            self.detection_metrics['offers_detected'] += 1
            
            logger.info(f"Support offer detected: {post.author_id[:8]}... "
                       f"(Types: {[t.value for t in support_types]}, Availability: {availability})")
            
            return offer
            
        except Exception as e:
            logger.error(f"Error detecting support offering: {e}")
            return None
    
    async def _is_false_positive_seeker(self, post: Post) -> bool:
        """Check if post is false positive for support seeking"""
        
        content_lower = post.content.lower()
        
        # Check explicit false positive patterns
        for fp_pattern in self.false_positive_filters['support_seeking_false_positives']:
            if fp_pattern in content_lower:
                return True
        
        # Check general false positives
        for fp_pattern in self.false_positive_filters['general_false_positives']:
            if fp_pattern in content_lower:
                return True
        
        # Check if post is informational/educational rather than personal
        educational_indicators = [
            'according to research', 'studies show', 'experts say',
            'general information', 'fyi', 'psa', 'reminder that'
        ]
        
        if any(indicator in content_lower for indicator in educational_indicators):
            return True
        
        # Check if post is about someone else
        third_person_indicators = ['my friend', 'someone i know', 'they are', 'their situation']
        if any(indicator in content_lower for indicator in third_person_indicators):
            return True
        
        return False
    
    async def _is_false_positive_offer(self, post: Post) -> bool:
        """Check if post is false positive for support offering"""
        
        content_lower = post.content.lower()
        
        # Check explicit false positive patterns
        for fp_pattern in self.false_positive_filters['support_offering_false_positives']:
            if fp_pattern in content_lower:
                return True
        
        # Check general false positives
        for fp_pattern in self.false_positive_filters['general_false_positives']:
            if fp_pattern in content_lower:
                return True
        
        return False
    
    async def _identify_needed_support_types(self, post: Post) -> List[SupportType]:
        """Identify what types of support the person needs"""
        
        content_lower = post.content.lower()
        detected_types = []
        
        for support_type, patterns in self.support_seeking_patterns.items():
            # Check keywords
            keyword_matches = sum(1 for keyword in patterns['keywords'] 
                                if keyword in content_lower)
            
            # Check phrases
            phrase_matches = sum(1 for phrase in patterns['phrases'] 
                               if phrase in content_lower)
            
            # Calculate confidence score
            total_patterns = len(patterns['keywords']) + len(patterns['phrases'])
            confidence = (keyword_matches + phrase_matches * 2) / total_patterns
            
            # Threshold for detection
            if confidence > 0.1:  # Fairly sensitive to catch subtle requests
                detected_types.append(support_type)
        
        return detected_types
    
    async def _identify_offered_support_types(self, post: Post) -> List[SupportType]:
        """Identify what types of support the person is offering"""
        
        content_lower = post.content.lower()
        detected_types = []
        
        for support_type, patterns in self.support_offering_patterns.items():
            # Check keywords
            keyword_matches = sum(1 for keyword in patterns['keywords'] 
                                if keyword in content_lower)
            
            # Check phrases  
            phrase_matches = sum(1 for phrase in patterns['phrases'] 
                               if phrase in content_lower)
            
            # Calculate confidence score
            total_patterns = len(patterns['keywords']) + len(patterns['phrases'])
            confidence = (keyword_matches + phrase_matches * 2) / total_patterns
            
            # Threshold for detection
            if confidence > 0.15:  # Slightly higher threshold for offers
                detected_types.append(support_type)
        
        return detected_types
    
    async def _determine_support_urgency(self, post: Post, care_assessment: CareAssessment) -> SupportUrgency:
        """Determine urgency level of support need"""
        
        # Crisis level care assessment means crisis urgency
        if care_assessment.care_level == CareLevel.CRISIS:
            return SupportUrgency.CRISIS
        
        content_lower = post.content.lower()
        
        # Check for crisis indicators in content
        crisis_keywords = ['crisis', 'emergency', 'cant take it', 'give up', 'end it all']
        if any(keyword in content_lower for keyword in crisis_keywords):
            return SupportUrgency.CRISIS
        
        # Check urgency indicators from patterns
        for support_type, patterns in self.support_seeking_patterns.items():
            urgency_indicators = patterns.get('urgency_indicators', {})
            
            for urgency_level, indicators in urgency_indicators.items():
                if any(indicator in content_lower for indicator in indicators):
                    return urgency_level
        
        # Default based on care level
        if care_assessment.care_level == CareLevel.HIGH:
            return SupportUrgency.HIGH
        elif care_assessment.care_level == CareLevel.MODERATE:
            return SupportUrgency.MODERATE
        else:
            return SupportUrgency.LOW
    
    async def _extract_specific_needs(self, post: Post) -> List[str]:
        """Extract specific needs mentioned in the post"""
        
        content_lower = post.content.lower()
        specific_needs = []
        
        # Common need patterns
        need_patterns = {
            'listening': ['someone to listen', 'need to talk', 'vent', 'share'],
            'advice': ['what should i do', 'advice', 'suggestions', 'guidance'],
            'resources': ['resources', 'help finding', 'recommendations', 'referrals'],
            'companionship': ['feel alone', 'isolated', 'lonely', 'connection'],
            'understanding': ['understand', 'relate', 'been there', 'similar'],
            'encouragement': ['motivation', 'encouragement', 'hope', 'strength'],
            'practical_help': ['practical help', 'steps', 'how to', 'what to do']
        }
        
        for need, keywords in need_patterns.items():
            if any(keyword in content_lower for keyword in keywords):
                specific_needs.append(need)
        
        return specific_needs
    
    async def _generate_context_summary(self, post: Post, care_assessment: CareAssessment) -> str:
        """Generate privacy-preserving context summary"""
        
        # Create summary without revealing personal details
        summary_parts = []
        
        # Care level context
        summary_parts.append(f"Care level: {care_assessment.care_level.value}")
        
        # Support types needed
        support_types = await self._identify_needed_support_types(post)
        if support_types:
            types_str = ', '.join([t.value.replace('_', ' ') for t in support_types])
            summary_parts.append(f"Support types: {types_str}")
        
        # General situation indicators (without specific details)
        situation_indicators = []
        content_lower = post.content.lower()
        
        general_situations = {
            'mental_health': ['depression', 'anxiety', 'mental health', 'therapy'],
            'life_transition': ['moving', 'job', 'relationship', 'school', 'change'],
            'loss_grief': ['loss', 'grief', 'died', 'passed away', 'mourning'],
            'health_concerns': ['health', 'medical', 'illness', 'condition', 'diagnosis'],
            'family_issues': ['family', 'parents', 'children', 'marriage', 'divorce'],
            'work_stress': ['work', 'job', 'career', 'workplace', 'employment']
        }
        
        for situation, keywords in general_situations.items():
            if any(keyword in content_lower for keyword in keywords):
                situation_indicators.append(situation.replace('_', ' '))
        
        if situation_indicators:
            summary_parts.append(f"Situation areas: {', '.join(situation_indicators)}")
        
        return '; '.join(summary_parts)
    
    async def _get_user_support_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user's support preferences (would be from user profile in production)"""
        
        # Default preferences - in production, would query user profile
        return {
            'demographics': {},
            'communication': ['direct_message'],
            'timezone': 'UTC',
            'available_times': ['anytime'],
            'consent_level': 'explicit_consent_required',
            'anonymity': 'partial_anonymous',
            'previous_support_count': 0
        }
    
    async def _get_user_support_profile(self, user_id: str) -> Dict[str, Any]:
        """Get user's support offering profile"""
        
        # Default profile - in production, would query user profile
        return {
            'demographics': {},
            'communication': ['direct_message'],
            'timezone': 'UTC', 
            'available_times': ['anytime'],
            'max_concurrent': 3,
            'current_count': 0,
            'consent_level': 'explicit_consent_required',
            'success_count': 0,
            'rating': 0.0
        }
    
    async def _determine_availability(self, post: Post) -> str:
        """Determine availability from support offering post"""
        
        content_lower = post.content.lower()
        
        immediate_indicators = ['right now', 'immediately', 'available now', 'online now']
        if any(indicator in content_lower for indicator in immediate_indicators):
            return 'immediate'
        
        hours_indicators = ['within hours', 'today', 'this evening', 'later today']
        if any(indicator in content_lower for indicator in hours_indicators):
            return 'within_hours'
        
        return 'flexible'
    
    async def _extract_experience_areas(self, post: Post) -> List[str]:
        """Extract areas of experience from support offering post"""
        
        content_lower = post.content.lower()
        experience_areas = []
        
        # Common experience areas
        areas = [
            'depression', 'anxiety', 'ptsd', 'grief', 'loss', 'addiction',
            'divorce', 'job loss', 'illness', 'disability', 'parenting',
            'relationships', 'school', 'work stress', 'family issues'
        ]
        
        for area in areas:
            if area in content_lower:
                experience_areas.append(area)
        
        return experience_areas
    
    async def _identify_credentials(self, post: Post) -> List[str]:
        """Identify any credentials mentioned in support offering post"""
        
        content_lower = post.content.lower()
        credentials = []
        
        credential_keywords = {
            'peer_counselor': ['peer counselor', 'peer support'],
            'therapist': ['therapist', 'licensed', 'lcsw', 'lpc', 'psychologist'],
            'lived_experience': ['been there', 'survived', 'recovered', 'overcame'],
            'training': ['trained', 'certified', 'educated', 'studied'],
            'volunteer': ['volunteer', 'crisis line', 'hotline']
        }
        
        for credential, keywords in credential_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                credentials.append(credential)
        
        return credentials
    
    async def _extract_support_keywords(self, post: Post) -> List[str]:
        """Extract keywords relevant to support matching"""
        
        content_lower = post.content.lower()
        keywords = []
        
        # Extract significant words (simple approach)
        words = re.findall(r'\b\w+\b', content_lower)
        
        # Filter for relevant support keywords
        relevant_keywords = [
            'depression', 'anxiety', 'grief', 'loss', 'stress', 'overwhelmed',
            'lonely', 'isolated', 'struggling', 'help', 'support', 'listen',
            'understand', 'care', 'connect', 'advice', 'guidance', 'hope'
        ]
        
        for word in words:
            if word in relevant_keywords:
                keywords.append(word)
        
        return list(set(keywords))  # Remove duplicates
    
    async def _identify_trigger_warnings(self, post: Post) -> List[str]:
        """Identify potential trigger warnings in content"""
        
        content_lower = post.content.lower()
        triggers = []
        
        trigger_keywords = {
            'suicide': ['suicide', 'kill myself', 'end it all', 'want to die'],
            'self_harm': ['self harm', 'cutting', 'hurt myself'],
            'abuse': ['abuse', 'violence', 'assault', 'trauma'],
            'eating_disorder': ['eating disorder', 'anorexia', 'bulimia', 'purging'],
            'addiction': ['addiction', 'substance', 'drugs', 'alcohol'],
            'death': ['death', 'died', 'funeral', 'grave']
        }
        
        for trigger, keywords in trigger_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                triggers.append(trigger)
        
        return triggers
    
    def get_active_seekers(self, urgency_filter: Optional[SupportUrgency] = None) -> List[SupportSeeker]:
        """Get active support seekers, optionally filtered by urgency"""
        
        current_time = datetime.utcnow()
        active = []
        
        for seeker in self.active_seekers.values():
            # Remove expired seekers
            if seeker.expires_at < current_time:
                continue
            
            # Apply urgency filter
            if urgency_filter and seeker.urgency != urgency_filter:
                continue
            
            active.append(seeker)
        
        # Sort by urgency and detection time
        urgency_order = {
            SupportUrgency.CRISIS: 0,
            SupportUrgency.HIGH: 1, 
            SupportUrgency.MODERATE: 2,
            SupportUrgency.LOW: 3,
            SupportUrgency.ONGOING: 4
        }
        
        active.sort(key=lambda s: (urgency_order[s.urgency], s.detected_at))
        return active
    
    def get_active_offers(self, support_type_filter: Optional[SupportType] = None) -> List[SupportOffer]:
        """Get active support offers, optionally filtered by support type"""
        
        current_time = datetime.utcnow()
        active = []
        
        for user_offers in self.active_offers.values():
            for offer in user_offers:
                # Remove expired offers
                if offer.expires_at < current_time:
                    continue
                
                # Check availability
                if offer.current_support_count >= offer.max_concurrent_support:
                    continue
                
                # Apply support type filter
                if support_type_filter and support_type_filter not in offer.support_types_offered:
                    continue
                
                active.append(offer)
        
        # Sort by rating and availability
        active.sort(key=lambda o: (-o.feedback_rating, o.current_support_count))
        return active
    
    def get_detection_metrics(self) -> Dict[str, Any]:
        """Get detection performance metrics"""
        
        return {
            'seekers_detected': self.detection_metrics['seekers_detected'],
            'offers_detected': self.detection_metrics['offers_detected'],
            'false_positives_filtered': self.detection_metrics['false_positives_filtered'],
            'successful_connections': self.detection_metrics['successful_connections'],
            'active_seekers': len(self.active_seekers),
            'active_offers': sum(len(offers) for offers in self.active_offers.values()),
            'detection_accuracy': self._calculate_detection_accuracy(),
            'system_health': {
                'care_engine_integration': True,
                'privacy_protection_active': True,
                'false_positive_filtering': True
            }
        }
    
    def _calculate_detection_accuracy(self) -> float:
        """Calculate detection accuracy based on successful connections"""
        
        total_detections = (self.detection_metrics['seekers_detected'] + 
                           self.detection_metrics['offers_detected'])
        
        if total_detections == 0:
            return 0.0
        
        # Simple accuracy metric - in production would be more sophisticated
        successful_rate = self.detection_metrics['successful_connections'] / total_detections
        return min(1.0, successful_rate)
    
    async def cleanup_expired_entries(self) -> None:
        """Clean up expired seekers and offers"""
        
        current_time = datetime.utcnow()
        
        # Clean up expired seekers
        expired_seekers = [
            user_id for user_id, seeker in self.active_seekers.items()
            if seeker.expires_at < current_time
        ]
        
        for user_id in expired_seekers:
            del self.active_seekers[user_id]
        
        # Clean up expired offers
        for user_id, offers in list(self.active_offers.items()):
            active_offers = [offer for offer in offers if offer.expires_at >= current_time]
            
            if not active_offers:
                del self.active_offers[user_id]
            else:
                self.active_offers[user_id] = active_offers
        
        if expired_seekers or any(len(offers) != len(self.active_offers.get(user_id, [])) 
                                 for user_id, offers in self.active_offers.items()):
            logger.info(f"Cleaned up {len(expired_seekers)} expired seekers and expired offers")