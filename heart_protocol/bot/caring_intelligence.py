"""
Caring Intelligence for Monarch Bot

The decision-making system that determines how the bot should respond
to different situations, prioritizing care, safety, and healing.
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
import logging

from .persona import ResponsePersonality
from .response_generator import ResponseType, ResponseContext
from ..core.base import CareLevel

logger = logging.getLogger(__name__)


class CareNeeds(Enum):
    """Different types of care needs the bot can identify"""
    IMMEDIATE_SAFETY = "immediate_safety"         # Crisis intervention needed
    EMOTIONAL_VALIDATION = "emotional_validation" # Need for emotional validation
    GENTLE_GUIDANCE = "gentle_guidance"           # Need for soft direction
    HOPE_AND_ENCOURAGEMENT = "hope_and_encouragement" # Need for hope
    PRACTICAL_SUPPORT = "practical_support"       # Need for resources/suggestions
    CELEBRATION = "celebration"                   # Achievement recognition
    BOUNDARY_RESPECT = "boundary_respect"         # Respecting user limits
    WISDOM_SHARING = "wisdom_sharing"             # Sharing relevant insights
    LOVING_PRESENCE = "loving_presence"           # Just being present with care
    TRAUMA_INFORMED_CARE = "trauma_informed_care" # Specialized trauma response


class InteractionIntent(Enum):
    """User's intent in the interaction"""
    SEEKING_SUPPORT = "seeking_support"           # Looking for emotional support
    ASKING_FOR_GUIDANCE = "asking_for_guidance"   # Want direction or advice
    SHARING_PROGRESS = "sharing_progress"         # Sharing achievements
    EXPRESSING_STRUGGLE = "expressing_struggle"   # Sharing difficulties
    IN_CRISIS = "in_crisis"                      # Crisis situation
    SETTING_BOUNDARIES = "setting_boundaries"    # Establishing limits
    SEEKING_VALIDATION = "seeking_validation"     # Want feelings validated
    REQUESTING_RESOURCES = "requesting_resources" # Looking for help/resources
    CASUAL_CONVERSATION = "casual_conversation"   # Light interaction
    ENDING_INTERACTION = "ending_interaction"     # Saying goodbye


@dataclass
class CaringAssessment:
    """Assessment of user's current care needs"""
    primary_care_need: CareNeeds
    secondary_care_needs: List[CareNeeds]
    interaction_intent: InteractionIntent
    emotional_state: str
    safety_level: CareLevel
    vulnerability_indicators: List[str]
    strength_indicators: List[str]
    cultural_considerations: List[str]
    trauma_sensitivity_needed: bool
    urgency_level: float  # 0.0 to 1.0
    support_capacity: float  # User's capacity to receive support (0.0 to 1.0)


@dataclass
class CaringResponse:
    """Complete caring response plan"""
    assessment: CaringAssessment
    response_type: ResponseType
    response_context: ResponseContext
    persona_config: ResponsePersonality
    safety_protocols: List[str]
    cultural_adaptations: List[str]
    follow_up_recommendations: List[str]
    monitoring_flags: List[str]


class CaringIntelligence:
    """
    The caring intelligence system that determines appropriate responses
    based on deep understanding of user needs and context.
    
    Core Principles:
    - Safety and wellbeing above all else
    - Meet people where they are
    - Honor user autonomy and choice
    - Provide trauma-informed care
    - Respect cultural context
    - Foster healing and growth
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Assessment algorithms
        self.care_need_patterns = self._initialize_care_need_patterns()
        self.intent_detection_patterns = self._initialize_intent_detection_patterns()
        self.emotional_state_indicators = self._initialize_emotional_state_indicators()
        self.safety_assessment_criteria = self._initialize_safety_assessment_criteria()
        self.vulnerability_indicators = self._initialize_vulnerability_indicators()
        
        # Response decision trees
        self.response_decision_matrix = self._initialize_response_decision_matrix()
        self.cultural_consideration_map = self._initialize_cultural_consideration_map()
        
        logger.info("Caring Intelligence initialized")
    
    def _initialize_care_need_patterns(self) -> Dict[CareNeeds, Dict[str, Any]]:
        """Initialize patterns for detecting different care needs"""
        
        return {
            CareNeeds.IMMEDIATE_SAFETY: {
                'keywords': [
                    'suicide', 'kill myself', 'end it all', 'don\'t want to live',
                    'hurt myself', 'self harm', 'emergency', 'crisis'
                ],
                'phrases': [
                    'want to die', 'no point in living', 'better off dead',
                    'can\'t go on', 'want to disappear', 'planning to hurt'
                ],
                'urgency_multiplier': 3.0
            },
            
            CareNeeds.EMOTIONAL_VALIDATION: {
                'keywords': [
                    'feel', 'feeling', 'emotional', 'upset', 'sad', 'angry',
                    'frustrated', 'confused', 'overwhelmed', 'hurt'
                ],
                'phrases': [
                    'no one understands', 'feeling alone', 'am i crazy',
                    'is this normal', 'feel so', 'feeling like'
                ],
                'context_indicators': ['sharing emotions', 'expressing feelings']
            },
            
            CareNeeds.GENTLE_GUIDANCE: {
                'keywords': [
                    'help', 'advice', 'guidance', 'direction', 'what should',
                    'how do i', 'don\'t know', 'confused', 'stuck'
                ],
                'phrases': [
                    'what should i do', 'how do i handle', 'need direction',
                    'not sure how', 'seeking advice', 'feel lost'
                ],
                'context_indicators': ['seeking direction', 'asking for help']
            },
            
            CareNeeds.HOPE_AND_ENCOURAGEMENT: {
                'keywords': [
                    'hopeless', 'despair', 'giving up', 'pointless', 'worthless',
                    'can\'t', 'impossible', 'never', 'always'
                ],
                'phrases': [
                    'no hope', 'nothing will change', 'always be like this',
                    'can\'t get better', 'no point trying', 'will never'
                ],
                'context_indicators': ['expressing despair', 'loss of hope']
            },
            
            CareNeeds.CELEBRATION: {
                'keywords': [
                    'progress', 'achievement', 'milestone', 'success', 'proud',
                    'accomplished', 'breakthrough', 'victory', 'win'
                ],
                'phrases': [
                    'i did it', 'finally achieved', 'made progress', 'so proud',
                    'big step', 'major breakthrough', 'reached my goal'
                ],
                'context_indicators': ['sharing achievement', 'positive progress']
            },
            
            CareNeeds.TRAUMA_INFORMED_CARE: {
                'keywords': [
                    'trauma', 'abuse', 'assault', 'ptsd', 'flashback',
                    'triggered', 'survivor', 'violence', 'attack'
                ],
                'phrases': [
                    'traumatic experience', 'been through', 'happened to me',
                    'can\'t forget', 'keeps coming back', 'reminds me of'
                ],
                'context_indicators': ['trauma disclosure', 'triggering content']
            },
            
            CareNeeds.LOVING_PRESENCE: {
                'keywords': [
                    'alone', 'lonely', 'isolated', 'nobody', 'empty',
                    'disconnected', 'abandoned', 'forgotten'
                ],
                'phrases': [
                    'feel so alone', 'nobody cares', 'all by myself',
                    'no one there', 'completely isolated', 'feel invisible'
                ],
                'context_indicators': ['expressing loneliness', 'need for connection']
            }
        }
    
    def _initialize_intent_detection_patterns(self) -> Dict[InteractionIntent, Dict[str, Any]]:
        """Initialize patterns for detecting user interaction intent"""
        
        return {
            InteractionIntent.SEEKING_SUPPORT: {
                'opening_patterns': [
                    'i need', 'could use', 'hoping for', 'looking for support',
                    'struggling with', 'having trouble', 'going through'
                ],
                'emotional_indicators': ['sad', 'upset', 'overwhelmed', 'stressed']
            },
            
            InteractionIntent.ASKING_FOR_GUIDANCE: {
                'question_patterns': [
                    'what should i', 'how do i', 'can you help me', 'advice on',
                    'what would you', 'how would you', 'suggestions for'
                ],
                'guidance_indicators': ['decision', 'choice', 'path', 'direction']
            },
            
            InteractionIntent.SHARING_PROGRESS: {
                'sharing_patterns': [
                    'i wanted to share', 'proud to say', 'excited to tell',
                    'good news', 'update on', 'progress report'
                ],
                'achievement_indicators': ['accomplished', 'achieved', 'succeeded', 'completed']
            },
            
            InteractionIntent.EXPRESSING_STRUGGLE: {
                'struggle_patterns': [
                    'struggling with', 'having a hard time', 'really difficult',
                    'can\'t handle', 'overwhelmed by', 'falling apart'
                ],
                'distress_indicators': ['difficult', 'hard', 'challenging', 'overwhelming']
            },
            
            InteractionIntent.IN_CRISIS: {
                'crisis_patterns': [
                    'emergency', 'urgent', 'immediate help', 'crisis',
                    'can\'t take it', 'breaking point', 'desperate'
                ],
                'safety_indicators': ['danger', 'harm', 'hurt', 'end', 'suicide']
            },
            
            InteractionIntent.SETTING_BOUNDARIES: {
                'boundary_patterns': [
                    'need to stop', 'not comfortable', 'don\'t want to',
                    'boundary', 'limit', 'enough for now', 'pause'
                ],
                'autonomy_indicators': ['choice', 'control', 'my decision', 'my pace']
            }
        }
    
    def _initialize_emotional_state_indicators(self) -> Dict[str, List[str]]:
        """Initialize indicators for different emotional states"""
        
        return {
            'distressed': [
                'upset', 'distressed', 'distraught', 'agitated', 'troubled',
                'tormented', 'anguished', 'devastated', 'shattered'
            ],
            'overwhelmed': [
                'overwhelmed', 'swamped', 'drowning', 'buried', 'crushed',
                'too much', 'can\'t handle', 'breaking point'
            ],
            'anxious': [
                'anxious', 'worried', 'nervous', 'scared', 'fearful',
                'panicked', 'terrified', 'afraid', 'apprehensive'
            ],
            'depressed': [
                'depressed', 'sad', 'down', 'blue', 'hopeless',
                'empty', 'numb', 'lifeless', 'despairing'
            ],
            'angry': [
                'angry', 'mad', 'furious', 'rage', 'livid',
                'frustrated', 'irritated', 'annoyed', 'outraged'
            ],
            'confused': [
                'confused', 'lost', 'uncertain', 'unclear', 'bewildered',
                'puzzled', 'perplexed', 'mixed up', 'disoriented'
            ],
            'hopeful': [
                'hopeful', 'optimistic', 'positive', 'encouraged', 'uplifted',
                'inspired', 'motivated', 'confident', 'believing'
            ],
            'proud': [
                'proud', 'accomplished', 'satisfied', 'successful', 'achieved',
                'victorious', 'triumphant', 'fulfilled', 'gratified'
            ]
        }
    
    def _initialize_safety_assessment_criteria(self) -> Dict[CareLevel, Dict[str, Any]]:
        """Initialize criteria for safety level assessment"""
        
        return {
            CareLevel.CRISIS: {
                'indicators': [
                    'suicide ideation', 'self harm plans', 'immediate danger',
                    'acute psychosis', 'severe dissociation', 'active crisis'
                ],
                'response_requirements': [
                    'immediate_intervention', 'professional_referral', 'safety_planning'
                ]
            },
            
            CareLevel.HIGH: {
                'indicators': [
                    'severe distress', 'overwhelming emotions', 'concerning behavior',
                    'significant impairment', 'high vulnerability', 'acute symptoms'
                ],
                'response_requirements': [
                    'enhanced_support', 'frequent_check_ins', 'resource_provision'
                ]
            },
            
            CareLevel.MODERATE: {
                'indicators': [
                    'noticeable distress', 'coping challenges', 'emotional difficulties',
                    'mild impairment', 'struggling but stable', 'manageable symptoms'
                ],
                'response_requirements': [
                    'supportive_care', 'skill_building', 'encouragement'
                ]
            },
            
            CareLevel.LOW: {
                'indicators': [
                    'stable functioning', 'good coping', 'positive progress',
                    'resilient responses', 'healthy engagement', 'growth-focused'
                ],
                'response_requirements': [
                    'maintenance_support', 'celebration', 'continued_growth'
                ]
            }
        }
    
    def _initialize_vulnerability_indicators(self) -> List[str]:
        """Initialize indicators of user vulnerability"""
        
        return [
            'recently traumatized', 'in crisis', 'isolated', 'overwhelmed',
            'new to healing', 'multiple stressors', 'limited support',
            'recent loss', 'major life change', 'health issues',
            'financial stress', 'relationship problems', 'work stress',
            'substance use', 'mental health episode', 'anniversary reactions'
        ]
    
    def _initialize_response_decision_matrix(self) -> Dict[str, Dict[str, Any]]:
        """Initialize decision matrix for response selection"""
        
        return {
            'crisis_immediate_safety': {
                'care_need': CareNeeds.IMMEDIATE_SAFETY,
                'response_type': ResponseType.CRISIS_SUPPORT,
                'response_context': ResponseContext.CRISIS_INTERVENTION,
                'persona_traits': ['protective', 'calm', 'professional'],
                'priority': 1  # Highest priority
            },
            
            'high_distress_validation': {
                'care_need': CareNeeds.EMOTIONAL_VALIDATION,
                'safety_level': CareLevel.HIGH,
                'response_type': ResponseType.EMPATHETIC_VALIDATION,
                'response_context': ResponseContext.EMOTIONAL_SUPPORT,
                'persona_traits': ['compassionate', 'gentle', 'understanding'],
                'priority': 2
            },
            
            'celebration_recognition': {
                'care_need': CareNeeds.CELEBRATION,
                'response_type': ResponseType.CELEBRATION,
                'response_context': ResponseContext.CELEBRATION_MOMENT,
                'persona_traits': ['encouraging', 'warm', 'joyful'],
                'priority': 3
            },
            
            'guidance_seeking': {
                'care_need': CareNeeds.GENTLE_GUIDANCE,
                'response_type': ResponseType.GENTLE_GUIDANCE,
                'response_context': ResponseContext.GUIDANCE_REQUEST,
                'persona_traits': ['wise', 'patient', 'supportive'],
                'priority': 4
            },
            
            'hope_needed': {
                'care_need': CareNeeds.HOPE_AND_ENCOURAGEMENT,
                'response_type': ResponseType.HOPE_OFFERING,
                'response_context': ResponseContext.EMOTIONAL_SUPPORT,
                'persona_traits': ['hopeful', 'encouraging', 'inspiring'],
                'priority': 5
            },
            
            'trauma_sensitive': {
                'care_need': CareNeeds.TRAUMA_INFORMED_CARE,
                'response_type': ResponseType.EMPATHETIC_VALIDATION,
                'response_context': ResponseContext.EMOTIONAL_SUPPORT,
                'persona_traits': ['gentle', 'protective', 'trauma_informed'],
                'priority': 2  # High priority
            }
        }
    
    def _initialize_cultural_consideration_map(self) -> Dict[str, List[str]]:
        """Initialize cultural considerations mapping"""
        
        return {
            'religious_context': [
                'respect_spiritual_beliefs', 'inclusive_language',
                'avoid_conflicting_guidance', 'honor_faith_journey'
            ],
            'cultural_background': [
                'respect_cultural_values', 'consider_family_dynamics',
                'acknowledge_cultural_strengths', 'adapt_communication_style'
            ],
            'marginalized_identity': [
                'affirm_identity', 'acknowledge_unique_challenges',
                'avoid_assumptions', 'provide_inclusive_support'
            ]
        }
    
    async def assess_caring_needs(self, user_input: str,
                                user_context: Dict[str, Any],
                                interaction_history: List[Dict[str, Any]]) -> CaringAssessment:
        """
        Assess user's caring needs based on input and context.
        
        Args:
            user_input: What the user said/wrote
            user_context: User's broader context
            interaction_history: Previous interactions
        """
        try:
            # Detect primary care need
            primary_care_need = await self._detect_primary_care_need(user_input, user_context)
            
            # Detect secondary care needs
            secondary_care_needs = await self._detect_secondary_care_needs(user_input, user_context)
            
            # Detect interaction intent
            interaction_intent = await self._detect_interaction_intent(user_input)
            
            # Assess emotional state
            emotional_state = await self._assess_emotional_state(user_input, user_context)
            
            # Assess safety level
            safety_level = await self._assess_safety_level(user_input, user_context)
            
            # Identify vulnerability indicators
            vulnerability_indicators = await self._identify_vulnerability_indicators(user_context)
            
            # Identify strength indicators
            strength_indicators = await self._identify_strength_indicators(user_input, user_context)
            
            # Assess cultural considerations
            cultural_considerations = await self._assess_cultural_considerations(user_context)
            
            # Assess trauma sensitivity needs
            trauma_sensitivity_needed = await self._assess_trauma_sensitivity_needs(user_input, user_context)
            
            # Calculate urgency level
            urgency_level = await self._calculate_urgency_level(primary_care_need, safety_level)
            
            # Assess support capacity
            support_capacity = await self._assess_support_capacity(user_context, emotional_state)
            
            assessment = CaringAssessment(
                primary_care_need=primary_care_need,
                secondary_care_needs=secondary_care_needs,
                interaction_intent=interaction_intent,
                emotional_state=emotional_state,
                safety_level=safety_level,
                vulnerability_indicators=vulnerability_indicators,
                strength_indicators=strength_indicators,
                cultural_considerations=cultural_considerations,
                trauma_sensitivity_needed=trauma_sensitivity_needed,
                urgency_level=urgency_level,
                support_capacity=support_capacity
            )
            
            logger.debug(f"Caring assessment completed: {primary_care_need.value} at {safety_level.value} level")
            
            return assessment
            
        except Exception as e:
            logger.error(f"Error in caring needs assessment: {e}")
            return await self._get_default_assessment()
    
    async def generate_caring_response_plan(self, assessment: CaringAssessment,
                                          user_context: Dict[str, Any],
                                          persona: Any) -> CaringResponse:
        """Generate complete caring response plan based on assessment"""
        
        try:
            # Determine response type and context
            response_type, response_context = await self._determine_response_approach(assessment)
            
            # Generate persona configuration
            persona_config = await persona.generate_persona_for_context(
                user_context, assessment.interaction_intent.value, assessment.emotional_state
            )
            
            # Determine safety protocols
            safety_protocols = await self._determine_safety_protocols(assessment)
            
            # Determine cultural adaptations
            cultural_adaptations = await self._determine_cultural_adaptations(assessment)
            
            # Generate follow-up recommendations
            follow_up_recommendations = await self._generate_follow_up_recommendations(assessment)
            
            # Set monitoring flags
            monitoring_flags = await self._set_monitoring_flags(assessment)
            
            caring_response = CaringResponse(
                assessment=assessment,
                response_type=response_type,
                response_context=response_context,
                persona_config=persona_config,
                safety_protocols=safety_protocols,
                cultural_adaptations=cultural_adaptations,
                follow_up_recommendations=follow_up_recommendations,
                monitoring_flags=monitoring_flags
            )
            
            logger.debug(f"Generated caring response plan: {response_type.value} with {persona_config.primary_trait.value} persona")
            
            return caring_response
            
        except Exception as e:
            logger.error(f"Error generating caring response plan: {e}")
            return await self._get_default_response_plan(assessment)
    
    async def _detect_primary_care_need(self, user_input: str, user_context: Dict[str, Any]) -> CareNeeds:
        """Detect the primary care need from user input"""
        
        user_input_lower = user_input.lower()
        care_need_scores = {}
        
        # Score each care need based on pattern matching
        for care_need, patterns in self.care_need_patterns.items():
            score = 0.0
            
            # Check keywords
            for keyword in patterns.get('keywords', []):
                if keyword in user_input_lower:
                    score += 2.0
            
            # Check phrases
            for phrase in patterns.get('phrases', []):
                if phrase in user_input_lower:
                    score += 3.0
            
            # Apply urgency multiplier if present
            urgency_multiplier = patterns.get('urgency_multiplier', 1.0)
            score *= urgency_multiplier
            
            if score > 0:
                care_need_scores[care_need] = score
        
        # Return highest scoring care need, or default
        if care_need_scores:
            return max(care_need_scores.items(), key=lambda x: x[1])[0]
        else:
            return CareNeeds.EMOTIONAL_VALIDATION  # Default
    
    async def _detect_secondary_care_needs(self, user_input: str, user_context: Dict[str, Any]) -> List[CareNeeds]:
        """Detect secondary care needs"""
        
        secondary_needs = []
        primary_need = await self._detect_primary_care_need(user_input, user_context)
        
        # Look for additional needs
        user_input_lower = user_input.lower()
        
        for care_need, patterns in self.care_need_patterns.items():
            if care_need == primary_need:
                continue
            
            # Check for secondary indicators
            keyword_matches = sum(1 for keyword in patterns.get('keywords', []) 
                                if keyword in user_input_lower)
            
            if keyword_matches >= 1:  # Lower threshold for secondary needs
                secondary_needs.append(care_need)
        
        return secondary_needs[:2]  # Limit to top 2 secondary needs
    
    async def _detect_interaction_intent(self, user_input: str) -> InteractionIntent:
        """Detect user's intent in the interaction"""
        
        user_input_lower = user_input.lower()
        intent_scores = {}
        
        for intent, patterns in self.intent_detection_patterns.items():
            score = 0.0
            
            # Check opening patterns
            for pattern in patterns.get('opening_patterns', []):
                if pattern in user_input_lower:
                    score += 2.0
            
            # Check question patterns
            for pattern in patterns.get('question_patterns', []):
                if pattern in user_input_lower:
                    score += 2.0
            
            # Check indicator words
            for indicator_type in ['emotional_indicators', 'guidance_indicators', 
                                 'achievement_indicators', 'distress_indicators',
                                 'safety_indicators', 'autonomy_indicators']:
                for indicator in patterns.get(indicator_type, []):
                    if indicator in user_input_lower:
                        score += 1.0
            
            if score > 0:
                intent_scores[intent] = score
        
        # Return highest scoring intent, or default
        if intent_scores:
            return max(intent_scores.items(), key=lambda x: x[1])[0]
        else:
            return InteractionIntent.SEEKING_SUPPORT  # Default
    
    async def _assess_emotional_state(self, user_input: str, user_context: Dict[str, Any]) -> str:
        """Assess user's current emotional state"""
        
        user_input_lower = user_input.lower()
        emotional_scores = {}
        
        for emotion, indicators in self.emotional_state_indicators.items():
            score = sum(1 for indicator in indicators if indicator in user_input_lower)
            if score > 0:
                emotional_scores[emotion] = score
        
        # Return highest scoring emotion, or neutral
        if emotional_scores:
            return max(emotional_scores.items(), key=lambda x: x[1])[0]
        else:
            return 'neutral'
    
    async def _assess_safety_level(self, user_input: str, user_context: Dict[str, Any]) -> CareLevel:
        """Assess user's safety level"""
        
        user_input_lower = user_input.lower()
        
        # Check for crisis indicators
        crisis_indicators = self.safety_assessment_criteria[CareLevel.CRISIS]['indicators']
        for indicator in crisis_indicators:
            if any(word in user_input_lower for word in indicator.split()):
                return CareLevel.CRISIS
        
        # Check for high concern indicators
        high_indicators = self.safety_assessment_criteria[CareLevel.HIGH]['indicators']
        for indicator in high_indicators:
            if any(word in user_input_lower for word in indicator.split()):
                return CareLevel.HIGH
        
        # Check context for additional risk factors
        if user_context.get('recent_crisis', False):
            return CareLevel.HIGH
        
        if user_context.get('multiple_stressors', False):
            return CareLevel.MODERATE
        
        return CareLevel.LOW  # Default to low if no concerning indicators
    
    async def _identify_vulnerability_indicators(self, user_context: Dict[str, Any]) -> List[str]:
        """Identify vulnerability indicators from context"""
        
        vulnerabilities = []
        
        for indicator in self.vulnerability_indicators:
            context_key = indicator.replace(' ', '_')
            if user_context.get(context_key, False):
                vulnerabilities.append(indicator)
        
        return vulnerabilities
    
    async def _identify_strength_indicators(self, user_input: str, user_context: Dict[str, Any]) -> List[str]:
        """Identify strength indicators"""
        
        strength_indicators = []
        user_input_lower = user_input.lower()
        
        # Strength keywords
        strength_keywords = [
            'proud', 'accomplished', 'achieved', 'overcame', 'survived',
            'strong', 'resilient', 'capable', 'determined', 'courageous'
        ]
        
        for keyword in strength_keywords:
            if keyword in user_input_lower:
                strength_indicators.append(f"demonstrates_{keyword}")
        
        # Context-based strengths
        if user_context.get('consistent_self_care', False):
            strength_indicators.append('consistent_self_care')
        
        if user_context.get('seeking_support', False):
            strength_indicators.append('help_seeking_behavior')
        
        return strength_indicators
    
    async def _assess_cultural_considerations(self, user_context: Dict[str, Any]) -> List[str]:
        """Assess cultural considerations needed"""
        
        considerations = []
        
        if user_context.get('religious_context'):
            considerations.extend(self.cultural_consideration_map['religious_context'])
        
        if user_context.get('cultural_background'):
            considerations.extend(self.cultural_consideration_map['cultural_background'])
        
        if user_context.get('marginalized_identity'):
            considerations.extend(self.cultural_consideration_map['marginalized_identity'])
        
        return considerations
    
    async def _assess_trauma_sensitivity_needs(self, user_input: str, user_context: Dict[str, Any]) -> bool:
        """Assess if trauma-informed approach is needed"""
        
        # Check for trauma-related content
        trauma_keywords = ['trauma', 'abuse', 'assault', 'ptsd', 'triggered', 'flashback']
        user_input_lower = user_input.lower()
        
        if any(keyword in user_input_lower for keyword in trauma_keywords):
            return True
        
        # Check context
        if user_context.get('trauma_history', False):
            return True
        
        if user_context.get('trauma_sensitive', False):
            return True
        
        return False
    
    async def _calculate_urgency_level(self, care_need: CareNeeds, safety_level: CareLevel) -> float:
        """Calculate urgency level (0.0 to 1.0)"""
        
        base_urgency = 0.3  # Default
        
        # Urgency based on care need
        care_need_urgency = {
            CareNeeds.IMMEDIATE_SAFETY: 1.0,
            CareNeeds.TRAUMA_INFORMED_CARE: 0.8,
            CareNeeds.EMOTIONAL_VALIDATION: 0.6,
            CareNeeds.HOPE_AND_ENCOURAGEMENT: 0.7,
            CareNeeds.GENTLE_GUIDANCE: 0.4,
            CareNeeds.CELEBRATION: 0.2,
            CareNeeds.LOVING_PRESENCE: 0.5
        }
        
        urgency = care_need_urgency.get(care_need, base_urgency)
        
        # Adjust based on safety level
        safety_urgency = {
            CareLevel.CRISIS: 1.0,
            CareLevel.HIGH: 0.8,
            CareLevel.MODERATE: 0.5,
            CareLevel.LOW: 0.2
        }
        
        safety_adjustment = safety_urgency.get(safety_level, 0.3)
        
        # Take the maximum of care need and safety urgency
        return max(urgency, safety_adjustment)
    
    async def _assess_support_capacity(self, user_context: Dict[str, Any], emotional_state: str) -> float:
        """Assess user's capacity to receive support (0.0 to 1.0)"""
        
        base_capacity = 0.7  # Default
        
        # Adjust based on emotional state
        if emotional_state in ['overwhelmed', 'crisis']:
            base_capacity -= 0.3
        elif emotional_state in ['hopeful', 'proud']:
            base_capacity += 0.2
        
        # Adjust based on context
        if user_context.get('emotionally_overwhelmed', False):
            base_capacity -= 0.2
        
        if user_context.get('strong_support_network', False):
            base_capacity += 0.1
        
        return max(0.1, min(1.0, base_capacity))
    
    async def _determine_response_approach(self, assessment: CaringAssessment) -> Tuple[ResponseType, ResponseContext]:
        """Determine response type and context based on assessment"""
        
        # Priority-based decision making
        if assessment.primary_care_need == CareNeeds.IMMEDIATE_SAFETY:
            return ResponseType.CRISIS_SUPPORT, ResponseContext.CRISIS_INTERVENTION
        
        elif assessment.primary_care_need == CareNeeds.CELEBRATION:
            return ResponseType.CELEBRATION, ResponseContext.CELEBRATION_MOMENT
        
        elif assessment.primary_care_need == CareNeeds.EMOTIONAL_VALIDATION:
            return ResponseType.EMPATHETIC_VALIDATION, ResponseContext.EMOTIONAL_SUPPORT
        
        elif assessment.primary_care_need == CareNeeds.GENTLE_GUIDANCE:
            return ResponseType.GENTLE_GUIDANCE, ResponseContext.GUIDANCE_REQUEST
        
        elif assessment.primary_care_need == CareNeeds.HOPE_AND_ENCOURAGEMENT:
            return ResponseType.HOPE_OFFERING, ResponseContext.EMOTIONAL_SUPPORT
        
        elif assessment.primary_care_need == CareNeeds.TRAUMA_INFORMED_CARE:
            return ResponseType.EMPATHETIC_VALIDATION, ResponseContext.EMOTIONAL_SUPPORT
        
        elif assessment.primary_care_need == CareNeeds.LOVING_PRESENCE:
            return ResponseType.LOVING_REMINDER, ResponseContext.EMOTIONAL_SUPPORT
        
        else:
            return ResponseType.EMPATHETIC_VALIDATION, ResponseContext.ONGOING_CONVERSATION
    
    async def _determine_safety_protocols(self, assessment: CaringAssessment) -> List[str]:
        """Determine safety protocols needed"""
        
        protocols = []
        
        if assessment.safety_level == CareLevel.CRISIS:
            protocols.extend([
                'immediate_safety_assessment',
                'crisis_resource_provision',
                'professional_referral_urgent',
                'follow_up_required'
            ])
        
        elif assessment.safety_level == CareLevel.HIGH:
            protocols.extend([
                'enhanced_monitoring',
                'resource_provision',
                'safety_check_recommended'
            ])
        
        if assessment.trauma_sensitivity_needed:
            protocols.append('trauma_informed_language')
        
        return protocols
    
    async def _determine_cultural_adaptations(self, assessment: CaringAssessment) -> List[str]:
        """Determine cultural adaptations needed"""
        
        return assessment.cultural_considerations
    
    async def _generate_follow_up_recommendations(self, assessment: CaringAssessment) -> List[str]:
        """Generate follow-up recommendations"""
        
        recommendations = []
        
        if assessment.urgency_level > 0.7:
            recommendations.append('follow_up_within_24_hours')
        elif assessment.urgency_level > 0.5:
            recommendations.append('follow_up_within_week')
        
        if assessment.primary_care_need == CareNeeds.GENTLE_GUIDANCE:
            recommendations.append('offer_resources')
        
        if assessment.primary_care_need == CareNeeds.CELEBRATION:
            recommendations.append('acknowledge_progress_in_future')
        
        return recommendations
    
    async def _set_monitoring_flags(self, assessment: CaringAssessment) -> List[str]:
        """Set monitoring flags for this interaction"""
        
        flags = []
        
        if assessment.safety_level in [CareLevel.CRISIS, CareLevel.HIGH]:
            flags.append('safety_monitoring')
        
        if assessment.trauma_sensitivity_needed:
            flags.append('trauma_sensitive')
        
        if assessment.vulnerability_indicators:
            flags.append('vulnerability_present')
        
        return flags
    
    async def _get_default_assessment(self) -> CaringAssessment:
        """Get default assessment when assessment fails"""
        
        return CaringAssessment(
            primary_care_need=CareNeeds.EMOTIONAL_VALIDATION,
            secondary_care_needs=[],
            interaction_intent=InteractionIntent.SEEKING_SUPPORT,
            emotional_state='neutral',
            safety_level=CareLevel.MODERATE,
            vulnerability_indicators=[],
            strength_indicators=[],
            cultural_considerations=[],
            trauma_sensitivity_needed=True,  # Err on side of caution
            urgency_level=0.5,
            support_capacity=0.7
        )
    
    async def _get_default_response_plan(self, assessment: CaringAssessment) -> CaringResponse:
        """Get default response plan when generation fails"""
        
        from .persona import PersonalityTrait, CommunicationStyle, EmotionalResonance, ResponsePersonality
        
        return CaringResponse(
            assessment=assessment,
            response_type=ResponseType.EMPATHETIC_VALIDATION,
            response_context=ResponseContext.EMOTIONAL_SUPPORT,
            persona_config=ResponsePersonality(
                primary_trait=PersonalityTrait.COMPASSIONATE,
                communication_style=CommunicationStyle.WARM_COMPANION,
                emotional_resonance=EmotionalResonance.GENTLE_VALIDATION,
                warmth_level=0.7,
                formality_level=0.3,
                wisdom_integration=0.5,
                protective_mode=True,
                cultural_sensitivity=0.8
            ),
            safety_protocols=['trauma_informed_language'],
            cultural_adaptations=[],
            follow_up_recommendations=[],
            monitoring_flags=['safety_monitoring']
        )