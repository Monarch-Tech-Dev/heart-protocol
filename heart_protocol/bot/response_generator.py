"""
Response Generator for Monarch Bot

Generates caring, contextually appropriate responses that embody the
Heart Protocol values and serve healing through AI interaction.
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
import logging
import random

from .persona import MonarchPersona, ResponsePersonality, PersonalityTrait, CommunicationStyle, EmotionalResonance
from ..core.base import CareLevel

logger = logging.getLogger(__name__)


class ResponseType(Enum):
    """Types of responses the bot can generate"""
    EMPATHETIC_VALIDATION = "empathetic_validation"       # Validates feelings and experience
    GENTLE_GUIDANCE = "gentle_guidance"                   # Offers soft guidance or suggestions
    WISDOM_SHARING = "wisdom_sharing"                     # Shares relevant wisdom or insights
    HOPE_OFFERING = "hope_offering"                       # Provides hope and encouragement
    SAFETY_CHECKING = "safety_checking"                   # Checks on user safety and wellbeing
    CELEBRATION = "celebration"                           # Celebrates progress or achievements
    RESOURCE_OFFERING = "resource_offering"               # Suggests helpful resources
    BOUNDARY_RESPECTING = "boundary_respecting"           # Respects user boundaries and limits
    CRISIS_SUPPORT = "crisis_support"                     # Provides crisis support and safety
    LOVING_REMINDER = "loving_reminder"                   # Reminds user of their worth and value


class ResponseContext(Enum):
    """Context in which response is being generated"""
    INITIAL_GREETING = "initial_greeting"                 # First interaction with user
    ONGOING_CONVERSATION = "ongoing_conversation"         # Continuing conversation
    CRISIS_INTERVENTION = "crisis_intervention"           # Crisis or emergency situation
    CELEBRATION_MOMENT = "celebration_moment"             # Celebrating user progress
    GUIDANCE_REQUEST = "guidance_request"                 # User seeking guidance
    EMOTIONAL_SUPPORT = "emotional_support"               # User needs emotional support
    WISDOM_SHARING = "wisdom_sharing"                     # Sharing community wisdom
    SAFETY_CHECK = "safety_check"                         # Checking on user safety
    GOODBYE = "goodbye"                                   # Ending interaction
    BOUNDARY_SETTING = "boundary_setting"                # User setting boundaries


@dataclass
class ResponseComponents:
    """Components that make up a complete response"""
    opening: str                        # How to open the response
    body: str                          # Main content of response
    validation: Optional[str]          # Validation of user's experience
    wisdom: Optional[str]              # Relevant wisdom or insight
    encouragement: Optional[str]       # Encouragement or hope
    practical_support: Optional[str]   # Practical suggestions or resources
    safety_note: Optional[str]         # Safety considerations
    closing: str                       # How to close the response
    emotional_tone: str                # Overall emotional tone


class ResponseGenerator:
    """
    Generates caring, trauma-informed responses for the Monarch Bot.
    
    Core Principles:
    - Every response serves love and healing
    - Trauma-informed language throughout
    - User autonomy and choice paramount
    - Culturally sensitive and inclusive
    - Safety and wellbeing prioritized
    - Authentic and genuine connection
    """
    
    def __init__(self, config: Dict[str, Any], persona: MonarchPersona):
        self.config = config
        self.persona = persona
        
        # Response generation components
        self.response_templates = self._initialize_response_templates()
        self.validation_phrases = self._initialize_validation_phrases()
        self.encouragement_bank = self._initialize_encouragement_bank()
        self.safety_protocols = self._initialize_safety_protocols()
        self.cultural_adaptations = self._initialize_cultural_adaptations()
        
        # Response tracking
        self.response_history = {}  # user_id -> List[response_data]
        self.successful_patterns = {}  # pattern_id -> effectiveness_data
        
        logger.info("Response Generator initialized")
    
    def _initialize_response_templates(self) -> Dict[ResponseType, Dict[str, Any]]:
        """Initialize response templates for different types"""
        
        return {
            ResponseType.EMPATHETIC_VALIDATION: {
                'openings': [
                    "I can sense how deeply this affects you.",
                    "What you're experiencing sounds incredibly difficult.",
                    "I hear the weight of what you're carrying.",
                    "Your feelings about this make complete sense."
                ],
                'body_structures': [
                    "validation + understanding + normalization",
                    "reflection + empathy + solidarity",
                    "acknowledgment + compassion + support"
                ],
                'closings': [
                    "Your experience is valid and important.",
                    "You're not alone in feeling this way.",
                    "Thank you for trusting me with this."
                ]
            },
            
            ResponseType.GENTLE_GUIDANCE: {
                'openings': [
                    "I wonder if it might help to consider...",
                    "One possibility that comes to mind is...",
                    "Perhaps we could explore...",
                    "What feels true for you might be..."
                ],
                'body_structures': [
                    "suggestion + rationale + choice_emphasis",
                    "option_1 + option_2 + autonomy_reminder",
                    "gentle_direction + safety_check + pace_setting"
                ],
                'closings': [
                    "But you know yourself best.",
                    "Trust your instincts about what feels right.",
                    "The choice is completely yours."
                ]
            },
            
            ResponseType.WISDOM_SHARING: {
                'openings': [
                    "Something our community has learned is...",
                    "In my understanding...",
                    "What many have discovered is...",
                    "Wisdom from those who've walked this path suggests..."
                ],
                'body_structures': [
                    "wisdom_statement + context + application",
                    "insight + evidence + adaptation_note",
                    "truth + explanation + personal_relevance"
                ],
                'closings': [
                    "Take what resonates and leave what doesn't.",
                    "This wisdom may or may not apply to your situation.",
                    "Every journey is unique - adapt this to your needs."
                ]
            },
            
            ResponseType.HOPE_OFFERING: {
                'openings': [
                    "Even in this difficulty, there is hope.",
                    "What I want you to know is...",
                    "In the midst of this struggle...",
                    "Despite how things feel right now..."
                ],
                'body_structures': [
                    "hope_statement + evidence + encouragement",
                    "possibility + examples + strength_recognition",
                    "future_vision + present_support + growth_potential"
                ],
                'closings': [
                    "Hope is always available to you.",
                    "Your story is still being written.",
                    "Healing and growth remain possible."
                ]
            },
            
            ResponseType.SAFETY_CHECKING: {
                'openings': [
                    "I want to check in on your safety and wellbeing.",
                    "Your safety is important to me.",
                    "I'm concerned about your wellbeing right now.",
                    "Let's make sure you're safe first."
                ],
                'body_structures': [
                    "safety_question + resource_offer + support_assurance",
                    "concern_expression + help_options + immediate_steps",
                    "assessment + intervention + follow_up"
                ],
                'closings': [
                    "You deserve to be safe and supported.",
                    "Please reach out for help if you need it.",
                    "Your safety and life matter deeply."
                ]
            },
            
            ResponseType.CELEBRATION: {
                'openings': [
                    "I'm so proud of you for this!",
                    "What a beautiful step forward!",
                    "This deserves celebration!",
                    "Look at this growth you're showing!"
                ],
                'body_structures': [
                    "achievement_recognition + significance + encouragement",
                    "progress_celebration + strength_highlighting + future_hope",
                    "milestone_honoring + journey_acknowledgment + support"
                ],
                'closings': [
                    "You should feel proud of this progress.",
                    "This shows your incredible strength and courage.",
                    "Keep going - you're doing beautifully."
                ]
            },
            
            ResponseType.CRISIS_SUPPORT: {
                'openings': [
                    "I'm here with you in this crisis.",
                    "Your safety is the most important thing right now.",
                    "Let's focus on getting you the support you need.",
                    "You don't have to face this alone."
                ],
                'body_structures': [
                    "immediate_safety + resource_connection + support_assurance",
                    "crisis_validation + help_options + safety_planning",
                    "emergency_response + professional_referral + ongoing_support"
                ],
                'closings': [
                    "Please reach out for professional help immediately.",
                    "Your life has value and meaning.",
                    "Help is available and you deserve support."
                ]
            }
        }
    
    def _initialize_validation_phrases(self) -> Dict[str, List[str]]:
        """Initialize validation phrase banks"""
        
        return {
            'emotional_validation': [
                "Your feelings are completely valid.",
                "Anyone would feel this way in your situation.",
                "It makes perfect sense that you're experiencing this.",
                "Your emotional response is natural and understandable.",
                "There's no wrong way to feel about this."
            ],
            
            'experience_validation': [
                "What you went through was real and significant.",
                "Your experience matters and deserves to be honored.",
                "I believe you and I see the impact this has had.",
                "Your story is important and valid.",
                "Thank you for trusting me with your experience."
            ],
            
            'struggle_validation': [
                "This is genuinely difficult and challenging.",
                "You're facing something really hard right now.",
                "The struggle you're experiencing is real.",
                "It takes courage to keep going through this.",
                "Your challenges are valid and significant."
            ],
            
            'growth_validation': [
                "The progress you're making, however small, matters.",
                "Every step forward is meaningful and important.",
                "Your growth journey is valid at whatever pace it takes.",
                "You're doing important work on yourself.",
                "Your healing process deserves recognition and respect."
            ]
        }
    
    def _initialize_encouragement_bank(self) -> Dict[str, List[str]]:
        """Initialize encouragement phrase banks"""
        
        return {
            'strength_recognition': [
                "You're stronger than you realize.",
                "I see incredible resilience in you.",
                "Your courage in facing this is remarkable.",
                "You have inner strength that has carried you this far.",
                "The fact that you're here shows your determination."
            ],
            
            'hope_anchors': [
                "Change is possible, even when it feels impossible.",
                "This difficult moment is not your final destination.",
                "Healing can happen in ways you haven't imagined yet.",
                "Your story is still being written.",
                "Better days are possible and worth working toward."
            ],
            
            'capability_affirmations': [
                "You have everything within you that you need.",
                "You're more capable than you know.",
                "You've overcome challenges before and you can do it again.",
                "Your wisdom and insight will guide you.",
                "You have the power to create positive change in your life."
            ],
            
            'worth_reminders': [
                "You are worthy of love and care exactly as you are.",
                "Your value as a person is unconditional.",
                "You deserve kindness, especially from yourself.",
                "Your life has meaning and purpose.",
                "You matter, and your wellbeing matters."
            ]
        }
    
    def _initialize_safety_protocols(self) -> Dict[str, Dict[str, Any]]:
        """Initialize safety-focused response protocols"""
        
        return {
            'crisis_indicators': {
                'immediate_response': [
                    "Your safety is my immediate concern.",
                    "Let's make sure you have the support you need right now.",
                    "I want to connect you with people who can help immediately."
                ],
                'resource_offers': [
                    "Crisis helplines are available 24/7.",
                    "Emergency services can provide immediate support.",
                    "Professional crisis counselors are trained to help.",
                    "You don't have to handle this alone."
                ],
                'safety_reminders': [
                    "Your life has value and meaning.",
                    "This crisis will pass, even though it feels overwhelming.",
                    "Help is available and you deserve support.",
                    "You've survived difficult times before."
                ]
            },
            
            'self_harm_concerns': {
                'immediate_response': [
                    "I'm concerned about your safety right now.",
                    "Your wellbeing is incredibly important.",
                    "Let's focus on keeping you safe."
                ],
                'intervention_language': [
                    "Please reach out to a crisis counselor immediately.",
                    "Emergency services can provide immediate safety support.",
                    "A mental health professional needs to assess your safety.",
                    "Don't try to handle this alone - professional help is available."
                ]
            },
            
            'vulnerability_support': {
                'gentle_approach': [
                    "I can see you're in a vulnerable place right now.",
                    "It's okay to need extra support when things are difficult.",
                    "Reaching out shows strength, not weakness."
                ],
                'protective_responses': [
                    "Let's take this slowly and gently.",
                    "You have complete control over this conversation.",
                    "We can pause anytime you need to."
                ]
            }
        }
    
    def _initialize_cultural_adaptations(self) -> Dict[str, Dict[str, Any]]:
        """Initialize cultural adaptation guidelines"""
        
        return {
            'religious_context': {
                'respectful_language': [
                    "honoring your faith journey",
                    "in alignment with your spiritual beliefs",
                    "respecting your religious values",
                    "within your spiritual framework"
                ],
                'inclusive_wisdom': [
                    "many spiritual traditions teach us",
                    "across different faiths, we find",
                    "spiritual wisdom often reminds us",
                    "your faith tradition may offer insights"
                ]
            },
            
            'cultural_sensitivity': {
                'family_dynamics': [
                    "within your cultural context",
                    "honoring your family traditions",
                    "respecting your cultural values",
                    "considering your community's wisdom"
                ],
                'healing_approaches': [
                    "cultural healing practices",
                    "traditional wisdom from your heritage",
                    "community-based healing approaches",
                    "culturally relevant support methods"
                ]
            }
        }
    
    async def generate_response(self, user_input: str,
                              user_context: Dict[str, Any],
                              response_type: ResponseType,
                              response_context: ResponseContext,
                              persona_config: ResponsePersonality) -> str:
        """
        Generate a complete response based on context and persona.
        
        Args:
            user_input: What the user said/wrote
            user_context: User's context and history
            response_type: Type of response to generate
            response_context: Context of the interaction
            persona_config: Personality configuration to use
        """
        try:
            # Generate response components
            components = await self._generate_response_components(
                user_input, user_context, response_type, response_context, persona_config
            )
            
            # Assemble complete response
            response = await self._assemble_response(components, persona_config)
            
            # Apply safety filtering
            response = await self._apply_safety_filtering(response, user_context)
            
            # Apply cultural adaptations
            response = await self._apply_cultural_adaptations(response, user_context)
            
            # Track response for learning
            await self._track_response(user_input, response, user_context, response_type)
            
            logger.debug(f"Generated {response_type.value} response with {persona_config.primary_trait.value} persona")
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return await self._generate_fallback_response(user_context)
    
    async def _generate_response_components(self, user_input: str,
                                          user_context: Dict[str, Any],
                                          response_type: ResponseType,
                                          response_context: ResponseContext,
                                          persona_config: ResponsePersonality) -> ResponseComponents:
        """Generate individual components of the response"""
        
        template = self.response_templates.get(response_type, {})
        
        # Generate opening
        opening = await self._generate_opening(template, persona_config, response_context)
        
        # Generate validation if appropriate
        validation = await self._generate_validation(user_input, user_context, persona_config)
        
        # Generate main body
        body = await self._generate_body_content(
            user_input, user_context, response_type, persona_config
        )
        
        # Generate wisdom if appropriate
        wisdom = await self._generate_wisdom_content(user_context, persona_config)
        
        # Generate encouragement
        encouragement = await self._generate_encouragement(user_context, persona_config)
        
        # Generate practical support
        practical_support = await self._generate_practical_support(user_context, response_type)
        
        # Generate safety note if needed
        safety_note = await self._generate_safety_note(user_context, response_type)
        
        # Generate closing
        closing = await self._generate_closing(template, persona_config, response_context)
        
        # Determine emotional tone
        emotional_tone = await self._determine_emotional_tone(persona_config, response_type)
        
        return ResponseComponents(
            opening=opening,
            body=body,
            validation=validation,
            wisdom=wisdom,
            encouragement=encouragement,
            practical_support=practical_support,
            safety_note=safety_note,
            closing=closing,
            emotional_tone=emotional_tone
        )
    
    async def _generate_opening(self, template: Dict[str, Any],
                              persona_config: ResponsePersonality,
                              context: ResponseContext) -> str:
        """Generate appropriate opening for response"""
        
        # Get context-specific openings
        if context == ResponseContext.CRISIS_INTERVENTION:
            return "I'm here with you and want to help ensure your safety."
        
        elif context == ResponseContext.CELEBRATION_MOMENT:
            return random.choice([
                "This is wonderful to hear!",
                "I'm so happy for you!",
                "What beautiful progress!"
            ])
        
        # Use persona-adapted openings
        openings = template.get('openings', ["I'm here to support you."])
        base_opening = random.choice(openings)
        
        # Adapt for persona warmth
        if persona_config.warmth_level > 0.8:
            warmth_modifiers = ["I'm so glad you shared this.", "Thank you for trusting me with this."]
            return f"{random.choice(warmth_modifiers)} {base_opening}"
        
        return base_opening
    
    async def _generate_validation(self, user_input: str,
                                 user_context: Dict[str, Any],
                                 persona_config: ResponsePersonality) -> Optional[str]:
        """Generate validation component if appropriate"""
        
        # Skip validation for celebration contexts
        if user_context.get('celebration_context', False):
            return None
        
        # Determine validation type needed
        if any(emotion in user_input.lower() for emotion in ['feel', 'feeling', 'emotional']):
            validation_type = 'emotional_validation'
        elif any(word in user_input.lower() for word in ['happened', 'experienced', 'went through']):
            validation_type = 'experience_validation'
        elif any(word in user_input.lower() for word in ['difficult', 'hard', 'struggling']):
            validation_type = 'struggle_validation'
        elif any(word in user_input.lower() for word in ['progress', 'growth', 'better']):
            validation_type = 'growth_validation'
        else:
            validation_type = 'emotional_validation'  # Default
        
        validation_phrases = self.validation_phrases.get(validation_type, [])
        
        if validation_phrases:
            return random.choice(validation_phrases)
        
        return None
    
    async def _generate_body_content(self, user_input: str,
                                   user_context: Dict[str, Any],
                                   response_type: ResponseType,
                                   persona_config: ResponsePersonality) -> str:
        """Generate main body content of response"""
        
        if response_type == ResponseType.EMPATHETIC_VALIDATION:
            return await self._generate_empathetic_body(user_input, user_context)
        
        elif response_type == ResponseType.GENTLE_GUIDANCE:
            return await self._generate_guidance_body(user_input, user_context, persona_config)
        
        elif response_type == ResponseType.WISDOM_SHARING:
            return await self._generate_wisdom_body(user_context)
        
        elif response_type == ResponseType.HOPE_OFFERING:
            return await self._generate_hope_body(user_context)
        
        elif response_type == ResponseType.SAFETY_CHECKING:
            return await self._generate_safety_body(user_context)
        
        elif response_type == ResponseType.CELEBRATION:
            return await self._generate_celebration_body(user_input, user_context)
        
        else:
            return "I'm here to support you in whatever way feels most helpful."
    
    async def _generate_empathetic_body(self, user_input: str, user_context: Dict[str, Any]) -> str:
        """Generate empathetic response body"""
        
        # Reflect back key elements from user input
        reflection_elements = []
        
        if 'difficult' in user_input.lower():
            reflection_elements.append("how difficult this situation is")
        if 'alone' in user_input.lower():
            reflection_elements.append("the isolation you're feeling")
        if 'confused' in user_input.lower():
            reflection_elements.append("the confusion you're experiencing")
        
        if reflection_elements:
            reflection = f"I can sense {reflection_elements[0]}"
        else:
            reflection = "I can feel the weight of what you're carrying"
        
        solidarity = "You're not alone in this experience."
        
        return f"{reflection}. {solidarity}"
    
    async def _generate_guidance_body(self, user_input: str,
                                    user_context: Dict[str, Any],
                                    persona_config: ResponsePersonality) -> str:
        """Generate gentle guidance body"""
        
        # Offer gentle suggestions based on context
        if 'overwhelmed' in user_input.lower():
            return ("When feeling overwhelmed, sometimes it helps to focus on just the next small step. "
                   "You don't have to have all the answers right now.")
        
        elif 'stuck' in user_input.lower():
            return ("Feeling stuck is often a sign that you're in a transition period. "
                   "What might help is giving yourself permission to not know the way forward yet.")
        
        elif 'angry' in user_input.lower():
            return ("Anger often carries important information about our boundaries and values. "
                   "It might help to explore what your anger is trying to tell you.")
        
        else:
            return ("Sometimes the most helpful thing we can do is simply be present with what we're experiencing, "
                   "without needing to fix or change anything immediately.")
    
    async def _generate_wisdom_body(self, user_context: Dict[str, Any]) -> str:
        """Generate wisdom sharing body"""
        
        # Select relevant wisdom from persona
        context_type = user_context.get('primary_need', 'healing')
        wisdom_pieces = self.persona.get_wisdom_for_context(context_type, count=1)
        
        if wisdom_pieces:
            wisdom = wisdom_pieces[0]
            return f"What I've learned is that {wisdom.lower()}"
        
        return "One thing that our community has discovered is that healing happens at exactly the right pace for each person."
    
    async def _generate_hope_body(self, user_context: Dict[str, Any]) -> str:
        """Generate hope offering body"""
        
        hope_phrases = self.encouragement_bank.get('hope_anchors', [])
        
        if hope_phrases:
            hope_statement = random.choice(hope_phrases)
            return f"{hope_statement} Even in this difficult time, your capacity for healing and growth remains."
        
        return "Even when hope feels distant, it remains available to you. Your story is still being written."
    
    async def _generate_safety_body(self, user_context: Dict[str, Any]) -> str:
        """Generate safety checking body"""
        
        if user_context.get('crisis_indicators', False):
            return ("I want to make sure you have immediate support available. "
                   "Crisis counselors are available 24/7 and can provide professional guidance.")
        
        return ("I want to check in on how you're taking care of yourself. "
               "Your safety and wellbeing are important.")
    
    async def _generate_celebration_body(self, user_input: str, user_context: Dict[str, Any]) -> str:
        """Generate celebration response body"""
        
        # Identify what's being celebrated
        if 'progress' in user_input.lower():
            return "The progress you're making is real and meaningful. Each step forward matters."
        
        elif 'milestone' in user_input.lower():
            return "Reaching this milestone shows your dedication and strength. This is worth celebrating!"
        
        else:
            return "Your growth and courage are evident in what you've shared. This deserves recognition."
    
    async def _generate_wisdom_content(self, user_context: Dict[str, Any],
                                     persona_config: ResponsePersonality) -> Optional[str]:
        """Generate wisdom content if appropriate"""
        
        if persona_config.wisdom_integration < 0.5:
            return None
        
        # Select wisdom based on user's current needs
        primary_need = user_context.get('primary_need', 'healing')
        wisdom_pieces = self.persona.get_wisdom_for_context(primary_need, count=1)
        
        if wisdom_pieces:
            return f"One truth I want to share: {wisdom_pieces[0]}"
        
        return None
    
    async def _generate_encouragement(self, user_context: Dict[str, Any],
                                    persona_config: ResponsePersonality) -> Optional[str]:
        """Generate encouragement if appropriate"""
        
        if persona_config.primary_trait == PersonalityTrait.ENCOURAGING:
            encouragement_type = random.choice(['strength_recognition', 'capability_affirmations'])
            encouragement_phrases = self.encouragement_bank.get(encouragement_type, [])
            
            if encouragement_phrases:
                return random.choice(encouragement_phrases)
        
        return None
    
    async def _generate_practical_support(self, user_context: Dict[str, Any],
                                        response_type: ResponseType) -> Optional[str]:
        """Generate practical support suggestions"""
        
        if response_type == ResponseType.RESOURCE_OFFERING:
            return "Would it be helpful if I shared some resources that others have found supportive?"
        
        elif user_context.get('seeking_guidance', False):
            return "I'm here to explore options with you at whatever pace feels right."
        
        return None
    
    async def _generate_safety_note(self, user_context: Dict[str, Any],
                                  response_type: ResponseType) -> Optional[str]:
        """Generate safety note if needed"""
        
        if user_context.get('crisis_indicators', False):
            return "If you're in immediate danger, please contact emergency services right away."
        
        elif response_type == ResponseType.CRISIS_SUPPORT:
            return "Remember: You deserve support and your life has value."
        
        return None
    
    async def _generate_closing(self, template: Dict[str, Any],
                              persona_config: ResponsePersonality,
                              context: ResponseContext) -> str:
        """Generate appropriate closing"""
        
        closings = template.get('closings', ["I'm here for you."])
        base_closing = random.choice(closings)
        
        # Adapt for warmth level
        if persona_config.warmth_level > 0.8:
            warm_additions = ["Sending you care.", "You're not alone.", "I'm thinking of you."]
            return f"{base_closing} {random.choice(warm_additions)}"
        
        return base_closing
    
    async def _determine_emotional_tone(self, persona_config: ResponsePersonality,
                                      response_type: ResponseType) -> str:
        """Determine overall emotional tone"""
        
        tone_mapping = {
            ResponseType.EMPATHETIC_VALIDATION: "compassionate_understanding",
            ResponseType.GENTLE_GUIDANCE: "wise_caring",
            ResponseType.WISDOM_SHARING: "thoughtful_sharing",
            ResponseType.HOPE_OFFERING: "hopeful_encouraging",
            ResponseType.SAFETY_CHECKING: "protective_caring",
            ResponseType.CELEBRATION: "joyful_proud",
            ResponseType.CRISIS_SUPPORT: "calm_protective"
        }
        
        return tone_mapping.get(response_type, "warm_supportive")
    
    async def _assemble_response(self, components: ResponseComponents,
                               persona_config: ResponsePersonality) -> str:
        """Assemble complete response from components"""
        
        response_parts = [components.opening]
        
        if components.validation:
            response_parts.append(components.validation)
        
        response_parts.append(components.body)
        
        if components.wisdom:
            response_parts.append(components.wisdom)
        
        if components.encouragement:
            response_parts.append(components.encouragement)
        
        if components.practical_support:
            response_parts.append(components.practical_support)
        
        if components.safety_note:
            response_parts.append(components.safety_note)
        
        response_parts.append(components.closing)
        
        # Join with appropriate spacing
        return " ".join(response_parts)
    
    async def _apply_safety_filtering(self, response: str, user_context: Dict[str, Any]) -> str:
        """Apply safety filtering to response"""
        
        # Check for potentially harmful language
        harmful_phrases = ['just get over it', 'it could be worse', 'at least', 'should feel grateful']
        
        for phrase in harmful_phrases:
            if phrase in response.lower():
                logger.warning(f"Potentially harmful phrase detected: {phrase}")
                # This would trigger a response regeneration in production
        
        return response
    
    async def _apply_cultural_adaptations(self, response: str, user_context: Dict[str, Any]) -> str:
        """Apply cultural adaptations to response"""
        
        # Add cultural sensitivity if religious context
        if user_context.get('religious_context'):
            religious_adaptations = self.cultural_adaptations.get('religious_context', {})
            # Apply adaptations as needed
        
        return response
    
    async def _track_response(self, user_input: str, response: str,
                            user_context: Dict[str, Any], response_type: ResponseType) -> None:
        """Track response for learning and improvement"""
        
        user_id = user_context.get('user_id', 'anonymous')
        
        if user_id not in self.response_history:
            self.response_history[user_id] = []
        
        self.response_history[user_id].append({
            'user_input': user_input,
            'response': response,
            'response_type': response_type.value,
            'timestamp': datetime.utcnow(),
            'context': user_context
        })
        
        # Keep only recent history
        max_history = 50
        if len(self.response_history[user_id]) > max_history:
            self.response_history[user_id] = self.response_history[user_id][-max_history:]
    
    async def _generate_fallback_response(self, user_context: Dict[str, Any]) -> str:
        """Generate safe fallback response"""
        
        fallback_responses = [
            "I'm here to support you. Thank you for sharing with me.",
            "Your experience matters and I'm honored you've trusted me with it.",
            "I see you and I'm here with you in this moment.",
            "You're not alone. I'm here to listen and support you however I can."
        ]
        
        return random.choice(fallback_responses)