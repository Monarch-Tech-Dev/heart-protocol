"""
Cultural Sensitivity Engine

Ensures affirmations respect diverse cultural values, beliefs, and communication styles.
Built on principles of cultural humility and inclusive design from the Open Source Love License.
Never imposes values - always adapts to honor user's cultural context.
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, date
from enum import Enum
from dataclasses import dataclass
import logging
import json

from .content_database import Affirmation, AffirmationType, CulturalContext

logger = logging.getLogger(__name__)


class CulturalDimension(Enum):
    """Cultural dimensions based on cross-cultural psychology research"""
    INDIVIDUALISM_COLLECTIVISM = "individualism_collectivism"
    POWER_DISTANCE = "power_distance"
    UNCERTAINTY_AVOIDANCE = "uncertainty_avoidance"
    MASCULINITY_FEMININITY = "masculinity_femininity"
    LONG_TERM_ORIENTATION = "long_term_orientation"
    INDULGENCE_RESTRAINT = "indulgence_restraint"
    CONTEXT_COMMUNICATION = "context_communication"  # High vs Low context
    RELATIONSHIP_TASK = "relationship_task"  # Relationship vs Task focused


class ReligiousContext(Enum):
    """Religious/spiritual contexts for appropriate messaging"""
    SECULAR = "secular"
    CHRISTIAN = "christian"
    ISLAMIC = "islamic"
    JEWISH = "jewish"
    BUDDHIST = "buddhist"
    HINDU = "hindu"
    INDIGENOUS_SPIRITUAL = "indigenous_spiritual"
    INTERFAITH = "interfaith"
    AGNOSTIC = "agnostic"
    ATHEIST = "atheist"
    SPIRITUAL_NOT_RELIGIOUS = "spiritual_not_religious"


class CommunicationStyle(Enum):
    """Cultural communication style preferences"""
    DIRECT = "direct"                    # Clear, explicit communication
    INDIRECT = "indirect"                # Subtle, contextual communication
    FORMAL = "formal"                    # Respectful, hierarchical
    INFORMAL = "informal"                # Casual, egalitarian
    EMOTIONAL = "emotional"              # Feelings-focused
    RATIONAL = "rational"                # Logic-focused
    STORY_BASED = "story_based"          # Narrative, metaphorical
    FACT_BASED = "fact_based"            # Evidence, data-driven


@dataclass
class CulturalProfile:
    """Represents a user's cultural context and preferences"""
    user_id: str
    cultural_contexts: List[CulturalContext]
    religious_context: Optional[ReligiousContext] = None
    communication_styles: List[CommunicationStyle] = None
    cultural_dimensions: Dict[CulturalDimension, float] = None  # 0.0 to 1.0 scales
    language_preferences: List[str] = None
    cultural_holidays: List[str] = None
    sensitive_topics: List[str] = None
    preferred_metaphors: List[str] = None
    avoided_concepts: List[str] = None
    family_structure: str = None  # nuclear, extended, chosen, etc.
    last_updated: datetime = None
    
    def __post_init__(self):
        if self.communication_styles is None:
            self.communication_styles = []
        if self.cultural_dimensions is None:
            self.cultural_dimensions = {}
        if self.language_preferences is None:
            self.language_preferences = ['english']
        if self.cultural_holidays is None:
            self.cultural_holidays = []
        if self.sensitive_topics is None:
            self.sensitive_topics = []
        if self.preferred_metaphors is None:
            self.preferred_metaphors = []
        if self.avoided_concepts is None:
            self.avoided_concepts = []
        if self.last_updated is None:
            self.last_updated = datetime.utcnow()


@dataclass
class CulturalAdaptation:
    """Represents how to adapt content for cultural appropriateness"""
    original_text: str
    adapted_text: str
    cultural_context: CulturalContext
    adaptations_made: List[str]
    confidence_score: float  # 0.0 to 1.0
    reasoning: str


class CulturalSensitivityEngine:
    """
    Ensures affirmations are culturally appropriate and respectful.
    
    Core Principles:
    - Cultural humility: Never assume our perspective is universal
    - Inclusive design: Create space for diverse values and beliefs
    - User agency: Users define their own cultural identity
    - Continuous learning: Adapt based on community feedback
    - Harm prevention: Avoid cultural stereotypes and appropriation
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cultural_profiles = {}  # In production, database-backed
        
        # Initialize cultural knowledge bases
        self.cultural_adaptations = self._initialize_cultural_adaptations()
        self.religious_considerations = self._initialize_religious_considerations()
        self.communication_style_guides = self._initialize_communication_styles()
        self.cultural_calendar = self._initialize_cultural_calendar()
        self.sensitive_topics_map = self._initialize_sensitive_topics()
        
        logger.info("Cultural Sensitivity Engine initialized")
    
    def _initialize_cultural_adaptations(self) -> Dict[str, Dict[str, Any]]:
        """Initialize cultural adaptation guidelines"""
        
        return {
            'individualism_to_collectivism': {
                'adaptations': {
                    'you are': 'you and your community are',
                    'your strength': 'the strength you share with others',
                    'your success': 'your community\'s success through you',
                    'you matter': 'you matter to your community',
                    'your journey': 'the path you walk with others',
                    'your worth': 'your value to the collective',
                    'independent': 'interconnected',
                    'self-reliant': 'mutually supportive'
                },
                'add_concepts': ['community', 'belonging', 'shared', 'together', 'collective'],
                'avoid_concepts': ['alone', 'individual', 'self-made', 'independent']
            },
            
            'collectivism_to_individualism': {
                'adaptations': {
                    'community': 'personal',
                    'collective': 'individual',
                    'shared journey': 'personal path',
                    'together': 'within yourself',
                    'interdependent': 'self-sufficient'
                },
                'add_concepts': ['personal', 'individual', 'unique', 'autonomous'],
                'avoid_concepts': ['duty to others', 'collective responsibility']
            },
            
            'high_context_to_low_context': {
                'adaptations': {
                    'you know in your heart': 'clearly and directly',
                    'wisdom flows': 'evidence shows',
                    'ancient knowledge': 'proven methods',
                    'spiritual understanding': 'logical reasoning'
                },
                'add_concepts': ['clear', 'direct', 'specific', 'explicit'],
                'avoid_concepts': ['mystical', 'implied', 'understood', 'felt']
            },
            
            'secular_to_spiritual': {
                'adaptations': {
                    'universe': 'divine presence',
                    'nature': 'creation',
                    'inner strength': 'divine strength within',
                    'potential': 'divine purpose',
                    'healing': 'divine healing',
                    'love': 'divine love',
                    'wisdom': 'sacred wisdom'
                },
                'add_concepts': ['blessed', 'sacred', 'divine', 'spiritual', 'holy'],
                'avoid_concepts': ['random', 'coincidence', 'luck', 'chance']
            },
            
            'spiritual_to_secular': {
                'adaptations': {
                    'divine': 'natural',
                    'blessed': 'fortunate',
                    'sacred': 'meaningful',
                    'spiritual': 'emotional',
                    'prayer': 'reflection',
                    'divine love': 'unconditional love',
                    'sacred wisdom': 'deep understanding'
                },
                'add_concepts': ['natural', 'human', 'psychological', 'evidence-based'],
                'avoid_concepts': ['divine', 'sacred', 'blessed', 'holy', 'spiritual']
            }
        }
    
    def _initialize_religious_considerations(self) -> Dict[ReligiousContext, Dict[str, Any]]:
        """Initialize religious sensitivity guidelines"""
        
        return {
            ReligiousContext.CHRISTIAN: {
                'preferred_concepts': ['blessed', 'grace', 'love', 'hope', 'faith', 'peace'],
                'neutral_adaptations': {
                    'universe': 'God\'s creation',
                    'inner strength': 'strength through faith',
                    'divine love': 'God\'s love'
                },
                'avoid_concepts': ['karma', 'reincarnation', 'chakras', 'enlightenment'],
                'holidays': ['christmas', 'easter', 'good_friday', 'pentecost'],
                'respectful_language': True
            },
            
            ReligiousContext.ISLAMIC: {
                'preferred_concepts': ['blessed', 'peace', 'guidance', 'mercy', 'gratitude'],
                'neutral_adaptations': {
                    'universe': 'Allah\'s creation',
                    'divine love': 'Allah\'s mercy',
                    'blessed': 'blessed by Allah'
                },
                'avoid_concepts': ['trinity', 'incarnation', 'meditation', 'yoga'],
                'holidays': ['ramadan', 'eid_al_fitr', 'eid_al_adha', 'hajj'],
                'respectful_language': True,
                'special_considerations': ['prayer_times', 'fasting_periods']
            },
            
            ReligiousContext.JEWISH: {
                'preferred_concepts': ['blessing', 'wisdom', 'community', 'justice', 'healing'],
                'neutral_adaptations': {
                    'divine': 'G-d',
                    'blessed': 'blessed by Hashem',
                    'sacred': 'holy'
                },
                'avoid_concepts': ['trinity', 'incarnation', 'new_testament'],
                'holidays': ['rosh_hashanah', 'yom_kippur', 'passover', 'shabbat'],
                'respectful_language': True,
                'special_considerations': ['sabbath_observance']
            },
            
            ReligiousContext.BUDDHIST: {
                'preferred_concepts': ['peace', 'compassion', 'wisdom', 'mindfulness', 'loving_kindness'],
                'neutral_adaptations': {
                    'divine love': 'loving kindness',
                    'strength': 'inner peace',
                    'healing': 'liberation from suffering'
                },
                'avoid_concepts': ['eternal_soul', 'creator_god', 'divine_judgment'],
                'holidays': ['vesak', 'bodhi_day', 'dharma_day'],
                'special_considerations': ['meditation_practices', 'noble_truths']
            },
            
            ReligiousContext.HINDU: {
                'preferred_concepts': ['divine', 'sacred', 'dharma', 'karma', 'peace', 'wisdom'],
                'neutral_adaptations': {
                    'universe': 'divine consciousness',
                    'inner strength': 'divine spark within',
                    'purpose': 'dharma'
                },
                'avoid_concepts': ['single_god', 'linear_time', 'one_life_only'],
                'holidays': ['diwali', 'holi', 'navaratri', 'dussehra'],
                'special_considerations': ['multiple_paths', 'cycles_of_time']
            },
            
            ReligiousContext.INDIGENOUS_SPIRITUAL: {
                'preferred_concepts': ['ancestors', 'wisdom', 'connection', 'earth', 'community'],
                'neutral_adaptations': {
                    'universe': 'great spirit',
                    'wisdom': 'ancestral wisdom',
                    'strength': 'strength of ancestors'
                },
                'avoid_concepts': ['primitive', 'savage', 'convert', 'civilize'],
                'special_considerations': ['land_connection', 'ancestor_reverence', 'community_focus'],
                'respectful_language': True,
                'cultural_sensitivity_critical': True
            },
            
            ReligiousContext.SECULAR: {
                'preferred_concepts': ['human', 'natural', 'psychological', 'scientific', 'evidence'],
                'neutral_adaptations': {
                    'divine': 'natural',
                    'blessed': 'fortunate',
                    'sacred': 'meaningful',
                    'spiritual': 'emotional'
                },
                'avoid_concepts': ['divine', 'sacred', 'blessed', 'spiritual', 'prayer'],
                'special_considerations': ['evidence_based', 'rational_approach']
            },
            
            ReligiousContext.ATHEIST: {
                'preferred_concepts': ['human', 'natural', 'rational', 'scientific', 'evidence'],
                'neutral_adaptations': {
                    'divine love': 'human compassion',
                    'blessed': 'fortunate',
                    'sacred': 'deeply meaningful',
                    'spiritual': 'emotional and psychological'
                },
                'avoid_concepts': ['divine', 'god', 'blessed', 'sacred', 'prayer', 'spiritual'],
                'special_considerations': ['strictly_secular', 'no_religious_language']
            }
        }
    
    def _initialize_communication_styles(self) -> Dict[CommunicationStyle, Dict[str, Any]]:
        """Initialize communication style adaptations"""
        
        return {
            CommunicationStyle.DIRECT: {
                'tone': 'clear and straightforward',
                'structure': 'simple declarative statements',
                'avoid': ['metaphors', 'indirect_suggestions', 'implied_meanings'],
                'examples': [
                    'You are valuable.',
                    'You have strength.',
                    'You can heal.',
                    'You matter.'
                ]
            },
            
            CommunicationStyle.INDIRECT: {
                'tone': 'gentle and suggestive',
                'structure': 'metaphorical and contextual',
                'prefer': ['stories', 'metaphors', 'gentle_suggestions'],
                'examples': [
                    'Like a river finding its way to the sea, you too will find your path.',
                    'In the garden of life, even the smallest seed can grow into something beautiful.',
                    'Sometimes the strongest trees bend in the wind rather than break.'
                ]
            },
            
            CommunicationStyle.FORMAL: {
                'tone': 'respectful and structured',
                'structure': 'polite and hierarchical',
                'language_features': ['respectful_address', 'formal_vocabulary'],
                'examples': [
                    'You are deserving of respect and honor.',
                    'Your contributions are valued and significant.',
                    'You have earned recognition for your efforts.'
                ]
            },
            
            CommunicationStyle.INFORMAL: {
                'tone': 'casual and friendly',
                'structure': 'conversational and warm',
                'language_features': ['casual_vocabulary', 'friendly_tone'],
                'examples': [
                    'You\'re doing great!',
                    'Keep being awesome!',
                    'You\'ve got this!'
                ]
            },
            
            CommunicationStyle.STORY_BASED: {
                'tone': 'narrative and metaphorical',
                'structure': 'story format with lessons',
                'prefer': ['metaphors', 'analogies', 'nature_examples'],
                'examples': [
                    'Like the oak tree that grows stronger in storms, your challenges are making you more resilient.',
                    'Every butterfly was once a caterpillar that refused to give up on transformation.',
                    'The mountain doesn\'t worry about the clouds that pass over it.'
                ]
            },
            
            CommunicationStyle.FACT_BASED: {
                'tone': 'evidence-based and logical',
                'structure': 'factual statements with reasoning',
                'prefer': ['research_references', 'logical_reasoning', 'clear_evidence'],
                'examples': [
                    'Research shows that self-compassion improves mental health outcomes.',
                    'Studies demonstrate that resilience can be developed through practice.',
                    'Neuroplasticity research proves your brain can form new, healthier patterns.'
                ]
            }
        }
    
    def _initialize_cultural_calendar(self) -> Dict[str, Dict[str, Any]]:
        """Initialize cultural holidays and significant dates"""
        
        return {
            'chinese_new_year': {
                'dates': ['february_variable'],
                'cultural_significance': 'renewal, family, prosperity',
                'appropriate_themes': ['new_beginnings', 'family_love', 'hope'],
                'avoid_themes': ['endings', 'loss', 'sadness']
            },
            
            'ramadan': {
                'dates': ['lunar_calendar_variable'],
                'cultural_significance': 'spiritual reflection, community, self-discipline',
                'appropriate_themes': ['spiritual_growth', 'community', 'patience'],
                'avoid_themes': ['indulgence', 'excess', 'materialism'],
                'special_considerations': ['fasting_hours', 'prayer_times']
            },
            
            'diwali': {
                'dates': ['october_november_variable'],
                'cultural_significance': 'light over darkness, good over evil',
                'appropriate_themes': ['light', 'hope', 'victory', 'renewal'],
                'avoid_themes': ['darkness', 'evil', 'despair']
            },
            
            'day_of_the_dead': {
                'dates': ['november_1_2'],
                'cultural_significance': 'honoring ancestors, celebration of life',
                'appropriate_themes': ['remembrance', 'celebration_of_life', 'connection'],
                'avoid_themes': ['fear_of_death', 'morbidity'],
                'cultural_sensitivity': 'not_halloween'
            },
            
            'indigenous_peoples_day': {
                'dates': ['october_second_monday'],
                'cultural_significance': 'honoring indigenous wisdom, resilience',
                'appropriate_themes': ['ancestral_wisdom', 'resilience', 'connection_to_land'],
                'cultural_sensitivity': 'high'
            }
        }
    
    def _initialize_sensitive_topics(self) -> Dict[str, List[str]]:
        """Initialize topics that require cultural sensitivity"""
        
        return {
            'family_structures': [
                'nuclear_family_assumptions',
                'heteronormative_language',
                'biological_parent_assumptions',
                'marriage_assumptions'
            ],
            
            'gender_and_sexuality': [
                'binary_gender_assumptions',
                'heteronormative_language',
                'traditional_gender_roles',
                'sexual_orientation_assumptions'
            ],
            
            'economic_assumptions': [
                'homeownership_assumptions',
                'car_ownership_assumptions',
                'disposable_income_assumptions',
                'educational_privilege_assumptions'
            ],
            
            'ability_assumptions': [
                'neurotypical_assumptions',
                'physical_ability_assumptions',
                'mental_health_stigma',
                'productivity_assumptions'
            ],
            
            'cultural_appropriation_risks': [
                'sacred_symbols_misuse',
                'spiritual_practice_misuse',
                'cultural_metaphors_misuse',
                'stereotypical_representations'
            ]
        }
    
    async def adapt_affirmation_for_culture(self, affirmation: Affirmation, 
                                          cultural_profile: CulturalProfile) -> CulturalAdaptation:
        """
        Adapt an affirmation to be culturally appropriate for the user.
        
        Args:
            affirmation: The original affirmation
            cultural_profile: User's cultural context and preferences
        """
        try:
            adaptations_made = []
            adapted_text = affirmation.text
            confidence_score = 1.0
            
            # Apply religious considerations
            if cultural_profile.religious_context:
                adapted_text, religious_adaptations = await self._apply_religious_adaptations(
                    adapted_text, cultural_profile.religious_context
                )
                adaptations_made.extend(religious_adaptations)
            
            # Apply cultural dimension adaptations
            adapted_text, dimension_adaptations = await self._apply_cultural_dimension_adaptations(
                adapted_text, cultural_profile.cultural_dimensions
            )
            adaptations_made.extend(dimension_adaptations)
            
            # Apply communication style adaptations
            adapted_text, style_adaptations = await self._apply_communication_style_adaptations(
                adapted_text, cultural_profile.communication_styles
            )
            adaptations_made.extend(style_adaptations)
            
            # Check for cultural sensitivity issues
            sensitivity_issues = await self._check_cultural_sensitivity(
                adapted_text, cultural_profile
            )
            
            if sensitivity_issues:
                adapted_text, sensitivity_adaptations = await self._resolve_sensitivity_issues(
                    adapted_text, sensitivity_issues, cultural_profile
                )
                adaptations_made.extend(sensitivity_adaptations)
                confidence_score *= 0.8  # Reduce confidence when issues found
            
            # Validate final adaptation
            final_validation = await self._validate_cultural_adaptation(
                adapted_text, cultural_profile
            )
            
            if not final_validation['appropriate']:
                confidence_score *= 0.5
                adaptations_made.append(f"Warning: {final_validation['concerns']}")
            
            return CulturalAdaptation(
                original_text=affirmation.text,
                adapted_text=adapted_text,
                cultural_context=cultural_profile.cultural_contexts[0] if cultural_profile.cultural_contexts else CulturalContext.UNIVERSAL,
                adaptations_made=adaptations_made,
                confidence_score=confidence_score,
                reasoning=self._generate_adaptation_reasoning(adaptations_made)
            )
            
        except Exception as e:
            logger.error(f"Error adapting affirmation for culture: {e}")
            return CulturalAdaptation(
                original_text=affirmation.text,
                adapted_text=affirmation.text,  # Fallback to original
                cultural_context=CulturalContext.UNIVERSAL,
                adaptations_made=['error_fallback'],
                confidence_score=0.1,
                reasoning="Error occurred, using original text"
            )
    
    async def _apply_religious_adaptations(self, text: str, 
                                         religious_context: ReligiousContext) -> Tuple[str, List[str]]:
        """Apply religious-appropriate language adaptations"""
        
        if religious_context not in self.religious_considerations:
            return text, []
        
        religious_config = self.religious_considerations[religious_context]
        adaptations = religious_config.get('neutral_adaptations', {})
        avoided_concepts = religious_config.get('avoid_concepts', [])
        adaptations_made = []
        
        adapted_text = text
        
        # Apply positive adaptations
        for original, replacement in adaptations.items():
            if original.lower() in adapted_text.lower():
                adapted_text = adapted_text.replace(original, replacement)
                adaptations_made.append(f"religious_adaptation: {original} -> {replacement}")
        
        # Check for avoided concepts
        for avoided_concept in avoided_concepts:
            if avoided_concept.lower() in adapted_text.lower():
                adaptations_made.append(f"religious_sensitivity: avoided {avoided_concept}")
                # This would trigger a more sophisticated replacement
                # For now, we flag it for manual review
        
        return adapted_text, adaptations_made
    
    async def _apply_cultural_dimension_adaptations(self, text: str,
                                                   cultural_dimensions: Dict[CulturalDimension, float]) -> Tuple[str, List[str]]:
        """Apply adaptations based on cultural dimensions"""
        
        adaptations_made = []
        adapted_text = text
        
        # Individualism vs Collectivism
        if CulturalDimension.INDIVIDUALISM_COLLECTIVISM in cultural_dimensions:
            individualism_score = cultural_dimensions[CulturalDimension.INDIVIDUALISM_COLLECTIVISM]
            
            if individualism_score < 0.3:  # Strongly collectivist
                adaptation_rules = self.cultural_adaptations['individualism_to_collectivism']
                
                for original, replacement in adaptation_rules['adaptations'].items():
                    if original.lower() in adapted_text.lower():
                        adapted_text = adapted_text.replace(original, replacement)
                        adaptations_made.append(f"collectivist_adaptation: {original} -> {replacement}")
            
            elif individualism_score > 0.7:  # Strongly individualist
                adaptation_rules = self.cultural_adaptations['collectivism_to_individualism']
                
                for original, replacement in adaptation_rules['adaptations'].items():
                    if original.lower() in adapted_text.lower():
                        adapted_text = adapted_text.replace(original, replacement)
                        adaptations_made.append(f"individualist_adaptation: {original} -> {replacement}")
        
        # High vs Low Context Communication
        if CulturalDimension.CONTEXT_COMMUNICATION in cultural_dimensions:
            context_score = cultural_dimensions[CulturalDimension.CONTEXT_COMMUNICATION]
            
            if context_score < 0.3:  # Low context (direct communication)
                adaptation_rules = self.cultural_adaptations['high_context_to_low_context']
                
                for original, replacement in adaptation_rules['adaptations'].items():
                    if original.lower() in adapted_text.lower():
                        adapted_text = adapted_text.replace(original, replacement)
                        adaptations_made.append(f"low_context_adaptation: {original} -> {replacement}")
        
        return adapted_text, adaptations_made
    
    async def _apply_communication_style_adaptations(self, text: str,
                                                    communication_styles: List[CommunicationStyle]) -> Tuple[str, List[str]]:
        """Apply communication style adaptations"""
        
        adaptations_made = []
        adapted_text = text
        
        for style in communication_styles:
            if style == CommunicationStyle.DIRECT:
                # Make language more direct and clear
                adapted_text = await self._make_more_direct(adapted_text)
                adaptations_made.append("direct_communication_style")
            
            elif style == CommunicationStyle.STORY_BASED:
                # Add narrative elements if not present
                if not any(word in adapted_text.lower() for word in ['like', 'as', 'imagine']):
                    adapted_text = await self._add_narrative_element(adapted_text)
                    adaptations_made.append("story_based_communication")
            
            elif style == CommunicationStyle.FORMAL:
                # Make language more formal and respectful
                adapted_text = await self._make_more_formal(adapted_text)
                adaptations_made.append("formal_communication_style")
            
            elif style == CommunicationStyle.FACT_BASED:
                # Add evidence-based language
                adapted_text = await self._add_evidence_language(adapted_text)
                adaptations_made.append("fact_based_communication")
        
        return adapted_text, adaptations_made
    
    async def _make_more_direct(self, text: str) -> str:
        """Make language more direct and clear"""
        # Remove metaphorical language and make statements clearer
        direct_replacements = {
            'you might': 'you',
            'perhaps': '',
            'in your heart': 'clearly',
            'flows through': 'exists in',
            'whispers': 'tells'
        }
        
        adapted = text
        for original, replacement in direct_replacements.items():
            adapted = adapted.replace(original, replacement)
        
        return adapted.strip()
    
    async def _add_narrative_element(self, text: str) -> str:
        """Add narrative/metaphorical element to text"""
        narrative_prefixes = [
            "Like a tree growing stronger through seasons, ",
            "In the same way that rivers find their path to the sea, ",
            "Just as the sun rises each day, "
        ]
        
        # Simple approach: add a narrative prefix
        return random.choice(narrative_prefixes) + text.lower()
    
    async def _make_more_formal(self, text: str) -> str:
        """Make language more formal and respectful"""
        formal_replacements = {
            'you\'re': 'you are',
            'can\'t': 'cannot',
            'won\'t': 'will not',
            'great': 'excellent',
            'awesome': 'remarkable'
        }
        
        adapted = text
        for original, replacement in formal_replacements.items():
            adapted = adapted.replace(original, replacement)
        
        return adapted
    
    async def _add_evidence_language(self, text: str) -> str:
        """Add evidence-based language elements"""
        evidence_phrases = [
            "Research confirms that ",
            "Studies show that ",
            "Evidence demonstrates that ",
            "Science validates that "
        ]
        
        # Add evidence phrase if text makes a claim
        if any(word in text.lower() for word in ['you are', 'you have', 'you can']):
            return random.choice(evidence_phrases) + text.lower()
        
        return text
    
    async def _check_cultural_sensitivity(self, text: str, 
                                        cultural_profile: CulturalProfile) -> List[str]:
        """Check for potential cultural sensitivity issues"""
        
        sensitivity_issues = []
        
        # Check against user's sensitive topics
        for sensitive_topic in cultural_profile.sensitive_topics:
            if sensitive_topic.lower() in text.lower():
                sensitivity_issues.append(f"contains_sensitive_topic: {sensitive_topic}")
        
        # Check against user's avoided concepts
        for avoided_concept in cultural_profile.avoided_concepts:
            if avoided_concept.lower() in text.lower():
                sensitivity_issues.append(f"contains_avoided_concept: {avoided_concept}")
        
        # Check for cultural appropriation risks
        appropriation_risks = self.sensitive_topics_map.get('cultural_appropriation_risks', [])
        for risk in appropriation_risks:
            if any(word in text.lower() for word in risk.split('_')):
                sensitivity_issues.append(f"appropriation_risk: {risk}")
        
        # Check for assumptions about family structures
        family_assumptions = ['mom', 'dad', 'parents', 'husband', 'wife']
        if any(assumption in text.lower() for assumption in family_assumptions):
            if cultural_profile.family_structure and cultural_profile.family_structure != 'nuclear':
                sensitivity_issues.append("family_structure_assumption")
        
        return sensitivity_issues
    
    async def _resolve_sensitivity_issues(self, text: str, issues: List[str],
                                        cultural_profile: CulturalProfile) -> Tuple[str, List[str]]:
        """Resolve identified cultural sensitivity issues"""
        
        resolutions = []
        adapted_text = text
        
        for issue in issues:
            if issue.startswith('contains_sensitive_topic:'):
                topic = issue.split(':', 1)[1].strip()
                # Remove or replace the sensitive topic
                adapted_text = adapted_text.replace(topic, '[topic_removed_for_sensitivity]')
                resolutions.append(f"removed_sensitive_topic: {topic}")
            
            elif issue.startswith('family_structure_assumption'):
                # Replace family assumptions with more inclusive language
                family_replacements = {
                    'parents': 'family',
                    'mom and dad': 'family',
                    'mother and father': 'caregivers',
                    'husband': 'partner',
                    'wife': 'partner'
                }
                
                for original, replacement in family_replacements.items():
                    if original in adapted_text.lower():
                        adapted_text = adapted_text.replace(original, replacement)
                        resolutions.append(f"inclusive_family_language: {original} -> {replacement}")
        
        return adapted_text, resolutions
    
    async def _validate_cultural_adaptation(self, text: str, 
                                          cultural_profile: CulturalProfile) -> Dict[str, Any]:
        """Validate that the cultural adaptation is appropriate"""
        
        validation_result = {
            'appropriate': True,
            'concerns': [],
            'confidence': 1.0
        }
        
        # Check length - some cultures prefer brevity, others detail
        if len(text) > 200 and CommunicationStyle.DIRECT in cultural_profile.communication_styles:
            validation_result['concerns'].append("text_too_long_for_direct_style")
            validation_result['confidence'] *= 0.8
        
        # Check for remaining sensitivity issues
        remaining_issues = await self._check_cultural_sensitivity(text, cultural_profile)
        if remaining_issues:
            validation_result['concerns'].extend(remaining_issues)
            validation_result['confidence'] *= 0.6
        
        # Check for coherence after adaptations
        if '[' in text and ']' in text:  # Placeholder text remaining
            validation_result['concerns'].append("incomplete_adaptation")
            validation_result['confidence'] *= 0.4
        
        if validation_result['concerns']:
            validation_result['appropriate'] = validation_result['confidence'] > 0.5
        
        return validation_result
    
    def _generate_adaptation_reasoning(self, adaptations_made: List[str]) -> str:
        """Generate human-readable reasoning for adaptations made"""
        
        if not adaptations_made:
            return "No cultural adaptations needed"
        
        reasoning_parts = []
        
        for adaptation in adaptations_made:
            if 'religious_adaptation' in adaptation:
                reasoning_parts.append("adapted for religious context")
            elif 'collectivist_adaptation' in adaptation:
                reasoning_parts.append("adapted for collectivist culture")
            elif 'individualist_adaptation' in adaptation:
                reasoning_parts.append("adapted for individualist culture")
            elif 'communication_style' in adaptation:
                reasoning_parts.append("adapted for communication preference")
            elif 'sensitivity' in adaptation:
                reasoning_parts.append("resolved cultural sensitivity concern")
        
        return "Culturally adapted: " + ", ".join(set(reasoning_parts))
    
    async def create_cultural_profile(self, user_id: str, 
                                    profile_data: Dict[str, Any]) -> CulturalProfile:
        """Create or update a user's cultural profile"""
        
        try:
            # Parse cultural contexts
            cultural_contexts = []
            if 'cultural_contexts' in profile_data:
                for context_str in profile_data['cultural_contexts']:
                    try:
                        cultural_contexts.append(CulturalContext(context_str))
                    except ValueError:
                        logger.warning(f"Unknown cultural context: {context_str}")
            
            # Parse religious context
            religious_context = None
            if 'religious_context' in profile_data:
                try:
                    religious_context = ReligiousContext(profile_data['religious_context'])
                except ValueError:
                    logger.warning(f"Unknown religious context: {profile_data['religious_context']}")
            
            # Parse communication styles
            communication_styles = []
            if 'communication_styles' in profile_data:
                for style_str in profile_data['communication_styles']:
                    try:
                        communication_styles.append(CommunicationStyle(style_str))
                    except ValueError:
                        logger.warning(f"Unknown communication style: {style_str}")
            
            # Create cultural profile
            profile = CulturalProfile(
                user_id=user_id,
                cultural_contexts=cultural_contexts,
                religious_context=religious_context,
                communication_styles=communication_styles,
                cultural_dimensions=profile_data.get('cultural_dimensions', {}),
                language_preferences=profile_data.get('language_preferences', ['english']),
                cultural_holidays=profile_data.get('cultural_holidays', []),
                sensitive_topics=profile_data.get('sensitive_topics', []),
                preferred_metaphors=profile_data.get('preferred_metaphors', []),
                avoided_concepts=profile_data.get('avoided_concepts', []),
                family_structure=profile_data.get('family_structure', 'not_specified')
            )
            
            # Store profile
            self.cultural_profiles[user_id] = profile
            
            logger.info(f"Created cultural profile for user {user_id[:8]}...")
            return profile
            
        except Exception as e:
            logger.error(f"Error creating cultural profile: {e}")
            # Return minimal default profile
            return CulturalProfile(
                user_id=user_id,
                cultural_contexts=[CulturalContext.UNIVERSAL]
            )
    
    async def get_cultural_profile(self, user_id: str) -> CulturalProfile:
        """Get user's cultural profile or create default"""
        
        if user_id in self.cultural_profiles:
            return self.cultural_profiles[user_id]
        
        # Create default profile
        return CulturalProfile(
            user_id=user_id,
            cultural_contexts=[CulturalContext.UNIVERSAL],
            communication_styles=[CommunicationStyle.DIRECT]
        )
    
    async def get_cultural_calendar_considerations(self, user_id: str, 
                                                 current_date: date) -> Dict[str, Any]:
        """Get cultural calendar considerations for current date"""
        
        profile = await self.get_cultural_profile(user_id)
        considerations = {
            'active_holidays': [],
            'cultural_themes': [],
            'sensitivity_reminders': [],
            'appropriate_content_types': []
        }
        
        # Check for active cultural holidays
        for holiday in profile.cultural_holidays:
            if holiday in self.cultural_calendar:
                holiday_config = self.cultural_calendar[holiday]
                # Simplified date checking - in production would be more sophisticated
                considerations['active_holidays'].append(holiday)
                considerations['cultural_themes'].extend(
                    holiday_config.get('appropriate_themes', [])
                )
        
        return considerations
    
    def get_cultural_sensitivity_metrics(self) -> Dict[str, Any]:
        """Get metrics about cultural sensitivity engine performance"""
        
        total_profiles = len(self.cultural_profiles)
        
        # Count profiles by cultural context
        context_counts = {}
        for profile in self.cultural_profiles.values():
            for context in profile.cultural_contexts:
                context_counts[context.value] = context_counts.get(context.value, 0) + 1
        
        # Count profiles by religious context
        religious_counts = {}
        for profile in self.cultural_profiles.values():
            if profile.religious_context:
                religious_counts[profile.religious_context.value] = \
                    religious_counts.get(profile.religious_context.value, 0) + 1
        
        return {
            'total_cultural_profiles': total_profiles,
            'cultural_contexts_represented': context_counts,
            'religious_contexts_represented': religious_counts,
            'cultural_adaptations_available': len(self.cultural_adaptations),
            'religious_considerations_available': len(self.religious_considerations),
            'communication_styles_supported': len(self.communication_style_guides),
            'cultural_holidays_recognized': len(self.cultural_calendar),
            'sensitivity_areas_monitored': len(self.sensitive_topics_map),
            'inclusive_design_active': True,
            'cultural_humility_practiced': True
        }