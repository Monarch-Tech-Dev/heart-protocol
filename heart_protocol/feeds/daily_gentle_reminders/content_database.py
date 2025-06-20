"""
Affirmation Content Database

A comprehensive, culturally-sensitive database of gentle reminders and affirmations.
Designed to respect diverse backgrounds while affirming universal human worth.
"""

import random
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)


class AffirmationType(Enum):
    """Types of affirmations available"""
    SELF_WORTH = "self_worth"
    PROGRESS_ACKNOWLEDGMENT = "progress_acknowledgment"
    RESILIENCE_BUILDING = "resilience_building"
    HOPE_NURTURING = "hope_nurturing"
    GRATITUDE_CULTIVATION = "gratitude_cultivation"
    SELF_COMPASSION = "self_compassion"
    STRENGTH_RECOGNITION = "strength_recognition"
    HEALING_SUPPORT = "healing_support"
    DAILY_ENCOURAGEMENT = "daily_encouragement"
    CRISIS_COMFORT = "crisis_comfort"


class CulturalContext(Enum):
    """Cultural contexts for affirmations"""
    UNIVERSAL = "universal"  # Works across all cultures
    WESTERN_INDIVIDUALISTIC = "western_individualistic"
    COLLECTIVIST = "collectivist"
    SPIRITUAL_GENERAL = "spiritual_general"
    SECULAR_HUMANISTIC = "secular_humanistic"
    INDIGENOUS_WISDOM = "indigenous_wisdom"
    EASTERN_PHILOSOPHICAL = "eastern_philosophical"
    RELIGIOUS_INTERFAITH = "religious_interfaith"


class Affirmation:
    """Represents a single affirmation with metadata"""
    
    def __init__(self, text: str, affirmation_type: AffirmationType, 
                 cultural_context: CulturalContext, emotional_intensity: float,
                 keywords: List[str] = None, avoid_when: List[str] = None):
        self.text = text
        self.type = affirmation_type
        self.cultural_context = cultural_context
        self.emotional_intensity = emotional_intensity  # 0.0 (gentle) to 1.0 (powerful)
        self.keywords = keywords or []
        self.avoid_when = avoid_when or []  # Situations to avoid this affirmation
        self.usage_count = 0
        self.positive_feedback_count = 0
        self.created_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/transmission"""
        return {
            'text': self.text,
            'type': self.type.value,
            'cultural_context': self.cultural_context.value,
            'emotional_intensity': self.emotional_intensity,
            'keywords': self.keywords,
            'avoid_when': self.avoid_when,
            'usage_count': self.usage_count,
            'positive_feedback_count': self.positive_feedback_count,
            'effectiveness_score': self.get_effectiveness_score(),
            'created_at': self.created_at.isoformat()
        }
    
    def get_effectiveness_score(self) -> float:
        """Calculate effectiveness score based on usage and feedback"""
        if self.usage_count == 0:
            return 0.5  # Neutral for new affirmations
        
        return min(1.0, self.positive_feedback_count / self.usage_count)
    
    def record_usage(self, positive_feedback: bool = False):
        """Record usage and feedback"""
        self.usage_count += 1
        if positive_feedback:
            self.positive_feedback_count += 1


class AffirmationDatabase:
    """
    Comprehensive database of caring affirmations with cultural sensitivity.
    
    Based on research in positive psychology, cultural psychology, and
    trauma-informed care practices.
    """
    
    def __init__(self):
        self.affirmations = []
        self.cultural_variants = {}
        self.seasonal_affirmations = {}
        self.crisis_specific_affirmations = []
        
        # Initialize with core affirmations
        self._initialize_core_affirmations()
        self._initialize_cultural_variants()
        self._initialize_seasonal_content()
        self._initialize_crisis_affirmations()
        
        logger.info(f"Affirmation database initialized with {len(self.affirmations)} affirmations")
    
    def _initialize_core_affirmations(self):
        """Initialize core affirmations for universal human worth"""
        
        # Self-Worth Affirmations
        self_worth_affirmations = [
            Affirmation(
                "You are worthy of love and belonging, exactly as you are.",
                AffirmationType.SELF_WORTH,
                CulturalContext.UNIVERSAL,
                0.8,
                keywords=["worth", "love", "belonging"],
                avoid_when=["acute_grief", "relationship_trauma"]
            ),
            Affirmation(
                "Your existence has value that doesn't depend on productivity or achievement.",
                AffirmationType.SELF_WORTH,
                CulturalContext.WESTERN_INDIVIDUALISTIC,
                0.7,
                keywords=["existence", "value", "productivity"],
                avoid_when=["work_trauma", "burnout"]
            ),
            Affirmation(
                "You matter to this world in ways you may never fully know.",
                AffirmationType.SELF_WORTH,
                CulturalContext.UNIVERSAL,
                0.6,
                keywords=["matter", "world", "impact"]
            ),
            Affirmation(
                "Your heart carries wisdom earned through every experience you've survived.",
                AffirmationType.SELF_WORTH,
                CulturalContext.UNIVERSAL,
                0.9,
                keywords=["wisdom", "survived", "experience"]
            ),
            Affirmation(
                "You are enough, have always been enough, and will always be enough.",
                AffirmationType.SELF_WORTH,
                CulturalContext.UNIVERSAL,
                0.8,
                keywords=["enough", "always", "sufficient"]
            )
        ]
        
        # Progress Acknowledgment
        progress_affirmations = [
            Affirmation(
                "Every small step forward is worth celebrating.",
                AffirmationType.PROGRESS_ACKNOWLEDGMENT,
                CulturalContext.UNIVERSAL,
                0.5,
                keywords=["step", "forward", "celebrating"]
            ),
            Affirmation(
                "Healing isn't linear, and that's perfectly okay.",
                AffirmationType.PROGRESS_ACKNOWLEDGMENT,
                CulturalContext.UNIVERSAL,
                0.7,
                keywords=["healing", "linear", "okay"]
            ),
            Affirmation(
                "You've grown in ways you might not even recognize yet.",
                AffirmationType.PROGRESS_ACKNOWLEDGMENT,
                CulturalContext.UNIVERSAL,
                0.6,
                keywords=["grown", "recognize", "change"]
            ),
            Affirmation(
                "Your journey is unfolding exactly as it needs to.",
                AffirmationType.PROGRESS_ACKNOWLEDGMENT,
                CulturalContext.SPIRITUAL_GENERAL,
                0.5,
                keywords=["journey", "unfolding", "needs"]
            )
        ]
        
        # Resilience Building
        resilience_affirmations = [
            Affirmation(
                "You have survived 100% of your difficult days so far.",
                AffirmationType.RESILIENCE_BUILDING,
                CulturalContext.UNIVERSAL,
                0.8,
                keywords=["survived", "difficult", "strength"]
            ),
            Affirmation(
                "Your resilience is built from every challenge you've faced.",
                AffirmationType.RESILIENCE_BUILDING,
                CulturalContext.UNIVERSAL,
                0.7,
                keywords=["resilience", "challenge", "faced"]
            ),
            Affirmation(
                "You are stronger than you think, braver than you feel, and more loved than you know.",
                AffirmationType.RESILIENCE_BUILDING,
                CulturalContext.UNIVERSAL,
                0.9,
                keywords=["stronger", "braver", "loved"]
            ),
            Affirmation(
                "The fact that you're still here means you're a fighter.",
                AffirmationType.RESILIENCE_BUILDING,
                CulturalContext.UNIVERSAL,
                0.8,
                keywords=["still here", "fighter", "perseverance"]
            )
        ]
        
        # Hope Nurturing
        hope_affirmations = [
            Affirmation(
                "Tomorrow holds possibilities you can't see today.",
                AffirmationType.HOPE_NURTURING,
                CulturalContext.UNIVERSAL,
                0.6,
                keywords=["tomorrow", "possibilities", "future"]
            ),
            Affirmation(
                "This difficult chapter is not your whole story.",
                AffirmationType.HOPE_NURTURING,
                CulturalContext.UNIVERSAL,
                0.7,
                keywords=["chapter", "story", "whole"]
            ),
            Affirmation(
                "You have weathered storms before, and you will again.",
                AffirmationType.HOPE_NURTURING,
                CulturalContext.UNIVERSAL,
                0.8,
                keywords=["weathered", "storms", "again"]
            ),
            Affirmation(
                "Your presence here today is proof of your incredible strength.",
                AffirmationType.HOPE_NURTURING,
                CulturalContext.UNIVERSAL,
                0.7,
                keywords=["presence", "proof", "strength"]
            )
        ]
        
        # Self-Compassion
        compassion_affirmations = [
            Affirmation(
                "You deserve the same kindness you would offer a dear friend.",
                AffirmationType.SELF_COMPASSION,
                CulturalContext.UNIVERSAL,
                0.6,
                keywords=["deserve", "kindness", "friend"]
            ),
            Affirmation(
                "It's okay to rest when you're tired, even if others don't understand.",
                AffirmationType.SELF_COMPASSION,
                CulturalContext.UNIVERSAL,
                0.5,
                keywords=["rest", "tired", "understand"]
            ),
            Affirmation(
                "Your feelings are valid, even when they're complicated.",
                AffirmationType.SELF_COMPASSION,
                CulturalContext.UNIVERSAL,
                0.6,
                keywords=["feelings", "valid", "complicated"]
            ),
            Affirmation(
                "You're allowed to take up space in this world.",
                AffirmationType.SELF_COMPASSION,
                CulturalContext.WESTERN_INDIVIDUALISTIC,
                0.7,
                keywords=["allowed", "space", "world"]
            )
        ]
        
        # Healing Support
        healing_affirmations = [
            Affirmation(
                "Healing happens in its own time, and that's wisdom, not weakness.",
                AffirmationType.HEALING_SUPPORT,
                CulturalContext.UNIVERSAL,
                0.7,
                keywords=["healing", "time", "wisdom"]
            ),
            Affirmation(
                "Your scars are evidence of your healing, not your damage.",
                AffirmationType.HEALING_SUPPORT,
                CulturalContext.UNIVERSAL,
                0.8,
                keywords=["scars", "evidence", "healing"]
            ),
            Affirmation(
                "Every breath you take is an act of courage and hope.",
                AffirmationType.HEALING_SUPPORT,
                CulturalContext.UNIVERSAL,
                0.6,
                keywords=["breath", "courage", "hope"]
            ),
            Affirmation(
                "You are allowed to heal at your own pace.",
                AffirmationType.HEALING_SUPPORT,
                CulturalContext.UNIVERSAL,
                0.5,
                keywords=["allowed", "heal", "pace"]
            )
        ]
        
        # Combine all affirmations
        all_affirmations = (
            self_worth_affirmations + progress_affirmations + 
            resilience_affirmations + hope_affirmations + 
            compassion_affirmations + healing_affirmations
        )
        
        self.affirmations.extend(all_affirmations)
    
    def _initialize_cultural_variants(self):
        """Initialize cultural variants of core affirmations"""
        
        # Collectivist culture variants (emphasizing community and relationships)
        collectivist_variants = [
            Affirmation(
                "Your community is stronger because you are part of it.",
                AffirmationType.SELF_WORTH,
                CulturalContext.COLLECTIVIST,
                0.7,
                keywords=["community", "stronger", "part"]
            ),
            Affirmation(
                "The wisdom of your ancestors flows through you.",
                AffirmationType.STRENGTH_RECOGNITION,
                CulturalContext.INDIGENOUS_WISDOM,
                0.8,
                keywords=["ancestors", "wisdom", "flows"]
            ),
            Affirmation(
                "You carry the love and hopes of those who came before you.",
                AffirmationType.SELF_WORTH,
                CulturalContext.COLLECTIVIST,
                0.9,
                keywords=["carry", "love", "hopes", "before"]
            )
        ]
        
        # Eastern philosophical variants
        eastern_variants = [
            Affirmation(
                "Like water, you have the strength to flow around any obstacle.",
                AffirmationType.RESILIENCE_BUILDING,
                CulturalContext.EASTERN_PHILOSOPHICAL,
                0.7,
                keywords=["water", "flow", "obstacle"]
            ),
            Affirmation(
                "In stillness, you can find the peace that was always within you.",
                AffirmationType.SELF_COMPASSION,
                CulturalContext.EASTERN_PHILOSOPHICAL,
                0.6,
                keywords=["stillness", "peace", "within"]
            ),
            Affirmation(
                "Your present moment contains infinite possibilities.",
                AffirmationType.HOPE_NURTURING,
                CulturalContext.EASTERN_PHILOSOPHICAL,
                0.6,
                keywords=["present", "infinite", "possibilities"]
            )
        ]
        
        # Spiritual (interfaith) variants
        spiritual_variants = [
            Affirmation(
                "You are a beloved child of the universe.",
                AffirmationType.SELF_WORTH,
                CulturalContext.SPIRITUAL_GENERAL,
                0.8,
                keywords=["beloved", "child", "universe"]
            ),
            Affirmation(
                "There is a light within you that cannot be extinguished.",
                AffirmationType.HOPE_NURTURING,
                CulturalContext.SPIRITUAL_GENERAL,
                0.9,
                keywords=["light", "within", "extinguished"]
            ),
            Affirmation(
                "You are held by love larger than you can comprehend.",
                AffirmationType.SELF_WORTH,
                CulturalContext.SPIRITUAL_GENERAL,
                0.8,
                keywords=["held", "love", "larger"]
            )
        ]
        
        # Secular humanistic variants
        secular_variants = [
            Affirmation(
                "Your capacity for growth and change is scientifically remarkable.",
                AffirmationType.PROGRESS_ACKNOWLEDGMENT,
                CulturalContext.SECULAR_HUMANISTIC,
                0.6,
                keywords=["capacity", "growth", "remarkable"]
            ),
            Affirmation(
                "Human connection and compassion are your evolutionary strengths.",
                AffirmationType.STRENGTH_RECOGNITION,
                CulturalContext.SECULAR_HUMANISTIC,
                0.7,
                keywords=["connection", "compassion", "evolutionary"]
            ),
            Affirmation(
                "Your mind has the power to literally rewire itself for healing.",
                AffirmationType.HEALING_SUPPORT,
                CulturalContext.SECULAR_HUMANISTIC,
                0.8,
                keywords=["mind", "rewire", "healing"]
            )
        ]
        
        # Add all cultural variants
        self.affirmations.extend(collectivist_variants + eastern_variants + 
                               spiritual_variants + secular_variants)
    
    def _initialize_seasonal_content(self):
        """Initialize seasonal and time-based affirmations"""
        
        self.seasonal_affirmations = {
            'winter': [
                Affirmation(
                    "Like trees in winter, rest is preparation for your next season of growth.",
                    AffirmationType.SELF_COMPASSION,
                    CulturalContext.UNIVERSAL,
                    0.6,
                    keywords=["winter", "rest", "growth"]
                ),
                Affirmation(
                    "Even in the darkest season, your inner light continues to shine.",
                    AffirmationType.HOPE_NURTURING,
                    CulturalContext.UNIVERSAL,
                    0.7,
                    keywords=["dark", "light", "shine"]
                )
            ],
            'spring': [
                Affirmation(
                    "Like the earth awakening, you too are ready for new beginnings.",
                    AffirmationType.HOPE_NURTURING,
                    CulturalContext.UNIVERSAL,
                    0.7,
                    keywords=["awakening", "beginnings", "ready"]
                ),
                Affirmation(
                    "Your growth, like spring flowers, happens gradually and beautifully.",
                    AffirmationType.PROGRESS_ACKNOWLEDGMENT,
                    CulturalContext.UNIVERSAL,
                    0.6,
                    keywords=["growth", "flowers", "gradually"]
                )
            ],
            'summer': [
                Affirmation(
                    "You are in full bloom, sharing your unique beauty with the world.",
                    AffirmationType.STRENGTH_RECOGNITION,
                    CulturalContext.UNIVERSAL,
                    0.8,
                    keywords=["bloom", "beauty", "sharing"]
                ),
                Affirmation(
                    "Like the sun, your warmth and light touch more lives than you know.",
                    AffirmationType.SELF_WORTH,
                    CulturalContext.UNIVERSAL,
                    0.7,
                    keywords=["sun", "warmth", "lives"]
                )
            ],
            'autumn': [
                Affirmation(
                    "Like autumn leaves, letting go can be beautiful and necessary.",
                    AffirmationType.HEALING_SUPPORT,
                    CulturalContext.UNIVERSAL,
                    0.6,
                    keywords=["letting go", "beautiful", "necessary"]
                ),
                Affirmation(
                    "Your wisdom, like autumn colors, becomes more vibrant with time.",
                    AffirmationType.STRENGTH_RECOGNITION,
                    CulturalContext.UNIVERSAL,
                    0.7,
                    keywords=["wisdom", "colors", "vibrant"]
                )
            ]
        }
    
    def _initialize_crisis_affirmations(self):
        """Initialize affirmations specifically for crisis situations"""
        
        self.crisis_specific_affirmations = [
            Affirmation(
                "Right now, in this moment, you are safe.",
                AffirmationType.CRISIS_COMFORT,
                CulturalContext.UNIVERSAL,
                0.5,
                keywords=["right now", "moment", "safe"]
            ),
            Affirmation(
                "This feeling is temporary, even when it doesn't feel that way.",
                AffirmationType.CRISIS_COMFORT,
                CulturalContext.UNIVERSAL,
                0.6,
                keywords=["temporary", "feeling", "way"]
            ),
            Affirmation(
                "You have people who care about you, even if you can't feel it right now.",
                AffirmationType.CRISIS_COMFORT,
                CulturalContext.UNIVERSAL,
                0.7,
                keywords=["people", "care", "feel"]
            ),
            Affirmation(
                "Your life has value that extends beyond this moment of pain.",
                AffirmationType.CRISIS_COMFORT,
                CulturalContext.UNIVERSAL,
                0.8,
                keywords=["life", "value", "beyond", "pain"]
            ),
            Affirmation(
                "There are people trained to help you through this. You don't have to face it alone.",
                AffirmationType.CRISIS_COMFORT,
                CulturalContext.UNIVERSAL,
                0.6,
                keywords=["trained", "help", "alone"]
            )
        ]
    
    async def get_personalized_affirmation(self, user_context: Dict[str, Any]) -> Optional[Affirmation]:
        """
        Get a personalized affirmation based on user context.
        
        Args:
            user_context: Dictionary containing:
                - emotional_capacity: Dict with user's current emotional state
                - cultural_preference: CulturalContext preference
                - recent_feedback: List of recent feedback on affirmations
                - crisis_indicators: Boolean indicating crisis state
                - time_context: Current time/season information
                - avoided_types: List of affirmation types to avoid
        """
        try:
            # Handle crisis situations first
            if user_context.get('crisis_indicators', False):
                return await self._get_crisis_affirmation(user_context)
            
            # Get appropriate affirmations pool
            candidate_affirmations = await self._filter_appropriate_affirmations(user_context)
            
            if not candidate_affirmations:
                # Fallback to universal affirmations
                candidate_affirmations = [
                    a for a in self.affirmations 
                    if a.cultural_context == CulturalContext.UNIVERSAL
                ]
            
            # Score and select best affirmation
            scored_affirmations = await self._score_affirmations_for_user(
                candidate_affirmations, user_context
            )
            
            if scored_affirmations:
                # Select from top-scored affirmations with some randomness for variety
                top_affirmations = scored_affirmations[:min(5, len(scored_affirmations))]
                selected = random.choice(top_affirmations)[1]  # (score, affirmation)
                
                logger.debug(f"Selected affirmation: {selected.type.value} for user context")
                return selected
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting personalized affirmation: {e}")
            return await self._get_fallback_affirmation()
    
    async def _get_crisis_affirmation(self, user_context: Dict[str, Any]) -> Affirmation:
        """Get appropriate affirmation for crisis situations"""
        # Use crisis-specific affirmations with gentle emotional intensity
        gentle_crisis_affirmations = [
            a for a in self.crisis_specific_affirmations 
            if a.emotional_intensity <= 0.7
        ]
        
        if gentle_crisis_affirmations:
            return random.choice(gentle_crisis_affirmations)
        
        # Fallback crisis affirmation
        return Affirmation(
            "You matter, and there are people who want to help you through this.",
            AffirmationType.CRISIS_COMFORT,
            CulturalContext.UNIVERSAL,
            0.6,
            keywords=["matter", "help", "through"]
        )
    
    async def _filter_appropriate_affirmations(self, user_context: Dict[str, Any]) -> List[Affirmation]:
        """Filter affirmations based on user context and preferences"""
        appropriate_affirmations = []
        
        cultural_pref = user_context.get('cultural_preference', CulturalContext.UNIVERSAL)
        emotional_capacity = user_context.get('emotional_capacity', {})
        avoided_types = user_context.get('avoided_types', [])
        
        for affirmation in self.affirmations:
            # Skip if type should be avoided
            if affirmation.type.value in avoided_types:
                continue
            
            # Check cultural compatibility
            if (affirmation.cultural_context != CulturalContext.UNIVERSAL and 
                affirmation.cultural_context != cultural_pref):
                continue
            
            # Check emotional intensity appropriateness
            max_intensity = emotional_capacity.get('can_handle_intensity', 0.7)
            if affirmation.emotional_intensity > max_intensity:
                continue
            
            # Check for avoided situations
            user_situation = user_context.get('current_situation', [])
            if any(avoid in user_situation for avoid in affirmation.avoid_when):
                continue
            
            appropriate_affirmations.append(affirmation)
        
        return appropriate_affirmations
    
    async def _score_affirmations_for_user(self, affirmations: List[Affirmation], 
                                         user_context: Dict[str, Any]) -> List[Tuple[float, Affirmation]]:
        """Score affirmations based on user context and effectiveness"""
        scored_affirmations = []
        
        for affirmation in affirmations:
            score = 0.0
            
            # Base effectiveness score
            score += affirmation.get_effectiveness_score() * 0.4
            
            # Emotional appropriateness
            emotional_capacity = user_context.get('emotional_capacity', {})
            if affirmation.emotional_intensity <= emotional_capacity.get('preferred_intensity', 0.7):
                score += 0.3
            
            # Type preference matching
            preferred_types = user_context.get('preferred_affirmation_types', [])
            if affirmation.type.value in preferred_types:
                score += 0.2
            
            # Keyword matching with user needs
            user_keywords = user_context.get('current_needs_keywords', [])
            keyword_matches = len(set(affirmation.keywords) & set(user_keywords))
            score += min(0.2, keyword_matches * 0.05)
            
            # Seasonal appropriateness
            current_season = user_context.get('season')
            if current_season and affirmation in self.seasonal_affirmations.get(current_season, []):
                score += 0.1
            
            # Diversity bonus (prefer less recently used affirmations)
            recent_affirmations = user_context.get('recent_affirmations', [])
            if affirmation.text not in recent_affirmations:
                score += 0.1
            
            scored_affirmations.append((score, affirmation))
        
        # Sort by score (highest first)
        scored_affirmations.sort(key=lambda x: x[0], reverse=True)
        return scored_affirmations
    
    async def _get_fallback_affirmation(self) -> Affirmation:
        """Get a safe fallback affirmation"""
        return Affirmation(
            "You are worthy of love and care.",
            AffirmationType.SELF_WORTH,
            CulturalContext.UNIVERSAL,
            0.5,
            keywords=["worthy", "love", "care"]
        )
    
    def add_affirmation(self, affirmation: Affirmation) -> bool:
        """Add a new affirmation to the database"""
        try:
            self.affirmations.append(affirmation)
            logger.info(f"Added new affirmation: {affirmation.type.value}")
            return True
        except Exception as e:
            logger.error(f"Error adding affirmation: {e}")
            return False
    
    def get_affirmations_by_type(self, affirmation_type: AffirmationType) -> List[Affirmation]:
        """Get all affirmations of a specific type"""
        return [a for a in self.affirmations if a.type == affirmation_type]
    
    def get_affirmations_by_culture(self, cultural_context: CulturalContext) -> List[Affirmation]:
        """Get all affirmations for a specific cultural context"""
        return [a for a in self.affirmations if a.cultural_context == cultural_context]
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the affirmation database"""
        stats = {
            'total_affirmations': len(self.affirmations),
            'by_type': {},
            'by_culture': {},
            'by_intensity': {
                'gentle (0.0-0.3)': 0,
                'moderate (0.4-0.6)': 0,
                'strong (0.7-1.0)': 0
            },
            'total_usage': sum(a.usage_count for a in self.affirmations),
            'average_effectiveness': sum(a.get_effectiveness_score() for a in self.affirmations) / len(self.affirmations),
            'crisis_affirmations': len(self.crisis_specific_affirmations),
            'seasonal_affirmations': sum(len(affirmations) for affirmations in self.seasonal_affirmations.values())
        }
        
        # Count by type
        for affirmation_type in AffirmationType:
            count = len([a for a in self.affirmations if a.type == affirmation_type])
            stats['by_type'][affirmation_type.value] = count
        
        # Count by culture
        for cultural_context in CulturalContext:
            count = len([a for a in self.affirmations if a.cultural_context == cultural_context])
            stats['by_culture'][cultural_context.value] = count
        
        # Count by intensity
        for affirmation in self.affirmations:
            if affirmation.emotional_intensity <= 0.3:
                stats['by_intensity']['gentle (0.0-0.3)'] += 1
            elif affirmation.emotional_intensity <= 0.6:
                stats['by_intensity']['moderate (0.4-0.6)'] += 1
            else:
                stats['by_intensity']['strong (0.7-1.0)'] += 1
        
        return stats


class CulturalAffirmations:
    """
    Helper class for managing cultural adaptations of affirmations.
    Ensures respectful and appropriate messaging across diverse backgrounds.
    """
    
    @staticmethod
    def adapt_for_culture(affirmation: Affirmation, target_culture: CulturalContext) -> Optional[Affirmation]:
        """Adapt an affirmation for a specific cultural context"""
        
        # Cultural adaptation mappings
        adaptations = {
            # Individual-focused to community-focused
            (CulturalContext.WESTERN_INDIVIDUALISTIC, CulturalContext.COLLECTIVIST): {
                "You are": "You and your community are",
                "your strength": "the strength you share with others",
                "you matter": "you matter to your community"
            },
            
            # Universal to spiritual
            (CulturalContext.UNIVERSAL, CulturalContext.SPIRITUAL_GENERAL): {
                "universe": "divine presence",
                "strength": "blessed strength",
                "love": "divine love"
            },
            
            # Add more adaptations as needed
        }
        
        adaptation_key = (affirmation.cultural_context, target_culture)
        if adaptation_key in adaptations:
            adapted_text = affirmation.text
            
            for original, replacement in adaptations[adaptation_key].items():
                adapted_text = adapted_text.replace(original, replacement)
            
            return Affirmation(
                adapted_text,
                affirmation.type,
                target_culture,
                affirmation.emotional_intensity,
                affirmation.keywords.copy(),
                affirmation.avoid_when.copy()
            )
        
        return None