"""
Emotional Capacity Personalization Engine

Personalizes gentle reminders based on user's current emotional capacity,
mental health status, and personal preferences. Ensures affirmations are
emotionally appropriate and supportive rather than overwhelming.
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum
import logging

from .content_database import AffirmationType, CulturalContext, Affirmation

logger = logging.getLogger(__name__)


class EmotionalCapacityLevel(Enum):
    """User's current emotional capacity levels"""
    VERY_LOW = "very_low"        # Crisis/severe distress
    LOW = "low"                  # Struggling, overwhelmed
    MODERATE = "moderate"        # Managing but stressed
    GOOD = "good"               # Stable and coping well
    HIGH = "high"               # Thriving, resilient


class PersonalizationFactors(Enum):
    """Factors used in personalization"""
    EMOTIONAL_CAPACITY = "emotional_capacity"
    TRAUMA_HISTORY = "trauma_history"
    CURRENT_STRESSORS = "current_stressors"
    SUPPORT_SYSTEM = "support_system"
    COPING_STRATEGIES = "coping_strategies"
    PERSONAL_VALUES = "personal_values"
    COMMUNICATION_STYLE = "communication_style"
    RECENT_FEEDBACK = "recent_feedback"


class EmotionalCapacityPersonalizer:
    """
    Personalizes affirmations based on emotional capacity and individual needs.
    
    Uses trauma-informed care principles and positive psychology research
    to ensure affirmations are supportive rather than triggering.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.user_profiles = {}  # In production, this would be database-backed
        
        # Personalization rules based on emotional capacity
        self.capacity_rules = self._initialize_capacity_rules()
        
        # Trauma-informed considerations
        self.trauma_considerations = self._initialize_trauma_considerations()
        
        logger.info("Emotional Capacity Personalizer initialized")
    
    def _initialize_capacity_rules(self) -> Dict[EmotionalCapacityLevel, Dict[str, Any]]:
        """Initialize personalization rules for different emotional capacity levels"""
        
        return {
            EmotionalCapacityLevel.VERY_LOW: {
                'max_emotional_intensity': 0.3,
                'preferred_types': [
                    AffirmationType.CRISIS_COMFORT,
                    AffirmationType.SELF_COMPASSION
                ],
                'avoid_types': [
                    AffirmationType.PROGRESS_ACKNOWLEDGMENT,  # May feel invalidating
                    AffirmationType.STRENGTH_RECOGNITION     # May feel overwhelming
                ],
                'message_length': 'short',  # Brief, simple messages
                'frequency': 'as_needed',   # Not scheduled, responsive
                'tone': 'very_gentle',
                'focus_areas': ['safety', 'immediate_comfort', 'basic_worth']
            },
            
            EmotionalCapacityLevel.LOW: {
                'max_emotional_intensity': 0.5,
                'preferred_types': [
                    AffirmationType.SELF_COMPASSION,
                    AffirmationType.HEALING_SUPPORT,
                    AffirmationType.SELF_WORTH
                ],
                'avoid_types': [
                    AffirmationType.RESILIENCE_BUILDING  # May feel pressuring
                ],
                'message_length': 'short_to_medium',
                'frequency': 'gentle_daily',
                'tone': 'gentle_supportive',
                'focus_areas': ['self_acceptance', 'healing', 'patience']
            },
            
            EmotionalCapacityLevel.MODERATE: {
                'max_emotional_intensity': 0.7,
                'preferred_types': [
                    AffirmationType.PROGRESS_ACKNOWLEDGMENT,
                    AffirmationType.SELF_WORTH,
                    AffirmationType.HOPE_NURTURING
                ],
                'avoid_types': [],  # No specific avoidances
                'message_length': 'medium',
                'frequency': 'daily',
                'tone': 'encouraging_supportive',
                'focus_areas': ['progress', 'hope', 'growth']
            },
            
            EmotionalCapacityLevel.GOOD: {
                'max_emotional_intensity': 0.8,
                'preferred_types': [
                    AffirmationType.STRENGTH_RECOGNITION,
                    AffirmationType.GRATITUDE_CULTIVATION,
                    AffirmationType.DAILY_ENCOURAGEMENT
                ],
                'avoid_types': [],
                'message_length': 'medium_to_long',
                'frequency': 'daily_plus',
                'tone': 'uplifting_empowering',
                'focus_areas': ['strengths', 'gratitude', 'contribution']
            },
            
            EmotionalCapacityLevel.HIGH: {
                'max_emotional_intensity': 1.0,
                'preferred_types': [
                    AffirmationType.STRENGTH_RECOGNITION,
                    AffirmationType.GRATITUDE_CULTIVATION,
                    AffirmationType.RESILIENCE_BUILDING
                ],
                'avoid_types': [],
                'message_length': 'flexible',
                'frequency': 'as_desired',
                'tone': 'empowering_inspirational',
                'focus_areas': ['leadership', 'service', 'growth']
            }
        }
    
    def _initialize_trauma_considerations(self) -> Dict[str, Dict[str, Any]]:
        """Initialize trauma-informed personalization considerations"""
        
        return {
            'attachment_trauma': {
                'avoid_keywords': ['abandoned', 'left', 'rejection'],
                'preferred_themes': ['belonging', 'connection', 'acceptance'],
                'gentle_approach': True,
                'emphasize_safety': True
            },
            
            'childhood_trauma': {
                'avoid_keywords': ['family', 'childhood', 'parent'],
                'preferred_themes': ['inner_child', 'healing', 'new_beginnings'],
                'gentle_approach': True,
                'emphasize_choice': True
            },
            
            'relationship_trauma': {
                'avoid_keywords': ['partner', 'relationship', 'love'],
                'preferred_themes': ['self_love', 'independence', 'healing'],
                'gentle_approach': True,
                'emphasize_self_worth': True
            },
            
            'work_trauma': {
                'avoid_keywords': ['productivity', 'achievement', 'success'],
                'preferred_themes': ['inherent_worth', 'rest', 'boundaries'],
                'gentle_approach': True,
                'emphasize_unconditional_worth': True
            },
            
            'grief_loss': {
                'avoid_keywords': ['moving_on', 'getting_over', 'replacement'],
                'preferred_themes': ['honoring_memory', 'carrying_love', 'gentle_healing'],
                'gentle_approach': True,
                'emphasize_patience': True
            }
        }
    
    async def assess_emotional_capacity(self, user_id: str, 
                                      current_indicators: Dict[str, Any]) -> EmotionalCapacityLevel:
        """
        Assess user's current emotional capacity based on various indicators.
        
        Args:
            user_id: User identifier
            current_indicators: Dict containing:
                - recent_care_requests: Number of recent help-seeking posts
                - crisis_indicators: Boolean for crisis state
                - stress_indicators: List of current stressors
                - coping_responses: Recent coping behaviors
                - social_engagement: Level of social interaction
                - sleep_patterns: If available, sleep quality indicators
                - user_self_assessment: Self-reported emotional state
        """
        try:
            # Start with moderate as baseline
            capacity_score = 0.5
            
            # Crisis indicators (immediate adjustment)
            if current_indicators.get('crisis_indicators', False):
                return EmotionalCapacityLevel.VERY_LOW
            
            # Recent care requests (higher frequency = lower capacity)
            care_requests = current_indicators.get('recent_care_requests', 0)
            if care_requests > 5:
                capacity_score -= 0.3
            elif care_requests > 2:
                capacity_score -= 0.2
            
            # Stress indicators
            stress_indicators = current_indicators.get('stress_indicators', [])
            stress_impact = len(stress_indicators) * 0.1
            capacity_score -= min(0.4, stress_impact)
            
            # Positive coping responses (increase capacity)
            coping_responses = current_indicators.get('positive_coping_responses', [])
            coping_boost = len(coping_responses) * 0.1
            capacity_score += min(0.3, coping_boost)
            
            # Social engagement (positive indicator)
            social_engagement = current_indicators.get('social_engagement_level', 0.5)  # 0-1 scale
            capacity_score += (social_engagement - 0.5) * 0.2
            
            # User self-assessment (if available)
            self_assessment = current_indicators.get('user_self_assessment')
            if self_assessment:
                # User knows themselves best - weight heavily
                capacity_score = (capacity_score * 0.3) + (self_assessment * 0.7)
            
            # Convert score to capacity level
            if capacity_score <= 0.2:
                return EmotionalCapacityLevel.VERY_LOW
            elif capacity_score <= 0.4:
                return EmotionalCapacityLevel.LOW
            elif capacity_score <= 0.6:
                return EmotionalCapacityLevel.MODERATE
            elif capacity_score <= 0.8:
                return EmotionalCapacityLevel.GOOD
            else:
                return EmotionalCapacityLevel.HIGH
                
        except Exception as e:
            logger.error(f"Error assessing emotional capacity: {e}")
            # Safe fallback
            return EmotionalCapacityLevel.MODERATE
    
    async def create_personalization_context(self, user_id: str, 
                                           capacity_level: EmotionalCapacityLevel,
                                           user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create comprehensive personalization context for affirmation selection.
        
        Args:
            user_id: User identifier
            capacity_level: Current emotional capacity
            user_profile: User's profile including preferences, history, etc.
        """
        try:
            # Get capacity rules
            capacity_rules = self.capacity_rules[capacity_level]
            
            # Build personalization context
            context = {
                'emotional_capacity': {
                    'level': capacity_level.value,
                    'can_handle_intensity': capacity_rules['max_emotional_intensity'],
                    'preferred_intensity': capacity_rules['max_emotional_intensity'] * 0.8,
                    'needs_gentle_approach': capacity_level in [
                        EmotionalCapacityLevel.VERY_LOW, 
                        EmotionalCapacityLevel.LOW
                    ]
                },
                
                'preferred_affirmation_types': [t.value for t in capacity_rules['preferred_types']],
                'avoided_types': [t.value for t in capacity_rules['avoid_types']],
                
                'cultural_preference': user_profile.get('cultural_context', CulturalContext.UNIVERSAL),
                'communication_style': capacity_rules['tone'],
                'message_length_preference': capacity_rules['message_length'],
                
                'current_needs_keywords': await self._identify_current_needs(
                    user_profile, capacity_level
                ),
                
                'avoided_keywords': await self._get_avoided_keywords(user_profile),
                
                'current_situation': user_profile.get('current_stressors', []),
                
                'recent_affirmations': user_profile.get('recent_affirmations', []),
                
                'personalization_factors': await self._analyze_personalization_factors(
                    user_profile, capacity_level
                )
            }
            
            # Add trauma-informed considerations
            trauma_history = user_profile.get('trauma_history', [])
            for trauma_type in trauma_history:
                if trauma_type in self.trauma_considerations:
                    trauma_config = self.trauma_considerations[trauma_type]
                    
                    # Add avoided keywords
                    context['avoided_keywords'].extend(trauma_config['avoid_keywords'])
                    
                    # Add preferred themes as keywords
                    context['current_needs_keywords'].extend(trauma_config['preferred_themes'])
                    
                    # Adjust emotional intensity if needed
                    if trauma_config['gentle_approach']:
                        context['emotional_capacity']['can_handle_intensity'] *= 0.8
            
            logger.debug(f"Created personalization context for capacity level: {capacity_level.value}")
            return context
            
        except Exception as e:
            logger.error(f"Error creating personalization context: {e}")
            return await self._get_fallback_context()
    
    async def _identify_current_needs(self, user_profile: Dict[str, Any], 
                                    capacity_level: EmotionalCapacityLevel) -> List[str]:
        """Identify keywords that match user's current needs"""
        
        needs_keywords = []
        
        # Based on capacity level
        capacity_needs = {
            EmotionalCapacityLevel.VERY_LOW: ['safe', 'comfort', 'present', 'breathe'],
            EmotionalCapacityLevel.LOW: ['healing', 'patience', 'gentle', 'care'],
            EmotionalCapacityLevel.MODERATE: ['progress', 'hope', 'growth', 'support'],
            EmotionalCapacityLevel.GOOD: ['strength', 'gratitude', 'connection', 'joy'],
            EmotionalCapacityLevel.HIGH: ['purpose', 'contribution', 'leadership', 'inspiration']
        }
        
        needs_keywords.extend(capacity_needs.get(capacity_level, []))
        
        # Based on current stressors
        stressors = user_profile.get('current_stressors', [])
        stressor_needs = {
            'work_stress': ['boundaries', 'worth', 'rest'],
            'relationship_issues': ['love', 'connection', 'understanding'],
            'health_concerns': ['strength', 'healing', 'hope'],
            'financial_stress': ['security', 'enough', 'abundance'],
            'family_issues': ['peace', 'acceptance', 'love'],
            'grief': ['memory', 'love', 'time', 'gentle']
        }
        
        for stressor in stressors:
            if stressor in stressor_needs:
                needs_keywords.extend(stressor_needs[stressor])
        
        # Based on personal values
        values = user_profile.get('personal_values', [])
        value_keywords = {
            'family': ['connection', 'love', 'belonging'],
            'achievement': ['progress', 'growth', 'purpose'],
            'spirituality': ['peace', 'divine', 'sacred'],
            'creativity': ['expression', 'beauty', 'unique'],
            'service': ['contribution', 'impact', 'others']
        }
        
        for value in values:
            if value in value_keywords:
                needs_keywords.extend(value_keywords[value])
        
        return list(set(needs_keywords))  # Remove duplicates
    
    async def _get_avoided_keywords(self, user_profile: Dict[str, Any]) -> List[str]:
        """Get keywords that should be avoided for this user"""
        
        avoided_keywords = []
        
        # User-specified triggers
        user_triggers = user_profile.get('trigger_warnings', [])
        avoided_keywords.extend(user_triggers)
        
        # Based on trauma history (handled in main function)
        
        # Based on current sensitivities
        sensitivities = user_profile.get('current_sensitivities', [])
        sensitivity_keywords = {
            'body_image': ['body', 'appearance', 'weight', 'looks'],
            'perfectionism': ['perfect', 'flawless', 'ideal', 'mistake'],
            'abandonment': ['left', 'alone', 'abandoned', 'forgotten'],
            'failure': ['failure', 'failed', 'unsuccessful', 'wrong']
        }
        
        for sensitivity in sensitivities:
            if sensitivity in sensitivity_keywords:
                avoided_keywords.extend(sensitivity_keywords[sensitivity])
        
        return list(set(avoided_keywords))
    
    async def _analyze_personalization_factors(self, user_profile: Dict[str, Any],
                                             capacity_level: EmotionalCapacityLevel) -> Dict[str, Any]:
        """Analyze various personalization factors"""
        
        factors = {}
        
        # Communication style preference
        communication_styles = user_profile.get('communication_preferences', [])
        factors['communication_style'] = {
            'prefers_direct': 'direct' in communication_styles,
            'prefers_metaphorical': 'metaphorical' in communication_styles,
            'prefers_spiritual': 'spiritual' in communication_styles,
            'prefers_scientific': 'scientific' in communication_styles
        }
        
        # Support system strength
        support_system = user_profile.get('support_system_strength', 0.5)  # 0-1 scale
        factors['support_system'] = {
            'strength': support_system,
            'emphasize_connection': support_system < 0.3,
            'emphasize_independence': support_system > 0.8
        }
        
        # Recent feedback patterns
        recent_feedback = user_profile.get('recent_affirmation_feedback', [])
        factors['feedback_patterns'] = {
            'positive_themes': await self._extract_positive_themes(recent_feedback),
            'negative_themes': await self._extract_negative_themes(recent_feedback),
            'preferred_intensity': await self._calculate_preferred_intensity(recent_feedback)
        }
        
        return factors
    
    async def _extract_positive_themes(self, feedback: List[Dict[str, Any]]) -> List[str]:
        """Extract themes from positive feedback"""
        positive_themes = []
        
        for item in feedback:
            if item.get('rating', 0) >= 4:  # 4-5 star rating
                affirmation_text = item.get('affirmation_text', '')
                # Simple keyword extraction (in production, would use NLP)
                themes = item.get('helpful_themes', [])
                positive_themes.extend(themes)
        
        # Count frequency and return most common
        theme_counts = {}
        for theme in positive_themes:
            theme_counts[theme] = theme_counts.get(theme, 0) + 1
        
        return sorted(theme_counts.keys(), key=lambda x: theme_counts[x], reverse=True)[:5]
    
    async def _extract_negative_themes(self, feedback: List[Dict[str, Any]]) -> List[str]:
        """Extract themes from negative feedback to avoid"""
        negative_themes = []
        
        for item in feedback:
            if item.get('rating', 5) <= 2:  # 1-2 star rating
                themes = item.get('unhelpful_themes', [])
                negative_themes.extend(themes)
        
        return list(set(negative_themes))
    
    async def _calculate_preferred_intensity(self, feedback: List[Dict[str, Any]]) -> float:
        """Calculate user's preferred emotional intensity based on feedback"""
        if not feedback:
            return 0.5  # Default moderate intensity
        
        intensity_ratings = []
        for item in feedback:
            if 'emotional_intensity' in item and 'rating' in item:
                # Weight by rating
                weight = item['rating'] / 5.0
                intensity_ratings.append(item['emotional_intensity'] * weight)
        
        if intensity_ratings:
            return sum(intensity_ratings) / len(intensity_ratings)
        
        return 0.5
    
    async def _get_fallback_context(self) -> Dict[str, Any]:
        """Get safe fallback personalization context"""
        return {
            'emotional_capacity': {
                'level': 'moderate',
                'can_handle_intensity': 0.6,
                'preferred_intensity': 0.5,
                'needs_gentle_approach': True
            },
            'preferred_affirmation_types': ['self_worth', 'self_compassion'],
            'avoided_types': [],
            'cultural_preference': CulturalContext.UNIVERSAL,
            'communication_style': 'gentle_supportive',
            'current_needs_keywords': ['care', 'support', 'worth'],
            'avoided_keywords': [],
            'current_situation': [],
            'recent_affirmations': []
        }
    
    async def update_user_profile(self, user_id: str, 
                                profile_updates: Dict[str, Any]) -> bool:
        """Update user profile with new information"""
        try:
            if user_id not in self.user_profiles:
                self.user_profiles[user_id] = {}
            
            self.user_profiles[user_id].update(profile_updates)
            self.user_profiles[user_id]['last_updated'] = datetime.utcnow().isoformat()
            
            logger.debug(f"Updated user profile for {user_id[:8]}...")
            return True
            
        except Exception as e:
            logger.error(f"Error updating user profile: {e}")
            return False
    
    async def get_personalization_recommendations(self, user_id: str) -> Dict[str, Any]:
        """Get recommendations for improving personalization"""
        
        user_profile = self.user_profiles.get(user_id, {})
        recent_feedback = user_profile.get('recent_affirmation_feedback', [])
        
        recommendations = {
            'data_gaps': [],
            'improvement_suggestions': [],
            'personalization_score': 0.5
        }
        
        # Check for missing personalization data
        important_fields = [
            'cultural_context', 'communication_preferences', 'trauma_history',
            'personal_values', 'current_stressors'
        ]
        
        for field in important_fields:
            if field not in user_profile:
                recommendations['data_gaps'].append(field)
        
        # Calculate personalization score
        available_data = len([f for f in important_fields if f in user_profile])
        recommendations['personalization_score'] = available_data / len(important_fields)
        
        # Provide improvement suggestions
        if len(recent_feedback) < 5:
            recommendations['improvement_suggestions'].append(
                "More feedback would help personalize affirmations better"
            )
        
        if 'cultural_context' not in user_profile:
            recommendations['improvement_suggestions'].append(
                "Cultural context would help provide more relevant affirmations"
            )
        
        return recommendations