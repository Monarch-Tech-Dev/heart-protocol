"""
Progress Detection for Guardian Energy Rising Feed

Detects healing progress, recovery milestones, and positive transformation
patterns in user posts and interactions. Built on strength-based and
trauma-informed care principles.
"""

import asyncio
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
import logging
import re

from ...core.base import Post, CareAssessment, CareLevel

logger = logging.getLogger(__name__)


class ProgressType(Enum):
    """Types of healing progress that can be detected"""
    CRISIS_STABILIZATION = "crisis_stabilization"     # Moving out of crisis
    EMOTIONAL_REGULATION = "emotional_regulation"     # Better emotional balance
    COPING_SKILLS = "coping_skills"                   # Learning healthy coping
    SELF_ADVOCACY = "self_advocacy"                   # Standing up for oneself
    BOUNDARY_SETTING = "boundary_setting"            # Setting healthy boundaries
    RELATIONSHIP_HEALING = "relationship_healing"     # Improving relationships
    TRAUMA_PROCESSING = "trauma_processing"           # Working through trauma
    SELF_COMPASSION = "self_compassion"              # Being kinder to self
    MEANING_MAKING = "meaning_making"                # Finding purpose/meaning
    RESILIENCE_BUILDING = "resilience_building"      # Developing resilience
    COMMUNITY_CONNECTION = "community_connection"     # Building support network
    SKILL_DEVELOPMENT = "skill_development"          # Learning new life skills
    CREATIVE_EXPRESSION = "creative_expression"      # Using creativity for healing
    PHYSICAL_WELLNESS = "physical_wellness"          # Improving physical health
    SPIRITUAL_GROWTH = "spiritual_growth"            # Deepening spirituality


class ProgressIntensity(Enum):
    """Intensity levels of progress indicators"""
    SUBTLE = "subtle"           # Small positive shifts
    MODERATE = "moderate"       # Clear positive changes
    SIGNIFICANT = "significant" # Major breakthroughs
    TRANSFORMATIVE = "transformative"  # Life-changing progress


class ProgressPattern(Enum):
    """Patterns of progress over time"""
    STEADY_IMPROVEMENT = "steady_improvement"         # Consistent upward trajectory
    BREAKTHROUGH_MOMENT = "breakthrough_moment"       # Sudden significant progress
    RECOVERY_FROM_SETBACK = "recovery_from_setback"   # Bouncing back from difficulty
    MILESTONE_ACHIEVEMENT = "milestone_achievement"   # Reaching specific goal
    WISDOM_SHARING = "wisdom_sharing"                # Helping others from experience
    GRATITUDE_EXPRESSION = "gratitude_expression"    # Expressing thankfulness
    FUTURE_ORIENTATION = "future_orientation"        # Looking forward with hope
    STRENGTH_RECOGNITION = "strength_recognition"    # Acknowledging own strength


@dataclass
class HealingMilestone:
    """Represents a detected healing milestone or progress indicator"""
    user_id: str
    milestone_type: ProgressType
    intensity: ProgressIntensity
    pattern: ProgressPattern
    post_id: str
    detected_at: datetime
    confidence_score: float  # 0.0 to 1.0
    supporting_evidence: List[str]
    progress_indicators: List[str]
    time_since_last_milestone: Optional[timedelta]
    celebration_worthiness: float  # 0.0 to 1.0 - how much this deserves celebration
    inspiration_potential: float   # 0.0 to 1.0 - how inspiring this could be to others
    privacy_level: str  # How much of this can be shared publicly
    user_sentiment: str # Overall emotional tone
    growth_areas: List[str]  # Areas where growth is evident
    
    def is_celebration_worthy(self) -> bool:
        """Check if this milestone deserves celebration"""
        return self.celebration_worthiness > 0.6
    
    def is_inspiration_worthy(self) -> bool:
        """Check if this could inspire others"""
        return self.inspiration_potential > 0.7 and self.privacy_level in ['public', 'community']


class ProgressDetector:
    """
    Detects healing progress and positive transformation patterns.
    
    Core Principles:
    - Strength-based: Focus on what's working and growing
    - Hope-amplifying: Highlight positive momentum
    - Culturally sensitive: Respect diverse healing paths
    - Trauma-informed: Acknowledge non-linear healing
    - User-honoring: Celebrate progress as defined by the user
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Progress detection patterns and indicators
        self.progress_patterns = self._initialize_progress_patterns()
        self.milestone_indicators = self._initialize_milestone_indicators()
        self.strength_keywords = self._initialize_strength_keywords()
        self.healing_language = self._initialize_healing_language()
        
        # User progress tracking
        self.user_progress_history = {}  # user_id -> List[HealingMilestone]
        self.baseline_assessments = {}   # user_id -> baseline_care_level
        
        # Detection metrics
        self.detection_metrics = {
            'milestones_detected': 0,
            'celebrations_triggered': 0,
            'inspiration_stories_created': 0,
            'users_with_progress': 0
        }
        
        logger.info("Progress Detector initialized")
    
    def _initialize_progress_patterns(self) -> Dict[ProgressType, Dict[str, Any]]:
        """Initialize patterns for detecting different types of progress"""
        
        return {
            ProgressType.CRISIS_STABILIZATION: {
                'positive_indicators': [
                    'feeling safer', 'crisis passed', 'stable now', 'not in danger',
                    'breathing easier', 'calm now', 'made it through', 'survived',
                    'got help', 'reaching out worked', 'support helped'
                ],
                'comparison_indicators': [
                    'better than yesterday', 'not as bad as', 'improving from',
                    'less scary than', 'more stable than before'
                ],
                'action_indicators': [
                    'called therapist', 'used coping skills', 'reached out',
                    'took medication', 'went to hospital', 'asked for help'
                ],
                'baseline_comparison': True,
                'intensity_multiplier': 2.0  # Crisis stabilization is highly significant
            },
            
            ProgressType.EMOTIONAL_REGULATION: {
                'positive_indicators': [
                    'managing emotions', 'staying calm', 'not triggered',
                    'breathing through', 'pause before reacting', 'feeling balanced',
                    'emotions settling', 'less reactive', 'more centered'
                ],
                'skill_indicators': [
                    'deep breathing', 'mindfulness', 'grounding techniques',
                    'emotional check-in', 'self-soothing', 'meditation'
                ],
                'progress_language': [
                    'getting better at', 'learning to', 'practicing',
                    'working on', 'developing'
                ]
            },
            
            ProgressType.COPING_SKILLS: {
                'positive_indicators': [
                    'using tools', 'healthy coping', 'new strategies',
                    'what works for me', 'tried something new', 'alternative approach'
                ],
                'skill_development': [
                    'learned', 'practicing', 'getting good at', 'mastering',
                    'discovered', 'found helpful'
                ],
                'specific_skills': [
                    'journaling', 'exercise', 'art therapy', 'music therapy',
                    'support group', 'therapy', 'medication', 'routine'
                ]
            },
            
            ProgressType.SELF_ADVOCACY: {
                'positive_indicators': [
                    'stood up for myself', 'said no', 'set boundary',
                    'asked for what i need', 'spoke up', 'defended myself',
                    'chose myself', 'prioritized my needs'
                ],
                'empowerment_language': [
                    'deserve better', 'worth more', 'not accepting',
                    'demanding respect', 'taking control'
                ]
            },
            
            ProgressType.BOUNDARY_SETTING: {
                'positive_indicators': [
                    'set boundaries', 'said no', 'limited contact',
                    'protected my energy', 'chose my circle', 'healthy distance',
                    'not my responsibility', 'their problem not mine'
                ],
                'empowerment_indicators': [
                    'feel empowered', 'taking control', 'protecting myself',
                    'choosing wisely', 'prioritizing me'
                ]
            },
            
            ProgressType.RELATIONSHIP_HEALING: {
                'positive_indicators': [
                    'better communication', 'resolved conflict', 'deeper connection',
                    'trust building', 'forgiveness', 'understanding each other',
                    'healthy relationship', 'supportive partner'
                ],
                'growth_indicators': [
                    'learning together', 'growing closer', 'working through',
                    'both committed', 'mutual respect'
                ]
            },
            
            ProgressType.TRAUMA_PROCESSING: {
                'positive_indicators': [
                    'processing trauma', 'working through', 'making sense',
                    'integrating experience', 'healing from', 'releasing',
                    'no longer controlling me', 'finding peace with'
                ],
                'therapy_progress': [
                    'breakthrough in therapy', 'emdr helping', 'therapist says',
                    'processing session', 'trauma work'
                ],
                'integration_signs': [
                    'makes sense now', 'understanding why', 'connecting dots',
                    'seeing patterns', 'not my fault'
                ]
            },
            
            ProgressType.SELF_COMPASSION: {
                'positive_indicators': [
                    'kind to myself', 'self-forgiveness', 'treating myself well',
                    'gentle with myself', 'self-care', 'loving myself',
                    'not beating myself up', 'inner kindness'
                ],
                'language_shifts': [
                    'i did my best', 'learning process', 'human mistake',
                    'growing and learning', 'being patient with myself'
                ]
            },
            
            ProgressType.MEANING_MAKING: {
                'positive_indicators': [
                    'finding purpose', 'making sense', 'deeper meaning',
                    'life has purpose', 'reason for', 'greater good',
                    'helping others', 'making difference'
                ],
                'perspective_shifts': [
                    'silver lining', 'growth opportunity', 'made me stronger',
                    'learned from', 'wisdom gained', 'experience taught me'
                ]
            },
            
            ProgressType.RESILIENCE_BUILDING: {
                'positive_indicators': [
                    'bouncing back', 'recovering quickly', 'handling stress',
                    'adapt to change', 'weather the storm', 'keep going',
                    'stronger than before', 'resilient'
                ],
                'strength_recognition': [
                    'survived before', 'can handle this', 'inner strength',
                    'tough times passed', 'overcome obstacles'
                ]
            },
            
            ProgressType.COMMUNITY_CONNECTION: {
                'positive_indicators': [
                    'found my tribe', 'support network', 'new friendships',
                    'community support', 'not alone', 'people who understand',
                    'meaningful connections', 'belonging'
                ],
                'engagement_signs': [
                    'joining group', 'volunteering', 'helping others',
                    'sharing story', 'participating in'
                ]
            },
            
            ProgressType.CREATIVE_EXPRESSION: {
                'positive_indicators': [
                    'art helped', 'creative outlet', 'expressing through',
                    'music therapy', 'writing helped', 'dance therapy',
                    'creative healing', 'artistic expression'
                ],
                'therapeutic_use': [
                    'processing through art', 'healing creativity',
                    'expressing emotions', 'creative breakthrough'
                ]
            }
        }
    
    def _initialize_milestone_indicators(self) -> Dict[str, List[str]]:
        """Initialize indicators for significant milestones"""
        
        return {
            'first_time_achievements': [
                'first time', 'never done before', 'milestone', 'breakthrough',
                'major step', 'huge progress', 'proud moment'
            ],
            
            'duration_milestones': [
                'days clean', 'weeks sober', 'months without', 'year anniversary',
                'days of', 'streak of', 'consecutive'
            ],
            
            'threshold_crossings': [
                'no longer', 'past the point', 'beyond', 'overcame',
                'left behind', 'moved past', 'healed from'
            ],
            
            'recognition_moments': [
                'realized', 'understood', 'finally see', 'clicked',
                'makes sense now', 'epiphany', 'aha moment'
            ],
            
            'celebration_language': [
                'celebrating', 'proud of', 'victory', 'win', 'success',
                'achievement', 'accomplished', 'reached goal'
            ],
            
            'gratitude_expressions': [
                'grateful for', 'thankful', 'appreciate', 'blessed',
                'lucky to have', 'grateful that'
            ],
            
            'strength_acknowledgment': [
                'stronger than i thought', 'discovered my strength',
                'inner power', 'capable of more', 'resilient'
            ]
        }
    
    def _initialize_strength_keywords(self) -> Dict[str, float]:
        """Initialize strength-based keywords with weights"""
        
        return {
            # Direct strength words
            'strong': 1.0, 'strength': 1.0, 'powerful': 0.9, 'resilient': 1.0,
            'brave': 0.9, 'courageous': 0.9, 'warrior': 0.8, 'fighter': 0.8,
            
            # Growth words
            'growing': 0.8, 'learning': 0.7, 'developing': 0.7, 'evolving': 0.8,
            'progressing': 0.9, 'improving': 0.8, 'advancing': 0.7,
            
            # Achievement words
            'accomplished': 0.8, 'achieved': 0.8, 'succeeded': 0.9, 'overcame': 1.0,
            'conquered': 0.9, 'mastered': 0.8, 'completed': 0.7,
            
            # Positive emotion words
            'proud': 0.8, 'confident': 0.8, 'hopeful': 0.8, 'optimistic': 0.7,
            'peaceful': 0.7, 'grateful': 0.8, 'blessed': 0.7,
            
            # Healing words
            'healing': 0.9, 'recovered': 1.0, 'transformed': 0.9, 'renewed': 0.8,
            'restored': 0.8, 'mended': 0.7, 'whole': 0.8
        }
    
    def _initialize_healing_language(self) -> Dict[str, List[str]]:
        """Initialize healing-oriented language patterns"""
        
        return {
            'progress_phrases': [
                'getting better', 'feeling stronger', 'making progress',
                'moving forward', 'on the right track', 'positive changes',
                'step in right direction', 'upward trajectory'
            ],
            
            'hope_language': [
                'looking forward', 'excited about', 'hopeful for',
                'bright future', 'good things coming', 'positive outlook'
            ],
            
            'empowerment_phrases': [
                'taking control', 'in charge of', 'choosing my path',
                'making decisions', 'standing up', 'advocating for myself'
            ],
            
            'wisdom_sharing': [
                'learned that', 'discovered', 'now i know', 'experience taught me',
                'wisdom gained', 'insight', 'understanding now'
            ],
            
            'gratitude_expressions': [
                'grateful for', 'thankful that', 'appreciate', 'blessed by',
                'lucky to have', 'value the'
            ]
        }
    
    async def detect_progress(self, post: Post, 
                            care_assessment: CareAssessment,
                            user_history: Dict[str, Any]) -> Optional[HealingMilestone]:
        """
        Detect healing progress and milestones in a user's post.
        
        Args:
            post: The user's post to analyze
            care_assessment: Current care assessment
            user_history: User's historical data for comparison
        """
        try:
            # Don't detect progress in posts that are primarily seeking help
            if care_assessment.care_level in [CareLevel.CRISIS, CareLevel.HIGH]:
                # Unless it's explicitly about overcoming crisis
                if not await self._contains_crisis_recovery_language(post.content):
                    return None
            
            # Analyze post content for progress indicators
            progress_analysis = await self._analyze_progress_content(post.content)
            
            if not progress_analysis['has_progress_indicators']:
                return None
            
            # Determine progress type and intensity
            progress_type = await self._identify_progress_type(post.content, progress_analysis)
            intensity = await self._assess_progress_intensity(post.content, progress_analysis)
            pattern = await self._identify_progress_pattern(post.content, user_history)
            
            # Calculate confidence and celebration scores
            confidence_score = await self._calculate_confidence(progress_analysis, user_history)
            celebration_worthiness = await self._assess_celebration_worthiness(
                progress_type, intensity, pattern, progress_analysis
            )
            inspiration_potential = await self._assess_inspiration_potential(
                post.content, progress_type, intensity
            )
            
            # Extract supporting evidence and growth areas
            supporting_evidence = progress_analysis['evidence']
            growth_areas = await self._identify_growth_areas(post.content, progress_type)
            
            # Determine privacy level
            privacy_level = await self._assess_privacy_level(post, user_history)
            
            # Assess user sentiment
            user_sentiment = await self._assess_sentiment(post.content)
            
            # Calculate time since last milestone
            time_since_last = await self._calculate_time_since_last_milestone(post.author_id, user_history)
            
            milestone = HealingMilestone(
                user_id=post.author_id,
                milestone_type=progress_type,
                intensity=intensity,
                pattern=pattern,
                post_id=post.id,
                detected_at=datetime.utcnow(),
                confidence_score=confidence_score,
                supporting_evidence=supporting_evidence,
                progress_indicators=progress_analysis['indicators'],
                time_since_last_milestone=time_since_last,
                celebration_worthiness=celebration_worthiness,
                inspiration_potential=inspiration_potential,
                privacy_level=privacy_level,
                user_sentiment=user_sentiment,
                growth_areas=growth_areas
            )
            
            # Store in user progress history
            await self._record_milestone(milestone)
            
            # Update metrics
            self.detection_metrics['milestones_detected'] += 1
            if milestone.is_celebration_worthy():
                self.detection_metrics['celebrations_triggered'] += 1
            if milestone.is_inspiration_worthy():
                self.detection_metrics['inspiration_stories_created'] += 1
            
            logger.info(f"Progress detected for user {post.author_id[:8]}... "
                       f"(Type: {progress_type.value}, Intensity: {intensity.value})")
            
            return milestone
            
        except Exception as e:
            logger.error(f"Error detecting progress: {e}")
            return None
    
    async def _contains_crisis_recovery_language(self, content: str) -> bool:
        """Check if content contains language about recovering from crisis"""
        
        recovery_phrases = [
            'made it through', 'survived the', 'crisis passed', 'feeling safer',
            'stabilized', 'got help', 'reached out', 'not in danger',
            'breathing again', 'storm passed'
        ]
        
        content_lower = content.lower()
        return any(phrase in content_lower for phrase in recovery_phrases)
    
    async def _analyze_progress_content(self, content: str) -> Dict[str, Any]:
        """Analyze content for progress indicators"""
        
        content_lower = content.lower()
        analysis = {
            'has_progress_indicators': False,
            'indicators': [],
            'evidence': [],
            'strength_score': 0.0,
            'growth_score': 0.0,
            'positive_emotion_score': 0.0
        }
        
        # Check for strength keywords
        strength_score = 0.0
        for keyword, weight in self.strength_keywords.items():
            if keyword in content_lower:
                strength_score += weight
                analysis['indicators'].append(f"strength_keyword: {keyword}")
        
        analysis['strength_score'] = min(1.0, strength_score / 5)  # Normalize
        
        # Check for milestone indicators
        for category, indicators in self.milestone_indicators.items():
            for indicator in indicators:
                if indicator in content_lower:
                    analysis['indicators'].append(f"milestone_{category}: {indicator}")
                    analysis['evidence'].append(f"Contains milestone language: '{indicator}'")
        
        # Check for healing language
        for category, phrases in self.healing_language.items():
            for phrase in phrases:
                if phrase in content_lower:
                    analysis['indicators'].append(f"healing_{category}: {phrase}")
                    analysis['evidence'].append(f"Healing language: '{phrase}'")
        
        # Calculate overall progress indication
        analysis['has_progress_indicators'] = (
            len(analysis['indicators']) > 0 or 
            analysis['strength_score'] > 0.3
        )
        
        return analysis
    
    async def _identify_progress_type(self, content: str, 
                                    analysis: Dict[str, Any]) -> ProgressType:
        """Identify the type of progress being demonstrated"""
        
        content_lower = content.lower()
        type_scores = {}
        
        # Score each progress type based on content patterns
        for progress_type, patterns in self.progress_patterns.items():
            score = 0.0
            
            # Check positive indicators
            for indicator in patterns.get('positive_indicators', []):
                if indicator in content_lower:
                    score += 1.0
            
            # Check skill indicators
            for indicator in patterns.get('skill_indicators', []):
                if indicator in content_lower:
                    score += 0.8
            
            # Check action indicators
            for indicator in patterns.get('action_indicators', []):
                if indicator in content_lower:
                    score += 0.6
            
            # Check progress language
            for indicator in patterns.get('progress_language', []):
                if indicator in content_lower:
                    score += 0.4
            
            if score > 0:
                type_scores[progress_type] = score
        
        # Return the highest scoring type, or default
        if type_scores:
            return max(type_scores.items(), key=lambda x: x[1])[0]
        else:
            return ProgressType.EMOTIONAL_REGULATION  # Default
    
    async def _assess_progress_intensity(self, content: str, 
                                       analysis: Dict[str, Any]) -> ProgressIntensity:
        """Assess the intensity/significance of the progress"""
        
        content_lower = content.lower()
        
        # High intensity indicators
        transformative_keywords = [
            'breakthrough', 'transformation', 'life-changing', 'major milestone',
            'huge step', 'completely different', 'total change'
        ]
        
        significant_keywords = [
            'significant', 'major', 'big step', 'important milestone',
            'real progress', 'meaningful change'
        ]
        
        moderate_keywords = [
            'progress', 'improvement', 'better', 'positive change',
            'step forward', 'moving ahead'
        ]
        
        # Check for intensity indicators
        if any(keyword in content_lower for keyword in transformative_keywords):
            return ProgressIntensity.TRANSFORMATIVE
        elif any(keyword in content_lower for keyword in significant_keywords):
            return ProgressIntensity.SIGNIFICANT
        elif any(keyword in content_lower for keyword in moderate_keywords):
            return ProgressIntensity.MODERATE
        else:
            return ProgressIntensity.SUBTLE
    
    async def _identify_progress_pattern(self, content: str, 
                                       user_history: Dict[str, Any]) -> ProgressPattern:
        """Identify the pattern of progress"""
        
        content_lower = content.lower()
        
        # Check for specific pattern indicators
        if any(phrase in content_lower for phrase in ['breakthrough', 'sudden', 'aha moment', 'clicked']):
            return ProgressPattern.BREAKTHROUGH_MOMENT
        
        if any(phrase in content_lower for phrase in ['milestone', 'goal', 'achievement', 'accomplished']):
            return ProgressPattern.MILESTONE_ACHIEVEMENT
        
        if any(phrase in content_lower for phrase in ['bounced back', 'recovered from', 'setback', 'relapse']):
            return ProgressPattern.RECOVERY_FROM_SETBACK
        
        if any(phrase in content_lower for phrase in ['helping others', 'sharing my story', 'advice for']):
            return ProgressPattern.WISDOM_SHARING
        
        if any(phrase in content_lower for phrase in ['grateful', 'thankful', 'blessed', 'appreciate']):
            return ProgressPattern.GRATITUDE_EXPRESSION
        
        if any(phrase in content_lower for phrase in ['future', 'tomorrow', 'next', 'looking forward']):
            return ProgressPattern.FUTURE_ORIENTATION
        
        if any(phrase in content_lower for phrase in ['realize', 'recognize', 'acknowledge', 'strength']):
            return ProgressPattern.STRENGTH_RECOGNITION
        
        # Default to steady improvement
        return ProgressPattern.STEADY_IMPROVEMENT
    
    async def _calculate_confidence(self, analysis: Dict[str, Any], 
                                  user_history: Dict[str, Any]) -> float:
        """Calculate confidence in progress detection"""
        
        confidence = 0.5  # Base confidence
        
        # Higher confidence with more indicators
        confidence += min(0.3, len(analysis['indicators']) * 0.05)
        
        # Higher confidence with strong strength scores
        confidence += analysis['strength_score'] * 0.2
        
        # Higher confidence with supporting evidence
        confidence += min(0.2, len(analysis['evidence']) * 0.04)
        
        # Consider user's baseline for comparison
        if user_history.get('baseline_care_level'):
            baseline = user_history['baseline_care_level']
            current = user_history.get('current_care_level', baseline)
            
            if current < baseline:  # Improvement from baseline
                confidence += 0.2
        
        return min(1.0, confidence)
    
    async def _assess_celebration_worthiness(self, progress_type: ProgressType,
                                           intensity: ProgressIntensity,
                                           pattern: ProgressPattern,
                                           analysis: Dict[str, Any]) -> float:
        """Assess how worthy this progress is of celebration"""
        
        base_score = 0.5
        
        # Intensity contributes to celebration worthiness
        intensity_scores = {
            ProgressIntensity.TRANSFORMATIVE: 1.0,
            ProgressIntensity.SIGNIFICANT: 0.8,
            ProgressIntensity.MODERATE: 0.6,
            ProgressIntensity.SUBTLE: 0.3
        }
        
        base_score += intensity_scores[intensity] * 0.3
        
        # Certain patterns are more celebration-worthy
        pattern_bonuses = {
            ProgressPattern.BREAKTHROUGH_MOMENT: 0.2,
            ProgressPattern.MILESTONE_ACHIEVEMENT: 0.3,
            ProgressPattern.RECOVERY_FROM_SETBACK: 0.25,
            ProgressPattern.WISDOM_SHARING: 0.15
        }
        
        base_score += pattern_bonuses.get(pattern, 0.0)
        
        # Crisis stabilization is always highly celebration-worthy
        if progress_type == ProgressType.CRISIS_STABILIZATION:
            base_score += 0.3
        
        return min(1.0, base_score)
    
    async def _assess_inspiration_potential(self, content: str,
                                          progress_type: ProgressType,
                                          intensity: ProgressIntensity) -> float:
        """Assess how inspiring this could be to others"""
        
        content_lower = content.lower()
        inspiration_score = 0.4
        
        # Transformative progress is highly inspiring
        if intensity == ProgressIntensity.TRANSFORMATIVE:
            inspiration_score += 0.4
        elif intensity == ProgressIntensity.SIGNIFICANT:
            inspiration_score += 0.3
        
        # Certain types are naturally more inspiring
        inspiring_types = {
            ProgressType.CRISIS_STABILIZATION,
            ProgressType.TRAUMA_PROCESSING,
            ProgressType.RESILIENCE_BUILDING,
            ProgressType.MEANING_MAKING
        }
        
        if progress_type in inspiring_types:
            inspiration_score += 0.2
        
        # Check for inspiring language
        inspiring_phrases = [
            'if i can do it', 'anyone can', 'hope for others',
            'sharing my story', 'want to help', 'message of hope'
        ]
        
        if any(phrase in content_lower for phrase in inspiring_phrases):
            inspiration_score += 0.2
        
        return min(1.0, inspiration_score)
    
    async def _identify_growth_areas(self, content: str, 
                                   progress_type: ProgressType) -> List[str]:
        """Identify specific areas where growth is evident"""
        
        growth_areas = []
        content_lower = content.lower()
        
        # Map content to growth areas
        growth_mapping = {
            'emotional': ['emotional regulation', 'feelings management'],
            'relationship': ['communication', 'connection', 'trust'],
            'coping': ['stress management', 'healthy habits'],
            'trauma': ['healing', 'integration', 'processing'],
            'boundary': ['self-advocacy', 'limits', 'protection'],
            'self': ['self-worth', 'self-care', 'self-compassion'],
            'spiritual': ['meaning', 'purpose', 'faith'],
            'physical': ['health', 'wellness', 'body']
        }
        
        for keyword, areas in growth_mapping.items():
            if keyword in content_lower:
                growth_areas.extend(areas)
        
        # Add progress type specific areas
        type_areas = {
            ProgressType.EMOTIONAL_REGULATION: ['emotional intelligence'],
            ProgressType.COPING_SKILLS: ['resilience tools'],
            ProgressType.SELF_ADVOCACY: ['empowerment'],
            ProgressType.BOUNDARY_SETTING: ['healthy limits'],
            ProgressType.TRAUMA_PROCESSING: ['healing integration'],
            ProgressType.COMMUNITY_CONNECTION: ['social support']
        }
        
        growth_areas.extend(type_areas.get(progress_type, []))
        
        return list(set(growth_areas))  # Remove duplicates
    
    async def _assess_privacy_level(self, post: Post, 
                                   user_history: Dict[str, Any]) -> str:
        """Assess appropriate privacy level for sharing this progress"""
        
        # Check post visibility settings
        if hasattr(post, 'visibility') and post.visibility == 'private':
            return 'private'
        
        # Check for sensitive content
        sensitive_keywords = [
            'abuse', 'assault', 'suicide', 'self-harm', 'addiction',
            'medication', 'therapy', 'hospital', 'crisis'
        ]
        
        if any(keyword in post.content.lower() for keyword in sensitive_keywords):
            return 'community'  # Share with supportive community only
        
        # Check user's historical preferences
        if user_history.get('prefers_privacy', False):
            return 'community'
        
        return 'public'  # Safe to share more broadly
    
    async def _assess_sentiment(self, content: str) -> str:
        """Assess overall emotional sentiment"""
        
        content_lower = content.lower()
        
        # Count positive vs challenging emotional indicators
        positive_emotions = ['happy', 'joy', 'grateful', 'proud', 'hopeful', 'peaceful', 'confident']
        mixed_emotions = ['bittersweet', 'complicated', 'processing', 'working through']
        challenging_emotions = ['difficult', 'hard', 'struggling', 'painful', 'tough']
        
        positive_count = sum(1 for emotion in positive_emotions if emotion in content_lower)
        mixed_count = sum(1 for emotion in mixed_emotions if emotion in content_lower)
        challenging_count = sum(1 for emotion in challenging_emotions if emotion in content_lower)
        
        if positive_count > challenging_count + mixed_count:
            return 'positive'
        elif mixed_count > 0 or (positive_count > 0 and challenging_count > 0):
            return 'mixed_positive'  # Acknowledging difficulty but with positive elements
        else:
            return 'reflective'
    
    async def _calculate_time_since_last_milestone(self, user_id: str,
                                                 user_history: Dict[str, Any]) -> Optional[timedelta]:
        """Calculate time since user's last milestone"""
        
        user_milestones = self.user_progress_history.get(user_id, [])
        
        if not user_milestones:
            return None
        
        last_milestone = max(user_milestones, key=lambda m: m.detected_at)
        return datetime.utcnow() - last_milestone.detected_at
    
    async def _record_milestone(self, milestone: HealingMilestone) -> None:
        """Record milestone in user's progress history"""
        
        user_id = milestone.user_id
        
        if user_id not in self.user_progress_history:
            self.user_progress_history[user_id] = []
        
        self.user_progress_history[user_id].append(milestone)
        
        # Keep only recent milestones
        max_milestones = 50
        if len(self.user_progress_history[user_id]) > max_milestones:
            self.user_progress_history[user_id] = self.user_progress_history[user_id][-max_milestones:]
        
        # Update unique users count
        self.detection_metrics['users_with_progress'] = len(self.user_progress_history)
    
    def get_user_progress_summary(self, user_id: str) -> Dict[str, Any]:
        """Get summary of user's progress journey"""
        
        user_milestones = self.user_progress_history.get(user_id, [])
        
        if not user_milestones:
            return {'no_progress_data': True}
        
        # Analyze progress patterns
        progress_types = {}
        for milestone in user_milestones:
            milestone_type = milestone.milestone_type.value
            progress_types[milestone_type] = progress_types.get(milestone_type, 0) + 1
        
        # Calculate recent progress
        recent_milestones = [
            m for m in user_milestones
            if m.detected_at > datetime.utcnow() - timedelta(days=30)
        ]
        
        return {
            'total_milestones': len(user_milestones),
            'recent_milestones': len(recent_milestones),
            'progress_areas': progress_types,
            'celebration_worthy_count': sum(1 for m in user_milestones if m.is_celebration_worthy()),
            'inspiration_worthy_count': sum(1 for m in user_milestones if m.is_inspiration_worthy()),
            'latest_milestone': user_milestones[-1].milestone_type.value if user_milestones else None,
            'progress_trajectory': 'positive' if len(recent_milestones) > 0 else 'stable'
        }
    
    def get_detection_metrics(self) -> Dict[str, Any]:
        """Get progress detection metrics"""
        
        return {
            'milestones_detected': self.detection_metrics['milestones_detected'],
            'celebrations_triggered': self.detection_metrics['celebrations_triggered'],
            'inspiration_stories_created': self.detection_metrics['inspiration_stories_created'],
            'users_with_progress': self.detection_metrics['users_with_progress'],
            'progress_types_detected': list(ProgressType),
            'system_health': {
                'progress_patterns_loaded': len(self.progress_patterns) > 0,
                'milestone_indicators_loaded': len(self.milestone_indicators) > 0,
                'strength_keywords_loaded': len(self.strength_keywords) > 0,
                'healing_language_loaded': len(self.healing_language) > 0
            }
        }