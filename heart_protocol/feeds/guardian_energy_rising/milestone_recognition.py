"""
Milestone Recognition for Guardian Energy Rising Feed

Recognizes specific healing milestones and achievements worthy of celebration
and community inspiration. Built on trauma-informed care principles and
strength-based approaches.
"""

import asyncio
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
import logging
import re

from .progress_detection import ProgressDetector, HealingMilestone, ProgressType, ProgressIntensity, ProgressPattern
from ...core.base import Post, CareAssessment, CareLevel

logger = logging.getLogger(__name__)


class MilestoneCategory(Enum):
    """Categories of recognizable milestones"""
    SURVIVAL_ACHIEVEMENT = "survival_achievement"    # Surviving difficult periods
    HEALING_BREAKTHROUGH = "healing_breakthrough"    # Major healing moments
    SKILL_MASTERY = "skill_mastery"                 # Mastering coping skills
    CONNECTION_SUCCESS = "connection_success"        # Building healthy relationships
    SELF_ADVOCACY_WIN = "self_advocacy_win"         # Standing up for oneself
    TRAUMA_INTEGRATION = "trauma_integration"        # Processing and integrating trauma
    WISDOM_SHARING = "wisdom_sharing"               # Helping others from experience
    RESILIENCE_DISPLAY = "resilience_display"       # Showing remarkable resilience
    BOUNDARY_ESTABLISHMENT = "boundary_establishment" # Setting healthy boundaries
    MEANING_DISCOVERY = "meaning_discovery"         # Finding purpose and meaning


class CelebrationLevel(Enum):
    """Levels of celebration worthiness"""
    QUIET_ACKNOWLEDGMENT = "quiet_acknowledgment"   # Gentle recognition
    COMMUNITY_CELEBRATION = "community_celebration" # Share with caring community
    INSPIRING_HIGHLIGHT = "inspiring_highlight"     # Inspire others facing similar challenges
    TRANSFORMATIVE_STORY = "transformative_story"   # Life-changing milestone


@dataclass
class CelebrationTrigger:
    """Represents a milestone worthy of celebration"""
    milestone_id: str
    user_id: str
    category: MilestoneCategory
    celebration_level: CelebrationLevel
    milestone_description: str
    significance_explanation: str
    inspiration_potential: float  # 0.0 to 1.0
    privacy_level: str  # How much can be shared
    celebration_message: str
    supporting_quotes: List[str]  # Quotes from the user's post
    growth_areas_highlighted: List[str]
    time_period_context: Optional[str]  # "after 6 months of...", etc.
    celebration_timing: datetime
    expiry_time: datetime
    anonymization_guidelines: Dict[str, Any]
    care_considerations: List[str]  # Special care needed when celebrating


class MilestoneRecognizer:
    """
    Recognizes healing milestones and achievements worthy of celebration.
    
    Core Principles:
    - Celebrate progress, not perfection
    - Honor diverse healing journeys
    - Respect privacy and consent
    - Amplify hope and possibility
    - Avoid toxic positivity
    - Support continued growth
    """
    
    def __init__(self, config: Dict[str, Any], progress_detector: ProgressDetector):
        self.config = config
        self.progress_detector = progress_detector
        
        # Milestone recognition patterns
        self.milestone_patterns = self._initialize_milestone_patterns()
        self.celebration_thresholds = self._initialize_celebration_thresholds()
        self.inspiration_indicators = self._initialize_inspiration_indicators()
        self.timing_considerations = self._initialize_timing_considerations()
        
        # Recognition tracking
        self.recognized_milestones = {}  # user_id -> List[CelebrationTrigger]
        self.celebration_history = {}    # milestone_id -> celebration_data
        
        # Recognition metrics
        self.recognition_metrics = {
            'milestones_recognized': 0,
            'celebrations_triggered': 0,
            'inspiring_stories_created': 0,
            'quiet_acknowledgments_given': 0,
            'users_celebrated': 0
        }
        
        logger.info("Milestone Recognizer initialized")
    
    def _initialize_milestone_patterns(self) -> Dict[MilestoneCategory, Dict[str, Any]]:
        """Initialize patterns for recognizing different milestone categories"""
        
        return {
            MilestoneCategory.SURVIVAL_ACHIEVEMENT: {
                'indicators': [
                    'made it through', 'survived', 'still here', 'didn\'t give up',
                    'kept going', 'pushed through', 'weathered the storm',
                    'came out the other side', 'endured', 'persevered'
                ],
                'time_markers': [
                    'days clean', 'weeks sober', 'months without', 'year since',
                    'anniversary of', 'it\'s been X since', 'X days ago'
                ],
                'significance_multiplier': 2.0,  # Survival is highly significant
                'inspiration_boost': 0.3
            },
            
            MilestoneCategory.HEALING_BREAKTHROUGH: {
                'indicators': [
                    'breakthrough', 'aha moment', 'suddenly understood', 'clicked',
                    'pieces fell together', 'everything made sense', 'realized',
                    'epiphany', 'breakthrough moment', 'major shift'
                ],
                'progress_markers': [
                    'therapy breakthrough', 'processing breakthrough', 'emotional breakthrough',
                    'healing moment', 'transformation', 'life-changing realization'
                ],
                'significance_multiplier': 1.8,
                'inspiration_boost': 0.4
            },
            
            MilestoneCategory.SKILL_MASTERY: {
                'indicators': [
                    'mastered', 'got good at', 'skilled now', 'natural now',
                    'second nature', 'automatic', 'instinctive', 'comes easily'
                ],
                'skill_areas': [
                    'breathing techniques', 'mindfulness', 'grounding', 'self-soothing',
                    'emotional regulation', 'communication', 'boundary setting',
                    'coping skills', 'stress management'
                ],
                'significance_multiplier': 1.4,
                'inspiration_boost': 0.2
            },
            
            MilestoneCategory.CONNECTION_SUCCESS: {
                'indicators': [
                    'healthy relationship', 'good friend', 'supportive partner',
                    'found my people', 'deep connection', 'mutual support',
                    'trust someone', 'feel understood', 'not alone anymore'
                ],
                'relationship_markers': [
                    'first time trusting', 'healthy communication', 'mutual respect',
                    'supportive friendship', 'loving relationship', 'found community'
                ],
                'significance_multiplier': 1.6,
                'inspiration_boost': 0.25
            },
            
            MilestoneCategory.SELF_ADVOCACY_WIN: {
                'indicators': [
                    'stood up for myself', 'said no', 'defended myself',
                    'spoke my truth', 'advocated for myself', 'demanded respect',
                    'protected my peace', 'chose myself', 'set boundaries'
                ],
                'empowerment_markers': [
                    'found my voice', 'worth more', 'deserve better',
                    'not accepting', 'taking control', 'empowered'
                ],
                'significance_multiplier': 1.5,
                'inspiration_boost': 0.35
            },
            
            MilestoneCategory.TRAUMA_INTEGRATION: {
                'indicators': [
                    'integrated', 'processed', 'made peace with', 'healed from',
                    'no longer controls me', 'transformed the pain', 'wisdom from',
                    'strength from struggle', 'post-traumatic growth'
                ],
                'integration_markers': [
                    'understand now', 'makes sense', 'not my fault',
                    'learned from', 'grew from', 'transformed by'
                ],
                'significance_multiplier': 2.2,  # Trauma integration is profound
                'inspiration_boost': 0.5
            },
            
            MilestoneCategory.WISDOM_SHARING: {
                'indicators': [
                    'helping others', 'sharing my story', 'mentor', 'guide',
                    'paying it forward', 'supporting someone', 'advice for',
                    'hope for others', 'if i can do it'
                ],
                'wisdom_markers': [
                    'learned that', 'discovered', 'wisdom gained',
                    'experience taught me', 'want others to know'
                ],
                'significance_multiplier': 1.7,
                'inspiration_boost': 0.6  # Sharing wisdom is highly inspiring
            },
            
            MilestoneCategory.RESILIENCE_DISPLAY: {
                'indicators': [
                    'bounced back', 'resilient', 'stronger than before',
                    'unbreakable', 'warrior', 'survivor spirit',
                    'can handle anything', 'adapt to anything'
                ],
                'resilience_markers': [
                    'overcome obstacles', 'weather any storm', 'inner strength',
                    'tough times pass', 'keep going no matter what'
                ],
                'significance_multiplier': 1.6,
                'inspiration_boost': 0.4
            },
            
            MilestoneCategory.MEANING_DISCOVERY: {
                'indicators': [
                    'found purpose', 'life has meaning', 'reason for',
                    'greater purpose', 'calling', 'mission', 'passionate about',
                    'making difference', 'serving others'
                ],
                'meaning_markers': [
                    'purpose-driven', 'meaningful work', 'serving community',
                    'higher calling', 'spiritual purpose', 'legacy'
                ],
                'significance_multiplier': 1.9,
                'inspiration_boost': 0.45
            }
        }
    
    def _initialize_celebration_thresholds(self) -> Dict[CelebrationLevel, Dict[str, Any]]:
        """Initialize thresholds for different celebration levels"""
        
        return {
            CelebrationLevel.QUIET_ACKNOWLEDGMENT: {
                'min_confidence': 0.6,
                'min_inspiration_potential': 0.0,
                'max_privacy_level': 'private',
                'celebration_approach': 'gentle_recognition'
            },
            
            CelebrationLevel.COMMUNITY_CELEBRATION: {
                'min_confidence': 0.75,
                'min_inspiration_potential': 0.4,
                'max_privacy_level': 'community',
                'celebration_approach': 'supportive_celebration'
            },
            
            CelebrationLevel.INSPIRING_HIGHLIGHT: {
                'min_confidence': 0.85,
                'min_inspiration_potential': 0.7,
                'max_privacy_level': 'public',
                'celebration_approach': 'inspiring_story'
            },
            
            CelebrationLevel.TRANSFORMATIVE_STORY: {
                'min_confidence': 0.9,
                'min_inspiration_potential': 0.85,
                'max_privacy_level': 'public',
                'celebration_approach': 'life_changing_story'
            }
        }
    
    def _initialize_inspiration_indicators(self) -> Dict[str, float]:
        """Initialize indicators that boost inspiration potential"""
        
        return {
            # Language that inspires hope
            'hope_language': 0.2,
            'if_i_can_do_it': 0.3,
            'anyone_can': 0.25,
            'message_of_hope': 0.35,
            'want_to_help_others': 0.4,
            
            # Overcoming significant challenges
            'major_obstacles_overcome': 0.3,
            'against_all_odds': 0.4,
            'thought_impossible': 0.35,
            
            # Transformation themes
            'complete_transformation': 0.4,
            'unrecognizable_from_before': 0.45,
            'new_person': 0.3,
            
            # Wisdom and growth
            'learned_so_much': 0.2,
            'wisdom_to_share': 0.35,
            'growth_mindset': 0.25
        }
    
    def _initialize_timing_considerations(self) -> Dict[str, Any]:
        """Initialize timing considerations for celebrations"""
        
        return {
            'minimum_time_between_celebrations': timedelta(days=7),
            'crisis_celebration_delay': timedelta(days=1),  # Wait after crisis
            'anniversary_recognition_window': timedelta(days=3),
            'seasonal_sensitivity': {
                'holidays': 'extra_gentle',
                'anniversaries': 'acknowledge_significance',
                'difficult_dates': 'approach_carefully'
            },
            'time_of_day_preferences': {
                'morning': 'hopeful_start',
                'afternoon': 'midday_boost',
                'evening': 'reflection_celebration'
            }
        }
    
    async def recognize_milestone(self, healing_milestone: HealingMilestone, 
                                user_context: Dict[str, Any],
                                post: Post) -> Optional[CelebrationTrigger]:
        """
        Recognize if a healing milestone deserves celebration.
        
        Args:
            healing_milestone: The detected milestone from progress detection
            user_context: User's context and preferences
            post: The original post that contained the milestone
        """
        try:
            # Check if this milestone meets recognition criteria
            if not await self._meets_recognition_criteria(healing_milestone, user_context):
                return None
            
            # Determine milestone category
            category = await self._categorize_milestone(healing_milestone, post.content)
            
            if not category:
                return None
            
            # Calculate celebration level
            celebration_level = await self._determine_celebration_level(
                healing_milestone, category, user_context
            )
            
            # Generate celebration content
            celebration_content = await self._generate_celebration_content(
                healing_milestone, category, celebration_level, post.content
            )
            
            # Create celebration trigger
            trigger = CelebrationTrigger(
                milestone_id=f"milestone_{healing_milestone.user_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                user_id=healing_milestone.user_id,
                category=category,
                celebration_level=celebration_level,
                milestone_description=celebration_content['description'],
                significance_explanation=celebration_content['significance'],
                inspiration_potential=healing_milestone.inspiration_potential,
                privacy_level=healing_milestone.privacy_level,
                celebration_message=celebration_content['message'],
                supporting_quotes=celebration_content['quotes'],
                growth_areas_highlighted=healing_milestone.growth_areas,
                time_period_context=await self._extract_time_context(post.content),
                celebration_timing=await self._calculate_optimal_timing(healing_milestone, user_context),
                expiry_time=datetime.utcnow() + timedelta(days=30),
                anonymization_guidelines=await self._create_anonymization_guidelines(
                    celebration_level, healing_milestone.privacy_level
                ),
                care_considerations=await self._identify_care_considerations(
                    healing_milestone, category
                )
            )
            
            # Record the recognition
            await self._record_milestone_recognition(trigger)
            
            # Update metrics
            self.recognition_metrics['milestones_recognized'] += 1
            if celebration_level != CelebrationLevel.QUIET_ACKNOWLEDGMENT:
                self.recognition_metrics['celebrations_triggered'] += 1
            if celebration_level in [CelebrationLevel.INSPIRING_HIGHLIGHT, CelebrationLevel.TRANSFORMATIVE_STORY]:
                self.recognition_metrics['inspiring_stories_created'] += 1
            
            logger.info(f"Milestone recognized for user {healing_milestone.user_id[:8]}... "
                       f"(Category: {category.value}, Level: {celebration_level.value})")
            
            return trigger
            
        except Exception as e:
            logger.error(f"Error recognizing milestone: {e}")
            return None
    
    async def _meets_recognition_criteria(self, milestone: HealingMilestone, 
                                        user_context: Dict[str, Any]) -> bool:
        """Check if milestone meets basic recognition criteria"""
        
        # Must be celebration worthy
        if not milestone.is_celebration_worthy():
            return False
        
        # Respect user's celebration preferences
        if user_context.get('celebration_preference') == 'none':
            return False
        
        # Check timing - don't celebrate too frequently
        last_celebration = await self._get_last_celebration_time(milestone.user_id)
        if last_celebration:
            time_since_last = datetime.utcnow() - last_celebration
            min_interval = self.timing_considerations['minimum_time_between_celebrations']
            if time_since_last < min_interval:
                return False
        
        # Don't celebrate immediately after crisis unless it's recovery
        if (milestone.milestone_type != ProgressType.CRISIS_STABILIZATION and
            user_context.get('recent_crisis', False)):
            return False
        
        return True
    
    async def _categorize_milestone(self, milestone: HealingMilestone, 
                                  content: str) -> Optional[MilestoneCategory]:
        """Categorize the type of milestone"""
        
        content_lower = content.lower()
        category_scores = {}
        
        # Score each category based on content patterns
        for category, patterns in self.milestone_patterns.items():
            score = 0.0
            
            # Check indicators
            for indicator in patterns.get('indicators', []):
                if indicator in content_lower:
                    score += 1.0
            
            # Check specific markers
            for marker_type in ['time_markers', 'progress_markers', 'skill_areas', 
                               'relationship_markers', 'empowerment_markers',
                               'integration_markers', 'wisdom_markers', 
                               'resilience_markers', 'meaning_markers']:
                for marker in patterns.get(marker_type, []):
                    if marker in content_lower:
                        score += 0.8
            
            # Apply significance multiplier
            if score > 0:
                score *= patterns.get('significance_multiplier', 1.0)
                category_scores[category] = score
        
        # Also consider the progress type from milestone
        type_category_mapping = {
            ProgressType.CRISIS_STABILIZATION: MilestoneCategory.SURVIVAL_ACHIEVEMENT,
            ProgressType.SELF_ADVOCACY: MilestoneCategory.SELF_ADVOCACY_WIN,
            ProgressType.TRAUMA_PROCESSING: MilestoneCategory.TRAUMA_INTEGRATION,
            ProgressType.MEANING_MAKING: MilestoneCategory.MEANING_DISCOVERY,
            ProgressType.RESILIENCE_BUILDING: MilestoneCategory.RESILIENCE_DISPLAY,
            ProgressType.COPING_SKILLS: MilestoneCategory.SKILL_MASTERY,
            ProgressType.COMMUNITY_CONNECTION: MilestoneCategory.CONNECTION_SUCCESS
        }
        
        mapped_category = type_category_mapping.get(milestone.milestone_type)
        if mapped_category:
            current_score = category_scores.get(mapped_category, 0.0)
            category_scores[mapped_category] = current_score + 2.0  # Boost mapped category
        
        # Return highest scoring category if above threshold
        if category_scores:
            best_category, best_score = max(category_scores.items(), key=lambda x: x[1])
            if best_score >= 1.0:
                return best_category
        
        return None
    
    async def _determine_celebration_level(self, milestone: HealingMilestone,
                                         category: MilestoneCategory,
                                         user_context: Dict[str, Any]) -> CelebrationLevel:
        """Determine appropriate celebration level"""
        
        # Start with quiet acknowledgment and work up
        celebration_level = CelebrationLevel.QUIET_ACKNOWLEDGMENT
        
        # Check thresholds for each level
        for level, thresholds in self.celebration_thresholds.items():
            if (milestone.confidence_score >= thresholds['min_confidence'] and
                milestone.inspiration_potential >= thresholds['min_inspiration_potential']):
                
                # Check privacy constraints
                privacy_allows = await self._check_privacy_level_allows(
                    milestone.privacy_level, thresholds['max_privacy_level']
                )
                
                if privacy_allows:
                    celebration_level = level
        
        # Apply category-specific adjustments
        category_patterns = self.milestone_patterns.get(category, {})
        inspiration_boost = category_patterns.get('inspiration_boost', 0.0)
        
        # Boost level for highly inspiring categories
        if inspiration_boost > 0.4 and celebration_level == CelebrationLevel.COMMUNITY_CELEBRATION:
            celebration_level = CelebrationLevel.INSPIRING_HIGHLIGHT
        elif inspiration_boost > 0.5 and celebration_level == CelebrationLevel.INSPIRING_HIGHLIGHT:
            celebration_level = CelebrationLevel.TRANSFORMATIVE_STORY
        
        # Respect user preferences
        max_user_level = user_context.get('max_celebration_level', 'transformative_story')
        level_hierarchy = [
            CelebrationLevel.QUIET_ACKNOWLEDGMENT,
            CelebrationLevel.COMMUNITY_CELEBRATION,
            CelebrationLevel.INSPIRING_HIGHLIGHT,
            CelebrationLevel.TRANSFORMATIVE_STORY
        ]
        
        max_user_index = next((i for i, level in enumerate(level_hierarchy) 
                              if level.value == max_user_level), len(level_hierarchy) - 1)
        current_index = level_hierarchy.index(celebration_level)
        
        if current_index > max_user_index:
            celebration_level = level_hierarchy[max_user_index]
        
        return celebration_level
    
    async def _generate_celebration_content(self, milestone: HealingMilestone,
                                          category: MilestoneCategory,
                                          level: CelebrationLevel,
                                          original_content: str) -> Dict[str, Any]:
        """Generate celebration content"""
        
        # Extract meaningful quotes from original content
        quotes = await self._extract_meaningful_quotes(original_content, category)
        
        # Generate description
        description = await self._generate_milestone_description(milestone, category)
        
        # Generate significance explanation
        significance = await self._generate_significance_explanation(milestone, category)
        
        # Generate celebration message based on level
        message = await self._generate_celebration_message(milestone, category, level)
        
        return {
            'description': description,
            'significance': significance,
            'message': message,
            'quotes': quotes
        }
    
    async def _extract_meaningful_quotes(self, content: str, 
                                       category: MilestoneCategory) -> List[str]:
        """Extract meaningful quotes from the original content"""
        
        quotes = []
        sentences = re.split(r'[.!?]+', content)
        
        # Get category patterns for relevant keywords
        category_patterns = self.milestone_patterns.get(category, {})
        relevant_keywords = []
        
        for pattern_list in category_patterns.values():
            if isinstance(pattern_list, list):
                relevant_keywords.extend(pattern_list)
        
        # Find sentences with relevant keywords
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:  # Meaningful length
                sentence_lower = sentence.lower()
                
                # Check for relevant keywords
                for keyword in relevant_keywords:
                    if keyword in sentence_lower:
                        quotes.append(f'"{sentence}"')
                        break
                
                # Also look for particularly inspiring phrases
                inspiring_phrases = [
                    'i can', 'i am', 'i did', 'i learned', 'i realized',
                    'i understand', 'i chose', 'i decided', 'i feel'
                ]
                
                for phrase in inspiring_phrases:
                    if sentence_lower.startswith(phrase):
                        quotes.append(f'"{sentence}"')
                        break
        
        # Return top 3 most relevant quotes
        return quotes[:3]
    
    async def _generate_milestone_description(self, milestone: HealingMilestone,
                                            category: MilestoneCategory) -> str:
        """Generate a description of the milestone"""
        
        category_descriptions = {
            MilestoneCategory.SURVIVAL_ACHIEVEMENT: "Survival and Perseverance Milestone",
            MilestoneCategory.HEALING_BREAKTHROUGH: "Healing Breakthrough Moment",
            MilestoneCategory.SKILL_MASTERY: "Coping Skill Mastery Achievement",
            MilestoneCategory.CONNECTION_SUCCESS: "Healthy Connection Success",
            MilestoneCategory.SELF_ADVOCACY_WIN: "Self-Advocacy Victory",
            MilestoneCategory.TRAUMA_INTEGRATION: "Trauma Integration Progress",
            MilestoneCategory.WISDOM_SHARING: "Wisdom Sharing and Helping Others",
            MilestoneCategory.RESILIENCE_DISPLAY: "Remarkable Resilience Display",
            MilestoneCategory.BOUNDARY_ESTABLISHMENT: "Healthy Boundary Setting",
            MilestoneCategory.MEANING_DISCOVERY: "Purpose and Meaning Discovery"
        }
        
        base_description = category_descriptions.get(category, "Personal Growth Milestone")
        
        # Add intensity context
        if milestone.intensity == ProgressIntensity.TRANSFORMATIVE:
            return f"Transformative {base_description}"
        elif milestone.intensity == ProgressIntensity.SIGNIFICANT:
            return f"Significant {base_description}"
        else:
            return base_description
    
    async def _generate_significance_explanation(self, milestone: HealingMilestone,
                                               category: MilestoneCategory) -> str:
        """Generate explanation of why this milestone is significant"""
        
        significance_templates = {
            MilestoneCategory.SURVIVAL_ACHIEVEMENT: 
                "This represents remarkable strength and perseverance through difficult times. "
                "Surviving and continuing forward is a testament to inner resilience.",
            
            MilestoneCategory.HEALING_BREAKTHROUGH:
                "Breakthrough moments in healing are profound shifts that can change everything. "
                "These realizations often unlock new levels of understanding and growth.",
            
            MilestoneCategory.SKILL_MASTERY:
                "Mastering healthy coping skills is a significant achievement that provides "
                "lasting tools for navigating life's challenges with greater ease.",
            
            MilestoneCategory.CONNECTION_SUCCESS:
                "Building healthy, supportive connections is fundamental to healing and growth. "
                "These relationships provide foundation for continued wellbeing.",
            
            MilestoneCategory.SELF_ADVOCACY_WIN:
                "Learning to advocate for yourself and set boundaries is empowering and protective. "
                "These skills honor your worth and support long-term wellbeing.",
            
            MilestoneCategory.TRAUMA_INTEGRATION:
                "Integrating traumatic experiences into your life story with wisdom and strength "
                "represents profound healing and post-traumatic growth.",
            
            MilestoneCategory.WISDOM_SHARING:
                "Sharing your wisdom and supporting others shows how healing can transform pain "
                "into purpose and create ripples of positive change.",
            
            MilestoneCategory.RESILIENCE_DISPLAY:
                "Demonstrating resilience in the face of adversity shows the incredible capacity "
                "for human strength and adaptation.",
            
            MilestoneCategory.MEANING_DISCOVERY:
                "Finding purpose and meaning, especially after difficult experiences, represents "
                "a profound shift toward hope and forward movement."
        }
        
        return significance_templates.get(category, 
                                        "This milestone represents meaningful progress in your healing journey.")
    
    async def _generate_celebration_message(self, milestone: HealingMilestone,
                                          category: MilestoneCategory,
                                          level: CelebrationLevel) -> str:
        """Generate appropriate celebration message"""
        
        if level == CelebrationLevel.QUIET_ACKNOWLEDGMENT:
            return "We see and honor your progress. Your strength is noticed and valued."
        
        elif level == CelebrationLevel.COMMUNITY_CELEBRATION:
            return ("Your progress deserves celebration! The growth and strength you've shown "
                   "is inspiring to witness. Our community celebrates this milestone with you.")
        
        elif level == CelebrationLevel.INSPIRING_HIGHLIGHT:
            return ("Your journey and this milestone could inspire others who are facing similar "
                   "challenges. Your strength and progress show what's possible in healing.")
        
        else:  # TRANSFORMATIVE_STORY
            return ("This transformative milestone in your healing journey could offer profound "
                   "hope to others. Your story demonstrates the incredible capacity for human "
                   "healing and growth.")
    
    async def _extract_time_context(self, content: str) -> Optional[str]:
        """Extract time period context from content"""
        
        time_patterns = [
            r'(\d+)\s+days?\s+(clean|sober|without|since)',
            r'(\d+)\s+(weeks?|months?|years?)\s+(clean|sober|without|since|of)',
            r'(first time in)\s+(\d+)\s+(days?|weeks?|months?|years?)',
            r'(anniversary of|year since|been\s+\d+)',
            r'(after\s+\d+\s+(?:days?|weeks?|months?|years?))'
        ]
        
        content_lower = content.lower()
        
        for pattern in time_patterns:
            match = re.search(pattern, content_lower)
            if match:
                return match.group(0)
        
        return None
    
    async def _calculate_optimal_timing(self, milestone: HealingMilestone,
                                      user_context: Dict[str, Any]) -> datetime:
        """Calculate optimal timing for celebration"""
        
        # Default to immediate timing
        optimal_time = datetime.utcnow()
        
        # If user just came out of crisis, wait a bit
        if user_context.get('recent_crisis', False):
            delay = self.timing_considerations['crisis_celebration_delay']
            optimal_time += delay
        
        # Consider user's timezone and preferences
        user_timezone = user_context.get('timezone', 'UTC')
        preferred_time = user_context.get('celebration_time_preference', 'morning')
        
        # Adjust for preferred time of day (simplified)
        current_hour = optimal_time.hour
        if preferred_time == 'morning' and current_hour > 12:
            optimal_time += timedelta(hours=24 - current_hour + 9)  # Next morning at 9am
        elif preferred_time == 'evening' and current_hour < 18:
            optimal_time += timedelta(hours=18 - current_hour)  # Today at 6pm
        
        return optimal_time
    
    async def _create_anonymization_guidelines(self, celebration_level: CelebrationLevel,
                                             privacy_level: str) -> Dict[str, Any]:
        """Create guidelines for anonymizing celebration content"""
        
        guidelines = {
            'remove_identifying_info': True,
            'generalize_specific_details': True,
            'preserve_emotional_core': True
        }
        
        if celebration_level == CelebrationLevel.QUIET_ACKNOWLEDGMENT:
            guidelines.update({
                'share_level': 'none',
                'anonymization_level': 'complete'
            })
        
        elif celebration_level == CelebrationLevel.COMMUNITY_CELEBRATION:
            guidelines.update({
                'share_level': 'community_only',
                'anonymization_level': 'moderate',
                'preserve_journey_context': True
            })
        
        elif celebration_level in [CelebrationLevel.INSPIRING_HIGHLIGHT, 
                                 CelebrationLevel.TRANSFORMATIVE_STORY]:
            guidelines.update({
                'share_level': 'public_anonymous',
                'anonymization_level': 'careful',
                'preserve_inspiring_elements': True,
                'remove_specific_locations': True,
                'generalize_timeframes': True
            })
        
        return guidelines
    
    async def _identify_care_considerations(self, milestone: HealingMilestone,
                                          category: MilestoneCategory) -> List[str]:
        """Identify special care considerations for celebrating this milestone"""
        
        considerations = []
        
        # General considerations
        considerations.append("Celebrate progress, not perfection")
        considerations.append("Avoid toxic positivity - acknowledge the journey's difficulty")
        
        # Category-specific considerations
        if category == MilestoneCategory.TRAUMA_INTEGRATION:
            considerations.extend([
                "Be sensitive to ongoing trauma processing",
                "Acknowledge that healing is non-linear",
                "Respect privacy around trauma details"
            ])
        
        elif category == MilestoneCategory.SURVIVAL_ACHIEVEMENT:
            considerations.extend([
                "Honor the struggle that preceded this achievement",
                "Avoid minimizing the difficulty faced",
                "Be aware of ongoing vulnerability"
            ])
        
        elif category == MilestoneCategory.SELF_ADVOCACY_WIN:
            considerations.extend([
                "Celebrate without creating pressure for continued advocacy",
                "Acknowledge that self-advocacy can be exhausting",
                "Respect boundaries around sharing details"
            ])
        
        # Intensity-based considerations
        if milestone.intensity == ProgressIntensity.TRANSFORMATIVE:
            considerations.append("Allow space for integration of major change")
        
        return considerations
    
    async def _check_privacy_level_allows(self, user_privacy: str, 
                                        max_allowed: str) -> bool:
        """Check if user's privacy level allows for celebration level"""
        
        privacy_hierarchy = ['private', 'community', 'public']
        
        try:
            user_index = privacy_hierarchy.index(user_privacy)
            max_index = privacy_hierarchy.index(max_allowed)
            return user_index >= max_index
        except ValueError:
            return False  # Unknown privacy level - err on side of caution
    
    async def _record_milestone_recognition(self, trigger: CelebrationTrigger) -> None:
        """Record milestone recognition"""
        
        user_id = trigger.user_id
        
        if user_id not in self.recognized_milestones:
            self.recognized_milestones[user_id] = []
        
        self.recognized_milestones[user_id].append(trigger)
        
        # Keep only recent recognitions
        max_recognitions = 20
        if len(self.recognized_milestones[user_id]) > max_recognitions:
            self.recognized_milestones[user_id] = self.recognized_milestones[user_id][-max_recognitions:]
        
        # Update unique users count
        self.recognition_metrics['users_celebrated'] = len(self.recognized_milestones)
    
    async def _get_last_celebration_time(self, user_id: str) -> Optional[datetime]:
        """Get time of user's last celebration"""
        
        user_recognitions = self.recognized_milestones.get(user_id, [])
        
        if not user_recognitions:
            return None
        
        last_recognition = max(user_recognitions, key=lambda r: r.celebration_timing)
        return last_recognition.celebration_timing
    
    def get_user_celebration_history(self, user_id: str) -> Dict[str, Any]:
        """Get user's celebration history"""
        
        user_recognitions = self.recognized_milestones.get(user_id, [])
        
        if not user_recognitions:
            return {'no_celebration_history': True}
        
        # Analyze celebration patterns
        categories = {}
        levels = {}
        
        for recognition in user_recognitions:
            category = recognition.category.value
            level = recognition.celebration_level.value
            
            categories[category] = categories.get(category, 0) + 1
            levels[level] = levels.get(level, 0) + 1
        
        recent_recognitions = [
            r for r in user_recognitions
            if r.celebration_timing > datetime.utcnow() - timedelta(days=30)
        ]
        
        return {
            'total_milestones_celebrated': len(user_recognitions),
            'recent_celebrations': len(recent_recognitions),
            'celebration_categories': categories,
            'celebration_levels': levels,
            'latest_milestone': user_recognitions[-1].category.value if user_recognitions else None,
            'inspiration_stories_shared': sum(1 for r in user_recognitions 
                                            if r.celebration_level in [CelebrationLevel.INSPIRING_HIGHLIGHT,
                                                                       CelebrationLevel.TRANSFORMATIVE_STORY]),
            'celebration_trajectory': 'positive' if len(recent_recognitions) > 0 else 'stable'
        }
    
    def get_recognition_metrics(self) -> Dict[str, Any]:
        """Get milestone recognition metrics"""
        
        return {
            'milestones_recognized': self.recognition_metrics['milestones_recognized'],
            'celebrations_triggered': self.recognition_metrics['celebrations_triggered'],
            'inspiring_stories_created': self.recognition_metrics['inspiring_stories_created'],
            'quiet_acknowledgments_given': self.recognition_metrics['quiet_acknowledgments_given'],
            'users_celebrated': self.recognition_metrics['users_celebrated'],
            'milestone_categories_recognized': list(MilestoneCategory),
            'celebration_levels_available': list(CelebrationLevel),
            'system_health': {
                'milestone_patterns_loaded': len(self.milestone_patterns) > 0,
                'celebration_thresholds_loaded': len(self.celebration_thresholds) > 0,
                'inspiration_indicators_loaded': len(self.inspiration_indicators) > 0,
                'timing_considerations_loaded': len(self.timing_considerations) > 0
            }
        }