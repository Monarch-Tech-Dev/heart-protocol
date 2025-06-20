"""
Guardian Energy Rising Feed - Main Implementation

The third of the Four Sacred Feeds that celebrates healing journeys, progress
milestones, and the strength that emerges from transformation. Amplifies 
stories of resilience and growth to inspire hope in others.

Core Philosophy: "From the depths of struggle, guardian energy rises."
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging

from ...core.base import FeedItem, CareLevel, FeedGenerator, Post, CareAssessment
from ...core.care_detection import CareDetectionEngine
from .progress_detection import ProgressDetector, HealingMilestone, ProgressType, ProgressIntensity
from .milestone_recognition import MilestoneRecognizer, CelebrationTrigger, CelebrationLevel, MilestoneCategory
from .celebration_engine import CelebrationEngine, CelebrationContent, CelebrationStyle

logger = logging.getLogger(__name__)


class GuardianEnergyRisingFeed(FeedGenerator):
    """
    The Guardian Energy Rising Feed celebrates healing journeys and milestones.
    
    Key Features:
    - Progress detection in user posts and interactions
    - Milestone recognition and categorization
    - Meaningful celebration creation
    - Inspiration story generation (anonymized)
    - Hope amplification through success stories
    - Trauma-informed celebration approaches
    - Privacy-respecting sharing guidelines
    """
    
    def __init__(self, config: Dict[str, Any], care_detection_engine: CareDetectionEngine):
        super().__init__(config)
        self.feed_type = "guardian_energy_rising"
        self.care_engine = care_detection_engine
        
        # Initialize core components
        self.progress_detector = ProgressDetector(config.get('progress_detection', {}))
        self.milestone_recognizer = MilestoneRecognizer(
            config.get('milestone_recognition', {}), 
            self.progress_detector
        )
        self.celebration_engine = CelebrationEngine(config.get('celebration', {}))
        
        # Feed state management
        self.recent_posts = {}          # post_id -> processed_post_data
        self.active_celebrations = {}   # celebration_id -> celebration_data
        self.inspiration_stories = {}   # story_id -> anonymized_story_data
        
        # Feed configuration
        self.max_celebrations_per_user_per_week = config.get('max_celebrations_per_user_per_week', 2)
        self.inspiration_story_ttl = timedelta(days=config.get('inspiration_story_ttl_days', 30))
        self.celebration_cooldown = timedelta(hours=config.get('celebration_cooldown_hours', 24))
        
        logger.info("Guardian Energy Rising Feed initialized")
    
    async def generate_feed_items(self, user_id: str, 
                                user_context: Dict[str, Any],
                                care_assessment: Optional[CareAssessment] = None) -> List[FeedItem]:
        """
        Generate Guardian Energy Rising feed items for a user.
        
        This includes:
        - Personal milestone celebrations (if any)
        - Inspiring anonymized stories from the community
        - Progress encouragement and hope amplification
        - Gentle reminders of growth and resilience
        """
        try:
            feed_items = []
            
            # Generate personal celebration items
            personal_items = await self._generate_personal_celebration_items(
                user_id, user_context, care_assessment
            )
            feed_items.extend(personal_items)
            
            # Generate inspiration story items
            inspiration_items = await self._generate_inspiration_story_items(
                user_id, user_context, care_assessment
            )
            feed_items.extend(inspiration_items)
            
            # Generate hope amplification items
            hope_items = await self._generate_hope_amplification_items(
                user_id, user_context, care_assessment
            )
            feed_items.extend(hope_items)
            
            # Generate community celebration items
            community_items = await self._generate_community_celebration_items(
                user_id, user_context
            )
            feed_items.extend(community_items)
            
            logger.info(f"Generated {len(feed_items)} Guardian Energy Rising items for user {user_id[:8]}...")
            
            return feed_items
            
        except Exception as e:
            logger.error(f"Error generating Guardian Energy Rising feed for user {user_id[:8]}...: {e}")
            return []
    
    async def process_post_for_progress_detection(self, post: Post, 
                                                care_assessment: CareAssessment,
                                                user_context: Dict[str, Any]) -> None:
        """Process a post to detect progress and potentially create celebrations"""
        
        try:
            # Store post for reference
            self.recent_posts[post.id] = {
                'post': post,
                'care_assessment': care_assessment,
                'user_context': user_context,
                'processed_at': datetime.utcnow()
            }
            
            # Detect healing progress/milestones
            healing_milestone = await self.progress_detector.detect_progress(
                post, care_assessment, user_context
            )
            
            if healing_milestone:
                await self._handle_detected_milestone(healing_milestone, post, user_context)
            
            # Clean up old posts
            await self._cleanup_old_posts()
            
        except Exception as e:
            logger.error(f"Error processing post for progress detection: {e}")
    
    async def _handle_detected_milestone(self, milestone: HealingMilestone, 
                                       post: Post, user_context: Dict[str, Any]) -> None:
        """Handle a detected healing milestone"""
        
        try:
            logger.info(f"Detected milestone for user {milestone.user_id[:8]}...: "
                       f"{milestone.milestone_type.value} (intensity: {milestone.intensity.value})")
            
            # Check if we should create a celebration
            if await self._should_create_celebration(milestone, user_context):
                
                # Recognize the milestone
                celebration_trigger = await self.milestone_recognizer.recognize_milestone(
                    milestone, user_context, post
                )
                
                if celebration_trigger:
                    await self._create_and_store_celebration(celebration_trigger, user_context)
                    
                    # Consider creating inspiration story if appropriate
                    if await self._should_create_inspiration_story(celebration_trigger):
                        await self._create_inspiration_story(celebration_trigger, milestone)
        
        except Exception as e:
            logger.error(f"Error handling detected milestone: {e}")
    
    async def _should_create_celebration(self, milestone: HealingMilestone, 
                                       user_context: Dict[str, Any]) -> bool:
        """Determine if we should create a celebration for this milestone"""
        
        # Check if milestone is celebration-worthy
        if not milestone.is_celebration_worthy():
            return False
        
        # Check user's celebration preferences
        if user_context.get('celebration_preference') == 'none':
            return False
        
        # Check celebration frequency limits
        recent_celebrations = await self._get_recent_user_celebrations(milestone.user_id)
        if len(recent_celebrations) >= self.max_celebrations_per_user_per_week:
            return False
        
        # Check cooldown period
        if recent_celebrations:
            last_celebration_time = max(c['created_at'] for c in recent_celebrations)
            if datetime.utcnow() - last_celebration_time < self.celebration_cooldown:
                return False
        
        # Don't celebrate if user is currently in crisis (unless it's crisis recovery)
        if (user_context.get('current_crisis', False) and 
            milestone.milestone_type != ProgressType.CRISIS_STABILIZATION):
            return False
        
        return True
    
    async def _create_and_store_celebration(self, trigger: CelebrationTrigger, 
                                          user_context: Dict[str, Any]) -> None:
        """Create and store a celebration"""
        
        try:
            # Create celebration content
            celebration_content = await self.celebration_engine.create_celebration(
                trigger, user_context
            )
            
            # Store celebration
            celebration_id = f"celebration_{trigger.milestone_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            self.active_celebrations[celebration_id] = {
                'celebration_id': celebration_id,
                'trigger': trigger,
                'content': celebration_content,
                'user_id': trigger.user_id,
                'created_at': datetime.utcnow(),
                'status': 'active',
                'privacy_level': trigger.privacy_level,
                'celebration_level': trigger.celebration_level,
                'expires_at': trigger.expiry_time
            }
            
            logger.info(f"Created celebration {celebration_id} for user {trigger.user_id[:8]}... "
                       f"(level: {trigger.celebration_level.value})")
            
        except Exception as e:
            logger.error(f"Error creating and storing celebration: {e}")
    
    async def _should_create_inspiration_story(self, trigger: CelebrationTrigger) -> bool:
        """Determine if we should create an inspiration story"""
        
        return (trigger.celebration_level in [CelebrationLevel.INSPIRING_HIGHLIGHT, 
                                            CelebrationLevel.TRANSFORMATIVE_STORY] and
                trigger.privacy_level in ['public', 'community'] and
                trigger.inspiration_potential > 0.7)
    
    async def _create_inspiration_story(self, trigger: CelebrationTrigger, 
                                      milestone: HealingMilestone) -> None:
        """Create anonymized inspiration story"""
        
        try:
            story_id = f"story_{trigger.milestone_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            # Create anonymized story
            anonymized_story = await self._generate_anonymized_story(trigger, milestone)
            
            self.inspiration_stories[story_id] = {
                'story_id': story_id,
                'category': trigger.category,
                'inspiration_level': trigger.inspiration_potential,
                'story_content': anonymized_story,
                'created_at': datetime.utcnow(),
                'expires_at': datetime.utcnow() + self.inspiration_story_ttl,
                'engagement_metrics': {
                    'views': 0,
                    'hope_reactions': 0,
                    'shares': 0
                }
            }
            
            logger.info(f"Created inspiration story {story_id} for category {trigger.category.value}")
            
        except Exception as e:
            logger.error(f"Error creating inspiration story: {e}")
    
    async def _generate_personal_celebration_items(self, user_id: str, 
                                                 user_context: Dict[str, Any],
                                                 care_assessment: Optional[CareAssessment]) -> List[FeedItem]:
        """Generate personal celebration items for user"""
        
        items = []
        
        # Find active celebrations for this user
        user_celebrations = [
            celebration for celebration in self.active_celebrations.values()
            if (celebration['user_id'] == user_id and 
                celebration['status'] == 'active' and
                celebration['expires_at'] > datetime.utcnow())
        ]
        
        for celebration in user_celebrations:
            item = await self._create_celebration_feed_item(celebration, user_context)
            if item:
                items.append(item)
        
        return items
    
    async def _generate_inspiration_story_items(self, user_id: str,
                                              user_context: Dict[str, Any],
                                              care_assessment: Optional[CareAssessment]) -> List[FeedItem]:
        """Generate inspiration story items"""
        
        items = []
        
        # Don't show inspiration stories if user is in crisis
        if care_assessment and care_assessment.care_level == CareLevel.CRISIS:
            return items
        
        # Find relevant inspiration stories
        relevant_stories = await self._find_relevant_inspiration_stories(user_context)
        
        # Create feed items for top stories (limit to 1-2)
        for story in relevant_stories[:2]:
            item = await self._create_inspiration_story_feed_item(story, user_context)
            if item:
                items.append(item)
        
        return items
    
    async def _generate_hope_amplification_items(self, user_id: str,
                                               user_context: Dict[str, Any],
                                               care_assessment: Optional[CareAssessment]) -> List[FeedItem]:
        """Generate hope amplification items"""
        
        items = []
        
        # Create hope amplification based on user's context
        if care_assessment and care_assessment.care_level in [CareLevel.MODERATE, CareLevel.HIGH]:
            item = await self._create_hope_amplification_item(user_context, care_assessment)
            if item:
                items.append(item)
        
        return items
    
    async def _generate_community_celebration_items(self, user_id: str,
                                                  user_context: Dict[str, Any]) -> List[FeedItem]:
        """Generate community celebration items"""
        
        items = []
        
        # Find recent community celebrations that can be shared
        community_celebrations = [
            celebration for celebration in self.active_celebrations.values()
            if (celebration['user_id'] != user_id and  # Not user's own celebration
                celebration['celebration_level'] in [CelebrationLevel.COMMUNITY_CELEBRATION,
                                                    CelebrationLevel.INSPIRING_HIGHLIGHT] and
                celebration['privacy_level'] in ['public', 'community'] and
                celebration['expires_at'] > datetime.utcnow())
        ]
        
        # Sort by inspiration potential and recency
        community_celebrations.sort(
            key=lambda c: (c['trigger'].inspiration_potential, c['created_at']), 
            reverse=True
        )
        
        # Create feed item for top community celebration
        if community_celebrations:
            item = await self._create_community_celebration_feed_item(
                community_celebrations[0], user_context
            )
            if item:
                items.append(item)
        
        return items
    
    async def _create_celebration_feed_item(self, celebration: Dict[str, Any],
                                          user_context: Dict[str, Any]) -> Optional[FeedItem]:
        """Create feed item for personal celebration"""
        
        try:
            trigger = celebration['trigger']
            content = celebration['content']
            
            # Create celebration message
            celebration_message = f"""## {content.headline}

{content.body_message}

{f'*"{content.inspiration_quote}"*' if content.inspiration_quote else ''}

### Reflection
{chr(10).join(f'• {prompt}' for prompt in content.reflection_prompts[:2])}

{f'### Celebration Ritual{chr(10)}{content.celebration_ritual}' if content.celebration_ritual else ''}

### Care Reminders
{chr(10).join(f'• {reminder}' for reminder in content.care_reminders)}

---
*Your progress matters. Your journey is honored. 💝*"""
            
            # Determine care level
            care_level = CareLevel.LOW  # Celebrations are generally uplifting
            if trigger.celebration_level == CelebrationLevel.TRANSFORMATIVE_STORY:
                care_level = CareLevel.MODERATE  # May need processing time
            
            metadata = {
                'feed_type': 'guardian_energy_rising',
                'item_type': 'personal_celebration',
                'celebration_id': celebration['celebration_id'],
                'milestone_category': trigger.category.value,
                'celebration_level': trigger.celebration_level.value,
                'milestone_type': trigger.milestone_description,
                'visual_elements': content.visual_elements,
                'privacy_level': trigger.privacy_level,
                'inspiration_potential': trigger.inspiration_potential
            }
            
            return FeedItem(
                content=celebration_message,
                care_level=care_level,
                source_type="personal_celebration",
                metadata=metadata,
                reasoning=f"Celebrating personal milestone: {trigger.category.value.replace('_', ' ')}",
                confidence_score=0.9,
                requires_human_review=False
            )
            
        except Exception as e:
            logger.error(f"Error creating celebration feed item: {e}")
            return None
    
    async def _create_inspiration_story_feed_item(self, story: Dict[str, Any],
                                                user_context: Dict[str, Any]) -> Optional[FeedItem]:
        """Create feed item for inspiration story"""
        
        try:
            story_content = f"""## 🌟 A Story of Hope and Resilience

{story['story_content']}

---

*This anonymized story comes from someone in our caring community. Their journey reminds us that healing is possible and that we're not alone in our struggles.*

### Reflection
• What resonates with you about this story?
• How might this person's journey offer hope for your own?

*Stories of resilience light the way for others walking similar paths. 💫*"""
            
            metadata = {
                'feed_type': 'guardian_energy_rising',
                'item_type': 'inspiration_story',
                'story_id': story['story_id'],
                'story_category': story['category'].value,
                'inspiration_level': story['inspiration_level'],
                'anonymized': True,
                'community_generated': True
            }
            
            return FeedItem(
                content=story_content,
                care_level=CareLevel.LOW,
                source_type="inspiration_story",
                metadata=metadata,
                reasoning="Inspiring anonymized story from community member's healing journey",
                confidence_score=0.8,
                requires_human_review=False
            )
            
        except Exception as e:
            logger.error(f"Error creating inspiration story feed item: {e}")
            return None
    
    async def _create_hope_amplification_item(self, user_context: Dict[str, Any],
                                            care_assessment: CareAssessment) -> Optional[FeedItem]:
        """Create hope amplification item"""
        
        try:
            # Generate hope-focused content based on care level
            if care_assessment.care_level == CareLevel.HIGH:
                content = """## 🌅 Gentle Reminder: You Are Stronger Than You Know

In moments when the weight feels heavy and the path ahead unclear, remember this truth: you have survived every difficult day in your life so far. That's a 100% success rate.

Healing isn't linear. Progress isn't always visible. But every breath you take, every moment you choose to continue, every small step forward matters more than you might realize.

### Today's Affirmation
*"I am exactly where I need to be in my healing journey. My pace is perfect for me."*

### Gentle Reminders
• Progress comes in all sizes - honor the small victories
• It's okay to rest when you need to
• You don't have to heal according to anyone else's timeline
• Your resilience is remarkable, even when you don't feel resilient

*You matter. Your healing matters. We're here with you. 💙*"""
                
            else:  # MODERATE
                content = """## ✨ Celebrating the Strength in Your Journey

Every person walking a healing path carries within them a unique form of guardian energy - the strength that emerges from choosing growth over stagnation, hope over despair, love over fear.

Today, we want to acknowledge the guardian energy rising within you. It shows up in your willingness to keep trying, your courage to be vulnerable, your commitment to your own growth.

### Recognition
Your journey is creating positive ripples you may not even see:
• Your courage inspires others to be brave
• Your healing contributes to collective healing
• Your growth makes space for others to grow

### Forward Movement
Consider this: What would it look like to fully embrace the guardian energy within you today?

*Your transformation is not just personal - it's a gift to the world. 🌟*"""
            
            metadata = {
                'feed_type': 'guardian_energy_rising',
                'item_type': 'hope_amplification',
                'care_level': care_assessment.care_level.value,
                'personalized': True,
                'inspirational': True
            }
            
            return FeedItem(
                content=content,
                care_level=care_assessment.care_level,
                source_type="hope_amplification",
                metadata=metadata,
                reasoning="Hope amplification content based on user's current care needs",
                confidence_score=0.85,
                requires_human_review=False
            )
            
        except Exception as e:
            logger.error(f"Error creating hope amplification item: {e}")
            return None
    
    async def _create_community_celebration_feed_item(self, celebration: Dict[str, Any],
                                                    user_context: Dict[str, Any]) -> Optional[FeedItem]:
        """Create feed item for community celebration"""
        
        try:
            trigger = celebration['trigger']
            
            # Create anonymized community celebration
            content = f"""## 🎉 Community Celebration

Today we celebrate a beautiful milestone achieved by one of our community members!

**Milestone Category:** {trigger.category.value.replace('_', ' ').title()}

{trigger.significance_explanation}

### What This Teaches Us
• Healing and growth are possible for all of us
• Every journey is unique and valuable
• Community support amplifies individual strength
• Progress deserves recognition and celebration

### Reflection
• How might this milestone inspire hope in your own journey?
• What progress in your own life deserves recognition today?

*When one of us grows, we all grow. When one of us heals, we all become more whole. 💝*"""
            
            metadata = {
                'feed_type': 'guardian_energy_rising',
                'item_type': 'community_celebration',
                'milestone_category': trigger.category.value,
                'anonymized': True,
                'community_milestone': True,
                'inspiration_potential': trigger.inspiration_potential
            }
            
            return FeedItem(
                content=content,
                care_level=CareLevel.LOW,
                source_type="community_celebration",
                metadata=metadata,
                reasoning="Celebrating anonymized community member's healing milestone",
                confidence_score=0.8,
                requires_human_review=False
            )
            
        except Exception as e:
            logger.error(f"Error creating community celebration feed item: {e}")
            return None
    
    async def _find_relevant_inspiration_stories(self, user_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find inspiration stories relevant to user's context"""
        
        # Get active stories
        active_stories = [
            story for story in self.inspiration_stories.values()
            if story['expires_at'] > datetime.utcnow()
        ]
        
        # Filter by relevance
        relevant_stories = []
        user_interests = user_context.get('healing_interests', [])
        user_stage = user_context.get('healing_stage', 'unknown')
        
        for story in active_stories:
            relevance_score = 0.0
            
            # Category relevance
            if story['category'].value in user_interests:
                relevance_score += 0.4
            
            # Inspiration level
            relevance_score += story['inspiration_level'] * 0.3
            
            # Recency bonus
            days_old = (datetime.utcnow() - story['created_at']).days
            recency_bonus = max(0, (7 - days_old) / 7 * 0.3)
            relevance_score += recency_bonus
            
            if relevance_score > 0.4:
                story['relevance_score'] = relevance_score
                relevant_stories.append(story)
        
        # Sort by relevance
        relevant_stories.sort(key=lambda s: s['relevance_score'], reverse=True)
        
        return relevant_stories
    
    async def _generate_anonymized_story(self, trigger: CelebrationTrigger, 
                                       milestone: HealingMilestone) -> str:
        """Generate anonymized inspiration story"""
        
        category_name = trigger.category.value.replace('_', ' ')
        
        story_templates = {
            MilestoneCategory.SURVIVAL_ACHIEVEMENT: 
                "A community member recently shared that they've reached an important survival milestone. "
                "Through incredibly difficult circumstances, they found the strength to keep going. "
                "Their journey reminds us that survival itself is an achievement worthy of celebration.",
            
            MilestoneCategory.HEALING_BREAKTHROUGH:
                "Someone in our community experienced a profound breakthrough in their healing journey. "
                "After working through challenging emotions and experiences, pieces began falling into place. "
                "Their story shows us that breakthrough moments are possible, even after long periods of struggle.",
            
            MilestoneCategory.SKILL_MASTERY:
                "A community member recently mastered important coping skills that have transformed their daily life. "
                "Through consistent practice and self-compassion, they developed tools that now serve them well. "
                "Their dedication shows us the power of investing in our own healing toolkit.",
            
            MilestoneCategory.WISDOM_SHARING:
                "One of our community members has begun helping others on similar healing journeys. "
                "By sharing their experience and wisdom, they're transforming their pain into purpose. "
                "Their generosity reminds us how healing can create ripples of positive change."
        }
        
        base_story = story_templates.get(
            trigger.category, 
            f"A community member achieved a meaningful milestone in their {category_name} journey."
        )
        
        # Add time context if available
        if trigger.time_period_context:
            base_story += f" This achievement came {trigger.time_period_context.lower()}, "
            base_story += "demonstrating the power of persistence and commitment to healing."
        
        # Add inspiration
        base_story += " Their story is a beacon of hope for anyone facing similar challenges, "
        base_story += "showing us that healing is possible and that every step forward matters."
        
        return base_story
    
    async def _get_recent_user_celebrations(self, user_id: str) -> List[Dict[str, Any]]:
        """Get user's recent celebrations"""
        
        one_week_ago = datetime.utcnow() - timedelta(days=7)
        
        return [
            celebration for celebration in self.active_celebrations.values()
            if (celebration['user_id'] == user_id and 
                celebration['created_at'] > one_week_ago)
        ]
    
    async def _cleanup_old_posts(self) -> None:
        """Clean up old processed posts"""
        
        cutoff_time = datetime.utcnow() - timedelta(hours=48)
        
        old_posts = [
            post_id for post_id, data in self.recent_posts.items()
            if data['processed_at'] < cutoff_time
        ]
        
        for post_id in old_posts:
            del self.recent_posts[post_id]
        
        # Also cleanup expired celebrations and stories
        await self._cleanup_expired_content()
    
    async def _cleanup_expired_content(self) -> None:
        """Clean up expired celebrations and stories"""
        
        now = datetime.utcnow()
        
        # Cleanup expired celebrations
        expired_celebrations = [
            celebration_id for celebration_id, celebration in self.active_celebrations.items()
            if celebration['expires_at'] < now
        ]
        
        for celebration_id in expired_celebrations:
            del self.active_celebrations[celebration_id]
        
        # Cleanup expired stories
        expired_stories = [
            story_id for story_id, story in self.inspiration_stories.items()
            if story['expires_at'] < now
        ]
        
        for story_id in expired_stories:
            del self.inspiration_stories[story_id]
    
    async def get_feed_analytics(self) -> Dict[str, Any]:
        """Get analytics for Guardian Energy Rising feed"""
        
        try:
            # Get component analytics
            progress_metrics = self.progress_detector.get_detection_metrics()
            recognition_metrics = self.milestone_recognizer.get_recognition_metrics()
            celebration_metrics = self.celebration_engine.get_celebration_metrics()
            
            # Calculate feed-specific metrics
            active_celebrations_count = len(self.active_celebrations)
            active_stories_count = len(self.inspiration_stories)
            
            total_inspiration_potential = sum(
                story['inspiration_level'] for story in self.inspiration_stories.values()
            ) / max(1, len(self.inspiration_stories))
            
            analytics = {
                'feed_type': 'guardian_energy_rising',
                'progress_detection': progress_metrics,
                'milestone_recognition': recognition_metrics,
                'celebration_engine': celebration_metrics,
                'feed_activity': {
                    'active_celebrations': active_celebrations_count,
                    'active_inspiration_stories': active_stories_count,
                    'recent_posts_processed': len(self.recent_posts),
                    'average_inspiration_potential': round(total_inspiration_potential, 2)
                },
                'content_distribution': {
                    'celebration_levels': self._analyze_celebration_levels(),
                    'milestone_categories': self._analyze_milestone_categories(),
                    'privacy_levels': self._analyze_privacy_levels()
                },
                'system_health': {
                    'progress_detector_active': True,
                    'milestone_recognizer_active': True,
                    'celebration_engine_active': True,
                    'content_generation_healthy': active_celebrations_count > 0 or active_stories_count > 0
                },
                'generated_at': datetime.utcnow().isoformat()
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error generating Guardian Energy Rising analytics: {e}")
            return {'error': 'Unable to generate analytics'}
    
    def _analyze_celebration_levels(self) -> Dict[str, int]:
        """Analyze distribution of celebration levels"""
        
        levels = {}
        for celebration in self.active_celebrations.values():
            level = celebration['celebration_level'].value
            levels[level] = levels.get(level, 0) + 1
        
        return levels
    
    def _analyze_milestone_categories(self) -> Dict[str, int]:
        """Analyze distribution of milestone categories"""
        
        categories = {}
        for celebration in self.active_celebrations.values():
            category = celebration['trigger'].category.value
            categories[category] = categories.get(category, 0) + 1
        
        for story in self.inspiration_stories.values():
            category = story['category'].value
            categories[category] = categories.get(category, 0) + 1
        
        return categories
    
    def _analyze_privacy_levels(self) -> Dict[str, int]:
        """Analyze distribution of privacy levels"""
        
        privacy_levels = {}
        for celebration in self.active_celebrations.values():
            level = celebration['privacy_level']
            privacy_levels[level] = privacy_levels.get(level, 0) + 1
        
        return privacy_levels