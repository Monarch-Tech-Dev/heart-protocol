"""
Base Feed Generator Framework

Core framework for building caring feed algorithms that prioritize
human wellbeing and authentic connection over engagement metrics.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

from ..base import BaseFeedGenerator, Post, FeedItem, FeedSkeleton, FeedType, CareLevel

logger = logging.getLogger(__name__)


@dataclass
class FeedConfiguration:
    """Configuration for feed generation"""
    feed_type: FeedType
    max_posts_per_hour: int = 10
    diversity_factor: float = 0.7  # 0.0 = no diversity, 1.0 = maximum diversity
    care_weight: float = 0.8  # How much to weight care indicators
    recency_weight: float = 0.3  # How much to weight recent posts
    user_preferences: Dict[str, Any] = None
    ethical_constraints: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.user_preferences is None:
            self.user_preferences = {}
        if self.ethical_constraints is None:
            self.ethical_constraints = {
                'no_trauma_dumping': True,
                'consent_required': True,
                'privacy_first': True,
                'opt_out_friendly': True
            }


class CaringFeedGenerator(BaseFeedGenerator):
    """
    Enhanced base class for caring feed generators with built-in
    ethical constraints and wellbeing-focused algorithms.
    """
    
    def __init__(self, feed_type: FeedType, config: Dict[str, Any]):
        super().__init__(feed_type, config)
        self.feed_config = FeedConfiguration(feed_type=feed_type, **config.get('feed_config', {}))
        self.post_cache = {}  # Simple in-memory cache
        self.user_preferences = {}
    
    async def generate_feed(self, user_id: str, cursor: Optional[str] = None, 
                           limit: int = 50) -> FeedSkeleton:
        """
        Generate a caring feed with ethical constraints and user preferences.
        """
        try:
            # Check user consent and preferences
            if not await self._check_user_consent(user_id):
                return FeedSkeleton(cursor=None, feed=[])
            
            # Get user preferences and context
            user_context = await self._get_user_context(user_id)
            
            # Get candidate posts
            candidate_posts = await self._get_candidate_posts(user_context, cursor, limit * 3)
            
            # Score and rank posts using caring algorithm
            scored_posts = await self._score_posts_for_care(candidate_posts, user_context)
            
            # Apply ethical filters
            filtered_posts = await self._apply_ethical_filters(scored_posts, user_context)
            
            # Ensure diversity and avoid overwhelming user
            final_posts = await self._ensure_feed_diversity(filtered_posts, limit)
            
            # Create feed skeleton
            feed_items = [{"post": post.uri} for post in final_posts]
            next_cursor = final_posts[-1].uri if final_posts else None
            
            # Log feed generation for metrics
            await self._log_feed_generation(user_id, len(final_posts), user_context)
            
            return FeedSkeleton(cursor=next_cursor, feed=feed_items)
            
        except Exception as e:
            self.logger.error(f"Error generating feed for {user_id}: {e}")
            return FeedSkeleton(cursor=None, feed=[])
    
    async def _check_user_consent(self, user_id: str) -> bool:
        """Check if user has consented to this type of feed"""
        # In production, this would check user preferences database
        return True  # Default to consent for now
    
    async def _get_user_context(self, user_id: str) -> Dict[str, Any]:
        """Get user context for personalized feed generation"""
        return {
            'user_id': user_id,
            'timezone': 'UTC',  # Would be fetched from user profile
            'emotional_capacity': await self._assess_user_emotional_capacity(user_id),
            'care_preferences': await self._get_user_care_preferences(user_id),
            'current_time': datetime.utcnow()
        }
    
    async def _assess_user_emotional_capacity(self, user_id: str) -> Dict[str, float]:
        """
        Assess user's current emotional capacity to receive different types of content.
        This helps avoid overwhelming users who are already struggling.
        """
        # This would integrate with user's recent activity, explicit preferences,
        # and possibly mood indicators they've shared
        return {
            'can_handle_heavy_content': 0.7,
            'needs_gentle_approach': 0.3,
            'seeking_inspiration': 0.5,
            'wants_practical_help': 0.6
        }
    
    async def _get_user_care_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user's care preferences and boundaries"""
        return {
            'care_types': ['gentle_reminders', 'peer_support', 'resources'],
            'trigger_warnings': ['eating_disorders', 'self_harm'],
            'preferred_response_tone': 'gentle_supportive',
            'max_care_posts_per_day': 5,
            'opt_out_keywords': ['unsubscribe', 'stop', 'no more']
        }
    
    @abstractmethod
    async def _get_candidate_posts(self, user_context: Dict[str, Any], 
                                  cursor: Optional[str], limit: int) -> List[Post]:
        """Get candidate posts for this feed type"""
        pass
    
    async def _score_posts_for_care(self, posts: List[Post], 
                                   user_context: Dict[str, Any]) -> List[Tuple[Post, float]]:
        """
        Score posts based on their caring potential and appropriateness for the user.
        This is where the caring algorithm magic happens.
        """
        scored_posts = []
        
        for post in posts:
            # Base score from feed-specific scoring
            base_score = await self.score_post(post, user_context)
            
            # Adjust for care indicators
            care_score = await self._calculate_care_score(post, user_context)
            
            # Adjust for user emotional capacity
            capacity_adjusted_score = await self._adjust_for_emotional_capacity(
                base_score, care_score, post, user_context
            )
            
            # Adjust for timing and recency
            time_adjusted_score = await self._adjust_for_timing(
                capacity_adjusted_score, post, user_context
            )
            
            scored_posts.append((post, time_adjusted_score))
        
        # Sort by score (highest first)
        scored_posts.sort(key=lambda x: x[1], reverse=True)
        return scored_posts
    
    async def _calculate_care_score(self, post: Post, user_context: Dict[str, Any]) -> float:
        """Calculate how much this post demonstrates caring/supportive qualities"""
        care_indicators = {
            'supportive_language': 0.0,
            'empathy_expressed': 0.0,
            'resources_shared': 0.0,
            'hope_offered': 0.0,
            'authentic_vulnerability': 0.0
        }
        
        text = post.text.lower()
        
        # Look for supportive language
        supportive_phrases = [
            'you\'re not alone', 'here for you', 'sending love', 'you matter',
            'proud of you', 'you\'re doing great', 'hang in there', 'you got this'
        ]
        for phrase in supportive_phrases:
            if phrase in text:
                care_indicators['supportive_language'] += 0.3
        
        # Look for empathy
        empathy_phrases = [
            'i understand', 'been there', 'felt that way', 'know how you feel',
            'similar experience', 'went through this'
        ]
        for phrase in empathy_phrases:
            if phrase in text:
                care_indicators['empathy_expressed'] += 0.4
        
        # Look for resource sharing
        resource_indicators = [
            'helped me', 'try this', 'found this helpful', 'resource',
            'therapy', 'counseling', 'support group', 'helpline'
        ]
        for indicator in resource_indicators:
            if indicator in text:
                care_indicators['resources_shared'] += 0.3
        
        # Look for hope and positivity (but not toxic positivity)
        hope_phrases = [
            'gets better', 'found hope', 'recovery', 'healing',
            'small steps', 'progress', 'growth', 'breakthrough'
        ]
        for phrase in hope_phrases:
            if phrase in text:
                care_indicators['hope_offered'] += 0.2
        
        # Calculate overall care score
        total_care_score = sum(care_indicators.values())
        return min(1.0, total_care_score)
    
    async def _adjust_for_emotional_capacity(self, base_score: float, care_score: float,
                                           post: Post, user_context: Dict[str, Any]) -> float:
        """
        Adjust post score based on user's current emotional capacity.
        Avoid overwhelming users who are already struggling.
        """
        capacity = user_context.get('emotional_capacity', {})
        
        # If user has low capacity for heavy content, boost lighter posts
        if capacity.get('can_handle_heavy_content', 0.5) < 0.3:
            if care_score > 0.5:  # This is caring/supportive content
                return base_score * 1.2
            else:  # Might be heavy content
                return base_score * 0.7
        
        # If user needs gentle approach, prioritize gentle content
        if capacity.get('needs_gentle_approach', 0.5) > 0.7:
            gentle_indicators = ['gentle', 'soft', 'kind', 'warm', 'peaceful']
            if any(word in post.text.lower() for word in gentle_indicators):
                return base_score * 1.3
        
        return base_score
    
    async def _adjust_for_timing(self, score: float, post: Post, 
                               user_context: Dict[str, Any]) -> float:
        """Adjust score based on timing and recency"""
        current_time = user_context.get('current_time', datetime.utcnow())
        post_age = current_time - post.created_at
        
        # Prefer recent posts but not too recent (avoid spam)
        if post_age < timedelta(hours=1):
            return score * 0.8  # Too recent, might be spam
        elif post_age < timedelta(hours=24):
            return score * 1.1  # Recent and good
        elif post_age < timedelta(days=7):
            return score * 1.0  # Still relevant
        else:
            return score * 0.9  # Older content
    
    async def _apply_ethical_filters(self, scored_posts: List[Tuple[Post, float]],
                                   user_context: Dict[str, Any]) -> List[Post]:
        """Apply ethical filters to ensure feed meets caring standards"""
        filtered_posts = []
        care_preferences = user_context.get('care_preferences', {})
        
        for post, score in scored_posts:
            # Check for trigger warnings
            if await self._contains_triggers(post, care_preferences.get('trigger_warnings', [])):
                continue
            
            # Check for opt-out keywords
            if await self._contains_opt_out_keywords(post, care_preferences.get('opt_out_keywords', [])):
                continue
            
            # Ensure post meets caring standards
            if score < 0.3:  # Minimum caring threshold
                continue
            
            filtered_posts.append(post)
        
        return filtered_posts
    
    async def _contains_triggers(self, post: Post, trigger_warnings: List[str]) -> bool:
        """Check if post contains content that user wants to avoid"""
        text = post.text.lower()
        return any(trigger in text for trigger in trigger_warnings)
    
    async def _contains_opt_out_keywords(self, post: Post, opt_out_keywords: List[str]) -> bool:
        """Check if post contains keywords indicating user wants to opt out"""
        text = post.text.lower()
        return any(keyword in text for keyword in opt_out_keywords)
    
    async def _ensure_feed_diversity(self, posts: List[Post], limit: int) -> List[Post]:
        """
        Ensure feed diversity to avoid overwhelming user with similar content.
        Balances care with variety.
        """
        if len(posts) <= limit:
            return posts
        
        selected_posts = []
        content_types_seen = set()
        authors_seen = set()
        
        for post in posts:
            # Avoid too many posts from same author
            if post.author in authors_seen and len([p for p in selected_posts if p.author == post.author]) >= 2:
                continue
            
            # Ensure content diversity (this is simplified)
            content_type = await self._classify_content_type(post)
            if content_type in content_types_seen and content_types_seen.count(content_type) >= 3:
                continue
            
            selected_posts.append(post)
            authors_seen.add(post.author)
            content_types_seen.add(content_type)
            
            if len(selected_posts) >= limit:
                break
        
        return selected_posts
    
    async def _classify_content_type(self, post: Post) -> str:
        """Classify the type of caring content"""
        text = post.text.lower()
        
        if any(word in text for word in ['help', 'support', 'advice']):
            return 'support_seeking'
        elif any(word in text for word in ['grateful', 'thank you', 'appreciation']):
            return 'gratitude'
        elif any(word in text for word in ['progress', 'better', 'healing', 'recovery']):
            return 'progress_sharing'
        elif any(word in text for word in ['resource', 'tip', 'helpful', 'try this']):
            return 'resource_sharing'
        else:
            return 'general_support'
    
    async def _log_feed_generation(self, user_id: str, post_count: int, 
                                 user_context: Dict[str, Any]) -> None:
        """Log feed generation for metrics and improvement"""
        self.logger.info(f"Generated feed for user {user_id}: {post_count} posts")
        # In production, this would log to metrics system


class FeedGeneratorManager:
    """
    Manages multiple feed generators and handles feed selection logic.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.generators = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def register_generator(self, feed_type: FeedType, generator: BaseFeedGenerator):
        """Register a feed generator for a specific feed type"""
        self.generators[feed_type] = generator
        self.logger.info(f"Registered generator for {feed_type.value}")
    
    async def generate_feed(self, feed_type: FeedType, user_id: str, 
                           cursor: Optional[str] = None, limit: int = 50) -> FeedSkeleton:
        """Generate feed using appropriate generator"""
        if feed_type not in self.generators:
            self.logger.error(f"No generator registered for {feed_type.value}")
            return FeedSkeleton(cursor=None, feed=[])
        
        generator = self.generators[feed_type]
        return await generator.generate_feed(user_id, cursor, limit)
    
    async def get_available_feeds(self, user_id: str) -> List[FeedType]:
        """Get list of feeds available to this user"""
        # Could be filtered based on user preferences, permissions, etc.
        return list(self.generators.keys())
    
    async def get_feed_info(self, feed_type: FeedType) -> Dict[str, Any]:
        """Get information about a specific feed type"""
        if feed_type not in self.generators:
            return {}
        
        return {
            'feed_type': feed_type.value,
            'description': self._get_feed_description(feed_type),
            'frequency': 'On-demand',
            'consent_required': True,
            'opt_out_available': True
        }
    
    def _get_feed_description(self, feed_type: FeedType) -> str:
        """Get human-readable description of feed type"""
        descriptions = {
            FeedType.DAILY_GENTLE_REMINDERS: "Gentle daily affirmations about your worth and value",
            FeedType.HEARTS_SEEKING_LIGHT: "Connects you with community members who understand and can help",
            FeedType.GUARDIAN_ENERGY_RISING: "Celebrates healing journeys and community care successes",
            FeedType.COMMUNITY_WISDOM: "Curated insights and resources from community healing experiences"
        }
        return descriptions.get(feed_type, "A caring feed designed to support human wellbeing")