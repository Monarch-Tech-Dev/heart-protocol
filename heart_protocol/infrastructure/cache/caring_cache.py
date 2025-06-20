"""
Caring Cache Strategy

Implements caching strategies that prioritize user wellbeing over pure performance.
Based on the Open Source Love License principle that technology should serve human flourishing.
"""

import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class CaringCacheDecision(Enum):
    """Types of caring cache decisions"""
    CACHE_NORMALLY = "cache_normally"
    CACHE_BRIEFLY = "cache_briefly"  # Short TTL for dynamic content
    NO_CACHE_FRESH_NEEDED = "no_cache_fresh_needed"  # User needs fresh content
    NO_CACHE_SENSITIVE = "no_cache_sensitive"  # Content too sensitive
    NO_CACHE_CRISIS = "no_cache_crisis"  # Crisis situation needs fresh resources


class CaringCacheStrategy:
    """
    Caching strategy that prioritizes user wellbeing over performance metrics.
    
    Key principles:
    - Fresh content for users in distress
    - No caching of crisis-related content
    - Adaptive TTL based on user emotional state
    - Prioritize user agency over system efficiency
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Caring cache settings
        self.default_ttl = config.get('default_ttl', 3600)  # 1 hour
        self.crisis_ttl = config.get('crisis_ttl', 0)  # No caching during crisis
        self.gentle_ttl = config.get('gentle_ttl', 1800)  # 30 mins for gentle content
        self.stable_ttl = config.get('stable_ttl', 7200)  # 2 hours for stable users
        
        # User state tracking for caring decisions
        self.user_states = {}  # In production, this would be in database
    
    async def should_use_cache(self, cache_key: str, user_id: Optional[str] = None) -> bool:
        """
        Determine if we should use cached content based on caring principles.
        """
        try:
            if not user_id:
                return True  # Default to caching for anonymous users
            
            user_state = await self._get_user_emotional_state(user_id)
            
            # Never use cache for users in crisis
            if user_state.get('in_crisis', False):
                logger.info(f"User {user_id[:8]}... in crisis, serving fresh content")
                return False
            
            # Don't use cache if user seems overwhelmed
            if user_state.get('overwhelmed', False):
                logger.info(f"User {user_id[:8]}... seems overwhelmed, serving fresh content")
                return False
            
            # Check if this is crisis-related content
            if await self._is_crisis_related_content(cache_key):
                logger.info(f"Crisis-related content detected, serving fresh: {cache_key}")
                return False
            
            # Check if user has requested fresh content recently
            if await self._user_requested_fresh_content(user_id):
                logger.info(f"User {user_id[:8]}... requested fresh content")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error determining cache usage: {e}")
            return True  # Fail safe - allow caching
    
    async def should_cache_content(self, content: Any, user_id: Optional[str] = None) -> bool:
        """
        Determine if content should be cached based on caring principles.
        """
        try:
            # Analyze content for sensitivity
            content_analysis = await self._analyze_content_for_caching(content)
            
            # Never cache crisis intervention content
            if content_analysis.get('is_crisis_content', False):
                logger.debug("Crisis content detected, not caching")
                return False
            
            # Don't cache highly personal content
            if content_analysis.get('is_highly_personal', False):
                logger.debug("Highly personal content detected, not caching")
                return False
            
            # Don't cache if content contains trigger warnings
            if content_analysis.get('has_trigger_warnings', False):
                logger.debug("Trigger warning content detected, not caching")
                return False
            
            # Check user-specific factors
            if user_id:
                user_state = await self._get_user_emotional_state(user_id)
                
                # Don't cache content for users who need dynamic support
                if user_state.get('needs_dynamic_support', False):
                    logger.debug(f"User {user_id[:8]}... needs dynamic support, not caching")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error determining if content should be cached: {e}")
            return True  # Fail safe - allow caching
    
    async def get_caring_ttl(self, content: Any, user_id: Optional[str] = None) -> int:
        """
        Get TTL based on caring principles rather than just performance.
        """
        try:
            if not user_id:
                return self.default_ttl
            
            user_state = await self._get_user_emotional_state(user_id)
            content_analysis = await self._analyze_content_for_caching(content)
            
            # Crisis or distress - very short or no caching
            if user_state.get('in_crisis', False) or user_state.get('in_distress', False):
                return self.crisis_ttl
            
            # User seems overwhelmed - shorter caching for more responsive content
            if user_state.get('overwhelmed', False):
                return self.gentle_ttl
            
            # Gentle content gets shorter TTL to allow for mood changes
            if content_analysis.get('is_gentle_content', False):
                return self.gentle_ttl
            
            # Stable users with positive content can have longer caching
            if user_state.get('stable', False) and content_analysis.get('is_positive_content', False):
                return self.stable_ttl
            
            # Educational/resource content can be cached longer
            if content_analysis.get('is_educational', False):
                return self.stable_ttl
            
            return self.default_ttl
            
        except Exception as e:
            logger.error(f"Error calculating caring TTL: {e}")
            return self.default_ttl
    
    async def _get_user_emotional_state(self, user_id: str) -> Dict[str, Any]:
        """
        Get user's current emotional state for caring cache decisions.
        """
        # In production, this would integrate with:
        # - Care detection system
        # - User's recent activity patterns
        # - Explicit user mood indicators
        # - Recent crisis interventions
        
        # For now, return default state
        return self.user_states.get(user_id, {
            'in_crisis': False,
            'in_distress': False,
            'overwhelmed': False,
            'stable': True,
            'needs_dynamic_support': False,
            'last_updated': datetime.utcnow().isoformat()
        })
    
    async def update_user_emotional_state(self, user_id: str, state_updates: Dict[str, Any]):
        """
        Update user's emotional state for caring cache decisions.
        Called by care detection system.
        """
        try:
            current_state = await self._get_user_emotional_state(user_id)
            
            # Update state
            current_state.update(state_updates)
            current_state['last_updated'] = datetime.utcnow().isoformat()
            
            self.user_states[user_id] = current_state
            
            logger.info(f"Updated emotional state for user {user_id[:8]}...")
            
            # If user is in crisis, trigger cache clearing
            if state_updates.get('in_crisis', False):
                await self._handle_crisis_cache_clearing(user_id)
            
        except Exception as e:
            logger.error(f"Error updating user emotional state: {e}")
    
    async def _analyze_content_for_caching(self, content: Any) -> Dict[str, bool]:
        """
        Analyze content to determine caching appropriateness.
        """
        analysis = {
            'is_crisis_content': False,
            'is_highly_personal': False,
            'has_trigger_warnings': False,
            'is_gentle_content': False,
            'is_positive_content': False,
            'is_educational': False
        }
        
        try:
            # Convert content to string for analysis
            if isinstance(content, dict):
                content_str = str(content).lower()
            elif isinstance(content, str):
                content_str = content.lower()
            else:
                content_str = str(content).lower()
            
            # Check for crisis indicators
            crisis_keywords = ['suicide', 'crisis', 'emergency', 'hotline', 'immediate help']
            analysis['is_crisis_content'] = any(keyword in content_str for keyword in crisis_keywords)
            
            # Check for trigger warnings
            trigger_keywords = ['trigger warning', 'tw:', 'content warning', 'cw:']
            analysis['has_trigger_warnings'] = any(keyword in content_str for keyword in trigger_keywords)
            
            # Check for highly personal content
            personal_keywords = ['my therapist', 'my medication', 'my diagnosis', 'personal story']
            analysis['is_highly_personal'] = any(keyword in content_str for keyword in personal_keywords)
            
            # Check for gentle content
            gentle_keywords = ['gentle reminder', 'you matter', 'worthy of love', 'take care', 'be kind']
            analysis['is_gentle_content'] = any(keyword in content_str for keyword in gentle_keywords)
            
            # Check for positive content
            positive_keywords = ['progress', 'healing', 'hope', 'grateful', 'celebration', 'achievement']
            analysis['is_positive_content'] = any(keyword in content_str for keyword in positive_keywords)
            
            # Check for educational content
            educational_keywords = ['resource', 'tip', 'guide', 'learn', 'information', 'research']
            analysis['is_educational'] = any(keyword in content_str for keyword in educational_keywords)
            
        except Exception as e:
            logger.error(f"Error analyzing content for caching: {e}")
        
        return analysis
    
    async def _is_crisis_related_content(self, cache_key: str) -> bool:
        """Check if cache key indicates crisis-related content"""
        crisis_indicators = ['crisis', 'emergency', 'hotline', 'suicide', 'immediate']
        return any(indicator in cache_key.lower() for indicator in crisis_indicators)
    
    async def _user_requested_fresh_content(self, user_id: str) -> bool:
        """Check if user has recently requested fresh content"""
        # In production, this would check user's recent requests
        # For now, return False
        return False
    
    async def _handle_crisis_cache_clearing(self, user_id: str):
        """Handle cache clearing when user enters crisis state"""
        try:
            logger.warning(f"User {user_id[:8]}... in crisis, clearing their cache for fresh resources")
            
            # In production, this would:
            # 1. Clear all cached content for this user
            # 2. Ensure fresh crisis resources are immediately available
            # 3. Notify crisis intervention system
            # 4. Log the incident for follow-up care
            
        except Exception as e:
            logger.error(f"Error handling crisis cache clearing: {e}")
    
    def get_caring_cache_metrics(self) -> Dict[str, Any]:
        """Get metrics about caring cache decisions"""
        total_users = len(self.user_states)
        
        crisis_users = sum(1 for state in self.user_states.values() 
                          if state.get('in_crisis', False))
        
        stable_users = sum(1 for state in self.user_states.values() 
                          if state.get('stable', False))
        
        return {
            'total_tracked_users': total_users,
            'users_in_crisis': crisis_users,
            'stable_users': stable_users,
            'crisis_cache_disabled_users': crisis_users,  # Crisis users don't get cached content
            'caring_principles_active': True,
            'wellbeing_prioritized_over_performance': True,
            'cache_settings': {
                'default_ttl': self.default_ttl,
                'crisis_ttl': self.crisis_ttl,
                'gentle_ttl': self.gentle_ttl,
                'stable_ttl': self.stable_ttl
            },
            'caring_message': 'Cache strategy prioritizes your wellbeing over speed 💙'
        }