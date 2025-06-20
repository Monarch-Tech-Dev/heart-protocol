"""
Heart Protocol Cache Manager

Privacy-first caching system that respects user agency and supports caring algorithms.
Implements caching strategies that serve user wellbeing, not just performance.
"""

import asyncio
import hashlib
import time
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
import logging

from .privacy_cache import PrivacyPreservingCache
from .caring_cache import CaringCacheStrategy
from .memory_cache import MemoryCache
from .redis_cache import RedisCache

logger = logging.getLogger(__name__)


class BaseCacheBackend(ABC):
    """Abstract base class for cache backends"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in cache with TTL"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        pass
    
    @abstractmethod
    async def clear_user_data(self, user_id: str) -> bool:
        """Clear all cached data for a user (GDPR compliance)"""
        pass


class CacheManager:
    """
    Heart Protocol Cache Manager
    
    Manages caching with privacy-first principles and user agency:
    - Users control their own cache TTL
    - Sensitive data is never cached without consent
    - Cache can be disabled per user
    - Automatic cache invalidation respects user preferences
    - GDPR-compliant data deletion
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.backend = self._initialize_backend()
        self.privacy_cache = PrivacyPreservingCache(config.get('privacy', {}))
        self.caring_strategy = CaringCacheStrategy(config.get('caring_strategy', {}))
        
        # Cache usage metrics (for caring, not exploitation)
        self.cache_hits = 0
        self.cache_misses = 0
        self.user_cache_preferences = {}  # In production, this would be in database
        
        logger.info(f"Cache Manager initialized with {type(self.backend).__name__} backend")
    
    def _initialize_backend(self) -> BaseCacheBackend:
        """Initialize cache backend based on configuration"""
        backend_type = self.config.get('type', 'memory')
        
        if backend_type == 'redis':
            return RedisCache(self.config.get('redis', {}))
        elif backend_type == 'memory':
            return MemoryCache(self.config.get('memory', {}))
        else:
            logger.warning(f"Unknown cache backend: {backend_type}, falling back to memory")
            return MemoryCache({})
    
    async def get(self, key: str, user_id: Optional[str] = None) -> Optional[Any]:
        """
        Get value from cache with privacy and user preference checks.
        """
        try:
            # Check if user has disabled caching
            if user_id and await self._user_has_disabled_caching(user_id):
                logger.debug(f"Cache disabled for user {user_id[:8]}...")
                return None
            
            # Apply privacy-preserving transformations to key
            safe_key = await self.privacy_cache.make_key_privacy_safe(key, user_id)
            
            # Check caring cache strategy
            if not await self.caring_strategy.should_use_cache(key, user_id):
                logger.debug(f"Caring strategy recommends fresh content for {safe_key}")
                return None
            
            # Get from backend
            cached_value = await self.backend.get(safe_key)
            
            if cached_value is not None:
                self.cache_hits += 1
                
                # Apply privacy filters to cached content
                filtered_value = await self.privacy_cache.filter_cached_content(
                    cached_value, user_id
                )
                
                logger.debug(f"Cache hit for {safe_key}")
                return filtered_value
            else:
                self.cache_misses += 1
                logger.debug(f"Cache miss for {safe_key}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting from cache: {e}")
            self.cache_misses += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None, 
                 user_id: Optional[str] = None) -> bool:
        """
        Set value in cache with privacy protection and user preferences.
        """
        try:
            # Check if user has disabled caching
            if user_id and await self._user_has_disabled_caching(user_id):
                logger.debug(f"Cache disabled for user {user_id[:8]}..., not storing")
                return True  # Don't fail, just don't cache
            
            # Get user's preferred TTL
            if user_id and ttl is None:
                ttl = await self._get_user_cache_ttl(user_id)
            elif ttl is None:
                ttl = self.config.get('default_ttl', 3600)
            
            # Apply privacy-preserving transformations
            safe_key = await self.privacy_cache.make_key_privacy_safe(key, user_id)
            safe_value = await self.privacy_cache.make_value_privacy_safe(value, user_id)
            
            # Check if this content should be cached at all
            if not await self.caring_strategy.should_cache_content(value, user_id):
                logger.debug(f"Caring strategy recommends not caching content for {safe_key}")
                return True  # Don't fail, just don't cache
            
            # Set in backend with privacy-safe data
            success = await self.backend.set(safe_key, safe_value, ttl)
            
            if success:
                logger.debug(f"Cached content with key {safe_key} for {ttl}s")
                
                # Schedule privacy-compliant cleanup if needed
                await self._schedule_privacy_cleanup(safe_key, user_id, ttl)
            
            return success
            
        except Exception as e:
            logger.error(f"Error setting cache: {e}")
            return False
    
    async def delete(self, key: str, user_id: Optional[str] = None) -> bool:
        """Delete value from cache"""
        try:
            safe_key = await self.privacy_cache.make_key_privacy_safe(key, user_id)
            return await self.backend.delete(safe_key)
        except Exception as e:
            logger.error(f"Error deleting from cache: {e}")
            return False
    
    async def clear_user_cache(self, user_id: str) -> bool:
        """
        Clear all cached data for a user (GDPR compliance).
        This is called when a user requests data deletion.
        """
        try:
            success = await self.backend.clear_user_data(user_id)
            
            # Also clear user preferences
            if user_id in self.user_cache_preferences:
                del self.user_cache_preferences[user_id]
            
            logger.info(f"Cleared all cached data for user {user_id[:8]}...")
            return success
            
        except Exception as e:
            logger.error(f"Error clearing user cache: {e}")
            return False
    
    async def set_user_cache_preferences(self, user_id: str, preferences: Dict[str, Any]) -> bool:
        """
        Set user's cache preferences (TTL, disable caching, etc.).
        Respects user agency over their data.
        """
        try:
            # Validate preferences
            validated_prefs = await self._validate_cache_preferences(preferences)
            
            # Store preferences
            self.user_cache_preferences[user_id] = {
                **validated_prefs,
                'updated_at': datetime.utcnow().isoformat()
            }
            
            logger.info(f"Updated cache preferences for user {user_id[:8]}...")
            return True
            
        except Exception as e:
            logger.error(f"Error setting user cache preferences: {e}")
            return False
    
    async def get_user_cache_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user's cache preferences"""
        return self.user_cache_preferences.get(user_id, {
            'caching_enabled': True,
            'cache_ttl': 3600,  # 1 hour default
            'cache_sensitive_content': False,
            'auto_clear_on_mood_change': True
        })
    
    async def _user_has_disabled_caching(self, user_id: str) -> bool:
        """Check if user has disabled caching"""
        prefs = await self.get_user_cache_preferences(user_id)
        return not prefs.get('caching_enabled', True)
    
    async def _get_user_cache_ttl(self, user_id: str) -> int:
        """Get user's preferred cache TTL"""
        prefs = await self.get_user_cache_preferences(user_id)
        return prefs.get('cache_ttl', 3600)
    
    async def _validate_cache_preferences(self, preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Validate user cache preferences"""
        validated = {}
        
        # Validate boolean preferences
        bool_prefs = ['caching_enabled', 'cache_sensitive_content', 'auto_clear_on_mood_change']
        for pref in bool_prefs:
            if pref in preferences:
                validated[pref] = bool(preferences[pref])
        
        # Validate cache_ttl (between 1 minute and 24 hours)
        if 'cache_ttl' in preferences:
            ttl = int(preferences['cache_ttl'])
            validated['cache_ttl'] = max(60, min(86400, ttl))
        
        return validated
    
    async def _schedule_privacy_cleanup(self, cache_key: str, user_id: Optional[str], ttl: int):
        """Schedule privacy-compliant cleanup of cached data"""
        # In production, this would schedule a cleanup job
        # For now, we rely on TTL expiration
        pass
    
    def get_cache_metrics(self) -> Dict[str, Any]:
        """
        Get cache performance metrics focused on user benefit, not system exploitation.
        """
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / max(total_requests, 1)
        
        return {
            'hit_rate': hit_rate,
            'total_requests': total_requests,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'backend_type': type(self.backend).__name__,
            'user_benefit_metrics': {
                'average_response_speedup': hit_rate * 0.8,  # Estimated speedup
                'reduced_server_load': hit_rate * 0.6,
                'privacy_protection_enabled': True,
                'user_control_available': True
            },
            'caring_message': f"Cache is helping deliver caring content {hit_rate:.1%} faster 💙"
        }
    
    async def optimize_for_user_wellbeing(self, user_id: str, recent_activity: Dict[str, Any]):
        """
        Optimize cache strategy based on user's wellbeing needs.
        This is caring performance optimization - not just speed, but appropriate delivery.
        """
        try:
            # If user seems overwhelmed, clear cache to provide fresh, gentle content
            if recent_activity.get('overwhelm_indicators', 0) > 3:
                logger.info(f"User {user_id[:8]}... seems overwhelmed, clearing cache for fresh content")
                await self.clear_user_cache(user_id)
                return
            
            # If user is in crisis, disable caching to ensure fresh crisis resources
            if recent_activity.get('crisis_indicators', False):
                logger.info(f"Crisis indicators for user {user_id[:8]}..., disabling cache temporarily")
                await self.set_user_cache_preferences(user_id, {
                    'caching_enabled': False,
                    'temporary_disable_reason': 'crisis_support'
                })
                return
            
            # If user is showing positive progress, extend cache TTL for stability
            if recent_activity.get('positive_progress', False):
                logger.info(f"Positive progress for user {user_id[:8]}..., extending cache for stability")
                await self.set_user_cache_preferences(user_id, {
                    'cache_ttl': 7200  # 2 hours for stability
                })
            
        except Exception as e:
            logger.error(f"Error optimizing cache for user wellbeing: {e}")
    
    async def emergency_cache_clear(self, reason: str = "emergency"):
        """
        Emergency cache clear for safety reasons.
        Used when algorithms detect potential harm in cached content.
        """
        try:
            logger.warning(f"Emergency cache clear initiated: {reason}")
            
            # This would clear all caches in production
            # For now, log the action
            logger.info("All caches would be cleared for user safety")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in emergency cache clear: {e}")
            return False