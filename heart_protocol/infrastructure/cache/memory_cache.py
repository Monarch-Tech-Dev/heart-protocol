"""
Memory Cache Backend

In-memory cache implementation for Heart Protocol with privacy-first design.
Suitable for development and small deployments.
"""

import time
import threading
from typing import Dict, Any, Optional
from datetime import datetime
import logging

from .manager import BaseCacheBackend

logger = logging.getLogger(__name__)


class CacheEntry:
    """Represents a cache entry with TTL and privacy metadata"""
    
    def __init__(self, value: Any, ttl: int, user_id: Optional[str] = None):
        self.value = value
        self.created_at = time.time()
        self.expires_at = self.created_at + ttl
        self.user_id = user_id
        self.access_count = 0
        self.last_accessed = self.created_at
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired"""
        return time.time() > self.expires_at
    
    def access(self) -> Any:
        """Access the cached value, updating access statistics"""
        self.access_count += 1
        self.last_accessed = time.time()
        return self.value
    
    def get_age_seconds(self) -> float:
        """Get age of cache entry in seconds"""
        return time.time() - self.created_at


class MemoryCache(BaseCacheBackend):
    """
    In-memory cache backend with privacy protection and caring features.
    
    Features:
    - Automatic TTL expiration
    - Privacy-safe user data handling
    - Memory usage monitoring
    - Graceful degradation under memory pressure
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cache = {}  # key -> CacheEntry
        self.lock = threading.RLock()
        self.max_size = config.get('max_size', 10000)
        self.cleanup_interval = config.get('cleanup_interval', 300)  # 5 minutes
        
        # Statistics for caring monitoring
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'expired_removals': 0,
            'user_data_deletions': 0
        }
        
        # Start cleanup thread
        self._start_cleanup_thread()
        
        logger.info(f"Memory cache initialized with max_size={self.max_size}")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from memory cache"""
        try:
            with self.lock:
                if key not in self.cache:
                    self.stats['misses'] += 1
                    return None
                
                entry = self.cache[key]
                
                # Check if expired
                if entry.is_expired():
                    del self.cache[key]
                    self.stats['expired_removals'] += 1
                    self.stats['misses'] += 1
                    return None
                
                # Return value and update access stats
                self.stats['hits'] += 1
                return entry.access()
                
        except Exception as e:
            logger.error(f"Error getting from memory cache: {e}")
            self.stats['misses'] += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in memory cache with TTL"""
        try:
            with self.lock:
                # Check if we need to make space
                if len(self.cache) >= self.max_size:
                    await self._evict_entries()
                
                # Create cache entry
                entry = CacheEntry(value, ttl)
                self.cache[key] = entry
                
                logger.debug(f"Cached entry {key} with TTL {ttl}s")
                return True
                
        except Exception as e:
            logger.error(f"Error setting memory cache: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from memory cache"""
        try:
            with self.lock:
                if key in self.cache:
                    del self.cache[key]
                    logger.debug(f"Deleted cache entry {key}")
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Error deleting from memory cache: {e}")
            return False
    
    async def clear_user_data(self, user_id: str) -> bool:
        """
        Clear all cached data for a user (GDPR compliance).
        For memory cache, we need to check all entries.
        """
        try:
            with self.lock:
                keys_to_delete = []
                
                for key, entry in self.cache.items():
                    # Check if this entry is associated with the user
                    # In memory cache, we identify user data by key prefix
                    user_hash = self._hash_user_id(user_id)
                    if key.startswith(f"hp:{user_hash}:"):
                        keys_to_delete.append(key)
                
                # Delete identified entries
                for key in keys_to_delete:
                    del self.cache[key]
                    self.stats['user_data_deletions'] += 1
                
                logger.info(f"Cleared {len(keys_to_delete)} cache entries for user {user_id[:8]}...")
                return True
                
        except Exception as e:
            logger.error(f"Error clearing user data from memory cache: {e}")
            return False
    
    async def _evict_entries(self):
        """
        Evict cache entries when approaching memory limits.
        Uses caring eviction strategy - prioritize keeping helpful content.
        """
        try:
            # Calculate how many entries to evict (10% of max size)
            entries_to_evict = max(1, self.max_size // 10)
            
            # Create list of (key, entry) tuples with eviction scores
            scored_entries = []
            
            for key, entry in self.cache.items():
                # Calculate eviction score (higher = more likely to evict)
                score = self._calculate_eviction_score(key, entry)
                scored_entries.append((score, key))
            
            # Sort by eviction score (highest first)
            scored_entries.sort(reverse=True)
            
            # Evict the highest-scored entries
            for i in range(min(entries_to_evict, len(scored_entries))):
                _, key_to_evict = scored_entries[i]
                del self.cache[key_to_evict]
                self.stats['evictions'] += 1
            
            logger.debug(f"Evicted {entries_to_evict} cache entries to make space")
            
        except Exception as e:
            logger.error(f"Error during cache eviction: {e}")
    
    def _calculate_eviction_score(self, key: str, entry: CacheEntry) -> float:
        """
        Calculate eviction score for cache entry.
        Higher score = more likely to be evicted.
        
        Caring eviction prioritizes:
        - Keep crisis resources (never evict)
        - Keep recently accessed helpful content
        - Keep gentle reminders and positive content
        - Evict old, unused content first
        """
        score = 0.0
        
        # Never evict crisis-related content
        if any(crisis_word in key.lower() for crisis_word in ['crisis', 'emergency', 'hotline']):
            return -1000.0  # Very low score = never evict
        
        # Age factor (older = higher eviction score)
        age_hours = entry.get_age_seconds() / 3600
        score += age_hours * 10
        
        # Access frequency factor (less accessed = higher eviction score)
        if entry.access_count == 0:
            score += 100  # Never accessed
        else:
            score += 50 / entry.access_count  # Inversely proportional to access count
        
        # Time since last access
        time_since_access = time.time() - entry.last_accessed
        score += time_since_access / 3600 * 5  # Hours since last access
        
        # Prefer to keep caring content (gentle reminders, positive content)
        caring_keywords = ['gentle', 'reminder', 'hope', 'love', 'care', 'support']
        if any(word in key.lower() for word in caring_keywords):
            score -= 50  # Lower eviction score for caring content
        
        return score
    
    def _start_cleanup_thread(self):
        """Start background thread for periodic cleanup"""
        def cleanup_worker():
            while True:
                try:
                    time.sleep(self.cleanup_interval)
                    self._cleanup_expired_entries()
                except Exception as e:
                    logger.error(f"Error in cache cleanup thread: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
        logger.info("Started cache cleanup thread")
    
    def _cleanup_expired_entries(self):
        """Remove expired entries from cache"""
        try:
            with self.lock:
                expired_keys = []
                
                for key, entry in self.cache.items():
                    if entry.is_expired():
                        expired_keys.append(key)
                
                for key in expired_keys:
                    del self.cache[key]
                    self.stats['expired_removals'] += 1
                
                if expired_keys:
                    logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
                    
        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")
    
    def _hash_user_id(self, user_id: str) -> str:
        """Hash user ID for privacy (simplified version)"""
        import hashlib
        return hashlib.sha256(user_id.encode()).hexdigest()[:16]
    
    def get_memory_usage_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        with self.lock:
            total_entries = len(self.cache)
            
            # Calculate memory usage (simplified estimation)
            estimated_memory = sum(
                len(str(key)) + len(str(entry.value)) + 100  # 100 bytes overhead per entry
                for key, entry in self.cache.items()
            )
            
            return {
                'total_entries': total_entries,
                'max_size': self.max_size,
                'utilization': total_entries / self.max_size,
                'estimated_memory_bytes': estimated_memory,
                'estimated_memory_mb': estimated_memory / (1024 * 1024),
                'stats': self.stats.copy(),
                'hit_rate': self.stats['hits'] / max(self.stats['hits'] + self.stats['misses'], 1),
                'caring_message': f"Memory cache serving {total_entries} caring entries 💙"
            }
    
    async def emergency_clear(self, reason: str = "emergency"):
        """Emergency cache clear for safety reasons"""
        try:
            with self.lock:
                cleared_count = len(self.cache)
                self.cache.clear()
                
                logger.warning(f"Emergency cache clear: {reason}, cleared {cleared_count} entries")
                return True
                
        except Exception as e:
            logger.error(f"Error in emergency cache clear: {e}")
            return False