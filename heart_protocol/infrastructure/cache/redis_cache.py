"""
Redis Cache Backend

Redis-based cache implementation for Heart Protocol with privacy-first design.
Suitable for production deployments with high availability requirements.
"""

import json
import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import hashlib

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available, falling back to memory cache")

from .manager import BaseCacheBackend

logger = logging.getLogger(__name__)


class RedisCache(BaseCacheBackend):
    """
    Redis cache backend with privacy protection and caring features.
    
    Features:
    - Async Redis operations
    - Automatic TTL management
    - Privacy-safe key generation
    - User data isolation
    - GDPR-compliant data deletion
    - Caring eviction policies
    """
    
    def __init__(self, config: Dict[str, Any]):
        if not REDIS_AVAILABLE:
            raise ImportError("Redis is required for RedisCache backend")
        
        self.config = config
        self.redis_client = None
        self.key_prefix = config.get('key_prefix', 'heart_protocol:')
        
        # Connection settings
        self.host = config.get('host', 'localhost')
        self.port = config.get('port', 6379)
        self.db = config.get('db', 0)
        self.password = config.get('password')
        self.ssl = config.get('ssl', False)
        
        # Caring cache settings
        self.max_memory_policy = config.get('max_memory_policy', 'allkeys-lfu')  # Least Frequently Used
        self.user_data_ttl = config.get('user_data_ttl', 3600)  # 1 hour default
        
        # Initialize connection
        asyncio.create_task(self._initialize_connection())
        
        logger.info(f"Redis cache configured for {self.host}:{self.port}")
    
    async def _initialize_connection(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                ssl=self.ssl,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True
            )
            
            # Test connection
            await self.redis_client.ping()
            
            # Configure Redis for caring cache behavior
            await self._configure_redis_for_caring()
            
            logger.info("Redis connection established")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    async def _configure_redis_for_caring(self):
        """Configure Redis settings for caring cache behavior"""
        try:
            # Set memory policy to prioritize frequently used caring content
            await self.redis_client.config_set('maxmemory-policy', self.max_memory_policy)
            
            # Enable keyspace notifications for TTL events (for caring cleanup)
            await self.redis_client.config_set('notify-keyspace-events', 'Ex')
            
            logger.info("Redis configured for caring cache behavior")
            
        except Exception as e:
            logger.warning(f"Could not configure Redis settings: {e}")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache"""
        try:
            if not self.redis_client:
                await self._initialize_connection()
            
            full_key = f"{self.key_prefix}{key}"
            
            # Get value from Redis
            cached_data = await self.redis_client.get(full_key)
            
            if cached_data is None:
                return None
            
            # Parse JSON data
            parsed_data = json.loads(cached_data)
            
            # Update access statistics for caring metrics
            await self._update_access_stats(full_key)
            
            logger.debug(f"Redis cache hit for {key}")
            return parsed_data
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for key {key}: {e}")
            # Remove corrupted data
            await self.delete(key)
            return None
        except Exception as e:
            logger.error(f"Error getting from Redis cache: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in Redis cache with TTL"""
        try:
            if not self.redis_client:
                await self._initialize_connection()
            
            full_key = f"{self.key_prefix}{key}"
            
            # Serialize value to JSON
            serialized_value = json.dumps(value, default=str)
            
            # Set in Redis with TTL
            success = await self.redis_client.setex(full_key, ttl, serialized_value)
            
            if success:
                # Set metadata for caring cache management
                await self._set_cache_metadata(full_key, ttl)
                logger.debug(f"Redis cache set for {key} with TTL {ttl}s")
            
            return bool(success)
            
        except Exception as e:
            logger.error(f"Error setting Redis cache: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from Redis cache"""
        try:
            if not self.redis_client:
                await self._initialize_connection()
            
            full_key = f"{self.key_prefix}{key}"
            
            # Delete main key and metadata
            pipeline = self.redis_client.pipeline()
            pipeline.delete(full_key)
            pipeline.delete(f"{full_key}:meta")
            
            results = await pipeline.execute()
            
            deleted = any(results)
            if deleted:
                logger.debug(f"Deleted Redis cache entry {key}")
            
            return deleted
            
        except Exception as e:
            logger.error(f"Error deleting from Redis cache: {e}")
            return False
    
    async def clear_user_data(self, user_id: str) -> bool:
        """
        Clear all cached data for a user (GDPR compliance).
        Uses Redis SCAN to find all user-related keys.
        """
        try:
            if not self.redis_client:
                await self._initialize_connection()
            
            # Create user-specific key pattern
            user_hash = self._hash_user_id(user_id)
            pattern = f"{self.key_prefix}hp:{user_hash}:*"
            
            deleted_count = 0
            
            # Use SCAN to find all matching keys (safer than KEYS for large datasets)
            async for key in self.redis_client.scan_iter(match=pattern, count=100):
                await self.redis_client.delete(key)
                # Also delete metadata
                await self.redis_client.delete(f"{key}:meta")
                deleted_count += 1
            
            logger.info(f"Cleared {deleted_count} Redis cache entries for user {user_id[:8]}...")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing user data from Redis: {e}")
            return False
    
    async def _set_cache_metadata(self, key: str, ttl: int):
        """Set metadata for cache entry for caring management"""
        try:
            metadata = {
                'created_at': datetime.utcnow().isoformat(),
                'ttl': ttl,
                'access_count': 0,
                'last_accessed': datetime.utcnow().isoformat()
            }
            
            meta_key = f"{key}:meta"
            await self.redis_client.setex(
                meta_key, 
                ttl + 300,  # Metadata lives slightly longer than data
                json.dumps(metadata)
            )
            
        except Exception as e:
            logger.error(f"Error setting cache metadata: {e}")
    
    async def _update_access_stats(self, key: str):
        """Update access statistics for caring metrics"""
        try:
            meta_key = f"{key}:meta"
            
            # Get current metadata
            meta_data = await self.redis_client.get(meta_key)
            if meta_data:
                metadata = json.loads(meta_data)
                metadata['access_count'] = metadata.get('access_count', 0) + 1
                metadata['last_accessed'] = datetime.utcnow().isoformat()
                
                # Update metadata (preserve TTL)
                ttl = await self.redis_client.ttl(meta_key)
                if ttl > 0:
                    await self.redis_client.setex(meta_key, ttl, json.dumps(metadata))
            
        except Exception as e:
            logger.error(f"Error updating access stats: {e}")
    
    def _hash_user_id(self, user_id: str) -> str:
        """Hash user ID for privacy"""
        return hashlib.sha256(user_id.encode()).hexdigest()[:16]
    
    async def get_redis_info(self) -> Dict[str, Any]:
        """Get Redis server information and caring cache metrics"""
        try:
            if not self.redis_client:
                await self._initialize_connection()
            
            # Get Redis server info
            redis_info = await self.redis_client.info()
            
            # Get Heart Protocol specific metrics
            hp_keys = 0
            async for key in self.redis_client.scan_iter(match=f"{self.key_prefix}*", count=100):
                hp_keys += 1
            
            # Calculate caring metrics
            used_memory = redis_info.get('used_memory', 0)
            max_memory = redis_info.get('maxmemory', 0)
            
            return {
                'redis_version': redis_info.get('redis_version', 'unknown'),
                'connected_clients': redis_info.get('connected_clients', 0),
                'used_memory_human': redis_info.get('used_memory_human', '0B'),
                'used_memory_peak_human': redis_info.get('used_memory_peak_human', '0B'),
                'memory_fragmentation_ratio': redis_info.get('mem_fragmentation_ratio', 0),
                'heart_protocol_keys': hp_keys,
                'max_memory_policy': redis_info.get('maxmemory_policy', 'unknown'),
                'keyspace_hits': redis_info.get('keyspace_hits', 0),
                'keyspace_misses': redis_info.get('keyspace_misses', 0),
                'hit_rate': self._calculate_hit_rate(redis_info),
                'caring_status': {
                    'memory_usage_healthy': (used_memory / max_memory) < 0.8 if max_memory > 0 else True,
                    'fragmentation_healthy': redis_info.get('mem_fragmentation_ratio', 1) < 1.5,
                    'caring_eviction_policy': redis_info.get('maxmemory_policy') in ['allkeys-lfu', 'volatile-lfu'],
                    'privacy_protection_active': True
                },
                'caring_message': f"Redis serving {hp_keys} caring cache entries 💙"
            }
            
        except Exception as e:
            logger.error(f"Error getting Redis info: {e}")
            return {'error': str(e)}
    
    def _calculate_hit_rate(self, redis_info: Dict[str, Any]) -> float:
        """Calculate cache hit rate"""
        hits = redis_info.get('keyspace_hits', 0)
        misses = redis_info.get('keyspace_misses', 0)
        total = hits + misses
        
        if total == 0:
            return 0.0
        
        return hits / total
    
    async def emergency_clear(self, reason: str = "emergency") -> bool:
        """Emergency cache clear for safety reasons"""
        try:
            if not self.redis_client:
                await self._initialize_connection()
            
            # Clear all Heart Protocol keys
            deleted_count = 0
            async for key in self.redis_client.scan_iter(match=f"{self.key_prefix}*", count=100):
                await self.redis_client.delete(key)
                deleted_count += 1
            
            logger.warning(f"Emergency Redis cache clear: {reason}, cleared {deleted_count} entries")
            return True
            
        except Exception as e:
            logger.error(f"Error in emergency Redis cache clear: {e}")
            return False
    
    async def optimize_for_caring_workload(self):
        """Optimize Redis configuration for caring workload patterns"""
        try:
            if not self.redis_client:
                await self._initialize_connection()
            
            # Configure for caring access patterns
            config_updates = {
                'maxmemory-policy': 'allkeys-lfu',  # Keep frequently accessed caring content
                'timeout': '300',  # 5 minute client timeout for gentle handling
                'tcp-keepalive': '60',  # Keep connections alive for better UX
            }
            
            for key, value in config_updates.items():
                try:
                    await self.redis_client.config_set(key, value)
                    logger.debug(f"Updated Redis config: {key} = {value}")
                except Exception as e:
                    logger.warning(f"Could not update Redis config {key}: {e}")
            
            logger.info("Redis optimized for caring workload")
            
        except Exception as e:
            logger.error(f"Error optimizing Redis for caring workload: {e}")
    
    async def close(self):
        """Close Redis connection gracefully"""
        try:
            if self.redis_client:
                await self.redis_client.close()
                logger.info("Redis connection closed")
        except Exception as e:
            logger.error(f"Error closing Redis connection: {e}")
    
    def __del__(self):
        """Ensure connection is closed on destruction"""
        if self.redis_client:
            try:
                asyncio.create_task(self.close())
            except Exception:
                pass  # Best effort cleanup