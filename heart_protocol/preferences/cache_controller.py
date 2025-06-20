"""
User Cache Controller

Provides users with granular control over data caching, retention, and 
performance optimization while maintaining privacy and autonomy.
"""

import asyncio
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import logging
import hashlib
import json

logger = logging.getLogger(__name__)


class CachePolicy(Enum):
    """Cache policies with different privacy/performance balances"""
    MINIMAL = "minimal"                      # Minimal caching, maximum privacy
    BALANCED = "balanced"                    # Balanced caching and privacy
    PERFORMANCE = "performance"              # Maximum performance, reasonable privacy
    USER_CONTROLLED = "user_controlled"      # User has granular control


class DataRetention(Enum):
    """Data retention policies"""
    SESSION_ONLY = "session_only"           # Data cleared at session end
    SHORT_TERM = "short_term"               # 24-48 hours
    MEDIUM_TERM = "medium_term"             # 1-7 days
    LONG_TERM = "long_term"                 # 1-4 weeks
    EXTENDED = "extended"                   # 1-12 months
    USER_DEFINED = "user_defined"           # User-specified duration


class CacheType(Enum):
    """Types of data that can be cached"""
    USER_PREFERENCES = "user_preferences"   # User preference data
    FEED_CONTENT = "feed_content"           # Generated feed content
    CARE_ASSESSMENTS = "care_assessments"   # Care assessment results
    PERSONALIZATION = "personalization"     # Personalization data
    ANALYTICS = "analytics"                 # Analytics and usage data
    SAFETY_MONITORING = "safety_monitoring" # Safety monitoring data
    CULTURAL_CONTEXT = "cultural_context"   # Cultural adaptation data
    INTERACTION_HISTORY = "interaction_history"  # User interaction patterns
    TEMPORARY_STATE = "temporary_state"     # Temporary UI/session state


class DataSensitivity(Enum):
    """Sensitivity levels for cached data"""
    PUBLIC = "public"                       # No privacy concerns
    PERSONAL = "personal"                   # Personal but not sensitive
    SENSITIVE = "sensitive"                 # Sensitive personal information
    HIGHLY_SENSITIVE = "highly_sensitive"   # Highly sensitive (health, crisis)
    CRITICAL = "critical"                   # Critical security information


@dataclass
class CacheConfiguration:
    """Configuration for a specific cache type"""
    cache_type: CacheType
    data_sensitivity: DataSensitivity
    default_ttl_hours: int
    max_ttl_hours: int
    min_ttl_hours: int
    allow_user_control: bool
    require_consent: bool
    auto_purge_on_privacy_change: bool
    encrypt_at_rest: bool
    description: str
    privacy_implications: str
    performance_benefits: str


@dataclass
class UserCacheSettings:
    """User's cache settings for a specific cache type"""
    user_id: str
    cache_type: CacheType
    enabled: bool
    ttl_hours: int
    last_updated: datetime
    user_consent_given: bool
    auto_purge_enabled: bool
    custom_rules: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheEntry:
    """Individual cache entry"""
    entry_id: str
    user_id: str
    cache_type: CacheType
    data_key: str
    data_value: Any
    created_at: datetime
    expires_at: datetime
    access_count: int
    last_accessed: datetime
    data_sensitivity: DataSensitivity
    encrypted: bool
    user_controlled: bool


class UserCacheController:
    """
    User-controlled cache management system that empowers users with
    granular control over their data caching and retention.
    
    Core Principles:
    - User control and transparency
    - Privacy by design
    - Performance optimization with consent
    - Granular data management
    - Healing-focused caching strategies
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Cache management
        self.cache_configurations = {}        # cache_type -> CacheConfiguration
        self.user_cache_settings = {}         # user_id -> Dict[cache_type, UserCacheSettings]
        self.cache_entries = {}               # entry_id -> CacheEntry
        self.cache_indices = {}               # user_id -> Dict[cache_type, List[entry_ids]]
        
        # Privacy and consent management
        self.consent_records = {}             # user_id -> consent_data
        self.data_retention_policies = {}     # user_id -> retention_policies
        self.purge_schedules = {}             # user_id -> purge_schedule
        
        # Performance tracking
        self.cache_performance_metrics = {}   # cache_type -> performance_data
        self.user_performance_preferences = {} # user_id -> performance_preferences
        
        # Security and encryption
        self.encryption_keys = {}             # user_id -> encryption_keys
        self.access_logs = {}                 # user_id -> access_log_entries
        
        self._initialize_cache_configurations()
        self._initialize_default_policies()
        
        logger.info("User Cache Controller initialized with privacy-first design")
    
    def _initialize_cache_configurations(self) -> None:
        """Initialize cache configurations for different data types"""
        
        self.cache_configurations[CacheType.USER_PREFERENCES] = CacheConfiguration(
            cache_type=CacheType.USER_PREFERENCES,
            data_sensitivity=DataSensitivity.PERSONAL,
            default_ttl_hours=168,  # 1 week
            max_ttl_hours=8760,     # 1 year
            min_ttl_hours=1,        # 1 hour
            allow_user_control=True,
            require_consent=False,
            auto_purge_on_privacy_change=True,
            encrypt_at_rest=False,
            description="User preference and settings data",
            privacy_implications="Preferences reveal personal choices and healing journey",
            performance_benefits="Faster loading of personalized interface and content"
        )
        
        self.cache_configurations[CacheType.FEED_CONTENT] = CacheConfiguration(
            cache_type=CacheType.FEED_CONTENT,
            data_sensitivity=DataSensitivity.PERSONAL,
            default_ttl_hours=24,   # 1 day
            max_ttl_hours=168,      # 1 week
            min_ttl_hours=1,        # 1 hour
            allow_user_control=True,
            require_consent=False,
            auto_purge_on_privacy_change=False,
            encrypt_at_rest=False,
            description="Generated feed content and recommendations",
            privacy_implications="Feed content reveals interests and healing focus areas",
            performance_benefits="Instant feed loading and reduced server processing"
        )
        
        self.cache_configurations[CacheType.CARE_ASSESSMENTS] = CacheConfiguration(
            cache_type=CacheType.CARE_ASSESSMENTS,
            data_sensitivity=DataSensitivity.HIGHLY_SENSITIVE,
            default_ttl_hours=24,   # 1 day
            max_ttl_hours=168,      # 1 week
            min_ttl_hours=1,        # 1 hour
            allow_user_control=True,
            require_consent=True,
            auto_purge_on_privacy_change=True,
            encrypt_at_rest=True,
            description="Care assessment results and mental health data",
            privacy_implications="Contains sensitive mental health and crisis information",
            performance_benefits="Faster care level determination and personalization"
        )
        
        self.cache_configurations[CacheType.PERSONALIZATION] = CacheConfiguration(
            cache_type=CacheType.PERSONALIZATION,
            data_sensitivity=DataSensitivity.SENSITIVE,
            default_ttl_hours=72,   # 3 days
            max_ttl_hours=720,      # 30 days
            min_ttl_hours=6,        # 6 hours
            allow_user_control=True,
            require_consent=True,
            auto_purge_on_privacy_change=True,
            encrypt_at_rest=True,
            description="Personalization algorithms and learned preferences",
            privacy_implications="Reveals detailed behavioral patterns and healing journey",
            performance_benefits="Highly personalized content and recommendations"
        )
        
        self.cache_configurations[CacheType.SAFETY_MONITORING] = CacheConfiguration(
            cache_type=CacheType.SAFETY_MONITORING,
            data_sensitivity=DataSensitivity.CRITICAL,
            default_ttl_hours=168,  # 1 week
            max_ttl_hours=720,      # 30 days
            min_ttl_hours=24,       # 1 day
            allow_user_control=False,  # Safety data controlled by system
            require_consent=True,
            auto_purge_on_privacy_change=False,  # Safety data preserved
            encrypt_at_rest=True,
            description="Safety monitoring patterns and crisis indicators",
            privacy_implications="Contains critical safety and crisis information",
            performance_benefits="Rapid safety assessment and crisis intervention"
        )
        
        self.cache_configurations[CacheType.ANALYTICS] = CacheConfiguration(
            cache_type=CacheType.ANALYTICS,
            data_sensitivity=DataSensitivity.PERSONAL,
            default_ttl_hours=168,  # 1 week
            max_ttl_hours=8760,     # 1 year
            min_ttl_hours=1,        # 1 hour
            allow_user_control=True,
            require_consent=True,
            auto_purge_on_privacy_change=True,
            encrypt_at_rest=False,
            description="Usage analytics and system improvement data",
            privacy_implications="Reveals usage patterns and system interaction",
            performance_benefits="System optimization and improved user experience"
        )
        
        self.cache_configurations[CacheType.CULTURAL_CONTEXT] = CacheConfiguration(
            cache_type=CacheType.CULTURAL_CONTEXT,
            data_sensitivity=DataSensitivity.SENSITIVE,
            default_ttl_hours=720,  # 30 days
            max_ttl_hours=8760,     # 1 year
            min_ttl_hours=24,       # 1 day
            allow_user_control=True,
            require_consent=False,
            auto_purge_on_privacy_change=True,
            encrypt_at_rest=False,
            description="Cultural adaptation and localization data",
            privacy_implications="Reveals cultural background and identity",
            performance_benefits="Culturally appropriate content and communication"
        )
        
        self.cache_configurations[CacheType.INTERACTION_HISTORY] = CacheConfiguration(
            cache_type=CacheType.INTERACTION_HISTORY,
            data_sensitivity=DataSensitivity.SENSITIVE,
            default_ttl_hours=168,  # 1 week
            max_ttl_hours=2160,     # 90 days
            min_ttl_hours=24,       # 1 day
            allow_user_control=True,
            require_consent=True,
            auto_purge_on_privacy_change=True,
            encrypt_at_rest=True,
            description="User interaction patterns and communication history",
            privacy_implications="Reveals detailed interaction and communication patterns",
            performance_benefits="Improved conversation context and personalization"
        )
        
        self.cache_configurations[CacheType.TEMPORARY_STATE] = CacheConfiguration(
            cache_type=CacheType.TEMPORARY_STATE,
            data_sensitivity=DataSensitivity.PERSONAL,
            default_ttl_hours=2,    # 2 hours
            max_ttl_hours=24,       # 1 day
            min_ttl_hours=1,        # 1 hour
            allow_user_control=True,
            require_consent=False,
            auto_purge_on_privacy_change=False,
            encrypt_at_rest=False,
            description="Temporary UI state and session data",
            privacy_implications="Contains current session activity",
            performance_benefits="Seamless user interface experience"
        )
    
    def _initialize_default_policies(self) -> None:
        """Initialize default cache policies for different user preferences"""
        
        self.default_policies = {
            CachePolicy.MINIMAL: {
                'ttl_multiplier': 0.5,     # Half the default TTL
                'sensitive_data_enabled': False,
                'analytics_enabled': False,
                'personalization_level': 'basic'
            },
            
            CachePolicy.BALANCED: {
                'ttl_multiplier': 1.0,     # Default TTL
                'sensitive_data_enabled': True,
                'analytics_enabled': True,
                'personalization_level': 'moderate'
            },
            
            CachePolicy.PERFORMANCE: {
                'ttl_multiplier': 1.5,     # 1.5x default TTL
                'sensitive_data_enabled': True,
                'analytics_enabled': True,
                'personalization_level': 'high'
            },
            
            CachePolicy.USER_CONTROLLED: {
                'ttl_multiplier': 1.0,     # User sets individual TTLs
                'sensitive_data_enabled': None,  # User decides per type
                'analytics_enabled': None,      # User decides
                'personalization_level': 'user_defined'
            }
        }
    
    async def initialize_user_cache_settings(self, user_id: str, 
                                           cache_policy: CachePolicy = CachePolicy.BALANCED,
                                           user_preferences: Optional[Dict[str, Any]] = None) -> None:
        """Initialize cache settings for a new user"""
        
        try:
            if user_id not in self.user_cache_settings:
                self.user_cache_settings[user_id] = {}
            
            if user_id not in self.cache_indices:
                self.cache_indices[user_id] = {}
            
            policy_config = self.default_policies[cache_policy]
            
            # Initialize settings for each cache type
            for cache_type, config in self.cache_configurations.items():
                # Calculate TTL based on policy
                if cache_policy == CachePolicy.USER_CONTROLLED:
                    ttl_hours = config.default_ttl_hours
                else:
                    ttl_hours = int(config.default_ttl_hours * policy_config['ttl_multiplier'])
                    ttl_hours = max(config.min_ttl_hours, min(ttl_hours, config.max_ttl_hours))
                
                # Determine if cache type should be enabled
                enabled = True
                if cache_policy == CachePolicy.MINIMAL:
                    if config.data_sensitivity in [DataSensitivity.SENSITIVE, DataSensitivity.HIGHLY_SENSITIVE]:
                        enabled = False
                
                # Apply user preferences if provided
                if user_preferences:
                    cache_pref_key = f"{cache_type.value}_enabled"
                    if cache_pref_key in user_preferences:
                        enabled = user_preferences[cache_pref_key]
                    
                    ttl_pref_key = f"{cache_type.value}_ttl_hours"
                    if ttl_pref_key in user_preferences:
                        requested_ttl = user_preferences[ttl_pref_key]
                        if config.min_ttl_hours <= requested_ttl <= config.max_ttl_hours:
                            ttl_hours = requested_ttl
                
                settings = UserCacheSettings(
                    user_id=user_id,
                    cache_type=cache_type,
                    enabled=enabled,
                    ttl_hours=ttl_hours,
                    last_updated=datetime.utcnow(),
                    user_consent_given=not config.require_consent,  # Will need proper consent flow
                    auto_purge_enabled=config.auto_purge_on_privacy_change
                )
                
                self.user_cache_settings[user_id][cache_type] = settings
                self.cache_indices[user_id][cache_type] = []
            
            logger.info(f"Initialized cache settings for user {user_id[:8]}... with {cache_policy.value} policy")
            
        except Exception as e:
            logger.error(f"Error initializing cache settings for user {user_id[:8]}...: {e}")
    
    async def set_user_cache_setting(self, user_id: str, cache_type: CacheType,
                                   enabled: Optional[bool] = None,
                                   ttl_hours: Optional[int] = None,
                                   custom_rules: Optional[Dict[str, Any]] = None) -> bool:
        """Set user's cache settings for a specific cache type"""
        
        try:
            # Ensure user has cache settings initialized
            if user_id not in self.user_cache_settings:
                await self.initialize_user_cache_settings(user_id)
            
            if cache_type not in self.user_cache_settings[user_id]:
                logger.error(f"Cache type {cache_type.value} not found for user {user_id[:8]}...")
                return False
            
            config = self.cache_configurations[cache_type]
            settings = self.user_cache_settings[user_id][cache_type]
            
            # Check if user is allowed to control this cache type
            if not config.allow_user_control and enabled is not None:
                logger.warning(f"User {user_id[:8]}... attempted to control non-user-controllable cache {cache_type.value}")
                return False
            
            # Validate TTL if provided
            if ttl_hours is not None:
                if not (config.min_ttl_hours <= ttl_hours <= config.max_ttl_hours):
                    logger.error(f"Invalid TTL {ttl_hours} for cache type {cache_type.value}")
                    return False
            
            # Check consent requirements
            if enabled and config.require_consent and not settings.user_consent_given:
                logger.warning(f"Consent required for cache type {cache_type.value} but not given")
                return False
            
            # Update settings
            if enabled is not None:
                old_enabled = settings.enabled
                settings.enabled = enabled
                
                # If disabling cache, purge existing data
                if old_enabled and not enabled:
                    await self._purge_user_cache_type(user_id, cache_type)
            
            if ttl_hours is not None:
                settings.ttl_hours = ttl_hours
                # Update expiration times for existing entries
                await self._update_cache_expiration_times(user_id, cache_type, ttl_hours)
            
            if custom_rules is not None:
                settings.custom_rules.update(custom_rules)
            
            settings.last_updated = datetime.utcnow()
            
            logger.info(f"Updated cache settings for user {user_id[:8]}... cache type {cache_type.value}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting cache settings for user {user_id[:8]}...: {e}")
            return False
    
    async def cache_user_data(self, user_id: str, cache_type: CacheType,
                            data_key: str, data_value: Any,
                            custom_ttl_hours: Optional[int] = None) -> bool:
        """Cache data for a user with their consent and settings"""
        
        try:
            # Check if user has caching enabled for this type
            if not await self._is_cache_enabled(user_id, cache_type):
                return False
            
            settings = self.user_cache_settings[user_id][cache_type]
            config = self.cache_configurations[cache_type]
            
            # Determine TTL
            ttl_hours = custom_ttl_hours if custom_ttl_hours else settings.ttl_hours
            expires_at = datetime.utcnow() + timedelta(hours=ttl_hours)
            
            # Generate entry ID
            entry_data = f"{user_id}_{cache_type.value}_{data_key}_{datetime.utcnow().isoformat()}"
            entry_id = hashlib.sha256(entry_data.encode()).hexdigest()[:32]\n            \n            # Encrypt data if required\n            encrypted = config.encrypt_at_rest\n            if encrypted:\n                data_value = await self._encrypt_data(user_id, data_value)\n            \n            # Create cache entry\n            entry = CacheEntry(\n                entry_id=entry_id,\n                user_id=user_id,\n                cache_type=cache_type,\n                data_key=data_key,\n                data_value=data_value,\n                created_at=datetime.utcnow(),\n                expires_at=expires_at,\n                access_count=0,\n                last_accessed=datetime.utcnow(),\n                data_sensitivity=config.data_sensitivity,\n                encrypted=encrypted,\n                user_controlled=config.allow_user_control\n            )\n            \n            # Store entry\n            self.cache_entries[entry_id] = entry\n            \n            # Update indices\n            if user_id not in self.cache_indices:\n                self.cache_indices[user_id] = {}\n            if cache_type not in self.cache_indices[user_id]:\n                self.cache_indices[user_id][cache_type] = []\n            \n            # Remove old entry with same key if exists\n            await self._remove_duplicate_cache_entries(user_id, cache_type, data_key)\n            \n            self.cache_indices[user_id][cache_type].append(entry_id)\n            \n            # Log access for audit\n            await self._log_cache_access(user_id, 'cache_write', cache_type, data_key)\n            \n            logger.debug(f\"Cached data for user {user_id[:8]}... type {cache_type.value} key {data_key}\")\n            return True\n            \n        except Exception as e:\n            logger.error(f\"Error caching data for user {user_id[:8]}...: {e}\")\n            return False\n    \n    async def get_cached_data(self, user_id: str, cache_type: CacheType, \n                            data_key: str) -> Optional[Any]:\n        \"\"\"Retrieve cached data for a user\"\"\"\n        \n        try:\n            # Check if user has caching enabled\n            if not await self._is_cache_enabled(user_id, cache_type):\n                return None\n            \n            # Find cache entry\n            entry = await self._find_cache_entry(user_id, cache_type, data_key)\n            if not entry:\n                return None\n            \n            # Check if entry has expired\n            if datetime.utcnow() > entry.expires_at:\n                await self._remove_cache_entry(entry.entry_id)\n                return None\n            \n            # Update access tracking\n            entry.access_count += 1\n            entry.last_accessed = datetime.utcnow()\n            \n            # Decrypt data if needed\n            data_value = entry.data_value\n            if entry.encrypted:\n                data_value = await self._decrypt_data(user_id, data_value)\n            \n            # Log access\n            await self._log_cache_access(user_id, 'cache_read', cache_type, data_key)\n            \n            logger.debug(f\"Retrieved cached data for user {user_id[:8]}... type {cache_type.value} key {data_key}\")\n            return data_value\n            \n        except Exception as e:\n            logger.error(f\"Error retrieving cached data for user {user_id[:8]}...: {e}\")\n            return None\n    \n    async def invalidate_user_cache(self, user_id: str, \n                                  cache_type: Optional[CacheType] = None,\n                                  data_key: Optional[str] = None) -> bool:\n        \"\"\"Invalidate user's cached data\"\"\"\n        \n        try:\n            if cache_type is None:\n                # Invalidate all cache types for user\n                if user_id in self.cache_indices:\n                    for ct in list(self.cache_indices[user_id].keys()):\n                        await self._purge_user_cache_type(user_id, ct)\n                return True\n            \n            elif data_key is None:\n                # Invalidate all data for specific cache type\n                await self._purge_user_cache_type(user_id, cache_type)\n                return True\n            \n            else:\n                # Invalidate specific data key\n                entry = await self._find_cache_entry(user_id, cache_type, data_key)\n                if entry:\n                    await self._remove_cache_entry(entry.entry_id)\n                return True\n            \n        except Exception as e:\n            logger.error(f\"Error invalidating cache for user {user_id[:8]}...: {e}\")\n            return False\n    \n    async def export_user_cache_data(self, user_id: str) -> Dict[str, Any]:\n        \"\"\"Export user's cache data for transparency\"\"\"\n        \n        try:\n            export_data = {\n                'user_id': user_id,\n                'exported_at': datetime.utcnow().isoformat(),\n                'cache_settings': {},\n                'cached_data': {},\n                'cache_statistics': {}\n            }\n            \n            # Export cache settings\n            if user_id in self.user_cache_settings:\n                for cache_type, settings in self.user_cache_settings[user_id].items():\n                    export_data['cache_settings'][cache_type.value] = {\n                        'enabled': settings.enabled,\n                        'ttl_hours': settings.ttl_hours,\n                        'last_updated': settings.last_updated.isoformat(),\n                        'user_consent_given': settings.user_consent_given,\n                        'auto_purge_enabled': settings.auto_purge_enabled,\n                        'custom_rules': settings.custom_rules\n                    }\n            \n            # Export cached data (non-sensitive only)\n            if user_id in self.cache_indices:\n                for cache_type, entry_ids in self.cache_indices[user_id].items():\n                    config = self.cache_configurations[cache_type]\n                    \n                    # Only export non-sensitive data for transparency\n                    if config.data_sensitivity in [DataSensitivity.PUBLIC, DataSensitivity.PERSONAL]:\n                        cache_data = []\n                        for entry_id in entry_ids:\n                            if entry_id in self.cache_entries:\n                                entry = self.cache_entries[entry_id]\n                                cache_data.append({\n                                    'data_key': entry.data_key,\n                                    'created_at': entry.created_at.isoformat(),\n                                    'expires_at': entry.expires_at.isoformat(),\n                                    'access_count': entry.access_count,\n                                    'last_accessed': entry.last_accessed.isoformat(),\n                                    'data_sensitivity': entry.data_sensitivity.value,\n                                    'encrypted': entry.encrypted\n                                })\n                        \n                        export_data['cached_data'][cache_type.value] = cache_data\n            \n            # Export cache statistics\n            export_data['cache_statistics'] = await self._get_user_cache_statistics(user_id)\n            \n            return export_data\n            \n        except Exception as e:\n            logger.error(f\"Error exporting cache data for user {user_id[:8]}...: {e}\")\n            return {}\n    \n    async def purge_expired_cache(self) -> Dict[str, int]:\n        \"\"\"Purge expired cache entries system-wide\"\"\"\n        \n        try:\n            purged_counts = {}\n            now = datetime.utcnow()\n            expired_entries = []\n            \n            # Find expired entries\n            for entry_id, entry in self.cache_entries.items():\n                if now > entry.expires_at:\n                    expired_entries.append(entry_id)\n                    cache_type = entry.cache_type.value\n                    purged_counts[cache_type] = purged_counts.get(cache_type, 0) + 1\n            \n            # Remove expired entries\n            for entry_id in expired_entries:\n                await self._remove_cache_entry(entry_id)\n            \n            logger.info(f\"Purged {len(expired_entries)} expired cache entries\")\n            return purged_counts\n            \n        except Exception as e:\n            logger.error(f\"Error purging expired cache: {e}\")\n            return {}\n    \n    async def get_user_cache_control_panel(self, user_id: str) -> Dict[str, Any]:\n        \"\"\"Get user's cache control panel data\"\"\"\n        \n        try:\n            control_panel = {\n                'user_id': user_id,\n                'cache_policies': {\n                    'current_policy': await self._detect_user_cache_policy(user_id),\n                    'available_policies': [\n                        {\n                            'policy': policy.value,\n                            'description': self._get_policy_description(policy),\n                            'privacy_level': self._get_policy_privacy_level(policy),\n                            'performance_level': self._get_policy_performance_level(policy)\n                        }\n                        for policy in CachePolicy\n                    ]\n                },\n                'cache_types': {},\n                'statistics': await self._get_user_cache_statistics(user_id),\n                'recommendations': await self._get_cache_recommendations(user_id)\n            }\n            \n            # Add cache type controls\n            if user_id in self.user_cache_settings:\n                for cache_type, settings in self.user_cache_settings[user_id].items():\n                    config = self.cache_configurations[cache_type]\n                    \n                    control_panel['cache_types'][cache_type.value] = {\n                        'configuration': {\n                            'display_name': cache_type.value.replace('_', ' ').title(),\n                            'description': config.description,\n                            'data_sensitivity': config.data_sensitivity.value,\n                            'privacy_implications': config.privacy_implications,\n                            'performance_benefits': config.performance_benefits,\n                            'allow_user_control': config.allow_user_control,\n                            'require_consent': config.require_consent\n                        },\n                        'current_settings': {\n                            'enabled': settings.enabled,\n                            'ttl_hours': settings.ttl_hours,\n                            'user_consent_given': settings.user_consent_given,\n                            'auto_purge_enabled': settings.auto_purge_enabled,\n                            'last_updated': settings.last_updated.isoformat()\n                        },\n                        'controls': {\n                            'min_ttl_hours': config.min_ttl_hours,\n                            'max_ttl_hours': config.max_ttl_hours,\n                            'default_ttl_hours': config.default_ttl_hours,\n                            'can_disable': config.allow_user_control,\n                            'can_adjust_ttl': config.allow_user_control\n                        }\n                    }\n            \n            return control_panel\n            \n        except Exception as e:\n            logger.error(f\"Error generating cache control panel for user {user_id[:8]}...: {e}\")\n            return {}\n    \n    async def _is_cache_enabled(self, user_id: str, cache_type: CacheType) -> bool:\n        \"\"\"Check if caching is enabled for user and cache type\"\"\"\n        \n        if user_id not in self.user_cache_settings:\n            return False\n        \n        if cache_type not in self.user_cache_settings[user_id]:\n            return False\n        \n        settings = self.user_cache_settings[user_id][cache_type]\n        config = self.cache_configurations[cache_type]\n        \n        # Check if enabled and consent given if required\n        return (settings.enabled and \n                (not config.require_consent or settings.user_consent_given))\n    \n    async def _find_cache_entry(self, user_id: str, cache_type: CacheType, \n                              data_key: str) -> Optional[CacheEntry]:\n        \"\"\"Find specific cache entry\"\"\"\n        \n        if (user_id not in self.cache_indices or \n            cache_type not in self.cache_indices[user_id]):\n            return None\n        \n        for entry_id in self.cache_indices[user_id][cache_type]:\n            if entry_id in self.cache_entries:\n                entry = self.cache_entries[entry_id]\n                if entry.data_key == data_key:\n                    return entry\n        \n        return None\n    \n    async def _remove_cache_entry(self, entry_id: str) -> None:\n        \"\"\"Remove cache entry and update indices\"\"\"\n        \n        if entry_id not in self.cache_entries:\n            return\n        \n        entry = self.cache_entries[entry_id]\n        \n        # Remove from indices\n        if (entry.user_id in self.cache_indices and \n            entry.cache_type in self.cache_indices[entry.user_id]):\n            if entry_id in self.cache_indices[entry.user_id][entry.cache_type]:\n                self.cache_indices[entry.user_id][entry.cache_type].remove(entry_id)\n        \n        # Remove entry\n        del self.cache_entries[entry_id]\n    \n    async def _remove_duplicate_cache_entries(self, user_id: str, cache_type: CacheType, \n                                            data_key: str) -> None:\n        \"\"\"Remove duplicate cache entries for the same data key\"\"\"\n        \n        existing_entry = await self._find_cache_entry(user_id, cache_type, data_key)\n        if existing_entry:\n            await self._remove_cache_entry(existing_entry.entry_id)\n    \n    async def _purge_user_cache_type(self, user_id: str, cache_type: CacheType) -> None:\n        \"\"\"Purge all cache entries for a user and cache type\"\"\"\n        \n        if (user_id not in self.cache_indices or \n            cache_type not in self.cache_indices[user_id]):\n            return\n        \n        entry_ids = list(self.cache_indices[user_id][cache_type])\n        \n        for entry_id in entry_ids:\n            await self._remove_cache_entry(entry_id)\n        \n        self.cache_indices[user_id][cache_type] = []\n    \n    async def _update_cache_expiration_times(self, user_id: str, cache_type: CacheType, \n                                           new_ttl_hours: int) -> None:\n        \"\"\"Update expiration times for existing cache entries\"\"\"\n        \n        if (user_id not in self.cache_indices or \n            cache_type not in self.cache_indices[user_id]):\n            return\n        \n        for entry_id in self.cache_indices[user_id][cache_type]:\n            if entry_id in self.cache_entries:\n                entry = self.cache_entries[entry_id]\n                entry.expires_at = entry.created_at + timedelta(hours=new_ttl_hours)\n    \n    async def _encrypt_data(self, user_id: str, data: Any) -> str:\n        \"\"\"Encrypt sensitive data (simplified implementation)\"\"\"\n        \n        # In production, use proper encryption with user-specific keys\n        data_str = json.dumps(data) if not isinstance(data, str) else data\n        return f\"encrypted_{hashlib.sha256(data_str.encode()).hexdigest()}\"\n    \n    async def _decrypt_data(self, user_id: str, encrypted_data: str) -> Any:\n        \"\"\"Decrypt sensitive data (simplified implementation)\"\"\"\n        \n        # In production, use proper decryption\n        if encrypted_data.startswith(\"encrypted_\"):\n            # This is a placeholder - in production, properly decrypt\n            return {\"decrypted\": True, \"placeholder\": \"actual decrypted data\"}\n        return encrypted_data\n    \n    async def _log_cache_access(self, user_id: str, action: str, \n                              cache_type: CacheType, data_key: str) -> None:\n        \"\"\"Log cache access for audit purposes\"\"\"\n        \n        if user_id not in self.access_logs:\n            self.access_logs[user_id] = []\n        \n        log_entry = {\n            'timestamp': datetime.utcnow().isoformat(),\n            'action': action,\n            'cache_type': cache_type.value,\n            'data_key': data_key\n        }\n        \n        self.access_logs[user_id].append(log_entry)\n        \n        # Keep only recent logs\n        max_logs = 1000\n        if len(self.access_logs[user_id]) > max_logs:\n            self.access_logs[user_id] = self.access_logs[user_id][-max_logs:]\n    \n    async def _get_user_cache_statistics(self, user_id: str) -> Dict[str, Any]:\n        \"\"\"Get cache statistics for a user\"\"\"\n        \n        stats = {\n            'total_cached_items': 0,\n            'cache_size_by_type': {},\n            'cache_hit_rate': 0.0,\n            'storage_used_mb': 0.0,\n            'oldest_cache_entry': None,\n            'most_accessed_items': []\n        }\n        \n        if user_id not in self.cache_indices:\n            return stats\n        \n        total_items = 0\n        total_size = 0\n        oldest_entry = None\n        most_accessed = []\n        \n        for cache_type, entry_ids in self.cache_indices[user_id].items():\n            type_count = len(entry_ids)\n            total_items += type_count\n            stats['cache_size_by_type'][cache_type.value] = type_count\n            \n            for entry_id in entry_ids:\n                if entry_id in self.cache_entries:\n                    entry = self.cache_entries[entry_id]\n                    \n                    # Calculate approximate size\n                    entry_size = len(str(entry.data_value)) if entry.data_value else 0\n                    total_size += entry_size\n                    \n                    # Track oldest entry\n                    if oldest_entry is None or entry.created_at < oldest_entry:\n                        oldest_entry = entry.created_at\n                    \n                    # Track most accessed\n                    if entry.access_count > 0:\n                        most_accessed.append({\n                            'cache_type': cache_type.value,\n                            'data_key': entry.data_key,\n                            'access_count': entry.access_count,\n                            'last_accessed': entry.last_accessed.isoformat()\n                        })\n        \n        stats['total_cached_items'] = total_items\n        stats['storage_used_mb'] = total_size / (1024 * 1024)  # Convert to MB\n        \n        if oldest_entry:\n            stats['oldest_cache_entry'] = oldest_entry.isoformat()\n        \n        # Sort most accessed and take top 10\n        most_accessed.sort(key=lambda x: x['access_count'], reverse=True)\n        stats['most_accessed_items'] = most_accessed[:10]\n        \n        return stats\n    \n    async def _detect_user_cache_policy(self, user_id: str) -> str:\n        \"\"\"Detect user's effective cache policy\"\"\"\n        \n        if user_id not in self.user_cache_settings:\n            return CachePolicy.BALANCED.value\n        \n        # Analyze user settings to determine effective policy\n        settings = self.user_cache_settings[user_id]\n        \n        # Count enabled cache types\n        enabled_count = sum(1 for s in settings.values() if s.enabled)\n        total_count = len(settings)\n        \n        if enabled_count == 0:\n            return CachePolicy.MINIMAL.value\n        elif enabled_count < total_count * 0.5:\n            return CachePolicy.MINIMAL.value\n        elif enabled_count == total_count:\n            # Check TTL settings to determine if performance-oriented\n            avg_ttl = sum(s.ttl_hours for s in settings.values()) / len(settings)\n            default_avg_ttl = sum(c.default_ttl_hours for c in self.cache_configurations.values()) / len(self.cache_configurations)\n            \n            if avg_ttl > default_avg_ttl * 1.2:\n                return CachePolicy.PERFORMANCE.value\n        \n        return CachePolicy.BALANCED.value\n    \n    def _get_policy_description(self, policy: CachePolicy) -> str:\n        \"\"\"Get human-readable policy description\"\"\"\n        \n        descriptions = {\n            CachePolicy.MINIMAL: \"Minimal caching for maximum privacy. Slower performance but highest data protection.\",\n            CachePolicy.BALANCED: \"Balanced approach between performance and privacy. Good for most users.\",\n            CachePolicy.PERFORMANCE: \"Optimized for speed and responsiveness. Enhanced caching with reasonable privacy.\",\n            CachePolicy.USER_CONTROLLED: \"Full user control over individual cache settings. Customize each data type.\"\n        }\n        \n        return descriptions.get(policy, \"Unknown policy\")\n    \n    def _get_policy_privacy_level(self, policy: CachePolicy) -> str:\n        \"\"\"Get privacy level for policy\"\"\"\n        \n        levels = {\n            CachePolicy.MINIMAL: \"Maximum\",\n            CachePolicy.BALANCED: \"High\", \n            CachePolicy.PERFORMANCE: \"Moderate\",\n            CachePolicy.USER_CONTROLLED: \"User Defined\"\n        }\n        \n        return levels.get(policy, \"Unknown\")\n    \n    def _get_policy_performance_level(self, policy: CachePolicy) -> str:\n        \"\"\"Get performance level for policy\"\"\"\n        \n        levels = {\n            CachePolicy.MINIMAL: \"Basic\",\n            CachePolicy.BALANCED: \"Good\",\n            CachePolicy.PERFORMANCE: \"Optimized\",\n            CachePolicy.USER_CONTROLLED: \"User Defined\"\n        }\n        \n        return levels.get(policy, \"Unknown\")\n    \n    async def _get_cache_recommendations(self, user_id: str) -> List[Dict[str, Any]]:\n        \"\"\"Get personalized cache recommendations for user\"\"\"\n        \n        recommendations = []\n        \n        try:\n            stats = await self._get_user_cache_statistics(user_id)\n            \n            # Recommend enabling feed content caching if disabled\n            if user_id in self.user_cache_settings:\n                feed_settings = self.user_cache_settings[user_id].get(CacheType.FEED_CONTENT)\n                if feed_settings and not feed_settings.enabled:\n                    recommendations.append({\n                        'type': 'performance_improvement',\n                        'cache_type': CacheType.FEED_CONTENT.value,\n                        'recommendation': 'Enable feed content caching',\n                        'reason': 'Faster feed loading without significant privacy impact',\n                        'impact': 'Improves responsiveness of healing content delivery',\n                        'privacy_cost': 'Low'\n                    })\n                \n                # Recommend adjusting TTL for frequently accessed data\n                if stats['total_cached_items'] > 0:\n                    for item in stats['most_accessed_items'][:3]:\n                        if item['access_count'] > 10:  # Frequently accessed\n                            recommendations.append({\n                                'type': 'ttl_optimization',\n                                'cache_type': item['cache_type'],\n                                'recommendation': 'Consider increasing TTL for frequently accessed data',\n                                'reason': f\"This data is accessed {item['access_count']} times\",\n                                'impact': 'Reduce re-computation and improve speed',\n                                'privacy_cost': 'Low'\n                            })\n            \n            return recommendations\n            \n        except Exception as e:\n            logger.error(f\"Error generating cache recommendations for user {user_id[:8]}...: {e}\")\n            return []\n    \n    async def get_cache_analytics(self) -> Dict[str, Any]:\n        \"\"\"Get system-wide cache analytics\"\"\"\n        \n        total_users = len(self.user_cache_settings)\n        total_entries = len(self.cache_entries)\n        \n        # Calculate cache type distribution\n        type_distribution = {}\n        for entry in self.cache_entries.values():\n            cache_type = entry.cache_type.value\n            type_distribution[cache_type] = type_distribution.get(cache_type, 0) + 1\n        \n        # Calculate policy distribution\n        policy_distribution = {}\n        for user_id in self.user_cache_settings:\n            policy = await self._detect_user_cache_policy(user_id)\n            policy_distribution[policy] = policy_distribution.get(policy, 0) + 1\n        \n        # Calculate storage usage\n        total_storage_mb = 0\n        for entry in self.cache_entries.values():\n            entry_size = len(str(entry.data_value)) if entry.data_value else 0\n            total_storage_mb += entry_size / (1024 * 1024)\n        \n        return {\n            'total_users_with_cache': total_users,\n            'total_cache_entries': total_entries,\n            'cache_type_distribution': type_distribution,\n            'user_policy_distribution': policy_distribution,\n            'total_storage_mb': round(total_storage_mb, 2),\n            'average_entries_per_user': total_entries / max(total_users, 1),\n            'cache_system_healthy': total_users > 0,\n            'user_control_enabled': True,\n            'generated_at': datetime.utcnow().isoformat()\n        }"