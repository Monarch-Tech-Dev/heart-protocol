"""
Privacy Settings Manager

Comprehensive privacy control system that empowers users with granular
control over their data sharing, visibility, and protection preferences.
"""

import asyncio
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


class PrivacyLevel(Enum):
    """Overall privacy levels"""
    MAXIMUM = "maximum"                      # Highest privacy, minimal data sharing
    HIGH = "high"                           # High privacy with selective sharing
    MODERATE = "moderate"                   # Balanced privacy and functionality
    OPEN = "open"                          # More open sharing for enhanced features
    CUSTOM = "custom"                      # User-defined granular settings


class DataSharingConsent(Enum):
    """Levels of data sharing consent"""
    NONE = "none"                          # No data sharing
    ANONYMIZED_ONLY = "anonymized_only"    # Only anonymized data
    AGGREGATED_ONLY = "aggregated_only"    # Only aggregated statistics
    RESEARCH_CONSENT = "research_consent"  # Research purposes with consent
    FULL_CONSENT = "full_consent"          # Full sharing with explicit consent


class DataCategory(Enum):
    """Categories of personal data"""
    BASIC_PROFILE = "basic_profile"        # Name, age, basic demographics
    HEALING_JOURNEY = "healing_journey"    # Healing progress and goals
    MENTAL_HEALTH = "mental_health"        # Mental health information
    CRISIS_DATA = "crisis_data"            # Crisis and safety information
    INTERACTION_PATTERNS = "interaction_patterns"  # How user interacts
    CONTENT_PREFERENCES = "content_preferences"    # Content likes/dislikes
    CULTURAL_IDENTITY = "cultural_identity"        # Cultural background
    SUPPORT_NETWORK = "support_network"    # Emergency contacts, supporters
    USAGE_ANALYTICS = "usage_analytics"    # System usage patterns
    COMMUNICATION_STYLE = "communication_style"   # Communication preferences


class SharingContext(Enum):
    """Contexts in which data might be shared"""
    EMERGENCY_ONLY = "emergency_only"      # Only in crisis/emergency
    HEALTHCARE_PROVIDERS = "healthcare_providers"  # With healthcare professionals
    RESEARCH_STUDIES = "research_studies"  # For research purposes
    SYSTEM_IMPROVEMENT = "system_improvement"      # To improve the platform
    COMMUNITY_MATCHING = "community_matching"     # For peer support matching
    PERSONALIZATION = "personalization"   # For content personalization
    ANALYTICS = "analytics"                # For usage analytics
    LEGAL_COMPLIANCE = "legal_compliance"  # Legal requirements only


@dataclass
class DataSharingRule:
    """Rule for sharing specific data category in specific context"""
    rule_id: str
    user_id: str
    data_category: DataCategory
    sharing_context: SharingContext
    consent_level: DataSharingConsent
    allowed: bool
    conditions: List[str] = field(default_factory=list)
    expiration_date: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_modified: datetime = field(default_factory=datetime.utcnow)
    user_notes: str = ""


@dataclass
class PrivacyAuditLog:
    """Log entry for privacy-related actions"""
    log_id: str
    user_id: str
    action: str
    data_category: Optional[DataCategory]
    sharing_context: Optional[SharingContext]
    details: Dict[str, Any]
    timestamp: datetime
    system_initiated: bool
    user_consent_verified: bool


@dataclass
class DataRetentionPolicy:
    """Data retention policy for specific data category"""
    policy_id: str
    user_id: str
    data_category: DataCategory
    retention_days: int  # -1 for indefinite
    auto_delete_enabled: bool
    deletion_method: str  # 'soft_delete', 'hard_delete', 'anonymize'
    grace_period_days: int
    user_notification_enabled: bool
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_modified: datetime = field(default_factory=datetime.utcnow)


class PrivacySettingsManager:
    """
    Comprehensive privacy settings management that puts users in complete
    control of their data sharing, retention, and protection preferences.
    
    Core Principles:
    - User sovereignty over personal data
    - Transparent data usage
    - Granular control options
    - Privacy by default
    - Informed consent
    - Right to be forgotten
    - Data minimization
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Privacy settings storage
        self.user_privacy_levels = {}         # user_id -> PrivacyLevel
        self.data_sharing_rules = {}          # user_id -> List[DataSharingRule]
        self.data_retention_policies = {}     # user_id -> List[DataRetentionPolicy]
        self.privacy_audit_logs = {}          # user_id -> List[PrivacyAuditLog]
        
        # Consent tracking
        self.consent_records = {}             # user_id -> consent_data
        self.consent_history = {}             # user_id -> List[consent_changes]
        
        # Data discovery and mapping
        self.user_data_inventory = {}         # user_id -> data_inventory
        self.data_flow_tracking = {}          # user_id -> data_flow_logs
        
        # Privacy templates and recommendations
        self.privacy_templates = {}           # template_id -> template_config
        self.privacy_recommendations = {}     # user_id -> recommendations
        
        self._initialize_default_templates()
        self._initialize_data_category_definitions()
        
        logger.info("Privacy Settings Manager initialized with user sovereignty focus")
    
    def _initialize_default_templates(self) -> None:
        """Initialize default privacy templates"""
        
        self.privacy_templates = {
            'maximum_privacy': {
                'name': 'Maximum Privacy',
                'description': 'Highest privacy protection with minimal data sharing',
                'default_sharing_consent': DataSharingConsent.NONE,
                'default_retention_days': 30,
                'emergency_sharing_only': True,
                'analytics_participation': False,
                'personalization_level': 'minimal',
                'rules': {
                    DataCategory.BASIC_PROFILE: {
                        SharingContext.EMERGENCY_ONLY: DataSharingConsent.ANONYMIZED_ONLY
                    },
                    DataCategory.CRISIS_DATA: {
                        SharingContext.EMERGENCY_ONLY: DataSharingConsent.FULL_CONSENT
                    }
                }
            },
            
            'balanced_privacy': {
                'name': 'Balanced Privacy',
                'description': 'Good privacy with selective sharing for enhanced features',
                'default_sharing_consent': DataSharingConsent.ANONYMIZED_ONLY,
                'default_retention_days': 180,
                'emergency_sharing_only': False,
                'analytics_participation': True,
                'personalization_level': 'moderate',
                'rules': {
                    DataCategory.HEALING_JOURNEY: {
                        SharingContext.PERSONALIZATION: DataSharingConsent.ANONYMIZED_ONLY,
                        SharingContext.RESEARCH_STUDIES: DataSharingConsent.ANONYMIZED_ONLY
                    },
                    DataCategory.USAGE_ANALYTICS: {
                        SharingContext.SYSTEM_IMPROVEMENT: DataSharingConsent.AGGREGATED_ONLY
                    },
                    DataCategory.CRISIS_DATA: {
                        SharingContext.EMERGENCY_ONLY: DataSharingConsent.FULL_CONSENT
                    }
                }
            },
            
            'open_sharing': {
                'name': 'Open Sharing',
                'description': 'More open sharing for enhanced personalization and community',
                'default_sharing_consent': DataSharingConsent.RESEARCH_CONSENT,
                'default_retention_days': 365,
                'emergency_sharing_only': False,
                'analytics_participation': True,
                'personalization_level': 'high',
                'rules': {
                    DataCategory.HEALING_JOURNEY: {
                        SharingContext.PERSONALIZATION: DataSharingConsent.FULL_CONSENT,
                        SharingContext.RESEARCH_STUDIES: DataSharingConsent.RESEARCH_CONSENT,
                        SharingContext.COMMUNITY_MATCHING: DataSharingConsent.ANONYMIZED_ONLY
                    },
                    DataCategory.CONTENT_PREFERENCES: {
                        SharingContext.PERSONALIZATION: DataSharingConsent.FULL_CONSENT,
                        SharingContext.SYSTEM_IMPROVEMENT: DataSharingConsent.AGGREGATED_ONLY
                    }
                }
            }
        }
    
    def _initialize_data_category_definitions(self) -> None:
        """Initialize definitions for data categories"""
        
        self.data_category_definitions = {
            DataCategory.BASIC_PROFILE: {
                'name': 'Basic Profile',
                'description': 'Name, age, location, basic demographic information',
                'sensitivity_level': 'low',
                'examples': ['Username', 'Age range', 'General location', 'Preferred language'],
                'default_retention_days': 365,
                'privacy_implications': 'Basic identifying information'
            },
            
            DataCategory.HEALING_JOURNEY: {
                'name': 'Healing Journey',
                'description': 'Your healing goals, progress, and journey milestones',
                'sensitivity_level': 'medium',
                'examples': ['Healing goals', 'Progress updates', 'Milestone achievements', 'Journey reflections'],
                'default_retention_days': 730,  # 2 years
                'privacy_implications': 'Reveals personal growth and healing path'
            },
            
            DataCategory.MENTAL_HEALTH: {
                'name': 'Mental Health Information',
                'description': 'Mental health status, conditions, and care information',
                'sensitivity_level': 'high',
                'examples': ['Mental health conditions', 'Treatment history', 'Medication information', 'Therapy notes'],
                'default_retention_days': 2555,  # 7 years (medical standard)
                'privacy_implications': 'Highly sensitive health information protected by healthcare privacy laws'
            },
            
            DataCategory.CRISIS_DATA: {
                'name': 'Crisis & Safety Data',
                'description': 'Crisis episodes, safety concerns, and emergency information',
                'sensitivity_level': 'critical',
                'examples': ['Crisis episodes', 'Safety assessments', 'Emergency contacts', 'Risk indicators'],
                'default_retention_days': 2555,  # 7 years
                'privacy_implications': 'Critical safety information that may be shared in emergencies'
            },
            
            DataCategory.INTERACTION_PATTERNS: {
                'name': 'Interaction Patterns',
                'description': 'How you interact with the system and community',
                'sensitivity_level': 'medium',
                'examples': ['Usage patterns', 'Communication style', 'Engagement levels', 'Response patterns'],
                'default_retention_days': 365,
                'privacy_implications': 'Reveals behavioral patterns and preferences'
            },
            
            DataCategory.CONTENT_PREFERENCES: {
                'name': 'Content Preferences',
                'description': 'Your content likes, dislikes, and personalization data',
                'sensitivity_level': 'low',
                'examples': ['Content ratings', 'Topic preferences', 'Filter settings', 'Personalization choices'],
                'default_retention_days': 365,
                'privacy_implications': 'Shows interests and content preferences'
            },
            
            DataCategory.CULTURAL_IDENTITY: {
                'name': 'Cultural Identity',
                'description': 'Cultural background, traditions, and identity information',
                'sensitivity_level': 'medium',
                'examples': ['Cultural background', 'Religious preferences', 'Language preferences', 'Cultural practices'],
                'default_retention_days': 730,
                'privacy_implications': 'Reveals cultural and religious identity'
            },
            
            DataCategory.SUPPORT_NETWORK: {
                'name': 'Support Network',
                'description': 'Emergency contacts, supporters, and care team information',
                'sensitivity_level': 'high',
                'examples': ['Emergency contacts', 'Healthcare providers', 'Support persons', 'Care team members'],
                'default_retention_days': 1095,  # 3 years
                'privacy_implications': 'Contains contact information of others and support relationships'
            },
            
            DataCategory.USAGE_ANALYTICS: {
                'name': 'Usage Analytics',
                'description': 'System usage data for improving platform performance',
                'sensitivity_level': 'low',
                'examples': ['Page views', 'Feature usage', 'Performance metrics', 'Error logs'],
                'default_retention_days': 180,
                'privacy_implications': 'Technical usage data, typically anonymized'
            },
            
            DataCategory.COMMUNICATION_STYLE: {
                'name': 'Communication Style',
                'description': 'Preferred communication methods and interaction styles',
                'sensitivity_level': 'low',
                'examples': ['Communication preferences', 'Notification settings', 'Interaction styles', 'Language choices'],
                'default_retention_days': 365,
                'privacy_implications': 'Shows communication preferences and interaction patterns'
            }
        }
    
    async def initialize_user_privacy_settings(self, user_id: str, 
                                             privacy_level: PrivacyLevel = PrivacyLevel.MODERATE,
                                             template_id: Optional[str] = None) -> bool:
        """Initialize privacy settings for a new user"""
        
        try:\n            # Set user's privacy level\n            self.user_privacy_levels[user_id] = privacy_level\n            \n            # Initialize empty collections\n            self.data_sharing_rules[user_id] = []\n            self.data_retention_policies[user_id] = []\n            self.privacy_audit_logs[user_id] = []\n            self.consent_records[user_id] = {}\n            self.consent_history[user_id] = []\n            \n            # Apply template if specified\n            if template_id and template_id in self.privacy_templates:\n                await self._apply_privacy_template(user_id, template_id)\n            else:\n                # Apply default privacy level settings\n                await self._apply_default_privacy_settings(user_id, privacy_level)\n            \n            # Log initialization\n            await self._log_privacy_action(\n                user_id, 'privacy_settings_initialized', \n                details={'privacy_level': privacy_level.value, 'template_id': template_id}\n            )\n            \n            logger.info(f\"Initialized privacy settings for user {user_id[:8]}... at {privacy_level.value} level\")\n            return True\n            \n        except Exception as e:\n            logger.error(f\"Error initializing privacy settings for user {user_id[:8]}...: {e}\")\n            return False\n    \n    async def set_data_sharing_rule(self, user_id: str, \n                                  data_category: DataCategory,\n                                  sharing_context: SharingContext,\n                                  consent_level: DataSharingConsent,\n                                  conditions: Optional[List[str]] = None,\n                                  expiration_date: Optional[datetime] = None) -> bool:\n        \"\"\"Set or update a data sharing rule\"\"\"\n        \n        try:\n            # Ensure user is initialized\n            if user_id not in self.data_sharing_rules:\n                await self.initialize_user_privacy_settings(user_id)\n            \n            # Generate rule ID\n            rule_id = f\"{user_id}_{data_category.value}_{sharing_context.value}_{int(datetime.utcnow().timestamp())}\"\n            \n            # Create new rule\n            rule = DataSharingRule(\n                rule_id=rule_id,\n                user_id=user_id,\n                data_category=data_category,\n                sharing_context=sharing_context,\n                consent_level=consent_level,\n                allowed=consent_level != DataSharingConsent.NONE,\n                conditions=conditions or [],\n                expiration_date=expiration_date,\n                created_at=datetime.utcnow(),\n                last_modified=datetime.utcnow()\n            )\n            \n            # Remove any existing rule for same category/context combination\n            existing_rules = self.data_sharing_rules[user_id]\n            self.data_sharing_rules[user_id] = [\n                r for r in existing_rules \n                if not (r.data_category == data_category and r.sharing_context == sharing_context)\n            ]\n            \n            # Add new rule\n            self.data_sharing_rules[user_id].append(rule)\n            \n            # Record consent\n            await self._record_consent_change(user_id, data_category, sharing_context, consent_level)\n            \n            # Log the change\n            await self._log_privacy_action(\n                user_id, 'data_sharing_rule_updated',\n                data_category=data_category,\n                sharing_context=sharing_context,\n                details={\n                    'consent_level': consent_level.value,\n                    'allowed': rule.allowed,\n                    'conditions': conditions or [],\n                    'has_expiration': expiration_date is not None\n                }\n            )\n            \n            logger.info(f\"Set data sharing rule for user {user_id[:8]}... \"\n                       f\"({data_category.value} in {sharing_context.value} context)\")\n            \n            return True\n            \n        except Exception as e:\n            logger.error(f\"Error setting data sharing rule for user {user_id[:8]}...: {e}\")\n            return False\n    \n    async def check_data_sharing_permission(self, user_id: str,\n                                          data_category: DataCategory,\n                                          sharing_context: SharingContext) -> Dict[str, Any]:\n        \"\"\"Check if data sharing is permitted for specific category and context\"\"\"\n        \n        try:\n            # Default to no permission\n            permission_result = {\n                'allowed': False,\n                'consent_level': DataSharingConsent.NONE,\n                'conditions': [],\n                'expiration_date': None,\n                'reason': 'No explicit permission found'\n            }\n            \n            if user_id not in self.data_sharing_rules:\n                return permission_result\n            \n            # Find applicable rule\n            for rule in self.data_sharing_rules[user_id]:\n                if (rule.data_category == data_category and \n                    rule.sharing_context == sharing_context):\n                    \n                    # Check if rule has expired\n                    if rule.expiration_date and datetime.utcnow() > rule.expiration_date:\n                        permission_result['reason'] = 'Permission expired'\n                        break\n                    \n                    permission_result = {\n                        'allowed': rule.allowed,\n                        'consent_level': rule.consent_level,\n                        'conditions': rule.conditions,\n                        'expiration_date': rule.expiration_date,\n                        'reason': 'Explicit user permission' if rule.allowed else 'User denied permission',\n                        'rule_id': rule.rule_id\n                    }\n                    break\n            \n            # Log permission check\n            await self._log_privacy_action(\n                user_id, 'data_sharing_permission_checked',\n                data_category=data_category,\n                sharing_context=sharing_context,\n                details={\n                    'permission_granted': permission_result['allowed'],\n                    'consent_level': permission_result['consent_level'].value if hasattr(permission_result['consent_level'], 'value') else str(permission_result['consent_level']),\n                    'reason': permission_result['reason']\n                },\n                system_initiated=True\n            )\n            \n            return permission_result\n            \n        except Exception as e:\n            logger.error(f\"Error checking data sharing permission for user {user_id[:8]}...: {e}\")\n            return {\n                'allowed': False,\n                'consent_level': DataSharingConsent.NONE,\n                'conditions': [],\n                'reason': 'Error checking permission'\n            }\n    \n    async def set_data_retention_policy(self, user_id: str,\n                                      data_category: DataCategory,\n                                      retention_days: int,\n                                      auto_delete_enabled: bool = True,\n                                      deletion_method: str = 'soft_delete') -> bool:\n        \"\"\"Set data retention policy for a data category\"\"\"\n        \n        try:\n            # Ensure user is initialized\n            if user_id not in self.data_retention_policies:\n                await self.initialize_user_privacy_settings(user_id)\n            \n            # Validate retention period\n            if retention_days < -1 or retention_days == 0:\n                logger.error(f\"Invalid retention period: {retention_days}\")\n                return False\n            \n            # Generate policy ID\n            policy_id = f\"{user_id}_{data_category.value}_retention_{int(datetime.utcnow().timestamp())}\"\n            \n            # Create retention policy\n            policy = DataRetentionPolicy(\n                policy_id=policy_id,\n                user_id=user_id,\n                data_category=data_category,\n                retention_days=retention_days,\n                auto_delete_enabled=auto_delete_enabled,\n                deletion_method=deletion_method,\n                grace_period_days=30,  # 30-day grace period before deletion\n                user_notification_enabled=True,\n                created_at=datetime.utcnow(),\n                last_modified=datetime.utcnow()\n            )\n            \n            # Remove existing policy for same category\n            existing_policies = self.data_retention_policies[user_id]\n            self.data_retention_policies[user_id] = [\n                p for p in existing_policies if p.data_category != data_category\n            ]\n            \n            # Add new policy\n            self.data_retention_policies[user_id].append(policy)\n            \n            # Log the change\n            await self._log_privacy_action(\n                user_id, 'data_retention_policy_set',\n                data_category=data_category,\n                details={\n                    'retention_days': retention_days,\n                    'auto_delete_enabled': auto_delete_enabled,\n                    'deletion_method': deletion_method\n                }\n            )\n            \n            logger.info(f\"Set data retention policy for user {user_id[:8]}... \"\n                       f\"({data_category.value}: {retention_days} days)\")\n            \n            return True\n            \n        except Exception as e:\n            logger.error(f\"Error setting data retention policy for user {user_id[:8]}...: {e}\")\n            return False\n    \n    async def get_user_privacy_dashboard(self, user_id: str) -> Dict[str, Any]:\n        \"\"\"Get comprehensive privacy dashboard for user\"\"\"\n        \n        try:\n            dashboard = {\n                'user_id': user_id,\n                'privacy_level': self.user_privacy_levels.get(user_id, PrivacyLevel.MODERATE).value,\n                'data_sharing_rules': {},\n                'data_retention_policies': {},\n                'data_categories': {},\n                'privacy_score': 0.0,\n                'recommendations': [],\n                'audit_summary': {},\n                'consent_status': {}\n            }\n            \n            # Add data category information\n            for category, definition in self.data_category_definitions.items():\n                dashboard['data_categories'][category.value] = {\n                    'name': definition['name'],\n                    'description': definition['description'],\n                    'sensitivity_level': definition['sensitivity_level'],\n                    'privacy_implications': definition['privacy_implications'],\n                    'current_sharing_rules': [],\n                    'current_retention_policy': None\n                }\n            \n            # Add current sharing rules\n            if user_id in self.data_sharing_rules:\n                for rule in self.data_sharing_rules[user_id]:\n                    category_key = rule.data_category.value\n                    if category_key in dashboard['data_categories']:\n                        rule_info = {\n                            'sharing_context': rule.sharing_context.value,\n                            'consent_level': rule.consent_level.value,\n                            'allowed': rule.allowed,\n                            'conditions': rule.conditions,\n                            'expires_at': rule.expiration_date.isoformat() if rule.expiration_date else None,\n                            'last_modified': rule.last_modified.isoformat()\n                        }\n                        dashboard['data_categories'][category_key]['current_sharing_rules'].append(rule_info)\n            \n            # Add current retention policies\n            if user_id in self.data_retention_policies:\n                for policy in self.data_retention_policies[user_id]:\n                    category_key = policy.data_category.value\n                    if category_key in dashboard['data_categories']:\n                        policy_info = {\n                            'retention_days': policy.retention_days,\n                            'auto_delete_enabled': policy.auto_delete_enabled,\n                            'deletion_method': policy.deletion_method,\n                            'grace_period_days': policy.grace_period_days,\n                            'last_modified': policy.last_modified.isoformat()\n                        }\n                        dashboard['data_categories'][category_key]['current_retention_policy'] = policy_info\n            \n            # Calculate privacy score\n            dashboard['privacy_score'] = await self._calculate_privacy_score(user_id)\n            \n            # Generate recommendations\n            dashboard['recommendations'] = await self._generate_privacy_recommendations(user_id)\n            \n            # Add audit summary\n            dashboard['audit_summary'] = await self._get_privacy_audit_summary(user_id)\n            \n            # Add consent status\n            dashboard['consent_status'] = await self._get_consent_status_summary(user_id)\n            \n            return dashboard\n            \n        except Exception as e:\n            logger.error(f\"Error generating privacy dashboard for user {user_id[:8]}...: {e}\")\n            return {}\n    \n    async def export_user_privacy_data(self, user_id: str) -> Dict[str, Any]:\n        \"\"\"Export all privacy settings and data for user transparency\"\"\"\n        \n        try:\n            export_data = {\n                'user_id': user_id,\n                'exported_at': datetime.utcnow().isoformat(),\n                'privacy_level': self.user_privacy_levels.get(user_id, PrivacyLevel.MODERATE).value,\n                'data_sharing_rules': [],\n                'data_retention_policies': [],\n                'consent_records': self.consent_records.get(user_id, {}),\n                'consent_history': [],\n                'privacy_audit_logs': []\n            }\n            \n            # Export sharing rules\n            if user_id in self.data_sharing_rules:\n                for rule in self.data_sharing_rules[user_id]:\n                    export_data['data_sharing_rules'].append({\n                        'rule_id': rule.rule_id,\n                        'data_category': rule.data_category.value,\n                        'sharing_context': rule.sharing_context.value,\n                        'consent_level': rule.consent_level.value,\n                        'allowed': rule.allowed,\n                        'conditions': rule.conditions,\n                        'expiration_date': rule.expiration_date.isoformat() if rule.expiration_date else None,\n                        'created_at': rule.created_at.isoformat(),\n                        'last_modified': rule.last_modified.isoformat(),\n                        'user_notes': rule.user_notes\n                    })\n            \n            # Export retention policies\n            if user_id in self.data_retention_policies:\n                for policy in self.data_retention_policies[user_id]:\n                    export_data['data_retention_policies'].append({\n                        'policy_id': policy.policy_id,\n                        'data_category': policy.data_category.value,\n                        'retention_days': policy.retention_days,\n                        'auto_delete_enabled': policy.auto_delete_enabled,\n                        'deletion_method': policy.deletion_method,\n                        'grace_period_days': policy.grace_period_days,\n                        'user_notification_enabled': policy.user_notification_enabled,\n                        'created_at': policy.created_at.isoformat(),\n                        'last_modified': policy.last_modified.isoformat()\n                    })\n            \n            # Export consent history\n            if user_id in self.consent_history:\n                for consent_change in self.consent_history[user_id]:\n                    export_data['consent_history'].append({\n                        'timestamp': consent_change['timestamp'].isoformat(),\n                        'data_category': consent_change['data_category'],\n                        'sharing_context': consent_change['sharing_context'],\n                        'old_consent_level': consent_change['old_consent_level'],\n                        'new_consent_level': consent_change['new_consent_level'],\n                        'user_initiated': consent_change['user_initiated']\n                    })\n            \n            # Export privacy audit logs (last 100 entries)\n            if user_id in self.privacy_audit_logs:\n                recent_logs = self.privacy_audit_logs[user_id][-100:]\n                for log_entry in recent_logs:\n                    export_data['privacy_audit_logs'].append({\n                        'log_id': log_entry.log_id,\n                        'action': log_entry.action,\n                        'data_category': log_entry.data_category.value if log_entry.data_category else None,\n                        'sharing_context': log_entry.sharing_context.value if log_entry.sharing_context else None,\n                        'details': log_entry.details,\n                        'timestamp': log_entry.timestamp.isoformat(),\n                        'system_initiated': log_entry.system_initiated,\n                        'user_consent_verified': log_entry.user_consent_verified\n                    })\n            \n            return export_data\n            \n        except Exception as e:\n            logger.error(f\"Error exporting privacy data for user {user_id[:8]}...: {e}\")\n            return {}\n    \n    async def delete_user_data(self, user_id: str, \n                             data_categories: Optional[List[DataCategory]] = None,\n                             deletion_method: str = 'hard_delete') -> Dict[str, Any]:\n        \"\"\"Delete user data according to their retention policies\"\"\"\n        \n        try:\n            deletion_result = {\n                'user_id': user_id,\n                'deletion_initiated_at': datetime.utcnow().isoformat(),\n                'deletion_method': deletion_method,\n                'categories_deleted': [],\n                'categories_failed': [],\n                'total_items_deleted': 0\n            }\n            \n            # Determine which categories to delete\n            categories_to_delete = data_categories or list(DataCategory)\n            \n            for category in categories_to_delete:\n                try:\n                    # Check retention policy\n                    retention_policy = None\n                    if user_id in self.data_retention_policies:\n                        for policy in self.data_retention_policies[user_id]:\n                            if policy.data_category == category:\n                                retention_policy = policy\n                                break\n                    \n                    # Perform deletion based on method\n                    if deletion_method == 'hard_delete':\n                        deleted_count = await self._hard_delete_category_data(user_id, category)\n                    elif deletion_method == 'soft_delete':\n                        deleted_count = await self._soft_delete_category_data(user_id, category)\n                    elif deletion_method == 'anonymize':\n                        deleted_count = await self._anonymize_category_data(user_id, category)\n                    else:\n                        logger.error(f\"Unknown deletion method: {deletion_method}\")\n                        deletion_result['categories_failed'].append(category.value)\n                        continue\n                    \n                    deletion_result['categories_deleted'].append({\n                        'category': category.value,\n                        'items_deleted': deleted_count,\n                        'deletion_method': deletion_method\n                    })\n                    \n                    deletion_result['total_items_deleted'] += deleted_count\n                    \n                except Exception as e:\n                    logger.error(f\"Error deleting {category.value} data for user {user_id[:8]}...: {e}\")\n                    deletion_result['categories_failed'].append(category.value)\n            \n            # Log deletion\n            await self._log_privacy_action(\n                user_id, 'user_data_deleted',\n                details={\n                    'deletion_method': deletion_method,\n                    'categories_deleted': [cat['category'] for cat in deletion_result['categories_deleted']],\n                    'total_items_deleted': deletion_result['total_items_deleted']\n                }\n            )\n            \n            logger.info(f\"Deleted data for user {user_id[:8]}... \"\n                       f\"({deletion_result['total_items_deleted']} items)\")\n            \n            return deletion_result\n            \n        except Exception as e:\n            logger.error(f\"Error deleting user data for {user_id[:8]}...: {e}\")\n            return {'error': str(e)}\n    \n    async def _apply_privacy_template(self, user_id: str, template_id: str) -> None:\n        \"\"\"Apply privacy template to user\"\"\"\n        \n        template = self.privacy_templates[template_id]\n        \n        # Apply template rules\n        for data_category, contexts in template['rules'].items():\n            for sharing_context, consent_level in contexts.items():\n                await self.set_data_sharing_rule(\n                    user_id, data_category, sharing_context, consent_level\n                )\n        \n        # Set retention policies based on template\n        retention_days = template['default_retention_days']\n        for category in DataCategory:\n            await self.set_data_retention_policy(\n                user_id, category, retention_days\n            )\n    \n    async def _apply_default_privacy_settings(self, user_id: str, \n                                            privacy_level: PrivacyLevel) -> None:\n        \"\"\"Apply default privacy settings based on privacy level\"\"\"\n        \n        # Apply settings based on privacy level\n        if privacy_level == PrivacyLevel.MAXIMUM:\n            await self._apply_privacy_template(user_id, 'maximum_privacy')\n        elif privacy_level == PrivacyLevel.MODERATE:\n            await self._apply_privacy_template(user_id, 'balanced_privacy')\n        elif privacy_level == PrivacyLevel.OPEN:\n            await self._apply_privacy_template(user_id, 'open_sharing')\n        else:\n            # For HIGH and CUSTOM, use balanced as starting point\n            await self._apply_privacy_template(user_id, 'balanced_privacy')\n    \n    async def _record_consent_change(self, user_id: str, \n                                   data_category: DataCategory,\n                                   sharing_context: SharingContext,\n                                   new_consent_level: DataSharingConsent) -> None:\n        \"\"\"Record consent change in history\"\"\"\n        \n        if user_id not in self.consent_history:\n            self.consent_history[user_id] = []\n        \n        # Find old consent level\n        old_consent_level = DataSharingConsent.NONE\n        for rule in self.data_sharing_rules.get(user_id, []):\n            if (rule.data_category == data_category and \n                rule.sharing_context == sharing_context):\n                old_consent_level = rule.consent_level\n                break\n        \n        consent_change = {\n            'timestamp': datetime.utcnow(),\n            'data_category': data_category.value,\n            'sharing_context': sharing_context.value,\n            'old_consent_level': old_consent_level.value,\n            'new_consent_level': new_consent_level.value,\n            'user_initiated': True\n        }\n        \n        self.consent_history[user_id].append(consent_change)\n        \n        # Keep only recent history\n        max_history = 1000\n        if len(self.consent_history[user_id]) > max_history:\n            self.consent_history[user_id] = self.consent_history[user_id][-max_history:]\n    \n    async def _log_privacy_action(self, user_id: str, action: str,\n                                data_category: Optional[DataCategory] = None,\n                                sharing_context: Optional[SharingContext] = None,\n                                details: Optional[Dict[str, Any]] = None,\n                                system_initiated: bool = False) -> None:\n        \"\"\"Log privacy-related action\"\"\"\n        \n        if user_id not in self.privacy_audit_logs:\n            self.privacy_audit_logs[user_id] = []\n        \n        log_id = f\"{user_id}_{action}_{int(datetime.utcnow().timestamp())}\"\n        \n        log_entry = PrivacyAuditLog(\n            log_id=log_id,\n            user_id=user_id,\n            action=action,\n            data_category=data_category,\n            sharing_context=sharing_context,\n            details=details or {},\n            timestamp=datetime.utcnow(),\n            system_initiated=system_initiated,\n            user_consent_verified=not system_initiated\n        )\n        \n        self.privacy_audit_logs[user_id].append(log_entry)\n        \n        # Keep only recent logs\n        max_logs = 2000\n        if len(self.privacy_audit_logs[user_id]) > max_logs:\n            self.privacy_audit_logs[user_id] = self.privacy_audit_logs[user_id][-max_logs:]\n    \n    async def _calculate_privacy_score(self, user_id: str) -> float:\n        \"\"\"Calculate privacy score (0.0 to 1.0) for user\"\"\"\n        \n        try:\n            score = 0.0\n            max_score = 0.0\n            \n            # Score based on data sharing restrictions\n            for category in DataCategory:\n                for context in SharingContext:\n                    max_score += 1.0\n                    \n                    permission = await self.check_data_sharing_permission(user_id, category, context)\n                    \n                    if not permission['allowed']:\n                        score += 1.0  # Higher score for more restricted sharing\n                    elif permission['consent_level'] == DataSharingConsent.ANONYMIZED_ONLY:\n                        score += 0.8\n                    elif permission['consent_level'] == DataSharingConsent.AGGREGATED_ONLY:\n                        score += 0.6\n                    elif permission['consent_level'] == DataSharingConsent.RESEARCH_CONSENT:\n                        score += 0.4\n                    # Full consent gets 0 points (less privacy)\n            \n            # Bonus for shorter retention periods\n            if user_id in self.data_retention_policies:\n                for policy in self.data_retention_policies[user_id]:\n                    if policy.retention_days < 90:  # Less than 3 months\n                        score += 0.1\n                    elif policy.retention_days < 365:  # Less than 1 year\n                        score += 0.05\n            \n            return min(score / max_score, 1.0) if max_score > 0 else 0.5\n            \n        except Exception as e:\n            logger.error(f\"Error calculating privacy score for user {user_id[:8]}...: {e}\")\n            return 0.5\n    \n    async def _generate_privacy_recommendations(self, user_id: str) -> List[Dict[str, Any]]:\n        \"\"\"Generate personalized privacy recommendations\"\"\"\n        \n        recommendations = []\n        \n        try:\n            privacy_score = await self._calculate_privacy_score(user_id)\n            \n            # Recommend higher privacy if score is low\n            if privacy_score < 0.4:\n                recommendations.append({\n                    'type': 'increase_privacy',\n                    'title': 'Consider Increasing Privacy Protection',\n                    'description': 'Your current settings allow broad data sharing. Consider restricting some categories.',\n                    'impact': 'Higher privacy protection',\n                    'trade_off': 'May reduce personalization features'\n                })\n            \n            # Recommend enabling analytics if completely disabled\n            analytics_allowed = await self.check_data_sharing_permission(\n                user_id, DataCategory.USAGE_ANALYTICS, SharingContext.SYSTEM_IMPROVEMENT\n            )\n            \n            if not analytics_allowed['allowed']:\n                recommendations.append({\n                    'type': 'enable_anonymized_analytics',\n                    'title': 'Consider Enabling Anonymized Analytics',\n                    'description': 'Anonymized usage analytics help improve the platform for everyone.',\n                    'impact': 'Better platform improvements',\n                    'trade_off': 'Minimal privacy impact with anonymization'\n                })\n            \n            # Recommend reviewing retention policies\n            if user_id in self.data_retention_policies:\n                long_retention_count = sum(\n                    1 for policy in self.data_retention_policies[user_id]\n                    if policy.retention_days > 365\n                )\n                \n                if long_retention_count > 3:\n                    recommendations.append({\n                        'type': 'review_retention',\n                        'title': 'Review Data Retention Periods',\n                        'description': 'You have several data categories with long retention periods.',\n                        'impact': 'Reduced data storage',\n                        'trade_off': 'May affect long-term personalization'\n                    })\n            \n            return recommendations\n            \n        except Exception as e:\n            logger.error(f\"Error generating privacy recommendations for user {user_id[:8]}...: {e}\")\n            return []\n    \n    async def _get_privacy_audit_summary(self, user_id: str) -> Dict[str, Any]:\n        \"\"\"Get summary of privacy audit logs\"\"\"\n        \n        summary = {\n            'total_actions': 0,\n            'recent_actions': 0,\n            'user_initiated_actions': 0,\n            'system_initiated_actions': 0,\n            'most_common_actions': [],\n            'last_activity': None\n        }\n        \n        if user_id not in self.privacy_audit_logs:\n            return summary\n        \n        logs = self.privacy_audit_logs[user_id]\n        summary['total_actions'] = len(logs)\n        \n        # Count recent actions (last 30 days)\n        thirty_days_ago = datetime.utcnow() - timedelta(days=30)\n        recent_logs = [log for log in logs if log.timestamp > thirty_days_ago]\n        summary['recent_actions'] = len(recent_logs)\n        \n        # Count user vs system initiated\n        summary['user_initiated_actions'] = sum(1 for log in logs if not log.system_initiated)\n        summary['system_initiated_actions'] = sum(1 for log in logs if log.system_initiated)\n        \n        # Find most common actions\n        action_counts = {}\n        for log in logs:\n            action_counts[log.action] = action_counts.get(log.action, 0) + 1\n        \n        summary['most_common_actions'] = sorted(\n            action_counts.items(), key=lambda x: x[1], reverse=True\n        )[:5]\n        \n        # Last activity\n        if logs:\n            summary['last_activity'] = logs[-1].timestamp.isoformat()\n        \n        return summary\n    \n    async def _get_consent_status_summary(self, user_id: str) -> Dict[str, Any]:\n        \"\"\"Get summary of consent status\"\"\"\n        \n        summary = {\n            'total_consent_decisions': 0,\n            'consents_given': 0,\n            'consents_denied': 0,\n            'pending_consent_requests': 0,\n            'consent_coverage': 0.0\n        }\n        \n        if user_id not in self.data_sharing_rules:\n            return summary\n        \n        rules = self.data_sharing_rules[user_id]\n        summary['total_consent_decisions'] = len(rules)\n        \n        summary['consents_given'] = sum(1 for rule in rules if rule.allowed)\n        summary['consents_denied'] = sum(1 for rule in rules if not rule.allowed)\n        \n        # Calculate consent coverage (percentage of possible data/context combinations with explicit rules)\n        total_possible_combinations = len(DataCategory) * len(SharingContext)\n        summary['consent_coverage'] = (len(rules) / total_possible_combinations) * 100\n        \n        return summary\n    \n    async def _hard_delete_category_data(self, user_id: str, category: DataCategory) -> int:\n        \"\"\"Hard delete all data for a category (placeholder implementation)\"\"\"\n        \n        # In production, this would delete actual data from various storage systems\n        logger.info(f\"Hard deleting {category.value} data for user {user_id[:8]}...\")\n        return 1  # Placeholder count\n    \n    async def _soft_delete_category_data(self, user_id: str, category: DataCategory) -> int:\n        \"\"\"Soft delete data for a category (mark as deleted)\"\"\"\n        \n        # In production, this would mark data as deleted without removing it\n        logger.info(f\"Soft deleting {category.value} data for user {user_id[:8]}...\")\n        return 1  # Placeholder count\n    \n    async def _anonymize_category_data(self, user_id: str, category: DataCategory) -> int:\n        \"\"\"Anonymize data for a category\"\"\"\n        \n        # In production, this would remove personally identifiable information\n        logger.info(f\"Anonymizing {category.value} data for user {user_id[:8]}...\")\n        return 1  # Placeholder count\n    \n    async def get_privacy_analytics(self) -> Dict[str, Any]:\n        \"\"\"Get system-wide privacy analytics\"\"\"\n        \n        total_users = len(self.user_privacy_levels)\n        \n        # Calculate privacy level distribution\n        privacy_level_distribution = {}\n        for level in self.user_privacy_levels.values():\n            level_value = level.value\n            privacy_level_distribution[level_value] = privacy_level_distribution.get(level_value, 0) + 1\n        \n        # Calculate average privacy score\n        total_score = 0.0\n        score_count = 0\n        for user_id in self.user_privacy_levels:\n            try:\n                score = await self._calculate_privacy_score(user_id)\n                total_score += score\n                score_count += 1\n            except:\n                pass\n        \n        average_privacy_score = total_score / max(score_count, 1)\n        \n        # Calculate consent statistics\n        total_consent_decisions = sum(len(rules) for rules in self.data_sharing_rules.values())\n        consents_given = 0\n        for rules in self.data_sharing_rules.values():\n            consents_given += sum(1 for rule in rules if rule.allowed)\n        \n        consent_approval_rate = (consents_given / max(total_consent_decisions, 1)) * 100\n        \n        return {\n            'total_users_with_privacy_settings': total_users,\n            'privacy_level_distribution': privacy_level_distribution,\n            'average_privacy_score': round(average_privacy_score, 2),\n            'total_consent_decisions': total_consent_decisions,\n            'consent_approval_rate': round(consent_approval_rate, 1),\n            'total_data_categories': len(DataCategory),\n            'total_sharing_contexts': len(SharingContext),\n            'privacy_system_healthy': total_users > 0,\n            'user_sovereignty_enabled': True,\n            'generated_at': datetime.utcnow().isoformat()\n        }"