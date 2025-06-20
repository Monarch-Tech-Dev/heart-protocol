"""
User Preference Management System

Comprehensive preference management that respects user autonomy and choice
while providing personalized, healing-focused experiences.
"""

import asyncio
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import logging
import json

logger = logging.getLogger(__name__)


class PreferenceCategory(Enum):
    """Categories of user preferences"""
    PRIVACY_CONTROL = "privacy_control"           # Privacy and data control
    CONTENT_PERSONALIZATION = "content_personalization"  # Content preferences
    COMMUNICATION_STYLE = "communication_style"   # How system communicates
    SAFETY_MONITORING = "safety_monitoring"       # Safety and crisis settings
    NOTIFICATION_SETTINGS = "notification_settings"  # Notification preferences
    ACCESSIBILITY = "accessibility"               # Accessibility options
    CULTURAL_ADAPTATION = "cultural_adaptation"   # Cultural preferences
    HEALING_JOURNEY = "healing_journey"          # Healing-specific settings
    CACHE_CONTROL = "cache_control"              # Data retention preferences
    INTERACTION_PREFERENCES = "interaction_preferences"  # Interaction style


class PreferenceType(Enum):
    """Types of preference values"""
    BOOLEAN = "boolean"                          # True/False
    INTEGER = "integer"                          # Numeric integer
    FLOAT = "float"                             # Numeric float
    STRING = "string"                           # Text string
    ENUM = "enum"                              # Enumerated choice
    LIST = "list"                              # List of values
    OBJECT = "object"                          # Complex object
    DURATION = "duration"                      # Time duration


@dataclass
class PreferenceDefinition:
    """Definition of a user preference"""
    preference_id: str
    category: PreferenceCategory
    preference_type: PreferenceType
    display_name: str
    description: str
    default_value: Any
    possible_values: Optional[List[Any]] = None
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    user_editable: bool = True
    requires_consent: bool = False
    healing_impact: str = ""
    privacy_implications: str = ""
    dependency_preferences: List[str] = field(default_factory=list)


@dataclass
class UserPreference:
    """User's specific preference setting"""
    user_id: str
    preference_id: str
    value: Any
    set_at: datetime
    last_modified: datetime
    source: str  # 'user_set', 'system_default', 'adaptive_learning'
    consent_given: bool = True
    locked: bool = False  # Prevents system from changing
    notes: str = ""


@dataclass
class PreferenceGroup:
    """Grouped preferences for easier management"""
    group_id: str
    group_name: str
    description: str
    preferences: List[str]  # List of preference_ids
    category: PreferenceCategory
    user_friendly_explanation: str
    healing_focus_area: str


class PreferenceManager:
    """
    Comprehensive preference management system that empowers users with
    granular control over their Heart Protocol experience.
    
    Core Principles:
    - User autonomy and choice paramount
    - Transparent about data usage
    - Healing-focused recommendations
    - Privacy by design
    - Accessible to all users
    - Cultural sensitivity
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Preference definitions and user settings
        self.preference_definitions = {}      # preference_id -> PreferenceDefinition
        self.user_preferences = {}            # user_id -> Dict[preference_id, UserPreference]
        self.preference_groups = {}           # group_id -> PreferenceGroup
        
        # Change tracking and analytics
        self.preference_change_history = {}  # user_id -> List[change_events]
        self.preference_analytics = {}        # Analytics about preference usage
        
        # Validation and consent management
        self.validation_rules = {}            # preference_id -> validation_functions
        self.consent_requirements = {}        # preference_id -> consent_details
        
        # System learning and adaptation
        self.adaptive_recommendations = {}    # user_id -> recommended_changes
        self.healing_impact_tracking = {}     # preference_id -> impact_metrics
        
        self._initialize_core_preferences()
        self._initialize_preference_groups()
        self._initialize_validation_rules()
        
        logger.info("Preference Manager initialized with user-centric control")
    
    def _initialize_core_preferences(self) -> None:
        """Initialize core preference definitions"""
        
        # Privacy Control Preferences
        self.preference_definitions['data_retention_days'] = PreferenceDefinition(
            preference_id='data_retention_days',
            category=PreferenceCategory.PRIVACY_CONTROL,
            preference_type=PreferenceType.INTEGER,
            display_name='Data Retention Period',
            description='How long to keep your data in our systems',
            default_value=365,  # 1 year default
            possible_values=[7, 30, 90, 180, 365, 730, -1],  # -1 means indefinite
            validation_rules={'min': 7, 'max': 3650},  # 7 days to 10 years
            user_editable=True,
            requires_consent=True,
            healing_impact="Longer retention allows better personalization but reduces privacy",
            privacy_implications="Shorter periods increase privacy, longer periods improve experience",
            dependency_preferences=['cache_policy', 'analytics_participation']
        )
        
        self.preference_definitions['cache_policy'] = PreferenceDefinition(
            preference_id='cache_policy',
            category=PreferenceCategory.CACHE_CONTROL,
            preference_type=PreferenceType.ENUM,
            display_name='Data Caching Policy',
            description='How aggressively to cache your data for performance',
            default_value='balanced',
            possible_values=['minimal', 'balanced', 'performance', 'user_controlled'],
            user_editable=True,
            requires_consent=False,
            healing_impact="Performance caching enables faster, more responsive care",
            privacy_implications="Less caching increases privacy, more caching improves performance"
        )
        
        self.preference_definitions['analytics_participation'] = PreferenceDefinition(
            preference_id='analytics_participation',
            category=PreferenceCategory.PRIVACY_CONTROL,
            preference_type=PreferenceType.ENUM,
            display_name='Analytics Participation',
            description='How much to participate in system improvement analytics',
            default_value='anonymized_only',
            possible_values=['none', 'anonymized_only', 'aggregated_only', 'full_consent'],
            user_editable=True,
            requires_consent=True,
            healing_impact="Analytics help improve the system for everyone's healing",
            privacy_implications="Higher participation improves system but may affect privacy"
        )
        
        # Content Personalization Preferences
        self.preference_definitions['content_personalization_level'] = PreferenceDefinition(
            preference_id='content_personalization_level',
            category=PreferenceCategory.CONTENT_PERSONALIZATION,
            preference_type=PreferenceType.ENUM,
            display_name='Content Personalization Level',
            description='How much to personalize content to your specific needs',
            default_value='moderate',
            possible_values=['minimal', 'moderate', 'high', 'adaptive'],
            user_editable=True,
            requires_consent=False,
            healing_impact="Personalization creates more relevant, healing-focused content",
            privacy_implications="Higher personalization requires more data analysis"
        )
        
        self.preference_definitions['content_sensitivity_level'] = PreferenceDefinition(
            preference_id='content_sensitivity_level',
            category=PreferenceCategory.CONTENT_PERSONALIZATION,
            preference_type=PreferenceType.ENUM,
            display_name='Content Sensitivity Level',
            description='How sensitive content filtering should be',
            default_value='moderate',
            possible_values=['low', 'moderate', 'high', 'maximum'],
            user_editable=True,
            requires_consent=False,
            healing_impact="Appropriate sensitivity creates safer healing spaces",
            privacy_implications="Higher sensitivity may limit content variety"
        )
        
        self.preference_definitions['trigger_content_filtering'] = PreferenceDefinition(
            preference_id='trigger_content_filtering',
            category=PreferenceCategory.CONTENT_PERSONALIZATION,
            preference_type=PreferenceType.BOOLEAN,
            display_name='Trigger Content Filtering',
            description='Filter potentially triggering content based on your profile',
            default_value=True,
            user_editable=True,
            requires_consent=False,
            healing_impact="Filtering creates safer spaces for healing and growth",
            privacy_implications="Requires analysis of your trigger patterns"
        )
        
        # Communication Style Preferences
        self.preference_definitions['communication_warmth'] = PreferenceDefinition(
            preference_id='communication_warmth',
            category=PreferenceCategory.COMMUNICATION_STYLE,
            preference_type=PreferenceType.ENUM,
            display_name='Communication Warmth Level',
            description='How warm and personal you want system communication',
            default_value='warm',
            possible_values=['minimal', 'gentle', 'warm', 'very_warm'],
            user_editable=True,
            requires_consent=False,
            healing_impact="Appropriate warmth creates connection and comfort",
            privacy_implications="No significant privacy implications"
        )
        
        self.preference_definitions['communication_formality'] = PreferenceDefinition(
            preference_id='communication_formality',
            category=PreferenceCategory.COMMUNICATION_STYLE,
            preference_type=PreferenceType.ENUM,
            display_name='Communication Formality',
            description='How formal or casual you want system communication',
            default_value='casual',
            possible_values=['very_formal', 'formal', 'casual', 'very_casual'],
            user_editable=True,
            requires_consent=False,
            healing_impact="Appropriate formality increases comfort and trust",
            privacy_implications="No significant privacy implications"
        )
        
        self.preference_definitions['preferred_communication_style'] = PreferenceDefinition(
            preference_id='preferred_communication_style',
            category=PreferenceCategory.COMMUNICATION_STYLE,
            preference_type=PreferenceType.ENUM,
            display_name='Preferred Communication Style',
            description='Your preferred style of interaction with the system',
            default_value='warm_companion',
            possible_values=[
                'warm_companion', 'gentle_guide', 'quiet_presence', 
                'encouraging_cheerleader', 'wise_elder', 'trauma_informed_therapist'
            ],
            user_editable=True,
            requires_consent=False,
            healing_impact="Matching communication style improves connection and healing",
            privacy_implications="No significant privacy implications"
        )
        
        # Safety Monitoring Preferences
        self.preference_definitions['safety_monitoring_level'] = PreferenceDefinition(
            preference_id='safety_monitoring_level',
            category=PreferenceCategory.SAFETY_MONITORING,
            preference_type=PreferenceType.ENUM,
            display_name='Safety Monitoring Level',
            description='How actively to monitor for safety concerns',
            default_value='standard',
            possible_values=['minimal', 'standard', 'enhanced', 'intensive'],
            user_editable=True,
            requires_consent=True,
            healing_impact="Appropriate monitoring provides safety while respecting autonomy",
            privacy_implications="Higher monitoring requires more data analysis"
        )
        
        self.preference_definitions['crisis_escalation_preference'] = PreferenceDefinition(
            preference_id='crisis_escalation_preference',
            category=PreferenceCategory.SAFETY_MONITORING,
            preference_type=PreferenceType.ENUM,
            display_name='Crisis Escalation Preference',
            description='How you want crisis situations to be handled',
            default_value='allow_with_consent',
            possible_values=['never', 'allow_with_consent', 'proactive', 'immediate_when_needed'],
            user_editable=True,
            requires_consent=True,
            healing_impact="Appropriate escalation can be life-saving in crisis",
            privacy_implications="May involve sharing data with crisis responders"
        )
        
        self.preference_definitions['emergency_contact_sharing'] = PreferenceDefinition(
            preference_id='emergency_contact_sharing',
            category=PreferenceCategory.SAFETY_MONITORING,
            preference_type=PreferenceType.BOOLEAN,
            display_name='Emergency Contact Sharing',
            description='Allow sharing your emergency contacts in crisis situations',
            default_value=False,
            user_editable=True,
            requires_consent=True,
            healing_impact="Emergency contacts can provide crucial support in crisis",
            privacy_implications="Shares personal contact information in emergencies only"
        )
        
        # Cultural Adaptation Preferences
        self.preference_definitions['cultural_background'] = PreferenceDefinition(
            preference_id='cultural_background',
            category=PreferenceCategory.CULTURAL_ADAPTATION,
            preference_type=PreferenceType.LIST,
            display_name='Cultural Background',
            description='Your cultural backgrounds for appropriate content adaptation',
            default_value=[],
            user_editable=True,
            requires_consent=False,
            healing_impact="Cultural adaptation creates more relevant and respectful experiences",
            privacy_implications="Cultural information helps personalization but reveals identity aspects"
        )
        
        self.preference_definitions['preferred_language'] = PreferenceDefinition(
            preference_id='preferred_language',
            category=PreferenceCategory.CULTURAL_ADAPTATION,
            preference_type=PreferenceType.STRING,
            display_name='Preferred Language',
            description='Your preferred language for system communication',
            default_value='English',
            possible_values=['English', 'Spanish', 'French', 'German', 'Italian', 'Portuguese'],
            user_editable=True,
            requires_consent=False,
            healing_impact="Native language communication improves understanding and comfort",
            privacy_implications="Language preference may indicate geographic location"
        )
        
        self.preference_definitions['religious_spiritual_considerations'] = PreferenceDefinition(
            preference_id='religious_spiritual_considerations',
            category=PreferenceCategory.CULTURAL_ADAPTATION,
            preference_type=PreferenceType.BOOLEAN,
            display_name='Religious/Spiritual Considerations',
            description='Include religious and spiritual considerations in content',
            default_value=False,
            user_editable=True,
            requires_consent=False,
            healing_impact="Spiritual integration can deepen healing for many people",
            privacy_implications="Spiritual preferences reveal personal beliefs"
        )
        
        # Healing Journey Preferences
        self.preference_definitions['healing_focus_areas'] = PreferenceDefinition(
            preference_id='healing_focus_areas',
            category=PreferenceCategory.HEALING_JOURNEY,
            preference_type=PreferenceType.LIST,
            display_name='Healing Focus Areas',
            description='Areas of healing you want to focus on',
            default_value=[],
            possible_values=[
                'trauma_recovery', 'anxiety_management', 'depression_support',
                'relationship_healing', 'self_compassion', 'grief_processing',
                'addiction_recovery', 'emotional_regulation', 'stress_management'
            ],
            user_editable=True,
            requires_consent=False,
            healing_impact="Focus areas enable targeted, relevant healing support",
            privacy_implications="Reveals specific mental health and healing needs"
        )
        
        self.preference_definitions['healing_pace_preference'] = PreferenceDefinition(
            preference_id='healing_pace_preference',
            category=PreferenceCategory.HEALING_JOURNEY,
            preference_type=PreferenceType.ENUM,
            display_name='Healing Pace Preference',
            description='Your preferred pace for healing journey content',
            default_value='your_own_pace',
            possible_values=['gentle_slow', 'your_own_pace', 'steady_progress', 'intensive_when_ready'],
            user_editable=True,
            requires_consent=False,
            healing_impact="Appropriate pacing respects individual healing timelines",
            privacy_implications="No significant privacy implications"
        )
        
        # Accessibility Preferences
        self.preference_definitions['accessibility_needs'] = PreferenceDefinition(
            preference_id='accessibility_needs',
            category=PreferenceCategory.ACCESSIBILITY,
            preference_type=PreferenceType.LIST,
            display_name='Accessibility Needs',
            description='Accessibility accommodations you need',
            default_value=[],
            possible_values=[
                'screen_reader_compatible', 'high_contrast', 'large_text',
                'simple_language', 'audio_alternatives', 'reduced_motion'
            ],
            user_editable=True,
            requires_consent=False,
            healing_impact="Accessibility ensures everyone can access healing resources",
            privacy_implications="Accessibility needs may indicate specific disabilities"
        )
        
        # Cache Control Preferences
        self.preference_definitions['cache_ttl_hours'] = PreferenceDefinition(
            preference_id='cache_ttl_hours',
            category=PreferenceCategory.CACHE_CONTROL,
            preference_type=PreferenceType.INTEGER,
            display_name='Cache Time-To-Live (Hours)',
            description='How long to cache your data for quick access',
            default_value=24,
            possible_values=[1, 6, 12, 24, 48, 72, 168],  # 1 hour to 1 week
            validation_rules={'min': 1, 'max': 168},
            user_editable=True,
            requires_consent=False,
            healing_impact="Longer cache improves responsiveness but reduces data freshness",
            privacy_implications="Longer cache keeps data in memory longer"
        )
        
        self.preference_definitions['cache_sensitive_data'] = PreferenceDefinition(
            preference_id='cache_sensitive_data',
            category=PreferenceCategory.CACHE_CONTROL,
            preference_type=PreferenceType.BOOLEAN,
            display_name='Cache Sensitive Data',
            description='Allow caching of sensitive personal information',
            default_value=False,
            user_editable=True,
            requires_consent=True,
            healing_impact="Caching sensitive data improves personalization speed",
            privacy_implications="Caching sensitive data increases privacy risk but improves experience"
        )
    
    def _initialize_preference_groups(self) -> None:
        """Initialize logical preference groups for easier user management"""
        
        self.preference_groups['privacy_fundamentals'] = PreferenceGroup(
            group_id='privacy_fundamentals',
            group_name='Privacy Fundamentals',
            description='Core privacy and data control settings',
            preferences=[
                'data_retention_days', 'analytics_participation', 
                'cache_policy', 'cache_sensitive_data'
            ],
            category=PreferenceCategory.PRIVACY_CONTROL,
            user_friendly_explanation="Control how your data is stored, shared, and used",
            healing_focus_area="Building trust through transparency and control"
        )
        
        self.preference_groups['personalization_balance'] = PreferenceGroup(
            group_id='personalization_balance',
            group_name='Personalization Balance',
            description='Balance between personalization and privacy',
            preferences=[
                'content_personalization_level', 'content_sensitivity_level',
                'trigger_content_filtering'
            ],
            category=PreferenceCategory.CONTENT_PERSONALIZATION,
            user_friendly_explanation="Choose how much the system adapts to your specific needs",
            healing_focus_area="Creating safe, relevant healing experiences"
        )
        
        self.preference_groups['communication_comfort'] = PreferenceGroup(
            group_id='communication_comfort',
            group_name='Communication Comfort',
            description='How the system communicates with you',
            preferences=[
                'communication_warmth', 'communication_formality',
                'preferred_communication_style'
            ],
            category=PreferenceCategory.COMMUNICATION_STYLE,
            user_friendly_explanation="Customize how the system talks with you",
            healing_focus_area="Building comfortable, healing-focused connection"
        )
        
        self.preference_groups['safety_autonomy'] = PreferenceGroup(
            group_id='safety_autonomy',
            group_name='Safety & Autonomy',
            description='Balance between safety monitoring and personal autonomy',
            preferences=[
                'safety_monitoring_level', 'crisis_escalation_preference',
                'emergency_contact_sharing'
            ],
            category=PreferenceCategory.SAFETY_MONITORING,
            user_friendly_explanation="Control how the system helps keep you safe",
            healing_focus_area="Providing safety while respecting your choices"
        )
        
        self.preference_groups['cultural_healing'] = PreferenceGroup(
            group_id='cultural_healing',
            group_name='Cultural & Healing Context',
            description='Cultural background and healing journey preferences',
            preferences=[
                'cultural_background', 'preferred_language',
                'religious_spiritual_considerations', 'healing_focus_areas',
                'healing_pace_preference'
            ],
            category=PreferenceCategory.CULTURAL_ADAPTATION,
            user_friendly_explanation="Adapt the system to your cultural background and healing journey",
            healing_focus_area="Honoring your unique healing path and cultural context"
        )
        
        self.preference_groups['accessibility_inclusion'] = PreferenceGroup(
            group_id='accessibility_inclusion',
            group_name='Accessibility & Inclusion',
            description='Accessibility accommodations and inclusive design',
            preferences=['accessibility_needs'],
            category=PreferenceCategory.ACCESSIBILITY,
            user_friendly_explanation="Ensure the system works well for your specific needs",
            healing_focus_area="Creating inclusive healing spaces for everyone"
        )
        
        self.preference_groups['performance_privacy'] = PreferenceGroup(
            group_id='performance_privacy',
            group_name='Performance vs Privacy',
            description='Balance system performance with privacy protection',
            preferences=['cache_ttl_hours', 'cache_policy'],
            category=PreferenceCategory.CACHE_CONTROL,
            user_friendly_explanation="Control how the system optimizes performance while protecting privacy",
            healing_focus_area="Fast, responsive care while maintaining data protection"
        )
    
    def _initialize_validation_rules(self) -> None:
        """Initialize validation rules for preferences"""
        
        self.validation_rules = {
            'data_retention_days': self._validate_data_retention_days,
            'cache_ttl_hours': self._validate_cache_ttl_hours,
            'healing_focus_areas': self._validate_healing_focus_areas,
            'cultural_background': self._validate_cultural_background,
            'accessibility_needs': self._validate_accessibility_needs
        }
    
    async def get_user_preference(self, user_id: str, preference_id: str) -> Any:
        """Get user's preference value"""
        
        try:
            # Check if user has set this preference
            if (user_id in self.user_preferences and 
                preference_id in self.user_preferences[user_id]):
                user_pref = self.user_preferences[user_id][preference_id]
                return user_pref.value
            
            # Return default value if not set
            if preference_id in self.preference_definitions:
                return self.preference_definitions[preference_id].default_value
            
            logger.warning(f"Unknown preference {preference_id} requested for user {user_id[:8]}...")
            return None
            
        except Exception as e:
            logger.error(f"Error getting preference {preference_id} for user {user_id[:8]}...: {e}")
            return None
    
    async def set_user_preference(self, user_id: str, preference_id: str, 
                                 value: Any, source: str = 'user_set') -> bool:
        """Set user's preference value"""
        
        try:
            # Validate preference exists
            if preference_id not in self.preference_definitions:
                logger.error(f"Unknown preference {preference_id}")
                return False
            
            pref_def = self.preference_definitions[preference_id]
            
            # Check if user can edit this preference
            if not pref_def.user_editable and source == 'user_set':
                logger.warning(f"User {user_id[:8]}... attempted to edit non-editable preference {preference_id}")
                return False
            
            # Validate the value
            if not await self._validate_preference_value(preference_id, value, user_id):
                logger.error(f"Invalid value for preference {preference_id}: {value}")
                return False
            
            # Check consent requirements
            if pref_def.requires_consent and source == 'user_set':
                if not await self._verify_preference_consent(user_id, preference_id):
                    logger.warning(f"Consent required for preference {preference_id} but not given")
                    return False
            
            # Initialize user preferences if needed
            if user_id not in self.user_preferences:
                self.user_preferences[user_id] = {}
            
            # Check if preference is locked
            existing_pref = self.user_preferences[user_id].get(preference_id)
            if existing_pref and existing_pref.locked and source != 'user_set':
                logger.debug(f"Preference {preference_id} is locked for user {user_id[:8]}...")
                return False
            
            # Create or update preference
            now = datetime.utcnow()
            user_preference = UserPreference(
                user_id=user_id,
                preference_id=preference_id,
                value=value,
                set_at=existing_pref.set_at if existing_pref else now,
                last_modified=now,
                source=source,
                consent_given=pref_def.requires_consent,
                locked=existing_pref.locked if existing_pref else False,
                notes=""
            )
            
            self.user_preferences[user_id][preference_id] = user_preference
            
            # Track change history
            await self._track_preference_change(user_id, preference_id, value, source)
            
            # Check for dependent preferences
            await self._update_dependent_preferences(user_id, preference_id, value)
            
            logger.info(f"Set preference {preference_id} for user {user_id[:8]}... to {value} (source: {source})")
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting preference {preference_id} for user {user_id[:8]}...: {e}")
            return False
    
    async def get_user_preferences_by_category(self, user_id: str, 
                                             category: PreferenceCategory) -> Dict[str, Any]:
        """Get all user preferences in a category"""
        
        category_prefs = {}
        
        for pref_id, pref_def in self.preference_definitions.items():
            if pref_def.category == category:
                value = await self.get_user_preference(user_id, pref_id)
                category_prefs[pref_id] = {
                    'value': value,
                    'definition': pref_def,
                    'user_set': (user_id in self.user_preferences and 
                               pref_id in self.user_preferences[user_id])
                }
        
        return category_prefs
    
    async def get_preference_group(self, user_id: str, group_id: str) -> Dict[str, Any]:
        """Get all preferences in a preference group"""
        
        if group_id not in self.preference_groups:
            return {}
        
        group = self.preference_groups[group_id]
        group_data = {
            'group_info': group,
            'preferences': {}
        }
        
        for pref_id in group.preferences:
            value = await self.get_user_preference(user_id, pref_id)
            pref_def = self.preference_definitions[pref_id]
            
            group_data['preferences'][pref_id] = {
                'value': value,
                'definition': pref_def,
                'user_set': (user_id in self.user_preferences and 
                           pref_id in self.user_preferences[user_id])
            }
        
        return group_data
    
    async def export_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Export all user preferences in a portable format"""
        
        export_data = {
            'user_id': user_id,
            'exported_at': datetime.utcnow().isoformat(),
            'preferences': {},
            'preference_groups': {}
        }
        
        # Export individual preferences
        if user_id in self.user_preferences:
            for pref_id, user_pref in self.user_preferences[user_id].items():
                export_data['preferences'][pref_id] = {
                    'value': user_pref.value,
                    'set_at': user_pref.set_at.isoformat(),
                    'last_modified': user_pref.last_modified.isoformat(),
                    'source': user_pref.source,
                    'consent_given': user_pref.consent_given,
                    'locked': user_pref.locked,
                    'notes': user_pref.notes
                }
        
        # Export preference groups with current values
        for group_id in self.preference_groups:
            group_data = await self.get_preference_group(user_id, group_id)
            export_data['preference_groups'][group_id] = group_data
        
        return export_data
    
    async def import_user_preferences(self, user_id: str, import_data: Dict[str, Any]) -> bool:
        """Import user preferences from exported data"""
        
        try:
            if 'preferences' not in import_data:
                logger.error("Invalid import data format")
                return False
            
            imported_count = 0
            
            for pref_id, pref_data in import_data['preferences'].items():
                # Validate preference still exists
                if pref_id not in self.preference_definitions:
                    logger.warning(f"Skipping unknown preference {pref_id}")
                    continue
                
                # Import the preference
                success = await self.set_user_preference(
                    user_id, pref_id, pref_data['value'], 'import'
                )
                
                if success:
                    # Restore additional metadata if possible
                    if user_id in self.user_preferences and pref_id in self.user_preferences[user_id]:
                        user_pref = self.user_preferences[user_id][pref_id]
                        user_pref.locked = pref_data.get('locked', False)
                        user_pref.notes = pref_data.get('notes', '')
                        user_pref.consent_given = pref_data.get('consent_given', True)
                    
                    imported_count += 1
            
            logger.info(f"Imported {imported_count} preferences for user {user_id[:8]}...")
            return True
            
        except Exception as e:
            logger.error(f"Error importing preferences for user {user_id[:8]}...: {e}")
            return False
    
    async def reset_user_preferences(self, user_id: str, 
                                   category: Optional[PreferenceCategory] = None) -> bool:
        """Reset user preferences to defaults"""
        
        try:
            if user_id not in self.user_preferences:
                return True  # Nothing to reset
            
            preferences_to_reset = []
            
            if category:
                # Reset only preferences in specified category
                for pref_id, pref_def in self.preference_definitions.items():
                    if pref_def.category == category and pref_id in self.user_preferences[user_id]:
                        preferences_to_reset.append(pref_id)
            else:
                # Reset all preferences
                preferences_to_reset = list(self.user_preferences[user_id].keys())
            
            for pref_id in preferences_to_reset:
                # Don't reset locked preferences
                user_pref = self.user_preferences[user_id][pref_id]
                if user_pref.locked:
                    continue
                
                # Set back to default
                default_value = self.preference_definitions[pref_id].default_value
                await self.set_user_preference(user_id, pref_id, default_value, 'system_reset')
            
            logger.info(f"Reset {len(preferences_to_reset)} preferences for user {user_id[:8]}...")
            return True
            
        except Exception as e:
            logger.error(f"Error resetting preferences for user {user_id[:8]}...: {e}")
            return False
    
    async def get_adaptive_recommendations(self, user_id: str) -> List[Dict[str, Any]]:
        """Get adaptive recommendations for preference improvements"""
        
        recommendations = []
        
        try:
            # Analyze user's current preferences and usage patterns
            user_prefs = self.user_preferences.get(user_id, {})
            
            # Recommendation: Enable analytics for better personalization
            if user_id in user_prefs:
                analytics_pref = user_prefs.get('analytics_participation')
                if analytics_pref and analytics_pref.value == 'none':
                    recommendations.append({
                        'type': 'privacy_balance',
                        'preference_id': 'analytics_participation',
                        'current_value': analytics_pref.value,
                        'recommended_value': 'anonymized_only',
                        'reason': 'Anonymized analytics help improve the system while protecting your privacy',
                        'healing_benefit': 'Better personalized healing experiences for everyone',
                        'privacy_impact': 'minimal'
                    })
            
            # Recommendation: Optimize cache settings
            cache_ttl = await self.get_user_preference(user_id, 'cache_ttl_hours')
            if cache_ttl and cache_ttl < 12:
                recommendations.append({
                    'type': 'performance_optimization',
                    'preference_id': 'cache_ttl_hours',
                    'current_value': cache_ttl,
                    'recommended_value': 24,
                    'reason': 'Longer cache improves system responsiveness',
                    'healing_benefit': 'Faster, more responsive healing support',
                    'privacy_impact': 'low'
                })
            
            # Recommendation: Enable trigger filtering for safety
            trigger_filtering = await self.get_user_preference(user_id, 'trigger_content_filtering')
            if not trigger_filtering:
                recommendations.append({
                    'type': 'safety_enhancement',
                    'preference_id': 'trigger_content_filtering',
                    'current_value': trigger_filtering,
                    'recommended_value': True,
                    'reason': 'Content filtering creates safer healing spaces',
                    'healing_benefit': 'Reduced exposure to potentially triggering content',
                    'privacy_impact': 'minimal'
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating adaptive recommendations for user {user_id[:8]}...: {e}")
            return []
    
    async def _validate_preference_value(self, preference_id: str, value: Any, user_id: str) -> bool:
        """Validate preference value"""
        
        try:
            pref_def = self.preference_definitions[preference_id]
            
            # Type validation
            if pref_def.preference_type == PreferenceType.BOOLEAN:
                if not isinstance(value, bool):
                    return False
            elif pref_def.preference_type == PreferenceType.INTEGER:
                if not isinstance(value, int):
                    return False
            elif pref_def.preference_type == PreferenceType.FLOAT:
                if not isinstance(value, (int, float)):
                    return False
            elif pref_def.preference_type == PreferenceType.STRING:
                if not isinstance(value, str):
                    return False
            elif pref_def.preference_type == PreferenceType.LIST:
                if not isinstance(value, list):
                    return False
            
            # Possible values validation
            if pref_def.possible_values and value not in pref_def.possible_values:
                return False
            
            # Custom validation rules
            validation_rules = pref_def.validation_rules
            if 'min' in validation_rules and value < validation_rules['min']:
                return False
            if 'max' in validation_rules and value > validation_rules['max']:
                return False
            
            # Custom validator
            if preference_id in self.validation_rules:
                validator = self.validation_rules[preference_id]
                return await validator(value, user_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating preference {preference_id} value: {e}")
            return False
    
    async def _verify_preference_consent(self, user_id: str, preference_id: str) -> bool:
        """Verify user has given consent for preference that requires it"""
        
        # In production, this would check actual consent records
        # For now, assume consent is given
        return True
    
    async def _track_preference_change(self, user_id: str, preference_id: str, 
                                     value: Any, source: str) -> None:
        """Track preference change for analytics"""
        
        if user_id not in self.preference_change_history:
            self.preference_change_history[user_id] = []
        
        change_event = {
            'timestamp': datetime.utcnow(),
            'preference_id': preference_id,
            'new_value': value,
            'source': source,
            'user_id': user_id
        }
        
        self.preference_change_history[user_id].append(change_event)
        
        # Keep only recent history
        max_history = 100
        if len(self.preference_change_history[user_id]) > max_history:
            self.preference_change_history[user_id] = self.preference_change_history[user_id][-max_history:]
    
    async def _update_dependent_preferences(self, user_id: str, preference_id: str, value: Any) -> None:
        """Update preferences that depend on this one"""
        
        pref_def = self.preference_definitions[preference_id]
        
        # Check if any other preferences depend on this one
        for other_pref_id, other_pref_def in self.preference_definitions.items():
            if preference_id in other_pref_def.dependency_preferences:
                # Update dependent preference logic would go here
                pass
    
    async def _validate_data_retention_days(self, value: int, user_id: str) -> bool:
        """Validate data retention days value"""
        return 7 <= value <= 3650 or value == -1  # 7 days to 10 years, or indefinite
    
    async def _validate_cache_ttl_hours(self, value: int, user_id: str) -> bool:
        """Validate cache TTL hours value"""
        return 1 <= value <= 168  # 1 hour to 1 week
    
    async def _validate_healing_focus_areas(self, value: List[str], user_id: str) -> bool:
        """Validate healing focus areas"""
        valid_areas = [
            'trauma_recovery', 'anxiety_management', 'depression_support',
            'relationship_healing', 'self_compassion', 'grief_processing',
            'addiction_recovery', 'emotional_regulation', 'stress_management'
        ]
        return all(area in valid_areas for area in value)
    
    async def _validate_cultural_background(self, value: List[str], user_id: str) -> bool:
        """Validate cultural background values"""
        # Accept any cultural background - validation would be more sophisticated in production
        return isinstance(value, list) and len(value) <= 10
    
    async def _validate_accessibility_needs(self, value: List[str], user_id: str) -> bool:
        """Validate accessibility needs"""
        valid_needs = [
            'screen_reader_compatible', 'high_contrast', 'large_text',
            'simple_language', 'audio_alternatives', 'reduced_motion'
        ]
        return all(need in valid_needs for need in value)
    
    async def get_preference_analytics(self) -> Dict[str, Any]:
        """Get analytics about preference usage and patterns"""
        
        total_users = len(self.user_preferences)
        total_preferences_set = sum(len(prefs) for prefs in self.user_preferences.values())
        
        # Calculate most commonly changed preferences
        preference_change_counts = {}
        for user_changes in self.preference_change_history.values():
            for change in user_changes:
                pref_id = change['preference_id']
                preference_change_counts[pref_id] = preference_change_counts.get(pref_id, 0) + 1
        
        # Calculate category distribution
        category_usage = {}
        for user_prefs in self.user_preferences.values():
            for pref_id in user_prefs:
                category = self.preference_definitions[pref_id].category.value
                category_usage[category] = category_usage.get(category, 0) + 1
        
        return {
            'total_users_with_preferences': total_users,
            'total_preferences_set': total_preferences_set,
            'average_preferences_per_user': total_preferences_set / max(total_users, 1),
            'most_changed_preferences': dict(sorted(preference_change_counts.items(), 
                                                   key=lambda x: x[1], reverse=True)[:10]),
            'category_usage_distribution': category_usage,
            'total_preference_definitions': len(self.preference_definitions),
            'total_preference_groups': len(self.preference_groups),
            'user_empowerment_healthy': total_users > 0,
            'generated_at': datetime.utcnow().isoformat()
        }