"""
Privacy-Aware Logging System

Logging system that prioritizes privacy, consent, and ethical data practices
while providing necessary observability for healing-focused systems.
"""

import asyncio
import logging
import json
import hashlib
import re
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
import threading
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Log levels with healing context"""
    HEALING = "HEALING"                                     # Healing events and progress
    CARE = "CARE"                                          # Care interactions and support
    SAFETY = "SAFETY"                                      # Safety and crisis events
    GROWTH = "GROWTH"                                      # Growth and learning events
    COMMUNITY = "COMMUNITY"                                # Community interactions
    SYSTEM = "SYSTEM"                                      # System operational events
    WARNING = "WARNING"                                    # Warning events
    ERROR = "ERROR"                                        # Error events
    CRITICAL = "CRITICAL"                                  # Critical system events


class PrivacyClassification(Enum):
    """Privacy classification for log entries"""
    PUBLIC = "public"                                      # Publicly shareable
    INTERNAL = "internal"                                  # Internal use only
    SENSITIVE = "sensitive"                                # Contains sensitive information
    CONFIDENTIAL = "confidential"                          # Highly confidential
    ANONYMOUS = "anonymous"                                # Anonymized data
    AGGREGATED = "aggregated"                              # Aggregated data only


class DataSensitivity(Enum):
    """Data sensitivity levels"""
    LOW = "low"                                            # General system information
    MEDIUM = "medium"                                      # User interaction data
    HIGH = "high"                                          # Personal information
    CRITICAL = "critical"                                  # Highly sensitive personal data


@dataclass
class SensitiveDataPattern:
    """Pattern for detecting sensitive data"""
    pattern_id: str
    regex_pattern: str
    data_type: str
    sensitivity_level: DataSensitivity
    replacement_strategy: str
    cultural_considerations: List[str]


@dataclass
class LogEntry:
    """Privacy-aware log entry"""
    entry_id: str
    timestamp: datetime
    log_level: LogLevel
    message: str
    context: Dict[str, Any]
    privacy_classification: PrivacyClassification
    data_sensitivity: DataSensitivity
    user_hash: Optional[str]
    cultural_context: List[str]
    healing_context: Dict[str, Any]
    consent_given: bool
    retention_period: timedelta
    anonymization_applied: List[str]
    filtered_fields: List[str]


class SensitiveDataFilter:
    """Filter for detecting and handling sensitive data in logs"""
    
    def __init__(self):
        self.patterns: List[SensitiveDataPattern] = []
        self.cultural_filters: Dict[str, List[Callable]] = {}
        self.replacement_strategies: Dict[str, Callable] = {}
        
        self._setup_default_patterns()
        self._setup_replacement_strategies()
        self._setup_cultural_filters()
    
    def _setup_default_patterns(self):
        """Setup default patterns for sensitive data detection"""
        self.patterns = [
            SensitiveDataPattern(
                pattern_id="email",
                regex_pattern=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                data_type="email_address",
                sensitivity_level=DataSensitivity.HIGH,
                replacement_strategy="hash_preserve_domain",
                cultural_considerations=[]
            ),
            SensitiveDataPattern(
                pattern_id="phone",
                regex_pattern=r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
                data_type="phone_number",
                sensitivity_level=DataSensitivity.HIGH,
                replacement_strategy="partial_mask",
                cultural_considerations=[]
            ),
            SensitiveDataPattern(
                pattern_id="ssn",
                regex_pattern=r'\b\d{3}-\d{2}-\d{4}\b',
                data_type="social_security_number",
                sensitivity_level=DataSensitivity.CRITICAL,
                replacement_strategy="full_redaction",
                cultural_considerations=[]
            ),
            SensitiveDataPattern(
                pattern_id="name",
                regex_pattern=r'\b[A-Z][a-z]+\s[A-Z][a-z]+\b',
                data_type="full_name",
                sensitivity_level=DataSensitivity.HIGH,
                replacement_strategy="initials_only",
                cultural_considerations=["name_structure_varies_by_culture"]
            ),
            SensitiveDataPattern(
                pattern_id="crisis_keywords",
                regex_pattern=r'\b(suicide|self-harm|abuse|crisis|emergency)\b',
                data_type="crisis_content",
                sensitivity_level=DataSensitivity.CRITICAL,
                replacement_strategy="category_replacement",
                cultural_considerations=["crisis_expression_cultural_variation"]
            ),
            SensitiveDataPattern(
                pattern_id="medical_info",
                regex_pattern=r'\b(diagnosis|medication|therapy|treatment|medical)\b',
                data_type="medical_information",
                sensitivity_level=DataSensitivity.CRITICAL,
                replacement_strategy="medical_category",
                cultural_considerations=["medical_terminology_cultural_differences"]
            ),
            SensitiveDataPattern(
                pattern_id="location",
                regex_pattern=r'\b\d{1,5}\s\w+\s(Street|St|Avenue|Ave|Road|Rd|Drive|Dr)\b',
                data_type="street_address",
                sensitivity_level=DataSensitivity.HIGH,
                replacement_strategy="region_generalization",
                cultural_considerations=["address_format_cultural_variation"]
            )
        ]
    
    def _setup_replacement_strategies(self):
        """Setup replacement strategies for different data types"""
        self.replacement_strategies = {
            "hash_preserve_domain": self._hash_preserve_domain,
            "partial_mask": self._partial_mask,
            "full_redaction": self._full_redaction,
            "initials_only": self._initials_only,
            "category_replacement": self._category_replacement,
            "medical_category": self._medical_category,
            "region_generalization": self._region_generalization
        }
    
    def _setup_cultural_filters(self):
        """Setup cultural-specific filtering approaches"""
        self.cultural_filters = {
            'high_privacy_cultures': [
                self._apply_maximum_privacy_protection,
                self._remove_indirect_identifiers
            ],
            'collectivist_cultures': [
                self._protect_family_information,
                self._generalize_group_references
            ],
            'trauma_informed_cultures': [
                self._apply_trauma_sensitive_filtering,
                self._protect_vulnerability_indicators
            ]
        }
    
    async def filter_sensitive_data(self, text: str, cultural_context: List[str] = None) -> Tuple[str, List[str]]:
        """Filter sensitive data from text with cultural awareness"""
        filtered_text = text
        applied_filters = []
        
        # Apply pattern-based filtering
        for pattern in self.patterns:
            if re.search(pattern.regex_pattern, filtered_text, re.IGNORECASE):
                replacement_func = self.replacement_strategies[pattern.replacement_strategy]
                filtered_text = await replacement_func(filtered_text, pattern)
                applied_filters.append(pattern.pattern_id)
        
        # Apply cultural filters
        if cultural_context:
            for culture in cultural_context:
                if culture in self.cultural_filters:
                    for filter_func in self.cultural_filters[culture]:
                        filtered_text = await filter_func(filtered_text)
                        applied_filters.append(f"cultural_{culture}")
        
        return filtered_text, applied_filters
    
    async def _hash_preserve_domain(self, text: str, pattern: SensitiveDataPattern) -> str:
        """Hash email while preserving domain for analytics"""
        def replace_email(match):
            email = match.group(0)
            local, domain = email.split('@')
            hashed_local = hashlib.sha256(local.encode()).hexdigest()[:8]
            return f"{hashed_local}@{domain}"
        
        return re.sub(pattern.regex_pattern, replace_email, text, flags=re.IGNORECASE)
    
    async def _partial_mask(self, text: str, pattern: SensitiveDataPattern) -> str:
        """Partially mask sensitive information"""
        def replace_match(match):
            original = match.group(0)
            if len(original) <= 4:
                return '*' * len(original)
            else:
                return original[:2] + '*' * (len(original) - 4) + original[-2:]
        
        return re.sub(pattern.regex_pattern, replace_match, text)
    
    async def _full_redaction(self, text: str, pattern: SensitiveDataPattern) -> str:
        """Fully redact sensitive information"""
        return re.sub(pattern.regex_pattern, f'[{pattern.data_type.upper()}_REDACTED]', text, flags=re.IGNORECASE)
    
    async def _initials_only(self, text: str, pattern: SensitiveDataPattern) -> str:
        """Replace names with initials only"""
        def replace_name(match):
            name = match.group(0)
            parts = name.split()
            return ' '.join([part[0] + '.' for part in parts if part])
        
        return re.sub(pattern.regex_pattern, replace_name, text)
    
    async def _category_replacement(self, text: str, pattern: SensitiveDataPattern) -> str:
        """Replace with category indicators"""
        category_map = {
            'suicide': '[CRISIS_IDEATION]',
            'self-harm': '[SELF_HARM_CONCERN]',
            'abuse': '[ABUSE_CONCERN]',
            'crisis': '[CRISIS_SITUATION]',
            'emergency': '[EMERGENCY_SITUATION]'
        }
        
        filtered_text = text
        for keyword, replacement in category_map.items():
            filtered_text = re.sub(rf'\b{keyword}\b', replacement, filtered_text, flags=re.IGNORECASE)
        
        return filtered_text
    
    async def _medical_category(self, text: str, pattern: SensitiveDataPattern) -> str:
        """Replace medical information with categories"""
        medical_categories = {
            'diagnosis': '[MEDICAL_DIAGNOSIS]',
            'medication': '[MEDICATION_REFERENCE]',
            'therapy': '[THERAPEUTIC_INTERVENTION]',
            'treatment': '[MEDICAL_TREATMENT]',
            'medical': '[MEDICAL_INFORMATION]'
        }
        
        filtered_text = text
        for term, replacement in medical_categories.items():
            filtered_text = re.sub(rf'\b{term}\b', replacement, filtered_text, flags=re.IGNORECASE)
        
        return filtered_text
    
    async def _region_generalization(self, text: str, pattern: SensitiveDataPattern) -> str:
        """Generalize location to region level"""
        return re.sub(pattern.regex_pattern, '[REGIONAL_LOCATION]', text, flags=re.IGNORECASE)
    
    async def _apply_maximum_privacy_protection(self, text: str) -> str:
        """Apply maximum privacy protection for high-privacy cultures"""
        # Remove any remaining potential identifiers
        identifiers = [r'\b\d+\b', r'\b[A-Z]{2,}\b']  # Numbers and uppercase sequences
        filtered_text = text
        for pattern in identifiers:
            filtered_text = re.sub(pattern, '[IDENTIFIER_REMOVED]', filtered_text)
        return filtered_text
    
    async def _remove_indirect_identifiers(self, text: str) -> str:
        """Remove indirect identifiers that could be used for identification"""
        # Remove specific times, dates, unique sequences
        time_patterns = [r'\b\d{1,2}:\d{2}(:\d{2})?\b', r'\b\d{1,2}/\d{1,2}/\d{4}\b']
        filtered_text = text
        for pattern in time_patterns:
            filtered_text = re.sub(pattern, '[TIME_GENERALIZED]', filtered_text)
        return filtered_text
    
    async def _protect_family_information(self, text: str) -> str:
        """Protect family and group information in collectivist cultures"""
        family_terms = ['family', 'mother', 'father', 'sibling', 'parent', 'child', 'spouse']
        filtered_text = text
        for term in family_terms:
            filtered_text = re.sub(rf'\b{term}\b', '[FAMILY_MEMBER]', filtered_text, flags=re.IGNORECASE)
        return filtered_text
    
    async def _generalize_group_references(self, text: str) -> str:
        """Generalize group references to protect community identity"""
        group_patterns = [r'\b\w+ community\b', r'\b\w+ group\b', r'\b\w+ organization\b']
        filtered_text = text
        for pattern in group_patterns:
            filtered_text = re.sub(pattern, '[COMMUNITY_GROUP]', filtered_text, flags=re.IGNORECASE)
        return filtered_text
    
    async def _apply_trauma_sensitive_filtering(self, text: str) -> str:
        """Apply trauma-sensitive filtering"""
        trauma_indicators = ['trigger', 'flashback', 'ptsd', 'trauma', 'survivor']
        filtered_text = text
        for indicator in trauma_indicators:
            filtered_text = re.sub(rf'\b{indicator}\b', '[TRAUMA_REFERENCE]', filtered_text, flags=re.IGNORECASE)
        return filtered_text
    
    async def _protect_vulnerability_indicators(self, text: str) -> str:
        """Protect indicators of vulnerability"""
        vulnerability_terms = ['vulnerable', 'struggling', 'difficulty', 'challenge', 'suffering']
        filtered_text = text
        for term in vulnerability_terms:
            filtered_text = re.sub(rf'\b{term}\b', '[SUPPORT_NEED]', filtered_text, flags=re.IGNORECASE)
        return filtered_text


class PrivacyAwareLogger:
    """
    Privacy-aware logging system that protects user privacy while maintaining
    necessary observability for healing-focused systems.
    
    Core Principles:
    - Privacy by design in all logging
    - Consent-based data collection
    - Cultural sensitivity in data handling
    - Minimal data collection for maximum insight
    - Automatic sensitive data detection and filtering
    - Configurable retention policies
    - Healing-focused log categorization
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.log_directory = Path(config.get('log_directory', './logs'))
        self.log_directory.mkdir(exist_ok=True)
        
        # Core components
        self.sensitive_data_filter = SensitiveDataFilter()
        self.log_entries: deque = deque(maxlen=10000)
        self.consent_tracker: Dict[str, Dict[str, bool]] = defaultdict(dict)
        self.retention_policies: Dict[LogLevel, timedelta] = {}
        self.cultural_adaptations: Dict[str, Any] = {}
        
        # Privacy settings
        self.default_privacy_classification = PrivacyClassification.INTERNAL
        self.auto_anonymization = config.get('auto_anonymization', True)
        self.consent_required_levels = config.get('consent_required_levels', [LogLevel.HEALING, LogLevel.CARE])
        
        # Logging infrastructure
        self.loggers: Dict[LogLevel, logging.Logger] = {}
        self.handlers: Dict[str, logging.Handler] = {}
        
        # Initialize logging system
        self._setup_retention_policies()
        self._setup_cultural_adaptations()
        self._setup_loggers()
        self._setup_background_tasks()
    
    def _setup_retention_policies(self):
        """Setup data retention policies by log level"""
        self.retention_policies = {
            LogLevel.HEALING: timedelta(days=365),      # Long retention for healing journey tracking
            LogLevel.CARE: timedelta(days=180),         # Medium retention for care effectiveness
            LogLevel.SAFETY: timedelta(days=730),       # Extended retention for safety analysis
            LogLevel.GROWTH: timedelta(days=90),        # Short retention for growth metrics
            LogLevel.COMMUNITY: timedelta(days=30),     # Short retention for community events
            LogLevel.SYSTEM: timedelta(days=30),        # Short retention for system logs
            LogLevel.WARNING: timedelta(days=90),       # Medium retention for warnings
            LogLevel.ERROR: timedelta(days=365),        # Long retention for error analysis
            LogLevel.CRITICAL: timedelta(days=730)      # Extended retention for critical events
        }
    
    def _setup_cultural_adaptations(self):
        """Setup cultural adaptations for logging"""
        self.cultural_adaptations = {
            'high_privacy_cultures': {
                'default_classification': PrivacyClassification.CONFIDENTIAL,
                'auto_anonymization': True,
                'consent_requirement': 'explicit_per_log',
                'retention_reduction': 0.5  # Reduce retention periods by 50%
            },
            'collectivist_cultures': {
                'group_privacy_protection': True,
                'family_information_sensitivity': 'high',
                'community_consent_preferred': True
            },
            'individualist_cultures': {
                'individual_consent_priority': True,
                'personal_data_emphasis': True
            },
            'trauma_informed_cultures': {
                'trauma_sensitive_logging': True,
                'trigger_content_filtering': True,
                'healing_context_prioritization': True
            }
        }
    
    def _setup_loggers(self):
        """Setup specialized loggers for different log levels"""
        for log_level in LogLevel:
            logger_name = f"heart_protocol.{log_level.value.lower()}"
            level_logger = logging.getLogger(logger_name)
            level_logger.setLevel(logging.INFO)
            
            # Create privacy-aware handler
            handler = self._create_privacy_aware_handler(log_level)
            level_logger.addHandler(handler)
            
            self.loggers[log_level] = level_logger
            self.handlers[log_level.value] = handler
    
    def _create_privacy_aware_handler(self, log_level: LogLevel) -> logging.Handler:
        """Create privacy-aware log handler"""
        log_file = self.log_directory / f"{log_level.value.lower()}.log"
        
        # Use rotating file handler to manage log size
        from logging.handlers import RotatingFileHandler
        handler = RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB per file
            backupCount=5
        )
        
        # Custom formatter that applies privacy filtering
        formatter = PrivacyAwareFormatter(self.sensitive_data_filter)
        handler.setFormatter(formatter)
        
        return handler
    
    def _setup_background_tasks(self):
        """Setup background tasks for maintenance"""
        # Start background task for log cleanup
        asyncio.create_task(self._cleanup_expired_logs())
    
    async def log_healing_event(self, message: str, user_id: Optional[str] = None,
                              context: Optional[Dict[str, Any]] = None,
                              cultural_context: Optional[List[str]] = None,
                              healing_context: Optional[Dict[str, Any]] = None) -> str:
        """Log a healing-related event with privacy protection"""
        return await self._log_event(
            LogLevel.HEALING,
            message,
            user_id=user_id,
            context=context,
            cultural_context=cultural_context,
            healing_context=healing_context
        )
    
    async def log_care_interaction(self, message: str, user_id: Optional[str] = None,
                                 context: Optional[Dict[str, Any]] = None,
                                 cultural_context: Optional[List[str]] = None) -> str:
        """Log a care interaction with privacy protection"""
        return await self._log_event(
            LogLevel.CARE,
            message,
            user_id=user_id,
            context=context,
            cultural_context=cultural_context
        )
    
    async def log_safety_event(self, message: str, user_id: Optional[str] = None,
                             context: Optional[Dict[str, Any]] = None,
                             cultural_context: Optional[List[str]] = None) -> str:
        """Log a safety-related event with privacy protection"""
        return await self._log_event(
            LogLevel.SAFETY,
            message,
            user_id=user_id,
            context=context,
            cultural_context=cultural_context,
            privacy_classification=PrivacyClassification.CONFIDENTIAL  # Safety events are highly sensitive
        )
    
    async def log_growth_event(self, message: str, user_id: Optional[str] = None,
                             context: Optional[Dict[str, Any]] = None,
                             cultural_context: Optional[List[str]] = None) -> str:
        """Log a growth-related event with privacy protection"""
        return await self._log_event(
            LogLevel.GROWTH,
            message,
            user_id=user_id,
            context=context,
            cultural_context=cultural_context
        )
    
    async def log_community_event(self, message: str, context: Optional[Dict[str, Any]] = None,
                                cultural_context: Optional[List[str]] = None) -> str:
        """Log a community event with privacy protection"""
        return await self._log_event(
            LogLevel.COMMUNITY,
            message,
            context=context,
            cultural_context=cultural_context,
            privacy_classification=PrivacyClassification.INTERNAL
        )
    
    async def log_system_event(self, message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Log a system event"""
        return await self._log_event(
            LogLevel.SYSTEM,
            message,
            context=context,
            privacy_classification=PrivacyClassification.INTERNAL
        )
    
    async def _log_event(self, log_level: LogLevel, message: str,
                       user_id: Optional[str] = None,
                       context: Optional[Dict[str, Any]] = None,
                       cultural_context: Optional[List[str]] = None,
                       healing_context: Optional[Dict[str, Any]] = None,
                       privacy_classification: Optional[PrivacyClassification] = None) -> str:
        """Core logging method with comprehensive privacy protection"""
        try:
            # Check consent if user is identified
            if user_id and log_level in self.consent_required_levels:
                if not await self._check_logging_consent(user_id, log_level):
                    logger.info(f"Logging consent not given for user {user_id}, level {log_level}")
                    return None
            
            # Apply cultural adaptations
            adapted_privacy_classification = await self._apply_cultural_privacy_adaptations(
                privacy_classification or self.default_privacy_classification,
                cultural_context or []
            )
            
            # Filter sensitive data from message
            filtered_message, applied_filters = await self.sensitive_data_filter.filter_sensitive_data(
                message, cultural_context
            )
            
            # Filter sensitive data from context
            filtered_context = await self._filter_context_data(context or {}, cultural_context or [])
            filtered_healing_context = await self._filter_context_data(healing_context or {}, cultural_context or [])
            
            # Anonymize user identifier
            user_hash = await self._anonymize_user_id(user_id) if user_id else None
            
            # Determine data sensitivity
            data_sensitivity = await self._assess_data_sensitivity(
                filtered_message, filtered_context, log_level
            )
            
            # Create log entry
            entry_id = f"{log_level.value}_{int(datetime.now().timestamp())}_{hash(filtered_message)}"
            log_entry = LogEntry(
                entry_id=entry_id,
                timestamp=datetime.now(),
                log_level=log_level,
                message=filtered_message,
                context=filtered_context,
                privacy_classification=adapted_privacy_classification,
                data_sensitivity=data_sensitivity,
                user_hash=user_hash,
                cultural_context=cultural_context or [],
                healing_context=filtered_healing_context,
                consent_given=True if user_id else False,
                retention_period=self.retention_policies.get(log_level, timedelta(days=30)),
                anonymization_applied=applied_filters,
                filtered_fields=[]  # Would track which fields were filtered
            )
            
            # Store log entry
            self.log_entries.append(log_entry)
            
            # Write to appropriate logger
            level_logger = self.loggers.get(log_level)
            if level_logger:
                log_data = {
                    'entry_id': entry_id,
                    'timestamp': log_entry.timestamp.isoformat(),
                    'level': log_level.value,
                    'message': filtered_message,
                    'context': filtered_context,
                    'privacy_classification': adapted_privacy_classification.value,
                    'cultural_context': cultural_context,
                    'healing_context': filtered_healing_context,
                    'anonymization_applied': applied_filters
                }
                level_logger.info(json.dumps(log_data))
            
            return entry_id
            
        except Exception as e:
            # Use standard logger for logging system errors
            logger.error(f"Error in privacy-aware logging: {e}")
            return None
    
    async def _check_logging_consent(self, user_id: str, log_level: LogLevel) -> bool:
        """Check if user has given consent for this type of logging"""
        user_consents = self.consent_tracker.get(user_id, {})
        return user_consents.get(log_level.value, False)
    
    async def _apply_cultural_privacy_adaptations(self, base_classification: PrivacyClassification,
                                                cultural_context: List[str]) -> PrivacyClassification:
        """Apply cultural adaptations to privacy classification"""
        adapted_classification = base_classification
        
        for culture in cultural_context:
            if culture in self.cultural_adaptations:
                adaptation = self.cultural_adaptations[culture]
                
                if 'default_classification' in adaptation:
                    # Use more restrictive classification
                    cultural_classification = adaptation['default_classification']
                    if cultural_classification.value == 'confidential':
                        adapted_classification = PrivacyClassification.CONFIDENTIAL
        
        return adapted_classification
    
    async def _filter_context_data(self, context: Dict[str, Any], 
                                 cultural_context: List[str]) -> Dict[str, Any]:
        """Filter sensitive data from context dictionary"""
        filtered_context = {}
        
        for key, value in context.items():
            if isinstance(value, str):
                filtered_value, _ = await self.sensitive_data_filter.filter_sensitive_data(
                    value, cultural_context
                )
                filtered_context[key] = filtered_value
            elif isinstance(value, dict):
                filtered_context[key] = await self._filter_context_data(value, cultural_context)
            elif isinstance(value, list):
                filtered_list = []
                for item in value:
                    if isinstance(item, str):
                        filtered_item, _ = await self.sensitive_data_filter.filter_sensitive_data(
                            item, cultural_context
                        )
                        filtered_list.append(filtered_item)
                    else:
                        filtered_list.append(item)
                filtered_context[key] = filtered_list
            else:
                filtered_context[key] = value
        
        return filtered_context
    
    async def _anonymize_user_id(self, user_id: str) -> str:
        """Create anonymized but consistent user identifier"""
        return hashlib.sha256(f"{user_id}_heart_protocol_logging_salt".encode()).hexdigest()[:16]
    
    async def _assess_data_sensitivity(self, message: str, context: Dict[str, Any], 
                                     log_level: LogLevel) -> DataSensitivity:
        """Assess the sensitivity level of the data being logged"""
        # Log level based sensitivity
        level_sensitivity = {
            LogLevel.HEALING: DataSensitivity.HIGH,
            LogLevel.CARE: DataSensitivity.HIGH,
            LogLevel.SAFETY: DataSensitivity.CRITICAL,
            LogLevel.GROWTH: DataSensitivity.MEDIUM,
            LogLevel.COMMUNITY: DataSensitivity.LOW,
            LogLevel.SYSTEM: DataSensitivity.LOW,
            LogLevel.WARNING: DataSensitivity.MEDIUM,
            LogLevel.ERROR: DataSensitivity.MEDIUM,
            LogLevel.CRITICAL: DataSensitivity.HIGH
        }
        
        base_sensitivity = level_sensitivity.get(log_level, DataSensitivity.MEDIUM)
        
        # Content-based sensitivity elevation
        sensitive_keywords = ['crisis', 'trauma', 'abuse', 'suicide', 'medical', 'diagnosis']
        if any(keyword in message.lower() for keyword in sensitive_keywords):
            return DataSensitivity.CRITICAL
        
        return base_sensitivity
    
    async def _cleanup_expired_logs(self):
        """Background task to clean up expired log entries"""
        while True:
            try:
                current_time = datetime.now()
                
                # Clean up in-memory log entries
                to_remove = []
                for entry in self.log_entries:
                    if current_time - entry.timestamp > entry.retention_period:
                        to_remove.append(entry)
                
                for entry in to_remove:
                    self.log_entries.remove(entry)
                
                # Clean up log files would be implemented here
                
                # Sleep for an hour before next cleanup
                await asyncio.sleep(3600)
                
            except Exception as e:
                logger.error(f"Error in log cleanup: {e}")
                await asyncio.sleep(3600)
    
    async def set_user_logging_consent(self, user_id: str, log_level: LogLevel, consent: bool):
        """Set user consent for specific log level"""
        self.consent_tracker[user_id][log_level.value] = consent
        logger.info(f"Logging consent set for user {user_id}, level {log_level.value}: {consent}")
    
    async def get_privacy_report(self) -> Dict[str, Any]:
        """Generate privacy compliance report for logging"""
        total_entries = len(self.log_entries)
        
        privacy_classifications = defaultdict(int)
        sensitivity_levels = defaultdict(int)
        consent_given = 0
        anonymization_applied = 0
        
        for entry in self.log_entries:
            privacy_classifications[entry.privacy_classification.value] += 1
            sensitivity_levels[entry.data_sensitivity.value] += 1
            if entry.consent_given:
                consent_given += 1
            if entry.anonymization_applied:
                anonymization_applied += 1
        
        return {
            'total_log_entries': total_entries,
            'privacy_classification_distribution': dict(privacy_classifications),
            'data_sensitivity_distribution': dict(sensitivity_levels),
            'consent_rate': consent_given / total_entries if total_entries > 0 else 0,
            'anonymization_rate': anonymization_applied / total_entries if total_entries > 0 else 0,
            'cultural_adaptations_active': len(self.cultural_adaptations),
            'sensitive_data_patterns_monitored': len(self.sensitive_data_filter.patterns),
            'retention_policies_configured': len(self.retention_policies),
            'report_generated': datetime.now().isoformat()
        }
    
    async def search_logs(self, query: Dict[str, Any], 
                        requester_permissions: List[str]) -> List[LogEntry]:
        """Search logs with privacy-aware access control"""
        results = []
        
        for entry in self.log_entries:
            # Check if requester has permission to access this log level
            if entry.privacy_classification.value not in requester_permissions:
                continue
            
            # Apply search filters
            if 'log_level' in query and entry.log_level.value != query['log_level']:
                continue
            
            if 'time_range' in query:
                start_time, end_time = query['time_range']
                if not (start_time <= entry.timestamp <= end_time):
                    continue
            
            if 'cultural_context' in query:
                if not any(ctx in entry.cultural_context for ctx in query['cultural_context']):
                    continue
            
            results.append(entry)
        
        return results
    
    def get_logger(self, log_level: LogLevel) -> logging.Logger:
        """Get logger for specific log level"""
        return self.loggers.get(log_level, self.loggers[LogLevel.SYSTEM])


class PrivacyAwareFormatter(logging.Formatter):
    """Custom formatter that applies privacy filtering to log records"""
    
    def __init__(self, sensitive_data_filter: SensitiveDataFilter):
        super().__init__()
        self.sensitive_data_filter = sensitive_data_filter
    
    def format(self, record):
        """Format log record with privacy filtering"""
        # Apply sensitivity filtering to the message
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            filtered_msg, _ = asyncio.create_task(
                self.sensitive_data_filter.filter_sensitive_data(record.msg)
            ).result() if asyncio.get_event_loop().is_running() else (record.msg, [])
            record.msg = filtered_msg
        
        return super().format(record)