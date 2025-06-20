"""
Privacy-Preserving Care Analysis

Implements privacy-by-design principles for analyzing user content
while protecting sensitive information and maintaining user agency.
"""

import hashlib
import hmac
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PrivacySettings:
    """User privacy preferences for care analysis"""
    user_id: str
    care_analysis_enabled: bool = False
    data_retention_days: int = 30
    sharing_with_helpers: bool = False
    crisis_intervention_enabled: bool = True
    anonymized_research_participation: bool = False
    last_updated: datetime = None
    
    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.utcnow()


class PrivacyPreservingAnalyzer:
    """
    Handles privacy-preserving analysis of user content for care detection.
    
    Key principles:
    - Minimal data collection
    - User consent for all analysis
    - Immediate anonymization
    - Differential privacy for research
    - User control over data retention
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.encryption_key = config.get('encryption_key', 'default-key-change-in-production')
        self.user_privacy_settings = {}  # In production, this would be a database
        
        # Default privacy settings
        self.default_settings = PrivacySettings(
            user_id="",
            care_analysis_enabled=False,  # Opt-in by default
            data_retention_days=30,
            sharing_with_helpers=False,
            crisis_intervention_enabled=True,  # Always enabled for safety
            anonymized_research_participation=False
        )
    
    async def check_analysis_consent(self, user_id: str) -> bool:
        """
        Check if user has consented to care analysis of their posts.
        
        Returns True only if user has explicitly opted in.
        """
        settings = await self.get_user_privacy_settings(user_id)
        return settings.care_analysis_enabled
    
    async def get_user_privacy_settings(self, user_id: str) -> PrivacySettings:
        """Get user's privacy settings, creating defaults if none exist"""
        if user_id not in self.user_privacy_settings:
            settings = PrivacySettings(user_id=user_id)
            self.user_privacy_settings[user_id] = settings
            return settings
        
        return self.user_privacy_settings[user_id]
    
    async def update_privacy_settings(self, user_id: str, settings: PrivacySettings) -> bool:
        """Update user's privacy settings"""
        try:
            settings.last_updated = datetime.utcnow()
            self.user_privacy_settings[user_id] = settings
            logger.info(f"Updated privacy settings for user {self._hash_user_id(user_id)}")
            return True
        except Exception as e:
            logger.error(f"Failed to update privacy settings: {e}")
            return False
    
    async def anonymize_for_analysis(self, post_content: str, user_id: str) -> Optional[str]:
        """
        Anonymize post content for care analysis while preserving
        emotional and linguistic patterns needed for detection.
        """
        try:
            # Check consent first
            if not await self.check_analysis_consent(user_id):
                return None
            
            # Remove personal identifiers
            anonymized_content = await self._remove_personal_identifiers(post_content)
            
            # Apply differential privacy noise if configured
            if self.config.get('differential_privacy', False):
                anonymized_content = await self._apply_differential_privacy(anonymized_content)
            
            return anonymized_content
            
        except Exception as e:
            logger.error(f"Error anonymizing content: {e}")
            return None
    
    async def _remove_personal_identifiers(self, content: str) -> str:
        """
        Remove or mask personal identifiers while preserving emotional content.
        """
        import re
        
        # Patterns to remove/mask
        patterns = [
            # Names (simple approach - could be improved with NER)
            (r'@[\w]+', '[MENTION]'),
            (r'my name is \w+', 'my name is [NAME]'),
            
            # Locations
            (r'\b\d{1,5}\s+\w+\s+(?:street|st|avenue|ave|road|rd|drive|dr|lane|ln|boulevard|blvd)\b', '[ADDRESS]'),
            (r'\b[A-Z][a-z]+,\s*[A-Z]{2}\b', '[CITY_STATE]'),
            
            # Phone numbers
            (r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]'),
            
            # Email addresses
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]'),
            
            # URLs
            (r'https?://[^\s]+', '[URL]'),
            
            # Specific personal details
            (r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', '[DATE]'),  # Birth dates
            (r'\bSSN\s*:?\s*\d{3}-?\d{2}-?\d{4}\b', '[SSN]'),
        ]
        
        anonymized = content
        for pattern, replacement in patterns:
            anonymized = re.sub(pattern, replacement, anonymized, flags=re.IGNORECASE)
        
        return anonymized
    
    async def _apply_differential_privacy(self, content: str) -> str:
        """
        Apply differential privacy techniques to add noise while preserving
        the essential emotional and linguistic patterns.
        """
        # Simplified differential privacy implementation
        # In production, this would use more sophisticated techniques
        
        # For now, we just add some semantic noise by occasionally
        # replacing words with synonyms or slightly altering structure
        # This is a placeholder - real differential privacy for text is complex
        
        return content  # Placeholder implementation
    
    async def create_anonymized_research_record(self, 
                                               analysis_result: Dict,
                                               user_id: str) -> Optional[Dict]:
        """
        Create anonymized record for research purposes if user has consented.
        """
        settings = await self.get_user_privacy_settings(user_id)
        
        if not settings.anonymized_research_participation:
            return None
        
        # Create fully anonymized record
        research_record = {
            'timestamp': datetime.utcnow().isoformat(),
            'care_level': analysis_result.get('care_level'),
            'confidence': analysis_result.get('confidence'),
            'indicators': analysis_result.get('indicators', []),
            'emotional_context': analysis_result.get('emotional_context', {}),
            'intervention_type': analysis_result.get('suggested_response_type'),
            'user_hash': self._hash_user_id(user_id),  # Irreversible hash
            'geographic_region': 'anonymized',  # Could be broad region if needed
            'age_range': 'anonymized',  # Could be age range if needed
        }
        
        return research_record
    
    def _hash_user_id(self, user_id: str) -> str:
        """Create irreversible hash of user ID for logging/research"""
        return hashlib.sha256(
            (user_id + self.encryption_key).encode()
        ).hexdigest()[:16]
    
    async def schedule_data_deletion(self, user_id: str, data_type: str) -> bool:
        """
        Schedule deletion of user data based on their retention preferences.
        """
        try:
            settings = await self.get_user_privacy_settings(user_id)
            deletion_date = datetime.utcnow() + timedelta(days=settings.data_retention_days)
            
            # In production, this would add to a deletion queue
            logger.info(f"Scheduled deletion of {data_type} for user {self._hash_user_id(user_id)} on {deletion_date}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to schedule data deletion: {e}")
            return False
    
    async def handle_data_deletion_request(self, user_id: str) -> bool:
        """
        Handle immediate data deletion request (GDPR compliance).
        """
        try:
            # Remove from privacy settings
            if user_id in self.user_privacy_settings:
                del self.user_privacy_settings[user_id]
            
            # In production, this would trigger deletion from all systems
            logger.info(f"Processed deletion request for user {self._hash_user_id(user_id)}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to process deletion request: {e}")
            return False
    
    async def audit_data_usage(self, user_id: str) -> Dict[str, Any]:
        """
        Provide audit trail of how user's data has been used.
        """
        try:
            settings = await self.get_user_privacy_settings(user_id)
            
            audit_record = {
                'user_id_hash': self._hash_user_id(user_id),
                'privacy_settings': {
                    'care_analysis_enabled': settings.care_analysis_enabled,
                    'data_retention_days': settings.data_retention_days,
                    'sharing_with_helpers': settings.sharing_with_helpers,
                    'crisis_intervention_enabled': settings.crisis_intervention_enabled,
                    'research_participation': settings.anonymized_research_participation
                },
                'last_updated': settings.last_updated.isoformat(),
                'data_usage_summary': {
                    'care_analyses_performed': 0,  # Would be tracked in production
                    'crisis_interventions': 0,
                    'helper_connections_facilitated': 0,
                    'research_records_created': 0
                }
            }
            
            return audit_record
            
        except Exception as e:
            logger.error(f"Failed to generate audit record: {e}")
            return {}
    
    async def get_privacy_dashboard(self, user_id: str) -> Dict[str, Any]:
        """
        Generate user-friendly privacy dashboard showing current settings
        and data usage.
        """
        settings = await self.get_user_privacy_settings(user_id)
        audit = await self.audit_data_usage(user_id)
        
        return {
            'current_settings': {
                'care_analysis': 'Enabled' if settings.care_analysis_enabled else 'Disabled',
                'data_retention': f"{settings.data_retention_days} days",
                'helper_connections': 'Enabled' if settings.sharing_with_helpers else 'Disabled',
                'crisis_support': 'Always enabled for safety',
                'research_participation': 'Enabled' if settings.anonymized_research_participation else 'Disabled'
            },
            'your_data_usage': audit.get('data_usage_summary', {}),
            'your_rights': [
                'You can disable care analysis at any time',
                'You can request all your data be deleted',
                'You can see exactly how your data is used',
                'You control who can help you',
                'Crisis support is always available regardless of settings'
            ],
            'last_updated': settings.last_updated.isoformat()
        }