"""
Heart Protocol API Validation

Request validation for caring feed APIs with user-friendly error messages.
"""

import re
from typing import Dict, Any, List
from urllib.parse import urlparse
import logging

from ...core.base import FeedType

logger = logging.getLogger(__name__)


def validate_feed_request(feed_uri: str, cursor: str, limit: int) -> Dict[str, Any]:
    """
    Validate feed generation request with caring error messages.
    """
    errors = []
    
    # Validate feed URI
    if not feed_uri:
        errors.append("Feed URI is required")
    elif not _is_valid_feed_uri(feed_uri):
        errors.append(f"Invalid feed URI format: {feed_uri}")
    
    # Validate cursor (if provided)
    if cursor and not _is_valid_cursor(cursor):
        errors.append("Invalid cursor format")
    
    # Validate limit
    if not isinstance(limit, int):
        errors.append("Limit must be an integer")
    elif limit < 1:
        errors.append("Limit must be at least 1")
    elif limit > 100:
        errors.append("Limit cannot exceed 100 (to prevent overwhelming you)")
    
    return {
        'valid': len(errors) == 0,
        'error': '; '.join(errors) if errors else None,
        'caring_message': _get_caring_validation_message(errors) if errors else None
    }


def validate_preferences_request(preferences: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate user preferences update request.
    """
    errors = []
    
    if not isinstance(preferences, dict):
        errors.append("Preferences must be a JSON object")
        return {
            'valid': False,
            'error': 'Preferences must be a JSON object',
            'caring_message': 'Please send your preferences as a properly formatted JSON object'
        }
    
    # Validate specific preference fields
    if 'disabled_feeds' in preferences:
        if not _validate_disabled_feeds(preferences['disabled_feeds']):
            errors.append("Invalid disabled_feeds format")
    
    if 'care_preferences' in preferences:
        if not _validate_care_preferences(preferences['care_preferences']):
            errors.append("Invalid care_preferences format")
    
    if 'privacy_settings' in preferences:
        if not _validate_privacy_settings(preferences['privacy_settings']):
            errors.append("Invalid privacy_settings format")
    
    if 'notification_preferences' in preferences:
        if not _validate_notification_preferences(preferences['notification_preferences']):
            errors.append("Invalid notification_preferences format")
    
    # Validate boolean fields
    boolean_fields = [
        'personalization_enabled', 'needs_gentle_approach', 'crisis_support_enabled',
        'allow_helper_matching', 'allow_celebration_content', 'disable_caching'
    ]
    
    for field in boolean_fields:
        if field in preferences and not isinstance(preferences[field], bool):
            errors.append(f"{field} must be true or false")
    
    # Validate numeric fields
    if 'cache_ttl' in preferences:
        ttl = preferences['cache_ttl']
        if not isinstance(ttl, int) or ttl < 60 or ttl > 86400:
            errors.append("cache_ttl must be between 60 and 86400 seconds")
    
    return {
        'valid': len(errors) == 0,
        'error': '; '.join(errors) if errors else None,
        'caring_message': _get_caring_preferences_message(errors) if errors else None
    }


def validate_feedback_request(feedback: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate feedback submission request.
    """
    errors = []
    
    # Required fields
    required_fields = ['feed_type', 'rating', 'feedback_type']
    for field in required_fields:
        if field not in feedback:
            errors.append(f"Missing required field: {field}")
    
    # Validate feed_type
    if 'feed_type' in feedback:
        try:
            FeedType(feedback['feed_type'])
        except ValueError:
            errors.append(f"Invalid feed_type: {feedback['feed_type']}")
    
    # Validate rating
    if 'rating' in feedback:
        rating = feedback['rating']
        if not isinstance(rating, (int, float)) or rating < 1 or rating > 5:
            errors.append("Rating must be between 1 and 5")
    
    # Validate feedback_type
    if 'feedback_type' in feedback:
        valid_types = ['feed_satisfaction', 'positive_feedback', 'negative_feedback', 'suggestion']
        if feedback['feedback_type'] not in valid_types:
            errors.append(f"feedback_type must be one of: {valid_types}")
    
    # Validate optional comment length
    if 'comment' in feedback:
        comment = feedback['comment']
        if not isinstance(comment, str):
            errors.append("Comment must be text")
        elif len(comment) > 1000:
            errors.append("Comment must be less than 1000 characters")
    
    return {
        'valid': len(errors) == 0,
        'error': '; '.join(errors) if errors else None,
        'caring_message': _get_caring_feedback_message(errors) if errors else None
    }


def _is_valid_feed_uri(feed_uri: str) -> bool:
    """Validate AT Protocol feed URI format"""
    try:
        # Basic URI validation
        parsed = urlparse(feed_uri)
        if parsed.scheme != 'at':
            return False
        
        # Check for expected feed URI pattern
        # Expected: at://did:plc:monarch/app.bsky.feed.generator/FEED_TYPE
        parts = feed_uri.split('/')
        if len(parts) < 4:
            return False
        
        # Check for valid feed type in URI
        valid_feed_identifiers = [
            'daily-gentle-reminders',
            'hearts-seeking-light', 
            'guardian-energy-rising',
            'community-wisdom'
        ]
        
        return any(identifier in feed_uri for identifier in valid_feed_identifiers)
        
    except Exception:
        return False


def _is_valid_cursor(cursor: str) -> bool:
    """Validate cursor format"""
    try:
        # Cursor should be a reasonable length string
        if not isinstance(cursor, str):
            return False
        
        if len(cursor) > 500:  # Reasonable max length
            return False
        
        # Basic format check (alphanumeric, hyphens, underscores)
        if not re.match(r'^[a-zA-Z0-9_-]+$', cursor):
            return False
        
        return True
        
    except Exception:
        return False


def _validate_disabled_feeds(disabled_feeds: Any) -> bool:
    """Validate disabled_feeds list"""
    if not isinstance(disabled_feeds, list):
        return False
    
    valid_feed_types = [ft.value for ft in FeedType]
    
    for feed_type in disabled_feeds:
        if not isinstance(feed_type, str) or feed_type not in valid_feed_types:
            return False
    
    return True


def _validate_care_preferences(care_prefs: Any) -> bool:
    """Validate care_preferences object"""
    if not isinstance(care_prefs, dict):
        return False
    
    # Validate trigger_warnings
    if 'trigger_warnings' in care_prefs:
        triggers = care_prefs['trigger_warnings']
        if not isinstance(triggers, list):
            return False
        if not all(isinstance(trigger, str) for trigger in triggers):
            return False
    
    # Validate care_types
    if 'care_types' in care_prefs:
        care_types = care_prefs['care_types']
        if not isinstance(care_types, list):
            return False
        valid_care_types = ['gentle_reminders', 'peer_support', 'resources', 'crisis_support']
        if not all(ct in valid_care_types for ct in care_types):
            return False
    
    # Validate max_care_posts_per_day
    if 'max_care_posts_per_day' in care_prefs:
        max_posts = care_prefs['max_care_posts_per_day']
        if not isinstance(max_posts, int) or max_posts < 0 or max_posts > 50:
            return False
    
    return True


def _validate_privacy_settings(privacy_settings: Any) -> bool:
    """Validate privacy_settings object"""
    if not isinstance(privacy_settings, dict):
        return False
    
    # Validate boolean privacy fields
    boolean_fields = [
        'care_analysis_enabled', 'sharing_with_helpers', 'anonymized_research_participation'
    ]
    
    for field in boolean_fields:
        if field in privacy_settings and not isinstance(privacy_settings[field], bool):
            return False
    
    # Validate data_retention_days
    if 'data_retention_days' in privacy_settings:
        retention = privacy_settings['data_retention_days']
        if not isinstance(retention, int) or retention < 1 or retention > 365:
            return False
    
    return True


def _validate_notification_preferences(notif_prefs: Any) -> bool:
    """Validate notification_preferences object"""
    if not isinstance(notif_prefs, dict):
        return False
    
    # Validate notification types
    if 'enabled_notifications' in notif_prefs:
        enabled = notif_prefs['enabled_notifications']
        if not isinstance(enabled, list):
            return False
        
        valid_types = ['gentle_reminders', 'crisis_support', 'helper_connections', 'feed_updates']
        if not all(nt in valid_types for nt in enabled):
            return False
    
    # Validate quiet_hours
    if 'quiet_hours' in notif_prefs:
        quiet_hours = notif_prefs['quiet_hours']
        if not isinstance(quiet_hours, dict):
            return False
        
        if 'start' in quiet_hours and 'end' in quiet_hours:
            start = quiet_hours['start']
            end = quiet_hours['end']
            
            # Validate time format (HH:MM)
            time_pattern = r'^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$'
            if not re.match(time_pattern, start) or not re.match(time_pattern, end):
                return False
    
    return True


def _get_caring_validation_message(errors: List[str]) -> str:
    """Get caring error message for validation failures"""
    if not errors:
        return None
    
    if len(errors) == 1:
        error = errors[0].lower()
        if 'uri' in error:
            return "The feed you're looking for seems to have an incorrect address. Please check the feed link."
        elif 'limit' in error:
            return "We want to make sure you get the right amount of caring content - please adjust your request size."
        elif 'cursor' in error:
            return "There seems to be an issue with loading more content. Please try refreshing."
        else:
            return "There's a small issue with your request. Please double-check and try again."
    else:
        return "There are a few issues with your request. Please review the details and try again - we're here to help!"


def _get_caring_preferences_message(errors: List[str]) -> str:
    """Get caring error message for preferences validation failures"""
    if not errors:
        return None
    
    return "We want to respect your preferences exactly, but there are some formatting issues. Please check your settings and try again."


def _get_caring_feedback_message(errors: List[str]) -> str:
    """Get caring error message for feedback validation failures"""
    if not errors:
        return None
    
    return "We really appreciate your feedback! There are just a few formatting issues to fix before we can process it."