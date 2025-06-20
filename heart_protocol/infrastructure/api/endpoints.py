"""
Heart Protocol API Endpoints

AT Protocol compatible endpoints for feed generation and user preferences.
"""

import asyncio
from flask import request, jsonify, g
from functools import wraps
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from ...core.base import FeedType, HeartProtocolError
from .validation import validate_feed_request, validate_preferences_request

logger = logging.getLogger(__name__)


def async_route(f):
    """Decorator to handle async functions in Flask routes"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(f(*args, **kwargs))
        finally:
            loop.close()
    return wrapper


def register_feed_endpoints(app, api):
    """Register all feed-related endpoints"""
    
    # AT Protocol Feed Skeleton Endpoint
    @app.route('/xrpc/app.bsky.feed.getFeedSkeleton', methods=['GET'])
    @async_route
    async def get_feed_skeleton():
        """
        AT Protocol compatible feed skeleton endpoint.
        Used by Bluesky and other AT Protocol clients.
        """
        try:
            # Extract and validate parameters
            feed_uri = request.args.get('feed')
            cursor = request.args.get('cursor')
            limit = int(request.args.get('limit', 50))
            
            # Validate request
            validation_result = validate_feed_request(feed_uri, cursor, limit)
            if not validation_result['valid']:
                return jsonify({
                    'error': 'InvalidRequest',
                    'message': validation_result['error'],
                    'caring_message': 'There was an issue with your feed request. Please check the format and try again.'
                }), 400
            
            # Parse feed type from URI
            feed_type = _parse_feed_type_from_uri(feed_uri)
            if not feed_type:
                return jsonify({
                    'error': 'UnsupportedFeed',
                    'message': f'Feed type not supported: {feed_uri}',
                    'caring_message': 'This feed type isn\'t available yet, but we have other caring feeds for you.',
                    'available_feeds': [f'at://did:plc:monarch/{ft.value}' for ft in FeedType]
                }), 404
            
            # Get user ID (in production, this would come from authentication)
            user_id = request.headers.get('X-User-ID', 'anonymous_user')
            
            # Generate feed
            result = await api.generate_feed(feed_type, user_id, cursor, limit)
            
            # Return AT Protocol compatible response
            return jsonify({
                'cursor': result['cursor'],
                'feed': result['feed']
            })
            
        except HeartProtocolError as e:
            logger.error(f"Heart Protocol error in feed generation: {e}")
            return jsonify({
                'error': 'FeedGenerationError',
                'message': str(e),
                'caring_message': 'We had trouble creating your feed. Our care team is looking into it.'
            }), 500
        
        except Exception as e:
            logger.error(f"Unexpected error in feed generation: {e}")
            return jsonify({
                'error': 'InternalError',
                'message': 'An unexpected error occurred',
                'caring_message': 'Something unexpected happened. We\'re working to fix it.'
            }), 500
    
    # Enhanced Feed Endpoint with Metadata
    @app.route('/api/v1/feeds/<feed_type>', methods=['GET'])
    @async_route
    async def get_enhanced_feed(feed_type):
        """
        Enhanced feed endpoint with caring metadata and user preferences.
        Provides additional context beyond AT Protocol requirements.
        """
        try:
            # Parse feed type
            try:
                feed_type_enum = FeedType(feed_type)
            except ValueError:
                return jsonify({
                    'error': 'InvalidFeedType',
                    'message': f'Unknown feed type: {feed_type}',
                    'available_feeds': [ft.value for ft in FeedType]
                }), 400
            
            # Get parameters
            cursor = request.args.get('cursor')
            limit = int(request.args.get('limit', 50))
            include_metadata = request.args.get('include_metadata', 'true').lower() == 'true'
            user_id = request.headers.get('X-User-ID', 'anonymous_user')
            
            # Generate feed with full metadata
            result = await api.generate_feed(feed_type_enum, user_id, cursor, limit)
            
            # Return enhanced response
            response = {
                'cursor': result['cursor'],
                'feed': result['feed']
            }
            
            if include_metadata:
                response['metadata'] = result['metadata']
                response['user_context'] = await _get_user_context(api, user_id)
                response['feed_info'] = await api.feed_manager.get_feed_info(feed_type_enum)
            
            return jsonify(response)
            
        except Exception as e:
            logger.error(f"Error in enhanced feed endpoint: {e}")
            return jsonify({
                'error': 'FeedError',
                'message': str(e),
                'caring_message': 'We had trouble with your enhanced feed request.'
            }), 500
    
    # User Preferences Endpoints
    @app.route('/api/v1/preferences', methods=['GET'])
    @async_route
    async def get_user_preferences():
        """Get user's caring preferences"""
        try:
            user_id = request.headers.get('X-User-ID', 'anonymous_user')
            preferences = await api.preferences_manager.get_user_preferences(user_id)
            
            return jsonify({
                'user_id': user_id,
                'preferences': preferences,
                'last_updated': preferences.get('last_updated'),
                'caring_message': 'Your preferences help us care for you better 💙'
            })
            
        except Exception as e:
            logger.error(f"Error getting user preferences: {e}")
            return jsonify({
                'error': 'PreferencesError',
                'message': str(e)
            }), 500
    
    @app.route('/api/v1/preferences', methods=['POST'])
    @async_route
    async def update_user_preferences():
        """Update user's caring preferences"""
        try:
            user_id = request.headers.get('X-User-ID', 'anonymous_user')
            
            # Validate request
            if not request.is_json:
                return jsonify({
                    'error': 'InvalidRequest',
                    'message': 'Request must be JSON'
                }), 400
            
            new_preferences = request.get_json()
            validation_result = validate_preferences_request(new_preferences)
            
            if not validation_result['valid']:
                return jsonify({
                    'error': 'InvalidPreferences',
                    'message': validation_result['error']
                }), 400
            
            # Update preferences
            success = await api.preferences_manager.update_user_preferences(
                user_id, new_preferences
            )
            
            if success:
                updated_preferences = await api.preferences_manager.get_user_preferences(user_id)
                return jsonify({
                    'success': True,
                    'preferences': updated_preferences,
                    'caring_message': 'Your preferences have been updated. We\'ll care for you accordingly 💙'
                })
            else:
                return jsonify({
                    'error': 'UpdateFailed',
                    'message': 'Failed to update preferences'
                }), 500
                
        except Exception as e:
            logger.error(f"Error updating user preferences: {e}")
            return jsonify({
                'error': 'PreferencesError',
                'message': str(e)
            }), 500
    
    # Feed Performance Metrics
    @app.route('/api/v1/metrics', methods=['GET'])
    @async_route
    async def get_feed_metrics():
        """Get feed performance and caring metrics"""
        try:
            user_id = request.headers.get('X-User-ID')
            
            if user_id:
                # User-specific metrics
                metrics = await _get_user_feed_metrics(api, user_id)
            else:
                # System-wide metrics (for admins)
                metrics = api.get_health_status()
                
                # Add caring-specific metrics
                metrics['caring_metrics'] = {
                    'feeds_generated_today': api.request_count,  # Simplified
                    'positive_outcomes_rate': 0.85,  # Would be calculated from real data
                    'crisis_interventions_today': 0,  # Would be tracked
                    'user_satisfaction_score': 4.2,  # Would be from user feedback
                    'community_connections_facilitated': 42  # Would be tracked
                }
            
            return jsonify(metrics)
            
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return jsonify({
                'error': 'MetricsError',
                'message': str(e)
            }), 500
    
    # Feed Discovery
    @app.route('/api/v1/feeds', methods=['GET'])
    async def discover_feeds():
        """Discover available caring feeds"""
        try:
            user_id = request.headers.get('X-User-ID', 'anonymous_user')
            
            # Get available feeds for user
            available_feeds = await api.feed_manager.get_available_feeds(user_id)
            
            feeds_info = []
            for feed_type in available_feeds:
                feed_info = await api.feed_manager.get_feed_info(feed_type)
                feeds_info.append(feed_info)
            
            return jsonify({
                'available_feeds': feeds_info,
                'total_feeds': len(feeds_info),
                'caring_message': 'These feeds are designed to support your wellbeing 💙',
                'customization_info': 'Each feed can be personalized in your preferences'
            })
            
        except Exception as e:
            logger.error(f"Error in feed discovery: {e}")
            return jsonify({
                'error': 'DiscoveryError',
                'message': str(e)
            }), 500
    
    # Caring Interaction Feedback
    @app.route('/api/v1/feedback', methods=['POST'])
    @async_route
    async def submit_feedback():
        """Submit feedback on feed quality and caring effectiveness"""
        try:
            user_id = request.headers.get('X-User-ID', 'anonymous_user')
            
            if not request.is_json:
                return jsonify({
                    'error': 'InvalidRequest',
                    'message': 'Feedback must be JSON'
                }), 400
            
            feedback = request.get_json()
            required_fields = ['feed_type', 'rating', 'feedback_type']
            
            if not all(field in feedback for field in required_fields):
                return jsonify({
                    'error': 'MissingFields',
                    'message': f'Required fields: {required_fields}'
                }), 400
            
            # Record feedback for A/B testing and improvement
            await api.algorithm_utils.record_feed_interaction(
                user_id=user_id,
                feed_type=FeedType(feedback['feed_type']),
                interaction_type=feedback['feedback_type'],
                outcome_value=feedback['rating'] / 5.0,  # Convert 1-5 to 0-1
                metadata={
                    'feedback_text': feedback.get('comment', ''),
                    'timestamp': datetime.utcnow().isoformat(),
                    'helpful_items': feedback.get('helpful_items', []),
                    'unhelpful_items': feedback.get('unhelpful_items', [])
                }
            )
            
            return jsonify({
                'success': True,
                'caring_message': 'Thank you for helping us care better 💙',
                'message': 'Your feedback helps us improve our caring algorithms'
            })
            
        except Exception as e:
            logger.error(f"Error submitting feedback: {e}")
            return jsonify({
                'error': 'FeedbackError',
                'message': str(e)
            }), 500


def _parse_feed_type_from_uri(feed_uri: str) -> Optional[FeedType]:
    """Parse feed type from AT Protocol feed URI"""
    if not feed_uri:
        return None
    
    # Expected format: at://did:plc:monarch/app.bsky.feed.generator/FEED_TYPE
    try:
        # Extract feed type from URI
        if 'daily-gentle-reminders' in feed_uri:
            return FeedType.DAILY_GENTLE_REMINDERS
        elif 'hearts-seeking-light' in feed_uri:
            return FeedType.HEARTS_SEEKING_LIGHT
        elif 'guardian-energy-rising' in feed_uri:
            return FeedType.GUARDIAN_ENERGY_RISING
        elif 'community-wisdom' in feed_uri:
            return FeedType.COMMUNITY_WISDOM
        else:
            return None
    except Exception:
        return None


async def _get_user_context(api, user_id: str) -> Dict[str, Any]:
    """Get user context for enhanced responses"""
    try:
        preferences = await api.preferences_manager.get_user_preferences(user_id)
        care_level = await api._assess_user_care_level(user_id)
        
        return {
            'care_level': care_level,
            'personalization_enabled': preferences.get('personalization_enabled', True),
            'gentle_mode': preferences.get('needs_gentle_approach', False),
            'crisis_support_available': preferences.get('crisis_support_enabled', True),
            'helper_matching_enabled': preferences.get('allow_helper_matching', True)
        }
    except Exception as e:
        logger.error(f"Error getting user context: {e}")
        return {'error': 'Unable to load user context'}


async def _get_user_feed_metrics(api, user_id: str) -> Dict[str, Any]:
    """Get user-specific feed metrics"""
    try:
        # In production, this would query actual user metrics
        return {
            'user_id': user_id,
            'feeds_accessed_today': 3,
            'favorite_feed_type': 'daily_gentle_reminders',
            'positive_interactions': 8,
            'total_interactions': 10,
            'wellbeing_trend': 'improving',
            'caring_message': 'Your engagement with caring content is encouraging 💙',
            'suggestions': [
                'Consider enabling helper matching for additional support',
                'Your gentle reminders seem most helpful - we\'ll prioritize similar content'
            ]
        }
    except Exception as e:
        logger.error(f"Error getting user metrics: {e}")
        return {'error': 'Unable to load user metrics'}