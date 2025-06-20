"""
Heart Protocol Feed API Server

Flask application serving caring feeds through AT Protocol compatible endpoints.
Designed to integrate with Bluesky and other AT Protocol networks.
"""

import asyncio
from flask import Flask, request, jsonify, g
from flask_cors import CORS
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import json
import os

from ...core.base import FeedType, HeartProtocolError
from ...core.feed_generators import FeedGeneratorManager, ExperimentManager, AlgorithmUtils
from ..cache import CacheManager
from ..preferences import UserPreferencesManager
from .middleware import setup_middleware
from .endpoints import register_feed_endpoints
from .validation import validate_feed_request

logger = logging.getLogger(__name__)


class FeedAPI:
    """
    Heart Protocol Feed API
    
    Provides AT Protocol compatible endpoints for caring feed generation.
    Integrates A/B testing, caching, and user preferences.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.app = Flask(__name__)
        
        # Initialize core components
        self.feed_manager = FeedGeneratorManager(config.get('feed_manager', {}))
        self.experiment_manager = ExperimentManager()
        self.algorithm_utils = AlgorithmUtils(self.experiment_manager)
        self.cache_manager = CacheManager(config.get('cache', {}))
        self.preferences_manager = UserPreferencesManager(config.get('preferences', {}))
        
        # Performance tracking
        self.request_count = 0
        self.total_response_time = 0.0
        self.error_count = 0
        
        # Configure Flask app
        self._setup_app()
        
        logger.info("Heart Protocol Feed API initialized")
    
    def _setup_app(self):
        """Configure Flask application"""
        # Basic configuration
        self.app.config['SECRET_KEY'] = self.config.get('secret_key', 'heart-protocol-dev-key')
        self.app.config['JSON_SORT_KEYS'] = False
        
        # CORS configuration
        CORS(self.app, origins=self.config.get('cors_origins', ['*']))
        
        # Setup middleware
        setup_middleware(self.app, self.config)
        
        # Register endpoints
        register_feed_endpoints(self.app, self)
        
        # Error handlers
        self._register_error_handlers()
    
    def _register_error_handlers(self):
        """Register error handlers for graceful error responses"""
        
        @self.app.errorhandler(HeartProtocolError)
        def handle_heart_protocol_error(error):
            """Handle Heart Protocol specific errors"""
            logger.error(f"Heart Protocol error: {error}")
            return jsonify({
                'error': 'heart_protocol_error',
                'message': str(error),
                'caring_message': 'We encountered an issue while trying to care for you. Please try again.',
                'timestamp': datetime.utcnow().isoformat()
            }), 500
        
        @self.app.errorhandler(400)
        def handle_bad_request(error):
            """Handle bad requests with caring message"""
            return jsonify({
                'error': 'invalid_request',
                'message': 'The request was not formatted correctly',
                'caring_message': 'It looks like there was an issue with your request. No worries, please try again.',
                'timestamp': datetime.utcnow().isoformat(),
                'help': 'Check the API documentation for correct request format'
            }), 400
        
        @self.app.errorhandler(404)
        def handle_not_found(error):
            """Handle not found errors"""
            return jsonify({
                'error': 'not_found',
                'message': 'The requested resource was not found',
                'caring_message': 'The feed you\'re looking for doesn\'t exist, but we have other caring feeds available.',
                'timestamp': datetime.utcnow().isoformat(),
                'available_feeds': [feed_type.value for feed_type in FeedType]
            }), 404
        
        @self.app.errorhandler(500)
        def handle_internal_error(error):
            """Handle internal server errors"""
            logger.error(f"Internal server error: {error}")
            self.error_count += 1
            
            return jsonify({
                'error': 'internal_error',
                'message': 'An internal error occurred',
                'caring_message': 'Something went wrong on our end. Our care team has been notified and will fix this.',
                'timestamp': datetime.utcnow().isoformat(),
                'support_contact': self.config.get('support_email', 'support@monarch-care.org')
            }), 500
    
    async def generate_feed(self, feed_type: FeedType, user_id: str, 
                           cursor: Optional[str] = None, limit: int = 50) -> Dict[str, Any]:
        """
        Generate a caring feed for the user.
        Integrates A/B testing, caching, and user preferences.
        """
        start_time = datetime.utcnow()
        
        try:
            # Validate user and feed type
            if not await self._validate_user_access(user_id, feed_type):
                raise HeartProtocolError(f"User {user_id} not authorized for {feed_type.value}")
            
            # Get user preferences
            user_preferences = await self.preferences_manager.get_user_preferences(user_id)
            
            # Check cache first
            cache_key = f"feed:{feed_type.value}:{user_id}:{cursor}:{limit}"
            cached_result = await self.cache_manager.get(cache_key)
            
            if cached_result and not user_preferences.get('disable_caching', False):
                logger.debug(f"Serving cached feed for {user_id}")
                await self._record_cache_hit(feed_type, user_id)
                return cached_result
            
            # Get algorithm configuration (potentially from A/B test)
            base_config = self.config.get('algorithms', {}).get(feed_type.value, {})
            algorithm_config = await self.algorithm_utils.get_algorithm_config_for_user(
                user_id, base_config
            )
            
            # Generate feed
            feed_skeleton = await self.feed_manager.generate_feed(
                feed_type, user_id, cursor, limit
            )
            
            # Enhance response with caring metadata
            response = {
                'cursor': feed_skeleton.cursor,
                'feed': feed_skeleton.feed,
                'metadata': {
                    'feed_type': feed_type.value,
                    'generated_at': datetime.utcnow().isoformat(),
                    'user_care_level': await self._assess_user_care_level(user_id),
                    'algorithm_version': algorithm_config.get('version', '1.0'),
                    'caring_message': await self._get_caring_message(feed_type, len(feed_skeleton.feed))
                }
            }
            
            # Cache the result
            cache_ttl = user_preferences.get('cache_ttl', 3600)  # 1 hour default
            await self.cache_manager.set(cache_key, response, ttl=cache_ttl)
            
            # Record metrics
            await self._record_feed_generation(feed_type, user_id, len(feed_skeleton.feed), start_time)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating feed: {e}")
            self.error_count += 1
            raise HeartProtocolError(f"Failed to generate {feed_type.value} feed: {str(e)}")
    
    async def _validate_user_access(self, user_id: str, feed_type: FeedType) -> bool:
        """Validate that user has access to requested feed type"""
        try:
            user_preferences = await self.preferences_manager.get_user_preferences(user_id)
            
            # Check if user has opted out of this feed type
            disabled_feeds = user_preferences.get('disabled_feeds', [])
            if feed_type.value in disabled_feeds:
                return False
            
            # Check if user meets any special requirements for this feed
            special_requirements = {
                FeedType.HEARTS_SEEKING_LIGHT: lambda prefs: prefs.get('allow_helper_matching', True),
                FeedType.GUARDIAN_ENERGY_RISING: lambda prefs: prefs.get('allow_celebration_content', True)
            }
            
            if feed_type in special_requirements:
                if not special_requirements[feed_type](user_preferences):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating user access: {e}")
            return True  # Fail open for user access
    
    async def _assess_user_care_level(self, user_id: str) -> str:
        """Assess user's current care level for metadata"""
        try:
            # This would integrate with care detection system
            # For now, return based on user preferences
            user_preferences = await self.preferences_manager.get_user_preferences(user_id)
            
            if user_preferences.get('needs_gentle_approach', False):
                return 'gentle_support'
            elif user_preferences.get('crisis_support_enabled', True):
                return 'standard_care'
            else:
                return 'minimal_intervention'
                
        except Exception as e:
            logger.error(f"Error assessing user care level: {e}")
            return 'standard_care'
    
    async def _get_caring_message(self, feed_type: FeedType, feed_length: int) -> str:
        """Get a caring message to include with the feed"""
        messages = {
            FeedType.DAILY_GENTLE_REMINDERS: [
                "Remember: you are worthy of love and care 💙",
                "Today's reminders are chosen especially for you",
                "You matter more than you know"
            ],
            FeedType.HEARTS_SEEKING_LIGHT: [
                "You're not alone in this journey",
                "Community care is here when you need it",
                "Helpers are standing by with open hearts"
            ],
            FeedType.GUARDIAN_ENERGY_RISING: [
                "Celebrating the strength in our community",
                "Your healing journey inspires others",
                "Together we rise and heal"
            ],
            FeedType.COMMUNITY_WISDOM: [
                "Wisdom from hearts that understand",
                "Learning from our collective healing",
                "Community insights to guide your path"
            ]
        }
        
        feed_messages = messages.get(feed_type, ["Caring content curated for you"])
        
        if feed_length == 0:
            return "No new content right now, but you're still valued and cared for 💙"
        elif feed_length == 1:
            return f"{feed_messages[0]} - One special item waiting for you"
        else:
            return f"{feed_messages[0]} - {feed_length} caring items selected for you"
    
    async def _record_cache_hit(self, feed_type: FeedType, user_id: str):
        """Record cache hit for performance metrics"""
        # In production, this would update metrics systems
        logger.debug(f"Cache hit for {feed_type.value} feed for user {user_id[:8]}...")
    
    async def _record_feed_generation(self, feed_type: FeedType, user_id: str, 
                                    feed_length: int, start_time: datetime):
        """Record feed generation metrics"""
        end_time = datetime.utcnow()
        response_time = (end_time - start_time).total_seconds()
        
        self.request_count += 1
        self.total_response_time += response_time
        
        # Log performance metrics
        logger.info(f"Generated {feed_type.value} feed: {feed_length} items in {response_time:.3f}s")
        
        # Record A/B testing metrics if applicable
        await self.algorithm_utils.record_feed_interaction(
            user_id=user_id,
            feed_type=feed_type,
            interaction_type='feed_generation',
            outcome_value=1.0 if feed_length > 0 else 0.5,  # Success metric
            metadata={
                'feed_length': feed_length,
                'response_time': response_time,
                'cached': False
            }
        )
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get API health status"""
        avg_response_time = (self.total_response_time / self.request_count 
                           if self.request_count > 0 else 0)
        
        return {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'metrics': {
                'total_requests': self.request_count,
                'total_errors': self.error_count,
                'error_rate': self.error_count / max(self.request_count, 1),
                'average_response_time': avg_response_time
            },
            'components': {
                'feed_manager': 'healthy',
                'cache_manager': 'healthy',
                'preferences_manager': 'healthy',
                'experiment_manager': 'healthy'
            },
            'caring_message': 'All systems caring for you properly 💙'
        }


def create_app(config: Optional[Dict[str, Any]] = None) -> Flask:
    """Create and configure Heart Protocol Feed API application"""
    if config is None:
        config = {
            'cache': {'type': 'memory'},
            'preferences': {'type': 'memory'},
            'cors_origins': ['*'],
            'debug': os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
        }
    
    api = FeedAPI(config)
    
    @api.app.route('/health')
    def health_check():
        """Health check endpoint"""
        return jsonify(api.get_health_status())
    
    @api.app.route('/')
    def index():
        """API information endpoint"""
        return jsonify({
            'name': 'Heart Protocol Feed API',
            'version': '1.0.0',
            'description': 'Caring algorithms that serve love, not extract from it',
            'caring_message': 'Welcome to feeds designed to support human flourishing 💙',
            'endpoints': {
                'feeds': '/xrpc/app.bsky.feed.getFeedSkeleton',
                'health': '/health',
                'preferences': '/api/v1/preferences',
                'metrics': '/api/v1/metrics'
            },
            'philosophy': 'Technology that helps people choose light, never forces them to see it'
        })
    
    return api.app


if __name__ == '__main__':
    # Development server
    app = create_app()
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 8080)),
        debug=os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    )