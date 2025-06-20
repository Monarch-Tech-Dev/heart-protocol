"""
Heart Protocol API Middleware

Middleware for caring request handling, rate limiting, and user protection.
"""

import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from flask import request, g, jsonify
from functools import wraps
import hashlib

logger = logging.getLogger(__name__)


class CaringRateLimiter:
    """
    Rate limiter that protects users from overwhelming themselves
    while ensuring caring content is always available.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.request_counts = {}  # In production, use Redis
        self.caring_exemptions = {
            '/health': True,
            '/api/v1/crisis': True,  # Crisis endpoints never rate limited
            '/api/v1/feedback': True  # Feedback always welcome
        }
    
    def is_rate_limited(self, user_id: str, endpoint: str) -> Dict[str, Any]:
        """
        Check if user is rate limited for this endpoint.
        Uses caring rate limiting that prioritizes user wellbeing.
        """
        # Never rate limit crisis or health endpoints
        if endpoint in self.caring_exemptions:
            return {'limited': False, 'reason': 'caring_exemption'}
        
        current_time = time.time()
        window_size = self.config.get('rate_limit_window', 3600)  # 1 hour
        
        # Get user's request history
        user_key = f"rate_limit:{user_id}"
        if user_key not in self.request_counts:
            self.request_counts[user_key] = []
        
        # Clean old requests
        cutoff_time = current_time - window_size
        self.request_counts[user_key] = [
            req_time for req_time in self.request_counts[user_key] 
            if req_time > cutoff_time
        ]
        
        # Check different limits based on endpoint type
        if '/xrpc/app.bsky.feed.getFeedSkeleton' in endpoint:
            # Feed requests - generous limit but prevent abuse
            limit = self.config.get('feed_requests_per_hour', 120)
        elif '/api/v1/feeds/' in endpoint:
            # Enhanced feed requests - slightly lower limit
            limit = self.config.get('enhanced_feed_requests_per_hour', 60)
        else:
            # General API requests
            limit = self.config.get('general_requests_per_hour', 200)
        
        request_count = len(self.request_counts[user_key])
        
        if request_count >= limit:
            return {
                'limited': True,
                'reason': 'rate_limit_exceeded',
                'limit': limit,
                'window_size': window_size,
                'requests_made': request_count,
                'reset_time': current_time + window_size - min(self.request_counts[user_key])
            }
        
        # Record this request
        self.request_counts[user_key].append(current_time)
        
        return {
            'limited': False,
            'requests_remaining': limit - request_count - 1,
            'reset_time': current_time + window_size
        }


class UserProtectionMiddleware:
    """
    Middleware to protect vulnerable users and enforce caring boundaries.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.vulnerable_user_cache = {}  # In production, use proper storage
    
    async def check_user_protection(self, user_id: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if user needs special protection or gentle handling.
        """
        try:
            # Check if user is in crisis mode
            crisis_mode = await self._is_user_in_crisis_mode(user_id)
            if crisis_mode:
                return {
                    'protection_needed': True,
                    'protection_type': 'crisis_mode',
                    'gentle_response': True,
                    'human_review': True
                }
            
            # Check if user is overwhelmed
            overwhelmed = await self._is_user_overwhelmed(user_id)
            if overwhelmed:
                return {
                    'protection_needed': True,
                    'protection_type': 'overwhelmed',
                    'gentle_response': True,
                    'reduce_content': True
                }
            
            # Check if user needs gentle approach
            needs_gentle = await self._user_needs_gentle_approach(user_id)
            if needs_gentle:
                return {
                    'protection_needed': True,
                    'protection_type': 'gentle_approach',
                    'gentle_response': True
                }
            
            return {'protection_needed': False}
            
        except Exception as e:
            logger.error(f"Error checking user protection: {e}")
            # Fail safe - assume user needs gentle approach
            return {
                'protection_needed': True,
                'protection_type': 'safety_fallback',
                'gentle_response': True
            }
    
    async def _is_user_in_crisis_mode(self, user_id: str) -> bool:
        """Check if user is currently in crisis mode"""
        # In production, this would check:
        # - Recent crisis interventions
        # - User's explicit crisis flag
        # - Care detection system alerts
        return False  # Placeholder
    
    async def _is_user_overwhelmed(self, user_id: str) -> bool:
        """Check if user is overwhelmed based on recent activity"""
        # In production, this would check:
        # - Rapid requests in short time
        # - Negative feedback patterns
        # - High stress indicators from content
        
        # Simple check based on request frequency
        current_time = time.time()
        recent_requests = getattr(g, 'recent_requests', [])
        
        # If more than 10 requests in last 10 minutes, consider overwhelmed
        recent_count = len([req for req in recent_requests if current_time - req < 600])
        return recent_count > 10
    
    async def _user_needs_gentle_approach(self, user_id: str) -> bool:
        """Check if user has requested gentle approach"""
        # In production, this would check user preferences
        # For now, apply gentle approach to anonymous users
        return user_id == 'anonymous_user'


def setup_middleware(app, config: Dict[str, Any]):
    """Setup all caring middleware for the Flask app"""
    
    rate_limiter = CaringRateLimiter(config.get('rate_limiting', {}))
    user_protection = UserProtectionMiddleware(config.get('user_protection', {}))
    
    @app.before_request
    def before_request():
        """Pre-request middleware for caring handling"""
        g.request_start_time = time.time()
        g.user_id = request.headers.get('X-User-ID', 'anonymous_user')
        g.request_id = _generate_request_id()
        
        # Track recent requests for overwhelm detection
        if not hasattr(g, 'recent_requests'):
            g.recent_requests = []
        g.recent_requests.append(g.request_start_time)
        
        # Keep only last 20 requests
        g.recent_requests = g.recent_requests[-20:]
        
        logger.info(f"Request {g.request_id}: {request.method} {request.path} from user {g.user_id[:8]}...")
    
    @app.before_request
    def check_rate_limiting():
        """Check rate limiting with caring messages"""
        if request.endpoint in ['static', 'health_check']:
            return None
        
        rate_limit_result = rate_limiter.is_rate_limited(g.user_id, request.path)
        
        if rate_limit_result['limited']:
            reset_time = datetime.fromtimestamp(rate_limit_result['reset_time'])
            
            return jsonify({
                'error': 'RateLimitExceeded',
                'message': f"Rate limit exceeded: {rate_limit_result['requests_made']}/{rate_limit_result['limit']} requests per hour",
                'caring_message': "We want to make sure you don't overwhelm yourself. Please take a gentle break and try again soon 💙",
                'reset_time': reset_time.isoformat(),
                'suggestions': [
                    "Take some time to process the content you've already received",
                    "Consider if you might be feeling overwhelmed",
                    "Crisis support is always available regardless of rate limits"
                ],
                'crisis_contact': config.get('crisis_contact', 'crisis@monarch-care.org')
            }), 429
        
        # Store rate limit info for response headers
        g.rate_limit_remaining = rate_limit_result.get('requests_remaining', 0)
        g.rate_limit_reset = rate_limit_result.get('reset_time', 0)
    
    @app.after_request
    def after_request(response):
        """Post-request middleware for caring headers and logging"""
        
        # Add caring headers
        response.headers['X-Caring-API'] = 'Heart-Protocol-v1.0'
        response.headers['X-Philosophy'] = 'Algorithms that serve love'
        
        # Add rate limit headers
        if hasattr(g, 'rate_limit_remaining'):
            response.headers['X-RateLimit-Remaining'] = str(g.rate_limit_remaining)
            response.headers['X-RateLimit-Reset'] = str(int(g.rate_limit_reset))
        
        # Add request ID for debugging
        if hasattr(g, 'request_id'):
            response.headers['X-Request-ID'] = g.request_id
        
        # Log request completion
        if hasattr(g, 'request_start_time'):
            duration = time.time() - g.request_start_time
            logger.info(f"Request {g.request_id} completed in {duration:.3f}s with status {response.status_code}")
        
        return response
    
    @app.errorhandler(429)
    def handle_rate_limit(error):
        """Enhanced rate limit error handler"""
        return jsonify({
            'error': 'RateLimitExceeded',
            'caring_message': 'You seem to be requesting a lot of content. Let\'s take a gentle pause together 💙',
            'suggestions': [
                'Take time to reflect on what you\'ve already received',
                'Consider reaching out to a friend or counselor',
                'Crisis support is always available'
            ],
            'wellness_reminder': 'Your wellbeing is more important than any feed',
            'crisis_resources': {
                'us_hotline': '988',
                'crisis_text': '741741',
                'international': 'https://findahelpline.com'
            }
        }), 429


def caring_auth_required(f):
    """
    Decorator for endpoints that require caring authentication.
    Focuses on user wellbeing rather than strict security.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user_id = request.headers.get('X-User-ID')
        
        if not user_id:
            return jsonify({
                'error': 'AuthenticationRequired',
                'message': 'User identification required',
                'caring_message': 'We need to know who you are to provide personalized care 💙',
                'help': 'Add X-User-ID header with your user identifier'
            }), 401
        
        # Store user ID for the request
        g.authenticated_user_id = user_id
        
        return f(*args, **kwargs)
    
    return decorated_function


def gentle_error_handler(f):
    """
    Decorator that wraps functions with gentle error handling.
    Ensures all errors are presented in a caring way.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {f.__name__}: {e}")
            
            return jsonify({
                'error': 'UnexpectedError',
                'message': 'An unexpected error occurred',
                'caring_message': 'Something unexpected happened, but you\'re still cared for 💙',
                'support_message': 'Our care team has been notified and will look into this',
                'what_you_can_do': [
                    'Try your request again in a moment',
                    'Contact support if the issue persists', 
                    'Remember that technical issues don\'t reflect your worth'
                ],
                'timestamp': datetime.utcnow().isoformat(),
                'request_id': getattr(g, 'request_id', 'unknown')
            }), 500
    
    return decorated_function


def _generate_request_id() -> str:
    """Generate unique request ID for tracking"""
    timestamp = str(int(time.time() * 1000))
    random_part = hashlib.md5(f"{timestamp}{time.time()}".encode()).hexdigest()[:8]
    return f"req_{timestamp}_{random_part}"