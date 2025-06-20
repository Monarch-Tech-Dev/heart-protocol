"""
AT Protocol Client for Heart Protocol

Trauma-informed, privacy-first client for interacting with the AT Protocol
and Bluesky network with focus on healing and user wellbeing.
"""

import asyncio
import aiohttp
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import logging
import json
from abc import ABC, abstractmethod
import base64
import jwt

logger = logging.getLogger(__name__)


class AuthMethod(Enum):
    """Authentication methods for AT Protocol"""
    APP_PASSWORD = "app_password"           # App-specific password
    OAUTH = "oauth"                        # OAuth authentication
    JWT_TOKEN = "jwt_token"                # JWT token authentication
    ANONYMOUS = "anonymous"                # Anonymous read-only access


class ConnectionState(Enum):
    """AT Protocol connection states"""
    DISCONNECTED = "disconnected"          # Not connected
    CONNECTING = "connecting"              # Connection in progress
    AUTHENTICATED = "authenticated"        # Connected and authenticated
    MONITORING = "monitoring"              # Actively monitoring feeds
    RATE_LIMITED = "rate_limited"          # Temporarily rate limited
    ERROR = "error"                       # Connection error state
    HEALING_PAUSE = "healing_pause"       # Paused for user wellbeing


class RequestPriority(Enum):
    """Priority levels for AT Protocol requests"""
    CRISIS_INTERVENTION = "crisis_intervention"    # Immediate crisis response
    CARE_DELIVERY = "care_delivery"               # Active care delivery
    MONITORING = "monitoring"                     # Background monitoring
    DISCOVERY = "discovery"                       # Content discovery
    MAINTENANCE = "maintenance"                   # System maintenance


@dataclass
class ATProtocolConfig:
    """Configuration for AT Protocol client"""
    service_url: str
    identifier: str
    auth_method: AuthMethod
    credentials: Dict[str, str]
    user_agent: str = "HeartProtocol/1.0 (Healing-Focused Social Care)"
    rate_limit_requests_per_minute: int = 300
    rate_limit_posts_per_hour: int = 100
    gentle_mode_enabled: bool = True
    privacy_protection_level: str = "maximum"
    healing_focus_enabled: bool = True
    trauma_informed_operations: bool = True
    cultural_sensitivity_enabled: bool = True
    accessibility_compliance: bool = True


@dataclass
class ATRequest:
    """AT Protocol request with healing-focused metadata"""
    request_id: str
    method: str
    endpoint: str
    params: Dict[str, Any]
    priority: RequestPriority
    created_at: datetime
    healing_purpose: str
    user_consent: bool
    privacy_safe: bool
    cultural_appropriate: bool
    accessibility_compliant: bool
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: int = 30


@dataclass
class ATResponse:
    """AT Protocol response with care-focused analysis"""
    request_id: str
    status_code: int
    data: Optional[Dict[str, Any]]
    headers: Dict[str, str]
    response_time_ms: float
    success: bool
    error_message: Optional[str]
    healing_relevant: bool
    care_signals_detected: List[str]
    privacy_compliant: bool
    cultural_considerations: List[str]
    accessibility_features: List[str]
    received_at: datetime


class ATProtocolClient:
    """
    AT Protocol client designed for healing-focused social care.
    
    Core Principles:
    - User privacy and consent are paramount
    - All operations are trauma-informed and gentle
    - Rate limiting respects user and platform wellbeing
    - Cultural sensitivity guides all interactions
    - Accessibility is built into every operation
    - Healing purpose drives all functionality
    - Crisis situations get immediate priority
    - Community care is fostered over individual gain
    """
    
    def __init__(self, config: ATProtocolConfig):
        self.config = config
        self.session = None
        self.connection_state = ConnectionState.DISCONNECTED
        self.auth_token = None
        self.refresh_token = None
        
        # Request management
        self.request_queue: Dict[RequestPriority, List[ATRequest]] = {
            priority: [] for priority in RequestPriority
        }
        self.active_requests: Dict[str, ATRequest] = {}
        self.request_history: List[ATResponse] = []
        
        # Rate limiting and gentle operations
        self.rate_limiter = self._initialize_rate_limiter()
        self.gentle_mode_active = config.gentle_mode_enabled
        self.last_request_time = None
        
        # Healing-focused tracking
        self.care_signals_detected = 0
        self.healing_interventions_made = 0
        self.user_wellbeing_prioritized = 0
        
        # Error handling and resilience
        self.consecutive_errors = 0
        self.healing_pause_until = None
        
        # Callbacks for different events
        self.care_signal_callbacks: List[Callable] = []
        self.connection_callbacks: List[Callable] = []
        self.error_callbacks: List[Callable] = []
    
    def _initialize_rate_limiter(self) -> Dict[str, Any]:
        """Initialize healing-focused rate limiter"""
        return {
            'requests_per_minute': self.config.rate_limit_requests_per_minute,
            'posts_per_hour': self.config.rate_limit_posts_per_hour,
            'current_minute_count': 0,
            'current_hour_count': 0,
            'last_minute_reset': datetime.utcnow(),
            'last_hour_reset': datetime.utcnow(),
            'gentle_delay_ms': 1000,  # 1 second gentle delay between requests
            'healing_pause_on_errors': True,
            'crisis_bypass_enabled': True
        }
    
    async def connect(self) -> bool:
        """
        Connect to AT Protocol with healing-focused authentication
        
        Returns:
            True if connection successful
        """
        try:
            if self.connection_state == ConnectionState.AUTHENTICATED:
                return True
            
            self.connection_state = ConnectionState.CONNECTING
            logger.info("Connecting to AT Protocol with healing focus")
            
            # Create aiohttp session with healing-focused headers
            connector = aiohttp.TCPConnector(
                limit=10,  # Gentle connection limit
                ttl_dns_cache=300,
                use_dns_cache=True
            )
            
            timeout = aiohttp.ClientTimeout(total=30)
            
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    'User-Agent': self.config.user_agent,
                    'X-Healing-Focus': 'true',
                    'X-Trauma-Informed': 'true',
                    'X-Privacy-First': 'true'
                }
            )
            
            # Authenticate based on method
            auth_success = await self._authenticate()
            
            if auth_success:
                self.connection_state = ConnectionState.AUTHENTICATED
                logger.info("Successfully connected to AT Protocol")
                
                # Trigger connection callbacks
                for callback in self.connection_callbacks:
                    try:
                        await callback({'event': 'connected', 'timestamp': datetime.utcnow()})
                    except Exception as e:
                        logger.error(f"Connection callback failed: {str(e)}")
                
                return True
            else:
                self.connection_state = ConnectionState.ERROR
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect to AT Protocol: {str(e)}")
            self.connection_state = ConnectionState.ERROR
            return False
    
    async def disconnect(self):
        """Disconnect from AT Protocol gracefully"""
        try:
            if self.session:
                await self.session.close()
                self.session = None
            
            self.connection_state = ConnectionState.DISCONNECTED
            self.auth_token = None
            self.refresh_token = None
            
            logger.info("Disconnected from AT Protocol")
            
            # Trigger connection callbacks
            for callback in self.connection_callbacks:
                try:
                    await callback({'event': 'disconnected', 'timestamp': datetime.utcnow()})
                except Exception as e:
                    logger.error(f"Disconnection callback failed: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error during disconnect: {str(e)}")
    
    async def _authenticate(self) -> bool:
        """Authenticate with AT Protocol using configured method"""
        try:
            if self.config.auth_method == AuthMethod.APP_PASSWORD:
                return await self._authenticate_app_password()
            elif self.config.auth_method == AuthMethod.OAUTH:
                return await self._authenticate_oauth()
            elif self.config.auth_method == AuthMethod.JWT_TOKEN:
                return await self._authenticate_jwt()
            elif self.config.auth_method == AuthMethod.ANONYMOUS:
                return True  # No authentication needed for read-only
            else:
                logger.error(f"Unsupported auth method: {self.config.auth_method}")
                return False
                
        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
            return False
    
    async def _authenticate_app_password(self) -> bool:
        """Authenticate using app password"""
        try:
            auth_data = {
                'identifier': self.config.identifier,
                'password': self.config.credentials.get('password')
            }
            
            response = await self._make_request(
                'POST',
                '/xrpc/com.atproto.server.createSession',
                data=auth_data,
                priority=RequestPriority.CARE_DELIVERY,
                healing_purpose="Authentication for healing-focused care delivery"
            )
            
            if response.success and response.data:
                self.auth_token = response.data.get('accessJwt')
                self.refresh_token = response.data.get('refreshJwt')
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"App password authentication failed: {str(e)}")
            return False
    
    async def _authenticate_oauth(self) -> bool:
        """Authenticate using OAuth (placeholder for future implementation)"""
        logger.warning("OAuth authentication not yet implemented")
        return False
    
    async def _authenticate_jwt(self) -> bool:
        """Authenticate using existing JWT token"""
        try:
            self.auth_token = self.config.credentials.get('jwt_token')
            
            # Validate token
            if self.auth_token:
                # Basic JWT validation (in production, would verify signature)
                try:
                    jwt.decode(self.auth_token, options={"verify_signature": False})
                    return True
                except jwt.InvalidTokenError:
                    logger.error("Invalid JWT token provided")
                    return False
            
            return False
            
        except Exception as e:
            logger.error(f"JWT authentication failed: {str(e)}")
            return False
    
    async def make_healing_request(self, method: str, endpoint: str,
                                 params: Optional[Dict[str, Any]] = None,
                                 data: Optional[Dict[str, Any]] = None,
                                 priority: RequestPriority = RequestPriority.MONITORING,
                                 healing_purpose: str = "General healing-focused operation",
                                 user_consent: bool = True) -> ATResponse:
        """
        Make a healing-focused request to AT Protocol
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: AT Protocol endpoint
            params: Query parameters
            data: Request body data
            priority: Request priority for queue management
            healing_purpose: Description of healing purpose
            user_consent: Whether user has consented to this operation
            
        Returns:
            AT Protocol response with care analysis
        """
        try:
            # Check if in healing pause
            if self.healing_pause_until and datetime.utcnow() < self.healing_pause_until:
                if priority != RequestPriority.CRISIS_INTERVENTION:
                    return self._create_paused_response(healing_purpose)
            
            # Check connection state
            if self.connection_state != ConnectionState.AUTHENTICATED:
                if not await self.connect():
                    return self._create_error_response("Not connected to AT Protocol")
            
            # Apply rate limiting (except for crisis)
            if priority != RequestPriority.CRISIS_INTERVENTION:
                if not await self._check_rate_limits():
                    return self._create_rate_limited_response()
            
            # Create request
            request = ATRequest(
                request_id=f"req_{datetime.utcnow().isoformat()}_{id(self)}",
                method=method,
                endpoint=endpoint,
                params=params or {},
                priority=priority,
                created_at=datetime.utcnow(),
                healing_purpose=healing_purpose,
                user_consent=user_consent,
                privacy_safe=await self._validate_privacy_safety(endpoint, params, data),
                cultural_appropriate=await self._validate_cultural_appropriateness(data),
                accessibility_compliant=await self._validate_accessibility(data)
            )
            
            # Make the actual request
            response = await self._execute_request(request, data)
            
            # Analyze response for care signals
            response = await self._analyze_care_signals(response)
            
            # Record request history
            self.request_history.append(response)
            
            # Update healing metrics
            await self._update_healing_metrics(response)
            
            return response
            
        except Exception as e:
            logger.error(f"Healing request failed: {str(e)}")
            return self._create_error_response(str(e))
    
    async def _make_request(self, method: str, endpoint: str,
                          params: Optional[Dict[str, Any]] = None,
                          data: Optional[Dict[str, Any]] = None,
                          priority: RequestPriority = RequestPriority.MONITORING,
                          healing_purpose: str = "AT Protocol operation") -> ATResponse:
        """Internal method for making requests"""
        return await self.make_healing_request(
            method, endpoint, params, data, priority, healing_purpose
        )
    
    async def _execute_request(self, request: ATRequest, data: Optional[Dict[str, Any]]) -> ATResponse:
        """Execute the actual HTTP request"""
        try:
            start_time = datetime.utcnow()
            
            # Prepare URL
            url = f"{self.config.service_url}{request.endpoint}"
            
            # Prepare headers
            headers = {
                'Content-Type': 'application/json',
                'X-Request-ID': request.request_id,
                'X-Healing-Purpose': request.healing_purpose,
                'X-Priority': request.priority.value
            }
            
            if self.auth_token:
                headers['Authorization'] = f'Bearer {self.auth_token}'
            
            # Apply gentle delay
            if self.gentle_mode_active and self.last_request_time:
                time_since_last = (datetime.utcnow() - self.last_request_time).total_seconds() * 1000
                gentle_delay = self.rate_limiter['gentle_delay_ms']
                
                if time_since_last < gentle_delay:
                    await asyncio.sleep((gentle_delay - time_since_last) / 1000)
            
            # Make request
            async with self.session.request(
                method=request.method,
                url=url,
                params=request.params,
                json=data,
                headers=headers
            ) as response:
                response_data = None
                try:
                    response_text = await response.text()
                    if response_text:
                        response_data = json.loads(response_text)
                except Exception as e:
                    logger.warning(f"Failed to parse response JSON: {str(e)}")
                
                end_time = datetime.utcnow()
                response_time_ms = (end_time - start_time).total_seconds() * 1000
                
                self.last_request_time = end_time
                
                return ATResponse(
                    request_id=request.request_id,
                    status_code=response.status,
                    data=response_data,
                    headers=dict(response.headers),
                    response_time_ms=response_time_ms,
                    success=200 <= response.status < 300,
                    error_message=None if 200 <= response.status < 300 else f"HTTP {response.status}",
                    healing_relevant=False,  # Will be analyzed later
                    care_signals_detected=[],
                    privacy_compliant=True,  # Will be validated
                    cultural_considerations=[],
                    accessibility_features=[],
                    received_at=end_time
                )
                
        except Exception as e:
            logger.error(f"Request execution failed: {str(e)}")
            return ATResponse(
                request_id=request.request_id,
                status_code=0,
                data=None,
                headers={},
                response_time_ms=0.0,
                success=False,
                error_message=str(e),
                healing_relevant=False,
                care_signals_detected=[],
                privacy_compliant=True,
                cultural_considerations=[],
                accessibility_features=[],
                received_at=datetime.utcnow()
            )
    
    async def _check_rate_limits(self) -> bool:
        """Check if request is within rate limits"""
        try:
            current_time = datetime.utcnow()
            
            # Check minute limit
            if (current_time - self.rate_limiter['last_minute_reset']).total_seconds() >= 60:
                self.rate_limiter['current_minute_count'] = 0
                self.rate_limiter['last_minute_reset'] = current_time
            
            if self.rate_limiter['current_minute_count'] >= self.rate_limiter['requests_per_minute']:
                logger.warning("Rate limit exceeded (per minute)")
                return False
            
            # Check hour limit
            if (current_time - self.rate_limiter['last_hour_reset']).total_seconds() >= 3600:
                self.rate_limiter['current_hour_count'] = 0
                self.rate_limiter['last_hour_reset'] = current_time
            
            if self.rate_limiter['current_hour_count'] >= self.rate_limiter['posts_per_hour']:
                logger.warning("Rate limit exceeded (per hour)")
                return False
            
            # Update counters
            self.rate_limiter['current_minute_count'] += 1
            self.rate_limiter['current_hour_count'] += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Rate limit check failed: {str(e)}")
            return True  # Default to allowing request
    
    async def _validate_privacy_safety(self, endpoint: str, params: Optional[Dict[str, Any]],
                                     data: Optional[Dict[str, Any]]) -> bool:
        """Validate that request maintains privacy safety"""
        try:
            # Check for sensitive endpoints
            sensitive_endpoints = [
                '/xrpc/com.atproto.server.createAccount',
                '/xrpc/com.atproto.server.createSession',
                '/xrpc/com.atproto.identity.updateHandle'
            ]
            
            if endpoint in sensitive_endpoints:
                # Extra validation for sensitive operations
                return True  # Placeholder - would implement actual validation
            
            # Check for PII in data
            if data:
                pii_fields = ['email', 'phone', 'ssn', 'address']
                for field in pii_fields:
                    if field in data:
                        logger.warning(f"PII field '{field}' detected in request data")
                        # In production, would apply data protection measures
            
            return True
            
        except Exception as e:
            logger.error(f"Privacy safety validation failed: {str(e)}")
            return False
    
    async def _validate_cultural_appropriateness(self, data: Optional[Dict[str, Any]]) -> bool:
        """Validate cultural appropriateness of request"""
        try:
            if not data:
                return True
            
            # Check for culturally sensitive content
            # This would integrate with cultural sensitivity engine
            return True
            
        except Exception as e:
            logger.error(f"Cultural appropriateness validation failed: {str(e)}")
            return True
    
    async def _validate_accessibility(self, data: Optional[Dict[str, Any]]) -> bool:
        """Validate accessibility compliance"""
        try:
            if not data:
                return True
            
            # Check for accessibility features in content
            # This would validate alt text, captions, etc.
            return True
            
        except Exception as e:
            logger.error(f"Accessibility validation failed: {str(e)}")
            return True
    
    async def _analyze_care_signals(self, response: ATResponse) -> ATResponse:
        """Analyze response for care signals and healing relevance"""
        try:
            if not response.success or not response.data:
                return response
            
            care_signals = []
            
            # Look for care-related content in response
            response_text = json.dumps(response.data).lower()
            
            # Crisis signals
            crisis_keywords = ['help', 'crisis', 'emergency', 'suicide', 'hurt', 'pain']
            for keyword in crisis_keywords:
                if keyword in response_text:
                    care_signals.append(f'crisis_signal_{keyword}')
            
            # Support signals
            support_keywords = ['support', 'lonely', 'sad', 'depressed', 'anxiety']
            for keyword in support_keywords:
                if keyword in response_text:
                    care_signals.append(f'support_signal_{keyword}')
            
            # Healing signals
            healing_keywords = ['healing', 'recovery', 'progress', 'better', 'grateful']
            for keyword in healing_keywords:
                if keyword in response_text:
                    care_signals.append(f'healing_signal_{keyword}')
            
            response.care_signals_detected = care_signals
            response.healing_relevant = len(care_signals) > 0
            
            # Trigger care signal callbacks if signals detected
            if care_signals:
                for callback in self.care_signal_callbacks:
                    try:
                        await callback({
                            'signals': care_signals,
                            'response': response,
                            'timestamp': datetime.utcnow()
                        })
                    except Exception as e:
                        logger.error(f"Care signal callback failed: {str(e)}")
            
            return response
            
        except Exception as e:
            logger.error(f"Care signal analysis failed: {str(e)}")
            return response
    
    async def _update_healing_metrics(self, response: ATResponse):
        """Update healing-focused metrics"""
        try:
            if response.success:
                if response.care_signals_detected:
                    self.care_signals_detected += len(response.care_signals_detected)
                
                if response.healing_relevant:
                    self.healing_interventions_made += 1
            
            # Reset error count on success
            if response.success:
                self.consecutive_errors = 0
            else:
                self.consecutive_errors += 1
                
                # Enter healing pause if too many errors
                if self.consecutive_errors >= 5:
                    await self._enter_healing_pause()
                    
        except Exception as e:
            logger.error(f"Failed to update healing metrics: {str(e)}")
    
    async def _enter_healing_pause(self):
        """Enter healing pause to protect user and system wellbeing"""
        try:
            pause_duration_minutes = min(30, self.consecutive_errors * 2)
            self.healing_pause_until = datetime.utcnow() + timedelta(minutes=pause_duration_minutes)
            
            logger.warning(f"Entering healing pause for {pause_duration_minutes} minutes due to consecutive errors")
            
            # Trigger error callbacks
            for callback in self.error_callbacks:
                try:
                    await callback({
                        'event': 'healing_pause_entered',
                        'duration_minutes': pause_duration_minutes,
                        'consecutive_errors': self.consecutive_errors,
                        'timestamp': datetime.utcnow()
                    })
                except Exception as e:
                    logger.error(f"Error callback failed: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Failed to enter healing pause: {str(e)}")
    
    def _create_error_response(self, error_message: str) -> ATResponse:
        """Create error response"""
        return ATResponse(
            request_id=f"error_{datetime.utcnow().isoformat()}",
            status_code=0,
            data=None,
            headers={},
            response_time_ms=0.0,
            success=False,
            error_message=error_message,
            healing_relevant=False,
            care_signals_detected=[],
            privacy_compliant=True,
            cultural_considerations=[],
            accessibility_features=[],
            received_at=datetime.utcnow()
        )
    
    def _create_rate_limited_response(self) -> ATResponse:
        """Create rate limited response"""
        return ATResponse(
            request_id=f"rate_limited_{datetime.utcnow().isoformat()}",
            status_code=429,
            data=None,
            headers={},
            response_time_ms=0.0,
            success=False,
            error_message="Rate limited - protecting system wellbeing",
            healing_relevant=False,
            care_signals_detected=[],
            privacy_compliant=True,
            cultural_considerations=[],
            accessibility_features=[],
            received_at=datetime.utcnow()
        )
    
    def _create_paused_response(self, healing_purpose: str) -> ATResponse:
        """Create healing pause response"""
        return ATResponse(
            request_id=f"paused_{datetime.utcnow().isoformat()}",
            status_code=503,
            data=None,
            headers={},
            response_time_ms=0.0,
            success=False,
            error_message="In healing pause - prioritizing wellbeing",
            healing_relevant=True,
            care_signals_detected=[],
            privacy_compliant=True,
            cultural_considerations=[],
            accessibility_features=[],
            received_at=datetime.utcnow()
        )
    
    # API Methods for common AT Protocol operations
    
    async def get_timeline(self, algorithm: str = "reverse-chronological",
                          limit: int = 50) -> ATResponse:
        """Get user timeline with healing focus"""
        return await self.make_healing_request(
            'GET',
            '/xrpc/app.bsky.feed.getTimeline',
            params={'algorithm': algorithm, 'limit': limit},
            priority=RequestPriority.MONITORING,
            healing_purpose="Monitor timeline for care opportunities"
        )
    
    async def get_author_feed(self, actor: str, limit: int = 50) -> ATResponse:
        """Get author feed for care monitoring"""
        return await self.make_healing_request(
            'GET',
            '/xrpc/app.bsky.feed.getAuthorFeed',
            params={'actor': actor, 'limit': limit},
            priority=RequestPriority.MONITORING,
            healing_purpose="Monitor author for potential care needs"
        )
    
    async def create_post(self, text: str, reply_to: Optional[str] = None,
                         healing_focused: bool = True) -> ATResponse:
        """Create a healing-focused post"""
        post_data = {
            '$type': 'app.bsky.feed.post',
            'text': text,
            'createdAt': datetime.utcnow().isoformat(),
            'langs': ['en']  # Would be configurable
        }
        
        if reply_to:
            post_data['reply'] = {'root': reply_to, 'parent': reply_to}
        
        if healing_focused:
            post_data['labels'] = ['healing-focused', 'trauma-informed']
        
        return await self.make_healing_request(
            'POST',
            '/xrpc/com.atproto.repo.createRecord',
            data={
                'repo': self.config.identifier,
                'collection': 'app.bsky.feed.post',
                'record': post_data
            },
            priority=RequestPriority.CARE_DELIVERY,
            healing_purpose="Deliver healing-focused content to community"
        )
    
    async def search_posts(self, query: str, limit: int = 25) -> ATResponse:
        """Search posts for care signals"""
        return await self.make_healing_request(
            'GET',
            '/xrpc/app.bsky.feed.searchPosts',
            params={'q': query, 'limit': limit},
            priority=RequestPriority.DISCOVERY,
            healing_purpose="Search for users needing care and support"
        )
    
    async def get_profile(self, actor: str) -> ATResponse:
        """Get user profile for care assessment"""
        return await self.make_healing_request(
            'GET',
            '/xrpc/app.bsky.actor.getProfile',
            params={'actor': actor},
            priority=RequestPriority.MONITORING,
            healing_purpose="Assess user profile for care context"
        )
    
    # Callback management
    
    def add_care_signal_callback(self, callback: Callable):
        """Add callback for care signal detection"""
        self.care_signal_callbacks.append(callback)
    
    def add_connection_callback(self, callback: Callable):
        """Add callback for connection state changes"""
        self.connection_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable):
        """Add callback for error events"""
        self.error_callbacks.append(callback)
    
    # Health and status methods
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get client health status"""
        return {
            'connection_state': self.connection_state.value,
            'consecutive_errors': self.consecutive_errors,
            'healing_pause_until': self.healing_pause_until.isoformat() if self.healing_pause_until else None,
            'care_signals_detected': self.care_signals_detected,
            'healing_interventions_made': self.healing_interventions_made,
            'gentle_mode_active': self.gentle_mode_active,
            'last_request_time': self.last_request_time.isoformat() if self.last_request_time else None,
            'rate_limit_status': {
                'requests_per_minute_remaining': max(0, self.rate_limiter['requests_per_minute'] - self.rate_limiter['current_minute_count']),
                'posts_per_hour_remaining': max(0, self.rate_limiter['posts_per_hour'] - self.rate_limiter['current_hour_count'])
            }
        }
    
    async def enable_gentle_mode(self):
        """Enable gentle mode for sensitive operations"""
        self.gentle_mode_active = True
        self.rate_limiter['gentle_delay_ms'] = 2000  # Increase delay
        logger.info("Gentle mode enabled for sensitive operations")
    
    async def disable_gentle_mode(self):
        """Disable gentle mode for urgent operations"""
        self.gentle_mode_active = False
        self.rate_limiter['gentle_delay_ms'] = 500  # Reduce delay
        logger.info("Gentle mode disabled for urgent operations")