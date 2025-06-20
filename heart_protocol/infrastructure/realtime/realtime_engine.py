"""
Real-time Engine for Healing-Focused Feed Updates

Core real-time engine that delivers feed updates in a trauma-informed,
gentle manner that supports healing rather than overwhelming users.
"""

import asyncio
import websockets
from typing import Dict, List, Optional, Any, Callable, Union, Set
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import logging
import json
from abc import ABC, abstractmethod
import weakref
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class UpdateType(Enum):
    """Types of real-time updates"""
    GENTLE_REMINDER = "gentle_reminder"              # Gentle daily reminders
    HEALING_PROGRESS = "healing_progress"            # Progress milestone updates
    COMMUNITY_SUPPORT = "community_support"         # Community support messages
    CRISIS_ALERT = "crisis_alert"                   # Crisis intervention alerts
    CELEBRATION = "celebration"                     # Achievement celebrations
    WISDOM_INSIGHT = "wisdom_insight"               # Community wisdom shares
    SAFETY_CHECK = "safety_check"                   # Safety monitoring updates
    PEER_CONNECTION = "peer_connection"             # Peer support connections
    CULTURAL_CONTENT = "cultural_content"           # Culturally relevant content
    ACCESSIBILITY_UPDATE = "accessibility_update"   # Accessibility accommodations


class DeliveryMode(Enum):
    """Modes for delivering real-time updates"""
    IMMEDIATE = "immediate"                         # Deliver immediately (crisis only)
    GENTLE_STREAM = "gentle_stream"                # Gentle, paced delivery
    BATCH_UPDATES = "batch_updates"                # Batched periodic updates
    USER_CONTROLLED = "user_controlled"            # User controls timing
    AMBIENT_AWARENESS = "ambient_awareness"        # Subtle, non-intrusive updates
    HEALING_MOMENTS = "healing_moments"            # Timed for optimal healing impact


class UserPresenceState(Enum):
    """User presence and availability states"""
    AVAILABLE = "available"                        # Available for updates
    BUSY = "busy"                                  # Busy, gentle updates only
    DO_NOT_DISTURB = "do_not_disturb"             # No updates except crisis
    HEALING_TIME = "healing_time"                  # In healing session, minimal updates
    CRISIS_MODE = "crisis_mode"                    # In crisis, priority updates only
    OFFLINE = "offline"                            # Not connected


class UpdatePriority(Enum):
    """Priority levels for real-time updates"""
    CRISIS = "crisis"                              # Immediate crisis intervention
    HIGH_CARE = "high_care"                        # High-priority care needs
    HEALING_SUPPORT = "healing_support"            # Active healing support
    GENTLE_REMINDER = "gentle_reminder"            # Gentle reminders and insights
    AMBIENT = "ambient"                            # Ambient awareness updates
    BACKGROUND = "background"                      # Background system updates


@dataclass
class RealtimeUpdate:
    """Real-time update message"""
    update_id: str
    user_id: str
    update_type: UpdateType
    priority: UpdatePriority
    content: Dict[str, Any]
    delivery_mode: DeliveryMode
    created_at: datetime
    expires_at: Optional[datetime]
    requires_acknowledgment: bool
    healing_impact_score: float
    emotional_intensity: float
    cultural_context: Optional[str]
    accessibility_features: List[str]
    safety_considerations: List[str]
    user_consent_required: bool
    gentle_delivery_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserConnection:
    """User's real-time connection state"""
    user_id: str
    connection_id: str
    websocket: Any  # WebSocket connection
    presence_state: UserPresenceState
    last_activity: datetime
    delivery_preferences: Dict[str, Any]
    emotional_capacity: float  # 0.0-1.0
    overwhelm_threshold: float
    current_session_updates: int
    gentle_mode_enabled: bool
    accessibility_needs: List[str]
    cultural_preferences: Dict[str, Any]
    safety_protocols: List[str]
    connection_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeliveryResult:
    """Result of real-time update delivery"""
    update_id: str
    user_id: str
    delivered: bool
    delivery_method: str
    delivered_at: Optional[datetime]
    user_response: Optional[str]
    emotional_impact_measured: Optional[float]
    gentle_delivery_successful: bool
    accessibility_accommodated: bool
    safety_maintained: bool
    delivery_notes: str


class RealtimeEngine:
    """
    Real-time engine for healing-focused feed updates.
    
    Core Principles:
    - User wellbeing comes before real-time performance
    - Gentle delivery prevents overwhelm and re-traumatization
    - Crisis updates get immediate priority with full resources
    - User presence and emotional capacity guide delivery timing
    - Cultural sensitivity is built into all communications
    - Accessibility is guaranteed for all real-time features
    - Privacy and consent are maintained in all interactions
    - Healing moments are optimized, not just engagement
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Connection management
        self.active_connections: Dict[str, UserConnection] = {}
        self.user_to_connection: Dict[str, str] = {}  # user_id -> connection_id
        
        # Update queues by priority
        self.update_queues: Dict[UpdatePriority, deque] = {
            priority: deque() for priority in UpdatePriority
        }
        
        # Delivery tracking
        self.pending_updates: Dict[str, RealtimeUpdate] = {}
        self.delivery_history: List[DeliveryResult] = []
        
        # Gentle delivery management
        self.user_update_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.overwhelm_detection: Dict[str, List[datetime]] = defaultdict(list)
        
        # Safety and crisis management
        self.crisis_connections: Set[str] = set()
        self.safety_monitors: Dict[str, Dict[str, Any]] = {}
        
        # Real-time engine state
        self.engine_active = False
        self.gentle_delivery_active = True
        
        # Callbacks and handlers
        self.update_handlers: Dict[UpdateType, List[Callable]] = defaultdict(list)
        self.delivery_callbacks: List[Callable] = []
        self.safety_callbacks: List[Callable] = []
        
        # Initialize delivery policies
        self._initialize_delivery_policies()
        
        # Initialize gentle delivery rules
        self._initialize_gentle_delivery_rules()
        
        # Initialize safety protocols
        self._initialize_safety_protocols()
    
    def _initialize_delivery_policies(self):
        """Initialize delivery policies for different update types"""
        
        self.delivery_policies = {
            UpdateType.CRISIS_ALERT: {
                'delivery_mode': DeliveryMode.IMMEDIATE,
                'priority': UpdatePriority.CRISIS,
                'bypass_user_preferences': True,
                'requires_acknowledgment': True,
                'max_delivery_delay_ms': 100,
                'retry_attempts': 5,
                'fallback_methods': ['push_notification', 'sms', 'email'],
                'gentle_delivery': False  # Crisis bypasses gentle delivery
            },
            
            UpdateType.GENTLE_REMINDER: {
                'delivery_mode': DeliveryMode.GENTLE_STREAM,
                'priority': UpdatePriority.GENTLE_REMINDER,
                'bypass_user_preferences': False,
                'requires_acknowledgment': False,
                'max_delivery_delay_ms': 5000,
                'retry_attempts': 2,
                'fallback_methods': ['delayed_delivery'],
                'gentle_delivery': True
            },
            
            UpdateType.HEALING_PROGRESS: {
                'delivery_mode': DeliveryMode.HEALING_MOMENTS,
                'priority': UpdatePriority.HEALING_SUPPORT,
                'bypass_user_preferences': False,
                'requires_acknowledgment': False,
                'max_delivery_delay_ms': 2000,
                'retry_attempts': 3,
                'fallback_methods': ['batch_delivery'],
                'gentle_delivery': True
            },
            
            UpdateType.COMMUNITY_SUPPORT: {
                'delivery_mode': DeliveryMode.AMBIENT_AWARENESS,
                'priority': UpdatePriority.HEALING_SUPPORT,
                'bypass_user_preferences': False,
                'requires_acknowledgment': False,
                'max_delivery_delay_ms': 10000,
                'retry_attempts': 2,
                'fallback_methods': ['batch_delivery'],
                'gentle_delivery': True
            },
            
            UpdateType.CELEBRATION: {
                'delivery_mode': DeliveryMode.HEALING_MOMENTS,
                'priority': UpdatePriority.HIGH_CARE,
                'bypass_user_preferences': False,
                'requires_acknowledgment': False,
                'max_delivery_delay_ms': 1000,
                'retry_attempts': 3,
                'fallback_methods': ['delayed_delivery'],
                'gentle_delivery': True
            },
            
            UpdateType.WISDOM_INSIGHT: {
                'delivery_mode': DeliveryMode.BATCH_UPDATES,
                'priority': UpdatePriority.GENTLE_REMINDER,
                'bypass_user_preferences': False,
                'requires_acknowledgment': False,
                'max_delivery_delay_ms': 30000,
                'retry_attempts': 1,
                'fallback_methods': ['next_session_delivery'],
                'gentle_delivery': True
            },
            
            UpdateType.SAFETY_CHECK: {
                'delivery_mode': DeliveryMode.GENTLE_STREAM,
                'priority': UpdatePriority.HIGH_CARE,
                'bypass_user_preferences': False,
                'requires_acknowledgment': True,
                'max_delivery_delay_ms': 3000,
                'retry_attempts': 3,
                'fallback_methods': ['delayed_delivery'],
                'gentle_delivery': True
            },
            
            UpdateType.PEER_CONNECTION: {
                'delivery_mode': DeliveryMode.AMBIENT_AWARENESS,
                'priority': UpdatePriority.HEALING_SUPPORT,
                'bypass_user_preferences': False,
                'requires_acknowledgment': False,
                'max_delivery_delay_ms': 15000,
                'retry_attempts': 2,
                'fallback_methods': ['batch_delivery'],
                'gentle_delivery': True
            }
        }
    
    def _initialize_gentle_delivery_rules(self):
        """Initialize rules for gentle delivery to prevent overwhelm"""
        
        self.gentle_delivery_rules = {
            'max_updates_per_minute': {
                UserPresenceState.AVAILABLE: 3,
                UserPresenceState.BUSY: 1,
                UserPresenceState.DO_NOT_DISTURB: 0,
                UserPresenceState.HEALING_TIME: 1,
                UserPresenceState.CRISIS_MODE: 10,  # More updates allowed during crisis
                UserPresenceState.OFFLINE: 0
            },
            
            'max_updates_per_hour': {
                UserPresenceState.AVAILABLE: 12,
                UserPresenceState.BUSY: 3,
                UserPresenceState.DO_NOT_DISTURB: 0,
                UserPresenceState.HEALING_TIME: 2,
                UserPresenceState.CRISIS_MODE: 50,
                UserPresenceState.OFFLINE: 0
            },
            
            'emotional_intensity_limits': {
                'very_low_capacity': 0.3,
                'low_capacity': 0.5,
                'moderate_capacity': 0.7,
                'good_capacity': 0.9,
                'high_capacity': 1.0
            },
            
            'delivery_spacing_seconds': {
                UpdatePriority.CRISIS: 0,  # No spacing for crisis
                UpdatePriority.HIGH_CARE: 30,
                UpdatePriority.HEALING_SUPPORT: 60,
                UpdatePriority.GENTLE_REMINDER: 120,
                UpdatePriority.AMBIENT: 300,
                UpdatePriority.BACKGROUND: 600
            },
            
            'overwhelm_detection': {
                'rapid_updates_threshold': 5,  # 5 updates in 5 minutes
                'rapid_updates_window_minutes': 5,
                'emotional_intensity_spike': 0.8,
                'recovery_period_minutes': 30
            }
        }
    
    def _initialize_safety_protocols(self):
        """Initialize safety protocols for real-time communications"""
        
        self.safety_protocols = {
            'crisis_detection': {
                'monitor_keywords': ['help', 'crisis', 'emergency', 'suicide', 'hurt'],
                'response_time_ms': 100,
                'escalation_required': True,
                'human_handoff': True
            },
            
            'overwhelm_prevention': {
                'monitor_user_responses': True,
                'detect_negative_patterns': True,
                'automatic_gentle_mode': True,
                'cooling_off_period_minutes': 15
            },
            
            'trauma_informed_delivery': {
                'avoid_triggering_content': True,
                'provide_content_warnings': True,
                'respect_boundaries': True,
                'offer_opt_out': True
            },
            
            'privacy_protection': {
                'encrypt_all_messages': True,
                'no_message_logging': True,
                'user_consent_required': True,
                'data_minimization': True
            }
        }
    
    async def start_realtime_engine(self):
        """Start the real-time engine with gentle delivery processing"""
        if self.engine_active:
            return
        
        self.engine_active = True
        logger.info("Starting healing-focused real-time engine")
        
        # Start core engine tasks
        asyncio.create_task(self._process_update_queues())
        asyncio.create_task(self._monitor_user_wellbeing())
        asyncio.create_task(self._manage_gentle_delivery())
        asyncio.create_task(self._handle_safety_monitoring())
        asyncio.create_task(self._cleanup_expired_updates())
    
    async def stop_realtime_engine(self):
        """Stop the real-time engine gracefully"""
        self.engine_active = False
        
        # Close all active connections gracefully
        for connection in self.active_connections.values():
            try:
                await self._close_connection_gracefully(connection)
            except Exception as e:
                logger.error(f"Error closing connection {connection.connection_id}: {str(e)}")
        
        logger.info("Real-time engine stopped")
    
    async def connect_user(self, user_id: str, websocket: Any,
                          connection_metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Connect a user to the real-time system
        
        Args:
            user_id: User identifier
            websocket: WebSocket connection
            connection_metadata: Additional connection information
            
        Returns:
            Connection ID for tracking
        """
        try:
            connection_id = f"conn_{user_id}_{datetime.utcnow().isoformat()}"
            
            # Get user preferences for real-time delivery
            delivery_preferences = await self._get_user_delivery_preferences(user_id)
            
            # Assess user's current emotional capacity
            emotional_capacity = await self._assess_user_emotional_capacity(user_id)
            
            # Create user connection
            connection = UserConnection(
                user_id=user_id,
                connection_id=connection_id,
                websocket=websocket,
                presence_state=UserPresenceState.AVAILABLE,
                last_activity=datetime.utcnow(),
                delivery_preferences=delivery_preferences,
                emotional_capacity=emotional_capacity,
                overwhelm_threshold=delivery_preferences.get('overwhelm_threshold', 0.7),
                current_session_updates=0,
                gentle_mode_enabled=delivery_preferences.get('gentle_mode', True),
                accessibility_needs=delivery_preferences.get('accessibility_needs', []),
                cultural_preferences=delivery_preferences.get('cultural_preferences', {}),
                safety_protocols=delivery_preferences.get('safety_protocols', []),
                connection_metadata=connection_metadata or {}
            )
            
            # Store connection
            self.active_connections[connection_id] = connection
            self.user_to_connection[user_id] = connection_id
            
            # Send welcome message with gentle introduction
            await self._send_welcome_message(connection)
            
            # Check for pending updates
            await self._deliver_pending_updates(user_id)
            
            logger.info(f"User {user_id} connected to real-time system")
            return connection_id
            
        except Exception as e:
            logger.error(f"Failed to connect user {user_id}: {str(e)}")
            raise
    
    async def disconnect_user(self, user_id: str):
        """Disconnect user from real-time system"""
        try:
            connection_id = self.user_to_connection.get(user_id)
            if not connection_id:
                return
            
            connection = self.active_connections.get(connection_id)
            if connection:
                await self._close_connection_gracefully(connection)
                del self.active_connections[connection_id]
                del self.user_to_connection[user_id]
            
            logger.info(f"User {user_id} disconnected from real-time system")
            
        except Exception as e:
            logger.error(f"Failed to disconnect user {user_id}: {str(e)}")
    
    async def send_realtime_update(self, update: RealtimeUpdate) -> bool:
        """
        Send a real-time update to a user
        
        Args:
            update: The update to send
            
        Returns:
            True if successfully queued/delivered
        """
        try:
            # Validate update
            if not await self._validate_update(update):
                logger.warning(f"Update validation failed for {update.update_id}")
                return False
            
            # Check user consent
            if update.user_consent_required:
                if not await self._check_user_consent(update.user_id, update.update_type):
                    logger.info(f"User consent not given for update {update.update_id}")
                    return False
            
            # Apply cultural and accessibility adaptations
            adapted_update = await self._adapt_update_for_user(update)
            
            # Store update
            self.pending_updates[update.update_id] = adapted_update
            
            # Add to appropriate priority queue
            self.update_queues[update.priority].append(adapted_update)
            
            logger.debug(f"Queued real-time update {update.update_id} for user {update.user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send real-time update: {str(e)}")
            return False
    
    async def update_user_presence(self, user_id: str, presence_state: UserPresenceState):
        """Update user's presence state"""
        try:
            connection_id = self.user_to_connection.get(user_id)
            if not connection_id:
                return
            
            connection = self.active_connections.get(connection_id)
            if connection:
                old_state = connection.presence_state
                connection.presence_state = presence_state
                connection.last_activity = datetime.utcnow()
                
                # Handle presence state changes
                await self._handle_presence_change(connection, old_state, presence_state)
                
                logger.debug(f"Updated presence for user {user_id}: {old_state.value} -> {presence_state.value}")
            
        except Exception as e:
            logger.error(f"Failed to update user presence: {str(e)}")
    
    async def _process_update_queues(self):
        """Process update queues with priority and gentle delivery"""
        while self.engine_active:
            try:
                # Process queues in priority order
                priority_order = [
                    UpdatePriority.CRISIS,
                    UpdatePriority.HIGH_CARE,
                    UpdatePriority.HEALING_SUPPORT,
                    UpdatePriority.GENTLE_REMINDER,
                    UpdatePriority.AMBIENT,
                    UpdatePriority.BACKGROUND
                ]
                
                updates_processed = 0
                for priority in priority_order:
                    queue = self.update_queues[priority]
                    
                    # Process up to 10 updates per cycle for each priority
                    batch_size = 10 if priority == UpdatePriority.CRISIS else 5
                    processed_in_batch = 0
                    
                    while queue and processed_in_batch < batch_size:
                        update = queue.popleft()
                        
                        # Check if update has expired
                        if update.expires_at and datetime.utcnow() > update.expires_at:
                            await self._handle_expired_update(update)
                            continue
                        
                        # Attempt delivery
                        delivery_result = await self._attempt_update_delivery(update)
                        
                        if delivery_result.delivered:
                            processed_in_batch += 1
                            updates_processed += 1
                        else:
                            # Requeue for retry if appropriate
                            if await self._should_retry_update(update, delivery_result):
                                queue.append(update)
                        
                        # Record delivery result
                        self.delivery_history.append(delivery_result)
                
                # Adaptive delay based on system load
                if updates_processed == 0:
                    await asyncio.sleep(1.0)  # No updates, longer sleep
                elif updates_processed < 5:
                    await asyncio.sleep(0.5)  # Light load
                else:
                    await asyncio.sleep(0.1)  # Heavy load, shorter sleep
                
            except Exception as e:
                logger.error(f"Error processing update queues: {str(e)}")
                await asyncio.sleep(5.0)  # Error recovery delay
    
    async def _attempt_update_delivery(self, update: RealtimeUpdate) -> DeliveryResult:
        """Attempt to deliver an update to the user"""
        try:
            connection_id = self.user_to_connection.get(update.user_id)
            if not connection_id:
                return DeliveryResult(
                    update_id=update.update_id,
                    user_id=update.user_id,
                    delivered=False,
                    delivery_method="no_connection",
                    delivered_at=None,
                    user_response=None,
                    emotional_impact_measured=None,
                    gentle_delivery_successful=False,
                    accessibility_accommodated=False,
                    safety_maintained=True,
                    delivery_notes="User not connected"
                )
            
            connection = self.active_connections.get(connection_id)
            if not connection:
                return self._create_failed_delivery_result(update, "connection_not_found")
            
            # Check if delivery is appropriate
            delivery_check = await self._check_delivery_appropriateness(update, connection)
            if not delivery_check['can_deliver']:
                return self._create_failed_delivery_result(update, delivery_check['reason'])
            
            # Apply gentle delivery if needed
            if update.gentle_delivery_config.get('enable_gentle_delivery', True):
                gentle_result = await self._apply_gentle_delivery(update, connection)
                if not gentle_result['proceed']:
                    return self._create_delayed_delivery_result(update, gentle_result['delay_reason'])
            
            # Prepare message for delivery
            message = await self._prepare_delivery_message(update, connection)
            
            # Send message via WebSocket
            delivery_successful = await self._send_websocket_message(connection, message)
            
            if delivery_successful:
                # Update connection state
                connection.current_session_updates += 1
                connection.last_activity = datetime.utcnow()
                
                # Track update count for overwhelm detection
                await self._track_update_delivery(update, connection)
                
                return DeliveryResult(
                    update_id=update.update_id,
                    user_id=update.user_id,
                    delivered=True,
                    delivery_method="websocket",
                    delivered_at=datetime.utcnow(),
                    user_response=None,
                    emotional_impact_measured=None,
                    gentle_delivery_successful=True,
                    accessibility_accommodated=len(connection.accessibility_needs) > 0,
                    safety_maintained=True,
                    delivery_notes="Successfully delivered via WebSocket"
                )
            else:
                return self._create_failed_delivery_result(update, "websocket_send_failed")
            
        except Exception as e:
            logger.error(f"Failed to deliver update {update.update_id}: {str(e)}")
            return self._create_failed_delivery_result(update, f"delivery_error: {str(e)}")
    
    async def _check_delivery_appropriateness(self, update: RealtimeUpdate,
                                            connection: UserConnection) -> Dict[str, Any]:
        """Check if update delivery is appropriate for current user state"""
        try:
            # Crisis updates always get delivered
            if update.priority == UpdatePriority.CRISIS:
                return {'can_deliver': True, 'reason': 'crisis_priority'}
            
            # Check user presence state
            presence_rules = {
                UserPresenceState.AVAILABLE: True,
                UserPresenceState.BUSY: update.priority in [UpdatePriority.CRISIS, UpdatePriority.HIGH_CARE],
                UserPresenceState.DO_NOT_DISTURB: update.priority == UpdatePriority.CRISIS,
                UserPresenceState.HEALING_TIME: update.priority in [UpdatePriority.CRISIS, UpdatePriority.HIGH_CARE],
                UserPresenceState.CRISIS_MODE: True,
                UserPresenceState.OFFLINE: False
            }
            
            if not presence_rules.get(connection.presence_state, False):
                return {'can_deliver': False, 'reason': f'user_presence_{connection.presence_state.value}'}
            
            # Check emotional capacity
            if update.emotional_intensity > connection.emotional_capacity:
                if update.priority not in [UpdatePriority.CRISIS, UpdatePriority.HIGH_CARE]:
                    return {'can_deliver': False, 'reason': 'emotional_capacity_insufficient'}
            
            # Check overwhelm threshold
            if connection.current_session_updates >= connection.overwhelm_threshold * 20:  # Rough conversion
                if update.priority not in [UpdatePriority.CRISIS, UpdatePriority.HIGH_CARE]:
                    return {'can_deliver': False, 'reason': 'overwhelm_threshold_reached'}
            
            # Check gentle delivery rules
            if connection.gentle_mode_enabled:
                gentle_check = await self._check_gentle_delivery_rules(update, connection)
                if not gentle_check['can_deliver']:
                    return gentle_check
            
            return {'can_deliver': True, 'reason': 'all_checks_passed'}
            
        except Exception as e:
            logger.error(f"Failed to check delivery appropriateness: {str(e)}")
            return {'can_deliver': False, 'reason': f'check_error: {str(e)}'}
    
    async def _check_gentle_delivery_rules(self, update: RealtimeUpdate,
                                         connection: UserConnection) -> Dict[str, Any]:
        """Check gentle delivery rules to prevent overwhelm"""
        try:
            current_time = datetime.utcnow()
            user_id = connection.user_id
            
            # Check minute-level limits
            minute_limit = self.gentle_delivery_rules['max_updates_per_minute'].get(
                connection.presence_state, 1
            )
            
            recent_updates = self.user_update_counts[user_id]
            current_minute = current_time.strftime('%Y-%m-%d-%H-%M')
            
            if recent_updates[current_minute] >= minute_limit:
                return {'can_deliver': False, 'reason': 'minute_limit_exceeded'}
            
            # Check hour-level limits
            hour_limit = self.gentle_delivery_rules['max_updates_per_hour'].get(
                connection.presence_state, 5
            )
            
            current_hour = current_time.strftime('%Y-%m-%d-%H')
            if recent_updates[current_hour] >= hour_limit:
                return {'can_deliver': False, 'reason': 'hour_limit_exceeded'}
            
            # Check delivery spacing
            spacing_seconds = self.gentle_delivery_rules['delivery_spacing_seconds'].get(
                update.priority, 60
            )
            
            if spacing_seconds > 0:
                time_since_last = (current_time - connection.last_activity).total_seconds()
                if time_since_last < spacing_seconds:
                    return {'can_deliver': False, 'reason': 'spacing_requirement_not_met'}
            
            return {'can_deliver': True, 'reason': 'gentle_rules_satisfied'}
            
        except Exception as e:
            logger.error(f"Failed to check gentle delivery rules: {str(e)}")
            return {'can_deliver': False, 'reason': f'gentle_check_error: {str(e)}'}
    
    async def _apply_gentle_delivery(self, update: RealtimeUpdate,
                                   connection: UserConnection) -> Dict[str, Any]:
        """Apply gentle delivery modifications to update"""
        try:
            # Check for overwhelm indicators
            overwhelm_detected = await self._detect_user_overwhelm(connection)
            if overwhelm_detected:
                return {'proceed': False, 'delay_reason': 'overwhelm_detected'}
            
            # Apply emotional intensity adjustments
            if update.emotional_intensity > connection.emotional_capacity:
                # Reduce emotional intensity if possible
                if 'emotional_intensity_reduction' in update.gentle_delivery_config:
                    update.emotional_intensity *= 0.8
                    update.content['gentle_mode'] = True
            
            # Apply timing adjustments
            gentle_config = update.gentle_delivery_config
            if gentle_config.get('apply_timing_delays'):
                optimal_timing = await self._calculate_optimal_timing(update, connection)
                if not optimal_timing['immediate']:
                    return {'proceed': False, 'delay_reason': 'timing_optimization'}
            
            return {'proceed': True, 'delay_reason': None}
            
        except Exception as e:
            logger.error(f"Failed to apply gentle delivery: {str(e)}")
            return {'proceed': True, 'delay_reason': None}  # Default to proceeding
    
    async def _detect_user_overwhelm(self, connection: UserConnection) -> bool:
        """Detect if user is showing signs of overwhelm"""
        try:
            user_id = connection.user_id
            current_time = datetime.utcnow()
            
            # Check rapid update threshold
            detection_config = self.gentle_delivery_rules['overwhelm_detection']
            threshold = detection_config['rapid_updates_threshold']
            window_minutes = detection_config['rapid_updates_window_minutes']
            
            # Get recent update timestamps
            recent_updates = self.overwhelm_detection[user_id]
            cutoff_time = current_time - timedelta(minutes=window_minutes)
            recent_updates = [t for t in recent_updates if t > cutoff_time]
            
            if len(recent_updates) >= threshold:
                logger.warning(f"Overwhelm detected for user {user_id}: {len(recent_updates)} updates in {window_minutes} minutes")
                return True
            
            # Check emotional capacity drop
            if connection.emotional_capacity < 0.3:
                return True
            
            # Check session update count relative to threshold
            if connection.current_session_updates > connection.overwhelm_threshold * 15:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to detect user overwhelm: {str(e)}")
            return False
    
    async def _prepare_delivery_message(self, update: RealtimeUpdate,
                                      connection: UserConnection) -> Dict[str, Any]:
        """Prepare message for delivery with all adaptations"""
        try:
            message = {
                'type': 'realtime_update',
                'update_id': update.update_id,
                'update_type': update.update_type.value,
                'priority': update.priority.value,
                'content': update.content.copy(),
                'timestamp': update.created_at.isoformat(),
                'requires_acknowledgment': update.requires_acknowledgment,
                'healing_focused': True,
                'gentle_delivery': True
            }
            
            # Apply accessibility adaptations
            if connection.accessibility_needs:
                message['accessibility_features'] = await self._apply_accessibility_adaptations(
                    message, connection.accessibility_needs
                )
            
            # Apply cultural adaptations
            if connection.cultural_preferences:
                message['cultural_adaptations'] = await self._apply_cultural_adaptations(
                    message, connection.cultural_preferences
                )
            
            # Add safety considerations
            if update.safety_considerations:
                message['safety_notes'] = update.safety_considerations
                message['content_warnings'] = await self._generate_content_warnings(update)
            
            # Add gentle delivery indicators
            if connection.gentle_mode_enabled:
                message['gentle_mode'] = True
                message['emotional_intensity'] = update.emotional_intensity
                message['can_dismiss'] = True
                message['opt_out_available'] = True
            
            return message
            
        except Exception as e:
            logger.error(f"Failed to prepare delivery message: {str(e)}")
            return {'error': 'message_preparation_failed'}
    
    async def _send_websocket_message(self, connection: UserConnection,
                                    message: Dict[str, Any]) -> bool:
        """Send message via WebSocket with error handling"""
        try:
            if not connection.websocket:
                return False
            
            # Convert message to JSON
            message_json = json.dumps(message, default=str)
            
            # Send via WebSocket
            await connection.websocket.send(message_json)
            
            return True
            
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"WebSocket connection closed for user {connection.user_id}")
            await self._handle_connection_closed(connection)
            return False
        except Exception as e:
            logger.error(f"Failed to send WebSocket message: {str(e)}")
            return False
    
    async def _track_update_delivery(self, update: RealtimeUpdate,
                                   connection: UserConnection):
        """Track update delivery for overwhelm detection"""
        try:
            current_time = datetime.utcnow()
            user_id = connection.user_id
            
            # Update delivery counts
            current_minute = current_time.strftime('%Y-%m-%d-%H-%M')
            current_hour = current_time.strftime('%Y-%m-%d-%H')
            
            self.user_update_counts[user_id][current_minute] += 1
            self.user_update_counts[user_id][current_hour] += 1
            
            # Add to overwhelm detection timeline
            self.overwhelm_detection[user_id].append(current_time)
            
            # Clean up old entries (keep last 24 hours)
            cutoff_time = current_time - timedelta(hours=24)
            self.overwhelm_detection[user_id] = [
                t for t in self.overwhelm_detection[user_id] if t > cutoff_time
            ]
            
        except Exception as e:
            logger.error(f"Failed to track update delivery: {str(e)}")
    
    async def _monitor_user_wellbeing(self):
        """Monitor user wellbeing and adjust real-time delivery accordingly"""
        while self.engine_active:
            try:
                for connection in self.active_connections.values():
                    # Check for overwhelm
                    overwhelm_detected = await self._detect_user_overwhelm(connection)
                    if overwhelm_detected:
                        await self._handle_user_overwhelm(connection)
                    
                    # Update emotional capacity assessment
                    new_capacity = await self._assess_user_emotional_capacity(connection.user_id)
                    if abs(new_capacity - connection.emotional_capacity) > 0.2:
                        connection.emotional_capacity = new_capacity
                        logger.info(f"Updated emotional capacity for user {connection.user_id}: {new_capacity:.2f}")
                    
                    # Check for inactive connections
                    if (datetime.utcnow() - connection.last_activity).total_seconds() > 1800:  # 30 minutes
                        await self._handle_inactive_connection(connection)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error monitoring user wellbeing: {str(e)}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def _handle_user_overwhelm(self, connection: UserConnection):
        """Handle detected user overwhelm"""
        try:
            logger.warning(f"Handling overwhelm for user {connection.user_id}")
            
            # Enable gentle mode
            connection.gentle_mode_enabled = True
            
            # Reduce emotional capacity temporarily
            connection.emotional_capacity = min(connection.emotional_capacity, 0.4)
            
            # Update presence to healing time
            if connection.presence_state not in [UserPresenceState.CRISIS_MODE, UserPresenceState.DO_NOT_DISTURB]:
                connection.presence_state = UserPresenceState.HEALING_TIME
            
            # Send gentle overwhelm recovery message
            recovery_message = {
                'type': 'gentle_support',
                'message': 'I notice you might be feeling overwhelmed. I\'m here for you and will give you some gentle space. Take all the time you need.',
                'actions': {
                    'pause_updates': True,
                    'enable_breathing_space': True,
                    'contact_support': 'available'
                },
                'healing_focused': True
            }
            
            await self._send_websocket_message(connection, recovery_message)
            
            # Trigger safety callbacks if configured
            for callback in self.safety_callbacks:
                try:
                    await callback({'type': 'overwhelm_detected', 'user_id': connection.user_id})
                except Exception as e:
                    logger.error(f"Safety callback failed: {str(e)}")
            
        except Exception as e:
            logger.error(f"Failed to handle user overwhelm: {str(e)}")
    
    def _create_failed_delivery_result(self, update: RealtimeUpdate, reason: str) -> DeliveryResult:
        """Create a failed delivery result"""
        return DeliveryResult(
            update_id=update.update_id,
            user_id=update.user_id,
            delivered=False,
            delivery_method="failed",
            delivered_at=None,
            user_response=None,
            emotional_impact_measured=None,
            gentle_delivery_successful=False,
            accessibility_accommodated=False,
            safety_maintained=True,
            delivery_notes=reason
        )
    
    def _create_delayed_delivery_result(self, update: RealtimeUpdate, reason: str) -> DeliveryResult:
        """Create a delayed delivery result"""
        return DeliveryResult(
            update_id=update.update_id,
            user_id=update.user_id,
            delivered=False,
            delivery_method="delayed",
            delivered_at=None,
            user_response=None,
            emotional_impact_measured=None,
            gentle_delivery_successful=True,  # Delayed is a successful gentle delivery choice
            accessibility_accommodated=False,
            safety_maintained=True,
            delivery_notes=f"Delivery delayed: {reason}"
        )
    
    # Placeholder methods for external integrations
    async def _get_user_delivery_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user's real-time delivery preferences"""
        # In production, this would integrate with preference system
        return {
            'gentle_mode': True,
            'overwhelm_threshold': 0.7,
            'accessibility_needs': [],
            'cultural_preferences': {},
            'safety_protocols': []
        }
    
    async def _assess_user_emotional_capacity(self, user_id: str) -> float:
        """Assess user's current emotional capacity"""
        # In production, this would integrate with emotional assessment system
        return 0.7  # Default moderate capacity
    
    async def _validate_update(self, update: RealtimeUpdate) -> bool:
        """Validate update before processing"""
        # Basic validation
        return (update.user_id and update.update_type and 
                update.priority and update.content is not None)
    
    async def _check_user_consent(self, user_id: str, update_type: UpdateType) -> bool:
        """Check if user has consented to receive this type of update"""
        # In production, this would check consent system
        return True  # Default to consented for now
    
    async def _adapt_update_for_user(self, update: RealtimeUpdate) -> RealtimeUpdate:
        """Adapt update for user's cultural and accessibility needs"""
        # In production, this would apply comprehensive adaptations
        return update
    
    # Additional placeholder methods would be implemented here
    async def _send_welcome_message(self, connection: UserConnection):
        """Send gentle welcome message to new connection"""
        pass
    
    async def _deliver_pending_updates(self, user_id: str):
        """Deliver any pending updates for newly connected user"""
        pass
    
    async def _close_connection_gracefully(self, connection: UserConnection):
        """Close connection gracefully with farewell message"""
        pass
    
    async def _handle_presence_change(self, connection: UserConnection, 
                                    old_state: UserPresenceState, 
                                    new_state: UserPresenceState):
        """Handle user presence state changes"""
        pass
    
    async def _handle_expired_update(self, update: RealtimeUpdate):
        """Handle expired updates"""
        pass
    
    async def _should_retry_update(self, update: RealtimeUpdate, 
                                 delivery_result: DeliveryResult) -> bool:
        """Determine if update should be retried"""
        return False  # Placeholder
    
    async def _manage_gentle_delivery(self):
        """Manage gentle delivery system"""
        pass
    
    async def _handle_safety_monitoring(self):
        """Handle safety monitoring for real-time communications"""
        pass
    
    async def _cleanup_expired_updates(self):
        """Clean up expired updates"""
        pass
    
    async def _calculate_optimal_timing(self, update: RealtimeUpdate, 
                                      connection: UserConnection) -> Dict[str, Any]:
        """Calculate optimal timing for update delivery"""
        return {'immediate': True}
    
    async def _apply_accessibility_adaptations(self, message: Dict[str, Any], 
                                             accessibility_needs: List[str]) -> Dict[str, Any]:
        """Apply accessibility adaptations to message"""
        return {}
    
    async def _apply_cultural_adaptations(self, message: Dict[str, Any], 
                                        cultural_preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Apply cultural adaptations to message"""
        return {}
    
    async def _generate_content_warnings(self, update: RealtimeUpdate) -> List[str]:
        """Generate appropriate content warnings"""
        return []
    
    async def _handle_connection_closed(self, connection: UserConnection):
        """Handle closed WebSocket connection"""
        pass
    
    async def _handle_inactive_connection(self, connection: UserConnection):
        """Handle inactive connection"""
        pass