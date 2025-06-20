"""
Firehose Monitor

Core real-time monitoring system for observing multiple social media streams,
detecting care signals, and coordinating healing-focused responses.
"""

import asyncio
import websockets
import aiohttp
from typing import Dict, List, Optional, Any, Callable, Set, Tuple, AsyncGenerator
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import logging
import json
import statistics
from collections import defaultdict, deque
import re
import hashlib

logger = logging.getLogger(__name__)


class StreamType(Enum):
    """Types of data streams to monitor"""
    BLUESKY_FIREHOSE = "bluesky_firehose"           # Bluesky AT Protocol firehose
    MASTODON_STREAMING = "mastodon_streaming"       # Mastodon streaming API
    TWITTER_STREAM = "twitter_stream"               # Twitter/X streaming (if available)
    DISCORD_WEBHOOKS = "discord_webhooks"           # Discord webhook monitoring
    TELEGRAM_UPDATES = "telegram_updates"           # Telegram bot updates
    MATRIX_EVENTS = "matrix_events"                 # Matrix protocol events
    CUSTOM_WEBHOOKS = "custom_webhooks"             # Custom webhook integrations
    RSS_FEEDS = "rss_feeds"                        # RSS feed monitoring
    WEB_SCRAPING = "web_scraping"                  # Ethical web scraping
    API_POLLING = "api_polling"                    # API polling for platforms without streaming


class MonitoringScope(Enum):
    """Scope of monitoring operations"""
    GLOBAL_MONITORING = "global_monitoring"         # Monitor global public streams
    KEYWORD_FILTERING = "keyword_filtering"         # Monitor specific keywords
    USER_FOLLOWING = "user_following"              # Monitor specific users
    HASHTAG_TRACKING = "hashtag_tracking"          # Track specific hashtags
    GEOGRAPHIC_REGION = "geographic_region"        # Monitor geographic regions
    LANGUAGE_SPECIFIC = "language_specific"        # Monitor specific languages
    COMMUNITY_FOCUSED = "community_focused"        # Monitor specific communities
    CRISIS_WATCH = "crisis_watch"                  # Enhanced crisis monitoring
    HEALING_FOCUS = "healing_focus"                # Focus on healing content


class MonitoringPriority(Enum):
    """Priority levels for different monitoring streams"""
    CRITICAL = "critical"                          # Crisis intervention priority
    HIGH = "high"                                  # High-priority care signals
    MEDIUM = "medium"                              # Standard monitoring
    LOW = "low"                                    # Background monitoring
    RESEARCH = "research"                          # Research and analytics


class StreamHealth(Enum):
    """Health status of monitoring streams"""
    HEALTHY = "healthy"                            # Stream operating normally
    DEGRADED = "degraded"                          # Stream experiencing issues
    RECONNECTING = "reconnecting"                  # Attempting to reconnect
    FAILED = "failed"                              # Stream connection failed
    RATE_LIMITED = "rate_limited"                  # Stream rate limited
    PAUSED = "paused"                              # Intentionally paused


@dataclass
class StreamConfig:
    """Configuration for a monitoring stream"""
    stream_id: str
    stream_type: StreamType
    monitoring_scope: MonitoringScope
    priority: MonitoringPriority
    connection_params: Dict[str, Any]
    filter_criteria: Dict[str, Any]
    rate_limits: Dict[str, Any]
    health_check_interval: int
    retry_config: Dict[str, Any]
    privacy_settings: Dict[str, Any]
    cultural_considerations: List[str]
    accessibility_features: List[str]
    enabled: bool = True
    gentle_mode: bool = True


@dataclass
class StreamEvent:
    """Event detected from a monitoring stream"""
    event_id: str
    stream_id: str
    stream_type: StreamType
    raw_content: str
    processed_content: str
    author_info: Dict[str, Any]
    metadata: Dict[str, Any]
    timestamp: datetime
    detected_signals: List[str]
    care_score: float
    crisis_score: float
    healing_score: float
    cultural_context: List[str]
    privacy_level: str
    processing_priority: MonitoringPriority
    requires_human_review: bool


@dataclass
class StreamMetrics:
    """Metrics for a monitoring stream"""
    stream_id: str
    events_processed: int
    care_signals_detected: int
    crisis_events_found: int
    healing_moments_identified: int
    processing_latency_ms: float
    error_count: int
    reconnection_count: int
    last_event_time: datetime
    health_status: StreamHealth
    throughput_per_minute: float
    quality_score: float


class FirehoseMonitor:
    """
    Core real-time monitoring system for social media streams.
    
    Core Principles:
    - Real-time detection of care opportunities and crisis situations
    - Privacy-preserving monitoring with user consent where possible
    - Trauma-informed processing of sensitive content
    - Cultural sensitivity in content interpretation
    - Gentle, non-intrusive monitoring approaches
    - Healing-focused signal detection over engagement metrics
    - Crisis intervention prioritization
    - Community wellbeing focus
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Monitoring settings
        self.max_concurrent_streams = self.config.get('max_concurrent_streams', 10)
        self.processing_batch_size = self.config.get('processing_batch_size', 100)
        self.event_retention_hours = self.config.get('event_retention_hours', 24)
        self.gentle_mode_enabled = self.config.get('gentle_mode_enabled', True)
        
        # Stream management
        self.active_streams: Dict[str, StreamConfig] = {}
        self.stream_connections: Dict[str, Any] = {}
        self.stream_metrics: Dict[str, StreamMetrics] = {}
        self.event_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Event processing
        self.event_processors: List[Callable] = []
        self.priority_queues: Dict[MonitoringPriority, asyncio.Queue] = {
            priority: asyncio.Queue() for priority in MonitoringPriority
        }
        
        # Pattern matching and filtering
        self.care_patterns = self._initialize_care_patterns()
        self.crisis_patterns = self._initialize_crisis_patterns()
        self.healing_patterns = self._initialize_healing_patterns()
        self.noise_filters = self._initialize_noise_filters()
        
        # Performance tracking
        self.total_events_processed = 0
        self.total_care_signals_detected = 0
        self.total_crisis_events_found = 0
        self.total_healing_moments_identified = 0
        self.processing_start_time = datetime.utcnow()
        
        # Safety and privacy systems
        self.privacy_filter = None              # Would integrate with privacy system
        self.content_moderator = None          # Would integrate with content moderation
        self.cultural_analyzer = None          # Would integrate with cultural sensitivity
        self.trauma_safety_checker = None     # Would integrate with trauma safety
        
        # Callbacks and integrations
        self.care_signal_callbacks: List[Callable] = []
        self.crisis_event_callbacks: List[Callable] = []
        self.healing_moment_callbacks: List[Callable] = []
        self.system_health_callbacks: List[Callable] = []
        
        # Monitoring state
        self.monitoring_active = False
        self.processing_tasks: Set[asyncio.Task] = set()
    
    def _initialize_care_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for detecting care signals"""
        return {
            'support_seeking': [
                r'\b(need|want|looking for)\s+(help|support|advice|guidance)\b',
                r'\b(struggling|difficult|hard time|overwhelmed)\b',
                r'\b(anyone|someone)\s+(who|that)\s+(can|could)\s+help\b',
                r'\b(feeling|feel)\s+(lost|alone|isolated|hopeless)\b'
            ],
            'emotional_distress': [
                r'\b(depressed|anxious|scared|worried|stressed)\b',
                r'\b(can\'t|cannot)\s+(cope|handle|deal)\b',
                r'\b(breaking down|falling apart|at my limit)\b',
                r'\b(tired of|exhausted|drained)\b'
            ],
            'support_offering': [
                r'\b(here for|available for|happy to help)\b',
                r'\b(if you need|reach out|DM me)\b',
                r'\b(offering|providing|sharing)\s+(support|help|advice)\b',
                r'\b(my experience|been through|understand)\b'
            ],
            'community_building': [
                r'\b(support group|community|circle|network)\b',
                r'\b(together|united|connected|belonging)\b',
                r'\b(safe space|welcoming|inclusive)\b',
                r'\b(join us|welcome|invite)\b'
            ],
            'gratitude_appreciation': [
                r'\b(thank you|grateful|appreciate|thankful)\b',
                r'\b(made my day|brightened|uplifted)\b',
                r'\b(blessed|fortunate|lucky)\b',
                r'\b(means a lot|significant|important)\b'
            ]
        }
    
    def _initialize_crisis_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for detecting crisis situations"""
        return {
            'suicidal_ideation': [
                r'\b(want to die|end it all|kill myself|suicide)\b',
                r'\b(better off dead|not worth living|can\'t go on)\b',
                r'\b(planning to|thinking about)\s+(suicide|ending)\b',
                r'\b(goodbye|final message|last time)\b'
            ],
            'self_harm': [
                r'\b(cutting|self harm|hurting myself)\b',
                r'\b(razor|blade|pills|overdose)\b',
                r'\b(want to hurt|need to hurt|deserve pain)\b'
            ],
            'immediate_danger': [
                r'\b(emergency|urgent|immediate help)\b',
                r'\b(in danger|unsafe|threatened)\b',
                r'\b(call 911|need police|ambulance)\b',
                r'\b(domestic violence|abuse|assault)\b'
            ],
            'severe_distress': [
                r'\b(panic attack|breakdown|crisis)\b',
                r'\b(can\'t breathe|heart racing|shaking)\b',
                r'\b(losing control|going crazy|breaking)\b'
            ]
        }
    
    def _initialize_healing_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for detecting healing moments"""
        return {
            'recovery_progress': [
                r'\b(getting better|improving|progress|healing)\b',
                r'\b(milestone|achievement|breakthrough)\b',
                r'\b(therapy|treatment|recovery)\s+(working|helping)\b',
                r'\b(feeling stronger|more hopeful|optimistic)\b'
            ],
            'wisdom_sharing': [
                r'\b(learned|discovered|realized|insight)\b',
                r'\b(advice|guidance|wisdom|experience)\b',
                r'\b(what helped me|in my experience)\b',
                r'\b(lesson|teaching|sharing)\b'
            ],
            'transformation': [
                r'\b(transformed|changed|different person)\b',
                r'\b(new beginning|fresh start|turning point)\b',
                r'\b(before and after|journey|evolution)\b'
            ],
            'hope_inspiration': [
                r'\b(hope|hopeful|optimistic|positive)\b',
                r'\b(inspired|motivated|encouraged)\b',
                r'\b(light|brightness|dawn|sunrise)\b',
                r'\b(possible|can do|will get better)\b'
            ]
        }
    
    def _initialize_noise_filters(self) -> Dict[str, List[str]]:
        """Initialize filters to reduce noise and false positives"""
        return {
            'promotional_content': [
                r'\b(buy|purchase|sale|discount|promo)\b',
                r'\b(click here|link in bio|DM for info)\b',
                r'\b(product|service|business|company)\b'
            ],
            'automated_content': [
                r'\b(bot|automated|scheduled|repost)\b',
                r'\b(generated|template|copy paste)\b'
            ],
            'irrelevant_context': [
                r'\b(movie|tv show|book|game|fiction)\b',
                r'\b(character|plot|story|narrative)\b'
            ]
        }
    
    async def add_stream(self, stream_config: StreamConfig) -> bool:
        """Add a new monitoring stream"""
        try:
            if len(self.active_streams) >= self.max_concurrent_streams:
                logger.warning(f"Maximum concurrent streams ({self.max_concurrent_streams}) reached")
                return False
            
            # Validate stream configuration
            if not await self._validate_stream_config(stream_config):
                logger.error(f"Invalid stream configuration: {stream_config.stream_id}")
                return False
            
            # Initialize stream metrics
            self.stream_metrics[stream_config.stream_id] = StreamMetrics(
                stream_id=stream_config.stream_id,
                events_processed=0,
                care_signals_detected=0,
                crisis_events_found=0,
                healing_moments_identified=0,
                processing_latency_ms=0.0,
                error_count=0,
                reconnection_count=0,
                last_event_time=datetime.utcnow(),
                health_status=StreamHealth.HEALTHY,
                throughput_per_minute=0.0,
                quality_score=0.0
            )
            
            # Store stream configuration
            self.active_streams[stream_config.stream_id] = stream_config
            
            # Start monitoring the stream if monitoring is active
            if self.monitoring_active:
                await self._start_stream_monitoring(stream_config)
            
            logger.info(f"Added monitoring stream: {stream_config.stream_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add stream {stream_config.stream_id}: {str(e)}")
            return False
    
    async def _validate_stream_config(self, config: StreamConfig) -> bool:
        """Validate stream configuration"""
        try:
            # Check required fields
            if not config.stream_id or not config.stream_type:
                return False
            
            # Check for duplicate stream IDs
            if config.stream_id in self.active_streams:
                logger.warning(f"Stream ID already exists: {config.stream_id}")
                return False
            
            # Validate connection parameters based on stream type
            required_params = {
                StreamType.BLUESKY_FIREHOSE: ['service_url', 'auth_token'],
                StreamType.MASTODON_STREAMING: ['instance_url', 'access_token'],
                StreamType.TWITTER_STREAM: ['api_key', 'api_secret', 'bearer_token'],
                StreamType.DISCORD_WEBHOOKS: ['webhook_url', 'auth_token'],
                StreamType.CUSTOM_WEBHOOKS: ['webhook_url'],
                StreamType.RSS_FEEDS: ['feed_url'],
                StreamType.API_POLLING: ['api_url', 'polling_interval']
            }
            
            if config.stream_type in required_params:
                required = required_params[config.stream_type]
                for param in required:
                    if param not in config.connection_params:
                        logger.error(f"Missing required parameter {param} for {config.stream_type}")
                        return False
            
            # Validate privacy settings
            if config.privacy_settings.get('requires_user_consent', True):
                if 'consent_mechanism' not in config.privacy_settings:
                    logger.warning(f"Stream {config.stream_id} requires consent but no mechanism specified")
            
            return True
            
        except Exception as e:
            logger.error(f"Stream configuration validation failed: {str(e)}")
            return False
    
    async def start_monitoring(self) -> bool:
        """Start monitoring all configured streams"""
        try:
            if self.monitoring_active:
                logger.warning("Monitoring already active")
                return True
            
            self.monitoring_active = True
            
            # Start event processing tasks
            await self._start_event_processors()
            
            # Start monitoring each configured stream
            for stream_config in self.active_streams.values():
                if stream_config.enabled:
                    await self._start_stream_monitoring(stream_config)
            
            # Start health monitoring
            await self._start_health_monitoring()
            
            logger.info("Firehose monitoring started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start monitoring: {str(e)}")
            self.monitoring_active = False
            return False
    
    async def stop_monitoring(self) -> bool:
        """Stop monitoring all streams"""
        try:
            self.monitoring_active = False
            
            # Stop all stream connections
            for stream_id, connection in self.stream_connections.items():
                try:
                    if hasattr(connection, 'close'):
                        await connection.close()
                except Exception as e:
                    logger.error(f"Error closing stream {stream_id}: {str(e)}")
            
            self.stream_connections.clear()
            
            # Cancel processing tasks
            for task in self.processing_tasks:
                if not task.done():
                    task.cancel()
            
            self.processing_tasks.clear()
            
            logger.info("Firehose monitoring stopped")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop monitoring: {str(e)}")
            return False
    
    async def _start_event_processors(self):
        """Start event processing tasks"""
        try:
            # Start priority-based processors
            for priority in MonitoringPriority:
                task = asyncio.create_task(self._process_priority_queue(priority))
                self.processing_tasks.add(task)
            
            # Start batch processor for efficiency
            task = asyncio.create_task(self._batch_event_processor())
            self.processing_tasks.add(task)
            
        except Exception as e:
            logger.error(f"Failed to start event processors: {str(e)}")
    
    async def _start_stream_monitoring(self, stream_config: StreamConfig):
        """Start monitoring a specific stream"""
        try:
            if stream_config.stream_type == StreamType.BLUESKY_FIREHOSE:
                task = asyncio.create_task(self._monitor_bluesky_firehose(stream_config))
            elif stream_config.stream_type == StreamType.MASTODON_STREAMING:
                task = asyncio.create_task(self._monitor_mastodon_stream(stream_config))
            elif stream_config.stream_type == StreamType.CUSTOM_WEBHOOKS:
                task = asyncio.create_task(self._monitor_webhook_stream(stream_config))
            elif stream_config.stream_type == StreamType.RSS_FEEDS:
                task = asyncio.create_task(self._monitor_rss_feed(stream_config))
            elif stream_config.stream_type == StreamType.API_POLLING:
                task = asyncio.create_task(self._monitor_api_polling(stream_config))
            else:
                logger.warning(f"Stream type {stream_config.stream_type} not yet implemented")
                return
            
            self.processing_tasks.add(task)
            
            # Store connection reference
            self.stream_connections[stream_config.stream_id] = task
            
        except Exception as e:
            logger.error(f"Failed to start stream monitoring for {stream_config.stream_id}: {str(e)}")
            self.stream_metrics[stream_config.stream_id].health_status = StreamHealth.FAILED
    
    async def _monitor_bluesky_firehose(self, stream_config: StreamConfig):
        """Monitor Bluesky AT Protocol firehose"""
        try:
            service_url = stream_config.connection_params['service_url']
            auth_token = stream_config.connection_params.get('auth_token')
            
            # WebSocket connection to Bluesky firehose
            firehose_url = f"{service_url}/xrpc/com.atproto.sync.subscribeRepos"
            
            headers = {}
            if auth_token:
                headers['Authorization'] = f'Bearer {auth_token}'
            
            while self.monitoring_active:
                try:
                    async with websockets.connect(firehose_url, extra_headers=headers) as websocket:
                        self.stream_metrics[stream_config.stream_id].health_status = StreamHealth.HEALTHY
                        
                        async for message in websocket:
                            if not self.monitoring_active:
                                break
                            
                            try:
                                # Parse Bluesky event
                                event_data = json.loads(message)
                                
                                # Convert to StreamEvent
                                stream_event = await self._parse_bluesky_event(
                                    event_data, stream_config.stream_id
                                )
                                
                                if stream_event:
                                    await self._process_stream_event(stream_event)
                                    
                            except Exception as e:
                                logger.error(f"Error processing Bluesky event: {str(e)}")
                                self.stream_metrics[stream_config.stream_id].error_count += 1
                                
                except websockets.exceptions.ConnectionClosed:
                    if self.monitoring_active:
                        logger.warning(f"Bluesky firehose connection closed, reconnecting...")
                        self.stream_metrics[stream_config.stream_id].health_status = StreamHealth.RECONNECTING
                        self.stream_metrics[stream_config.stream_id].reconnection_count += 1
                        await asyncio.sleep(5)  # Wait before reconnecting
                        
                except Exception as e:
                    logger.error(f"Bluesky firehose error: {str(e)}")
                    self.stream_metrics[stream_config.stream_id].health_status = StreamHealth.FAILED
                    if self.monitoring_active:
                        await asyncio.sleep(30)  # Wait longer on errors
                        
        except Exception as e:
            logger.error(f"Fatal error in Bluesky firehose monitoring: {str(e)}")
            self.stream_metrics[stream_config.stream_id].health_status = StreamHealth.FAILED
    
    async def _monitor_mastodon_stream(self, stream_config: StreamConfig):
        """Monitor Mastodon streaming API"""
        try:
            instance_url = stream_config.connection_params['instance_url']
            access_token = stream_config.connection_params['access_token']
            
            # Mastodon streaming endpoint
            stream_endpoint = f"{instance_url}/api/v1/streaming/public"
            
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Accept': 'text/event-stream'
            }
            
            while self.monitoring_active:
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(stream_endpoint, headers=headers) as response:
                            self.stream_metrics[stream_config.stream_id].health_status = StreamHealth.HEALTHY
                            
                            async for line in response.content:
                                if not self.monitoring_active:
                                    break
                                
                                try:
                                    line_str = line.decode('utf-8').strip()
                                    if line_str.startswith('data: '):
                                        event_data = json.loads(line_str[6:])
                                        
                                        # Convert to StreamEvent
                                        stream_event = await self._parse_mastodon_event(
                                            event_data, stream_config.stream_id
                                        )
                                        
                                        if stream_event:
                                            await self._process_stream_event(stream_event)
                                            
                                except Exception as e:
                                    logger.error(f"Error processing Mastodon event: {str(e)}")
                                    self.stream_metrics[stream_config.stream_id].error_count += 1
                                    
                except aiohttp.ClientError as e:
                    if self.monitoring_active:
                        logger.warning(f"Mastodon stream connection error, reconnecting: {str(e)}")
                        self.stream_metrics[stream_config.stream_id].health_status = StreamHealth.RECONNECTING
                        self.stream_metrics[stream_config.stream_id].reconnection_count += 1
                        await asyncio.sleep(10)
                        
        except Exception as e:
            logger.error(f"Fatal error in Mastodon stream monitoring: {str(e)}")
            self.stream_metrics[stream_config.stream_id].health_status = StreamHealth.FAILED
    
    async def _monitor_webhook_stream(self, stream_config: StreamConfig):
        """Monitor custom webhook stream (placeholder for webhook server)"""
        try:
            # This would integrate with a webhook server
            # For now, simulate webhook events
            webhook_url = stream_config.connection_params['webhook_url']
            
            while self.monitoring_active:
                try:
                    # Simulate receiving webhook events
                    await asyncio.sleep(5)  # Check every 5 seconds
                    
                    # In real implementation, this would receive actual webhook events
                    # For simulation, we'll create a dummy event occasionally
                    if datetime.utcnow().second % 30 == 0:  # Every 30 seconds
                        dummy_event = {
                            'content': 'Sample webhook content for testing',
                            'author': 'webhook_user',
                            'timestamp': datetime.utcnow().isoformat(),
                            'platform': 'custom_platform'
                        }
                        
                        stream_event = await self._parse_webhook_event(
                            dummy_event, stream_config.stream_id
                        )
                        
                        if stream_event:
                            await self._process_stream_event(stream_event)
                    
                    self.stream_metrics[stream_config.stream_id].health_status = StreamHealth.HEALTHY
                    
                except Exception as e:
                    logger.error(f"Error in webhook monitoring: {str(e)}")
                    self.stream_metrics[stream_config.stream_id].error_count += 1
                    await asyncio.sleep(30)
                    
        except Exception as e:
            logger.error(f"Fatal error in webhook monitoring: {str(e)}")
            self.stream_metrics[stream_config.stream_id].health_status = StreamHealth.FAILED
    
    async def _monitor_rss_feed(self, stream_config: StreamConfig):
        """Monitor RSS feed"""
        try:
            feed_url = stream_config.connection_params['feed_url']
            polling_interval = stream_config.connection_params.get('polling_interval', 300)  # 5 minutes default
            
            last_check = datetime.utcnow() - timedelta(hours=1)  # Start with 1 hour ago
            
            while self.monitoring_active:
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(feed_url) as response:
                            content = await response.text()
                            
                            # Parse RSS feed (simplified)
                            # In production, would use feedparser or similar
                            items = await self._parse_rss_content(content, last_check)
                            
                            for item in items:
                                stream_event = await self._parse_rss_event(
                                    item, stream_config.stream_id
                                )
                                
                                if stream_event:
                                    await self._process_stream_event(stream_event)
                            
                            last_check = datetime.utcnow()
                            self.stream_metrics[stream_config.stream_id].health_status = StreamHealth.HEALTHY
                    
                    await asyncio.sleep(polling_interval)
                    
                except Exception as e:
                    logger.error(f"Error in RSS feed monitoring: {str(e)}")
                    self.stream_metrics[stream_config.stream_id].error_count += 1
                    await asyncio.sleep(polling_interval)
                    
        except Exception as e:
            logger.error(f"Fatal error in RSS feed monitoring: {str(e)}")
            self.stream_metrics[stream_config.stream_id].health_status = StreamHealth.FAILED
    
    async def _monitor_api_polling(self, stream_config: StreamConfig):
        """Monitor API through polling"""
        try:
            api_url = stream_config.connection_params['api_url']
            polling_interval = stream_config.connection_params.get('polling_interval', 60)
            headers = stream_config.connection_params.get('headers', {})
            
            last_id = None
            
            while self.monitoring_active:
                try:
                    params = stream_config.connection_params.get('params', {})
                    if last_id:
                        params['since_id'] = last_id
                    
                    async with aiohttp.ClientSession() as session:
                        async with session.get(api_url, headers=headers, params=params) as response:
                            data = await response.json()
                            
                            events = data if isinstance(data, list) else data.get('data', [])
                            
                            for event_data in events:
                                stream_event = await self._parse_api_event(
                                    event_data, stream_config.stream_id
                                )
                                
                                if stream_event:
                                    await self._process_stream_event(stream_event)
                                    last_id = event_data.get('id', last_id)
                            
                            self.stream_metrics[stream_config.stream_id].health_status = StreamHealth.HEALTHY
                    
                    await asyncio.sleep(polling_interval)
                    
                except Exception as e:
                    logger.error(f"Error in API polling: {str(e)}")
                    self.stream_metrics[stream_config.stream_id].error_count += 1
                    await asyncio.sleep(polling_interval)
                    
        except Exception as e:
            logger.error(f"Fatal error in API polling: {str(e)}")
            self.stream_metrics[stream_config.stream_id].health_status = StreamHealth.FAILED
    
    async def _parse_bluesky_event(self, event_data: Dict[str, Any], stream_id: str) -> Optional[StreamEvent]:
        """Parse Bluesky firehose event into StreamEvent"""
        try:
            # Extract content from Bluesky event structure
            if 'commit' not in event_data:
                return None
            
            commit = event_data['commit']
            if 'ops' not in commit:
                return None
            
            for op in commit['ops']:
                if op.get('action') == 'create' and 'record' in op:
                    record = op['record']
                    
                    if record.get('$type') == 'app.bsky.feed.post':
                        content = record.get('text', '')
                        author_did = event_data.get('repo', 'unknown')
                        
                        # Analyze content for signals
                        detected_signals = await self._analyze_content_signals(content)
                        
                        # Calculate scores
                        care_score = await self._calculate_care_score(content, detected_signals)
                        crisis_score = await self._calculate_crisis_score(content, detected_signals)
                        healing_score = await self._calculate_healing_score(content, detected_signals)
                        
                        # Determine processing priority
                        priority = await self._determine_processing_priority(
                            care_score, crisis_score, healing_score
                        )
                        
                        return StreamEvent(
                            event_id=f"bluesky_{datetime.utcnow().isoformat()}_{hash(content)}",
                            stream_id=stream_id,
                            stream_type=StreamType.BLUESKY_FIREHOSE,
                            raw_content=json.dumps(event_data),
                            processed_content=content,
                            author_info={'did': author_did},
                            metadata={'commit': commit},
                            timestamp=datetime.utcnow(),
                            detected_signals=detected_signals,
                            care_score=care_score,
                            crisis_score=crisis_score,
                            healing_score=healing_score,
                            cultural_context=[],
                            privacy_level='public',
                            processing_priority=priority,
                            requires_human_review=crisis_score > 0.8
                        )
            
            return None
            
        except Exception as e:
            logger.error(f"Error parsing Bluesky event: {str(e)}")
            return None
    
    async def _parse_mastodon_event(self, event_data: Dict[str, Any], stream_id: str) -> Optional[StreamEvent]:
        """Parse Mastodon event into StreamEvent"""
        try:
            if event_data.get('event') != 'update':
                return None
            
            payload = json.loads(event_data.get('payload', '{}'))
            content = payload.get('content', '')
            
            # Strip HTML tags from Mastodon content
            content = re.sub(r'<[^>]+>', '', content)
            
            author_info = {
                'username': payload.get('account', {}).get('username', 'unknown'),
                'display_name': payload.get('account', {}).get('display_name', '')
            }
            
            # Analyze content
            detected_signals = await self._analyze_content_signals(content)
            care_score = await self._calculate_care_score(content, detected_signals)
            crisis_score = await self._calculate_crisis_score(content, detected_signals)
            healing_score = await self._calculate_healing_score(content, detected_signals)
            priority = await self._determine_processing_priority(care_score, crisis_score, healing_score)
            
            return StreamEvent(
                event_id=f"mastodon_{payload.get('id', datetime.utcnow().isoformat())}",
                stream_id=stream_id,
                stream_type=StreamType.MASTODON_STREAMING,
                raw_content=json.dumps(event_data),
                processed_content=content,
                author_info=author_info,
                metadata=payload,
                timestamp=datetime.utcnow(),
                detected_signals=detected_signals,
                care_score=care_score,
                crisis_score=crisis_score,
                healing_score=healing_score,
                cultural_context=[],
                privacy_level=payload.get('visibility', 'public'),
                processing_priority=priority,
                requires_human_review=crisis_score > 0.8
            )
            
        except Exception as e:
            logger.error(f"Error parsing Mastodon event: {str(e)}")
            return None
    
    async def _parse_webhook_event(self, event_data: Dict[str, Any], stream_id: str) -> Optional[StreamEvent]:
        """Parse webhook event into StreamEvent"""
        try:
            content = event_data.get('content', '')
            author_info = {'username': event_data.get('author', 'unknown')}
            
            detected_signals = await self._analyze_content_signals(content)
            care_score = await self._calculate_care_score(content, detected_signals)
            crisis_score = await self._calculate_crisis_score(content, detected_signals)
            healing_score = await self._calculate_healing_score(content, detected_signals)
            priority = await self._determine_processing_priority(care_score, crisis_score, healing_score)
            
            return StreamEvent(
                event_id=f"webhook_{datetime.utcnow().isoformat()}_{hash(content)}",
                stream_id=stream_id,
                stream_type=StreamType.CUSTOM_WEBHOOKS,
                raw_content=json.dumps(event_data),
                processed_content=content,
                author_info=author_info,
                metadata=event_data,
                timestamp=datetime.utcnow(),
                detected_signals=detected_signals,
                care_score=care_score,
                crisis_score=crisis_score,
                healing_score=healing_score,
                cultural_context=[],
                privacy_level='unknown',
                processing_priority=priority,
                requires_human_review=crisis_score > 0.8
            )
            
        except Exception as e:
            logger.error(f"Error parsing webhook event: {str(e)}")
            return None
    
    async def _parse_rss_content(self, content: str, since: datetime) -> List[Dict[str, Any]]:
        """Parse RSS content and extract recent items"""
        try:
            # Simplified RSS parsing - in production would use feedparser
            items = []
            
            # Extract items from RSS (basic regex parsing)
            item_pattern = r'<item>(.*?)</item>'
            title_pattern = r'<title>(.*?)</title>'
            description_pattern = r'<description>(.*?)</description>'
            pubdate_pattern = r'<pubDate>(.*?)</pubDate>'
            
            item_matches = re.findall(item_pattern, content, re.DOTALL)
            
            for item_content in item_matches:
                title_match = re.search(title_pattern, item_content)
                desc_match = re.search(description_pattern, item_content)
                date_match = re.search(pubdate_pattern, item_content)
                
                if title_match:
                    item = {
                        'title': title_match.group(1),
                        'description': desc_match.group(1) if desc_match else '',
                        'pub_date': date_match.group(1) if date_match else '',
                        'content': f"{title_match.group(1)} {desc_match.group(1) if desc_match else ''}"
                    }
                    items.append(item)
            
            return items
            
        except Exception as e:
            logger.error(f"Error parsing RSS content: {str(e)}")
            return []
    
    async def _parse_rss_event(self, item_data: Dict[str, Any], stream_id: str) -> Optional[StreamEvent]:
        """Parse RSS item into StreamEvent"""
        try:
            content = item_data.get('content', '')
            
            detected_signals = await self._analyze_content_signals(content)
            care_score = await self._calculate_care_score(content, detected_signals)
            crisis_score = await self._calculate_crisis_score(content, detected_signals)
            healing_score = await self._calculate_healing_score(content, detected_signals)
            priority = await self._determine_processing_priority(care_score, crisis_score, healing_score)
            
            return StreamEvent(
                event_id=f"rss_{datetime.utcnow().isoformat()}_{hash(content)}",
                stream_id=stream_id,
                stream_type=StreamType.RSS_FEEDS,
                raw_content=json.dumps(item_data),
                processed_content=content,
                author_info={'source': 'rss_feed'},
                metadata=item_data,
                timestamp=datetime.utcnow(),
                detected_signals=detected_signals,
                care_score=care_score,
                crisis_score=crisis_score,
                healing_score=healing_score,
                cultural_context=[],
                privacy_level='public',
                processing_priority=priority,
                requires_human_review=crisis_score > 0.8
            )
            
        except Exception as e:
            logger.error(f"Error parsing RSS event: {str(e)}")
            return None
    
    async def _parse_api_event(self, event_data: Dict[str, Any], stream_id: str) -> Optional[StreamEvent]:
        """Parse API polling event into StreamEvent"""
        try:
            content = event_data.get('text', event_data.get('content', ''))
            author_info = {
                'username': event_data.get('user', {}).get('username', 'unknown'),
                'id': event_data.get('user', {}).get('id', 'unknown')
            }
            
            detected_signals = await self._analyze_content_signals(content)
            care_score = await self._calculate_care_score(content, detected_signals)
            crisis_score = await self._calculate_crisis_score(content, detected_signals)
            healing_score = await self._calculate_healing_score(content, detected_signals)
            priority = await self._determine_processing_priority(care_score, crisis_score, healing_score)
            
            return StreamEvent(
                event_id=f"api_{event_data.get('id', datetime.utcnow().isoformat())}",
                stream_id=stream_id,
                stream_type=StreamType.API_POLLING,
                raw_content=json.dumps(event_data),
                processed_content=content,
                author_info=author_info,
                metadata=event_data,
                timestamp=datetime.utcnow(),
                detected_signals=detected_signals,
                care_score=care_score,
                crisis_score=crisis_score,
                healing_score=healing_score,
                cultural_context=[],
                privacy_level='public',
                processing_priority=priority,
                requires_human_review=crisis_score > 0.8
            )
            
        except Exception as e:
            logger.error(f"Error parsing API event: {str(e)}")
            return None
    
    async def _analyze_content_signals(self, content: str) -> List[str]:
        """Analyze content and detect care/crisis/healing signals"""
        try:
            signals = []
            content_lower = content.lower()
            
            # Check care patterns
            for signal_type, patterns in self.care_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, content_lower):
                        signals.append(f"care_{signal_type}")
                        break
            
            # Check crisis patterns
            for signal_type, patterns in self.crisis_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, content_lower):
                        signals.append(f"crisis_{signal_type}")
                        break
            
            # Check healing patterns
            for signal_type, patterns in self.healing_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, content_lower):
                        signals.append(f"healing_{signal_type}")
                        break
            
            return signals
            
        except Exception as e:
            logger.error(f"Error analyzing content signals: {str(e)}")
            return []
    
    async def _calculate_care_score(self, content: str, signals: List[str]) -> float:
        """Calculate care signal score for content"""
        try:
            care_signals = [s for s in signals if s.startswith('care_')]
            
            if not care_signals:
                return 0.0
            
            # Base score from number of signals
            base_score = min(0.7, len(care_signals) * 0.2)
            
            # Boost for specific high-value care signals
            high_value_signals = [
                'care_support_seeking', 'care_emotional_distress', 'care_support_offering'
            ]
            
            high_value_count = sum(1 for signal in care_signals if signal in high_value_signals)
            high_value_boost = high_value_count * 0.2
            
            # Content length consideration (longer posts might be more meaningful)
            length_factor = min(1.0, len(content) / 500) * 0.1
            
            total_score = base_score + high_value_boost + length_factor
            
            return min(1.0, total_score)
            
        except Exception as e:
            logger.error(f"Error calculating care score: {str(e)}")
            return 0.0
    
    async def _calculate_crisis_score(self, content: str, signals: List[str]) -> float:
        """Calculate crisis signal score for content"""
        try:
            crisis_signals = [s for s in signals if s.startswith('crisis_')]
            
            if not crisis_signals:
                return 0.0
            
            # Crisis scoring is more aggressive due to safety implications
            base_score = min(0.9, len(crisis_signals) * 0.3)
            
            # Immediate danger signals get maximum score
            if 'crisis_immediate_danger' in crisis_signals:
                return 1.0
            
            # Suicidal ideation gets very high score
            if 'crisis_suicidal_ideation' in crisis_signals:
                return 0.95
            
            # Self-harm gets high score
            if 'crisis_self_harm' in crisis_signals:
                return 0.9
            
            return base_score
            
        except Exception as e:
            logger.error(f"Error calculating crisis score: {str(e)}")
            return 0.0
    
    async def _calculate_healing_score(self, content: str, signals: List[str]) -> float:
        """Calculate healing signal score for content"""
        try:
            healing_signals = [s for s in signals if s.startswith('healing_')]
            
            if not healing_signals:
                return 0.0
            
            # Base score from healing signals
            base_score = min(0.8, len(healing_signals) * 0.25)
            
            # Boost for transformation and wisdom sharing
            high_impact_signals = [
                'healing_transformation', 'healing_wisdom_sharing'
            ]
            
            high_impact_count = sum(1 for signal in healing_signals if signal in high_impact_signals)
            impact_boost = high_impact_count * 0.2
            
            total_score = base_score + impact_boost
            
            return min(1.0, total_score)
            
        except Exception as e:
            logger.error(f"Error calculating healing score: {str(e)}")
            return 0.0
    
    async def _determine_processing_priority(self, care_score: float, 
                                           crisis_score: float, healing_score: float) -> MonitoringPriority:
        """Determine processing priority based on scores"""
        try:
            # Crisis takes absolute priority
            if crisis_score > 0.7:
                return MonitoringPriority.CRITICAL
            elif crisis_score > 0.4:
                return MonitoringPriority.HIGH
            
            # High care scores get priority
            if care_score > 0.7:
                return MonitoringPriority.HIGH
            elif care_score > 0.4:
                return MonitoringPriority.MEDIUM
            
            # Healing signals get medium priority
            if healing_score > 0.5:
                return MonitoringPriority.MEDIUM
            
            # Default to low priority
            return MonitoringPriority.LOW
            
        except Exception as e:
            logger.error(f"Error determining processing priority: {str(e)}")
            return MonitoringPriority.LOW
    
    async def _process_stream_event(self, event: StreamEvent):
        """Process a stream event through the priority system"""
        try:
            # Add to appropriate priority queue
            await self.priority_queues[event.processing_priority].put(event)
            
            # Update stream metrics
            if event.stream_id in self.stream_metrics:
                metrics = self.stream_metrics[event.stream_id]
                metrics.events_processed += 1
                metrics.last_event_time = event.timestamp
                
                if any('care_' in signal for signal in event.detected_signals):
                    metrics.care_signals_detected += 1
                
                if any('crisis_' in signal for signal in event.detected_signals):
                    metrics.crisis_events_found += 1
                
                if any('healing_' in signal for signal in event.detected_signals):
                    metrics.healing_moments_identified += 1
            
            # Add to event buffer
            self.event_buffer[event.stream_id].append(event)
            
            # Update global metrics
            self.total_events_processed += 1
            
            if any('care_' in signal for signal in event.detected_signals):
                self.total_care_signals_detected += 1
            
            if any('crisis_' in signal for signal in event.detected_signals):
                self.total_crisis_events_found += 1
            
            if any('healing_' in signal for signal in event.detected_signals):
                self.total_healing_moments_identified += 1
            
        except Exception as e:
            logger.error(f"Error processing stream event: {str(e)}")
    
    async def _process_priority_queue(self, priority: MonitoringPriority):
        """Process events from a specific priority queue"""
        try:
            queue = self.priority_queues[priority]
            
            while self.monitoring_active:
                try:
                    # Wait for event with timeout
                    event = await asyncio.wait_for(queue.get(), timeout=1.0)
                    
                    # Process the event
                    await self._handle_prioritized_event(event, priority)
                    
                    # Mark task as done
                    queue.task_done()
                    
                except asyncio.TimeoutError:
                    # No events in queue, continue waiting
                    continue
                except Exception as e:
                    logger.error(f"Error in priority queue {priority}: {str(e)}")
                    await asyncio.sleep(1)
                    
        except Exception as e:
            logger.error(f"Fatal error in priority queue processor {priority}: {str(e)}")
    
    async def _handle_prioritized_event(self, event: StreamEvent, priority: MonitoringPriority):
        """Handle an event based on its priority"""
        try:
            # Critical events (crisis) get immediate attention
            if priority == MonitoringPriority.CRITICAL:
                await self._handle_crisis_event(event)
            
            # High priority events get rapid processing
            elif priority == MonitoringPriority.HIGH:
                await self._handle_high_priority_event(event)
            
            # Medium priority events get standard processing
            elif priority == MonitoringPriority.MEDIUM:
                await self._handle_medium_priority_event(event)
            
            # Low priority events get batch processing
            elif priority == MonitoringPriority.LOW:
                await self._handle_low_priority_event(event)
            
            # Research events get analytical processing
            elif priority == MonitoringPriority.RESEARCH:
                await self._handle_research_event(event)
            
            # Trigger callbacks for relevant event processors
            await self._trigger_event_callbacks(event)
            
        except Exception as e:
            logger.error(f"Error handling prioritized event: {str(e)}")
    
    async def _handle_crisis_event(self, event: StreamEvent):
        """Handle crisis-level events with immediate response"""
        try:
            logger.critical(f"Crisis event detected: {event.event_id}")
            
            # Trigger all crisis callbacks immediately
            for callback in self.crisis_event_callbacks:
                try:
                    await callback(event)
                except Exception as e:
                    logger.error(f"Crisis callback failed: {str(e)}")
            
            # Log for human review
            logger.critical(f"Crisis content: {event.processed_content[:200]}...")
            
        except Exception as e:
            logger.error(f"Error handling crisis event: {str(e)}")
    
    async def _handle_high_priority_event(self, event: StreamEvent):
        """Handle high-priority events with rapid response"""
        try:
            # Trigger care signal callbacks
            for callback in self.care_signal_callbacks:
                try:
                    await callback(event)
                except Exception as e:
                    logger.error(f"Care signal callback failed: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error handling high priority event: {str(e)}")
    
    async def _handle_medium_priority_event(self, event: StreamEvent):
        """Handle medium-priority events with standard processing"""
        try:
            # Process healing moments
            if any('healing_' in signal for signal in event.detected_signals):
                for callback in self.healing_moment_callbacks:
                    try:
                        await callback(event)
                    except Exception as e:
                        logger.error(f"Healing moment callback failed: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error handling medium priority event: {str(e)}")
    
    async def _handle_low_priority_event(self, event: StreamEvent):
        """Handle low-priority events with batch processing"""
        try:
            # Low priority events are processed in batches for efficiency
            # For now, just log them
            if len(event.detected_signals) > 0:
                logger.debug(f"Low priority signals detected: {event.detected_signals}")
            
        except Exception as e:
            logger.error(f"Error handling low priority event: {str(e)}")
    
    async def _handle_research_event(self, event: StreamEvent):
        """Handle research events for analytics and learning"""
        try:
            # Research events contribute to analytics and pattern learning
            # For now, just track them
            logger.debug(f"Research event processed: {event.event_id}")
            
        except Exception as e:
            logger.error(f"Error handling research event: {str(e)}")
    
    async def _trigger_event_callbacks(self, event: StreamEvent):
        """Trigger appropriate callbacks based on event characteristics"""
        try:
            # Care signal callbacks
            if any('care_' in signal for signal in event.detected_signals):
                for callback in self.care_signal_callbacks:
                    try:
                        await callback(event)
                    except Exception as e:
                        logger.error(f"Care signal callback failed: {str(e)}")
            
            # Crisis event callbacks
            if any('crisis_' in signal for signal in event.detected_signals):
                for callback in self.crisis_event_callbacks:
                    try:
                        await callback(event)
                    except Exception as e:
                        logger.error(f"Crisis event callback failed: {str(e)}")
            
            # Healing moment callbacks
            if any('healing_' in signal for signal in event.detected_signals):
                for callback in self.healing_moment_callbacks:
                    try:
                        await callback(event)
                    except Exception as e:
                        logger.error(f"Healing moment callback failed: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error triggering event callbacks: {str(e)}")
    
    async def _batch_event_processor(self):
        """Process events in batches for efficiency"""
        try:
            batch_size = self.processing_batch_size
            batch_interval = 5  # Process every 5 seconds
            
            while self.monitoring_active:
                try:
                    await asyncio.sleep(batch_interval)
                    
                    # Collect events from all buffers
                    batch_events = []
                    
                    for stream_id, buffer in self.event_buffer.items():
                        while buffer and len(batch_events) < batch_size:
                            batch_events.append(buffer.popleft())
                    
                    if batch_events:
                        await self._process_event_batch(batch_events)
                    
                except Exception as e:
                    logger.error(f"Error in batch processor: {str(e)}")
                    await asyncio.sleep(batch_interval)
                    
        except Exception as e:
            logger.error(f"Fatal error in batch event processor: {str(e)}")
    
    async def _process_event_batch(self, events: List[StreamEvent]):
        """Process a batch of events for analytics and patterns"""
        try:
            # Batch processing for analytics, pattern detection, etc.
            logger.debug(f"Processing batch of {len(events)} events")
            
            # Update throughput metrics
            for event in events:
                if event.stream_id in self.stream_metrics:
                    # Update throughput calculation
                    metrics = self.stream_metrics[event.stream_id]
                    time_diff = (datetime.utcnow() - self.processing_start_time).total_seconds() / 60
                    if time_diff > 0:
                        metrics.throughput_per_minute = metrics.events_processed / time_diff
            
        except Exception as e:
            logger.error(f"Error processing event batch: {str(e)}")
    
    async def _start_health_monitoring(self):
        """Start monitoring system health"""
        try:
            task = asyncio.create_task(self._health_monitor_loop())
            self.processing_tasks.add(task)
            
        except Exception as e:
            logger.error(f"Failed to start health monitoring: {str(e)}")
    
    async def _health_monitor_loop(self):
        """Monitor system health continuously"""
        try:
            check_interval = 30  # Check every 30 seconds
            
            while self.monitoring_active:
                try:
                    await self._check_system_health()
                    await asyncio.sleep(check_interval)
                    
                except Exception as e:
                    logger.error(f"Error in health monitor: {str(e)}")
                    await asyncio.sleep(check_interval)
                    
        except Exception as e:
            logger.error(f"Fatal error in health monitor: {str(e)}")
    
    async def _check_system_health(self):
        """Check and update system health status"""
        try:
            current_time = datetime.utcnow()
            
            for stream_id, metrics in self.stream_metrics.items():
                # Check for stale streams
                time_since_last_event = (current_time - metrics.last_event_time).total_seconds()
                
                if time_since_last_event > 300:  # 5 minutes
                    if metrics.health_status == StreamHealth.HEALTHY:
                        metrics.health_status = StreamHealth.DEGRADED
                        logger.warning(f"Stream {stream_id} appears stale")
                
                if time_since_last_event > 600:  # 10 minutes
                    if metrics.health_status == StreamHealth.DEGRADED:
                        metrics.health_status = StreamHealth.FAILED
                        logger.error(f"Stream {stream_id} appears failed")
                
                # Calculate quality score
                if metrics.events_processed > 0:
                    signal_ratio = (metrics.care_signals_detected + 
                                  metrics.crisis_events_found + 
                                  metrics.healing_moments_identified) / metrics.events_processed
                    
                    error_ratio = metrics.error_count / max(1, metrics.events_processed)
                    
                    quality_score = max(0.0, signal_ratio - error_ratio)
                    metrics.quality_score = min(1.0, quality_score)
            
            # Trigger health callbacks
            for callback in self.system_health_callbacks:
                try:
                    await callback(self.stream_metrics)
                except Exception as e:
                    logger.error(f"Health callback failed: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error checking system health: {str(e)}")
    
    # Callback management
    def add_care_signal_callback(self, callback: Callable):
        """Add callback for care signal events"""
        self.care_signal_callbacks.append(callback)
    
    def add_crisis_event_callback(self, callback: Callable):
        """Add callback for crisis events"""
        self.crisis_event_callbacks.append(callback)
    
    def add_healing_moment_callback(self, callback: Callable):
        """Add callback for healing moments"""
        self.healing_moment_callbacks.append(callback)
    
    def add_system_health_callback(self, callback: Callable):
        """Add callback for system health updates"""
        self.system_health_callbacks.append(callback)
    
    # Analytics and reporting
    def get_monitoring_analytics(self) -> Dict[str, Any]:
        """Get comprehensive monitoring analytics"""
        try:
            uptime_seconds = (datetime.utcnow() - self.processing_start_time).total_seconds()
            uptime_hours = uptime_seconds / 3600
            
            # Stream health summary
            healthy_streams = sum(1 for m in self.stream_metrics.values() 
                                if m.health_status == StreamHealth.HEALTHY)
            total_streams = len(self.stream_metrics)
            
            # Average metrics
            if self.stream_metrics:
                avg_throughput = statistics.mean([m.throughput_per_minute for m in self.stream_metrics.values()])
                avg_quality = statistics.mean([m.quality_score for m in self.stream_metrics.values()])
                total_errors = sum(m.error_count for m in self.stream_metrics.values())
            else:
                avg_throughput = 0.0
                avg_quality = 0.0
                total_errors = 0
            
            return {
                'uptime_hours': uptime_hours,
                'total_events_processed': self.total_events_processed,
                'total_care_signals_detected': self.total_care_signals_detected,
                'total_crisis_events_found': self.total_crisis_events_found,
                'total_healing_moments_identified': self.total_healing_moments_identified,
                'active_streams': len(self.active_streams),
                'healthy_streams': healthy_streams,
                'stream_health_ratio': healthy_streams / max(1, total_streams),
                'average_throughput_per_minute': avg_throughput,
                'average_quality_score': avg_quality,
                'total_errors': total_errors,
                'events_per_hour': self.total_events_processed / max(1, uptime_hours),
                'care_signal_rate': self.total_care_signals_detected / max(1, self.total_events_processed),
                'crisis_detection_rate': self.total_crisis_events_found / max(1, self.total_events_processed),
                'healing_moment_rate': self.total_healing_moments_identified / max(1, self.total_events_processed)
            }
            
        except Exception as e:
            logger.error(f"Error generating monitoring analytics: {str(e)}")
            return {}
    
    def get_stream_analytics(self, stream_id: str) -> Optional[Dict[str, Any]]:
        """Get analytics for a specific stream"""
        try:
            if stream_id not in self.stream_metrics:
                return None
            
            metrics = self.stream_metrics[stream_id]
            config = self.active_streams.get(stream_id)
            
            return {
                'stream_id': stream_id,
                'stream_type': config.stream_type.value if config else 'unknown',
                'health_status': metrics.health_status.value,
                'events_processed': metrics.events_processed,
                'care_signals_detected': metrics.care_signals_detected,
                'crisis_events_found': metrics.crisis_events_found,
                'healing_moments_identified': metrics.healing_moments_identified,
                'error_count': metrics.error_count,
                'reconnection_count': metrics.reconnection_count,
                'throughput_per_minute': metrics.throughput_per_minute,
                'quality_score': metrics.quality_score,
                'last_event_time': metrics.last_event_time.isoformat(),
                'signal_detection_rate': (metrics.care_signals_detected + 
                                        metrics.crisis_events_found + 
                                        metrics.healing_moments_identified) / max(1, metrics.events_processed)
            }
            
        except Exception as e:
            logger.error(f"Error generating stream analytics for {stream_id}: {str(e)}")
            return None