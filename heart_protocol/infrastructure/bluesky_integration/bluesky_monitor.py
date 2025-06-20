"""
Bluesky Monitor for Care Signal Detection

Monitoring system for detecting care needs, support opportunities, and crisis
situations across the Bluesky network using trauma-informed approaches.
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import logging
import re
from collections import defaultdict, deque
import statistics

from .at_protocol_client import ATProtocolClient, RequestPriority

logger = logging.getLogger(__name__)


class MonitoringScope(Enum):
    """Scope of monitoring operations"""
    GLOBAL_FIREHOSE = "global_firehose"        # Monitor global activity stream
    TARGETED_KEYWORDS = "targeted_keywords"    # Monitor specific care-related keywords
    USER_TIMELINE = "user_timeline"           # Monitor specific user timelines
    COMMUNITY_FEEDS = "community_feeds"       # Monitor community/list feeds
    HASHTAG_MONITORING = "hashtag_monitoring" # Monitor specific hashtags
    REPLY_THREADS = "reply_threads"           # Monitor reply threads for support
    SEARCH_QUERIES = "search_queries"         # Active search for care signals


class CareSignal(Enum):
    """Types of care signals detected"""
    IMMEDIATE_CRISIS = "immediate_crisis"      # Suicidal ideation, self-harm
    MENTAL_HEALTH_CRISIS = "mental_health_crisis"  # Severe mental health episode
    SUPPORT_SEEKING = "support_seeking"       # Actively seeking help/support
    ISOLATION_LONELINESS = "isolation_loneliness"  # Expressing loneliness/isolation
    GRIEF_LOSS = "grief_loss"                # Processing grief or loss
    ANXIETY_STRESS = "anxiety_stress"         # High anxiety or stress
    DEPRESSION_INDICATORS = "depression_indicators"  # Signs of depression
    TRAUMA_PROCESSING = "trauma_processing"   # Processing trauma
    HEALING_PROGRESS = "healing_progress"     # Sharing healing journey
    OFFERING_SUPPORT = "offering_support"    # Offering help to others
    CELEBRATION_MILESTONE = "celebration_milestone"  # Celebrating progress
    COMMUNITY_BUILDING = "community_building"  # Building supportive community


class SignalConfidence(Enum):
    """Confidence levels for detected signals"""
    VERY_LOW = "very_low"      # 0.0-0.2 - Possible false positive
    LOW = "low"                # 0.2-0.4 - Low confidence
    MODERATE = "moderate"      # 0.4-0.6 - Moderate confidence
    HIGH = "high"              # 0.6-0.8 - High confidence
    VERY_HIGH = "very_high"    # 0.8-1.0 - Very high confidence


class MonitoringStrategy(Enum):
    """Strategies for monitoring content"""
    PASSIVE_OBSERVATION = "passive_observation"    # Observe without interaction
    GENTLE_ENGAGEMENT = "gentle_engagement"        # Gentle supportive responses
    ACTIVE_OUTREACH = "active_outreach"           # Proactive support outreach
    CRISIS_INTERVENTION = "crisis_intervention"    # Immediate crisis response
    COMMUNITY_AMPLIFICATION = "community_amplification"  # Amplify positive content


@dataclass
class CareSignalDetection:
    """Detected care signal with metadata"""
    signal_id: str
    user_handle: str
    user_did: str
    post_uri: str
    post_content: str
    signal_type: CareSignal
    confidence_score: float
    confidence_level: SignalConfidence
    detected_keywords: List[str]
    context_analysis: Dict[str, Any]
    emotional_intensity: float
    urgency_level: str
    cultural_considerations: List[str]
    privacy_sensitivity: str
    detected_at: datetime
    post_created_at: datetime
    user_profile_context: Dict[str, Any]
    intervention_recommended: bool
    intervention_type: str
    gentle_approach_required: bool
    trauma_informed_response: bool


@dataclass
class MonitoringPattern:
    """Pattern for monitoring specific types of content"""
    pattern_id: str
    care_signal: CareSignal
    keywords: List[str]
    regex_patterns: List[str]
    context_indicators: List[str]
    exclusion_patterns: List[str]
    confidence_weights: Dict[str, float]
    cultural_adaptations: Dict[str, List[str]]
    trauma_sensitive: bool
    requires_immediate_response: bool
    monitoring_strategy: MonitoringStrategy


class BlueSkyMonitor:
    """
    Monitoring system for detecting care needs across Bluesky network.
    
    Core Principles:
    - Trauma-informed detection respects user vulnerability
    - Privacy-first monitoring with explicit consent where possible
    - Cultural sensitivity guides interpretation of signals
    - False positive prevention protects users from unwanted intervention
    - Crisis signals get immediate priority with human handoff
    - Community care is fostered through gentle amplification
    - User agency is preserved in all monitoring activities
    - Healing-focused rather than engagement-focused monitoring
    """
    
    def __init__(self, at_client: ATProtocolClient, config: Dict[str, Any]):
        self.at_client = at_client
        self.config = config
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_scopes: Set[MonitoringScope] = set()
        self.monitored_users: Set[str] = set()
        self.monitored_keywords: Set[str] = set()
        
        # Detection patterns
        self.monitoring_patterns = self._initialize_monitoring_patterns()
        self.detection_history: List[CareSignalDetection] = []
        self.user_signal_history: Dict[str, List[CareSignalDetection]] = defaultdict(list)
        
        # Rate limiting and gentle monitoring
        self.monitoring_rate_limiter = {
            'max_requests_per_minute': 60,
            'max_posts_analyzed_per_minute': 100,
            'gentle_delay_ms': 2000,
            'crisis_bypass_enabled': True
        }
        
        # False positive prevention
        self.false_positive_tracking: Dict[str, int] = defaultdict(int)
        self.user_intervention_history: Dict[str, List[datetime]] = defaultdict(list)
        
        # Callbacks for different types of signals
        self.care_signal_callbacks: Dict[CareSignal, List[Callable]] = defaultdict(list)
        self.crisis_callbacks: List[Callable] = []
        self.community_care_callbacks: List[Callable] = []
        
        # Cultural and accessibility adaptation
        self.cultural_adaptations = self._initialize_cultural_adaptations()
        self.accessibility_features = self._initialize_accessibility_features()
        
        # Privacy and consent tracking
        self.user_consent_status: Dict[str, Dict[str, bool]] = defaultdict(dict)
        self.privacy_protection_active = True
    
    def _initialize_monitoring_patterns(self) -> Dict[CareSignal, MonitoringPattern]:
        """Initialize patterns for detecting different care signals"""
        
        patterns = {}
        
        # Immediate Crisis Pattern
        patterns[CareSignal.IMMEDIATE_CRISIS] = MonitoringPattern(
            pattern_id="immediate_crisis_v1",
            care_signal=CareSignal.IMMEDIATE_CRISIS,
            keywords=[
                "want to die", "kill myself", "end it all", "suicide", "can't go on",
                "hurt myself", "cutting", "overdose", "jump off", "hanging",
                "worthless", "burden", "everyone better without me"
            ],
            regex_patterns=[
                r"(?:want|going) to (?:die|kill myself)",
                r"(?:can't|cannot) (?:go on|take it)",
                r"(?:end|ending) it all",
                r"(?:hurt|harm) myself"
            ],
            context_indicators=[
                "feeling hopeless", "no way out", "final goodbye", "last post",
                "thank you for everything", "sorry everyone"
            ],
            exclusion_patterns=[
                "movie", "book", "game", "character", "fiction", "story",
                "lyrics", "song", "quote", "joking", "kidding"
            ],
            confidence_weights={
                'direct_statement': 0.9,
                'context_indicators': 0.7,
                'emotional_intensity': 0.8,
                'user_history': 0.6
            },
            cultural_adaptations={
                'expressions_of_pain': {
                    'western': ["can't take it", "want to end it"],
                    'collectivist': ["burden to family", "bringing shame"],
                    'spiritual': ["join the ancestors", "eternal rest"]
                }
            },
            trauma_sensitive=True,
            requires_immediate_response=True,
            monitoring_strategy=MonitoringStrategy.CRISIS_INTERVENTION
        )
        
        # Mental Health Crisis Pattern
        patterns[CareSignal.MENTAL_HEALTH_CRISIS] = MonitoringPattern(
            pattern_id="mental_health_crisis_v1",
            care_signal=CareSignal.MENTAL_HEALTH_CRISIS,
            keywords=[
                "panic attack", "can't breathe", "losing my mind", "going crazy",
                "breaking down", "falling apart", "can't function", "overwhelming",
                "spiraling", "dissociating", "manic episode", "psychotic"
            ],
            regex_patterns=[
                r"(?:panic|anxiety) attack",
                r"(?:losing|lost) my mind",
                r"(?:breaking|falling) (?:down|apart)",
                r"can't (?:function|cope|handle)"
            ],
            context_indicators=[
                "need help now", "emergency", "crisis mode", "can't handle this",
                "everything is too much", "system overload"
            ],
            exclusion_patterns=[
                "work stress", "exam stress", "normal anxiety", "just tired"
            ],
            confidence_weights={
                'crisis_keywords': 0.8,
                'context_severity': 0.7,
                'temporal_clustering': 0.6,
                'user_pattern': 0.5
            },
            cultural_adaptations={},
            trauma_sensitive=True,
            requires_immediate_response=True,
            monitoring_strategy=MonitoringStrategy.ACTIVE_OUTREACH
        )
        
        # Support Seeking Pattern
        patterns[CareSignal.SUPPORT_SEEKING] = MonitoringPattern(
            pattern_id="support_seeking_v1",
            care_signal=CareSignal.SUPPORT_SEEKING,
            keywords=[
                "need help", "looking for support", "anyone else", "advice needed",
                "going through", "struggling with", "how do you", "has anyone",
                "support group", "therapy", "counseling", "resources"
            ],
            regex_patterns=[
                r"(?:need|looking for|seeking) (?:help|support|advice)",
                r"(?:anyone|has anyone) (?:else|experienced|been through)",
                r"(?:struggling|dealing) with",
                r"how (?:do you|did you) (?:cope|handle|deal)"
            ],
            context_indicators=[
                "first time", "don't know what to do", "feeling lost",
                "any suggestions", "please help", "desperate"
            ],
            exclusion_patterns=[
                "technical help", "homework help", "work advice", "relationship drama"
            ],
            confidence_weights={
                'explicit_request': 0.8,
                'vulnerability_indicators': 0.7,
                'community_seeking': 0.6,
                'emotional_openness': 0.5
            },
            cultural_adaptations={},
            trauma_sensitive=True,
            requires_immediate_response=False,
            monitoring_strategy=MonitoringStrategy.GENTLE_ENGAGEMENT
        )
        
        # Isolation and Loneliness Pattern
        patterns[CareSignal.ISOLATION_LONELINESS] = MonitoringPattern(
            pattern_id="isolation_loneliness_v1",
            care_signal=CareSignal.ISOLATION_LONELINESS,
            keywords=[
                "so lonely", "no friends", "all alone", "nobody cares", "isolated",
                "disconnected", "empty house", "silence", "no one to talk to",
                "forgotten", "invisible", "doesn't matter"
            ],
            regex_patterns=[
                r"(?:so|very|really) lonely",
                r"(?:no|don't have) (?:friends|anyone|one)",
                r"(?:all|completely) alone",
                r"no one (?:cares|understands|listens)"
            ],
            context_indicators=[
                "weekends are hard", "holidays hurt", "seeing couples",
                "everyone has plans", "just me again"
            ],
            exclusion_patterns=[
                "enjoying solitude", "choosing to be alone", "introvert life",
                "peaceful alone time"
            ],
            confidence_weights={
                'loneliness_expressions': 0.7,
                'social_isolation': 0.6,
                'emotional_pain': 0.8,
                'temporal_pattern': 0.5
            },
            cultural_adaptations={},
            trauma_sensitive=True,
            requires_immediate_response=False,
            monitoring_strategy=MonitoringStrategy.GENTLE_ENGAGEMENT
        )
        
        # Healing Progress Pattern
        patterns[CareSignal.HEALING_PROGRESS] = MonitoringPattern(
            pattern_id="healing_progress_v1",
            care_signal=CareSignal.HEALING_PROGRESS,
            keywords=[
                "getting better", "healing journey", "therapy helping", "progress",
                "small wins", "breakthrough", "growing", "learning", "recovering",
                "stronger", "grateful", "hope", "light at the end"
            ],
            regex_patterns=[
                r"(?:getting|feeling) better",
                r"(?:healing|recovery) (?:journey|process)",
                r"(?:small|little) (?:wins|victories|steps)",
                r"making progress"
            ],
            context_indicators=[
                "therapist said", "support group", "medication helping",
                "coping strategies", "self care", "boundaries"
            ],
            exclusion_patterns=[
                "fake it till you make it", "pretending", "putting on a face"
            ],
            confidence_weights={
                'positive_indicators': 0.8,
                'growth_language': 0.7,
                'professional_support': 0.6,
                'sustained_pattern': 0.9
            },
            cultural_adaptations={},
            trauma_sensitive=False,
            requires_immediate_response=False,
            monitoring_strategy=MonitoringStrategy.COMMUNITY_AMPLIFICATION
        )
        
        # Add more patterns for other care signals...
        
        return patterns
    
    def _initialize_cultural_adaptations(self) -> Dict[str, Dict[str, Any]]:
        """Initialize cultural adaptations for monitoring"""
        return {
            'expression_styles': {
                'direct': ['western', 'individualistic'],
                'indirect': ['collectivistic', 'high_context'],
                'metaphorical': ['spiritual', 'indigenous'],
                'somatic': ['embodied', 'traditional_healing']
            },
            'help_seeking_patterns': {
                'individual_focused': ['personal therapy', 'self-help'],
                'family_oriented': ['family involvement', 'collective healing'],
                'community_based': ['support groups', 'peer support'],
                'spiritual_practices': ['prayer', 'ritual', 'ceremony']
            }
        }
    
    def _initialize_accessibility_features(self) -> Dict[str, Any]:
        """Initialize accessibility features for monitoring"""
        return {
            'content_analysis': {
                'alt_text_detection': True,
                'caption_analysis': True,
                'audio_description_check': True,
                'sign_language_detection': True
            },
            'communication_adaptations': {
                'simple_language_preferred': True,
                'visual_cues_important': True,
                'audio_alternatives_needed': True,
                'cognitive_load_considerations': True
            }
        }
    
    async def start_monitoring(self, scopes: List[MonitoringScope],
                             keywords: Optional[List[str]] = None,
                             users: Optional[List[str]] = None):
        """
        Start monitoring Bluesky for care signals
        
        Args:
            scopes: List of monitoring scopes to activate
            keywords: Specific keywords to monitor (for targeted monitoring)
            users: Specific users to monitor (for user timeline monitoring)
        """
        try:
            if self.monitoring_active:
                logger.warning("Monitoring already active")
                return
            
            self.monitoring_active = True
            self.monitoring_scopes = set(scopes)
            
            if keywords:
                self.monitored_keywords.update(keywords)
            
            if users:
                self.monitored_users.update(users)
            
            logger.info(f"Starting Bluesky monitoring with scopes: {[s.value for s in scopes]}")
            
            # Start monitoring tasks based on scopes
            tasks = []
            
            if MonitoringScope.GLOBAL_FIREHOSE in scopes:
                tasks.append(asyncio.create_task(self._monitor_global_firehose()))
            
            if MonitoringScope.TARGETED_KEYWORDS in scopes:
                tasks.append(asyncio.create_task(self._monitor_targeted_keywords()))
            
            if MonitoringScope.USER_TIMELINE in scopes:
                tasks.append(asyncio.create_task(self._monitor_user_timelines()))
            
            if MonitoringScope.SEARCH_QUERIES in scopes:
                tasks.append(asyncio.create_task(self._monitor_search_queries()))
            
            # Start care signal processing
            tasks.append(asyncio.create_task(self._process_care_signals()))
            
            # Start false positive monitoring
            tasks.append(asyncio.create_task(self._monitor_false_positives()))
            
            # Wait for tasks to complete or monitoring to stop
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"Failed to start monitoring: {str(e)}")
            self.monitoring_active = False
    
    async def stop_monitoring(self):
        """Stop all monitoring activities"""
        try:
            self.monitoring_active = False
            logger.info("Stopping Bluesky monitoring")
            
            # Clear monitoring state
            self.monitoring_scopes.clear()
            self.monitored_keywords.clear()
            self.monitored_users.clear()
            
        except Exception as e:
            logger.error(f"Error stopping monitoring: {str(e)}")
    
    async def _monitor_global_firehose(self):
        """Monitor global Bluesky firehose for care signals"""
        while self.monitoring_active:
            try:
                # Note: This would connect to the AT Protocol firehose
                # For now, we'll simulate with timeline monitoring
                response = await self.at_client.get_timeline(limit=50)
                
                if response.success and response.data:
                    posts = response.data.get('feed', [])
                    
                    for post_data in posts:
                        await self._analyze_post_for_care_signals(post_data)
                
                # Gentle delay between firehose checks
                await asyncio.sleep(self.monitoring_rate_limiter['gentle_delay_ms'] / 1000)
                
            except Exception as e:
                logger.error(f"Global firehose monitoring error: {str(e)}")
                await asyncio.sleep(60)  # Error recovery delay
    
    async def _monitor_targeted_keywords(self):
        """Monitor for specific care-related keywords"""
        while self.monitoring_active:
            try:
                # Search for each monitored keyword
                for keyword in self.monitored_keywords:
                    if not self.monitoring_active:
                        break
                    
                    response = await self.at_client.search_posts(keyword, limit=25)
                    
                    if response.success and response.data:
                        posts = response.data.get('posts', [])
                        
                        for post_data in posts:
                            await self._analyze_post_for_care_signals(post_data, keyword_triggered=keyword)
                    
                    # Gentle delay between keyword searches
                    await asyncio.sleep(5)
                
                # Longer delay between full keyword cycles
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"Targeted keyword monitoring error: {str(e)}")
                await asyncio.sleep(180)
    
    async def _monitor_user_timelines(self):
        """Monitor specific user timelines for care signals"""
        while self.monitoring_active:
            try:
                for user_handle in self.monitored_users:
                    if not self.monitoring_active:
                        break
                    
                    response = await self.at_client.get_author_feed(user_handle, limit=20)
                    
                    if response.success and response.data:
                        posts = response.data.get('feed', [])
                        
                        for post_data in posts:
                            await self._analyze_post_for_care_signals(post_data, monitored_user=user_handle)
                    
                    # Gentle delay between user timeline checks
                    await asyncio.sleep(10)
                
                # Longer delay between full user cycles
                await asyncio.sleep(600)  # 10 minutes
                
            except Exception as e:
                logger.error(f"User timeline monitoring error: {str(e)}")
                await asyncio.sleep(300)
    
    async def _monitor_search_queries(self):
        """Actively search for care signals using predefined queries"""
        while self.monitoring_active:
            try:
                # Care-focused search queries
                care_queries = [
                    "need help depression",
                    "anxiety support",
                    "feeling suicidal",
                    "mental health crisis",
                    "therapy resources",
                    "support group",
                    "healing journey",
                    "recovery progress"
                ]
                
                for query in care_queries:
                    if not self.monitoring_active:
                        break
                    
                    response = await self.at_client.search_posts(query, limit=20)
                    
                    if response.success and response.data:
                        posts = response.data.get('posts', [])
                        
                        for post_data in posts:
                            await self._analyze_post_for_care_signals(post_data, search_query=query)
                    
                    # Gentle delay between searches
                    await asyncio.sleep(15)
                
                # Longer delay between search cycles
                await asyncio.sleep(900)  # 15 minutes
                
            except Exception as e:
                logger.error(f"Search query monitoring error: {str(e)}")
                await asyncio.sleep(300)
    
    async def _analyze_post_for_care_signals(self, post_data: Dict[str, Any],
                                           keyword_triggered: Optional[str] = None,
                                           monitored_user: Optional[str] = None,
                                           search_query: Optional[str] = None):
        """Analyze a post for care signals"""
        try:
            # Extract post information
            post_record = post_data.get('post', {}).get('record', {})
            post_author = post_data.get('post', {}).get('author', {})
            
            post_text = post_record.get('text', '')
            user_handle = post_author.get('handle', '')
            user_did = post_author.get('did', '')
            post_uri = post_data.get('post', {}).get('uri', '')
            post_created_at = post_record.get('createdAt', '')
            
            if not post_text or not user_handle:
                return
            
            # Skip if we've already analyzed this post recently
            if await self._is_recently_analyzed(post_uri):
                return
            
            # Apply monitoring patterns
            detected_signals = []
            
            for care_signal, pattern in self.monitoring_patterns.items():
                signal_detection = await self._apply_monitoring_pattern(
                    pattern, post_text, post_data, keyword_triggered, search_query
                )
                
                if signal_detection:
                    detected_signals.append(signal_detection)
            
            # Process detected signals
            for detection in detected_signals:
                detection.user_handle = user_handle
                detection.user_did = user_did
                detection.post_uri = post_uri
                detection.post_content = post_text
                detection.detected_at = datetime.utcnow()
                detection.post_created_at = datetime.fromisoformat(post_created_at.replace('Z', '+00:00')) if post_created_at else datetime.utcnow()
                
                # Get user profile context
                detection.user_profile_context = await self._get_user_context(user_handle)
                
                # Apply cultural and accessibility considerations
                detection = await self._apply_cultural_analysis(detection)
                detection = await self._apply_accessibility_analysis(detection)
                
                # Determine intervention recommendations
                detection = await self._determine_intervention_recommendations(detection)
                
                # Store detection
                self.detection_history.append(detection)
                self.user_signal_history[user_handle].append(detection)
                
                # Trigger appropriate callbacks
                await self._trigger_care_signal_callbacks(detection)
                
                logger.info(f"Detected {detection.signal_type.value} signal from {user_handle} with confidence {detection.confidence_score:.2f}")
            
        except Exception as e:
            logger.error(f"Failed to analyze post for care signals: {str(e)}")
    
    async def _apply_monitoring_pattern(self, pattern: MonitoringPattern,
                                      post_text: str, post_data: Dict[str, Any],
                                      keyword_triggered: Optional[str] = None,
                                      search_query: Optional[str] = None) -> Optional[CareSignalDetection]:
        """Apply a monitoring pattern to detect care signals"""
        try:
            post_text_lower = post_text.lower()
            confidence_score = 0.0
            detected_keywords = []
            
            # Check for keyword matches
            keyword_matches = 0
            for keyword in pattern.keywords:
                if keyword.lower() in post_text_lower:
                    detected_keywords.append(keyword)
                    keyword_matches += 1
            
            if keyword_matches > 0:
                confidence_score += pattern.confidence_weights.get('direct_statement', 0.5) * (keyword_matches / len(pattern.keywords))
            
            # Check regex patterns
            regex_matches = 0
            for regex_pattern in pattern.regex_patterns:
                if re.search(regex_pattern, post_text_lower):
                    regex_matches += 1
            
            if regex_matches > 0:
                confidence_score += pattern.confidence_weights.get('context_indicators', 0.4) * (regex_matches / len(pattern.regex_patterns))
            
            # Check context indicators
            context_matches = 0
            for context_indicator in pattern.context_indicators:
                if context_indicator.lower() in post_text_lower:
                    context_matches += 1
            
            if context_matches > 0:
                confidence_score += pattern.confidence_weights.get('emotional_intensity', 0.3) * (context_matches / len(pattern.context_indicators))
            
            # Check exclusion patterns (reduce confidence)
            exclusion_matches = 0
            for exclusion_pattern in pattern.exclusion_patterns:
                if exclusion_pattern.lower() in post_text_lower:
                    exclusion_matches += 1
            
            if exclusion_matches > 0:
                confidence_score *= (1.0 - (exclusion_matches * 0.2))  # Reduce confidence
            
            # Boost confidence if triggered by keyword or search
            if keyword_triggered or search_query:
                confidence_score *= 1.2
            
            # Minimum threshold for detection
            if confidence_score < 0.3:
                return None
            
            # Cap confidence at 1.0
            confidence_score = min(1.0, confidence_score)
            
            # Calculate emotional intensity
            emotional_intensity = await self._calculate_emotional_intensity(post_text, pattern)
            
            # Determine confidence level
            confidence_level = self._get_confidence_level(confidence_score)
            
            # Create detection
            detection = CareSignalDetection(
                signal_id=f"signal_{pattern.care_signal.value}_{datetime.utcnow().isoformat()}",
                user_handle="",  # Will be filled in by caller
                user_did="",
                post_uri="",
                post_content=post_text,
                signal_type=pattern.care_signal,
                confidence_score=confidence_score,
                confidence_level=confidence_level,
                detected_keywords=detected_keywords,
                context_analysis={
                    'keyword_matches': keyword_matches,
                    'regex_matches': regex_matches,
                    'context_matches': context_matches,
                    'exclusion_matches': exclusion_matches,
                    'keyword_triggered': keyword_triggered,
                    'search_query': search_query
                },
                emotional_intensity=emotional_intensity,
                urgency_level=self._determine_urgency_level(pattern, confidence_score, emotional_intensity),
                cultural_considerations=[],  # Will be filled later
                privacy_sensitivity=self._determine_privacy_sensitivity(pattern),
                detected_at=datetime.utcnow(),
                post_created_at=datetime.utcnow(),  # Will be updated
                user_profile_context={},  # Will be filled later
                intervention_recommended=pattern.requires_immediate_response,
                intervention_type=pattern.monitoring_strategy.value,
                gentle_approach_required=pattern.trauma_sensitive,
                trauma_informed_response=pattern.trauma_sensitive
            )
            
            return detection
            
        except Exception as e:
            logger.error(f"Failed to apply monitoring pattern: {str(e)}")
            return None
    
    async def _calculate_emotional_intensity(self, post_text: str, pattern: MonitoringPattern) -> float:
        """Calculate emotional intensity of the post"""
        try:
            intensity = 0.5  # Base intensity
            
            # High intensity indicators
            high_intensity_words = [
                'extremely', 'completely', 'totally', 'absolutely', 'desperately',
                'unbearable', 'overwhelming', 'devastating', 'crushing', 'destroying'
            ]
            
            for word in high_intensity_words:
                if word in post_text.lower():
                    intensity += 0.1
            
            # Punctuation intensity indicators
            exclamation_count = post_text.count('!')
            question_count = post_text.count('?')
            caps_ratio = sum(1 for c in post_text if c.isupper()) / max(1, len(post_text))
            
            intensity += min(0.2, exclamation_count * 0.05)
            intensity += min(0.1, question_count * 0.03)
            intensity += min(0.2, caps_ratio * 0.5)
            
            # Crisis signals increase intensity
            if pattern.care_signal in [CareSignal.IMMEDIATE_CRISIS, CareSignal.MENTAL_HEALTH_CRISIS]:
                intensity *= 1.5
            
            return min(1.0, intensity)
            
        except Exception as e:
            logger.error(f"Failed to calculate emotional intensity: {str(e)}")
            return 0.5
    
    def _get_confidence_level(self, confidence_score: float) -> SignalConfidence:
        """Convert confidence score to confidence level"""
        if confidence_score >= 0.8:
            return SignalConfidence.VERY_HIGH
        elif confidence_score >= 0.6:
            return SignalConfidence.HIGH
        elif confidence_score >= 0.4:
            return SignalConfidence.MODERATE
        elif confidence_score >= 0.2:
            return SignalConfidence.LOW
        else:
            return SignalConfidence.VERY_LOW
    
    def _determine_urgency_level(self, pattern: MonitoringPattern, confidence_score: float, emotional_intensity: float) -> str:
        """Determine urgency level for response"""
        if pattern.care_signal == CareSignal.IMMEDIATE_CRISIS:
            return "immediate"
        elif pattern.care_signal == CareSignal.MENTAL_HEALTH_CRISIS:
            return "urgent"
        elif confidence_score > 0.7 and emotional_intensity > 0.7:
            return "high"
        elif confidence_score > 0.5:
            return "moderate"
        else:
            return "low"
    
    def _determine_privacy_sensitivity(self, pattern: MonitoringPattern) -> str:
        """Determine privacy sensitivity level"""
        if pattern.care_signal in [CareSignal.IMMEDIATE_CRISIS, CareSignal.MENTAL_HEALTH_CRISIS]:
            return "very_high"
        elif pattern.trauma_sensitive:
            return "high"
        else:
            return "moderate"
    
    async def _is_recently_analyzed(self, post_uri: str) -> bool:
        """Check if post was recently analyzed to avoid duplicates"""
        cutoff_time = datetime.utcnow() - timedelta(hours=1)
        
        for detection in self.detection_history:
            if (detection.post_uri == post_uri and 
                detection.detected_at > cutoff_time):
                return True
        
        return False
    
    async def _get_user_context(self, user_handle: str) -> Dict[str, Any]:
        """Get user profile context for better signal interpretation"""
        try:
            # Get basic profile
            response = await self.at_client.get_profile(user_handle)
            
            if response.success and response.data:
                profile_data = response.data
                
                # Extract relevant context
                context = {
                    'display_name': profile_data.get('displayName', ''),
                    'description': profile_data.get('description', ''),
                    'followers_count': profile_data.get('followersCount', 0),
                    'posts_count': profile_data.get('postsCount', 0),
                    'created_at': profile_data.get('createdAt', ''),
                    'profile_analysis': await self._analyze_profile_for_care_context(profile_data)
                }
                
                return context
            
            return {}
            
        except Exception as e:
            logger.error(f"Failed to get user context: {str(e)}")
            return {}
    
    async def _analyze_profile_for_care_context(self, profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze user profile for care-relevant context"""
        try:
            description = profile_data.get('description', '').lower()
            
            analysis = {
                'mental_health_awareness': False,
                'support_seeking_indicators': False,
                'helping_others_indicators': False,
                'professional_helper': False,
                'recovery_journey': False
            }
            
            # Mental health awareness indicators
            mental_health_keywords = ['mental health', 'depression', 'anxiety', 'therapy', 'counseling']
            analysis['mental_health_awareness'] = any(keyword in description for keyword in mental_health_keywords)
            
            # Support seeking indicators
            support_keywords = ['looking for support', 'need help', 'support group', 'peer support']
            analysis['support_seeking_indicators'] = any(keyword in description for keyword in support_keywords)
            
            # Helping others indicators
            helping_keywords = ['here to help', 'support others', 'mental health advocate', 'peer counselor']
            analysis['helping_others_indicators'] = any(keyword in description for keyword in helping_keywords)
            
            # Professional helper indicators
            professional_keywords = ['therapist', 'counselor', 'psychologist', 'social worker', 'crisis counselor']
            analysis['professional_helper'] = any(keyword in description for keyword in professional_keywords)
            
            # Recovery journey indicators
            recovery_keywords = ['recovery', 'healing journey', 'survivor', 'in recovery', 'sober']
            analysis['recovery_journey'] = any(keyword in description for keyword in recovery_keywords)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze profile for care context: {str(e)}")
            return {}
    
    async def _apply_cultural_analysis(self, detection: CareSignalDetection) -> CareSignalDetection:
        """Apply cultural analysis to detection"""
        # Placeholder for cultural analysis implementation
        detection.cultural_considerations = ['general_western_context']
        return detection
    
    async def _apply_accessibility_analysis(self, detection: CareSignalDetection) -> CareSignalDetection:
        """Apply accessibility analysis to detection"""
        # Placeholder for accessibility analysis implementation
        return detection
    
    async def _determine_intervention_recommendations(self, detection: CareSignalDetection) -> CareSignalDetection:
        """Determine intervention recommendations based on detection"""
        try:
            # Crisis signals always recommend intervention
            if detection.signal_type in [CareSignal.IMMEDIATE_CRISIS, CareSignal.MENTAL_HEALTH_CRISIS]:
                detection.intervention_recommended = True
                detection.intervention_type = "crisis_intervention"
                return detection
            
            # High confidence signals recommend gentle intervention
            if detection.confidence_score > 0.7:
                detection.intervention_recommended = True
                detection.intervention_type = "gentle_outreach"
            
            # Support seeking signals recommend resource sharing
            elif detection.signal_type == CareSignal.SUPPORT_SEEKING:
                detection.intervention_recommended = True
                detection.intervention_type = "resource_sharing"
            
            # Healing progress signals recommend amplification
            elif detection.signal_type == CareSignal.HEALING_PROGRESS:
                detection.intervention_recommended = True
                detection.intervention_type = "positive_amplification"
            
            else:
                detection.intervention_recommended = False
                detection.intervention_type = "monitoring_only"
            
            return detection
            
        except Exception as e:
            logger.error(f"Failed to determine intervention recommendations: {str(e)}")
            return detection
    
    async def _trigger_care_signal_callbacks(self, detection: CareSignalDetection):
        """Trigger appropriate callbacks for detected care signals"""
        try:
            # Trigger signal-specific callbacks
            signal_callbacks = self.care_signal_callbacks.get(detection.signal_type, [])
            for callback in signal_callbacks:
                try:
                    await callback(detection)
                except Exception as e:
                    logger.error(f"Care signal callback failed: {str(e)}")
            
            # Trigger crisis callbacks for crisis signals
            if (detection.signal_type in [CareSignal.IMMEDIATE_CRISIS, CareSignal.MENTAL_HEALTH_CRISIS] and
                detection.confidence_score > 0.6):
                for callback in self.crisis_callbacks:
                    try:
                        await callback(detection)
                    except Exception as e:
                        logger.error(f"Crisis callback failed: {str(e)}")
            
            # Trigger community care callbacks for positive signals
            if detection.signal_type in [CareSignal.HEALING_PROGRESS, CareSignal.OFFERING_SUPPORT]:
                for callback in self.community_care_callbacks:
                    try:
                        await callback(detection)
                    except Exception as e:
                        logger.error(f"Community care callback failed: {str(e)}")
                        
        except Exception as e:
            logger.error(f"Failed to trigger care signal callbacks: {str(e)}")
    
    async def _process_care_signals(self):
        """Process detected care signals for patterns and follow-up"""
        while self.monitoring_active:
            try:
                # Clean up old detections (keep last 7 days)
                cutoff_time = datetime.utcnow() - timedelta(days=7)
                self.detection_history = [
                    detection for detection in self.detection_history
                    if detection.detected_at > cutoff_time
                ]
                
                # Clean up user signal history
                for user_handle in list(self.user_signal_history.keys()):
                    self.user_signal_history[user_handle] = [
                        detection for detection in self.user_signal_history[user_handle]
                        if detection.detected_at > cutoff_time
                    ]
                    
                    if not self.user_signal_history[user_handle]:
                        del self.user_signal_history[user_handle]
                
                await asyncio.sleep(3600)  # Process every hour
                
            except Exception as e:
                logger.error(f"Care signal processing error: {str(e)}")
                await asyncio.sleep(1800)  # Wait 30 minutes on error
    
    async def _monitor_false_positives(self):
        """Monitor for false positive patterns to improve detection"""
        while self.monitoring_active:
            try:
                # Analyze false positive patterns
                # This would integrate with user feedback systems
                await asyncio.sleep(7200)  # Check every 2 hours
                
            except Exception as e:
                logger.error(f"False positive monitoring error: {str(e)}")
                await asyncio.sleep(3600)
    
    # Callback management methods
    
    def add_care_signal_callback(self, signal_type: CareSignal, callback: Callable):
        """Add callback for specific care signal type"""
        self.care_signal_callbacks[signal_type].append(callback)
    
    def add_crisis_callback(self, callback: Callable):
        """Add callback for crisis situations"""
        self.crisis_callbacks.append(callback)
    
    def add_community_care_callback(self, callback: Callable):
        """Add callback for community care opportunities"""
        self.community_care_callbacks.append(callback)
    
    # Status and reporting methods
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status"""
        return {
            'monitoring_active': self.monitoring_active,
            'monitoring_scopes': [scope.value for scope in self.monitoring_scopes],
            'monitored_users_count': len(self.monitored_users),
            'monitored_keywords_count': len(self.monitored_keywords),
            'total_detections_today': len([
                d for d in self.detection_history
                if d.detected_at > datetime.utcnow() - timedelta(days=1)
            ]),
            'crisis_detections_today': len([
                d for d in self.detection_history
                if d.detected_at > datetime.utcnow() - timedelta(days=1)
                and d.signal_type in [CareSignal.IMMEDIATE_CRISIS, CareSignal.MENTAL_HEALTH_CRISIS]
            ]),
            'privacy_protection_active': self.privacy_protection_active
        }
    
    async def get_care_signal_report(self, time_range_hours: int = 24) -> Dict[str, Any]:
        """Get comprehensive care signal detection report"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=time_range_hours)
            
            recent_detections = [
                detection for detection in self.detection_history
                if detection.detected_at > cutoff_time
            ]
            
            if not recent_detections:
                return {'no_data': True, 'time_range_hours': time_range_hours}
            
            # Signal type distribution
            signal_distribution = defaultdict(int)
            confidence_distribution = defaultdict(int)
            urgency_distribution = defaultdict(int)
            
            for detection in recent_detections:
                signal_distribution[detection.signal_type.value] += 1
                confidence_distribution[detection.confidence_level.value] += 1
                urgency_distribution[detection.urgency_level] += 1
            
            # Average metrics
            avg_confidence = statistics.mean([d.confidence_score for d in recent_detections])
            avg_emotional_intensity = statistics.mean([d.emotional_intensity for d in recent_detections])
            
            return {
                'time_range_hours': time_range_hours,
                'total_detections': len(recent_detections),
                'signal_type_distribution': dict(signal_distribution),
                'confidence_level_distribution': dict(confidence_distribution),
                'urgency_level_distribution': dict(urgency_distribution),
                'average_confidence_score': avg_confidence,
                'average_emotional_intensity': avg_emotional_intensity,
                'interventions_recommended': len([d for d in recent_detections if d.intervention_recommended]),
                'trauma_informed_responses': len([d for d in recent_detections if d.trauma_informed_response]),
                'unique_users_detected': len(set(d.user_handle for d in recent_detections))
            }
            
        except Exception as e:
            logger.error(f"Failed to generate care signal report: {str(e)}")
            return {'error': str(e)}