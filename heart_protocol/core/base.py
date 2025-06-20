"""
Heart Protocol Core Base Classes

Foundation classes and interfaces for the caring algorithm system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
import asyncio
import logging

logger = logging.getLogger(__name__)


class CareLevel(Enum):
    """Levels of care intervention needed"""
    NONE = "none"
    GENTLE_SUPPORT = "gentle_support" 
    ACTIVE_OUTREACH = "active_outreach"
    CRISIS_INTERVENTION = "crisis_intervention"


class FeedType(Enum):
    """Types of caring feeds available"""
    DAILY_GENTLE_REMINDERS = "daily_gentle_reminders"
    HEARTS_SEEKING_LIGHT = "hearts_seeking_light"
    GUARDIAN_ENERGY_RISING = "guardian_energy_rising"
    COMMUNITY_WISDOM = "community_wisdom"


@dataclass
class Post:
    """Represents a social media post"""
    uri: str
    author: str
    text: str
    created_at: datetime
    reply_to: Optional[str] = None
    embed: Optional[Dict] = None
    labels: Optional[List[str]] = None
    
    def __post_init__(self):
        if isinstance(self.created_at, str):
            self.created_at = datetime.fromisoformat(self.created_at.replace('Z', '+00:00'))


@dataclass
class CareAssessment:
    """Assessment of care needs for a post or user"""
    post_uri: str
    care_level: CareLevel
    confidence: float  # 0.0 to 1.0
    indicators: List[str]
    suggested_response: Optional[str] = None
    human_review_needed: bool = False
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


@dataclass
class FeedItem:
    """Item in a caring feed"""
    post: Post
    reason: str  # Why this was included
    care_context: Optional[str] = None
    priority_score: float = 0.0


@dataclass
class FeedSkeleton:
    """Feed response structure for AT Protocol"""
    cursor: Optional[str]
    feed: List[Dict[str, str]]  # [{"post": uri}, ...]


class BaseCareDetector(ABC):
    """Base class for care detection algorithms"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    async def assess_care_needs(self, post: Post) -> CareAssessment:
        """Assess if and what kind of care a post indicates is needed"""
        pass
    
    @abstractmethod
    async def detect_crisis_indicators(self, post: Post) -> bool:
        """Detect if post contains crisis indicators requiring immediate response"""
        pass


class BaseFeedGenerator(ABC):
    """Base class for caring feed generators"""
    
    def __init__(self, feed_type: FeedType, config: Dict[str, Any]):
        self.feed_type = feed_type
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    async def generate_feed(self, user_id: str, cursor: Optional[str] = None, 
                           limit: int = 50) -> FeedSkeleton:
        """Generate a caring feed for the user"""
        pass
    
    @abstractmethod
    async def is_eligible_post(self, post: Post) -> bool:
        """Check if post is eligible for this feed type"""
        pass
    
    async def score_post(self, post: Post, context: Dict[str, Any]) -> float:
        """Score how well a post fits this feed (0.0 to 1.0)"""
        return 0.5  # Default neutral score


class BaseCareResponder(ABC):
    """Base class for generating caring responses"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    async def generate_response(self, assessment: CareAssessment, 
                               context: Dict[str, Any]) -> Optional[str]:
        """Generate appropriate caring response"""
        pass
    
    @abstractmethod 
    async def should_respond(self, assessment: CareAssessment) -> bool:
        """Determine if a response should be generated"""
        pass


class BaseConnectionMatcher(ABC):
    """Base class for matching people who need care with helpers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    async def find_potential_helpers(self, assessment: CareAssessment) -> List[str]:
        """Find users who might be able to help with this specific need"""
        pass
    
    @abstractmethod
    async def check_helper_availability(self, user_id: str) -> bool:
        """Check if a potential helper is currently available and willing"""
        pass


class CareMetrics:
    """Tracks meaningful care outcomes, not engagement metrics"""
    
    def __init__(self):
        self.connections_facilitated = 0
        self.crisis_interventions = 0
        self.positive_outcomes = 0
        self.user_wellbeing_reports = []
        self.feed_satisfaction_scores = []
    
    def record_connection(self, helper_id: str, helped_id: str):
        """Record a successful connection between users"""
        self.connections_facilitated += 1
        logger.info(f"Connection facilitated: {helper_id} -> {helped_id}")
    
    def record_crisis_intervention(self, post_uri: str, outcome: str):
        """Record crisis intervention and outcome"""
        self.crisis_interventions += 1
        logger.info(f"Crisis intervention: {post_uri} -> {outcome}")
    
    def record_wellbeing_feedback(self, user_id: str, score: float, feedback: str):
        """Record user wellbeing feedback"""
        self.user_wellbeing_reports.append({
            'user_id': user_id,
            'score': score,
            'feedback': feedback,
            'timestamp': datetime.utcnow()
        })


class HeartProtocolError(Exception):
    """Base exception for Heart Protocol errors"""
    pass


class CareDetectionError(HeartProtocolError):
    """Errors in care detection process"""
    pass


class FeedGenerationError(HeartProtocolError):
    """Errors in feed generation process"""
    pass


class CrisisEscalationError(HeartProtocolError):
    """Errors in crisis escalation process"""
    pass