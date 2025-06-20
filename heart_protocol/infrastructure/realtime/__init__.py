"""
Real-time Feed Updates System

Real-time communication system for delivering healing-focused content updates
with gentle, non-overwhelming user experiences and trauma-informed design.

Core Philosophy: "Real-time care, delivered gently."
"""

from .realtime_engine import RealtimeEngine, UpdateType, DeliveryMode
from .gentle_streaming import GentleStreamer, StreamingPolicy, UserComfort
from .healing_websockets import HealingWebSocketManager, ConnectionType, SafetyProtocol
from .feed_synchronizer import FeedSynchronizer, SyncStrategy, ConflictResolution

__all__ = [
    'RealtimeEngine',
    'UpdateType',
    'DeliveryMode',
    'GentleStreamer',
    'StreamingPolicy',
    'UserComfort',
    'HealingWebSocketManager',
    'ConnectionType',
    'SafetyProtocol',
    'FeedSynchronizer',
    'SyncStrategy',
    'ConflictResolution'
]