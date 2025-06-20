"""
Real-time Firehose Monitoring

Real-time monitoring infrastructure for detecting care opportunities, crisis situations,
and healing moments across social media streams and communication platforms.

Core Philosophy: "Vigilant care, gentle response, healing focus."
"""

from .firehose_monitor import FirehoseMonitor, StreamType, MonitoringScope
from .care_signal_processor import CareSignalProcessor, SignalType, ProcessingPriority
from .crisis_detector import CrisisDetector, CrisisLevel, InterventionUrgency
from .healing_moment_tracker import HealingMomentTracker, MomentType, ResonanceLevel
from .stream_aggregator import StreamAggregator, AggregationStrategy, FilterCriteria
from .realtime_analytics import RealtimeAnalytics, MetricType, AnalysisWindow

__all__ = [
    'FirehoseMonitor',
    'StreamType',
    'MonitoringScope',
    'CareSignalProcessor',
    'SignalType',
    'ProcessingPriority',
    'CrisisDetector',
    'CrisisLevel',
    'InterventionUrgency',
    'HealingMomentTracker',
    'MomentType',
    'ResonanceLevel',
    'StreamAggregator',
    'AggregationStrategy',
    'FilterCriteria',
    'RealtimeAnalytics',
    'MetricType',
    'AnalysisWindow'
]