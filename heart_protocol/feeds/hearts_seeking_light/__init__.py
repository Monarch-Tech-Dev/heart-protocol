"""
Hearts Seeking Light Feed

The second of the Four Sacred Feeds - connecting those who need support
with those who can offer it. Built on principles of consent, privacy,
and authentic human connection.

Core Philosophy: "In our shared vulnerability, we find strength and connection."
"""

from .support_detection import SupportSeeker, SupportOffer, SupportDetectionEngine
from .matching import ConnectionMatcher, MatchingCriteria
from .intervention_timing import InterventionTimingEngine
from .consent_system import ConsentManager, ConsentLevel
from .feed import HeartsSeekingLightFeed

__all__ = [
    'HeartsSeekingLightFeed',
    'SupportSeeker',
    'SupportOffer', 
    'SupportDetectionEngine',
    'ConnectionMatcher',
    'MatchingCriteria',
    'InterventionTimingEngine',
    'ConsentManager',
    'ConsentLevel'
]