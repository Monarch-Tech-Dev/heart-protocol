"""
Care Detection Engine

Ethical detection of support needs through natural language understanding,
context awareness, and consent-based intervention with privacy-preserving analysis.
"""

from .engine import CareDetectionEngine
from .patterns import CarePatternMatcher
from .privacy import PrivacyPreservingAnalyzer

__all__ = ['CareDetectionEngine', 'CarePatternMatcher', 'PrivacyPreservingAnalyzer']