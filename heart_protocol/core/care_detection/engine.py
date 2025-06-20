"""
Care Detection Engine

The heart of Heart Protocol's ethical care detection system.
Analyzes posts to identify when someone might benefit from support,
while respecting privacy and requiring consent.
"""

import re
import asyncio
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

from ..base import BaseCareDetector, Post, CareAssessment, CareLevel
from .patterns import CarePatternMatcher
from .privacy import PrivacyPreservingAnalyzer

logger = logging.getLogger(__name__)


class CareDetectionEngine(BaseCareDetector):
    """
    Ethical detection of support needs through:
    - Natural language understanding
    - Context awareness  
    - Consent-based intervention
    - Privacy-preserving analysis
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.pattern_matcher = CarePatternMatcher(config.get('patterns', {}))
        self.privacy_analyzer = PrivacyPreservingAnalyzer(config.get('privacy', {}))
        
        # Crisis keywords that require immediate attention
        self.crisis_keywords = {
            'explicit': [
                'kill myself', 'end my life', 'want to die', 'suicide',
                'not worth living', 'better off dead', 'end it all'
            ],
            'concerning': [
                'hopeless', 'no point', 'give up', 'can\'t go on',
                'too much pain', 'nothing left', 'tired of living'
            ]
        }
        
        # Support-seeking language patterns
        self.support_patterns = {
            'direct_help': [
                r'need help', r'please help', r'can someone help',
                r'looking for support', r'need advice', r'feeling lost'
            ],
            'indirect_help': [
                r'don\'t know what to do', r'feeling alone', r'struggling with',
                r'having a hard time', r'going through', r'dealing with'
            ],
            'isolation': [
                r'no one understands', r'all alone', r'nobody cares',
                r'feel invisible', r'isolated', r'lonely'
            ]
        }
    
    async def assess_care_needs(self, post: Post) -> CareAssessment:
        """
        Assess if and what kind of care a post indicates is needed.
        
        This is the core function that determines:
        - What level of care intervention might be helpful
        - How confident we are in this assessment
        - What specific indicators led to this conclusion
        - Whether human review is needed
        """
        try:
            # First check if we have consent to analyze this post
            if not await self._has_analysis_consent(post):
                return CareAssessment(
                    post_uri=post.uri,
                    care_level=CareLevel.NONE,
                    confidence=0.0,
                    indicators=["no_consent_for_analysis"]
                )
            
            # Check for crisis indicators first
            crisis_score, crisis_indicators = await self._detect_crisis_language(post)
            if crisis_score > 0.8:
                return CareAssessment(
                    post_uri=post.uri,
                    care_level=CareLevel.CRISIS_INTERVENTION,
                    confidence=crisis_score,
                    indicators=crisis_indicators,
                    human_review_needed=True,
                    suggested_response="crisis_escalation"
                )
            
            # Check for support-seeking language
            support_score, support_indicators = await self._detect_support_seeking(post)
            
            # Check for isolation expressions
            isolation_score, isolation_indicators = await self._detect_isolation(post)
            
            # Check for struggle sharing
            struggle_score, struggle_indicators = await self._detect_struggle_sharing(post)
            
            # Combine scores and determine care level
            all_indicators = support_indicators + isolation_indicators + struggle_indicators
            
            if crisis_score > 0.6:
                care_level = CareLevel.ACTIVE_OUTREACH
                confidence = max(crisis_score, support_score)
                human_review = True
            elif support_score > 0.6 or isolation_score > 0.7:
                care_level = CareLevel.GENTLE_SUPPORT
                confidence = max(support_score, isolation_score)
                human_review = support_score > 0.8
            elif struggle_score > 0.5:
                care_level = CareLevel.GENTLE_SUPPORT
                confidence = struggle_score
                human_review = False
            else:
                care_level = CareLevel.NONE
                confidence = 0.0
                human_review = False
            
            return CareAssessment(
                post_uri=post.uri,
                care_level=care_level,
                confidence=confidence,
                indicators=all_indicators,
                human_review_needed=human_review,
                suggested_response=self._suggest_response_type(care_level, all_indicators)
            )
            
        except Exception as e:
            logger.error(f"Error assessing care needs for {post.uri}: {e}")
            # Fail safely - don't intervene if we can't assess properly
            return CareAssessment(
                post_uri=post.uri,
                care_level=CareLevel.NONE,
                confidence=0.0,
                indicators=["assessment_error"]
            )
    
    async def detect_crisis_indicators(self, post: Post) -> bool:
        """
        Detect if post contains crisis indicators requiring immediate response.
        This is a simplified version focused on safety.
        """
        try:
            crisis_score, _ = await self._detect_crisis_language(post)
            return crisis_score > 0.8
        except Exception as e:
            logger.error(f"Error detecting crisis indicators: {e}")
            return False
    
    async def _has_analysis_consent(self, post: Post) -> bool:
        """
        Check if user has consented to analysis of their posts.
        For now, we only analyze public posts. In future, users can
        explicitly opt-in to care analysis.
        """
        # For now, only analyze public posts (implied consent)
        # TODO: Implement explicit consent system
        return True
    
    async def _detect_crisis_language(self, post: Post) -> Tuple[float, List[str]]:
        """Detect explicit crisis language requiring immediate intervention"""
        text = post.text.lower()
        indicators = []
        score = 0.0
        
        # Check for explicit crisis keywords
        for keyword in self.crisis_keywords['explicit']:
            if keyword in text:
                indicators.append(f"explicit_crisis: {keyword}")
                score = max(score, 0.9)
        
        # Check for concerning language
        for keyword in self.crisis_keywords['concerning']:
            if keyword in text:
                indicators.append(f"concerning_language: {keyword}")
                score = max(score, 0.7)
        
        # Check for patterns indicating imminent danger
        danger_patterns = [
            r'tonight.*end', r'today.*last day', r'final.*goodbye',
            r'plan.*suicide', r'method.*kill', r'ready.*die'
        ]
        
        for pattern in danger_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                indicators.append(f"danger_pattern: {pattern}")
                score = max(score, 0.95)
        
        return score, indicators
    
    async def _detect_support_seeking(self, post: Post) -> Tuple[float, List[str]]:
        """Detect language indicating someone is seeking support"""
        text = post.text.lower()
        indicators = []
        score = 0.0
        
        # Direct help requests
        for pattern in self.support_patterns['direct_help']:
            if re.search(pattern, text, re.IGNORECASE):
                indicators.append(f"direct_help: {pattern}")
                score = max(score, 0.8)
        
        # Indirect help requests
        for pattern in self.support_patterns['indirect_help']:
            if re.search(pattern, text, re.IGNORECASE):
                indicators.append(f"indirect_help: {pattern}")
                score = max(score, 0.6)
        
        # Question marks often indicate seeking help/advice
        if '?' in text and any(word in text for word in ['help', 'advice', 'support', 'what', 'how']):
            indicators.append("help_question")
            score = max(score, 0.5)
        
        return score, indicators
    
    async def _detect_isolation(self, post: Post) -> Tuple[float, List[str]]:
        """Detect expressions of loneliness or isolation"""
        text = post.text.lower()
        indicators = []
        score = 0.0
        
        for pattern in self.support_patterns['isolation']:
            if re.search(pattern, text, re.IGNORECASE):
                indicators.append(f"isolation: {pattern}")
                score = max(score, 0.7)
        
        # Time-based isolation indicators
        isolation_times = ['3am', 'late night', 'middle of the night', 'can\'t sleep']
        for time_phrase in isolation_times:
            if time_phrase in text:
                indicators.append(f"isolation_time: {time_phrase}")
                score = max(score, 0.5)
        
        return score, indicators
    
    async def _detect_struggle_sharing(self, post: Post) -> Tuple[float, List[str]]:
        """Detect when someone is sharing their struggles (potential support opportunity)"""
        text = post.text.lower()
        indicators = []
        score = 0.0
        
        # Common struggle topics
        struggle_topics = [
            'depression', 'anxiety', 'panic', 'stress', 'overwhelmed',
            'exhausted', 'burned out', 'difficult day', 'tough week',
            'mental health', 'therapy', 'medication', 'grief', 'loss'
        ]
        
        for topic in struggle_topics:
            if topic in text:
                indicators.append(f"struggle_topic: {topic}")
                score = max(score, 0.6)
        
        # Emotional expressions
        emotional_words = [
            'crying', 'tears', 'heartbroken', 'devastated', 'scared',
            'worried', 'afraid', 'confused', 'lost', 'hurt'
        ]
        
        for emotion in emotional_words:
            if emotion in text:
                indicators.append(f"emotional_expression: {emotion}")
                score = max(score, 0.5)
        
        return score, indicators
    
    def _suggest_response_type(self, care_level: CareLevel, indicators: List[str]) -> Optional[str]:
        """Suggest what type of response would be most appropriate"""
        if care_level == CareLevel.CRISIS_INTERVENTION:
            return "crisis_resources"
        elif care_level == CareLevel.ACTIVE_OUTREACH:
            return "gentle_outreach"
        elif care_level == CareLevel.GENTLE_SUPPORT:
            if any("help" in indicator for indicator in indicators):
                return "resource_offer"
            else:
                return "gentle_validation"
        else:
            return None
    
    async def get_care_statistics(self) -> Dict:
        """Get statistics about care detection performance"""
        # TODO: Implement proper metrics tracking
        return {
            "assessments_made": 0,
            "crisis_interventions": 0,
            "support_connections": 0,
            "false_positive_rate": 0.0,
            "user_satisfaction": 0.0
        }