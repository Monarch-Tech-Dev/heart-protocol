"""
Care Pattern Matcher

Advanced pattern recognition for identifying different types of care needs
using linguistic patterns, emotional markers, and contextual cues.
"""

import re
import asyncio
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class EmotionalState(Enum):
    """Detected emotional states"""
    DISTRESSED = "distressed"
    LONELY = "lonely"
    OVERWHELMED = "overwhelmed"
    HOPEFUL = "hopeful"
    GRATEFUL = "grateful"
    ANGRY = "angry"
    FEARFUL = "fearful"
    CONFUSED = "confused"


@dataclass
class PatternMatch:
    """Result of pattern matching"""
    pattern_type: str
    confidence: float
    matched_text: str
    emotional_state: Optional[EmotionalState] = None
    urgency_level: int = 0  # 0-5, 5 being most urgent


class CarePatternMatcher:
    """
    Advanced pattern matching for care detection.
    Uses linguistic analysis to identify different types of support needs.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.patterns = self._initialize_patterns()
    
    def _initialize_patterns(self) -> Dict:
        """Initialize comprehensive pattern database"""
        return {
            'crisis_immediate': {
                'patterns': [
                    r'(?:going to|gonna|about to|plan to|planning to)\s+(?:kill myself|end my life|die)',
                    r'(?:tonight|today|this (?:morning|afternoon|evening))\s+(?:will be|is)\s+(?:my last|the end)',
                    r'(?:have|got|found)\s+(?:a|the)\s+(?:gun|pills|rope|bridge|method)',
                    r'(?:final|last|goodbye)\s+(?:message|post|words)',
                    r'(?:can\'t|cannot)\s+(?:take|handle|do)\s+(?:it|this)\s+(?:anymore|any longer)'
                ],
                'urgency': 5,
                'emotional_state': EmotionalState.DISTRESSED
            },
            
            'crisis_planning': {
                'patterns': [
                    r'(?:thinking about|considering|planning)\s+(?:suicide|killing myself|ending it)',
                    r'(?:researching|looking up|googling)\s+(?:ways to|how to)\s+(?:die|kill|suicide)',
                    r'(?:writing|wrote)\s+(?:a|my)\s+(?:suicide note|final letter|will)',
                    r'(?:giving away|sold|donating)\s+(?:my|all my)\s+(?:stuff|things|belongings)'
                ],
                'urgency': 4,
                'emotional_state': EmotionalState.DISTRESSED
            },
            
            'crisis_ideation': {
                'patterns': [
                    r'(?:wish I|want to|hoping to)\s+(?:was|were)\s+(?:dead|gone|never born)',
                    r'(?:world|everyone)\s+(?:would be|is)\s+better (?:off|without me)',
                    r'(?:tired of|sick of|done with)\s+(?:living|life|being alive)',
                    r'(?:no reason|nothing)\s+to\s+(?:live|keep going|continue)',
                    r'(?:just|only)\s+(?:want|need)\s+(?:the pain|it all)\s+to\s+(?:stop|end)'
                ],
                'urgency': 3,
                'emotional_state': EmotionalState.DISTRESSED
            },
            
            'support_direct': {
                'patterns': [
                    r'(?:please|can someone|need someone to)\s+help\s+(?:me|please)',
                    r'(?:looking for|need|seeking|could use)\s+(?:help|support|advice|guidance)',
                    r'(?:don\'t know|not sure|unclear)\s+(?:what to do|how to handle|where to turn)',
                    r'(?:anyone|someone)\s+(?:been through|experienced|dealt with)\s+(?:this|something similar)'
                ],
                'urgency': 2,
                'emotional_state': EmotionalState.CONFUSED
            },
            
            'support_indirect': {
                'patterns': [
                    r'(?:having|going through)\s+(?:a|such a|really)\s+(?:tough|hard|difficult|rough)\s+(?:time|day|week|month)',
                    r'(?:struggling|dealing)\s+with\s+(?:depression|anxiety|grief|loss|trauma)',
                    r'(?:feel|feeling|felt)\s+(?:so|really|completely|totally)\s+(?:alone|lonely|isolated|lost)',
                    r'(?:everything|life|things)\s+(?:feels?|seems?)\s+(?:overwhelming|too much|impossible)',
                    r'(?:can\'t|cannot|unable to)\s+(?:cope|handle|manage|deal with)\s+(?:this|everything|life)'
                ],
                'urgency': 2,
                'emotional_state': EmotionalState.OVERWHELMED
            },
            
            'isolation_acute': {
                'patterns': [
                    r'(?:completely|totally|so|really)\s+(?:alone|lonely|isolated)',
                    r'(?:no one|nobody|not a soul)\s+(?:understands|cares|gets it|knows)',
                    r'(?:have|got)\s+(?:no one|nobody)\s+to\s+(?:talk to|turn to|call)',
                    r'(?:all my|my)\s+(?:friends|family)\s+(?:left|abandoned|don\'t care)',
                    r'(?:feel|feeling)\s+(?:invisible|forgotten|unwanted|unloved)'
                ],
                'urgency': 2,
                'emotional_state': EmotionalState.LONELY
            },
            
            'emotional_distress': {
                'patterns': [
                    r'(?:crying|sobbing|tears)\s+(?:uncontrollably|nonstop|all day|for hours)',
                    r'(?:panic|anxiety)\s+(?:attack|attacks)\s+(?:all day|constantly|won\'t stop)',
                    r'(?:can\'t|cannot)\s+(?:stop|quit|help)\s+(?:crying|shaking|panicking)',
                    r'(?:heart|chest)\s+(?:racing|pounding|tight|hurts)\s+(?:with|from)\s+(?:anxiety|panic|fear)',
                    r'(?:feel|feeling)\s+(?:numb|empty|hollow|broken|shattered)'
                ],
                'urgency': 2,
                'emotional_state': EmotionalState.DISTRESSED
            },
            
            'hope_seeking': {
                'patterns': [
                    r'(?:does|will)\s+(?:it|this|things|life)\s+(?:get|become)\s+(?:better|easier)',
                    r'(?:when|how)\s+(?:does|will)\s+(?:the pain|this feeling|it)\s+(?:stop|end|go away)',
                    r'(?:is there|any)\s+(?:hope|light|reason)\s+(?:for|to)\s+(?:me|keep going|continue)',
                    r'(?:what|how)\s+(?:helps|works|makes)\s+(?:you|people)\s+(?:feel better|cope|heal)',
                    r'(?:looking for|need|seeking)\s+(?:hope|inspiration|reasons to live)'
                ],
                'urgency': 1,
                'emotional_state': EmotionalState.HOPEFUL
            },
            
            'recovery_sharing': {
                'patterns': [
                    r'(?:getting|starting to feel|feeling)\s+(?:better|stronger|hopeful)',
                    r'(?:therapy|counseling|treatment)\s+(?:is helping|helped|working)',
                    r'(?:grateful|thankful)\s+for\s+(?:support|help|friends|community)',
                    r'(?:small|baby)\s+(?:steps|progress|victories|wins)',
                    r'(?:hope|believe)\s+(?:things|it)\s+(?:will|can)\s+(?:get better|improve|work out)'
                ],
                'urgency': 0,
                'emotional_state': EmotionalState.GRATEFUL
            }
        }
    
    async def analyze_patterns(self, text: str) -> List[PatternMatch]:
        """
        Analyze text for care-related patterns.
        Returns list of matches sorted by urgency.
        """
        matches = []
        text_lower = text.lower()
        
        for pattern_type, pattern_data in self.patterns.items():
            for pattern in pattern_data['patterns']:
                match = re.search(pattern, text_lower, re.IGNORECASE)
                if match:
                    matches.append(PatternMatch(
                        pattern_type=pattern_type,
                        confidence=self._calculate_confidence(match, text),
                        matched_text=match.group(),
                        emotional_state=pattern_data.get('emotional_state'),
                        urgency_level=pattern_data.get('urgency', 0)
                    ))
        
        # Sort by urgency level (highest first)
        matches.sort(key=lambda x: x.urgency_level, reverse=True)
        return matches
    
    def _calculate_confidence(self, match, text: str) -> float:
        """Calculate confidence score based on match quality and context"""
        base_confidence = 0.7
        
        # Longer matches are generally more confident
        match_length = len(match.group())
        length_boost = min(0.2, match_length / 100)
        
        # Check for amplifying words
        amplifiers = ['really', 'so', 'very', 'extremely', 'completely', 'totally']
        amplifier_boost = 0.1 * sum(1 for amp in amplifiers if amp in text.lower())
        
        # Check for negation that might reduce confidence
        negation_words = ['not', 'never', 'don\'t', 'can\'t', 'won\'t']
        negation_penalty = 0.3 if any(neg in text.lower() for neg in negation_words) else 0
        
        confidence = base_confidence + length_boost + amplifier_boost - negation_penalty
        return max(0.0, min(1.0, confidence))
    
    async def get_emotional_context(self, text: str) -> Dict[EmotionalState, float]:
        """
        Get emotional context scores for different states.
        Returns dictionary of emotional states and their confidence scores.
        """
        matches = await self.analyze_patterns(text)
        emotional_scores = {}
        
        for match in matches:
            if match.emotional_state:
                current_score = emotional_scores.get(match.emotional_state, 0.0)
                emotional_scores[match.emotional_state] = max(current_score, match.confidence)
        
        return emotional_scores
    
    async def assess_urgency(self, text: str) -> Tuple[int, List[str]]:
        """
        Assess overall urgency level and return contributing factors.
        Returns (urgency_level, reasons)
        """
        matches = await self.analyze_patterns(text)
        
        if not matches:
            return 0, []
        
        max_urgency = max(match.urgency_level for match in matches)
        high_urgency_matches = [match for match in matches if match.urgency_level >= 3]
        
        reasons = [f"{match.pattern_type}: {match.matched_text}" for match in high_urgency_matches]
        
        return max_urgency, reasons
    
    async def suggest_care_approach(self, text: str) -> Dict[str, any]:
        """
        Suggest appropriate care approach based on pattern analysis.
        """
        matches = await self.analyze_patterns(text)
        emotional_context = await self.get_emotional_context(text)
        urgency, reasons = await self.assess_urgency(text)
        
        if urgency >= 4:
            approach = "crisis_intervention"
            priority = "immediate"
        elif urgency >= 3:
            approach = "active_outreach"
            priority = "high"
        elif urgency >= 2:
            approach = "gentle_support"
            priority = "medium"
        elif urgency >= 1:
            approach = "resource_offer"
            priority = "low"
        else:
            approach = "positive_reinforcement"
            priority = "low"
        
        return {
            "approach": approach,
            "priority": priority,
            "urgency_level": urgency,
            "emotional_context": emotional_context,
            "contributing_factors": reasons,
            "recommended_response_tone": self._suggest_response_tone(emotional_context),
            "human_review_recommended": urgency >= 3
        }
    
    def _suggest_response_tone(self, emotional_context: Dict[EmotionalState, float]) -> str:
        """Suggest appropriate tone for response based on emotional context"""
        if not emotional_context:
            return "gentle_supportive"
        
        dominant_emotion = max(emotional_context.items(), key=lambda x: x[1])[0]
        
        tone_map = {
            EmotionalState.DISTRESSED: "calm_reassuring",
            EmotionalState.LONELY: "warm_connecting",
            EmotionalState.OVERWHELMED: "grounding_supportive",
            EmotionalState.HOPEFUL: "encouraging_realistic",
            EmotionalState.GRATEFUL: "celebratory_supportive",
            EmotionalState.ANGRY: "validating_peaceful",
            EmotionalState.FEARFUL: "safe_reassuring",
            EmotionalState.CONFUSED: "clarifying_gentle"
        }
        
        return tone_map.get(dominant_emotion, "gentle_supportive")