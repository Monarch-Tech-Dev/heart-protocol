"""
Wisdom Curation for Community Wisdom Feed

Identifies, extracts, and curates healing wisdom and insights from community
posts and interactions. Built on principles of honoring lived experience
and amplifying knowledge that supports collective healing.
"""

import asyncio
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
import logging
import re

from ...core.base import Post, CareAssessment, CareLevel

logger = logging.getLogger(__name__)


class WisdomCategory(Enum):
    """Categories of wisdom that can be curated"""
    COPING_STRATEGIES = "coping_strategies"           # Practical coping techniques
    HEALING_INSIGHTS = "healing_insights"             # Deep insights about healing process
    RELATIONSHIP_WISDOM = "relationship_wisdom"       # Wisdom about healthy relationships
    SELF_CARE_PRACTICES = "self_care_practices"       # Self-care and nurturing approaches
    TRAUMA_RECOVERY = "trauma_recovery"               # Trauma-informed recovery wisdom
    EMOTIONAL_REGULATION = "emotional_regulation"     # Emotional management insights
    MEANING_MAKING = "meaning_making"                 # Finding purpose and meaning
    RESILIENCE_BUILDING = "resilience_building"       # Building strength and resilience
    COMMUNITY_SUPPORT = "community_support"           # Supporting others and building community
    SPIRITUAL_GROWTH = "spiritual_growth"             # Spiritual and existential insights
    CRISIS_NAVIGATION = "crisis_navigation"           # Getting through crisis situations
    BOUNDARY_SETTING = "boundary_setting"             # Healthy boundaries and self-advocacy
    FORGIVENESS_HEALING = "forgiveness_healing"       # Forgiveness and letting go
    CREATIVE_EXPRESSION = "creative_expression"       # Healing through creativity
    MINDFULNESS_PRESENCE = "mindfulness_presence"     # Mindfulness and present-moment awareness


class WisdomType(Enum):
    """Types of wisdom content"""
    PRACTICAL_TIP = "practical_tip"                   # Actionable advice or technique
    DEEP_INSIGHT = "deep_insight"                     # Profound realization or understanding
    LIVED_EXPERIENCE = "lived_experience"             # Wisdom from personal experience
    TRANSFORMED_PERSPECTIVE = "transformed_perspective" # Changed way of seeing things
    HELPFUL_REFRAME = "helpful_reframe"               # Useful way to reframe situations
    ENCOURAGING_TRUTH = "encouraging_truth"           # Hopeful and encouraging insight
    WARNING_WISDOM = "warning_wisdom"                 # What to avoid or be careful of
    INTEGRATION_GUIDANCE = "integration_guidance"     # How to integrate healing experiences


@dataclass
class WisdomInsight:
    """Represents a piece of curated wisdom"""
    insight_id: str
    author_id: str  # Anonymous identifier
    category: WisdomCategory
    wisdom_type: WisdomType
    content: str
    source_post_id: str
    extracted_at: datetime
    wisdom_score: float  # 0.0 to 1.0 - quality and helpfulness
    validation_score: float  # 0.0 to 1.0 - community validation
    applicability_tags: List[str]  # Who this might help
    context_requirements: List[str]  # When this applies
    care_considerations: List[str]  # Special care needed when sharing
    supporting_evidence: List[str]  # What makes this wisdom valuable
    cultural_context: Optional[str]  # Cultural considerations
    trauma_informed: bool  # Whether this follows trauma-informed principles
    anonymization_level: str  # How much to anonymize when sharing
    expiry_date: Optional[datetime]  # When this wisdom might become less relevant
    engagement_metrics: Dict[str, int]  # How community has engaged with this
    wisdom_lineage: List[str]  # If this builds on other wisdom


class WisdomCurator:
    """
    Curates healing wisdom and insights from community content.
    
    Core Principles:
    - Honor lived experience as valid knowledge
    - Amplify wisdom that serves collective healing
    - Maintain anonymity and privacy
    - Validate wisdom through multiple perspectives
    - Ensure trauma-informed approaches
    - Respect cultural contexts and diversity
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Wisdom detection patterns
        self.wisdom_patterns = self._initialize_wisdom_patterns()
        self.insight_indicators = self._initialize_insight_indicators()
        self.experience_markers = self._initialize_experience_markers()
        self.transformation_signals = self._initialize_transformation_signals()
        
        # Wisdom storage and tracking
        self.curated_wisdom = {}      # insight_id -> WisdomInsight
        self.category_collections = {} # category -> List[insight_ids]
        self.user_contributions = {}   # user_id -> List[insight_ids]
        
        # Curation metrics
        self.curation_metrics = {
            'insights_identified': 0,
            'insights_curated': 0,
            'insights_validated': 0,
            'insights_shared': 0,
            'contributing_users': 0,
            'wisdom_categories_active': 0
        }
        
        logger.info("Wisdom Curator initialized")
    
    def _initialize_wisdom_patterns(self) -> Dict[WisdomCategory, Dict[str, Any]]:
        """Initialize patterns for detecting different types of wisdom"""
        
        return {
            WisdomCategory.COPING_STRATEGIES: {
                'indicators': [
                    'what helps me', 'my go-to technique', 'this works for me',
                    'found helpful', 'coping strategy', 'technique that works',
                    'when i need to calm down', 'grounding technique'
                ],
                'action_words': [
                    'breathe', 'ground', 'center', 'practice', 'use',
                    'try', 'do', 'remind myself', 'tell myself'
                ],
                'effectiveness_markers': [
                    'really helps', 'works well', 'makes a difference',
                    'calms me down', 'brings me back', 'grounds me'
                ],
                'wisdom_weight': 1.0
            },
            
            WisdomCategory.HEALING_INSIGHTS: {
                'indicators': [
                    'realized', 'learned', 'discovered', 'understood',
                    'insight', 'revelation', 'breakthrough', 'aha moment',
                    'now i know', 'truth is', 'what i\'ve learned'
                ],
                'depth_markers': [
                    'deep truth', 'profound realization', 'life-changing insight',
                    'fundamentally changed', 'shifted everything'
                ],
                'integration_signs': [
                    'now i understand', 'makes sense now', 'see clearly',
                    'perspective shifted', 'view changed'
                ],
                'wisdom_weight': 1.2
            },
            
            WisdomCategory.RELATIONSHIP_WISDOM: {
                'indicators': [
                    'healthy relationships', 'good boundaries', 'communication',
                    'learned about love', 'relationship insight', 'connection wisdom'
                ],
                'relationship_markers': [
                    'partner', 'friend', 'family', 'relationship', 'connection',
                    'communication', 'boundary', 'support', 'love'
                ],
                'health_indicators': [
                    'healthy', 'supportive', 'mutual', 'respectful',
                    'balanced', 'caring', 'understanding'
                ],
                'wisdom_weight': 1.0
            },
            
            WisdomCategory.SELF_CARE_PRACTICES: {
                'indicators': [
                    'self-care', 'taking care of myself', 'nurturing myself',
                    'self-compassion', 'being kind to myself', 'gentle with myself'
                ],
                'practice_markers': [
                    'routine', 'ritual', 'practice', 'habit', 'daily',
                    'regular', 'consistent', 'discipline'
                ],
                'nurturing_words': [
                    'gentle', 'kind', 'compassionate', 'loving', 'caring',
                    'patient', 'understanding', 'forgiving'
                ],
                'wisdom_weight': 0.9
            },
            
            WisdomCategory.TRAUMA_RECOVERY: {
                'indicators': [
                    'trauma recovery', 'healing from trauma', 'post-traumatic growth',
                    'trauma-informed', 'processing trauma', 'integrating trauma'
                ],
                'recovery_markers': [
                    'healing', 'recovery', 'processing', 'integrating',
                    'working through', 'dealing with', 'overcoming'
                ],
                'growth_indicators': [
                    'stronger', 'wiser', 'resilient', 'post-traumatic growth',
                    'transformed', 'evolved', 'learned from'
                ],
                'wisdom_weight': 1.3  # High weight due to importance
            },
            
            WisdomCategory.EMOTIONAL_REGULATION: {
                'indicators': [
                    'managing emotions', 'emotional regulation', 'feeling my feelings',
                    'emotional intelligence', 'processing emotions'
                ],
                'regulation_techniques': [
                    'breathing', 'mindfulness', 'grounding', 'pausing',
                    'observing', 'accepting', 'allowing', 'sitting with'
                ],
                'emotional_awareness': [
                    'notice', 'observe', 'aware', 'conscious', 'mindful',
                    'present', 'attentive', 'tuned in'
                ],
                'wisdom_weight': 1.0
            },
            
            WisdomCategory.MEANING_MAKING: {
                'indicators': [
                    'purpose', 'meaning', 'why', 'calling', 'mission',
                    'reason for', 'makes sense', 'understanding why'
                ],
                'purpose_markers': [
                    'purpose', 'calling', 'mission', 'meaning', 'reason',
                    'why', 'significance', 'importance', 'value'
                ],
                'transformation_signs': [
                    'transformed', 'changed', 'shifted', 'evolved',
                    'growth', 'learning', 'wisdom', 'insight'
                ],
                'wisdom_weight': 1.2
            },
            
            WisdomCategory.RESILIENCE_BUILDING: {
                'indicators': [
                    'resilience', 'bounce back', 'inner strength', 'perseverance',
                    'endurance', 'toughness', 'durability', 'adaptability'
                ],
                'strength_markers': [
                    'strong', 'tough', 'resilient', 'durable', 'enduring',
                    'persistent', 'determined', 'courageous'
                ],
                'adaptation_signs': [
                    'adapt', 'adjust', 'flexible', 'bend not break',
                    'change', 'evolve', 'grow', 'learn'
                ],
                'wisdom_weight': 1.1
            },
            
            WisdomCategory.COMMUNITY_SUPPORT: {
                'indicators': [
                    'supporting others', 'helping community', 'being there',
                    'mutual support', 'community care', 'collective healing'
                ],
                'support_actions': [
                    'listen', 'hold space', 'be present', 'show up',
                    'offer', 'give', 'share', 'connect'
                ],
                'community_values': [
                    'together', 'connected', 'supported', 'belonging',
                    'community', 'family', 'tribe', 'circle'
                ],
                'wisdom_weight': 1.0
            },
            
            WisdomCategory.CRISIS_NAVIGATION: {
                'indicators': [
                    'getting through crisis', 'survival mode', 'crisis management',
                    'emergency coping', 'acute distress', 'crisis wisdom'
                ],
                'crisis_markers': [
                    'crisis', 'emergency', 'acute', 'severe', 'intense',
                    'overwhelming', 'breaking point', 'rock bottom'
                ],
                'navigation_strategies': [
                    'survive', 'endure', 'get through', 'hold on',
                    'one moment at a time', 'just breathe', 'stay safe'
                ],
                'wisdom_weight': 1.4  # Very high weight due to urgency
            }
        }
    
    def _initialize_insight_indicators(self) -> Dict[str, float]:
        """Initialize indicators that suggest valuable insights"""
        
        return {
            # Learning and realization words
            'learned': 1.2, 'realized': 1.3, 'discovered': 1.2, 'understood': 1.1,
            'insight': 1.4, 'revelation': 1.3, 'epiphany': 1.3, 'breakthrough': 1.4,
            
            # Wisdom sharing words
            'wisdom': 1.3, 'advice': 0.8, 'tip': 0.9, 'suggestion': 0.8,
            'recommend': 0.9, 'helpful': 1.0, 'works for me': 1.1,
            
            # Transformation indicators
            'transformed': 1.3, 'changed': 1.0, 'shifted': 1.1, 'evolved': 1.2,
            'growth': 1.1, 'progress': 1.0, 'healing': 1.2,
            
            # Experience depth markers
            'deep': 1.1, 'profound': 1.3, 'powerful': 1.1, 'significant': 1.0,
            'meaningful': 1.1, 'important': 0.9, 'life-changing': 1.4,
            
            # Truth and authenticity
            'truth': 1.2, 'honest': 1.0, 'real': 0.9, 'authentic': 1.1,
            'genuine': 1.0, 'sincere': 1.0
        }
    
    def _initialize_experience_markers(self) -> Dict[str, List[str]]:
        """Initialize markers that indicate lived experience"""
        
        return {
            'personal_experience': [
                'in my experience', 'what i\'ve learned', 'from my journey',
                'personally', 'for me', 'i\'ve found', 'i discovered',
                'my story', 'my path', 'my process'
            ],
            
            'time_depth': [
                'years of', 'months of', 'after years', 'over time',
                'in time', 'eventually', 'gradually', 'slowly learned'
            ],
            
            'struggle_context': [
                'through struggle', 'after trauma', 'in recovery',
                'healing from', 'working through', 'dealing with',
                'overcoming', 'surviving'
            ],
            
            'professional_context': [
                'in therapy', 'with therapist', 'counselor taught me',
                'professional help', 'treatment', 'therapy work'
            ],
            
            'wisdom_sharing': [
                'want to share', 'hope this helps', 'might help others',
                'passing along', 'sharing wisdom', 'for anyone who'
            ]
        }
    
    def _initialize_transformation_signals(self) -> Dict[str, List[str]]:
        """Initialize signals that indicate meaningful transformation"""
        
        return {
            'before_after': [
                'used to', 'before i', 'now i', 'no longer', 'instead of',
                'different now', 'changed from', 'evolved from'
            ],
            
            'perspective_shift': [
                'see differently', 'new perspective', 'way of looking',
                'reframe', 'shift in thinking', 'changed my view'
            ],
            
            'capacity_change': [
                'can now', 'able to', 'capable of', 'stronger',
                'more resilient', 'better at', 'improved'
            ],
            
            'integration_markers': [
                'integrated', 'internalized', 'embody', 'live by',
                'part of me', 'who i am now', 'natural'
            ]
        }
    
    async def curate_wisdom_from_post(self, post: Post, 
                                    care_assessment: CareAssessment,
                                    user_context: Dict[str, Any]) -> List[WisdomInsight]:
        """
        Curate wisdom insights from a user's post.
        
        Args:
            post: The post to analyze for wisdom
            care_assessment: Current care assessment
            user_context: User's context and history
        """
        try:
            curated_insights = []
            
            # Don't curate from crisis posts unless they contain recovery wisdom
            if care_assessment.care_level == CareLevel.CRISIS:
                if not await self._contains_crisis_recovery_wisdom(post.content):
                    return curated_insights
            
            # Analyze post content for wisdom indicators
            wisdom_analysis = await self._analyze_wisdom_content(post.content)
            
            if not wisdom_analysis['has_wisdom_indicators']:
                return curated_insights
            
            # Extract specific wisdom insights
            potential_insights = await self._extract_wisdom_insights(
                post, wisdom_analysis, user_context
            )
            
            # Validate and score each insight
            for insight_data in potential_insights:
                validated_insight = await self._create_validated_insight(
                    insight_data, post, user_context
                )
                
                if validated_insight and validated_insight.wisdom_score >= 0.6:
                    curated_insights.append(validated_insight)
                    await self._store_curated_insight(validated_insight)
            
            # Update metrics
            self.curation_metrics['insights_identified'] += len(potential_insights)
            self.curation_metrics['insights_curated'] += len(curated_insights)
            
            if curated_insights:
                logger.info(f"Curated {len(curated_insights)} wisdom insights from user {post.author_id[:8]}...")
            
            return curated_insights
            
        except Exception as e:
            logger.error(f"Error curating wisdom from post: {e}")
            return []
    
    async def _contains_crisis_recovery_wisdom(self, content: str) -> bool:
        """Check if crisis post contains recovery wisdom"""
        
        recovery_wisdom_phrases = [
            'what helped me get through', 'crisis survival tip', 'emergency coping',
            'when in crisis', 'acute distress management', 'survival strategy',
            'crisis navigation', 'getting through the worst'
        ]
        
        content_lower = content.lower()
        return any(phrase in content_lower for phrase in recovery_wisdom_phrases)
    
    async def _analyze_wisdom_content(self, content: str) -> Dict[str, Any]:
        """Analyze content for wisdom indicators"""
        
        content_lower = content.lower()
        analysis = {
            'has_wisdom_indicators': False,
            'wisdom_score': 0.0,
            'categories_detected': [],
            'insight_types': [],
            'experience_depth': 0.0,
            'transformation_signals': [],
            'wisdom_phrases': []
        }
        
        # Check for insight indicators
        wisdom_score = 0.0
        for indicator, weight in self.insight_indicators.items():
            if indicator in content_lower:
                wisdom_score += weight
                analysis['wisdom_phrases'].append(indicator)
        
        analysis['wisdom_score'] = min(1.0, wisdom_score / 5.0)  # Normalize
        
        # Check for category-specific patterns
        categories_detected = []
        for category, patterns in self.wisdom_patterns.items():
            category_score = 0.0
            
            # Check indicators
            for indicator in patterns.get('indicators', []):
                if indicator in content_lower:
                    category_score += 1.0
            
            # Check specific markers
            for marker_list in ['action_words', 'effectiveness_markers', 'depth_markers',
                               'relationship_markers', 'practice_markers', 'recovery_markers']:
                for marker in patterns.get(marker_list, []):
                    if marker in content_lower:
                        category_score += 0.5
            
            if category_score > 0:
                weighted_score = category_score * patterns.get('wisdom_weight', 1.0)
                categories_detected.append((category, weighted_score))
        
        # Sort categories by score
        categories_detected.sort(key=lambda x: x[1], reverse=True)
        analysis['categories_detected'] = [cat for cat, score in categories_detected]
        
        # Check for experience markers
        experience_depth = 0.0
        for marker_type, markers in self.experience_markers.items():
            for marker in markers:
                if marker in content_lower:
                    experience_depth += 0.2
        
        analysis['experience_depth'] = min(1.0, experience_depth)
        
        # Check for transformation signals
        transformation_signals = []
        for signal_type, signals in self.transformation_signals.items():
            for signal in signals:
                if signal in content_lower:
                    transformation_signals.append(signal_type)
        
        analysis['transformation_signals'] = list(set(transformation_signals))
        
        # Determine if content has wisdom indicators
        analysis['has_wisdom_indicators'] = (
            analysis['wisdom_score'] > 0.3 or
            len(analysis['categories_detected']) > 0 or
            analysis['experience_depth'] > 0.3
        )
        
        return analysis
    
    async def _extract_wisdom_insights(self, post: Post, 
                                     analysis: Dict[str, Any],
                                     user_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract specific wisdom insights from the post"""
        
        insights = []
        content = post.content
        
        # Split into sentences for insight extraction
        sentences = re.split(r'[.!?]+', content)
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if len(sentence) < 20:  # Skip very short sentences
                continue
            
            sentence_lower = sentence.lower()
            
            # Check if sentence contains wisdom indicators
            has_wisdom = False
            wisdom_strength = 0.0
            
            for indicator, weight in self.insight_indicators.items():
                if indicator in sentence_lower:
                    has_wisdom = True
                    wisdom_strength += weight
            
            if not has_wisdom:
                continue
            
            # Determine category and type
            insight_category = await self._determine_insight_category(
                sentence, analysis['categories_detected']
            )
            
            insight_type = await self._determine_insight_type(sentence)
            
            # Extract context (surrounding sentences)
            context_start = max(0, i - 1)
            context_end = min(len(sentences), i + 2)
            context = '. '.join(sentences[context_start:context_end]).strip()
            
            insights.append({
                'content': sentence,
                'context': context,
                'category': insight_category,
                'type': insight_type,
                'wisdom_strength': min(1.0, wisdom_strength),
                'sentence_index': i
            })
        
        # Also look for multi-sentence insights
        paragraph_insights = await self._extract_paragraph_insights(content, analysis)
        insights.extend(paragraph_insights)
        
        return insights
    
    async def _determine_insight_category(self, sentence: str, 
                                        detected_categories: List[WisdomCategory]) -> WisdomCategory:
        """Determine the category of a specific insight"""
        
        sentence_lower = sentence.lower()
        
        # Score each detected category for this sentence
        category_scores = {}
        
        for category in detected_categories:
            patterns = self.wisdom_patterns.get(category, {})
            score = 0.0
            
            # Check category-specific patterns
            for pattern_list in patterns.values():
                if isinstance(pattern_list, list):
                    for pattern in pattern_list:
                        if pattern in sentence_lower:
                            score += 1.0
            
            if score > 0:
                category_scores[category] = score
        
        # Return highest scoring category, or default
        if category_scores:
            return max(category_scores.items(), key=lambda x: x[1])[0]
        else:
            return WisdomCategory.HEALING_INSIGHTS  # Default category
    
    async def _determine_insight_type(self, sentence: str) -> WisdomType:
        """Determine the type of insight"""
        
        sentence_lower = sentence.lower()
        
        # Check for different insight types
        if any(word in sentence_lower for word in ['tip', 'technique', 'strategy', 'practice']):
            return WisdomType.PRACTICAL_TIP
        
        elif any(word in sentence_lower for word in ['realized', 'understood', 'insight', 'revelation']):
            return WisdomType.DEEP_INSIGHT
        
        elif any(word in sentence_lower for word in ['my experience', 'personally', 'for me', 'i found']):
            return WisdomType.LIVED_EXPERIENCE
        
        elif any(word in sentence_lower for word in ['now i see', 'perspective', 'reframe', 'different way']):
            return WisdomType.TRANSFORMED_PERSPECTIVE
        
        elif any(word in sentence_lower for word in ['instead', 'rather than', 'better to', 'helpful to']):
            return WisdomType.HELPFUL_REFRAME
        
        elif any(word in sentence_lower for word in ['hope', 'possible', 'can', 'will', 'strength']):
            return WisdomType.ENCOURAGING_TRUTH
        
        elif any(word in sentence_lower for word in ['avoid', 'careful', 'watch out', 'beware']):
            return WisdomType.WARNING_WISDOM
        
        elif any(word in sentence_lower for word in ['integrate', 'apply', 'use', 'practice']):
            return WisdomType.INTEGRATION_GUIDANCE
        
        else:
            return WisdomType.DEEP_INSIGHT  # Default type
    
    async def _extract_paragraph_insights(self, content: str, 
                                        analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract insights that span multiple sentences"""
        
        insights = []
        
        # Split content into paragraphs
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        for paragraph in paragraphs:
            if len(paragraph) < 100:  # Skip short paragraphs
                continue
            
            paragraph_lower = paragraph.lower()
            
            # Check if paragraph contains significant wisdom
            wisdom_density = 0.0
            for indicator in self.insight_indicators.keys():
                if indicator in paragraph_lower:
                    wisdom_density += 1.0
            
            wisdom_density = wisdom_density / len(paragraph.split())  # Normalize by length
            
            if wisdom_density > 0.02:  # Threshold for wisdom density
                # This paragraph has high wisdom density
                category = analysis['categories_detected'][0] if analysis['categories_detected'] else WisdomCategory.HEALING_INSIGHTS
                
                insights.append({
                    'content': paragraph,
                    'context': paragraph,  # Full paragraph is context
                    'category': category,
                    'type': WisdomType.DEEP_INSIGHT,
                    'wisdom_strength': min(1.0, wisdom_density * 20),  # Scale up
                    'sentence_index': -1  # Indicates paragraph-level insight
                })
        
        return insights
    
    async def _create_validated_insight(self, insight_data: Dict[str, Any], 
                                      post: Post, 
                                      user_context: Dict[str, Any]) -> Optional[WisdomInsight]:
        """Create and validate a wisdom insight"""
        
        try:
            # Generate unique insight ID
            insight_id = f"wisdom_{post.author_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{insight_data['sentence_index']}"
            
            # Calculate comprehensive wisdom score
            wisdom_score = await self._calculate_wisdom_score(insight_data, user_context)
            
            # Determine applicability and context requirements
            applicability_tags = await self._determine_applicability(insight_data)
            context_requirements = await self._determine_context_requirements(insight_data)
            care_considerations = await self._determine_care_considerations(insight_data)
            
            # Assess trauma-informed nature
            trauma_informed = await self._assess_trauma_informed(insight_data['content'])
            
            # Determine anonymization level
            anonymization_level = await self._determine_anonymization_level(
                insight_data, user_context
            )
            
            # Extract supporting evidence
            supporting_evidence = await self._extract_supporting_evidence(
                insight_data, post.content
            )
            
            # Create wisdom insight
            insight = WisdomInsight(
                insight_id=insight_id,
                author_id=f"anonymous_{hash(post.author_id) % 10000}",  # Anonymous ID
                category=insight_data['category'],
                wisdom_type=insight_data['type'],
                content=insight_data['content'],
                source_post_id=post.id,
                extracted_at=datetime.utcnow(),
                wisdom_score=wisdom_score,
                validation_score=0.0,  # Will be populated by community validation
                applicability_tags=applicability_tags,
                context_requirements=context_requirements,
                care_considerations=care_considerations,
                supporting_evidence=supporting_evidence,
                cultural_context=user_context.get('cultural_context'),
                trauma_informed=trauma_informed,
                anonymization_level=anonymization_level,
                expiry_date=None,  # Most wisdom doesn't expire
                engagement_metrics={'views': 0, 'helpful_votes': 0, 'shares': 0},
                wisdom_lineage=[]  # Will be populated if this builds on other wisdom
            )
            
            return insight
            
        except Exception as e:
            logger.error(f"Error creating validated insight: {e}")
            return None
    
    async def _calculate_wisdom_score(self, insight_data: Dict[str, Any], 
                                    user_context: Dict[str, Any]) -> float:
        """Calculate comprehensive wisdom score"""
        
        base_score = insight_data['wisdom_strength']
        
        # Factor in user's credibility and experience
        user_experience = user_context.get('healing_experience_years', 0)
        experience_bonus = min(0.2, user_experience * 0.05)
        
        # Factor in content quality
        content_length = len(insight_data['content'])
        length_factor = min(1.0, content_length / 200)  # Optimal length around 200 chars
        
        # Factor in specificity vs generality
        specificity = len([word for word in insight_data['content'].lower().split() 
                          if word in ['specific', 'exactly', 'precisely', 'particular']])
        specificity_bonus = min(0.1, specificity * 0.05)
        
        # Factor in actionability
        action_words = ['do', 'try', 'practice', 'use', 'apply', 'start', 'begin']
        actionability = len([word for word in insight_data['content'].lower().split() 
                           if word in action_words])
        actionability_bonus = min(0.15, actionability * 0.05)
        
        final_score = base_score + experience_bonus + specificity_bonus + actionability_bonus
        final_score *= length_factor  # Penalize very short or very long content
        
        return min(1.0, final_score)
    
    async def _determine_applicability(self, insight_data: Dict[str, Any]) -> List[str]:
        """Determine who this wisdom might help"""
        
        content_lower = insight_data['content'].lower()
        applicability = []
        
        # General applicability based on category
        category_applicability = {
            WisdomCategory.COPING_STRATEGIES: ['anxiety', 'stress', 'overwhelm'],
            WisdomCategory.TRAUMA_RECOVERY: ['trauma_survivors', 'ptsd', 'complex_trauma'],
            WisdomCategory.RELATIONSHIP_WISDOM: ['relationship_issues', 'communication_struggles'],
            WisdomCategory.CRISIS_NAVIGATION: ['crisis', 'emergency', 'severe_distress'],
            WisdomCategory.EMOTIONAL_REGULATION: ['emotional_instability', 'mood_swings'],
        }
        
        applicability.extend(category_applicability.get(insight_data['category'], []))
        
        # Specific applicability based on content
        if any(word in content_lower for word in ['anxiety', 'anxious', 'worried']):
            applicability.append('anxiety')
        
        if any(word in content_lower for word in ['depression', 'depressed', 'sad']):
            applicability.append('depression')
        
        if any(word in content_lower for word in ['trauma', 'ptsd', 'flashback']):
            applicability.append('trauma')
        
        if any(word in content_lower for word in ['crisis', 'emergency', 'suicidal']):
            applicability.append('crisis')
        
        return list(set(applicability))  # Remove duplicates
    
    async def _determine_context_requirements(self, insight_data: Dict[str, Any]) -> List[str]:
        """Determine when this wisdom applies"""
        
        content_lower = insight_data['content'].lower()
        requirements = []
        
        # Time-based requirements
        if any(word in content_lower for word in ['morning', 'night', 'evening']):
            requirements.append('time_specific')
        
        # Emotional state requirements
        if any(word in content_lower for word in ['when anxious', 'when sad', 'when triggered']):
            requirements.append('emotional_state_specific')
        
        # Skill level requirements
        if any(word in content_lower for word in ['advanced', 'beginner', 'experienced']):
            requirements.append('skill_level_specific')
        
        # Safety requirements
        if any(word in content_lower for word in ['safe space', 'private', 'alone']):
            requirements.append('safety_environment_needed')
        
        # Professional support requirements
        if any(word in content_lower for word in ['with therapist', 'professional help']):
            requirements.append('professional_support_recommended')
        
        return requirements
    
    async def _determine_care_considerations(self, insight_data: Dict[str, Any]) -> List[str]:
        """Determine care considerations when sharing this wisdom"""
        
        considerations = []
        content_lower = insight_data['content'].lower()
        
        # Trauma considerations
        if any(word in content_lower for word in ['trauma', 'abuse', 'violence']):
            considerations.append('trauma_content_warning')
        
        # Crisis considerations
        if any(word in content_lower for word in ['crisis', 'suicidal', 'self-harm']):
            considerations.append('crisis_content_warning')
        
        # Professional guidance needed
        if insight_data['category'] in [WisdomCategory.TRAUMA_RECOVERY, WisdomCategory.CRISIS_NAVIGATION]:
            considerations.append('professional_guidance_recommended')
        
        # Individual variation
        considerations.append('individual_results_may_vary')
        
        # Not universal applicability
        considerations.append('context_dependent')
        
        return considerations
    
    async def _assess_trauma_informed(self, content: str) -> bool:
        """Assess if the wisdom follows trauma-informed principles"""
        
        content_lower = content.lower()
        
        # Positive indicators
        trauma_informed_indicators = [
            'your pace', 'when you\'re ready', 'if it feels safe',
            'gentle', 'compassionate', 'non-judgmental', 'choice',
            'control', 'empowerment', 'safety first'
        ]
        
        # Negative indicators
        trauma_uninformed_indicators = [
            'just get over it', 'move on', 'forget about it',
            'should', 'must', 'have to', 'need to',
            'everyone should', 'always', 'never'
        ]
        
        positive_count = sum(1 for indicator in trauma_informed_indicators 
                           if indicator in content_lower)
        negative_count = sum(1 for indicator in trauma_uninformed_indicators 
                           if indicator in content_lower)
        
        return positive_count > negative_count
    
    async def _determine_anonymization_level(self, insight_data: Dict[str, Any], 
                                           user_context: Dict[str, Any]) -> str:
        """Determine level of anonymization needed"""
        
        content_lower = insight_data['content'].lower()
        
        # High anonymization for sensitive content
        if any(word in content_lower for word in ['trauma', 'abuse', 'assault', 'crisis']):
            return 'high'
        
        # Medium anonymization for personal details
        if any(word in content_lower for word in ['my', 'i', 'personal', 'family']):
            return 'medium'
        
        # Low anonymization for general wisdom
        return 'low'
    
    async def _extract_supporting_evidence(self, insight_data: Dict[str, Any], 
                                         full_content: str) -> List[str]:
        """Extract evidence that supports this wisdom"""
        
        evidence = []
        
        # Look for effectiveness statements in context
        context_lower = insight_data.get('context', '').lower()
        
        effectiveness_phrases = [
            'really helps', 'works well', 'made a difference',
            'transformed my', 'changed everything', 'game changer',
            'life saver', 'breakthrough', 'profound impact'
        ]
        
        for phrase in effectiveness_phrases:
            if phrase in context_lower:
                evidence.append(f"User reports: '{phrase}'")
        
        # Look for time-tested evidence
        time_phrases = ['years of', 'months of', 'long time', 'consistently']
        for phrase in time_phrases:
            if phrase in context_lower:
                evidence.append("Tested over time")
                break
        
        # Look for professional validation
        if any(word in context_lower for word in ['therapist', 'counselor', 'professional']):
            evidence.append("Professionally supported approach")
        
        return evidence
    
    async def _store_curated_insight(self, insight: WisdomInsight) -> None:
        """Store curated insight in collections"""
        
        # Store in main collection
        self.curated_wisdom[insight.insight_id] = insight
        
        # Store in category collection
        category = insight.category
        if category not in self.category_collections:
            self.category_collections[category] = []
        self.category_collections[category].append(insight.insight_id)
        
        # Store in user contributions
        author_id = insight.author_id
        if author_id not in self.user_contributions:
            self.user_contributions[author_id] = []
        self.user_contributions[author_id].append(insight.insight_id)
        
        # Update metrics
        if author_id not in self.user_contributions or len(self.user_contributions[author_id]) == 1:
            self.curation_metrics['contributing_users'] += 1
        
        active_categories = len([cat for cat in self.category_collections if self.category_collections[cat]])
        self.curation_metrics['wisdom_categories_active'] = active_categories
        
        logger.debug(f"Stored wisdom insight {insight.insight_id} in category {category.value}")
    
    def get_wisdom_by_category(self, category: WisdomCategory, 
                              limit: int = 10) -> List[WisdomInsight]:
        """Get wisdom insights by category"""
        
        insight_ids = self.category_collections.get(category, [])
        insights = [self.curated_wisdom[id] for id in insight_ids if id in self.curated_wisdom]
        
        # Sort by wisdom score and validation score
        insights.sort(key=lambda i: (i.wisdom_score + i.validation_score), reverse=True)
        
        return insights[:limit]
    
    def get_curation_metrics(self) -> Dict[str, Any]:
        """Get wisdom curation metrics"""
        
        return {
            'insights_identified': self.curation_metrics['insights_identified'],
            'insights_curated': self.curation_metrics['insights_curated'],
            'insights_validated': self.curation_metrics['insights_validated'],
            'insights_shared': self.curation_metrics['insights_shared'],
            'contributing_users': self.curation_metrics['contributing_users'],
            'wisdom_categories_active': self.curation_metrics['wisdom_categories_active'],
            'total_curated_wisdom': len(self.curated_wisdom),
            'category_distribution': {cat.value: len(insights) for cat, insights in self.category_collections.items()},
            'system_health': {
                'wisdom_patterns_loaded': len(self.wisdom_patterns) > 0,
                'insight_indicators_loaded': len(self.insight_indicators) > 0,
                'experience_markers_loaded': len(self.experience_markers) > 0,
                'transformation_signals_loaded': len(self.transformation_signals) > 0
            }
        }