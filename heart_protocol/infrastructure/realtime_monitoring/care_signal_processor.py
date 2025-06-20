"""
Care Signal Processor

Advanced processor for analyzing and responding to care signals detected
in real-time streams with trauma-informed, culturally sensitive approaches.
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import logging
import statistics
import json
from collections import defaultdict, deque
import re

from .firehose_monitor import StreamEvent, MonitoringPriority

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Types of care signals that can be processed"""
    SUPPORT_SEEKING = "support_seeking"                 # Actively seeking help
    EMOTIONAL_DISTRESS = "emotional_distress"           # Emotional pain expression
    SUPPORT_OFFERING = "support_offering"               # Offering help to others
    COMMUNITY_BUILDING = "community_building"           # Building supportive community
    GRATITUDE_APPRECIATION = "gratitude_appreciation"   # Expressing gratitude
    VULNERABILITY_SHARING = "vulnerability_sharing"     # Courageous vulnerability
    PROGRESS_CELEBRATION = "progress_celebration"       # Celebrating growth
    WISDOM_SHARING = "wisdom_sharing"                   # Sharing learned wisdom
    CONNECTION_SEEKING = "connection_seeking"           # Seeking human connection
    VALIDATION_SEEKING = "validation_seeking"           # Seeking validation/understanding


class ProcessingPriority(Enum):
    """Priority levels for processing care signals"""
    IMMEDIATE = "immediate"                             # Requires immediate response
    URGENT = "urgent"                                   # High priority processing
    STANDARD = "standard"                               # Normal processing
    BACKGROUND = "background"                           # Low priority processing
    RESEARCH = "research"                               # For analytics only


class ResponseType(Enum):
    """Types of responses to care signals"""
    GENTLE_ENGAGEMENT = "gentle_engagement"             # Gentle, supportive response
    RESOURCE_SHARING = "resource_sharing"               # Share relevant resources
    COMMUNITY_CONNECTION = "community_connection"       # Connect to community
    PROFESSIONAL_REFERRAL = "professional_referral"    # Refer to professional help
    PEER_MATCHING = "peer_matching"                     # Match with peer support
    CONTENT_AMPLIFICATION = "content_amplification"     # Amplify positive content
    VALIDATION_RESPONSE = "validation_response"         # Provide validation
    WISDOM_APPRECIATION = "wisdom_appreciation"         # Appreciate shared wisdom
    PASSIVE_MONITORING = "passive_monitoring"           # Monitor without direct response
    NO_RESPONSE = "no_response"                         # No response needed


class ProcessingStage(Enum):
    """Stages of care signal processing"""
    DETECTION = "detection"                             # Signal detected
    ANALYSIS = "analysis"                               # Analyzing signal context
    VALIDATION = "validation"                           # Validating signal authenticity
    PRIORITIZATION = "prioritization"                   # Determining response priority
    RESPONSE_PLANNING = "response_planning"             # Planning appropriate response
    RESPONSE_EXECUTION = "response_execution"           # Executing response
    FOLLOW_UP = "follow_up"                            # Following up on response
    COMPLETED = "completed"                             # Processing completed


@dataclass
class CareSignalAnalysis:
    """Detailed analysis of a care signal"""
    signal_id: str
    original_event: StreamEvent
    signal_type: SignalType
    confidence_score: float
    authenticity_score: float
    urgency_level: float
    vulnerability_indicators: List[str]
    cultural_context: List[str]
    trauma_considerations: List[str]
    accessibility_needs: List[str]
    emotional_intensity: float
    social_context: Dict[str, Any]
    support_type_needed: List[str]
    potential_responses: List[ResponseType]
    recommended_response: ResponseType
    response_timing: str
    safety_considerations: List[str]
    privacy_requirements: List[str]
    analyzed_at: datetime


@dataclass
class ResponsePlan:
    """Plan for responding to a care signal"""
    plan_id: str
    signal_analysis: CareSignalAnalysis
    response_type: ResponseType
    response_priority: ProcessingPriority
    response_approach: str
    personalization_factors: Dict[str, Any]
    cultural_adaptations: List[str]
    accessibility_accommodations: List[str]
    safety_measures: List[str]
    trauma_informed_elements: List[str]
    resource_recommendations: List[Dict[str, Any]]
    community_connections: List[Dict[str, Any]]
    timing_strategy: Dict[str, Any]
    success_metrics: Dict[str, Any]
    gentle_constraints: Dict[str, Any]
    created_at: datetime
    execution_window: Tuple[datetime, datetime]


@dataclass
class ProcessingResult:
    """Result of care signal processing"""
    result_id: str
    signal_id: str
    processing_stage: ProcessingStage
    response_executed: bool
    response_type: Optional[ResponseType]
    execution_details: Dict[str, Any]
    user_response: Optional[str]
    effectiveness_score: float
    healing_impact: float
    user_satisfaction: float
    lessons_learned: List[str]
    follow_up_needed: bool
    follow_up_timing: Optional[datetime]
    completed_at: datetime
    processing_duration_ms: float


class CareSignalProcessor:
    """
    Advanced processor for care signals with trauma-informed, culturally sensitive,
    and healing-focused approaches.
    
    Core Principles:
    - Trauma-informed processing prevents re-traumatization
    - Cultural sensitivity guides all response approaches
    - User autonomy and consent are respected throughout
    - Gentle, non-intrusive responses prioritized
    - Healing outcomes measured over engagement metrics
    - Privacy and safety are paramount
    - Authentic human connection facilitated
    - Community care fostered over individual interventions
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Processing settings
        self.processing_enabled = self.config.get('processing_enabled', True)
        self.gentle_mode = self.config.get('gentle_mode', True)
        self.cultural_sensitivity_level = self.config.get('cultural_sensitivity_level', 'high')
        self.trauma_informed_processing = self.config.get('trauma_informed_processing', True)
        
        # Signal processing
        self.processing_queue: asyncio.Queue = asyncio.Queue()
        self.active_analyses: Dict[str, CareSignalAnalysis] = {}
        self.processing_results: List[ProcessingResult] = []
        self.response_plans: Dict[str, ResponsePlan] = {}
        
        # Pattern analysis
        self.signal_patterns = self._initialize_signal_patterns()
        self.response_strategies = self._initialize_response_strategies()
        self.cultural_adaptations = self._initialize_cultural_adaptations()
        
        # Performance tracking
        self.signals_processed = 0
        self.responses_executed = 0
        self.average_processing_time_ms = 0.0
        self.total_healing_impact = 0.0
        self.user_satisfaction_average = 0.0
        
        # Safety and cultural systems
        self.trauma_safety_validator = None    # Would integrate with trauma safety system
        self.cultural_context_analyzer = None # Would integrate with cultural analysis
        self.accessibility_adapter = None     # Would integrate with accessibility system
        self.privacy_protector = None         # Would integrate with privacy system
        
        # Callbacks and integrations
        self.signal_analyzed_callbacks: List[Callable] = []
        self.response_planned_callbacks: List[Callable] = []
        self.response_executed_callbacks: List[Callable] = []
        self.processing_completed_callbacks: List[Callable] = []
        
        # Processing state
        self.processing_active = False
        self.processing_tasks: Set[asyncio.Task] = set()
    
    def _initialize_signal_patterns(self) -> Dict[SignalType, Dict[str, Any]]:
        """Initialize patterns for analyzing different types of care signals"""
        return {
            SignalType.SUPPORT_SEEKING: {
                'keywords': [
                    'need help', 'looking for advice', 'struggling with', 'don\'t know what to do',
                    'any suggestions', 'has anyone', 'please help', 'advice needed'
                ],
                'emotional_indicators': [
                    'overwhelmed', 'lost', 'confused', 'scared', 'anxious', 'desperate',
                    'hopeless', 'stuck', 'isolated', 'alone'
                ],
                'urgency_markers': [
                    'urgent', 'emergency', 'crisis', 'immediate', 'asap', 'desperate',
                    'can\'t wait', 'right now'
                ],
                'vulnerability_indicators': [
                    'afraid to ask', 'embarrassed', 'ashamed', 'vulnerable', 'scared to share',
                    'first time posting', 'hard to admit'
                ]
            },
            
            SignalType.EMOTIONAL_DISTRESS: {
                'keywords': [
                    'depressed', 'anxious', 'panic', 'breakdown', 'crying', 'exhausted',
                    'drained', 'overwhelmed', 'burnt out', 'falling apart'
                ],
                'emotional_indicators': [
                    'can\'t cope', 'breaking down', 'losing it', 'at my limit', 'can\'t handle',
                    'too much', 'drowning', 'suffocating'
                ],
                'intensity_markers': [
                    'severe', 'intense', 'unbearable', 'extreme', 'constant', 'chronic',
                    'persistent', 'overwhelming'
                ],
                'physical_symptoms': [
                    'can\'t sleep', 'no appetite', 'headaches', 'chest tight', 'shaking',
                    'nauseous', 'dizzy', 'fatigue'
                ]
            },
            
            SignalType.SUPPORT_OFFERING: {
                'keywords': [
                    'here for you', 'happy to help', 'if you need', 'reach out', 'DM me',
                    'been through this', 'understand', 'support you'
                ],
                'experience_indicators': [
                    'went through', 'experienced', 'dealt with', 'overcame', 'survived',
                    'recovered from', 'learned from'
                ],
                'availability_markers': [
                    'available', 'free to talk', 'message me', 'here to listen',
                    'open to chat', 'willing to help'
                ],
                'expertise_indicators': [
                    'professional', 'trained in', 'experience with', 'knowledge of',
                    'specialized in', 'certified'
                ]
            },
            
            SignalType.VULNERABILITY_SHARING: {
                'keywords': [
                    'admitting', 'confessing', 'sharing', 'opening up', 'being honest',
                    'vulnerable moment', 'hard to say'
                ],
                'courage_indicators': [
                    'first time sharing', 'never told anyone', 'scary to admit',
                    'taking courage', 'brave enough', 'ready to share'
                ],
                'authenticity_markers': [
                    'real talk', 'honestly', 'truth is', 'reality', 'raw',
                    'unfiltered', 'genuine'
                ],
                'trust_indicators': [
                    'trusting you', 'safe space', 'feel comfortable', 'no judgment',
                    'understanding community'
                ]
            },
            
            SignalType.WISDOM_SHARING: {
                'keywords': [
                    'learned', 'discovered', 'realized', 'insight', 'lesson',
                    'advice', 'tip', 'what helped', 'experience taught'
                ],
                'wisdom_indicators': [
                    'looking back', 'in hindsight', 'now I know', 'wish I knew',
                    'important lesson', 'key insight'
                ],
                'teaching_markers': [
                    'sharing because', 'hope this helps', 'might help others',
                    'for anyone going through', 'pay it forward'
                ],
                'transformation_indicators': [
                    'changed my perspective', 'shifted my thinking', 'opened my eyes',
                    'transformed', 'growth', 'evolution'
                ]
            }
        }
    
    def _initialize_response_strategies(self) -> Dict[ResponseType, Dict[str, Any]]:
        """Initialize response strategies for different signal types"""
        return {
            ResponseType.GENTLE_ENGAGEMENT: {
                'approach': 'soft_supportive_response',
                'tone': 'warm_empathetic',
                'timing': 'prompt_but_not_immediate',
                'personalization': 'high',
                'trauma_informed': True,
                'cultural_sensitivity': 'high'
            },
            
            ResponseType.RESOURCE_SHARING: {
                'approach': 'helpful_resource_provision',
                'tone': 'informative_supportive',
                'timing': 'timely_relevant',
                'personalization': 'medium',
                'trauma_informed': True,
                'cultural_sensitivity': 'high'
            },
            
            ResponseType.COMMUNITY_CONNECTION: {
                'approach': 'facilitated_introduction',
                'tone': 'welcoming_inclusive',
                'timing': 'when_user_ready',
                'personalization': 'high',
                'trauma_informed': True,
                'cultural_sensitivity': 'very_high'
            },
            
            ResponseType.PROFESSIONAL_REFERRAL: {
                'approach': 'gentle_professional_suggestion',
                'tone': 'caring_non_judgmental',
                'timing': 'appropriate_moment',
                'personalization': 'high',
                'trauma_informed': True,
                'cultural_sensitivity': 'very_high'
            },
            
            ResponseType.VALIDATION_RESPONSE: {
                'approach': 'affirming_validating',
                'tone': 'understanding_accepting',
                'timing': 'prompt_response',
                'personalization': 'high',
                'trauma_informed': True,
                'cultural_sensitivity': 'high'
            }
        }
    
    def _initialize_cultural_adaptations(self) -> Dict[str, Dict[str, Any]]:
        """Initialize cultural adaptation strategies"""
        return {
            'collectivist_cultures': {
                'community_emphasis': True,
                'family_consideration': True,
                'group_harmony_focus': True,
                'indirect_communication': True,
                'elder_respect': True
            },
            
            'individualist_cultures': {
                'personal_autonomy': True,
                'direct_communication': True,
                'self_reliance_respect': True,
                'individual_achievement': True,
                'personal_space': True
            },
            
            'high_context_cultures': {
                'nonverbal_awareness': True,
                'implicit_understanding': True,
                'relationship_focus': True,
                'context_sensitivity': True,
                'subtle_communication': True
            },
            
            'low_context_cultures': {
                'explicit_communication': True,
                'direct_messaging': True,
                'clear_instructions': True,
                'specific_details': True,
                'straightforward_approach': True
            }
        }
    
    async def start_processing(self) -> bool:
        """Start care signal processing"""
        try:
            if self.processing_active:
                logger.warning("Care signal processing already active")
                return True
            
            self.processing_active = True
            
            # Start processing tasks
            await self._start_processing_tasks()
            
            logger.info("Care signal processing started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start care signal processing: {str(e)}")
            self.processing_active = False
            return False
    
    async def stop_processing(self) -> bool:
        """Stop care signal processing"""
        try:
            self.processing_active = False
            
            # Cancel processing tasks
            for task in self.processing_tasks:
                if not task.done():
                    task.cancel()
            
            self.processing_tasks.clear()
            
            logger.info("Care signal processing stopped")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop care signal processing: {str(e)}")
            return False
    
    async def _start_processing_tasks(self):
        """Start processing tasks"""
        try:
            # Start main processing loop
            task = asyncio.create_task(self._processing_loop())
            self.processing_tasks.add(task)
            
            # Start follow-up processor
            task = asyncio.create_task(self._follow_up_processor())
            self.processing_tasks.add(task)
            
            # Start analytics processor
            task = asyncio.create_task(self._analytics_processor())
            self.processing_tasks.add(task)
            
        except Exception as e:
            logger.error(f"Failed to start processing tasks: {str(e)}")
    
    async def process_care_signal(self, event: StreamEvent) -> Optional[CareSignalAnalysis]:
        """Process a care signal from a stream event"""
        try:
            if not self.processing_enabled:
                return None
            
            # Add to processing queue
            await self.processing_queue.put(event)
            
            return None  # Processing happens asynchronously
            
        except Exception as e:
            logger.error(f"Failed to queue care signal for processing: {str(e)}")
            return None
    
    async def _processing_loop(self):
        """Main processing loop for care signals"""
        try:
            while self.processing_active:
                try:
                    # Wait for event with timeout
                    event = await asyncio.wait_for(self.processing_queue.get(), timeout=1.0)
                    
                    # Process the care signal
                    analysis = await self._analyze_care_signal(event)
                    
                    if analysis:
                        # Create response plan
                        response_plan = await self._create_response_plan(analysis)
                        
                        if response_plan:
                            # Execute response if appropriate
                            result = await self._execute_response_plan(response_plan)
                            
                            if result:
                                self.processing_results.append(result)
                    
                    # Mark queue task as done
                    self.processing_queue.task_done()
                    
                except asyncio.TimeoutError:
                    # No events in queue, continue waiting
                    continue
                except Exception as e:
                    logger.error(f"Error in processing loop: {str(e)}")
                    await asyncio.sleep(1)
                    
        except Exception as e:
            logger.error(f"Fatal error in processing loop: {str(e)}")
    
    async def _analyze_care_signal(self, event: StreamEvent) -> Optional[CareSignalAnalysis]:
        """Analyze a care signal in detail"""
        try:
            start_time = datetime.utcnow()
            
            # Determine signal type
            signal_type = await self._classify_signal_type(event)
            
            if not signal_type:
                return None
            
            # Calculate analysis scores
            confidence_score = await self._calculate_confidence_score(event, signal_type)
            authenticity_score = await self._calculate_authenticity_score(event, signal_type)
            urgency_level = await self._calculate_urgency_level(event, signal_type)
            emotional_intensity = await self._calculate_emotional_intensity(event)
            
            # Analyze context factors
            vulnerability_indicators = await self._identify_vulnerability_indicators(event)
            cultural_context = await self._analyze_cultural_context(event)
            trauma_considerations = await self._analyze_trauma_considerations(event)
            accessibility_needs = await self._analyze_accessibility_needs(event)
            social_context = await self._analyze_social_context(event)
            
            # Determine support needs and responses
            support_type_needed = await self._identify_support_types_needed(event, signal_type)
            potential_responses = await self._identify_potential_responses(signal_type, urgency_level)
            recommended_response = await self._recommend_response(
                signal_type, urgency_level, cultural_context, trauma_considerations
            )
            
            # Determine timing and safety
            response_timing = await self._determine_response_timing(urgency_level, trauma_considerations)
            safety_considerations = await self._analyze_safety_considerations(event, trauma_considerations)
            privacy_requirements = await self._analyze_privacy_requirements(event)
            
            analysis = CareSignalAnalysis(
                signal_id=f"care_signal_{datetime.utcnow().isoformat()}_{id(self)}",
                original_event=event,
                signal_type=signal_type,
                confidence_score=confidence_score,
                authenticity_score=authenticity_score,
                urgency_level=urgency_level,
                vulnerability_indicators=vulnerability_indicators,
                cultural_context=cultural_context,
                trauma_considerations=trauma_considerations,
                accessibility_needs=accessibility_needs,
                emotional_intensity=emotional_intensity,
                social_context=social_context,
                support_type_needed=support_type_needed,
                potential_responses=potential_responses,
                recommended_response=recommended_response,
                response_timing=response_timing,
                safety_considerations=safety_considerations,
                privacy_requirements=privacy_requirements,
                analyzed_at=datetime.utcnow()
            )
            
            # Store analysis
            self.active_analyses[analysis.signal_id] = analysis
            self.signals_processed += 1
            
            # Update processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.average_processing_time_ms = (
                (self.average_processing_time_ms * (self.signals_processed - 1) + processing_time) 
                / self.signals_processed
            )
            
            # Trigger callbacks
            for callback in self.signal_analyzed_callbacks:
                try:
                    await callback(analysis)
                except Exception as e:
                    logger.error(f"Signal analyzed callback failed: {str(e)}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing care signal: {str(e)}")
            return None
    
    async def _classify_signal_type(self, event: StreamEvent) -> Optional[SignalType]:
        """Classify the type of care signal"""
        try:
            content = event.processed_content.lower()
            signal_scores = {}
            
            # Score each signal type
            for signal_type, patterns in self.signal_patterns.items():
                score = 0.0
                
                # Check keywords
                keyword_matches = sum(
                    1 for keyword in patterns.get('keywords', []) 
                    if keyword in content
                )
                if patterns.get('keywords'):
                    score += (keyword_matches / len(patterns['keywords'])) * 0.4
                
                # Check emotional indicators
                emotion_matches = sum(
                    1 for emotion in patterns.get('emotional_indicators', [])
                    if emotion in content
                )
                if patterns.get('emotional_indicators'):
                    score += (emotion_matches / len(patterns['emotional_indicators'])) * 0.3
                
                # Check specific markers (urgency, vulnerability, etc.)
                for marker_type in ['urgency_markers', 'vulnerability_indicators', 
                                   'intensity_markers', 'authenticity_markers']:
                    markers = patterns.get(marker_type, [])
                    if markers:
                        marker_matches = sum(1 for marker in markers if marker in content)
                        score += (marker_matches / len(markers)) * 0.1
                
                signal_scores[signal_type] = score
            
            # Return highest scoring signal type if above threshold
            if signal_scores:
                best_signal, best_score = max(signal_scores.items(), key=lambda x: x[1])
                if best_score > 0.3:  # Threshold for classification
                    return best_signal
            
            return None
            
        except Exception as e:
            logger.error(f"Error classifying signal type: {str(e)}")
            return None
    
    async def _calculate_confidence_score(self, event: StreamEvent, signal_type: SignalType) -> float:
        """Calculate confidence in signal classification"""
        try:
            content = event.processed_content.lower()
            patterns = self.signal_patterns.get(signal_type, {})
            
            confidence = 0.5  # Base confidence
            
            # Multiple pattern matches increase confidence
            pattern_types = ['keywords', 'emotional_indicators', 'urgency_markers', 
                           'vulnerability_indicators']
            
            matches_found = 0
            for pattern_type in pattern_types:
                pattern_list = patterns.get(pattern_type, [])
                if pattern_list and any(pattern in content for pattern in pattern_list):
                    matches_found += 1
            
            confidence += (matches_found / len(pattern_types)) * 0.3
            
            # Content length consideration (longer posts might be more confident)
            if len(event.processed_content) > 100:
                confidence += 0.1
            
            # Signal strength from event
            if hasattr(event, 'care_score') and event.care_score > 0.7:
                confidence += 0.1
            
            return min(1.0, confidence)
            
        except Exception as e:
            logger.error(f"Error calculating confidence score: {str(e)}")
            return 0.5
    
    async def _calculate_authenticity_score(self, event: StreamEvent, signal_type: SignalType) -> float:
        """Calculate authenticity of the care signal"""
        try:
            content = event.processed_content.lower()
            authenticity = 0.5  # Base authenticity
            
            # Personal experience indicators
            personal_indicators = [
                'i feel', 'my experience', 'i\'ve been', 'for me', 'in my case',
                'personally', 'i struggle', 'i\'m going through'
            ]
            
            personal_matches = sum(1 for indicator in personal_indicators if indicator in content)
            authenticity += min(0.3, personal_matches * 0.1)
            
            # Vulnerability indicators
            vulnerability_indicators = [
                'hard to admit', 'scared to share', 'vulnerable', 'embarrassed',
                'ashamed', 'first time', 'never told'
            ]
            
            vulnerability_matches = sum(1 for indicator in vulnerability_indicators if indicator in content)
            authenticity += min(0.2, vulnerability_matches * 0.1)
            
            # Specificity (authentic posts often have specific details)
            specific_words = [
                'yesterday', 'today', 'this morning', 'last week', 'specific',
                'exactly', 'particularly', 'precisely'
            ]
            
            specificity_matches = sum(1 for word in specific_words if word in content)
            authenticity += min(0.1, specificity_matches * 0.05)
            
            # Penalty for promotional language
            promotional_words = [
                'buy', 'purchase', 'sale', 'discount', 'promo', 'link in bio',
                'dm for info', 'click here'
            ]
            
            promotional_matches = sum(1 for word in promotional_words if word in content)
            authenticity -= promotional_matches * 0.1
            
            return max(0.0, min(1.0, authenticity))
            
        except Exception as e:
            logger.error(f"Error calculating authenticity score: {str(e)}")
            return 0.5
    
    async def _calculate_urgency_level(self, event: StreamEvent, signal_type: SignalType) -> float:
        """Calculate urgency level of the care signal"""
        try:
            content = event.processed_content.lower()
            urgency = 0.0
            
            # Crisis indicators (highest urgency)
            crisis_indicators = [
                'emergency', 'urgent', 'crisis', 'immediate', 'help me now',
                'can\'t wait', 'asap', 'desperate'
            ]
            
            crisis_matches = sum(1 for indicator in crisis_indicators if indicator in content)
            if crisis_matches > 0:
                urgency = 0.9
            
            # High urgency indicators
            high_urgency_indicators = [
                'breaking down', 'can\'t cope', 'at my limit', 'losing it',
                'falling apart', 'overwhelmed'
            ]
            
            high_urgency_matches = sum(1 for indicator in high_urgency_indicators if indicator in content)
            if high_urgency_matches > 0 and urgency < 0.7:
                urgency = 0.7
            
            # Moderate urgency indicators
            moderate_urgency_indicators = [
                'struggling', 'difficult', 'hard time', 'need help',
                'looking for advice', 'don\'t know what to do'
            ]
            
            moderate_urgency_matches = sum(1 for indicator in moderate_urgency_indicators if indicator in content)
            if moderate_urgency_matches > 0 and urgency < 0.5:
                urgency = 0.5
            
            # Signal type influences urgency
            if signal_type == SignalType.SUPPORT_SEEKING:
                urgency = max(urgency, 0.6)
            elif signal_type == SignalType.EMOTIONAL_DISTRESS:
                urgency = max(urgency, 0.7)
            
            return urgency
            
        except Exception as e:
            logger.error(f"Error calculating urgency level: {str(e)}")
            return 0.3
    
    async def _calculate_emotional_intensity(self, event: StreamEvent) -> float:
        """Calculate emotional intensity of the content"""
        try:
            content = event.processed_content.lower()
            
            # High intensity emotions
            high_intensity = [
                'devastated', 'destroyed', 'shattered', 'broken', 'crushed',
                'overwhelmed', 'desperate', 'hopeless', 'terrified'
            ]
            
            # Medium intensity emotions
            medium_intensity = [
                'sad', 'worried', 'anxious', 'stressed', 'frustrated',
                'confused', 'lost', 'tired', 'scared'
            ]
            
            # Low intensity emotions
            low_intensity = [
                'concerned', 'unsure', 'uncertain', 'questioning',
                'wondering', 'thinking about'
            ]
            
            high_matches = sum(1 for emotion in high_intensity if emotion in content)
            medium_matches = sum(1 for emotion in medium_intensity if emotion in content)
            low_matches = sum(1 for emotion in low_intensity if emotion in content)
            
            # Calculate weighted intensity
            intensity = (high_matches * 0.9 + medium_matches * 0.6 + low_matches * 0.3) / 3
            
            # Content features that indicate intensity
            intensity_features = [
                '!!!', 'all caps words', 'repeated letters', 'multiple question marks'
            ]
            
            # Check for all caps (simplified)
            if any(word.isupper() and len(word) > 3 for word in event.processed_content.split()):
                intensity += 0.2
            
            # Check for multiple exclamation marks
            if '!!!' in event.processed_content:
                intensity += 0.1
            
            return min(1.0, intensity)
            
        except Exception as e:
            logger.error(f"Error calculating emotional intensity: {str(e)}")
            return 0.3
    
    async def _identify_vulnerability_indicators(self, event: StreamEvent) -> List[str]:
        """Identify vulnerability indicators in the content"""
        try:
            content = event.processed_content.lower()
            indicators = []
            
            vulnerability_patterns = {
                'first_time_sharing': ['first time', 'never shared', 'never told anyone'],
                'fear_of_judgment': ['afraid to ask', 'scared to share', 'embarrassed', 'ashamed'],
                'trust_building': ['trusting you', 'taking courage', 'brave enough'],
                'isolation': ['alone', 'no one to talk to', 'isolated', 'lonely'],
                'emotional_overwhelm': ['can\'t handle', 'too much', 'overwhelming'],
                'self_doubt': ['not sure if', 'maybe i\'m wrong', 'probably stupid'],
                'desperation': ['desperate', 'at my limit', 'last resort']
            }
            
            for indicator_type, patterns in vulnerability_patterns.items():
                if any(pattern in content for pattern in patterns):
                    indicators.append(indicator_type)
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error identifying vulnerability indicators: {str(e)}")
            return []
    
    async def _analyze_cultural_context(self, event: StreamEvent) -> List[str]:
        """Analyze cultural context from the content"""
        try:
            content = event.processed_content.lower()
            cultural_context = []
            
            # Cultural markers
            cultural_indicators = {
                'family_oriented': ['family', 'parents', 'children', 'siblings', 'relatives'],
                'community_focused': ['community', 'neighborhood', 'village', 'tribe'],
                'religious_spiritual': ['god', 'prayer', 'faith', 'spiritual', 'church', 'temple'],
                'traditional_values': ['tradition', 'custom', 'heritage', 'ancestors'],
                'collectivist_language': ['we', 'us', 'our community', 'together'],
                'individualist_language': ['i', 'me', 'myself', 'my own', 'personal']
            }
            
            for context_type, indicators in cultural_indicators.items():
                if any(indicator in content for indicator in indicators):
                    cultural_context.append(context_type)
            
            # Language patterns (simplified - would integrate with language detection)
            # This would be expanded with actual language detection and cultural analysis
            
            return cultural_context
            
        except Exception as e:
            logger.error(f"Error analyzing cultural context: {str(e)}")
            return []
    
    async def _analyze_trauma_considerations(self, event: StreamEvent) -> List[str]:
        """Analyze trauma considerations for response planning"""
        try:
            content = event.processed_content.lower()
            considerations = []
            
            trauma_indicators = {
                'ptsd_indicators': ['flashbacks', 'nightmares', 'triggered', 'trauma'],
                'abuse_survivor': ['abuse', 'survivor', 'victim', 'perpetrator'],
                'trust_issues': ['trust issues', 'hard to trust', 'betrayed'],
                'hypervigilance': ['on edge', 'hypervigilant', 'always watching'],
                'dissociation': ['disconnected', 'not real', 'floating', 'detached'],
                'emotional_numbing': ['numb', 'empty', 'void', 'nothing'],
                'avoidance': ['avoiding', 'can\'t face', 'too scary', 'run away']
            }
            
            for trauma_type, indicators in trauma_indicators.items():
                if any(indicator in content for indicator in indicators):
                    considerations.append(trauma_type)
            
            # General trauma-informed considerations
            if any(word in content for word in ['trauma', 'ptsd', 'abuse', 'survivor']):
                considerations.append('trauma_informed_approach_required')
            
            if any(word in content for word in ['trigger', 'triggered', 'triggering']):
                considerations.append('trigger_awareness_needed')
            
            return considerations
            
        except Exception as e:
            logger.error(f"Error analyzing trauma considerations: {str(e)}")
            return []
    
    async def _analyze_accessibility_needs(self, event: StreamEvent) -> List[str]:
        """Analyze accessibility needs for response"""
        try:
            content = event.processed_content.lower()
            accessibility_needs = []
            
            # Accessibility indicators
            accessibility_indicators = {
                'visual_impairment': ['blind', 'visually impaired', 'screen reader'],
                'hearing_impairment': ['deaf', 'hard of hearing', 'hearing aid'],
                'cognitive_accessibility': ['learning disability', 'dyslexia', 'adhd', 'autism'],
                'motor_impairment': ['mobility', 'wheelchair', 'motor disability'],
                'language_processing': ['english second language', 'esl', 'non-native speaker'],
                'cognitive_overload': ['overwhelmed', 'too much information', 'confusing']
            }
            
            for need_type, indicators in accessibility_indicators.items():
                if any(indicator in content for indicator in indicators):
                    accessibility_needs.append(need_type)
            
            # Standard accessibility needs
            accessibility_needs.extend([
                'plain_language_preferred',
                'clear_structure_needed',
                'mobile_friendly_required'
            ])
            
            return accessibility_needs
            
        except Exception as e:
            logger.error(f"Error analyzing accessibility needs: {str(e)}")
            return []
    
    async def _analyze_social_context(self, event: StreamEvent) -> Dict[str, Any]:
        """Analyze social context of the signal"""
        try:
            # This would integrate with social network analysis
            # For now, provide basic context from event metadata
            
            social_context = {
                'platform': event.stream_type.value,
                'public_post': event.privacy_level == 'public',
                'author_info_available': bool(event.author_info),
                'has_replies': False,  # Would be determined from platform data
                'community_context': 'general',  # Would be determined from platform/community
                'time_of_day': event.timestamp.hour,
                'day_of_week': event.timestamp.weekday()
            }
            
            # Analyze for social isolation indicators
            isolation_indicators = [
                'no one to talk to', 'alone', 'isolated', 'no friends',
                'family doesn\'t understand', 'nobody cares'
            ]
            
            if any(indicator in event.processed_content.lower() for indicator in isolation_indicators):
                social_context['isolation_indicators'] = True
            
            return social_context
            
        except Exception as e:
            logger.error(f"Error analyzing social context: {str(e)}")
            return {}
    
    async def _identify_support_types_needed(self, event: StreamEvent, signal_type: SignalType) -> List[str]:
        """Identify what types of support are needed"""
        try:
            content = event.processed_content.lower()
            support_types = []
            
            # Map signal types to likely support needs
            signal_support_mapping = {
                SignalType.SUPPORT_SEEKING: ['emotional_support', 'practical_advice', 'resource_sharing'],
                SignalType.EMOTIONAL_DISTRESS: ['emotional_support', 'crisis_intervention', 'professional_help'],
                SignalType.SUPPORT_OFFERING: ['community_connection', 'peer_matching'],
                SignalType.VULNERABILITY_SHARING: ['validation', 'emotional_support', 'safe_space'],
                SignalType.WISDOM_SHARING: ['appreciation', 'amplification', 'community_sharing']
            }
            
            support_types.extend(signal_support_mapping.get(signal_type, []))
            
            # Content-specific support identification
            support_patterns = {
                'emotional_support': ['need someone to listen', 'feeling alone', 'emotional support'],
                'practical_advice': ['how to', 'what should i do', 'advice', 'suggestions'],
                'resource_sharing': ['resources', 'information', 'where to find'],
                'professional_help': ['therapy', 'counseling', 'professional', 'doctor'],
                'crisis_intervention': ['crisis', 'emergency', 'urgent', 'immediate'],
                'peer_connection': ['others who', 'people like me', 'similar experience'],
                'validation': ['am i crazy', 'normal', 'validate', 'understand'],
                'community_belonging': ['belong', 'accepted', 'welcome', 'community']
            }
            
            for support_type, patterns in support_patterns.items():
                if any(pattern in content for pattern in patterns):
                    support_types.append(support_type)
            
            return list(set(support_types))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Error identifying support types needed: {str(e)}")
            return []
    
    async def _identify_potential_responses(self, signal_type: SignalType, urgency_level: float) -> List[ResponseType]:
        """Identify potential response types"""
        try:
            responses = []
            
            # Urgency-based responses
            if urgency_level > 0.8:
                responses.extend([
                    ResponseType.PROFESSIONAL_REFERRAL,
                    ResponseType.GENTLE_ENGAGEMENT,
                    ResponseType.RESOURCE_SHARING
                ])
            elif urgency_level > 0.6:
                responses.extend([
                    ResponseType.GENTLE_ENGAGEMENT,
                    ResponseType.COMMUNITY_CONNECTION,
                    ResponseType.RESOURCE_SHARING
                ])
            else:
                responses.extend([
                    ResponseType.VALIDATION_RESPONSE,
                    ResponseType.RESOURCE_SHARING,
                    ResponseType.COMMUNITY_CONNECTION
                ])
            
            # Signal type specific responses
            type_responses = {
                SignalType.SUPPORT_SEEKING: [
                    ResponseType.GENTLE_ENGAGEMENT,
                    ResponseType.RESOURCE_SHARING,
                    ResponseType.COMMUNITY_CONNECTION
                ],
                SignalType.EMOTIONAL_DISTRESS: [
                    ResponseType.GENTLE_ENGAGEMENT,
                    ResponseType.PROFESSIONAL_REFERRAL,
                    ResponseType.VALIDATION_RESPONSE
                ],
                SignalType.SUPPORT_OFFERING: [
                    ResponseType.COMMUNITY_CONNECTION,
                    ResponseType.PEER_MATCHING,
                    ResponseType.WISDOM_APPRECIATION
                ],
                SignalType.VULNERABILITY_SHARING: [
                    ResponseType.VALIDATION_RESPONSE,
                    ResponseType.GENTLE_ENGAGEMENT,
                    ResponseType.COMMUNITY_CONNECTION
                ],
                SignalType.WISDOM_SHARING: [
                    ResponseType.WISDOM_APPRECIATION,
                    ResponseType.CONTENT_AMPLIFICATION,
                    ResponseType.COMMUNITY_CONNECTION
                ]
            }
            
            responses.extend(type_responses.get(signal_type, []))
            
            return list(set(responses))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Error identifying potential responses: {str(e)}")
            return []
    
    async def _recommend_response(self, signal_type: SignalType, urgency_level: float,
                                cultural_context: List[str], trauma_considerations: List[str]) -> ResponseType:
        """Recommend the best response type"""
        try:
            # High urgency gets immediate gentle engagement
            if urgency_level > 0.8:
                if 'trauma_informed_approach_required' in trauma_considerations:
                    return ResponseType.PROFESSIONAL_REFERRAL
                else:
                    return ResponseType.GENTLE_ENGAGEMENT
            
            # Signal type based recommendations
            if signal_type == SignalType.SUPPORT_SEEKING:
                if 'family_oriented' in cultural_context:
                    return ResponseType.COMMUNITY_CONNECTION
                else:
                    return ResponseType.GENTLE_ENGAGEMENT
            
            elif signal_type == SignalType.EMOTIONAL_DISTRESS:
                if trauma_considerations:
                    return ResponseType.PROFESSIONAL_REFERRAL
                else:
                    return ResponseType.VALIDATION_RESPONSE
            
            elif signal_type == SignalType.VULNERABILITY_SHARING:
                return ResponseType.VALIDATION_RESPONSE
            
            elif signal_type == SignalType.WISDOM_SHARING:
                return ResponseType.WISDOM_APPRECIATION
            
            elif signal_type == SignalType.SUPPORT_OFFERING:
                return ResponseType.COMMUNITY_CONNECTION
            
            # Default to gentle engagement
            return ResponseType.GENTLE_ENGAGEMENT
            
        except Exception as e:
            logger.error(f"Error recommending response: {str(e)}")
            return ResponseType.GENTLE_ENGAGEMENT
    
    async def _determine_response_timing(self, urgency_level: float, 
                                       trauma_considerations: List[str]) -> str:
        """Determine appropriate timing for response"""
        try:
            # Crisis situations need immediate response
            if urgency_level > 0.8:
                return 'immediate'
            
            # High urgency gets prompt response
            if urgency_level > 0.6:
                return 'within_minutes'
            
            # Trauma considerations may require slower approach
            if trauma_considerations:
                return 'gentle_timing_allow_processing'
            
            # Standard timing for most situations
            if urgency_level > 0.4:
                return 'within_hours'
            
            # Low urgency can wait
            return 'within_day'
            
        except Exception as e:
            logger.error(f"Error determining response timing: {str(e)}")
            return 'standard_timing'
    
    async def _analyze_safety_considerations(self, event: StreamEvent, 
                                           trauma_considerations: List[str]) -> List[str]:
        """Analyze safety considerations for response"""
        try:
            safety_considerations = []
            
            # Trauma-specific safety
            if trauma_considerations:
                safety_considerations.extend([
                    'trauma_informed_approach',
                    'gentle_language_required',
                    'avoid_triggering_content',
                    'respect_boundaries'
                ])
            
            # Vulnerability-based safety
            if any('crisis_' in signal for signal in event.detected_signals):
                safety_considerations.extend([
                    'crisis_protocols_active',
                    'professional_backup_ready',
                    'suicide_prevention_resources'
                ])
            
            # Privacy safety
            if event.privacy_level != 'public':
                safety_considerations.append('respect_privacy_level')
            
            # General safety
            safety_considerations.extend([
                'no_medical_advice',
                'no_therapy_replacement',
                'encourage_professional_help_when_appropriate'
            ])
            
            return safety_considerations
            
        except Exception as e:
            logger.error(f"Error analyzing safety considerations: {str(e)}")
            return []
    
    async def _analyze_privacy_requirements(self, event: StreamEvent) -> List[str]:
        """Analyze privacy requirements for response"""
        try:
            privacy_requirements = []
            
            # Platform-based privacy
            if event.privacy_level == 'private':
                privacy_requirements.append('private_response_only')
            elif event.privacy_level == 'limited':
                privacy_requirements.append('limited_visibility_response')
            
            # Content-based privacy
            if any(word in event.processed_content.lower() for word in 
                   ['confidential', 'private', 'secret', 'don\'t share']):
                privacy_requirements.append('explicit_confidentiality_request')
            
            # Standard privacy
            privacy_requirements.extend([
                'no_personal_info_sharing',
                'user_consent_for_connections',
                'data_minimization'
            ])
            
            return privacy_requirements
            
        except Exception as e:
            logger.error(f"Error analyzing privacy requirements: {str(e)}")
            return []
    
    async def _create_response_plan(self, analysis: CareSignalAnalysis) -> Optional[ResponsePlan]:
        """Create a detailed response plan"""
        try:
            # Get response strategy
            strategy = self.response_strategies.get(analysis.recommended_response, {})
            
            # Determine priority
            if analysis.urgency_level > 0.8:
                priority = ProcessingPriority.IMMEDIATE
            elif analysis.urgency_level > 0.6:
                priority = ProcessingPriority.URGENT
            else:
                priority = ProcessingPriority.STANDARD
            
            # Create personalization factors
            personalization = await self._create_personalization_factors(analysis)
            
            # Create cultural adaptations
            cultural_adaptations = await self._create_cultural_adaptations(analysis)
            
            # Create accessibility accommodations
            accessibility_accommodations = await self._create_accessibility_accommodations(analysis)
            
            # Create safety measures
            safety_measures = await self._create_safety_measures(analysis)
            
            # Create trauma-informed elements
            trauma_informed_elements = await self._create_trauma_informed_elements(analysis)
            
            # Generate resource recommendations
            resources = await self._generate_resource_recommendations(analysis)
            
            # Generate community connections
            community_connections = await self._generate_community_connections(analysis)
            
            # Create timing strategy
            timing_strategy = await self._create_timing_strategy(analysis)
            
            # Define success metrics
            success_metrics = await self._define_success_metrics(analysis)
            
            # Set gentle constraints
            gentle_constraints = await self._set_gentle_constraints(analysis)
            
            # Calculate execution window
            execution_window = await self._calculate_execution_window(analysis)
            
            plan = ResponsePlan(
                plan_id=f"response_plan_{datetime.utcnow().isoformat()}_{id(self)}",
                signal_analysis=analysis,
                response_type=analysis.recommended_response,
                response_priority=priority,
                response_approach=strategy.get('approach', 'gentle_supportive'),
                personalization_factors=personalization,
                cultural_adaptations=cultural_adaptations,
                accessibility_accommodations=accessibility_accommodations,
                safety_measures=safety_measures,
                trauma_informed_elements=trauma_informed_elements,
                resource_recommendations=resources,
                community_connections=community_connections,
                timing_strategy=timing_strategy,
                success_metrics=success_metrics,
                gentle_constraints=gentle_constraints,
                created_at=datetime.utcnow(),
                execution_window=execution_window
            )
            
            # Store plan
            self.response_plans[plan.plan_id] = plan
            
            # Trigger callbacks
            for callback in self.response_planned_callbacks:
                try:
                    await callback(plan)
                except Exception as e:
                    logger.error(f"Response planned callback failed: {str(e)}")
            
            return plan
            
        except Exception as e:
            logger.error(f"Error creating response plan: {str(e)}")
            return None
    
    async def _create_personalization_factors(self, analysis: CareSignalAnalysis) -> Dict[str, Any]:
        """Create personalization factors for response"""
        try:
            return {
                'signal_type': analysis.signal_type.value,
                'urgency_level': analysis.urgency_level,
                'emotional_intensity': analysis.emotional_intensity,
                'vulnerability_indicators': analysis.vulnerability_indicators,
                'cultural_context': analysis.cultural_context,
                'support_types_needed': analysis.support_type_needed,
                'accessibility_needs': analysis.accessibility_needs,
                'trauma_informed_required': len(analysis.trauma_considerations) > 0,
                'privacy_sensitive': len(analysis.privacy_requirements) > 0
            }
        except Exception as e:
            logger.error(f"Error creating personalization factors: {str(e)}")
            return {}
    
    async def _create_cultural_adaptations(self, analysis: CareSignalAnalysis) -> List[str]:
        """Create cultural adaptations for response"""
        try:
            adaptations = []
            
            for context in analysis.cultural_context:
                if context in self.cultural_adaptations:
                    adaptation_config = self.cultural_adaptations[context]
                    
                    if adaptation_config.get('community_emphasis'):
                        adaptations.append('emphasize_community_support')
                    
                    if adaptation_config.get('family_consideration'):
                        adaptations.append('consider_family_dynamics')
                    
                    if adaptation_config.get('indirect_communication'):
                        adaptations.append('use_indirect_communication_style')
                    
                    if adaptation_config.get('elder_respect'):
                        adaptations.append('show_respect_for_wisdom_traditions')
            
            return adaptations
            
        except Exception as e:
            logger.error(f"Error creating cultural adaptations: {str(e)}")
            return []
    
    async def _create_accessibility_accommodations(self, analysis: CareSignalAnalysis) -> List[str]:
        """Create accessibility accommodations for response"""
        try:
            accommodations = []
            
            for need in analysis.accessibility_needs:
                if need == 'visual_impairment':
                    accommodations.extend([
                        'screen_reader_friendly_format',
                        'descriptive_text_only',
                        'avoid_visual_only_information'
                    ])
                elif need == 'cognitive_accessibility':
                    accommodations.extend([
                        'plain_language_only',
                        'clear_structure',
                        'bullet_points_for_clarity',
                        'avoid_overwhelming_information'
                    ])
                elif need == 'language_processing':
                    accommodations.extend([
                        'simple_sentence_structure',
                        'avoid_idioms_and_slang',
                        'cultural_translation_awareness'
                    ])
            
            # Standard accommodations
            accommodations.extend([
                'mobile_friendly_formatting',
                'clear_action_steps',
                'multiple_format_options'
            ])
            
            return list(set(accommodations))
            
        except Exception as e:
            logger.error(f"Error creating accessibility accommodations: {str(e)}")
            return []
    
    async def _create_safety_measures(self, analysis: CareSignalAnalysis) -> List[str]:
        """Create safety measures for response"""
        try:
            safety_measures = []
            
            # Based on safety considerations
            for consideration in analysis.safety_considerations:
                if consideration == 'crisis_protocols_active':
                    safety_measures.extend([
                        'crisis_hotline_information_ready',
                        'professional_escalation_prepared',
                        'emergency_contact_protocols'
                    ])
                elif consideration == 'trauma_informed_approach':
                    safety_measures.extend([
                        'avoid_triggering_language',
                        'respect_emotional_boundaries',
                        'gentle_pacing'
                    ])
            
            # Standard safety measures
            safety_measures.extend([
                'no_medical_diagnosis_or_advice',
                'encourage_professional_help_when_appropriate',
                'respect_user_autonomy_and_choice'
            ])
            
            return list(set(safety_measures))
            
        except Exception as e:
            logger.error(f"Error creating safety measures: {str(e)}")
            return []
    
    async def _create_trauma_informed_elements(self, analysis: CareSignalAnalysis) -> List[str]:
        """Create trauma-informed elements for response"""
        try:
            elements = []
            
            if analysis.trauma_considerations:
                elements.extend([
                    'acknowledge_courage_in_sharing',
                    'validate_feelings_and_experiences',
                    'emphasize_user_control_and_choice',
                    'avoid_assumptions_about_experience',
                    'provide_grounding_techniques_if_appropriate',
                    'respect_boundaries_and_pacing',
                    'focus_on_strengths_and_resilience'
                ])
            
            # Always include basic trauma-informed principles
            elements.extend([
                'use_empowering_language',
                'avoid_blame_or_judgment',
                'respect_cultural_and_gender_identities'
            ])
            
            return list(set(elements))
            
        except Exception as e:
            logger.error(f"Error creating trauma-informed elements: {str(e)}")
            return []
    
    async def _generate_resource_recommendations(self, analysis: CareSignalAnalysis) -> List[Dict[str, Any]]:
        """Generate resource recommendations"""
        try:
            resources = []
            
            # Based on support types needed
            for support_type in analysis.support_type_needed:
                if support_type == 'crisis_intervention':
                    resources.append({
                        'type': 'crisis_hotline',
                        'name': 'Crisis Text Line',
                        'contact': 'Text HOME to 741741',
                        'description': '24/7 crisis support'
                    })
                elif support_type == 'professional_help':
                    resources.append({
                        'type': 'therapy_finder',
                        'name': 'Psychology Today',
                        'url': 'psychologytoday.com/therapists',
                        'description': 'Find therapists in your area'
                    })
                elif support_type == 'emotional_support':
                    resources.append({
                        'type': 'support_community',
                        'name': 'Heart Protocol Community',
                        'description': 'Healing-focused support community'
                    })
            
            return resources
            
        except Exception as e:
            logger.error(f"Error generating resource recommendations: {str(e)}")
            return []
    
    async def _generate_community_connections(self, analysis: CareSignalAnalysis) -> List[Dict[str, Any]]:
        """Generate community connection recommendations"""
        try:
            connections = []
            
            # Based on signal type and cultural context
            if analysis.signal_type == SignalType.SUPPORT_SEEKING:
                connections.append({
                    'type': 'peer_support_group',
                    'description': 'Connect with others seeking similar support',
                    'approach': 'gentle_introduction'
                })
            
            if 'community_focused' in analysis.cultural_context:
                connections.append({
                    'type': 'cultural_community',
                    'description': 'Connect with culturally resonant support',
                    'approach': 'cultural_bridge_building'
                })
            
            return connections
            
        except Exception as e:
            logger.error(f"Error generating community connections: {str(e)}")
            return []
    
    async def _create_timing_strategy(self, analysis: CareSignalAnalysis) -> Dict[str, Any]:
        """Create timing strategy for response"""
        try:
            return {
                'response_timing': analysis.response_timing,
                'urgency_level': analysis.urgency_level,
                'follow_up_schedule': 'based_on_user_response',
                'gentle_pacing': analysis.urgency_level < 0.6,
                'immediate_if_crisis': analysis.urgency_level > 0.8
            }
        except Exception as e:
            logger.error(f"Error creating timing strategy: {str(e)}")
            return {}
    
    async def _define_success_metrics(self, analysis: CareSignalAnalysis) -> Dict[str, Any]:
        """Define success metrics for response"""
        try:
            return {
                'user_engagement': 'positive_response_or_acknowledgment',
                'healing_progress': 'indicators_of_improved_wellbeing',
                'connection_quality': 'meaningful_human_connection_established',
                'resource_utilization': 'helpful_resources_accessed',
                'safety_maintained': 'no_harm_caused_by_response',
                'autonomy_respected': 'user_choice_and_control_maintained'
            }
        except Exception as e:
            logger.error(f"Error defining success metrics: {str(e)}")
            return {}
    
    async def _set_gentle_constraints(self, analysis: CareSignalAnalysis) -> Dict[str, Any]:
        """Set gentle constraints for response"""
        try:
            return {
                'maximum_response_length': 'three_paragraphs_or_less',
                'tone_requirement': 'warm_supportive_non_judgmental',
                'advice_limitation': 'no_medical_or_legal_advice',
                'privacy_protection': 'respect_all_privacy_requirements',
                'cultural_sensitivity': 'adapt_to_cultural_context',
                'trauma_awareness': 'avoid_potential_triggers',
                'user_autonomy': 'respect_user_choice_and_boundaries'
            }
        except Exception as e:
            logger.error(f"Error setting gentle constraints: {str(e)}")
            return {}
    
    async def _calculate_execution_window(self, analysis: CareSignalAnalysis) -> Tuple[datetime, datetime]:
        """Calculate execution window for response"""
        try:
            current_time = datetime.utcnow()
            
            if analysis.response_timing == 'immediate':
                start_time = current_time
                end_time = current_time + timedelta(minutes=5)
            elif analysis.response_timing == 'within_minutes':
                start_time = current_time + timedelta(minutes=2)
                end_time = current_time + timedelta(minutes=15)
            elif analysis.response_timing == 'within_hours':
                start_time = current_time + timedelta(minutes=30)
                end_time = current_time + timedelta(hours=4)
            elif analysis.response_timing == 'within_day':
                start_time = current_time + timedelta(hours=2)
                end_time = current_time + timedelta(hours=24)
            else:
                start_time = current_time + timedelta(minutes=10)
                end_time = current_time + timedelta(hours=2)
            
            return (start_time, end_time)
            
        except Exception as e:
            logger.error(f"Error calculating execution window: {str(e)}")
            current_time = datetime.utcnow()
            return (current_time, current_time + timedelta(hours=1))
    
    async def _execute_response_plan(self, plan: ResponsePlan) -> Optional[ProcessingResult]:
        """Execute a response plan"""
        try:
            start_time = datetime.utcnow()
            
            # Check execution timing
            current_time = datetime.utcnow()
            start_window, end_window = plan.execution_window
            
            if current_time < start_window:
                # Too early, schedule for later
                logger.debug(f"Response plan {plan.plan_id} scheduled for later execution")
                return None
            
            if current_time > end_window:
                # Too late, may not be relevant anymore
                logger.warning(f"Response plan {plan.plan_id} executed past optimal window")
            
            # Simulate response execution (in production, would integrate with response systems)
            execution_details = await self._simulate_response_execution(plan)
            
            # Calculate processing duration
            processing_duration = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Simulate effectiveness scoring (in production, would be based on actual outcomes)
            effectiveness_score = await self._simulate_effectiveness_scoring(plan)
            healing_impact = await self._simulate_healing_impact(plan)
            user_satisfaction = await self._simulate_user_satisfaction(plan)
            
            # Create result
            result = ProcessingResult(
                result_id=f"processing_result_{datetime.utcnow().isoformat()}_{id(self)}",
                signal_id=plan.signal_analysis.signal_id,
                processing_stage=ProcessingStage.COMPLETED,
                response_executed=True,
                response_type=plan.response_type,
                execution_details=execution_details,
                user_response=None,  # Would be captured from actual user interaction
                effectiveness_score=effectiveness_score,
                healing_impact=healing_impact,
                user_satisfaction=user_satisfaction,
                lessons_learned=await self._extract_lessons_learned(plan, execution_details),
                follow_up_needed=effectiveness_score < 0.7,
                follow_up_timing=datetime.utcnow() + timedelta(days=1) if effectiveness_score < 0.7 else None,
                completed_at=datetime.utcnow(),
                processing_duration_ms=processing_duration
            )
            
            # Update metrics
            self.responses_executed += 1
            self.total_healing_impact += healing_impact
            self.user_satisfaction_average = (
                (self.user_satisfaction_average * (self.responses_executed - 1) + user_satisfaction)
                / self.responses_executed
            )
            
            # Remove from active plans
            if plan.plan_id in self.response_plans:
                del self.response_plans[plan.plan_id]
            
            # Remove from active analyses
            if plan.signal_analysis.signal_id in self.active_analyses:
                del self.active_analyses[plan.signal_analysis.signal_id]
            
            # Trigger callbacks
            for callback in self.response_executed_callbacks:
                try:
                    await callback(result)
                except Exception as e:
                    logger.error(f"Response executed callback failed: {str(e)}")
            
            for callback in self.processing_completed_callbacks:
                try:
                    await callback(result)
                except Exception as e:
                    logger.error(f"Processing completed callback failed: {str(e)}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing response plan: {str(e)}")
            return None
    
    async def _simulate_response_execution(self, plan: ResponsePlan) -> Dict[str, Any]:
        """Simulate response execution (placeholder for real implementation)"""
        try:
            # In production, this would integrate with actual response delivery systems
            return {
                'response_delivered': True,
                'delivery_method': 'platform_native',
                'personalization_applied': len(plan.personalization_factors) > 0,
                'cultural_adaptations_used': len(plan.cultural_adaptations) > 0,
                'accessibility_accommodations_applied': len(plan.accessibility_accommodations) > 0,
                'safety_measures_implemented': len(plan.safety_measures) > 0,
                'resources_shared': len(plan.resource_recommendations),
                'community_connections_facilitated': len(plan.community_connections),
                'delivery_timestamp': datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error simulating response execution: {str(e)}")
            return {}
    
    async def _simulate_effectiveness_scoring(self, plan: ResponsePlan) -> float:
        """Simulate effectiveness scoring (placeholder for real measurement)"""
        try:
            # In production, would be based on actual user engagement and outcomes
            base_score = 0.7
            
            # Boost for high-quality planning
            if len(plan.cultural_adaptations) > 0:
                base_score += 0.1
            
            if len(plan.accessibility_accommodations) > 0:
                base_score += 0.1
            
            if len(plan.trauma_informed_elements) > 0:
                base_score += 0.1
            
            return min(1.0, base_score)
            
        except Exception as e:
            logger.error(f"Error simulating effectiveness scoring: {str(e)}")
            return 0.5
    
    async def _simulate_healing_impact(self, plan: ResponsePlan) -> float:
        """Simulate healing impact measurement (placeholder for real measurement)"""
        try:
            # In production, would be based on long-term user wellbeing outcomes
            base_impact = 0.6
            
            # Higher impact for vulnerable users who receive appropriate support
            if len(plan.signal_analysis.vulnerability_indicators) > 0:
                base_impact += 0.2
            
            # Higher impact for trauma-informed responses
            if len(plan.trauma_informed_elements) > 0:
                base_impact += 0.1
            
            # Higher impact for culturally sensitive responses
            if len(plan.cultural_adaptations) > 0:
                base_impact += 0.1
            
            return min(1.0, base_impact)
            
        except Exception as e:
            logger.error(f"Error simulating healing impact: {str(e)}")
            return 0.5
    
    async def _simulate_user_satisfaction(self, plan: ResponsePlan) -> float:
        """Simulate user satisfaction measurement (placeholder for real measurement)"""
        try:
            # In production, would be based on actual user feedback
            base_satisfaction = 0.7
            
            # Higher satisfaction for personalized responses
            if len(plan.personalization_factors) > 0:
                base_satisfaction += 0.1
            
            # Higher satisfaction for accessible responses
            if len(plan.accessibility_accommodations) > 0:
                base_satisfaction += 0.1
            
            # Higher satisfaction for appropriate timing
            if plan.response_priority == ProcessingPriority.IMMEDIATE:
                base_satisfaction += 0.1
            
            return min(1.0, base_satisfaction)
            
        except Exception as e:
            logger.error(f"Error simulating user satisfaction: {str(e)}")
            return 0.5
    
    async def _extract_lessons_learned(self, plan: ResponsePlan, execution_details: Dict[str, Any]) -> List[str]:
        """Extract lessons learned from response execution"""
        try:
            lessons = []
            
            # Plan quality lessons
            if len(plan.cultural_adaptations) > 0:
                lessons.append("Cultural adaptations improve response relevance")
            
            if len(plan.accessibility_accommodations) > 0:
                lessons.append("Accessibility accommodations increase user engagement")
            
            if len(plan.trauma_informed_elements) > 0:
                lessons.append("Trauma-informed approaches build trust and safety")
            
            # Execution lessons
            if execution_details.get('delivery_method') == 'platform_native':
                lessons.append("Platform-native delivery maintains user context")
            
            # Signal type lessons
            signal_type = plan.signal_analysis.signal_type
            if signal_type == SignalType.VULNERABILITY_SHARING:
                lessons.append("Vulnerability sharing requires validation and gentle response")
            elif signal_type == SignalType.SUPPORT_SEEKING:
                lessons.append("Support seeking benefits from resource sharing and connection")
            
            return lessons
            
        except Exception as e:
            logger.error(f"Error extracting lessons learned: {str(e)}")
            return []
    
    async def _follow_up_processor(self):
        """Process follow-up tasks"""
        try:
            while self.processing_active:
                try:
                    # Check for results that need follow-up
                    current_time = datetime.utcnow()
                    
                    for result in self.processing_results:
                        if (result.follow_up_needed and 
                            result.follow_up_timing and 
                            current_time >= result.follow_up_timing):
                            
                            await self._execute_follow_up(result)
                            result.follow_up_needed = False
                    
                    await asyncio.sleep(300)  # Check every 5 minutes
                    
                except Exception as e:
                    logger.error(f"Error in follow-up processor: {str(e)}")
                    await asyncio.sleep(300)
                    
        except Exception as e:
            logger.error(f"Fatal error in follow-up processor: {str(e)}")
    
    async def _execute_follow_up(self, result: ProcessingResult):
        """Execute follow-up actions"""
        try:
            logger.info(f"Executing follow-up for result {result.result_id}")
            
            # In production, would implement actual follow-up logic
            # For now, just log the follow-up
            
        except Exception as e:
            logger.error(f"Error executing follow-up: {str(e)}")
    
    async def _analytics_processor(self):
        """Process analytics and patterns"""
        try:
            while self.processing_active:
                try:
                    await asyncio.sleep(3600)  # Run every hour
                    
                    # Analyze patterns in processed signals
                    await self._analyze_processing_patterns()
                    
                except Exception as e:
                    logger.error(f"Error in analytics processor: {str(e)}")
                    await asyncio.sleep(3600)
                    
        except Exception as e:
            logger.error(f"Fatal error in analytics processor: {str(e)}")
    
    async def _analyze_processing_patterns(self):
        """Analyze patterns in signal processing"""
        try:
            if len(self.processing_results) < 10:
                return
            
            # Analyze effectiveness patterns
            recent_results = self.processing_results[-100:]  # Last 100 results
            
            avg_effectiveness = statistics.mean([r.effectiveness_score for r in recent_results])
            avg_healing_impact = statistics.mean([r.healing_impact for r in recent_results])
            avg_satisfaction = statistics.mean([r.user_satisfaction for r in recent_results])
            
            logger.info(f"Processing analytics: effectiveness={avg_effectiveness:.3f}, "
                       f"healing_impact={avg_healing_impact:.3f}, satisfaction={avg_satisfaction:.3f}")
            
        except Exception as e:
            logger.error(f"Error analyzing processing patterns: {str(e)}")
    
    # Callback management
    def add_signal_analyzed_callback(self, callback: Callable):
        """Add callback for signal analysis completion"""
        self.signal_analyzed_callbacks.append(callback)
    
    def add_response_planned_callback(self, callback: Callable):
        """Add callback for response plan creation"""
        self.response_planned_callbacks.append(callback)
    
    def add_response_executed_callback(self, callback: Callable):
        """Add callback for response execution"""
        self.response_executed_callbacks.append(callback)
    
    def add_processing_completed_callback(self, callback: Callable):
        """Add callback for processing completion"""
        self.processing_completed_callbacks.append(callback)
    
    # Analytics and reporting
    def get_processing_analytics(self) -> Dict[str, Any]:
        """Get comprehensive processing analytics"""
        try:
            if not self.processing_results:
                return {
                    'signals_processed': self.signals_processed,
                    'responses_executed': self.responses_executed,
                    'average_processing_time_ms': self.average_processing_time_ms,
                    'total_healing_impact': self.total_healing_impact,
                    'user_satisfaction_average': self.user_satisfaction_average,
                    'active_analyses': len(self.active_analyses),
                    'active_response_plans': len(self.response_plans)
                }
            
            # Calculate detailed analytics
            recent_results = self.processing_results[-100:] if len(self.processing_results) > 100 else self.processing_results
            
            effectiveness_scores = [r.effectiveness_score for r in recent_results]
            healing_impacts = [r.healing_impact for r in recent_results]
            satisfaction_scores = [r.user_satisfaction for r in recent_results]
            processing_times = [r.processing_duration_ms for r in recent_results]
            
            return {
                'signals_processed': self.signals_processed,
                'responses_executed': self.responses_executed,
                'average_processing_time_ms': self.average_processing_time_ms,
                'total_healing_impact': self.total_healing_impact,
                'user_satisfaction_average': self.user_satisfaction_average,
                'active_analyses': len(self.active_analyses),
                'active_response_plans': len(self.response_plans),
                'recent_effectiveness_average': statistics.mean(effectiveness_scores),
                'recent_healing_impact_average': statistics.mean(healing_impacts),
                'recent_satisfaction_average': statistics.mean(satisfaction_scores),
                'recent_processing_time_average': statistics.mean(processing_times),
                'follow_up_rate': sum(1 for r in recent_results if r.follow_up_needed) / len(recent_results),
                'response_success_rate': sum(1 for r in recent_results if r.response_executed) / len(recent_results)
            }
            
        except Exception as e:
            logger.error(f"Error generating processing analytics: {str(e)}")
            return {}