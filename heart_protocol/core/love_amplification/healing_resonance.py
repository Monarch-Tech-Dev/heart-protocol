"""
Healing Resonance Detector

System for detecting, measuring, and amplifying healing resonance patterns
across social networks to create cascading waves of healing and recovery.
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import logging
import statistics
import math
from collections import defaultdict, deque
import numpy as np

logger = logging.getLogger(__name__)


class ResonanceType(Enum):
    """Types of healing resonance patterns"""
    COLLECTIVE_HEALING = "collective_healing"           # Multiple people healing together
    INSPIRATION_WAVE = "inspiration_wave"               # Inspiring others to start healing
    WISDOM_CASCADE = "wisdom_cascade"                   # Sharing wisdom triggers more sharing
    SUPPORT_NETWORK = "support_network"                 # Support creates more support
    BREAKTHROUGH_RIPPLE = "breakthrough_ripple"         # Breakthroughs inspire others
    GRATITUDE_AMPLIFICATION = "gratitude_amplification" # Gratitude creates more gratitude
    COURAGE_CONTAGION = "courage_contagion"            # Courage spreads to others
    HOPE_MULTIPLICATION = "hope_multiplication"         # Hope spreads exponentially
    RECOVERY_MOMENTUM = "recovery_momentum"             # Recovery builds on recovery
    VULNERABILITY_SAFETY = "vulnerability_safety"       # Safe vulnerability creates more safety


class HealingPhase(Enum):
    """Phases of healing journey"""
    CRISIS_INTERVENTION = "crisis_intervention"         # Immediate crisis support
    INITIAL_AWARENESS = "initial_awareness"             # Becoming aware of need for healing
    SEEKING_HELP = "seeking_help"                       # Actively seeking support
    EARLY_RECOVERY = "early_recovery"                   # Beginning healing process
    ACTIVE_HEALING = "active_healing"                   # Engaged in healing work
    INTEGRATION = "integration"                         # Integrating healing insights
    STABLE_RECOVERY = "stable_recovery"                 # Stable in recovery
    GIVING_BACK = "giving_back"                        # Helping others heal
    WISDOM_SHARING = "wisdom_sharing"                   # Sharing wisdom gained


class ResonanceStrength(Enum):
    """Strength levels of healing resonance"""
    SUBTLE = "subtle"                                   # 0.0-0.3 - Gentle influence
    MODERATE = "moderate"                               # 0.3-0.6 - Clear impact
    STRONG = "strong"                                   # 0.6-0.8 - Significant influence
    POWERFUL = "powerful"                               # 0.8-0.9 - Major transformation
    TRANSFORMATIONAL = "transformational"               # 0.9-1.0 - Life-changing impact


@dataclass
class HealingWave:
    """A detected wave of healing resonance"""
    wave_id: str
    origin_content: str
    origin_author: str
    resonance_type: ResonanceType
    initial_healing_phase: HealingPhase
    wave_strength: float
    strength_level: ResonanceStrength
    propagation_speed: float
    healing_trajectory: Dict[str, float]
    affected_users: List[str]
    secondary_waves: List[str]
    cultural_adaptation: Dict[str, Any]
    accessibility_features: List[str]
    trauma_informed_propagation: bool
    detected_at: datetime
    peak_resonance_time: Optional[datetime]
    wave_completion_time: Optional[datetime]
    total_healing_impact: float
    sustainability_score: float
    authenticity_preservation: float


@dataclass
class ResonancePattern:
    """Pattern for detecting healing resonance"""
    pattern_id: str
    resonance_type: ResonanceType
    trigger_keywords: List[str]
    emotional_indicators: List[str]
    healing_phase_markers: Dict[HealingPhase, List[str]]
    propagation_conditions: Dict[str, Any]
    cultural_variations: Dict[str, List[str]]
    authenticity_markers: List[str]
    sustainability_factors: List[str]


@dataclass
class HealingContext:
    """Context information for healing resonance"""
    user_id: str
    current_healing_phase: HealingPhase
    vulnerability_level: float
    support_network_size: int
    cultural_background: List[str]
    accessibility_needs: List[str]
    trauma_history_indicators: List[str]
    healing_goals: List[str]
    preferred_support_types: List[str]
    safety_requirements: List[str]


class HealingResonanceDetector:
    """
    System for detecting and amplifying healing resonance patterns to create
    cascading waves of healing and recovery across communities.
    
    Core Principles:
    - Healing resonance creates authentic, sustainable change
    - Each person's healing contributes to collective healing
    - Trauma-informed approaches prevent re-traumatization
    - Cultural sensitivity honors diverse healing traditions
    - Authenticity is preserved throughout amplification
    - Vulnerable users receive priority protection and support
    - Healing waves are sustainable and self-reinforcing
    - Community resilience is built through shared healing
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Detection settings
        self.resonance_threshold = self.config.get('resonance_threshold', 0.4)
        self.wave_tracking_window = self.config.get('wave_tracking_window_hours', 72)
        self.max_wave_propagation_hops = self.config.get('max_propagation_hops', 6)
        self.authenticity_minimum = self.config.get('authenticity_minimum', 0.6)
        
        # Resonance patterns
        self.resonance_patterns = self._initialize_resonance_patterns()
        
        # Wave tracking
        self.active_waves: Dict[str, HealingWave] = {}
        self.completed_waves: List[HealingWave] = []
        self.user_healing_contexts: Dict[str, HealingContext] = {}
        self.resonance_network: Dict[str, Set[str]] = defaultdict(set)
        
        # Performance metrics
        self.waves_detected = 0
        self.total_healing_impact = 0.0
        self.users_reached_by_waves = set()
        self.secondary_waves_triggered = 0
        
        # Safety and cultural systems
        self.trauma_safety_validator = None     # Would integrate with trauma safety system
        self.cultural_adaptation_engine = None  # Would integrate with cultural sensitivity engine
        self.accessibility_optimizer = None     # Would integrate with accessibility system
        
        # Callbacks
        self.wave_detected_callbacks: List[Callable] = []
        self.wave_completion_callbacks: List[Callable] = []
        self.healing_impact_callbacks: List[Callable] = []
    
    def _initialize_resonance_patterns(self) -> Dict[ResonanceType, ResonancePattern]:
        """Initialize patterns for detecting different types of healing resonance"""
        return {
            ResonanceType.COLLECTIVE_HEALING: ResonancePattern(
                pattern_id="collective_healing_v1",
                resonance_type=ResonanceType.COLLECTIVE_HEALING,
                trigger_keywords=[
                    "together", "healing circle", "support group", "community healing",
                    "shared journey", "collective growth", "healing together", "united in healing"
                ],
                emotional_indicators=[
                    "connected", "united", "supported", "belonging", "shared strength",
                    "community power", "collective wisdom", "healing bond"
                ],
                healing_phase_markers={
                    HealingPhase.ACTIVE_HEALING: ["working together", "healing alongside", "shared process"],
                    HealingPhase.INTEGRATION: ["learning together", "growing together", "integrating as community"],
                    HealingPhase.GIVING_BACK: ["supporting others", "sharing strength", "lifting each other"]
                },
                propagation_conditions={
                    "community_engagement": 0.7,
                    "shared_vulnerability": 0.6,
                    "mutual_support": 0.8
                },
                cultural_variations={
                    "collectivist_cultures": ["family healing", "ancestor wisdom", "community strength"],
                    "indigenous_traditions": ["circle healing", "tribal wisdom", "generational healing"]
                },
                authenticity_markers=[
                    "personal experience", "vulnerability sharing", "mutual growth",
                    "genuine connection", "real transformation"
                ],
                sustainability_factors=[
                    "ongoing support", "community commitment", "shared responsibility",
                    "healing infrastructure", "collective resources"
                ]
            ),
            
            ResonanceType.INSPIRATION_WAVE: ResonancePattern(
                pattern_id="inspiration_wave_v1",
                resonance_type=ResonanceType.INSPIRATION_WAVE,
                trigger_keywords=[
                    "inspired by", "motivated me", "showed me", "gave me hope",
                    "sparked something", "lit a fire", "opened my eyes", "changed my perspective"
                ],
                emotional_indicators=[
                    "hopeful", "motivated", "inspired", "energized", "awakened",
                    "empowered", "encouraged", "uplifted", "transformed"
                ],
                healing_phase_markers={
                    HealingPhase.INITIAL_AWARENESS: ["realized I need", "opened my eyes", "saw possibility"],
                    HealingPhase.SEEKING_HELP: ["inspired to seek", "motivated to find", "encouraged to try"],
                    HealingPhase.EARLY_RECOVERY: ["started because of", "took first step", "began journey"]
                },
                propagation_conditions={
                    "inspirational_authenticity": 0.8,
                    "relatable_story": 0.7,
                    "hope_generation": 0.9
                },
                cultural_variations={
                    "individual_achievement": ["personal breakthrough", "individual success", "self-discovery"],
                    "community_impact": ["community change", "collective inspiration", "shared awakening"]
                },
                authenticity_markers=[
                    "personal story", "real experience", "honest struggle",
                    "genuine transformation", "vulnerable sharing"
                ],
                sustainability_factors=[
                    "ongoing inspiration", "continued motivation", "sustained hope",
                    "inspirational community", "role model presence"
                ]
            ),
            
            ResonanceType.WISDOM_CASCADE: ResonancePattern(
                pattern_id="wisdom_cascade_v1",
                resonance_type=ResonanceType.WISDOM_CASCADE,
                trigger_keywords=[
                    "learned from", "wisdom shared", "insight from", "teaching moment",
                    "passed on knowledge", "shared experience", "valuable lesson", "wisdom tradition"
                ],
                emotional_indicators=[
                    "enlightened", "understood", "clarified", "illuminated",
                    "wise", "insightful", "knowing", "aware"
                ],
                healing_phase_markers={
                    HealingPhase.INTEGRATION: ["integrating wisdom", "applying insights", "using knowledge"],
                    HealingPhase.WISDOM_SHARING: ["sharing what I learned", "passing on wisdom", "teaching others"],
                    HealingPhase.GIVING_BACK: ["helping others learn", "sharing experience", "mentoring"]
                },
                propagation_conditions={
                    "wisdom_relevance": 0.8,
                    "teaching_clarity": 0.7,
                    "learning_receptivity": 0.6
                },
                cultural_variations={
                    "elder_wisdom": ["elder teaching", "traditional knowledge", "ancestral wisdom"],
                    "peer_learning": ["peer insight", "shared learning", "collaborative wisdom"]
                },
                authenticity_markers=[
                    "earned wisdom", "lived experience", "hard-won insights",
                    "practical knowledge", "tested understanding"
                ],
                sustainability_factors=[
                    "wisdom preservation", "teaching continuity", "learning culture",
                    "knowledge sharing", "mentorship tradition"
                ]
            ),
            
            ResonanceType.SUPPORT_NETWORK: ResonancePattern(
                pattern_id="support_network_v1",
                resonance_type=ResonanceType.SUPPORT_NETWORK,
                trigger_keywords=[
                    "support network", "circle of care", "helping hands", "community support",
                    "network of love", "support system", "caring community", "help network"
                ],
                emotional_indicators=[
                    "supported", "cared for", "surrounded by love", "held up",
                    "carried", "protected", "safe", "secure"
                ],
                healing_phase_markers={
                    HealingPhase.SEEKING_HELP: ["finding support", "building network", "seeking community"],
                    HealingPhase.ACTIVE_HEALING: ["supported healing", "network assistance", "community care"],
                    HealingPhase.STABLE_RECOVERY: ["maintaining support", "ongoing network", "continued care"]
                },
                propagation_conditions={
                    "mutual_support": 0.9,
                    "network_strength": 0.8,
                    "caring_reciprocity": 0.7
                },
                cultural_variations={
                    "family_networks": ["family support", "kinship care", "extended family"],
                    "community_networks": ["neighborhood support", "community care", "village help"]
                },
                authenticity_markers=[
                    "genuine care", "real support", "authentic concern",
                    "consistent help", "reliable presence"
                ],
                sustainability_factors=[
                    "network maintenance", "support sustainability", "care continuity",
                    "relationship strength", "mutual aid"
                ]
            ),
            
            ResonanceType.BREAKTHROUGH_RIPPLE: ResonancePattern(
                pattern_id="breakthrough_ripple_v1",
                resonance_type=ResonanceType.BREAKTHROUGH_RIPPLE,
                trigger_keywords=[
                    "breakthrough", "major progress", "turning point", "transformation",
                    "life-changing", "paradigm shift", "awakening", "revelation"
                ],
                emotional_indicators=[
                    "breakthrough feeling", "transformed", "liberated", "free",
                    "enlightened", "empowered", "renewed", "reborn"
                ],
                healing_phase_markers={
                    HealingPhase.ACTIVE_HEALING: ["major breakthrough", "significant progress", "turning point"],
                    HealingPhase.INTEGRATION: ["integrating breakthrough", "processing change", "embodying shift"],
                    HealingPhase.STABLE_RECOVERY: ["sustained breakthrough", "lasting change", "transformation maintained"]
                },
                propagation_conditions={
                    "breakthrough_authenticity": 0.9,
                    "transformation_depth": 0.8,
                    "inspirational_impact": 0.8
                },
                cultural_variations={
                    "spiritual_breakthrough": ["spiritual awakening", "soul transformation", "divine connection"],
                    "psychological_breakthrough": ["mental shift", "cognitive change", "emotional breakthrough"]
                },
                authenticity_markers=[
                    "real transformation", "genuine change", "authentic shift",
                    "lived breakthrough", "embodied transformation"
                ],
                sustainability_factors=[
                    "breakthrough integration", "change sustainability", "transformation support",
                    "continued growth", "ongoing evolution"
                ]
            ),
            
            ResonanceType.HOPE_MULTIPLICATION: ResonancePattern(
                pattern_id="hope_multiplication_v1",
                resonance_type=ResonanceType.HOPE_MULTIPLICATION,
                trigger_keywords=[
                    "hope", "possibility", "future bright", "things will get better",
                    "light ahead", "reason to believe", "faith in tomorrow", "optimism"
                ],
                emotional_indicators=[
                    "hopeful", "optimistic", "positive", "believing", "trusting",
                    "confident", "encouraged", "uplifted", "brightened"
                ],
                healing_phase_markers={
                    HealingPhase.CRISIS_INTERVENTION: ["finding hope", "glimpse of light", "reason to continue"],
                    HealingPhase.INITIAL_AWARENESS: ["hope for change", "possibility seen", "future imagined"],
                    HealingPhase.SEEKING_HELP: ["hope for healing", "belief in recovery", "faith in process"]
                },
                propagation_conditions={
                    "hope_authenticity": 0.8,
                    "believable_possibility": 0.7,
                    "inspirational_power": 0.9
                },
                cultural_variations={
                    "spiritual_hope": ["divine hope", "faith-based optimism", "spiritual confidence"],
                    "secular_hope": ["human resilience", "natural healing", "evidence-based optimism"]
                },
                authenticity_markers=[
                    "grounded hope", "realistic optimism", "experienced hope",
                    "earned confidence", "witnessed possibility"
                ],
                sustainability_factors=[
                    "hope maintenance", "optimism cultivation", "possibility reinforcement",
                    "confidence building", "faith nurturing"
                ]
            )
        }
    
    async def detect_healing_resonance(self, content: str, author_id: str, 
                                     context: Optional[Dict[str, Any]] = None) -> Optional[HealingWave]:
        """
        Detect healing resonance patterns in content that could create healing waves
        
        Args:
            content: Text content to analyze
            author_id: ID of the content author
            context: Additional context about the content and situation
            
        Returns:
            HealingWave if resonance detected, None otherwise
        """
        try:
            detected_resonance = None
            highest_resonance_score = 0.0
            
            content_lower = content.lower()
            
            # Analyze for each resonance type
            for resonance_type, pattern in self.resonance_patterns.items():
                resonance_score = await self._analyze_resonance_pattern(content_lower, pattern, context)
                
                if resonance_score > highest_resonance_score and resonance_score >= self.resonance_threshold:
                    highest_resonance_score = resonance_score
                    detected_resonance = resonance_type
            
            if not detected_resonance:
                return None
            
            # Get the pattern for the detected resonance
            pattern = self.resonance_patterns[detected_resonance]
            
            # Determine healing phase
            healing_phase = await self._determine_healing_phase(content_lower, pattern, context)
            
            # Calculate wave properties
            wave_strength = highest_resonance_score
            strength_level = await self._determine_strength_level(wave_strength)
            propagation_speed = await self._calculate_propagation_speed(detected_resonance, wave_strength, context)
            healing_trajectory = await self._project_healing_trajectory(detected_resonance, healing_phase, context)
            
            # Cultural and accessibility analysis
            cultural_adaptation = await self._analyze_cultural_adaptation_needs(content, pattern, context)
            accessibility_features = await self._analyze_accessibility_requirements(content, context)
            trauma_informed_safe = await self._validate_trauma_informed_safety(content, context)
            
            # Calculate sustainability and authenticity
            sustainability_score = await self._calculate_sustainability_score(content, pattern, context)
            authenticity_preservation = await self._calculate_authenticity_preservation(content, pattern)
            
            # Create healing wave
            wave = HealingWave(
                wave_id=f"healing_wave_{datetime.utcnow().isoformat()}_{id(self)}",
                origin_content=content,
                origin_author=author_id,
                resonance_type=detected_resonance,
                initial_healing_phase=healing_phase,
                wave_strength=wave_strength,
                strength_level=strength_level,
                propagation_speed=propagation_speed,
                healing_trajectory=healing_trajectory,
                affected_users=[author_id],
                secondary_waves=[],
                cultural_adaptation=cultural_adaptation,
                accessibility_features=accessibility_features,
                trauma_informed_propagation=trauma_informed_safe,
                detected_at=datetime.utcnow(),
                peak_resonance_time=None,
                wave_completion_time=None,
                total_healing_impact=0.0,
                sustainability_score=sustainability_score,
                authenticity_preservation=authenticity_preservation
            )
            
            # Track the wave
            self.active_waves[wave.wave_id] = wave
            self.waves_detected += 1
            
            # Update user healing context
            if author_id not in self.user_healing_contexts:
                self.user_healing_contexts[author_id] = await self._create_healing_context(author_id, context)
            
            # Trigger callbacks
            for callback in self.wave_detected_callbacks:
                try:
                    await callback(wave)
                except Exception as e:
                    logger.error(f"Wave detection callback failed: {str(e)}")
            
            return wave
            
        except Exception as e:
            logger.error(f"Healing resonance detection failed: {str(e)}")
            return None
    
    async def _analyze_resonance_pattern(self, content: str, pattern: ResonancePattern,
                                       context: Optional[Dict[str, Any]]) -> float:
        """Analyze content for specific resonance pattern"""
        try:
            resonance_score = 0.0
            
            # Trigger keyword analysis
            trigger_matches = sum(1 for keyword in pattern.trigger_keywords if keyword in content)
            if pattern.trigger_keywords:
                resonance_score += (trigger_matches / len(pattern.trigger_keywords)) * 0.3
            
            # Emotional indicator analysis
            emotional_matches = sum(1 for indicator in pattern.emotional_indicators if indicator in content)
            if pattern.emotional_indicators:
                resonance_score += (emotional_matches / len(pattern.emotional_indicators)) * 0.2
            
            # Healing phase marker analysis
            phase_score = 0.0
            for phase, markers in pattern.healing_phase_markers.items():
                phase_matches = sum(1 for marker in markers if marker in content)
                if markers:
                    phase_score = max(phase_score, phase_matches / len(markers))
            resonance_score += phase_score * 0.2
            
            # Authenticity marker analysis
            authenticity_matches = sum(1 for marker in pattern.authenticity_markers if marker in content)
            if pattern.authenticity_markers:
                resonance_score += (authenticity_matches / len(pattern.authenticity_markers)) * 0.2
            
            # Context-based propagation conditions
            if context:
                propagation_score = await self._evaluate_propagation_conditions(pattern, context)
                resonance_score += propagation_score * 0.1
            
            return min(1.0, resonance_score)
            
        except Exception as e:
            logger.error(f"Resonance pattern analysis failed: {str(e)}")
            return 0.0
    
    async def _evaluate_propagation_conditions(self, pattern: ResonancePattern,
                                             context: Dict[str, Any]) -> float:
        """Evaluate if conditions are right for resonance propagation"""
        try:
            propagation_score = 0.0
            conditions_met = 0
            total_conditions = len(pattern.propagation_conditions)
            
            for condition, required_threshold in pattern.propagation_conditions.items():
                if condition == "community_engagement" and context.get('community_engagement', 0) >= required_threshold:
                    conditions_met += 1
                elif condition == "shared_vulnerability" and context.get('vulnerability_sharing', 0) >= required_threshold:
                    conditions_met += 1
                elif condition == "mutual_support" and context.get('support_presence', 0) >= required_threshold:
                    conditions_met += 1
                elif condition == "inspirational_authenticity" and context.get('authenticity_score', 0) >= required_threshold:
                    conditions_met += 1
                elif condition == "hope_generation" and context.get('hope_indicators', 0) >= required_threshold:
                    conditions_met += 1
                # Add more condition evaluations as needed
            
            if total_conditions > 0:
                propagation_score = conditions_met / total_conditions
            
            return propagation_score
            
        except Exception as e:
            logger.error(f"Propagation conditions evaluation failed: {str(e)}")
            return 0.0
    
    async def _determine_healing_phase(self, content: str, pattern: ResonancePattern,
                                     context: Optional[Dict[str, Any]]) -> HealingPhase:
        """Determine the healing phase indicated by the content"""
        try:
            phase_scores = {}
            
            # Analyze content for each healing phase
            for phase, markers in pattern.healing_phase_markers.items():
                phase_matches = sum(1 for marker in markers if marker in content)
                if markers:
                    phase_scores[phase] = phase_matches / len(markers)
            
            # Also consider context clues
            if context:
                user_phase = context.get('user_healing_phase')
                if user_phase and user_phase in [phase.value for phase in HealingPhase]:
                    phase_from_context = HealingPhase(user_phase)
                    phase_scores[phase_from_context] = phase_scores.get(phase_from_context, 0) + 0.3
            
            # Crisis indicators override other phases
            crisis_indicators = ['crisis', 'emergency', 'urgent', 'suicide', 'self-harm', 'help me']
            if any(indicator in content for indicator in crisis_indicators):
                return HealingPhase.CRISIS_INTERVENTION
            
            # Return phase with highest score
            if phase_scores:
                return max(phase_scores.items(), key=lambda x: x[1])[0]
            else:
                return HealingPhase.INITIAL_AWARENESS
                
        except Exception as e:
            logger.error(f"Healing phase determination failed: {str(e)}")
            return HealingPhase.INITIAL_AWARENESS
    
    async def _determine_strength_level(self, wave_strength: float) -> ResonanceStrength:
        """Determine the strength level of the resonance"""
        try:
            if wave_strength >= 0.9:
                return ResonanceStrength.TRANSFORMATIONAL
            elif wave_strength >= 0.8:
                return ResonanceStrength.POWERFUL
            elif wave_strength >= 0.6:
                return ResonanceStrength.STRONG
            elif wave_strength >= 0.3:
                return ResonanceStrength.MODERATE
            else:
                return ResonanceStrength.SUBTLE
                
        except Exception as e:
            logger.error(f"Strength level determination failed: {str(e)}")
            return ResonanceStrength.MODERATE
    
    async def _calculate_propagation_speed(self, resonance_type: ResonanceType, 
                                         wave_strength: float, context: Optional[Dict[str, Any]]) -> float:
        """Calculate how fast the healing wave will propagate"""
        try:
            base_speed = wave_strength * 0.5
            
            # Type-specific speed modifiers
            type_speed_modifiers = {
                ResonanceType.COLLECTIVE_HEALING: 1.2,     # Spreads through community bonds
                ResonanceType.INSPIRATION_WAVE: 1.5,       # Inspiration spreads quickly
                ResonanceType.WISDOM_CASCADE: 0.8,         # Wisdom spreads more thoughtfully
                ResonanceType.SUPPORT_NETWORK: 1.0,        # Moderate spread through networks
                ResonanceType.BREAKTHROUGH_RIPPLE: 1.3,    # Breakthroughs create excitement
                ResonanceType.GRATITUDE_AMPLIFICATION: 1.1, # Gratitude spreads steadily
                ResonanceType.COURAGE_CONTAGION: 1.4,      # Courage inspires quickly
                ResonanceType.HOPE_MULTIPLICATION: 1.6,    # Hope spreads very fast
                ResonanceType.RECOVERY_MOMENTUM: 0.9,      # Recovery is steady but gradual
                ResonanceType.VULNERABILITY_SAFETY: 0.7    # Vulnerability spreads carefully
            }
            
            speed_modifier = type_speed_modifiers.get(resonance_type, 1.0)
            
            # Context-based adjustments
            if context:
                # Crisis situations propagate faster
                if context.get('crisis_context', False):
                    speed_modifier *= 1.5
                
                # High engagement communities propagate faster
                community_engagement = context.get('community_engagement', 0.5)
                speed_modifier *= (0.5 + community_engagement)
                
                # Network size affects speed
                network_size = context.get('network_size', 100)
                network_factor = min(2.0, math.log10(network_size) / 2)
                speed_modifier *= network_factor
            
            propagation_speed = base_speed * speed_modifier
            
            return min(1.0, propagation_speed)
            
        except Exception as e:
            logger.error(f"Propagation speed calculation failed: {str(e)}")
            return 0.5
    
    async def _project_healing_trajectory(self, resonance_type: ResonanceType, 
                                        healing_phase: HealingPhase, 
                                        context: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """Project the healing trajectory this wave will likely follow"""
        try:
            trajectory = {}
            
            # Base trajectory templates by resonance type
            if resonance_type == ResonanceType.COLLECTIVE_HEALING:
                trajectory = {
                    'community_bonding': 0.8,
                    'mutual_support_increase': 0.9,
                    'shared_healing_acceleration': 0.7,
                    'collective_resilience_building': 0.8,
                    'long_term_community_health': 0.9
                }
            
            elif resonance_type == ResonanceType.INSPIRATION_WAVE:
                trajectory = {
                    'motivation_increase': 0.9,
                    'action_initiation': 0.8,
                    'hope_restoration': 0.8,
                    'goal_pursuit_activation': 0.7,
                    'sustained_inspiration': 0.6
                }
            
            elif resonance_type == ResonanceType.WISDOM_CASCADE:
                trajectory = {
                    'knowledge_integration': 0.8,
                    'wisdom_application': 0.7,
                    'teaching_others': 0.6,
                    'wisdom_preservation': 0.9,
                    'cultural_knowledge_transfer': 0.8
                }
            
            elif resonance_type == ResonanceType.SUPPORT_NETWORK:
                trajectory = {
                    'network_expansion': 0.8,
                    'support_quality_improvement': 0.9,
                    'reciprocal_care_development': 0.7,
                    'network_resilience': 0.8,
                    'sustained_mutual_aid': 0.9
                }
            
            elif resonance_type == ResonanceType.BREAKTHROUGH_RIPPLE:
                trajectory = {
                    'transformation_inspiration': 0.9,
                    'breakthrough_replication': 0.7,
                    'paradigm_shift_spread': 0.6,
                    'liberation_movement': 0.8,
                    'sustained_transformation': 0.7
                }
            
            elif resonance_type == ResonanceType.HOPE_MULTIPLICATION:
                trajectory = {
                    'hope_restoration': 0.9,
                    'optimism_cultivation': 0.8,
                    'future_vision_clarity': 0.7,
                    'despair_reduction': 0.8,
                    'sustained_hopefulness': 0.7
                }
            
            else:
                # Default trajectory
                trajectory = {
                    'positive_impact': 0.7,
                    'healing_acceleration': 0.6,
                    'community_benefit': 0.6,
                    'sustained_change': 0.5
                }
            
            # Adjust trajectory based on healing phase
            phase_adjustments = {
                HealingPhase.CRISIS_INTERVENTION: {
                    'immediate_stabilization': 0.9,
                    'crisis_resolution': 0.8,
                    'safety_establishment': 0.9
                },
                HealingPhase.INITIAL_AWARENESS: {
                    'awareness_deepening': 0.8,
                    'readiness_development': 0.7,
                    'motivation_building': 0.8
                },
                HealingPhase.SEEKING_HELP: {
                    'help_connection': 0.9,
                    'resource_access': 0.8,
                    'support_engagement': 0.8
                },
                HealingPhase.ACTIVE_HEALING: {
                    'healing_acceleration': 0.9,
                    'progress_momentum': 0.8,
                    'skill_development': 0.7
                },
                HealingPhase.GIVING_BACK: {
                    'helper_empowerment': 0.9,
                    'wisdom_sharing': 0.8,
                    'community_strengthening': 0.9
                }
            }
            
            # Add phase-specific adjustments
            if healing_phase in phase_adjustments:
                trajectory.update(phase_adjustments[healing_phase])
            
            # Context-based adjustments
            if context:
                # Adjust for user vulnerability
                vulnerability = context.get('user_vulnerability', 0.5)
                safety_factor = 1.0 - (vulnerability * 0.3)
                for key in trajectory:
                    if 'safety' in key or 'stabilization' in key:
                        trajectory[key] *= (1.0 + vulnerability * 0.5)  # More focus on safety
                    else:
                        trajectory[key] *= safety_factor
                
                # Adjust for community support
                community_support = context.get('community_support', 0.5)
                for key in trajectory:
                    if 'community' in key or 'network' in key or 'collective' in key:
                        trajectory[key] *= (0.5 + community_support * 0.5)
            
            return trajectory
            
        except Exception as e:
            logger.error(f"Healing trajectory projection failed: {str(e)}")
            return {'general_healing': 0.5}
    
    async def _analyze_cultural_adaptation_needs(self, content: str, pattern: ResonancePattern,
                                               context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze cultural adaptation needs for healing wave propagation"""
        try:
            adaptations = {
                'cultural_sensitivity_required': False,
                'language_adaptations': [],
                'cultural_metaphors': [],
                'traditional_healing_integration': [],
                'family_structure_considerations': [],
                'religious_spiritual_adaptations': []
            }
            
            content_lower = content.lower()
            
            # Check for cultural variation patterns
            for culture, variations in pattern.cultural_variations.items():
                variation_matches = sum(1 for variation in variations if variation in content_lower)
                if variation_matches > 0:
                    adaptations['cultural_sensitivity_required'] = True
                    adaptations['cultural_metaphors'].append(culture)
            
            # Detect religious/spiritual content
            spiritual_markers = ['pray', 'faith', 'spiritual', 'divine', 'sacred', 'blessed']
            if any(marker in content_lower for marker in spiritual_markers):
                adaptations['religious_spiritual_adaptations'].append('interfaith_sensitivity')
                adaptations['cultural_sensitivity_required'] = True
            
            # Detect family/community structure references
            family_markers = ['family', 'community', 'tribe', 'clan', 'ancestors', 'elders']
            if any(marker in content_lower for marker in family_markers):
                adaptations['family_structure_considerations'].append('diverse_family_structures')
                adaptations['cultural_sensitivity_required'] = True
            
            # Context-based adaptations
            if context:
                user_culture = context.get('cultural_background', [])
                if user_culture:
                    adaptations['language_adaptations'].extend(user_culture)
                    adaptations['cultural_sensitivity_required'] = True
                
                # Traditional healing preferences
                if context.get('traditional_healing_preference', False):
                    adaptations['traditional_healing_integration'].append('traditional_modern_integration')
            
            return adaptations
            
        except Exception as e:
            logger.error(f"Cultural adaptation analysis failed: {str(e)}")
            return {'cultural_sensitivity_required': False}
    
    async def _analyze_accessibility_requirements(self, content: str,
                                                context: Optional[Dict[str, Any]]) -> List[str]:
        """Analyze accessibility requirements for healing wave propagation"""
        try:
            requirements = []
            
            content_lower = content.lower()
            
            # Visual content accessibility
            visual_markers = ['image', 'photo', 'visual', 'see', 'look', 'watch']
            if any(marker in content_lower for marker in visual_markers):
                requirements.extend([
                    'alt_text_descriptions',
                    'screen_reader_optimization',
                    'high_contrast_support'
                ])
            
            # Audio content accessibility
            audio_markers = ['audio', 'sound', 'music', 'listen', 'hear', 'voice']
            if any(marker in content_lower for marker in audio_markers):
                requirements.extend([
                    'captions_needed',
                    'transcripts_required',
                    'sign_language_interpretation'
                ])
            
            # Cognitive accessibility
            if len(content.split()) > 100:  # Long content
                requirements.extend([
                    'plain_language_summary',
                    'content_chunking',
                    'reading_assistance'
                ])
            
            # Emotional accessibility
            emotional_intensity_markers = ['trauma', 'trigger', 'intense', 'overwhelming', 'crisis']
            if any(marker in content_lower for marker in emotional_intensity_markers):
                requirements.extend([
                    'content_warnings',
                    'emotional_safety_measures',
                    'grounding_techniques',
                    'support_resource_links'
                ])
            
            # Context-based requirements
            if context:
                user_accessibility_needs = context.get('accessibility_needs', [])
                requirements.extend(user_accessibility_needs)
                
                # Vulnerability-based requirements
                if context.get('user_vulnerability', 0) > 0.7:
                    requirements.extend([
                        'gentle_delivery_mode',
                        'paced_content_delivery',
                        'emotional_check_ins'
                    ])
            
            # Standard accessibility requirements
            requirements.extend([
                'keyboard_navigation',
                'mobile_optimization',
                'font_size_flexibility',
                'color_blind_friendly'
            ])
            
            return list(set(requirements))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Accessibility requirements analysis failed: {str(e)}")
            return ['basic_accessibility']
    
    async def _validate_trauma_informed_safety(self, content: str,
                                             context: Optional[Dict[str, Any]]) -> bool:
        """Validate that the healing wave propagation is trauma-informed and safe"""
        try:
            # Check for potentially re-traumatizing content
            trauma_risk_markers = [
                'graphic', 'detailed trauma', 'explicit description',
                'graphic violence', 'detailed abuse', 'triggering content'
            ]
            
            content_lower = content.lower()
            has_trauma_risk = any(marker in content_lower for marker in trauma_risk_markers)
            
            if has_trauma_risk:
                return False
            
            # Check for healing-supportive elements
            healing_supportive_markers = [
                'gentle', 'safe', 'supported', 'at your pace',
                'no pressure', 'choice', 'consent', 'boundaries'
            ]
            
            has_healing_support = any(marker in content_lower for marker in healing_supportive_markers)
            
            # Context-based safety validation
            if context:
                # Check user trauma history
                trauma_history = context.get('trauma_history_indicators', [])
                if trauma_history and not has_healing_support:
                    return False
                
                # Check current vulnerability level
                vulnerability = context.get('user_vulnerability', 0)
                if vulnerability > 0.8 and not has_healing_support:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Trauma-informed safety validation failed: {str(e)}")
            return True  # Default to safe
    
    async def _calculate_sustainability_score(self, content: str, pattern: ResonancePattern,
                                            context: Optional[Dict[str, Any]]) -> float:
        """Calculate how sustainable the healing wave will be"""
        try:
            sustainability_score = 0.5  # Base score
            
            # Check for sustainability factors
            sustainability_matches = sum(
                1 for factor in pattern.sustainability_factors 
                if factor.replace('_', ' ') in content.lower()
            )
            
            if pattern.sustainability_factors:
                factor_score = sustainability_matches / len(pattern.sustainability_factors)
                sustainability_score += factor_score * 0.3
            
            # Structural sustainability indicators
            structure_indicators = [
                'ongoing', 'continued', 'sustainable', 'long-term',
                'building', 'growing', 'strengthening', 'developing'
            ]
            
            structure_matches = sum(1 for indicator in structure_indicators if indicator in content.lower())
            sustainability_score += (structure_matches / len(structure_indicators)) * 0.2
            
            # Community-based sustainability
            community_indicators = [
                'community', 'together', 'collective', 'shared',
                'network', 'support system', 'infrastructure'
            ]
            
            community_matches = sum(1 for indicator in community_indicators if indicator in content.lower())
            sustainability_score += (community_matches / len(community_indicators)) * 0.2
            
            # Context-based adjustments
            if context:
                # Strong support networks increase sustainability
                network_strength = context.get('support_network_strength', 0.5)
                sustainability_score += network_strength * 0.2
                
                # Community engagement increases sustainability
                community_engagement = context.get('community_engagement', 0.5)
                sustainability_score += community_engagement * 0.1
                
                # Resource availability affects sustainability
                resource_availability = context.get('resource_availability', 0.5)
                sustainability_score += resource_availability * 0.1
            
            return min(1.0, sustainability_score)
            
        except Exception as e:
            logger.error(f"Sustainability score calculation failed: {str(e)}")
            return 0.5
    
    async def _calculate_authenticity_preservation(self, content: str, pattern: ResonancePattern) -> float:
        """Calculate how well authenticity will be preserved during wave propagation"""
        try:
            authenticity_score = 0.5  # Base score
            
            # Check for authenticity markers
            authenticity_matches = sum(
                1 for marker in pattern.authenticity_markers
                if marker.replace('_', ' ') in content.lower()
            )
            
            if pattern.authenticity_markers:
                marker_score = authenticity_matches / len(pattern.authenticity_markers)
                authenticity_score += marker_score * 0.4
            
            # Personal experience indicators
            personal_indicators = [
                'I experienced', 'my journey', 'what I learned', 'in my case',
                'for me', 'I found', 'my experience', 'personally'
            ]
            
            personal_matches = sum(1 for indicator in personal_indicators if indicator in content.lower())
            authenticity_score += (personal_matches / len(personal_indicators)) * 0.3
            
            # Vulnerability and honesty indicators
            vulnerability_indicators = [
                'honestly', 'truthfully', 'vulnerable', 'real', 'genuine',
                'authentic', 'raw', 'unfiltered', 'honest'
            ]
            
            vulnerability_matches = sum(1 for indicator in vulnerability_indicators if indicator in content.lower())
            authenticity_score += (vulnerability_matches / len(vulnerability_indicators)) * 0.2
            
            # Specificity and detail indicators (authentic stories have details)
            specificity_indicators = [
                'specifically', 'exactly', 'particular', 'precise',
                'detail', 'moment when', 'time that'
            ]
            
            specificity_matches = sum(1 for indicator in specificity_indicators if indicator in content.lower())
            authenticity_score += (specificity_matches / len(specificity_indicators)) * 0.1
            
            return min(1.0, authenticity_score)
            
        except Exception as e:
            logger.error(f"Authenticity preservation calculation failed: {str(e)}")
            return 0.5
    
    async def _create_healing_context(self, user_id: str, context: Optional[Dict[str, Any]]) -> HealingContext:
        """Create healing context for a user"""
        try:
            # Extract information from provided context
            if context:
                healing_phase_str = context.get('user_healing_phase', 'initial_awareness')
                healing_phase = HealingPhase(healing_phase_str) if healing_phase_str in [p.value for p in HealingPhase] else HealingPhase.INITIAL_AWARENESS
                
                vulnerability_level = context.get('user_vulnerability', 0.5)
                support_network_size = context.get('support_network_size', 10)
                cultural_background = context.get('cultural_background', [])
                accessibility_needs = context.get('accessibility_needs', [])
                trauma_history = context.get('trauma_history_indicators', [])
                healing_goals = context.get('healing_goals', [])
                support_preferences = context.get('preferred_support_types', [])
                safety_requirements = context.get('safety_requirements', [])
            else:
                # Default values
                healing_phase = HealingPhase.INITIAL_AWARENESS
                vulnerability_level = 0.5
                support_network_size = 10
                cultural_background = []
                accessibility_needs = []
                trauma_history = []
                healing_goals = []
                support_preferences = []
                safety_requirements = []
            
            return HealingContext(
                user_id=user_id,
                current_healing_phase=healing_phase,
                vulnerability_level=vulnerability_level,
                support_network_size=support_network_size,
                cultural_background=cultural_background,
                accessibility_needs=accessibility_needs,
                trauma_history_indicators=trauma_history,
                healing_goals=healing_goals,
                preferred_support_types=support_preferences,
                safety_requirements=safety_requirements
            )
            
        except Exception as e:
            logger.error(f"Healing context creation failed: {str(e)}")
            # Return minimal default context
            return HealingContext(
                user_id=user_id,
                current_healing_phase=HealingPhase.INITIAL_AWARENESS,
                vulnerability_level=0.5,
                support_network_size=10,
                cultural_background=[],
                accessibility_needs=[],
                trauma_history_indicators=[],
                healing_goals=[],
                preferred_support_types=[],
                safety_requirements=[]
            )
    
    async def track_wave_propagation(self, wave_id: str, 
                                   propagation_data: Dict[str, Any]) -> bool:
        """Track the propagation of a healing wave"""
        try:
            if wave_id not in self.active_waves:
                logger.warning(f"Attempting to track unknown wave: {wave_id}")
                return False
            
            wave = self.active_waves[wave_id]
            
            # Update affected users
            new_users = propagation_data.get('new_affected_users', [])
            wave.affected_users.extend(new_users)
            self.users_reached_by_waves.update(new_users)
            
            # Track secondary waves
            secondary_waves = propagation_data.get('secondary_waves_triggered', [])
            wave.secondary_waves.extend(secondary_waves)
            self.secondary_waves_triggered += len(secondary_waves)
            
            # Update healing impact
            additional_impact = propagation_data.get('healing_impact', 0.0)
            wave.total_healing_impact += additional_impact
            self.total_healing_impact += additional_impact
            
            # Check for peak resonance
            current_resonance = propagation_data.get('current_resonance_strength', 0.0)
            if current_resonance > wave.wave_strength:
                wave.wave_strength = current_resonance
                wave.peak_resonance_time = datetime.utcnow()
            
            # Update resonance network connections
            for user in new_users:
                self.resonance_network[wave.origin_author].add(user)
                self.resonance_network[user].add(wave.origin_author)
            
            return True
            
        except Exception as e:
            logger.error(f"Wave propagation tracking failed: {str(e)}")
            return False
    
    async def complete_wave(self, wave_id: str, completion_data: Dict[str, Any]) -> bool:
        """Mark a healing wave as completed and analyze its impact"""
        try:
            if wave_id not in self.active_waves:
                logger.warning(f"Attempting to complete unknown wave: {wave_id}")
                return False
            
            wave = self.active_waves[wave_id]
            
            # Set completion time
            wave.wave_completion_time = datetime.utcnow()
            
            # Update final metrics
            wave.total_healing_impact = completion_data.get('total_healing_impact', wave.total_healing_impact)
            final_user_count = completion_data.get('total_users_affected', len(wave.affected_users))
            
            # Calculate final impact metrics
            if final_user_count > 0:
                average_impact_per_user = wave.total_healing_impact / final_user_count
                logger.info(f"Wave {wave_id} completed: {final_user_count} users, {average_impact_per_user:.3f} avg impact")
            
            # Move to completed waves
            self.completed_waves.append(wave)
            del self.active_waves[wave_id]
            
            # Trigger completion callbacks
            for callback in self.wave_completion_callbacks:
                try:
                    await callback(wave)
                except Exception as e:
                    logger.error(f"Wave completion callback failed: {str(e)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Wave completion failed: {str(e)}")
            return False
    
    # Callback management
    def add_wave_detected_callback(self, callback: Callable):
        """Add callback for wave detection events"""
        self.wave_detected_callbacks.append(callback)
    
    def add_wave_completion_callback(self, callback: Callable):
        """Add callback for wave completion events"""
        self.wave_completion_callbacks.append(callback)
    
    def add_healing_impact_callback(self, callback: Callable):
        """Add callback for healing impact measurement events"""
        self.healing_impact_callbacks.append(callback)
    
    # Analytics and reporting
    def get_resonance_analytics(self) -> Dict[str, Any]:
        """Get analytics on healing resonance detection and impact"""
        try:
            total_waves = len(self.completed_waves) + len(self.active_waves)
            
            if total_waves == 0:
                return {
                    'total_waves_detected': 0,
                    'active_waves': 0,
                    'completed_waves': 0,
                    'total_healing_impact': 0.0,
                    'users_reached': 0,
                    'average_wave_impact': 0.0,
                    'resonance_type_distribution': {},
                    'healing_phase_distribution': {},
                    'cultural_adaptation_stats': {},
                    'sustainability_average': 0.0
                }
            
            # Calculate metrics
            completed_impact = sum(wave.total_healing_impact for wave in self.completed_waves)
            active_impact = sum(wave.total_healing_impact for wave in self.active_waves.values())
            total_impact = completed_impact + active_impact
            
            average_impact = total_impact / total_waves if total_waves > 0 else 0.0
            
            # Resonance type distribution
            all_waves = self.completed_waves + list(self.active_waves.values())
            resonance_types = [wave.resonance_type.value for wave in all_waves]
            resonance_distribution = {
                rt.value: resonance_types.count(rt.value) for rt in ResonanceType
            }
            
            # Healing phase distribution
            healing_phases = [wave.initial_healing_phase.value for wave in all_waves]
            phase_distribution = {
                hp.value: healing_phases.count(hp.value) for hp in HealingPhase
            }
            
            # Cultural adaptation statistics
            cultural_adaptations = sum(
                1 for wave in all_waves 
                if wave.cultural_adaptation.get('cultural_sensitivity_required', False)
            )
            
            # Sustainability average
            sustainability_scores = [wave.sustainability_score for wave in all_waves]
            average_sustainability = statistics.mean(sustainability_scores) if sustainability_scores else 0.0
            
            return {
                'total_waves_detected': total_waves,
                'active_waves': len(self.active_waves),
                'completed_waves': len(self.completed_waves),
                'total_healing_impact': total_impact,
                'users_reached': len(self.users_reached_by_waves),
                'secondary_waves_triggered': self.secondary_waves_triggered,
                'average_wave_impact': average_impact,
                'resonance_type_distribution': resonance_distribution,
                'healing_phase_distribution': phase_distribution,
                'cultural_adaptations_made': cultural_adaptations,
                'sustainability_average': average_sustainability,
                'resonance_network_size': len(self.resonance_network),
                'resonance_connections': sum(len(connections) for connections in self.resonance_network.values())
            }
            
        except Exception as e:
            logger.error(f"Resonance analytics generation failed: {str(e)}")
            return {}
    
    def get_healing_network_analysis(self) -> Dict[str, Any]:
        """Analyze the healing resonance network"""
        try:
            if not self.resonance_network:
                return {
                    'network_nodes': 0,
                    'network_connections': 0,
                    'network_density': 0.0,
                    'most_influential_healers': [],
                    'healing_clusters': [],
                    'average_connections': 0.0
                }
            
            # Basic network statistics
            total_nodes = len(self.resonance_network)
            total_connections = sum(len(connections) for connections in self.resonance_network.values())
            average_connections = total_connections / total_nodes if total_nodes > 0 else 0.0
            
            # Network density
            max_connections = total_nodes * (total_nodes - 1)
            network_density = total_connections / max_connections if max_connections > 0 else 0.0
            
            # Most influential healers (users with most connections)
            user_influence = [
                (user, len(connections)) 
                for user, connections in self.resonance_network.items()
            ]
            most_influential = sorted(user_influence, key=lambda x: x[1], reverse=True)[:10]
            
            return {
                'network_nodes': total_nodes,
                'network_connections': total_connections,
                'network_density': network_density,
                'most_influential_healers': most_influential,
                'average_connections': average_connections,
                'resonance_strength': 'strong' if network_density > 0.1 else 'moderate' if network_density > 0.05 else 'emerging'
            }
            
        except Exception as e:
            logger.error(f"Healing network analysis failed: {str(e)}")
            return {}