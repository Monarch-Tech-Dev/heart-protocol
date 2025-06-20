"""
Connection Catalyst

System for facilitating meaningful human connections and catalyzing the formation
of healing relationships, support networks, and caring communities.
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

logger = logging.getLogger(__name__)


class ConnectionType(Enum):
    """Types of connections the catalyst can facilitate"""
    HEALING_PARTNERSHIP = "healing_partnership"           # Mutual healing support
    MENTOR_MENTEE = "mentor_mentee"                       # Wisdom sharing relationship
    CRISIS_SUPPORT_BUDDY = "crisis_support_buddy"         # Crisis support partnership
    ACCOUNTABILITY_PARTNER = "accountability_partner"     # Mutual accountability
    SHARED_JOURNEY = "shared_journey"                     # Similar healing paths
    COMPLEMENTARY_STRENGTHS = "complementary_strengths"   # Different but complementary skills
    COMMUNITY_BUILDING = "community_building"             # Building larger communities
    CULTURAL_BRIDGE = "cultural_bridge"                   # Cross-cultural connections
    INTERGENERATIONAL = "intergenerational"               # Different age groups
    SKILL_EXCHANGE = "skill_exchange"                     # Mutual skill sharing
    EMOTIONAL_SUPPORT = "emotional_support"               # Emotional support network
    PRACTICAL_ASSISTANCE = "practical_assistance"         # Practical help network


class CatalystAction(Enum):
    """Actions the catalyst can take to facilitate connections"""
    GENTLE_INTRODUCTION = "gentle_introduction"           # Soft introduction with context
    SHARED_INTEREST_HIGHLIGHT = "shared_interest_highlight" # Highlight common interests
    MUTUAL_NEED_MATCHING = "mutual_need_matching"         # Match complementary needs
    COMMUNITY_INVITATION = "community_invitation"         # Invite to relevant community
    RESOURCE_SHARING = "resource_sharing"                # Share relevant resources
    COLLABORATION_SUGGESTION = "collaboration_suggestion" # Suggest collaboration
    SUPPORT_CIRCLE_FORMATION = "support_circle_formation" # Form support circles
    HEALING_CIRCLE_INVITATION = "healing_circle_invitation" # Invite to healing circles
    MENTORSHIP_FACILITATION = "mentorship_facilitation"   # Facilitate mentorship
    PEER_CONNECTION = "peer_connection"                   # Connect peers


class ConnectionStrength(Enum):
    """Strength levels of potential connections"""
    WEAK = "weak"                                         # 0.0-0.3 - Surface level
    MODERATE = "moderate"                                 # 0.3-0.6 - Some compatibility
    STRONG = "strong"                                     # 0.6-0.8 - Good compatibility
    VERY_STRONG = "very_strong"                          # 0.8-0.9 - High compatibility
    TRANSFORMATIONAL = "transformational"                # 0.9-1.0 - Life-changing potential


class ConnectionStage(Enum):
    """Stages of connection development"""
    POTENTIAL_IDENTIFIED = "potential_identified"         # Connection possibility spotted
    INTRODUCTION_MADE = "introduction_made"               # Initial introduction completed
    INITIAL_INTERACTION = "initial_interaction"           # First meaningful interaction
    DEVELOPING_RAPPORT = "developing_rapport"             # Building relationship
    ESTABLISHED_CONNECTION = "established_connection"     # Stable, ongoing connection
    DEEP_BOND = "deep_bond"                              # Deep, meaningful relationship
    MUTUAL_TRANSFORMATION = "mutual_transformation"       # Transformative relationship


@dataclass
class ConnectionProfile:
    """Profile for understanding a user's connection needs and preferences"""
    user_id: str
    healing_phase: str
    vulnerability_level: float
    support_seeking: List[str]
    support_offering: List[str]
    communication_style: str
    cultural_background: List[str]
    accessibility_needs: List[str]
    trauma_considerations: List[str]
    connection_preferences: Dict[str, Any]
    boundaries: Dict[str, Any]
    availability: Dict[str, Any]
    past_connection_success: List[str]
    connection_goals: List[str]
    safety_requirements: List[str]


@dataclass
class ConnectionOpportunity:
    """An identified opportunity for meaningful connection"""
    opportunity_id: str
    user_1_id: str
    user_2_id: str
    connection_type: ConnectionType
    connection_strength: float
    strength_level: ConnectionStrength
    compatibility_factors: Dict[str, float]
    mutual_benefits: List[str]
    potential_challenges: List[str]
    recommended_action: CatalystAction
    introduction_approach: str
    safety_considerations: List[str]
    cultural_considerations: List[str]
    timing_recommendation: str
    success_probability: float
    healing_potential: float
    identified_at: datetime
    expiration_time: Optional[datetime]
    privacy_safe: bool
    consent_required: bool


@dataclass
class ConnectionAttempt:
    """Record of an attempt to facilitate a connection"""
    attempt_id: str
    opportunity_id: str
    action_taken: CatalystAction
    approach_used: str
    user_responses: Dict[str, str]
    success_indicators: List[str]
    challenges_encountered: List[str]
    cultural_adaptations_made: List[str]
    accessibility_accommodations: List[str]
    safety_measures_applied: List[str]
    attempt_timestamp: datetime
    outcome: str
    lessons_learned: List[str]
    follow_up_needed: bool


@dataclass
class EstablishedConnection:
    """Record of a successfully established connection"""
    connection_id: str
    user_1_id: str
    user_2_id: str
    connection_type: ConnectionType
    connection_stage: ConnectionStage
    relationship_quality: float
    mutual_satisfaction: float
    healing_impact: Dict[str, float]
    support_exchanges: List[Dict[str, Any]]
    milestones_achieved: List[str]
    challenges_overcome: List[str]
    growth_observed: List[str]
    established_at: datetime
    last_interaction: datetime
    relationship_trajectory: Dict[str, float]
    sustainability_indicators: List[str]


class ConnectionCatalyst:
    """
    System for identifying, facilitating, and nurturing meaningful human connections
    that support healing, growth, and community wellbeing.
    
    Core Principles:
    - Meaningful connections accelerate healing and growth
    - Consent and safety are paramount in all connections
    - Cultural sensitivity guides all matching and introductions
    - Trauma-informed approaches prevent re-traumatization
    - Authentic connections are prioritized over superficial ones
    - Vulnerable users receive extra protection and gentle approaches
    - Community resilience is built through interconnected support
    - Diversity and inclusion strengthen all connections
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Catalyst settings
        self.connection_threshold = self.config.get('connection_threshold', 0.6)
        self.safety_priority_level = self.config.get('safety_priority_level', 'maximum')
        self.consent_required_always = self.config.get('consent_required_always', True)
        self.cultural_sensitivity_mode = self.config.get('cultural_sensitivity_mode', 'high')
        
        # Connection tracking
        self.user_profiles: Dict[str, ConnectionProfile] = {}
        self.identified_opportunities: Dict[str, ConnectionOpportunity] = {}
        self.connection_attempts: List[ConnectionAttempt] = []
        self.established_connections: Dict[str, EstablishedConnection] = {}
        self.community_networks: Dict[str, Set[str]] = defaultdict(set)
        
        # Performance metrics
        self.opportunities_identified = 0
        self.connections_attempted = 0
        self.connections_established = 0
        self.total_healing_impact = 0.0
        self.community_networks_formed = 0
        
        # Matching algorithms
        self.compatibility_algorithms = self._initialize_compatibility_algorithms()
        
        # Safety and cultural systems
        self.safety_validator = None            # Would integrate with safety system
        self.cultural_matcher = None           # Would integrate with cultural sensitivity
        self.trauma_safety_checker = None     # Would integrate with trauma-informed care
        self.accessibility_adapter = None     # Would integrate with accessibility system
        
        # Callbacks
        self.opportunity_identified_callbacks: List[Callable] = []
        self.connection_attempted_callbacks: List[Callable] = []
        self.connection_established_callbacks: List[Callable] = []
        self.healing_impact_callbacks: List[Callable] = []
    
    def _initialize_compatibility_algorithms(self) -> Dict[str, Callable]:
        """Initialize compatibility algorithms for different connection types"""
        return {
            'healing_phase_compatibility': self._calculate_healing_phase_compatibility,
            'support_needs_matching': self._calculate_support_needs_matching,
            'cultural_compatibility': self._calculate_cultural_compatibility,
            'communication_style_match': self._calculate_communication_style_match,
            'vulnerability_level_balance': self._calculate_vulnerability_balance,
            'shared_interests_overlap': self._calculate_shared_interests_overlap,
            'complementary_strengths': self._calculate_complementary_strengths,
            'availability_alignment': self._calculate_availability_alignment,
            'safety_compatibility': self._calculate_safety_compatibility,
            'growth_trajectory_alignment': self._calculate_growth_trajectory_alignment
        }
    
    async def create_connection_profile(self, user_id: str, 
                                      profile_data: Dict[str, Any]) -> ConnectionProfile:
        """Create or update a connection profile for a user"""
        try:
            profile = ConnectionProfile(
                user_id=user_id,
                healing_phase=profile_data.get('healing_phase', 'initial_awareness'),
                vulnerability_level=profile_data.get('vulnerability_level', 0.5),
                support_seeking=profile_data.get('support_seeking', []),
                support_offering=profile_data.get('support_offering', []),
                communication_style=profile_data.get('communication_style', 'balanced'),
                cultural_background=profile_data.get('cultural_background', []),
                accessibility_needs=profile_data.get('accessibility_needs', []),
                trauma_considerations=profile_data.get('trauma_considerations', []),
                connection_preferences=profile_data.get('connection_preferences', {}),
                boundaries=profile_data.get('boundaries', {}),
                availability=profile_data.get('availability', {}),
                past_connection_success=profile_data.get('past_connection_success', []),
                connection_goals=profile_data.get('connection_goals', []),
                safety_requirements=profile_data.get('safety_requirements', [])
            )
            
            self.user_profiles[user_id] = profile
            
            # After creating/updating profile, look for new connection opportunities
            await self._identify_new_opportunities_for_user(user_id)
            
            return profile
            
        except Exception as e:
            logger.error(f"Connection profile creation failed: {str(e)}")
            raise
    
    async def identify_connection_opportunities(self, user_id: str) -> List[ConnectionOpportunity]:
        """Identify potential connection opportunities for a specific user"""
        try:
            if user_id not in self.user_profiles:
                logger.warning(f"No profile found for user {user_id}")
                return []
            
            user_profile = self.user_profiles[user_id]
            opportunities = []
            
            # Compare with all other users
            for other_user_id, other_profile in self.user_profiles.items():
                if other_user_id == user_id:
                    continue
                
                # Check if connection already exists
                existing_connection = self._find_existing_connection(user_id, other_user_id)
                if existing_connection:
                    continue
                
                # Calculate compatibility
                compatibility = await self._calculate_overall_compatibility(user_profile, other_profile)
                
                if compatibility['total_score'] >= self.connection_threshold:
                    # Determine connection type and action
                    connection_type = await self._determine_optimal_connection_type(
                        user_profile, other_profile, compatibility
                    )
                    
                    recommended_action = await self._determine_recommended_action(
                        connection_type, user_profile, other_profile
                    )
                    
                    # Create opportunity
                    opportunity = await self._create_connection_opportunity(
                        user_id, other_user_id, connection_type, compatibility, recommended_action
                    )
                    
                    if opportunity:
                        opportunities.append(opportunity)
                        self.identified_opportunities[opportunity.opportunity_id] = opportunity
                        self.opportunities_identified += 1
            
            # Sort by strength and healing potential
            opportunities.sort(
                key=lambda x: (x.connection_strength, x.healing_potential), 
                reverse=True
            )
            
            # Trigger callbacks
            for opportunity in opportunities:
                for callback in self.opportunity_identified_callbacks:
                    try:
                        await callback(opportunity)
                    except Exception as e:
                        logger.error(f"Opportunity identification callback failed: {str(e)}")
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Connection opportunity identification failed: {str(e)}")
            return []
    
    async def _identify_new_opportunities_for_user(self, user_id: str):
        """Internal method to identify new opportunities when a user profile is updated"""
        try:
            opportunities = await self.identify_connection_opportunities(user_id)
            
            # Also check if this user could be a good match for existing users
            for other_user_id in self.user_profiles:
                if other_user_id != user_id:
                    other_opportunities = await self.identify_connection_opportunities(other_user_id)
                    # Filter for opportunities involving the new user
                    relevant_opportunities = [
                        opp for opp in other_opportunities 
                        if opp.user_1_id == user_id or opp.user_2_id == user_id
                    ]
                    opportunities.extend(relevant_opportunities)
            
            logger.info(f"Identified {len(opportunities)} new connection opportunities involving user {user_id}")
            
        except Exception as e:
            logger.error(f"New opportunity identification failed: {str(e)}")
    
    async def _calculate_overall_compatibility(self, profile1: ConnectionProfile, 
                                             profile2: ConnectionProfile) -> Dict[str, Any]:
        """Calculate overall compatibility between two user profiles"""
        try:
            compatibility_scores = {}
            
            # Run all compatibility algorithms
            for algorithm_name, algorithm_func in self.compatibility_algorithms.items():
                try:
                    score = await algorithm_func(profile1, profile2)
                    compatibility_scores[algorithm_name] = score
                except Exception as e:
                    logger.error(f"Compatibility algorithm {algorithm_name} failed: {str(e)}")
                    compatibility_scores[algorithm_name] = 0.0
            
            # Calculate weighted total score
            weights = {
                'safety_compatibility': 0.25,          # Safety is highest priority
                'support_needs_matching': 0.20,        # Core purpose compatibility
                'healing_phase_compatibility': 0.15,   # Healing journey alignment
                'cultural_compatibility': 0.10,        # Cultural understanding
                'communication_style_match': 0.10,     # Communication effectiveness
                'vulnerability_level_balance': 0.05,   # Emotional safety balance
                'shared_interests_overlap': 0.05,      # Common ground
                'complementary_strengths': 0.05,       # Mutual growth potential
                'availability_alignment': 0.03,        # Practical compatibility
                'growth_trajectory_alignment': 0.02    # Long-term alignment
            }
            
            total_score = 0.0
            for algorithm, score in compatibility_scores.items():
                weight = weights.get(algorithm, 0.01)
                total_score += score * weight
            
            return {
                'total_score': min(1.0, total_score),
                'individual_scores': compatibility_scores,
                'weights_applied': weights
            }
            
        except Exception as e:
            logger.error(f"Overall compatibility calculation failed: {str(e)}")
            return {'total_score': 0.0, 'individual_scores': {}, 'weights_applied': {}}
    
    async def _calculate_healing_phase_compatibility(self, profile1: ConnectionProfile,
                                                   profile2: ConnectionProfile) -> float:
        """Calculate compatibility based on healing phases"""
        try:
            # Define phase compatibility matrix
            phase_compatibility = {
                'crisis_intervention': {
                    'crisis_intervention': 0.9,    # Crisis support together
                    'stable_recovery': 0.8,        # Stable person can help
                    'giving_back': 0.9,           # Helper can provide support
                    'wisdom_sharing': 0.7          # Wisdom can help in crisis
                },
                'initial_awareness': {
                    'initial_awareness': 0.8,      # Mutual discovery
                    'seeking_help': 0.7,          # Slightly ahead can guide
                    'early_recovery': 0.6,        # Experience can encourage
                    'giving_back': 0.8             # Helper can guide
                },
                'seeking_help': {
                    'seeking_help': 0.9,          # Mutual support in seeking
                    'early_recovery': 0.8,        # Experience can help
                    'active_healing': 0.7,        # Active healer can guide
                    'giving_back': 0.9             # Helper can provide resources
                },
                'early_recovery': {
                    'early_recovery': 0.9,        # Mutual early support
                    'active_healing': 0.8,        # More experienced can guide
                    'integration': 0.7,           # Integration wisdom helpful
                    'giving_back': 0.8             # Helper can encourage
                },
                'active_healing': {
                    'active_healing': 0.9,        # Mutual active work
                    'integration': 0.8,           # Integration experience helpful
                    'stable_recovery': 0.7,       # Stability can inspire
                    'giving_back': 0.8             # Helper can support process
                },
                'integration': {
                    'integration': 0.9,           # Mutual integration work
                    'stable_recovery': 0.8,       # Stability experience
                    'giving_back': 0.7,           # Helper perspective
                    'wisdom_sharing': 0.8          # Wisdom exchange
                },
                'stable_recovery': {
                    'stable_recovery': 0.8,       # Mutual stability
                    'giving_back': 0.9,           # Natural progression
                    'wisdom_sharing': 0.9,        # Wisdom development
                    'crisis_intervention': 0.8     # Can help those in crisis
                },
                'giving_back': {
                    'giving_back': 0.9,           # Mutual service
                    'wisdom_sharing': 0.9,        # Natural combination
                    'crisis_intervention': 0.9,    # Crisis help
                    'seeking_help': 0.9            # Help seekers
                },
                'wisdom_sharing': {
                    'wisdom_sharing': 0.9,        # Mutual wisdom exchange
                    'giving_back': 0.9,           # Complementary service
                    'integration': 0.8,           # Integration guidance
                    'stable_recovery': 0.8         # Stability wisdom
                }
            }
            
            phase1 = profile1.healing_phase
            phase2 = profile2.healing_phase
            
            # Get compatibility score from matrix
            if phase1 in phase_compatibility and phase2 in phase_compatibility[phase1]:
                return phase_compatibility[phase1][phase2]
            elif phase2 in phase_compatibility and phase1 in phase_compatibility[phase2]:
                return phase_compatibility[phase2][phase1]
            else:
                return 0.5  # Default moderate compatibility
                
        except Exception as e:
            logger.error(f"Healing phase compatibility calculation failed: {str(e)}")
            return 0.5
    
    async def _calculate_support_needs_matching(self, profile1: ConnectionProfile,
                                              profile2: ConnectionProfile) -> float:
        """Calculate how well users' support needs and offerings match"""
        try:
            # Calculate bidirectional matching
            user1_needs_met = 0
            user1_total_needs = len(profile1.support_seeking)
            
            if user1_total_needs > 0:
                for need in profile1.support_seeking:
                    if need in profile2.support_offering:
                        user1_needs_met += 1
                user1_satisfaction = user1_needs_met / user1_total_needs
            else:
                user1_satisfaction = 0.5  # No specific needs = moderate satisfaction
            
            user2_needs_met = 0
            user2_total_needs = len(profile2.support_seeking)
            
            if user2_total_needs > 0:
                for need in profile2.support_seeking:
                    if need in profile1.support_offering:
                        user2_needs_met += 1
                user2_satisfaction = user2_needs_met / user2_total_needs
            else:
                user2_satisfaction = 0.5  # No specific needs = moderate satisfaction
            
            # Mutual satisfaction is average of both directions
            mutual_satisfaction = (user1_satisfaction + user2_satisfaction) / 2
            
            # Bonus for complementary offerings (both can give different things)
            complementary_offerings = set(profile1.support_offering) - set(profile2.support_offering)
            complementary_bonus = min(0.2, len(complementary_offerings) * 0.05)
            
            return min(1.0, mutual_satisfaction + complementary_bonus)
            
        except Exception as e:
            logger.error(f"Support needs matching calculation failed: {str(e)}")
            return 0.5
    
    async def _calculate_cultural_compatibility(self, profile1: ConnectionProfile,
                                              profile2: ConnectionProfile) -> float:
        """Calculate cultural compatibility between users"""
        try:
            # Shared cultural background
            shared_cultures = set(profile1.cultural_background) & set(profile2.cultural_background)
            total_cultures = set(profile1.cultural_background) | set(profile2.cultural_background)
            
            if len(total_cultures) == 0:
                cultural_overlap = 0.5  # No cultural info = neutral
            else:
                cultural_overlap = len(shared_cultures) / len(total_cultures)
            
            # Cultural bridge bonus (different cultures can be enriching)
            different_cultures = set(profile1.cultural_background) - set(profile2.cultural_background)
            bridge_bonus = min(0.3, len(different_cultures) * 0.1) if len(different_cultures) <= 3 else 0
            
            # Cultural learning opportunity
            learning_opportunity = 0.0
            if len(profile1.cultural_background) > 0 and len(profile2.cultural_background) > 0:
                if not shared_cultures:  # Completely different cultures
                    learning_opportunity = 0.2
            
            cultural_compatibility = cultural_overlap + bridge_bonus + learning_opportunity
            
            return min(1.0, cultural_compatibility)
            
        except Exception as e:
            logger.error(f"Cultural compatibility calculation failed: {str(e)}")
            return 0.5
    
    async def _calculate_communication_style_match(self, profile1: ConnectionProfile,
                                                 profile2: ConnectionProfile) -> float:
        """Calculate communication style compatibility"""
        try:
            # Define communication style compatibility matrix
            style_compatibility = {
                'direct': {'direct': 0.9, 'balanced': 0.7, 'gentle': 0.5, 'empathetic': 0.6},
                'balanced': {'direct': 0.7, 'balanced': 0.9, 'gentle': 0.8, 'empathetic': 0.8},
                'gentle': {'direct': 0.5, 'balanced': 0.8, 'gentle': 0.9, 'empathetic': 0.9},
                'empathetic': {'direct': 0.6, 'balanced': 0.8, 'gentle': 0.9, 'empathetic': 0.9}
            }
            
            style1 = profile1.communication_style
            style2 = profile2.communication_style
            
            if style1 in style_compatibility and style2 in style_compatibility[style1]:
                return style_compatibility[style1][style2]
            else:
                return 0.7  # Default good compatibility
                
        except Exception as e:
            logger.error(f"Communication style compatibility calculation failed: {str(e)}")
            return 0.7
    
    async def _calculate_vulnerability_balance(self, profile1: ConnectionProfile,
                                             profile2: ConnectionProfile) -> float:
        """Calculate balance of vulnerability levels for emotional safety"""
        try:
            vuln1 = profile1.vulnerability_level
            vuln2 = profile2.vulnerability_level
            
            # Ideal scenarios
            if 0.3 <= vuln1 <= 0.7 and 0.3 <= vuln2 <= 0.7:
                # Both moderate vulnerability - good balance
                return 0.9
            
            elif (vuln1 > 0.8 and 0.4 <= vuln2 <= 0.7) or (vuln2 > 0.8 and 0.4 <= vuln1 <= 0.7):
                # One high vulnerability with one stable - can work well
                return 0.8
            
            elif vuln1 > 0.8 and vuln2 > 0.8:
                # Both highly vulnerable - might overwhelm each other
                return 0.4
            
            elif vuln1 < 0.3 and vuln2 < 0.3:
                # Both low vulnerability - might lack emotional depth
                return 0.6
            
            else:
                # Other combinations
                difference = abs(vuln1 - vuln2)
                if difference > 0.5:
                    return 0.5  # Large difference might be challenging
                else:
                    return 0.7  # Moderate difference is manageable
                    
        except Exception as e:
            logger.error(f"Vulnerability balance calculation failed: {str(e)}")
            return 0.5
    
    async def _calculate_shared_interests_overlap(self, profile1: ConnectionProfile,
                                                profile2: ConnectionProfile) -> float:
        """Calculate shared interests and common ground"""
        try:
            # This would integrate with a more detailed interest/hobby system
            # For now, use connection goals as proxy for interests
            
            goals1 = set(profile1.connection_goals)
            goals2 = set(profile2.connection_goals)
            
            shared_goals = goals1 & goals2
            total_goals = goals1 | goals2
            
            if len(total_goals) == 0:
                return 0.5  # No goals specified = neutral
            
            overlap_ratio = len(shared_goals) / len(total_goals)
            
            # Bonus for having any shared goals at all
            shared_bonus = 0.2 if len(shared_goals) > 0 else 0.0
            
            return min(1.0, overlap_ratio + shared_bonus)
            
        except Exception as e:
            logger.error(f"Shared interests calculation failed: {str(e)}")
            return 0.5
    
    async def _calculate_complementary_strengths(self, profile1: ConnectionProfile,
                                               profile2: ConnectionProfile) -> float:
        """Calculate how well users' strengths complement each other"""
        try:
            # Check if what one offers is different from what the other offers
            offerings1 = set(profile1.support_offering)
            offerings2 = set(profile2.support_offering)
            
            unique_offerings1 = offerings1 - offerings2
            unique_offerings2 = offerings2 - offerings1
            
            total_unique_offerings = len(unique_offerings1) + len(unique_offerings2)
            total_offerings = len(offerings1) + len(offerings2)
            
            if total_offerings == 0:
                return 0.5  # No offerings specified = neutral
            
            complementary_ratio = total_unique_offerings / total_offerings
            
            # Bonus for having some overlap (shows common ground) and some differences (complementary)
            overlap = len(offerings1 & offerings2)
            if overlap > 0 and total_unique_offerings > 0:
                balance_bonus = 0.2
            else:
                balance_bonus = 0.0
            
            return min(1.0, complementary_ratio + balance_bonus)
            
        except Exception as e:
            logger.error(f"Complementary strengths calculation failed: {str(e)}")
            return 0.5
    
    async def _calculate_availability_alignment(self, profile1: ConnectionProfile,
                                              profile2: ConnectionProfile) -> float:
        """Calculate availability compatibility"""
        try:
            # This would integrate with detailed availability/timezone systems
            # For now, use simple heuristics
            
            avail1 = profile1.availability
            avail2 = profile2.availability
            
            # Check time zone compatibility
            tz1 = avail1.get('timezone', 'UTC')
            tz2 = avail2.get('timezone', 'UTC')
            
            # Simplified timezone compatibility (in reality would calculate actual time differences)
            if tz1 == tz2:
                timezone_score = 1.0
            else:
                timezone_score = 0.7  # Different timezones can work but are challenging
            
            # Check availability preferences
            pref1 = avail1.get('preferred_times', [])
            pref2 = avail2.get('preferred_times', [])
            
            if pref1 and pref2:
                shared_times = set(pref1) & set(pref2)
                total_times = set(pref1) | set(pref2)
                time_overlap = len(shared_times) / len(total_times) if total_times else 0.5
            else:
                time_overlap = 0.7  # No specific preferences = moderate compatibility
            
            return (timezone_score * 0.6) + (time_overlap * 0.4)
            
        except Exception as e:
            logger.error(f"Availability alignment calculation failed: {str(e)}")
            return 0.7
    
    async def _calculate_safety_compatibility(self, profile1: ConnectionProfile,
                                            profile2: ConnectionProfile) -> float:
        """Calculate safety compatibility (most important factor)"""
        try:
            safety_score = 1.0
            
            # Check trauma considerations
            trauma1 = set(profile1.trauma_considerations)
            trauma2 = set(profile2.trauma_considerations)
            
            # Check for potentially conflicting trauma histories
            conflicting_traumas = {
                'abuse_survivor': 'authority_issues',
                'abandonment_trauma': 'attachment_avoidance',
                'trust_issues': 'emotional_overwhelm'
            }
            
            for trauma_type1 in trauma1:
                for trauma_type2 in trauma2:
                    if (trauma_type1 in conflicting_traumas and 
                        conflicting_traumas[trauma_type1] == trauma_type2):
                        safety_score -= 0.2
            
            # Check boundaries compatibility
            boundaries1 = profile1.boundaries
            boundaries2 = profile2.boundaries
            
            # Check for boundary conflicts
            if boundaries1.get('no_direct_contact') and boundaries2.get('prefers_direct_contact'):
                safety_score -= 0.3
            
            if boundaries1.get('slow_trust_building') and boundaries2.get('fast_connection_preference'):
                safety_score -= 0.2
            
            # Check safety requirements
            safety1 = set(profile1.safety_requirements)
            safety2 = set(profile2.safety_requirements)
            
            # Ensure both users' safety requirements can be met
            if 'professional_mediation' in safety1 or 'professional_mediation' in safety2:
                # Both need professional mediation - compatible
                if 'professional_mediation' in safety1 and 'professional_mediation' in safety2:
                    safety_score += 0.1
                else:
                    safety_score -= 0.1
            
            # High vulnerability users need extra safety considerations
            if (profile1.vulnerability_level > 0.8 or profile2.vulnerability_level > 0.8):
                if 'gentle_approach' not in safety1 and 'gentle_approach' not in safety2:
                    safety_score -= 0.2
            
            return max(0.0, safety_score)
            
        except Exception as e:
            logger.error(f"Safety compatibility calculation failed: {str(e)}")
            return 0.5  # Default to moderate safety when calculation fails
    
    async def _calculate_growth_trajectory_alignment(self, profile1: ConnectionProfile,
                                                   profile2: ConnectionProfile) -> float:
        """Calculate alignment of growth trajectories"""
        try:
            # This would integrate with more detailed growth tracking systems
            # For now, use connection goals and healing phases as proxies
            
            goals1 = set(profile1.connection_goals)
            goals2 = set(profile2.connection_goals)
            
            # Growth-oriented goals
            growth_goals = {
                'personal_development', 'skill_building', 'emotional_growth',
                'healing_acceleration', 'wisdom_development', 'service_to_others'
            }
            
            growth_goals1 = goals1 & growth_goals
            growth_goals2 = goals2 & growth_goals
            
            # Shared growth orientation
            shared_growth_goals = growth_goals1 & growth_goals2
            total_growth_goals = growth_goals1 | growth_goals2
            
            if len(total_growth_goals) == 0:
                return 0.5  # No growth goals = neutral
            
            growth_alignment = len(shared_growth_goals) / len(total_growth_goals)
            
            # Bonus for complementary growth goals
            if len(growth_goals1) > 0 and len(growth_goals2) > 0:
                complementary_bonus = 0.2
            else:
                complementary_bonus = 0.0
            
            return min(1.0, growth_alignment + complementary_bonus)
            
        except Exception as e:
            logger.error(f"Growth trajectory alignment calculation failed: {str(e)}")
            return 0.5
    
    async def _determine_optimal_connection_type(self, profile1: ConnectionProfile,
                                               profile2: ConnectionProfile,
                                               compatibility: Dict[str, Any]) -> ConnectionType:
        """Determine the optimal type of connection for two users"""
        try:
            scores = compatibility['individual_scores']
            
            # Analyze user needs and phases to determine connection type
            phase1 = profile1.healing_phase
            phase2 = profile2.healing_phase
            
            # Crisis support connections
            if phase1 == 'crisis_intervention' or phase2 == 'crisis_intervention':
                return ConnectionType.CRISIS_SUPPORT_BUDDY
            
            # Mentorship connections
            phases_ordered = [
                'initial_awareness', 'seeking_help', 'early_recovery', 
                'active_healing', 'integration', 'stable_recovery', 
                'giving_back', 'wisdom_sharing'
            ]
            
            try:
                phase1_index = phases_ordered.index(phase1)
                phase2_index = phases_ordered.index(phase2)
                
                if abs(phase1_index - phase2_index) >= 3:
                    return ConnectionType.MENTOR_MENTEE
            except ValueError:
                pass  # Phase not in ordered list
            
            # Support type analysis
            seeking1 = set(profile1.support_seeking)
            offering1 = set(profile1.support_offering)
            seeking2 = set(profile2.support_seeking)
            offering2 = set(profile2.support_offering)
            
            # Mutual support if both seek and offer similar things
            mutual_support_indicators = seeking1 & seeking2
            if len(mutual_support_indicators) >= 2:
                if 'emotional_support' in mutual_support_indicators:
                    return ConnectionType.EMOTIONAL_SUPPORT
                elif 'practical_help' in mutual_support_indicators:
                    return ConnectionType.PRACTICAL_ASSISTANCE
                else:
                    return ConnectionType.HEALING_PARTNERSHIP
            
            # Complementary strengths
            if scores.get('complementary_strengths', 0) > 0.7:
                if 'skill_sharing' in offering1 or 'skill_sharing' in offering2:
                    return ConnectionType.SKILL_EXCHANGE
                else:
                    return ConnectionType.COMPLEMENTARY_STRENGTHS
            
            # Cultural bridge connections
            if scores.get('cultural_compatibility', 0) > 0.7 and len(
                set(profile1.cultural_background) - set(profile2.cultural_background)
            ) > 0:
                return ConnectionType.CULTURAL_BRIDGE
            
            # Accountability partnerships
            if ('accountability' in seeking1 or 'accountability' in seeking2) and \
               ('motivation' in offering1 or 'motivation' in offering2):
                return ConnectionType.ACCOUNTABILITY_PARTNER
            
            # Shared journey (similar phases and needs)
            if phase1 == phase2 and len(mutual_support_indicators) > 0:
                return ConnectionType.SHARED_JOURNEY
            
            # Default to healing partnership
            return ConnectionType.HEALING_PARTNERSHIP
            
        except Exception as e:
            logger.error(f"Connection type determination failed: {str(e)}")
            return ConnectionType.HEALING_PARTNERSHIP
    
    async def _determine_recommended_action(self, connection_type: ConnectionType,
                                          profile1: ConnectionProfile,
                                          profile2: ConnectionProfile) -> CatalystAction:
        """Determine the recommended action for facilitating the connection"""
        try:
            # Safety-first approach for vulnerable users
            high_vulnerability = (profile1.vulnerability_level > 0.7 or 
                                profile2.vulnerability_level > 0.7)
            
            if high_vulnerability:
                return CatalystAction.GENTLE_INTRODUCTION
            
            # Action based on connection type
            type_action_mapping = {
                ConnectionType.CRISIS_SUPPORT_BUDDY: CatalystAction.GENTLE_INTRODUCTION,
                ConnectionType.MENTOR_MENTEE: CatalystAction.MENTORSHIP_FACILITATION,
                ConnectionType.HEALING_PARTNERSHIP: CatalystAction.SHARED_INTEREST_HIGHLIGHT,
                ConnectionType.EMOTIONAL_SUPPORT: CatalystAction.SUPPORT_CIRCLE_FORMATION,
                ConnectionType.PRACTICAL_ASSISTANCE: CatalystAction.MUTUAL_NEED_MATCHING,
                ConnectionType.SKILL_EXCHANGE: CatalystAction.COLLABORATION_SUGGESTION,
                ConnectionType.CULTURAL_BRIDGE: CatalystAction.SHARED_INTEREST_HIGHLIGHT,
                ConnectionType.COMMUNITY_BUILDING: CatalystAction.COMMUNITY_INVITATION,
                ConnectionType.ACCOUNTABILITY_PARTNER: CatalystAction.MUTUAL_NEED_MATCHING,
                ConnectionType.SHARED_JOURNEY: CatalystAction.PEER_CONNECTION,
                ConnectionType.COMPLEMENTARY_STRENGTHS: CatalystAction.COLLABORATION_SUGGESTION
            }
            
            return type_action_mapping.get(connection_type, CatalystAction.GENTLE_INTRODUCTION)
            
        except Exception as e:
            logger.error(f"Recommended action determination failed: {str(e)}")
            return CatalystAction.GENTLE_INTRODUCTION
    
    async def _create_connection_opportunity(self, user1_id: str, user2_id: str,
                                           connection_type: ConnectionType,
                                           compatibility: Dict[str, Any],
                                           recommended_action: CatalystAction) -> Optional[ConnectionOpportunity]:
        """Create a connection opportunity object"""
        try:
            profile1 = self.user_profiles[user1_id]
            profile2 = self.user_profiles[user2_id]
            
            connection_strength = compatibility['total_score']
            strength_level = await self._determine_connection_strength_level(connection_strength)
            
            # Analyze mutual benefits
            mutual_benefits = await self._analyze_mutual_benefits(profile1, profile2, connection_type)
            
            # Identify potential challenges
            potential_challenges = await self._identify_potential_challenges(profile1, profile2, compatibility)
            
            # Determine introduction approach
            introduction_approach = await self._determine_introduction_approach(
                recommended_action, profile1, profile2
            )
            
            # Safety and cultural considerations
            safety_considerations = await self._analyze_safety_considerations(profile1, profile2)
            cultural_considerations = await self._analyze_cultural_considerations(profile1, profile2)
            
            # Timing and success estimation
            timing_recommendation = await self._determine_optimal_timing(profile1, profile2)
            success_probability = await self._estimate_success_probability(compatibility, safety_considerations)
            healing_potential = await self._estimate_healing_potential(connection_type, compatibility)
            
            # Privacy and consent checks
            privacy_safe = await self._validate_privacy_safety(profile1, profile2)
            consent_required = self.consent_required_always or (
                profile1.vulnerability_level > 0.6 or profile2.vulnerability_level > 0.6
            )
            
            # Expiration time (opportunities should be timely)
            expiration_time = datetime.utcnow() + timedelta(days=7)
            
            opportunity = ConnectionOpportunity(
                opportunity_id=f"conn_opp_{datetime.utcnow().isoformat()}_{id(self)}",
                user_1_id=user1_id,
                user_2_id=user2_id,
                connection_type=connection_type,
                connection_strength=connection_strength,
                strength_level=strength_level,
                compatibility_factors=compatibility['individual_scores'],
                mutual_benefits=mutual_benefits,
                potential_challenges=potential_challenges,
                recommended_action=recommended_action,
                introduction_approach=introduction_approach,
                safety_considerations=safety_considerations,
                cultural_considerations=cultural_considerations,
                timing_recommendation=timing_recommendation,
                success_probability=success_probability,
                healing_potential=healing_potential,
                identified_at=datetime.utcnow(),
                expiration_time=expiration_time,
                privacy_safe=privacy_safe,
                consent_required=consent_required
            )
            
            return opportunity
            
        except Exception as e:
            logger.error(f"Connection opportunity creation failed: {str(e)}")
            return None
    
    async def _determine_connection_strength_level(self, strength: float) -> ConnectionStrength:
        """Determine connection strength level from numeric score"""
        if strength >= 0.9:
            return ConnectionStrength.TRANSFORMATIONAL
        elif strength >= 0.8:
            return ConnectionStrength.VERY_STRONG
        elif strength >= 0.6:
            return ConnectionStrength.STRONG
        elif strength >= 0.3:
            return ConnectionStrength.MODERATE
        else:
            return ConnectionStrength.WEAK
    
    async def _analyze_mutual_benefits(self, profile1: ConnectionProfile, profile2: ConnectionProfile,
                                     connection_type: ConnectionType) -> List[str]:
        """Analyze mutual benefits of the connection"""
        try:
            benefits = []
            
            # Type-specific benefits
            if connection_type == ConnectionType.HEALING_PARTNERSHIP:
                benefits.extend([
                    'Mutual emotional support during healing journey',
                    'Shared understanding of healing challenges',
                    'Accountability for healing goals',
                    'Reduced isolation during recovery'
                ])
            
            elif connection_type == ConnectionType.MENTOR_MENTEE:
                benefits.extend([
                    'Experienced guidance for healing journey',
                    'Wisdom sharing from lived experience',
                    'Hope through seeing recovery success',
                    'Mentorship fulfillment for mentor'
                ])
            
            elif connection_type == ConnectionType.CRISIS_SUPPORT_BUDDY:
                benefits.extend([
                    'Immediate emotional support during crisis',
                    'Crisis intervention and safety planning',
                    'Connection to professional resources',
                    'Reduced crisis isolation'
                ])
            
            elif connection_type == ConnectionType.SKILL_EXCHANGE:
                benefits.extend([
                    'Mutual skill development',
                    'Practical capability building',
                    'Knowledge sharing and growth',
                    'Collaborative learning experience'
                ])
            
            # Support matching benefits
            seeking1 = set(profile1.support_seeking)
            offering2 = set(profile2.support_offering)
            matched_support = seeking1 & offering2
            
            for support_type in matched_support:
                benefits.append(f"User 1 receives {support_type} from User 2")
            
            seeking2 = set(profile2.support_seeking)
            offering1 = set(profile1.support_offering)
            matched_support_reverse = seeking2 & offering1
            
            for support_type in matched_support_reverse:
                benefits.append(f"User 2 receives {support_type} from User 1")
            
            # Cultural benefits
            cultures1 = set(profile1.cultural_background)
            cultures2 = set(profile2.cultural_background)
            
            if cultures1 != cultures2 and len(cultures1) > 0 and len(cultures2) > 0:
                benefits.append('Cross-cultural learning and understanding')
                benefits.append('Expanded cultural perspectives')
            
            return benefits
            
        except Exception as e:
            logger.error(f"Mutual benefits analysis failed: {str(e)}")
            return ['Mutual support and growth']
    
    async def _identify_potential_challenges(self, profile1: ConnectionProfile, profile2: ConnectionProfile,
                                           compatibility: Dict[str, Any]) -> List[str]:
        """Identify potential challenges in the connection"""
        try:
            challenges = []
            scores = compatibility['individual_scores']
            
            # Communication challenges
            if scores.get('communication_style_match', 0) < 0.6:
                challenges.append('Different communication styles may require adjustment')
            
            # Vulnerability imbalance
            vuln_diff = abs(profile1.vulnerability_level - profile2.vulnerability_level)
            if vuln_diff > 0.4:
                challenges.append('Different vulnerability levels may affect emotional safety')
            
            # Cultural differences
            if scores.get('cultural_compatibility', 0) < 0.5:
                challenges.append('Cultural differences may require sensitivity and patience')
            
            # Availability mismatches
            if scores.get('availability_alignment', 0) < 0.5:
                challenges.append('Scheduling and availability differences')
            
            # Safety considerations
            trauma1 = set(profile1.trauma_considerations)
            trauma2 = set(profile2.trauma_considerations)
            
            if len(trauma1) > 0 or len(trauma2) > 0:
                challenges.append('Trauma histories require careful, gentle approach')
            
            # Boundary conflicts
            boundaries1 = profile1.boundaries
            boundaries2 = profile2.boundaries
            
            if boundaries1.get('slow_trust_building') and boundaries2.get('fast_connection_preference'):
                challenges.append('Different paces of trust building')
            
            # High vulnerability warnings
            if profile1.vulnerability_level > 0.8 or profile2.vulnerability_level > 0.8:
                challenges.append('High vulnerability requires extra care and support')
            
            # Support needs imbalance
            if scores.get('support_needs_matching', 0) < 0.4:
                challenges.append('Support needs may not align well')
            
            return challenges
            
        except Exception as e:
            logger.error(f"Potential challenges identification failed: {str(e)}")
            return ['General relationship building challenges']
    
    async def _determine_introduction_approach(self, action: CatalystAction,
                                             profile1: ConnectionProfile,
                                             profile2: ConnectionProfile) -> str:
        """Determine the specific approach for making introductions"""
        try:
            # Safety-first for vulnerable users
            if profile1.vulnerability_level > 0.7 or profile2.vulnerability_level > 0.7:
                return "gentle_mediated_introduction_with_safety_emphasis"
            
            # Action-specific approaches
            approach_mapping = {
                CatalystAction.GENTLE_INTRODUCTION: "warm_personal_introduction_highlighting_compatibility",
                CatalystAction.SHARED_INTEREST_HIGHLIGHT: "introduction_focusing_on_shared_interests_and_goals",
                CatalystAction.MUTUAL_NEED_MATCHING: "introduction_emphasizing_mutual_support_potential",
                CatalystAction.MENTORSHIP_FACILITATION: "structured_mentorship_introduction_with_guidelines",
                CatalystAction.SUPPORT_CIRCLE_FORMATION: "invitation_to_support_circle_with_both_users",
                CatalystAction.COLLABORATION_SUGGESTION: "project_collaboration_introduction",
                CatalystAction.PEER_CONNECTION: "peer_introduction_emphasizing_shared_journey",
                CatalystAction.COMMUNITY_INVITATION: "community_event_introduction",
                CatalystAction.HEALING_CIRCLE_INVITATION: "healing_circle_invitation_with_both_users"
            }
            
            base_approach = approach_mapping.get(action, "gentle_personal_introduction")
            
            # Cultural adaptations
            cultures1 = set(profile1.cultural_background)
            cultures2 = set(profile2.cultural_background)
            
            if len(cultures1) > 0 or len(cultures2) > 0:
                if cultures1 == cultures2:
                    base_approach += "_with_cultural_resonance_emphasis"
                else:
                    base_approach += "_with_cultural_sensitivity_and_bridge_building"
            
            # Trauma-informed adaptations
            if (len(profile1.trauma_considerations) > 0 or 
                len(profile2.trauma_considerations) > 0):
                base_approach += "_with_trauma_informed_safety_measures"
            
            return base_approach
            
        except Exception as e:
            logger.error(f"Introduction approach determination failed: {str(e)}")
            return "gentle_personal_introduction"
    
    async def _analyze_safety_considerations(self, profile1: ConnectionProfile,
                                           profile2: ConnectionProfile) -> List[str]:
        """Analyze safety considerations for the connection"""
        try:
            safety_considerations = []
            
            # Vulnerability-based considerations
            if profile1.vulnerability_level > 0.7:
                safety_considerations.append('User 1 requires gentle, trauma-informed approach')
            
            if profile2.vulnerability_level > 0.7:
                safety_considerations.append('User 2 requires gentle, trauma-informed approach')
            
            # Trauma considerations
            trauma1 = profile1.trauma_considerations
            trauma2 = profile2.trauma_considerations
            
            if 'abuse_survivor' in trauma1 or 'abuse_survivor' in trauma2:
                safety_considerations.append('Abuse survivor present - extra safety protocols needed')
            
            if 'trust_issues' in trauma1 or 'trust_issues' in trauma2:
                safety_considerations.append('Trust issues present - slow trust building required')
            
            if 'abandonment_trauma' in trauma1 or 'abandonment_trauma' in trauma2:
                safety_considerations.append('Abandonment trauma present - consistent, reliable contact needed')
            
            # Safety requirements
            safety1 = set(profile1.safety_requirements)
            safety2 = set(profile2.safety_requirements)
            
            if 'professional_mediation' in safety1 or 'professional_mediation' in safety2:
                safety_considerations.append('Professional mediation or oversight recommended')
            
            if 'group_setting_only' in safety1 or 'group_setting_only' in safety2:
                safety_considerations.append('Initial meetings should be in group settings')
            
            if 'public_space_meetings' in safety1 or 'public_space_meetings' in safety2:
                safety_considerations.append('Meetings should be in public, safe spaces')
            
            # Boundary considerations
            boundaries1 = profile1.boundaries
            boundaries2 = profile2.boundaries
            
            if boundaries1.get('no_personal_info_sharing') or boundaries2.get('no_personal_info_sharing'):
                safety_considerations.append('Personal information sharing boundaries must be respected')
            
            if boundaries1.get('limited_contact_frequency') or boundaries2.get('limited_contact_frequency'):
                safety_considerations.append('Contact frequency boundaries must be established and respected')
            
            return safety_considerations
            
        except Exception as e:
            logger.error(f"Safety considerations analysis failed: {str(e)}")
            return ['General safety protocols apply']
    
    async def _analyze_cultural_considerations(self, profile1: ConnectionProfile,
                                             profile2: ConnectionProfile) -> List[str]:
        """Analyze cultural considerations for the connection"""
        try:
            cultural_considerations = []
            
            cultures1 = set(profile1.cultural_background)
            cultures2 = set(profile2.cultural_background)
            
            # Same culture considerations
            shared_cultures = cultures1 & cultures2
            if shared_cultures:
                cultural_considerations.append(f'Shared cultural background: {", ".join(shared_cultures)}')
            
            # Different culture considerations
            different_cultures = cultures1 ^ cultures2  # Symmetric difference
            if different_cultures:
                cultural_considerations.append('Cross-cultural sensitivity required')
                cultural_considerations.append('Opportunity for cultural learning and exchange')
            
            # Specific cultural considerations
            all_cultures = cultures1 | cultures2
            
            if any(culture in ['indigenous', 'native_american', 'aboriginal'] for culture in all_cultures):
                cultural_considerations.append('Indigenous cultural protocols and wisdom traditions should be honored')
            
            if any(culture in ['collectivist', 'asian', 'african', 'latin_american'] for culture in all_cultures):
                cultural_considerations.append('Collectivist cultural values and family considerations important')
            
            if any(culture in ['religious', 'spiritual'] for culture in all_cultures):
                cultural_considerations.append('Religious/spiritual beliefs and practices should be respected')
            
            # Language considerations
            # This would integrate with language preference system
            cultural_considerations.append('Language preferences and communication styles should be considered')
            
            return cultural_considerations
            
        except Exception as e:
            logger.error(f"Cultural considerations analysis failed: {str(e)}")
            return ['Cultural sensitivity and respect required']
    
    async def _determine_optimal_timing(self, profile1: ConnectionProfile,
                                      profile2: ConnectionProfile) -> str:
        """Determine optimal timing for the connection"""
        try:
            # Crisis situations need immediate connection
            if profile1.healing_phase == 'crisis_intervention' or profile2.healing_phase == 'crisis_intervention':
                return 'immediate_with_crisis_protocols'
            
            # High vulnerability needs careful timing
            if profile1.vulnerability_level > 0.8 or profile2.vulnerability_level > 0.8:
                return 'gentle_timing_with_preparation'
            
            # Availability-based timing
            avail1 = profile1.availability
            avail2 = profile2.availability
            
            # Check for preferred timing
            pref1 = avail1.get('preferred_introduction_timing', 'flexible')
            pref2 = avail2.get('preferred_introduction_timing', 'flexible')
            
            if pref1 == 'immediate' and pref2 == 'immediate':
                return 'immediate_mutual_preference'
            elif pref1 == 'gradual' or pref2 == 'gradual':
                return 'gradual_with_preparation'
            elif pref1 == 'when_ready' or pref2 == 'when_ready':
                return 'when_both_users_confirm_readiness'
            
            # Default timing based on healing phases
            phases = [profile1.healing_phase, profile2.healing_phase]
            
            if 'seeking_help' in phases:
                return 'timely_while_help_seeking'
            elif 'active_healing' in phases:
                return 'during_active_healing_phase'
            elif 'giving_back' in phases:
                return 'when_service_oriented'
            
            return 'flexible_timing_based_on_mutual_availability'
            
        except Exception as e:
            logger.error(f"Optimal timing determination failed: {str(e)}")
            return 'flexible_timing'
    
    async def _estimate_success_probability(self, compatibility: Dict[str, Any],
                                          safety_considerations: List[str]) -> float:
        """Estimate probability of successful connection"""
        try:
            base_probability = compatibility['total_score']
            
            # Safety considerations affect success probability
            safety_challenges = len([c for c in safety_considerations if 'requires' in c or 'must' in c])
            safety_penalty = safety_challenges * 0.05
            
            # Adjust for specific factors
            scores = compatibility['individual_scores']
            
            # High safety compatibility boosts success
            if scores.get('safety_compatibility', 0) > 0.8:
                base_probability += 0.1
            
            # Good communication match boosts success
            if scores.get('communication_style_match', 0) > 0.8:
                base_probability += 0.05
            
            # Strong support needs matching boosts success
            if scores.get('support_needs_matching', 0) > 0.8:
                base_probability += 0.1
            
            success_probability = base_probability - safety_penalty
            
            return max(0.0, min(1.0, success_probability))
            
        except Exception as e:
            logger.error(f"Success probability estimation failed: {str(e)}")
            return 0.5
    
    async def _estimate_healing_potential(self, connection_type: ConnectionType,
                                        compatibility: Dict[str, Any]) -> float:
        """Estimate healing potential of the connection"""
        try:
            # Base healing potential by connection type
            type_healing_potential = {
                ConnectionType.HEALING_PARTNERSHIP: 0.9,
                ConnectionType.MENTOR_MENTEE: 0.8,
                ConnectionType.CRISIS_SUPPORT_BUDDY: 0.95,
                ConnectionType.EMOTIONAL_SUPPORT: 0.85,
                ConnectionType.SHARED_JOURNEY: 0.8,
                ConnectionType.COMPLEMENTARY_STRENGTHS: 0.7,
                ConnectionType.COMMUNITY_BUILDING: 0.75,
                ConnectionType.CULTURAL_BRIDGE: 0.7,
                ConnectionType.SKILL_EXCHANGE: 0.6,
                ConnectionType.PRACTICAL_ASSISTANCE: 0.65,
                ConnectionType.ACCOUNTABILITY_PARTNER: 0.7,
                ConnectionType.INTERGENERATIONAL: 0.75
            }
            
            base_potential = type_healing_potential.get(connection_type, 0.6)
            
            # Adjust based on compatibility factors
            scores = compatibility['individual_scores']
            
            # Strong support matching increases healing potential
            support_bonus = scores.get('support_needs_matching', 0) * 0.2
            
            # Good healing phase compatibility increases potential
            phase_bonus = scores.get('healing_phase_compatibility', 0) * 0.15
            
            # Cultural compatibility can enhance healing
            cultural_bonus = scores.get('cultural_compatibility', 0) * 0.1
            
            healing_potential = base_potential + support_bonus + phase_bonus + cultural_bonus
            
            return min(1.0, healing_potential)
            
        except Exception as e:
            logger.error(f"Healing potential estimation failed: {str(e)}")
            return 0.6
    
    async def _validate_privacy_safety(self, profile1: ConnectionProfile,
                                     profile2: ConnectionProfile) -> bool:
        """Validate that the connection preserves privacy and safety"""
        try:
            # Check for privacy conflicts
            if (profile1.boundaries.get('no_personal_info_sharing') and 
                profile2.connection_preferences.get('prefers_personal_sharing')):
                return False
            
            if (profile2.boundaries.get('no_personal_info_sharing') and 
                profile1.connection_preferences.get('prefers_personal_sharing')):
                return False
            
            # Check accessibility requirements don't create privacy issues
            access1 = profile1.accessibility_needs
            access2 = profile2.accessibility_needs
            
            # Some accessibility needs might conflict with privacy preferences
            if ('public_record_keeping' in access1 and 
                'private_communication_only' in profile2.boundaries):
                return False
            
            # Default to privacy safe
            return True
            
        except Exception as e:
            logger.error(f"Privacy safety validation failed: {str(e)}")
            return True
    
    def _find_existing_connection(self, user1_id: str, user2_id: str) -> Optional[EstablishedConnection]:
        """Find existing connection between two users"""
        for connection in self.established_connections.values():
            if ((connection.user_1_id == user1_id and connection.user_2_id == user2_id) or
                (connection.user_1_id == user2_id and connection.user_2_id == user1_id)):
                return connection
        return None
    
    # Analytics and reporting
    def get_catalyst_analytics(self) -> Dict[str, Any]:
        """Get analytics on connection catalyst performance"""
        try:
            total_opportunities = len(self.identified_opportunities)
            
            if total_opportunities == 0:
                return {
                    'opportunities_identified': 0,
                    'connections_attempted': 0,
                    'connections_established': 0,
                    'success_rate': 0.0,
                    'average_healing_impact': 0.0,
                    'connection_type_distribution': {},
                    'cultural_bridge_connections': 0,
                    'total_healing_impact': 0.0
                }
            
            # Calculate success rate
            success_rate = self.connections_established / max(1, self.connections_attempted)
            
            # Average healing impact
            if self.established_connections:
                healing_impacts = [
                    sum(conn.healing_impact.values()) 
                    for conn in self.established_connections.values()
                ]
                average_healing_impact = statistics.mean(healing_impacts)
            else:
                average_healing_impact = 0.0
            
            # Connection type distribution
            connection_types = [
                opp.connection_type.value 
                for opp in self.identified_opportunities.values()
            ]
            type_distribution = {
                ct.value: connection_types.count(ct.value) 
                for ct in ConnectionType
            }
            
            # Cultural bridge connections
            cultural_bridge_count = sum(
                1 for opp in self.identified_opportunities.values()
                if opp.connection_type == ConnectionType.CULTURAL_BRIDGE
            )
            
            return {
                'opportunities_identified': total_opportunities,
                'connections_attempted': self.connections_attempted,
                'connections_established': self.connections_established,
                'success_rate': success_rate,
                'average_healing_impact': average_healing_impact,
                'connection_type_distribution': type_distribution,
                'cultural_bridge_connections': cultural_bridge_count,
                'total_healing_impact': self.total_healing_impact,
                'community_networks_formed': self.community_networks_formed,
                'active_user_profiles': len(self.user_profiles)
            }
            
        except Exception as e:
            logger.error(f"Catalyst analytics generation failed: {str(e)}")
            return {}