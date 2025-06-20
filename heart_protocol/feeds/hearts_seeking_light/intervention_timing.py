"""
Context-Appropriate Intervention Timing

Determines the optimal timing for support interventions based on:
- User's emotional state and capacity
- Social context and appropriateness
- Availability of support providers
- Privacy and consent considerations
- Crisis vs. routine support timing

Built on trauma-informed care principles and respect for user agency.
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta, time
from enum import Enum
from dataclasses import dataclass
import logging

from .support_detection import SupportSeeker, SupportOffer, SupportUrgency, SupportType

logger = logging.getLogger(__name__)


class InterventionType(Enum):
    """Types of support interventions"""
    IMMEDIATE_CRISIS = "immediate_crisis"        # Crisis response within minutes
    URGENT_SUPPORT = "urgent_support"           # High-priority support within hours
    GENTLE_OUTREACH = "gentle_outreach"         # Non-urgent supportive contact
    RESOURCE_SHARING = "resource_sharing"       # Information and resources
    CONNECTION_OFFER = "connection_offer"       # Offer to connect with support
    CHECK_IN = "check_in"                       # Follow-up after support
    CELEBRATION = "celebration"                 # Acknowledging progress/wins


class InterventionContext(Enum):
    """Context for intervention timing"""
    PUBLIC_POST = "public_post"                 # Response to public post
    PRIVATE_MESSAGE = "private_message"         # Direct private outreach
    COMMUNITY_SPACE = "community_space"         # Group/community context
    CRISIS_PROTOCOL = "crisis_protocol"         # Emergency intervention
    FOLLOW_UP = "follow_up"                     # Continuing support relationship


class TimingWindow(Enum):
    """Timing windows for interventions"""
    IMMEDIATE = "immediate"                     # 0-5 minutes
    VERY_URGENT = "very_urgent"                # 5-30 minutes
    URGENT = "urgent"                          # 30 minutes - 2 hours
    PRIORITY = "priority"                      # 2-8 hours
    ROUTINE = "routine"                        # 8-24 hours
    FLEXIBLE = "flexible"                      # 1-7 days
    ONGOING = "ongoing"                        # Ongoing relationship


@dataclass
class InterventionTiming:
    """Represents optimal timing for a support intervention"""
    intervention_type: InterventionType
    timing_window: TimingWindow
    earliest_time: datetime
    latest_time: datetime
    optimal_time: datetime
    context: InterventionContext
    confidence_score: float  # 0.0 to 1.0
    reasoning: str
    prerequisites: List[str]  # What must happen before intervention
    considerations: List[str]  # Important factors to consider
    escalation_triggers: List[str]  # When to escalate urgency
    
    def is_within_window(self, current_time: datetime) -> bool:
        """Check if current time is within the intervention window"""
        return self.earliest_time <= current_time <= self.latest_time
    
    def time_until_window(self, current_time: datetime) -> Optional[timedelta]:
        """Get time until intervention window opens"""
        if current_time >= self.earliest_time:
            return timedelta(0)
        return self.earliest_time - current_time
    
    def time_remaining(self, current_time: datetime) -> Optional[timedelta]:
        """Get time remaining in intervention window"""
        if current_time > self.latest_time:
            return None
        return self.latest_time - current_time


class InterventionTimingEngine:
    """
    Determines context-appropriate timing for support interventions.
    
    Core Principles:
    - Respect user privacy and agency
    - Match intervention intensity to need urgency
    - Consider cultural and personal boundaries
    - Balance immediate help with non-intrusive support
    - Prioritize human connection over algorithmic efficiency
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Timing rules for different scenarios
        self.timing_rules = self._initialize_timing_rules()
        self.cultural_considerations = self._initialize_cultural_timing()
        self.crisis_protocols = self._initialize_crisis_protocols()
        self.boundary_rules = self._initialize_boundary_rules()
        
        # Track intervention history for learning
        self.intervention_history = {}  # user_id -> List[InterventionTiming]
        
        logger.info("Intervention Timing Engine initialized")
    
    def _initialize_timing_rules(self) -> Dict[SupportUrgency, Dict[str, Any]]:
        """Initialize timing rules based on support urgency"""
        
        return {
            SupportUrgency.CRISIS: {
                'intervention_type': InterventionType.IMMEDIATE_CRISIS,
                'timing_window': TimingWindow.IMMEDIATE,
                'max_delay_minutes': 5,
                'context_requirements': ['crisis_protocol_active'],
                'allowed_times': ['any_time'],  # Crisis overrides normal timing
                'minimum_responders': 2,  # Multiple people should respond
                'escalation_required': True,
                'human_oversight': True
            },
            
            SupportUrgency.HIGH: {
                'intervention_type': InterventionType.URGENT_SUPPORT,
                'timing_window': TimingWindow.URGENT,
                'max_delay_minutes': 120,
                'context_requirements': ['consent_verified'],
                'allowed_times': ['daytime_hours', 'early_evening'],
                'minimum_responders': 1,
                'escalation_required': False,
                'human_oversight': True
            },
            
            SupportUrgency.MODERATE: {
                'intervention_type': InterventionType.GENTLE_OUTREACH,
                'timing_window': TimingWindow.PRIORITY,
                'max_delay_minutes': 480,  # 8 hours
                'context_requirements': ['explicit_consent', 'appropriate_context'],
                'allowed_times': ['business_hours', 'evening_hours'],
                'minimum_responders': 1,
                'escalation_required': False,
                'human_oversight': False
            },
            
            SupportUrgency.LOW: {
                'intervention_type': InterventionType.RESOURCE_SHARING,
                'timing_window': TimingWindow.ROUTINE,
                'max_delay_minutes': 1440,  # 24 hours
                'context_requirements': ['user_comfort', 'natural_opportunity'],
                'allowed_times': ['business_hours'],
                'minimum_responders': 1,
                'escalation_required': False,
                'human_oversight': False
            },
            
            SupportUrgency.ONGOING: {
                'intervention_type': InterventionType.CHECK_IN,
                'timing_window': TimingWindow.FLEXIBLE,
                'max_delay_minutes': 10080,  # 7 days
                'context_requirements': ['established_relationship', 'user_consent'],
                'allowed_times': ['preferred_user_times'],
                'minimum_responders': 1,
                'escalation_required': False,
                'human_oversight': False
            }
        }
    
    def _initialize_cultural_timing(self) -> Dict[str, Dict[str, Any]]:
        """Initialize cultural considerations for intervention timing"""
        
        return {
            'individualistic_cultures': {
                'respect_privacy': True,
                'direct_approach_ok': True,
                'family_involvement': 'only_if_requested',
                'preferred_communication': ['private_message', 'direct_reply'],
                'timing_preferences': ['business_hours', 'early_evening']
            },
            
            'collectivist_cultures': {
                'respect_privacy': True,
                'community_context_important': True,
                'family_involvement': 'may_be_helpful',
                'preferred_communication': ['community_space', 'group_context'],
                'timing_preferences': ['evening_hours', 'weekend_time']
            },
            
            'high_context_cultures': {
                'indirect_approach_preferred': True,
                'relationship_building_first': True,
                'subtle_intervention': True,
                'timing_preferences': ['relationship_context', 'natural_opportunities']
            },
            
            'religious_communities': {
                'spiritual_considerations': True,
                'community_support_valued': True,
                'prayer_times_respected': True,
                'holiday_awareness': True,
                'timing_preferences': ['community_gathering_times', 'spiritual_practice_times']
            }
        }
    
    def _initialize_crisis_protocols(self) -> Dict[str, Any]:
        """Initialize crisis intervention protocols"""
        
        return {
            'immediate_response': {
                'max_response_time_minutes': 5,
                'required_actions': [
                    'assess_immediate_safety',
                    'provide_crisis_resources',
                    'initiate_human_contact',
                    'monitor_continuously'
                ],
                'escalation_triggers': [
                    'explicit_suicide_ideation',
                    'immediate_harm_plan',
                    'no_response_to_outreach',
                    'escalating_crisis_language'
                ],
                'override_permissions': ['privacy_settings', 'timing_preferences']
            },
            
            'crisis_team_notification': {
                'immediate_notification': True,
                'human_responder_required': True,
                'professional_backup': True,
                'documentation_required': True
            },
            
            'safety_protocols': {
                'never_leave_alone': True,
                'continuous_monitoring': True,
                'multiple_responders': True,
                'professional_handoff': True
            }
        }
    
    def _initialize_boundary_rules(self) -> Dict[str, List[str]]:
        """Initialize user boundary and privacy rules"""
        
        return {
            'consent_requirements': [
                'explicit_consent_for_private_contact',
                'respect_communication_preferences',
                'honor_do_not_contact_requests',
                'verify_identity_for_crisis'
            ],
            
            'privacy_protections': [
                'no_unsolicited_contact',
                'respect_anonymous_preferences',
                'protect_shared_information',
                'secure_communication_channels'
            ],
            
            'cultural_boundaries': [
                'respect_religious_practices',
                'honor_family_structures',
                'consider_gender_preferences',
                'respect_cultural_timing'
            ],
            
            'professional_boundaries': [
                'clear_role_definitions',
                'appropriate_scope_of_support',
                'referral_when_needed',
                'documentation_and_oversight'
            ]
        }
    
    async def calculate_optimal_intervention_timing(self, 
                                                  seeker: SupportSeeker,
                                                  offer: SupportOffer,
                                                  context: Dict[str, Any]) -> InterventionTiming:
        """
        Calculate optimal timing for a support intervention.
        
        Args:
            seeker: Person seeking support
            offer: Support offer being considered
            context: Additional context (user preferences, current time, etc.)
        """
        try:
            # Get base timing rules for urgency level
            base_rules = self.timing_rules[seeker.urgency]
            
            # Calculate timing window
            current_time = context.get('current_time', datetime.utcnow())
            earliest_time, latest_time, optimal_time = await self._calculate_timing_window(
                seeker, offer, base_rules, current_time, context
            )
            
            # Determine intervention context
            intervention_context = await self._determine_intervention_context(
                seeker, offer, context
            )
            
            # Apply cultural considerations
            cultural_adjustments = await self._apply_cultural_timing_adjustments(
                seeker, offer, context, earliest_time, latest_time
            )
            
            earliest_time = cultural_adjustments.get('earliest_time', earliest_time)
            latest_time = cultural_adjustments.get('latest_time', latest_time)
            optimal_time = cultural_adjustments.get('optimal_time', optimal_time)
            
            # Check prerequisites and constraints
            prerequisites = await self._check_prerequisites(seeker, offer, context)
            considerations = await self._identify_considerations(seeker, offer, context)
            escalation_triggers = await self._identify_escalation_triggers(seeker, context)
            
            # Calculate confidence score
            confidence_score = await self._calculate_timing_confidence(
                seeker, offer, context, prerequisites
            )
            
            # Generate reasoning
            reasoning = await self._generate_timing_reasoning(
                seeker, offer, base_rules, cultural_adjustments, context
            )
            
            intervention_timing = InterventionTiming(
                intervention_type=base_rules['intervention_type'],
                timing_window=base_rules['timing_window'],
                earliest_time=earliest_time,
                latest_time=latest_time,
                optimal_time=optimal_time,
                context=intervention_context,
                confidence_score=confidence_score,
                reasoning=reasoning,
                prerequisites=prerequisites,
                considerations=considerations,
                escalation_triggers=escalation_triggers
            )
            
            # Store in history for learning
            await self._record_intervention_timing(seeker.user_id, intervention_timing)
            
            logger.info(f"Calculated intervention timing for {seeker.user_id[:8]}... "
                       f"(Window: {base_rules['timing_window'].value}, "
                       f"Confidence: {confidence_score:.2f})")
            
            return intervention_timing
            
        except Exception as e:
            logger.error(f"Error calculating intervention timing: {e}")
            return await self._get_fallback_timing(seeker, context)
    
    async def _calculate_timing_window(self, seeker: SupportSeeker, offer: SupportOffer,
                                     base_rules: Dict[str, Any], current_time: datetime,
                                     context: Dict[str, Any]) -> Tuple[datetime, datetime, datetime]:
        """Calculate the timing window for intervention"""
        
        # Base delay from rules
        max_delay = timedelta(minutes=base_rules['max_delay_minutes'])
        
        # Crisis gets immediate timing
        if seeker.urgency == SupportUrgency.CRISIS:
            earliest_time = current_time
            latest_time = current_time + timedelta(minutes=5)
            optimal_time = current_time + timedelta(minutes=1)
            return earliest_time, latest_time, optimal_time
        
        # Consider user timezone and preferences
        user_tz = context.get('user_timezone', 'UTC')
        user_time = current_time  # In production, would convert to user timezone
        
        # Apply time-of-day constraints
        earliest_time = await self._apply_time_constraints(
            current_time, base_rules['allowed_times'], context, 'earliest'
        )
        
        latest_time = earliest_time + max_delay
        
        # Apply latest time constraints
        latest_time = await self._apply_time_constraints(
            latest_time, base_rules['allowed_times'], context, 'latest'
        )
        
        # Calculate optimal time within window
        optimal_time = await self._calculate_optimal_time_in_window(
            earliest_time, latest_time, seeker, offer, context
        )
        
        return earliest_time, latest_time, optimal_time
    
    async def _apply_time_constraints(self, target_time: datetime, allowed_times: List[str],
                                    context: Dict[str, Any], bound_type: str) -> datetime:
        """Apply time-of-day constraints to timing"""
        
        if 'any_time' in allowed_times:
            return target_time
        
        target_hour = target_time.hour
        
        # Define time ranges
        time_ranges = {
            'business_hours': (9, 17),
            'daytime_hours': (7, 18),
            'evening_hours': (17, 21),
            'early_evening': (17, 19),
            'late_evening': (19, 23),
            'preferred_user_times': context.get('user_preferred_hours', (9, 17))
        }
        
        # Find the most restrictive applicable range
        valid_ranges = []
        for time_type in allowed_times:
            if time_type in time_ranges:
                valid_ranges.append(time_ranges[time_type])
        
        if not valid_ranges:
            return target_time
        
        # Apply most restrictive range
        if bound_type == 'earliest':
            earliest_hour = max(start for start, end in valid_ranges)
            if target_hour < earliest_hour:
                return target_time.replace(hour=earliest_hour, minute=0, second=0)
        else:  # latest
            latest_hour = min(end for start, end in valid_ranges)
            if target_hour > latest_hour:
                return target_time.replace(hour=latest_hour, minute=0, second=0)
        
        return target_time
    
    async def _calculate_optimal_time_in_window(self, earliest: datetime, latest: datetime,
                                              seeker: SupportSeeker, offer: SupportOffer,
                                              context: Dict[str, Any]) -> datetime:
        """Calculate optimal time within the intervention window"""
        
        # For crisis, optimal is immediate
        if seeker.urgency == SupportUrgency.CRISIS:
            return earliest
        
        # Consider support provider availability
        provider_optimal = await self._get_provider_optimal_time(offer, earliest, latest)
        
        # Consider user receptivity patterns
        user_optimal = await self._get_user_optimal_time(seeker, earliest, latest, context)
        
        # Balance between provider and user optimal times
        if provider_optimal and user_optimal:
            # Find midpoint
            provider_timestamp = provider_optimal.timestamp()
            user_timestamp = user_optimal.timestamp()
            optimal_timestamp = (provider_timestamp + user_timestamp) / 2
            optimal_time = datetime.fromtimestamp(optimal_timestamp)
            
            # Ensure it's within window
            optimal_time = max(earliest, min(latest, optimal_time))
        elif provider_optimal:
            optimal_time = provider_optimal
        elif user_optimal:
            optimal_time = user_optimal
        else:
            # Default to middle of window
            window_duration = latest - earliest
            optimal_time = earliest + (window_duration / 2)
        
        return optimal_time
    
    async def _get_provider_optimal_time(self, offer: SupportOffer, 
                                       earliest: datetime, latest: datetime) -> Optional[datetime]:
        """Get optimal time based on support provider availability"""
        
        # Parse availability
        if offer.availability == 'immediate':
            return earliest
        elif offer.availability == 'within_hours':
            # Prefer within 2 hours if possible
            preferred = earliest + timedelta(hours=2)
            return min(preferred, latest)
        else:  # flexible
            # Prefer middle of provider's available times
            available_times = offer.available_times
            if 'anytime' in available_times:
                return None  # No preference
            
            # Simple heuristic - prefer middle of day if "daytime" mentioned
            if any('day' in time_pref.lower() for time_pref in available_times):
                midday = earliest.replace(hour=12, minute=0, second=0)
                if earliest <= midday <= latest:
                    return midday
        
        return None
    
    async def _get_user_optimal_time(self, seeker: SupportSeeker, 
                                   earliest: datetime, latest: datetime,
                                   context: Dict[str, Any]) -> Optional[datetime]:
        """Get optimal time based on user preferences and patterns"""
        
        # Check user's stated available times
        available_times = seeker.available_times
        if 'anytime' in available_times:
            return None  # No preference
        
        # Parse user preferences
        preferred_hours = []
        for time_pref in available_times:
            if 'morning' in time_pref:
                preferred_hours.extend([8, 9, 10])
            elif 'afternoon' in time_pref:
                preferred_hours.extend([13, 14, 15])
            elif 'evening' in time_pref:
                preferred_hours.extend([18, 19, 20])
        
        if preferred_hours:
            # Find preferred hour within window
            for hour in preferred_hours:
                candidate = earliest.replace(hour=hour, minute=0, second=0)
                if earliest <= candidate <= latest:
                    return candidate
        
        return None
    
    async def _determine_intervention_context(self, seeker: SupportSeeker, 
                                            offer: SupportOffer,
                                            context: Dict[str, Any]) -> InterventionContext:
        """Determine the appropriate context for intervention"""
        
        # Crisis always uses crisis protocol
        if seeker.urgency == SupportUrgency.CRISIS:
            return InterventionContext.CRISIS_PROTOCOL
        
        # Check communication preferences
        seeker_prefs = seeker.communication_preferences
        offer_prefs = offer.communication_preferences
        
        # Find common preferred communication method
        common_prefs = set(seeker_prefs) & set(offer_prefs)
        
        if 'private_message' in common_prefs:
            return InterventionContext.PRIVATE_MESSAGE
        elif 'public_reply' in common_prefs:
            return InterventionContext.PUBLIC_POST
        elif 'community_space' in common_prefs:
            return InterventionContext.COMMUNITY_SPACE
        
        # Default to private for safety
        return InterventionContext.PRIVATE_MESSAGE
    
    async def _apply_cultural_timing_adjustments(self, seeker: SupportSeeker, 
                                               offer: SupportOffer, context: Dict[str, Any],
                                               earliest: datetime, latest: datetime) -> Dict[str, Any]:
        """Apply cultural considerations to timing"""
        
        adjustments = {
            'earliest_time': earliest,
            'latest_time': latest,
            'optimal_time': earliest + (latest - earliest) / 2
        }
        
        # Get cultural context
        cultural_context = context.get('cultural_context', 'individualistic_cultures')
        cultural_rules = self.cultural_considerations.get(cultural_context, {})
        
        # Apply cultural timing preferences
        timing_prefs = cultural_rules.get('timing_preferences', [])
        
        if 'relationship_context' in timing_prefs:
            # Prefer timing that builds on existing relationship context
            # In a real system, this would check for prior interactions
            pass
        
        if 'natural_opportunities' in timing_prefs:
            # Prefer timing that feels natural rather than algorithmic
            # Add some randomness within the window
            import random
            window_duration = latest - earliest
            random_offset = timedelta(seconds=random.randint(0, int(window_duration.total_seconds() / 4)))
            adjustments['optimal_time'] = earliest + random_offset
        
        return adjustments
    
    async def _check_prerequisites(self, seeker: SupportSeeker, offer: SupportOffer,
                                 context: Dict[str, Any]) -> List[str]:
        """Check prerequisites that must be met before intervention"""
        
        prerequisites = []
        urgency_rules = self.timing_rules[seeker.urgency]
        
        # Check required context
        for requirement in urgency_rules.get('context_requirements', []):
            if requirement == 'crisis_protocol_active':
                if not context.get('crisis_protocol_enabled', False):
                    prerequisites.append('Activate crisis response protocol')
            
            elif requirement == 'consent_verified':
                if not context.get('consent_verified', False):
                    prerequisites.append('Verify user consent for contact')
            
            elif requirement == 'explicit_consent':
                if not context.get('explicit_consent', False):
                    prerequisites.append('Obtain explicit consent for intervention')
            
            elif requirement == 'appropriate_context':
                if not context.get('appropriate_context', False):
                    prerequisites.append('Ensure appropriate social context')
            
            elif requirement == 'user_comfort':
                if not context.get('user_comfort_verified', False):
                    prerequisites.append('Verify user comfort with intervention')
            
            elif requirement == 'natural_opportunity':
                if not context.get('natural_opportunity', False):
                    prerequisites.append('Wait for natural intervention opportunity')
            
            elif requirement == 'established_relationship':
                if not context.get('relationship_exists', False):
                    prerequisites.append('Establish rapport before intervention')
        
        return prerequisites
    
    async def _identify_considerations(self, seeker: SupportSeeker, offer: SupportOffer,
                                     context: Dict[str, Any]) -> List[str]:
        """Identify important considerations for the intervention"""
        
        considerations = []
        
        # Privacy considerations
        if seeker.anonymity_preference == 'fully_anonymous':
            considerations.append('Maintain full anonymity of support seeker')
        elif seeker.anonymity_preference == 'partial_anonymous':
            considerations.append('Protect identifying details while enabling support')
        
        # Cultural considerations
        cultural_context = context.get('cultural_context')
        if cultural_context in self.cultural_considerations:
            cultural_rules = self.cultural_considerations[cultural_context]
            
            if cultural_rules.get('indirect_approach_preferred'):
                considerations.append('Use indirect, culturally appropriate approach')
            
            if cultural_rules.get('relationship_building_first'):
                considerations.append('Focus on relationship building before advice')
            
            if cultural_rules.get('community_context_important'):
                considerations.append('Consider community and family context')
        
        # Trigger warnings
        if seeker.trigger_warnings:
            considerations.append(f"Be mindful of trigger warnings: {', '.join(seeker.trigger_warnings)}")
        
        # Support type considerations
        if SupportType.CRISIS_INTERVENTION in seeker.support_types_needed:
            considerations.append('Crisis intervention protocols required')
        
        if SupportType.PROFESSIONAL_RESOURCES in seeker.support_types_needed:
            considerations.append('Professional referrals may be needed')
        
        return considerations
    
    async def _identify_escalation_triggers(self, seeker: SupportSeeker, 
                                          context: Dict[str, Any]) -> List[str]:
        """Identify triggers that would require escalating intervention urgency"""
        
        triggers = []
        
        # General escalation triggers
        triggers.extend([
            'No response to initial outreach within window',
            'Indication of worsening crisis',
            'Request for immediate professional help',
            'Safety concerns expressed'
        ])
        
        # Urgency-specific triggers
        if seeker.urgency != SupportUrgency.CRISIS:
            triggers.extend([
                'Crisis language detected in follow-up',
                'Explicit mention of self-harm',
                'Request for emergency intervention'
            ])
        
        # Context-specific triggers
        if context.get('first_time_seeker', False):
            triggers.append('Overwhelmed by support outreach')
        
        return triggers
    
    async def _calculate_timing_confidence(self, seeker: SupportSeeker, offer: SupportOffer,
                                         context: Dict[str, Any], 
                                         prerequisites: List[str]) -> float:
        """Calculate confidence in the timing recommendation"""
        
        confidence = 1.0
        
        # Reduce confidence for unmet prerequisites
        confidence -= len(prerequisites) * 0.1
        
        # Reduce confidence for mismatched communication preferences
        seeker_prefs = set(seeker.communication_preferences)
        offer_prefs = set(offer.communication_preferences)
        if not (seeker_prefs & offer_prefs):
            confidence -= 0.2
        
        # Reduce confidence for timezone mismatches
        if seeker.timezone != offer.timezone:
            confidence -= 0.1
        
        # Increase confidence for crisis situations (clear protocols)
        if seeker.urgency == SupportUrgency.CRISIS:
            confidence += 0.2
        
        # Reduce confidence for first-time seekers (more uncertainty)
        if context.get('first_time_seeker', False):
            confidence -= 0.1
        
        return max(0.1, min(1.0, confidence))
    
    async def _generate_timing_reasoning(self, seeker: SupportSeeker, offer: SupportOffer,
                                       base_rules: Dict[str, Any], 
                                       cultural_adjustments: Dict[str, Any],
                                       context: Dict[str, Any]) -> str:
        """Generate human-readable reasoning for timing decision"""
        
        reasoning_parts = []
        
        # Base urgency reasoning
        urgency_reasoning = {
            SupportUrgency.CRISIS: "Crisis situation requires immediate intervention",
            SupportUrgency.HIGH: "High urgency requires prompt but thoughtful response",
            SupportUrgency.MODERATE: "Moderate need allows time for appropriate context",
            SupportUrgency.LOW: "Low urgency enables patient, relationship-building approach",
            SupportUrgency.ONGOING: "Ongoing support focuses on sustainable connection"
        }
        
        reasoning_parts.append(urgency_reasoning[seeker.urgency])
        
        # Cultural considerations
        cultural_context = context.get('cultural_context')
        if cultural_context and cultural_context in self.cultural_considerations:
            reasoning_parts.append(f"Cultural context ({cultural_context}) considered")
        
        # Communication preferences
        common_prefs = set(seeker.communication_preferences) & set(offer.communication_preferences)
        if common_prefs:
            reasoning_parts.append(f"Matched communication preferences: {', '.join(common_prefs)}")
        
        # Timing constraints
        if base_rules.get('allowed_times') and 'any_time' not in base_rules['allowed_times']:
            reasoning_parts.append(f"Timing respects constraints: {', '.join(base_rules['allowed_times'])}")
        
        return '; '.join(reasoning_parts)
    
    async def _get_fallback_timing(self, seeker: SupportSeeker, 
                                 context: Dict[str, Any]) -> InterventionTiming:
        """Get safe fallback timing when calculation fails"""
        
        current_time = context.get('current_time', datetime.utcnow())
        
        # Safe defaults based on urgency
        if seeker.urgency == SupportUrgency.CRISIS:
            timing_window = TimingWindow.IMMEDIATE
            earliest = current_time
            latest = current_time + timedelta(minutes=5)
        else:
            timing_window = TimingWindow.ROUTINE
            earliest = current_time + timedelta(hours=1)
            latest = current_time + timedelta(hours=24)
        
        optimal = earliest + (latest - earliest) / 2
        
        return InterventionTiming(
            intervention_type=InterventionType.GENTLE_OUTREACH,
            timing_window=timing_window,
            earliest_time=earliest,
            latest_time=latest,
            optimal_time=optimal,
            context=InterventionContext.PRIVATE_MESSAGE,
            confidence_score=0.3,  # Low confidence for fallback
            reasoning="Fallback timing due to calculation error",
            prerequisites=['verify_user_consent'],
            considerations=['exercise_extra_caution'],
            escalation_triggers=['no_response_within_window']
        )
    
    async def _record_intervention_timing(self, user_id: str, 
                                        timing: InterventionTiming) -> None:
        """Record intervention timing for learning and analysis"""
        
        if user_id not in self.intervention_history:
            self.intervention_history[user_id] = []
        
        self.intervention_history[user_id].append(timing)
        
        # Keep only recent history
        max_history = 50
        if len(self.intervention_history[user_id]) > max_history:
            self.intervention_history[user_id] = self.intervention_history[user_id][-max_history:]
    
    async def evaluate_intervention_effectiveness(self, user_id: str, 
                                                timing: InterventionTiming,
                                                outcome: Dict[str, Any]) -> float:
        """Evaluate how effective the intervention timing was"""
        
        effectiveness_score = 0.5  # Base score
        
        # Positive outcome indicators
        if outcome.get('user_responded', False):
            effectiveness_score += 0.3
        
        if outcome.get('user_felt_supported', False):
            effectiveness_score += 0.2
        
        if outcome.get('connection_established', False):
            effectiveness_score += 0.2
        
        if outcome.get('crisis_resolved', False):
            effectiveness_score += 0.3
        
        # Negative outcome indicators
        if outcome.get('user_felt_intruded_upon', False):
            effectiveness_score -= 0.4
        
        if outcome.get('timing_inappropriate', False):
            effectiveness_score -= 0.3
        
        if outcome.get('cultural_insensitivity', False):
            effectiveness_score -= 0.5
        
        # Timing-specific factors
        actual_intervention_time = outcome.get('actual_intervention_time')
        if actual_intervention_time:
            if timing.is_within_window(actual_intervention_time):
                effectiveness_score += 0.1
            else:
                effectiveness_score -= 0.2
        
        effectiveness_score = max(0.0, min(1.0, effectiveness_score))
        
        logger.info(f"Intervention effectiveness for {user_id[:8]}...: {effectiveness_score:.2f}")
        return effectiveness_score
    
    def get_timing_analytics(self) -> Dict[str, Any]:
        """Get analytics about intervention timing patterns"""
        
        total_interventions = sum(len(history) for history in self.intervention_history.values())
        
        if total_interventions == 0:
            return {'no_data': True}
        
        # Analyze timing patterns
        intervention_types = {}
        timing_windows = {}
        confidence_scores = []
        
        for user_history in self.intervention_history.values():
            for timing in user_history:
                # Count intervention types
                intervention_types[timing.intervention_type.value] = \
                    intervention_types.get(timing.intervention_type.value, 0) + 1
                
                # Count timing windows
                timing_windows[timing.timing_window.value] = \
                    timing_windows.get(timing.timing_window.value, 0) + 1
                
                # Collect confidence scores
                confidence_scores.append(timing.confidence_score)
        
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        
        return {
            'total_interventions_timed': total_interventions,
            'unique_users': len(self.intervention_history),
            'intervention_type_distribution': intervention_types,
            'timing_window_distribution': timing_windows,
            'average_confidence': round(avg_confidence, 3),
            'timing_engine_health': {
                'rules_loaded': len(self.timing_rules) > 0,
                'cultural_considerations_active': len(self.cultural_considerations) > 0,
                'crisis_protocols_ready': len(self.crisis_protocols) > 0,
                'boundary_rules_enforced': len(self.boundary_rules) > 0
            }
        }