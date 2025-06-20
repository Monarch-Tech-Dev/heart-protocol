"""
Optimal Timing Engine for Daily Gentle Reminders

Determines the best times to deliver affirmations for maximum positive impact.
Based on chronobiology, psychology research, and individual user patterns.
Prioritizes user wellbeing over engagement metrics.
"""

import asyncio
import pytz
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta, time
from enum import Enum
import logging
import json
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class TimeOfDay(Enum):
    """Times of day with different psychological characteristics"""
    EARLY_MORNING = "early_morning"      # 5:00-7:00 AM - Fresh start energy
    MORNING = "morning"                  # 7:00-10:00 AM - Peak alertness
    LATE_MORNING = "late_morning"        # 10:00-12:00 PM - High productivity
    EARLY_AFTERNOON = "early_afternoon"  # 12:00-2:00 PM - Post-lunch dip
    AFTERNOON = "afternoon"              # 2:00-5:00 PM - Second peak
    EVENING = "evening"                  # 5:00-8:00 PM - Wind down
    NIGHT = "night"                      # 8:00-10:00 PM - Reflection time
    LATE_NIGHT = "late_night"           # 10:00 PM+ - Rest preparation


class TimingStrategy(Enum):
    """Different strategies for timing affirmations"""
    PREVENTIVE = "preventive"           # Before predicted difficult times
    RESPONSIVE = "responsive"           # After detecting distress
    ROUTINE = "routine"                 # Regular scheduled times
    ADAPTIVE = "adaptive"               # Based on user patterns
    CRISIS_RESPONSIVE = "crisis_responsive"  # Immediate crisis support


class MoodPattern(Enum):
    """Common mood patterns throughout the day"""
    MORNING_PERSON = "morning_person"       # Peak mood in morning
    EVENING_PERSON = "evening_person"       # Peak mood in evening
    STEADY_STATE = "steady_state"           # Consistent throughout day
    AFTERNOON_DIP = "afternoon_dip"         # Energy drop mid-day
    VARIABLE = "variable"                   # Unpredictable patterns


@dataclass
class TimingWindow:
    """Represents an optimal timing window for affirmations"""
    start_time: time
    end_time: time
    effectiveness_score: float  # 0.0 to 1.0
    reasoning: str
    priority_level: int  # 1-5, 5 being highest priority
    user_receptivity: float  # How receptive user typically is during this time
    avoid_if: List[str] = None  # Conditions to avoid this window
    
    def __post_init__(self):
        if self.avoid_if is None:
            self.avoid_if = []


class OptimalTimingEngine:
    """
    Determines optimal timing for delivering gentle reminders based on:
    - Individual user patterns and preferences
    - Chronobiology and circadian rhythm research
    - Psychological states throughout the day
    - Crisis detection and responsive delivery
    - Cultural considerations for timing
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.user_timing_profiles = {}  # In production, database-backed
        
        # Initialize timing research and patterns
        self.chronobiology_patterns = self._initialize_chronobiology_patterns()
        self.crisis_timing_rules = self._initialize_crisis_timing_rules()
        self.cultural_timing_preferences = self._initialize_cultural_timing()
        
        logger.info("Optimal Timing Engine initialized")
    
    def _initialize_chronobiology_patterns(self) -> Dict[TimeOfDay, Dict[str, Any]]:
        """
        Initialize timing patterns based on chronobiology research.
        
        Based on research about cortisol rhythms, circadian biology,
        and optimal times for different types of psychological interventions.
        """
        return {
            TimeOfDay.EARLY_MORNING: {
                'cortisol_level': 'rising',
                'alertness': 'increasing',
                'receptivity_to_affirmations': 0.7,
                'best_for': ['hope', 'new_beginnings', 'energy'],
                'avoid_for': ['heavy_emotional_content', 'complex_processing'],
                'typical_mood': 'neutral_to_positive',
                'effectiveness_multiplier': 0.8
            },
            
            TimeOfDay.MORNING: {
                'cortisol_level': 'peak',
                'alertness': 'peak',
                'receptivity_to_affirmations': 0.9,
                'best_for': ['strength', 'motivation', 'daily_intention'],
                'avoid_for': ['relaxation', 'vulnerability'],
                'typical_mood': 'energetic',
                'effectiveness_multiplier': 1.0
            },
            
            TimeOfDay.LATE_MORNING: {
                'cortisol_level': 'declining',
                'alertness': 'high',
                'receptivity_to_affirmations': 0.8,
                'best_for': ['progress', 'achievement', 'capability'],
                'avoid_for': ['rest', 'emotional_processing'],
                'typical_mood': 'productive',
                'effectiveness_multiplier': 0.9
            },
            
            TimeOfDay.EARLY_AFTERNOON: {
                'cortisol_level': 'low',
                'alertness': 'dipping',
                'receptivity_to_affirmations': 0.6,
                'best_for': ['self_compassion', 'patience', 'gentleness'],
                'avoid_for': ['high_energy', 'major_decisions'],
                'typical_mood': 'tired',
                'effectiveness_multiplier': 0.7
            },
            
            TimeOfDay.AFTERNOON: {
                'cortisol_level': 'moderate',
                'alertness': 'recovering',
                'receptivity_to_affirmations': 0.8,
                'best_for': ['perseverance', 'second_chances', 'renewal'],
                'avoid_for': ['new_concepts', 'overwhelm'],
                'typical_mood': 'recovering',
                'effectiveness_multiplier': 0.8
            },
            
            TimeOfDay.EVENING: {
                'cortisol_level': 'declining',
                'alertness': 'moderate',
                'receptivity_to_affirmations': 0.9,
                'best_for': ['reflection', 'gratitude', 'connection'],
                'avoid_for': ['stimulating_content', 'stress'],
                'typical_mood': 'reflective',
                'effectiveness_multiplier': 1.0
            },
            
            TimeOfDay.NIGHT: {
                'cortisol_level': 'low',
                'alertness': 'declining',
                'receptivity_to_affirmations': 0.8,
                'best_for': ['peace', 'self_acceptance', 'healing'],
                'avoid_for': ['activation', 'problem_solving'],
                'typical_mood': 'contemplative',
                'effectiveness_multiplier': 0.9
            },
            
            TimeOfDay.LATE_NIGHT: {
                'cortisol_level': 'lowest',
                'alertness': 'low',
                'receptivity_to_affirmations': 0.5,
                'best_for': ['comfort', 'safety', 'rest'],
                'avoid_for': ['complex_emotions', 'major_realizations'],
                'typical_mood': 'vulnerable',
                'effectiveness_multiplier': 0.6
            }
        }
    
    def _initialize_crisis_timing_rules(self) -> Dict[str, Any]:
        """Initialize timing rules for crisis situations"""
        return {
            'immediate_response_window': timedelta(minutes=5),
            'follow_up_windows': [
                timedelta(hours=1),
                timedelta(hours=6),
                timedelta(hours=24)
            ],
            'safe_hours': {
                'start': time(6, 0),  # 6 AM
                'end': time(23, 0)    # 11 PM
            },
            'overnight_crisis_protocol': {
                'immediate_comfort': True,
                'gentle_reminder_only': True,
                'human_handoff_priority': True
            },
            'max_crisis_reminders_per_hour': 2,
            'crisis_spacing_minutes': 30
        }
    
    def _initialize_cultural_timing(self) -> Dict[str, Dict[str, Any]]:
        """Initialize cultural considerations for timing"""
        return {
            'western_individualistic': {
                'preferred_morning_start': time(7, 0),
                'work_day_consideration': True,
                'weekend_preference': 'later_start',
                'meal_times': [time(7, 0), time(12, 0), time(18, 0)]
            },
            
            'collectivist': {
                'family_time_consideration': True,
                'meal_times': [time(6, 30), time(12, 0), time(19, 0)],
                'respect_quiet_hours': True,
                'community_schedule_awareness': True
            },
            
            'spiritual_practices': {
                'prayer_times_consideration': True,
                'meditation_windows': [time(6, 0), time(18, 0)],
                'sabbath_awareness': True,
                'religious_calendar_integration': True
            },
            
            'work_intensive_cultures': {
                'avoid_peak_work_hours': True,
                'lunch_break_optimization': True,
                'commute_time_utilization': True,
                'weekend_recovery_respect': True
            }
        }
    
    async def calculate_optimal_timing(self, user_id: str, 
                                     user_context: Dict[str, Any]) -> List[TimingWindow]:
        """
        Calculate optimal timing windows for delivering affirmations to a user.
        
        Args:
            user_id: User identifier
            user_context: Contains timezone, emotional_capacity, preferences, etc.
        """
        try:
            # Get user's timing profile
            timing_profile = await self._get_user_timing_profile(user_id)
            
            # Get current user state
            current_state = await self._assess_current_user_state(user_context)
            
            # Handle crisis situations with immediate timing
            if current_state.get('crisis_indicators', False):
                return await self._calculate_crisis_timing(user_context, timing_profile)
            
            # Calculate based on user patterns and chronobiology
            optimal_windows = await self._calculate_regular_timing(
                user_context, timing_profile, current_state
            )
            
            # Apply cultural considerations
            culturally_adjusted_windows = await self._apply_cultural_timing_adjustments(
                optimal_windows, user_context
            )
            
            # Sort by effectiveness and return top windows
            sorted_windows = sorted(
                culturally_adjusted_windows, 
                key=lambda w: w.effectiveness_score * w.user_receptivity,
                reverse=True
            )
            
            logger.debug(f"Calculated {len(sorted_windows)} optimal timing windows for user")
            return sorted_windows[:5]  # Return top 5 windows
            
        except Exception as e:
            logger.error(f"Error calculating optimal timing: {e}")
            return await self._get_fallback_timing(user_context)
    
    async def _get_user_timing_profile(self, user_id: str) -> Dict[str, Any]:
        """Get or create user's timing profile"""
        if user_id not in self.user_timing_profiles:
            # Initialize with defaults
            self.user_timing_profiles[user_id] = {
                'mood_pattern': MoodPattern.STEADY_STATE,
                'preferred_times': [],
                'avoided_times': [],
                'timezone': 'UTC',
                'weekend_preferences': {},
                'response_history': [],
                'effectiveness_by_time': {},
                'last_updated': datetime.utcnow()
            }
        
        return self.user_timing_profiles[user_id]
    
    async def _assess_current_user_state(self, user_context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess user's current state for timing decisions"""
        return {
            'emotional_capacity': user_context.get('emotional_capacity', {}),
            'crisis_indicators': user_context.get('crisis_indicators', False),
            'stress_level': user_context.get('stress_level', 0.5),
            'time_since_last_affirmation': user_context.get('time_since_last_affirmation', timedelta(hours=24)),
            'current_activity': user_context.get('current_activity', 'unknown'),
            'recent_interactions': user_context.get('recent_interactions', []),
            'user_reported_state': user_context.get('user_reported_state', 'neutral')
        }
    
    async def _calculate_crisis_timing(self, user_context: Dict[str, Any], 
                                     timing_profile: Dict[str, Any]) -> List[TimingWindow]:
        """Calculate timing for crisis situations - immediate and gentle"""
        crisis_rules = self.crisis_timing_rules
        user_timezone = pytz.timezone(timing_profile.get('timezone', 'UTC'))
        current_time = datetime.now(user_timezone).time()
        
        # Immediate response window
        immediate_window = TimingWindow(
            start_time=current_time,
            end_time=(datetime.combine(datetime.today(), current_time) + 
                     crisis_rules['immediate_response_window']).time(),
            effectiveness_score=1.0,
            reasoning="Immediate crisis support response",
            priority_level=5,
            user_receptivity=1.0,
            avoid_if=[]
        )
        
        # Follow-up windows for continued support
        follow_up_windows = []
        for i, follow_up_delta in enumerate(crisis_rules['follow_up_windows']):
            follow_up_time = (datetime.combine(datetime.today(), current_time) + follow_up_delta).time()
            
            # Check if it's within safe hours or if overnight protocol applies
            safe_start = crisis_rules['safe_hours']['start']
            safe_end = crisis_rules['safe_hours']['end']
            
            if safe_start <= follow_up_time <= safe_end:
                effectiveness = 0.8 - (i * 0.1)  # Decreasing effectiveness over time
                reasoning = f"Crisis follow-up {i+1} - continued support"
            else:
                effectiveness = 0.6  # Overnight protocol
                reasoning = f"Overnight crisis support {i+1} - gentle comfort"
            
            follow_up_windows.append(TimingWindow(
                start_time=follow_up_time,
                end_time=(datetime.combine(datetime.today(), follow_up_time) + timedelta(minutes=30)).time(),
                effectiveness_score=effectiveness,
                reasoning=reasoning,
                priority_level=4 - i,
                user_receptivity=0.9 - (i * 0.1),
                avoid_if=['user_explicitly_requested_space']
            ))
        
        return [immediate_window] + follow_up_windows
    
    async def _calculate_regular_timing(self, user_context: Dict[str, Any],
                                      timing_profile: Dict[str, Any],
                                      current_state: Dict[str, Any]) -> List[TimingWindow]:
        """Calculate regular timing windows based on patterns and research"""
        
        windows = []
        user_timezone = pytz.timezone(timing_profile.get('timezone', 'UTC'))
        emotional_capacity = current_state['emotional_capacity']
        
        # Get user's mood pattern
        mood_pattern = timing_profile.get('mood_pattern', MoodPattern.STEADY_STATE)
        
        # Calculate effectiveness for each time of day
        for time_of_day, chrono_data in self.chronobiology_patterns.items():
            
            # Base effectiveness from chronobiology
            base_effectiveness = chrono_data['effectiveness_multiplier']
            
            # Adjust for user's mood pattern
            pattern_adjustment = await self._get_mood_pattern_adjustment(
                time_of_day, mood_pattern
            )
            
            # Adjust for emotional capacity
            capacity_adjustment = await self._get_capacity_timing_adjustment(
                time_of_day, emotional_capacity
            )
            
            # Adjust for user's historical preferences
            history_adjustment = await self._get_historical_effectiveness(
                timing_profile, time_of_day
            )
            
            # Calculate final effectiveness
            final_effectiveness = (
                base_effectiveness * 0.4 +
                pattern_adjustment * 0.3 +
                capacity_adjustment * 0.2 +
                history_adjustment * 0.1
            )
            
            # Create timing window
            if final_effectiveness > 0.3:  # Only include reasonably effective times
                start_time, end_time = self._get_time_window_for_period(time_of_day)
                
                windows.append(TimingWindow(
                    start_time=start_time,
                    end_time=end_time,
                    effectiveness_score=final_effectiveness,
                    reasoning=f"Optimal for {', '.join(chrono_data['best_for'])}",
                    priority_level=max(1, int(final_effectiveness * 5)),
                    user_receptivity=chrono_data['receptivity_to_affirmations'],
                    avoid_if=chrono_data['avoid_for']
                ))
        
        return windows
    
    async def _get_mood_pattern_adjustment(self, time_of_day: TimeOfDay, 
                                         mood_pattern: MoodPattern) -> float:
        """Adjust effectiveness based on user's mood pattern"""
        
        adjustments = {
            MoodPattern.MORNING_PERSON: {
                TimeOfDay.EARLY_MORNING: 1.2,
                TimeOfDay.MORNING: 1.3,
                TimeOfDay.LATE_MORNING: 1.1,
                TimeOfDay.AFTERNOON: 0.8,
                TimeOfDay.EVENING: 0.7,
                TimeOfDay.NIGHT: 0.6
            },
            
            MoodPattern.EVENING_PERSON: {
                TimeOfDay.EARLY_MORNING: 0.5,
                TimeOfDay.MORNING: 0.6,
                TimeOfDay.AFTERNOON: 0.9,
                TimeOfDay.EVENING: 1.3,
                TimeOfDay.NIGHT: 1.2,
                TimeOfDay.LATE_NIGHT: 1.1
            },
            
            MoodPattern.AFTERNOON_DIP: {
                TimeOfDay.EARLY_AFTERNOON: 0.5,  # Very low during dip
                TimeOfDay.AFTERNOON: 0.7,
                TimeOfDay.EVENING: 1.2,  # Recovery period
                TimeOfDay.MORNING: 1.1
            },
            
            MoodPattern.STEADY_STATE: {
                # No major adjustments - consistent throughout day
                time_period: 1.0 for time_period in TimeOfDay
            },
            
            MoodPattern.VARIABLE: {
                # Conservative approach - moderate effectiveness all times
                time_period: 0.8 for time_period in TimeOfDay
            }
        }
        
        pattern_adjustments = adjustments.get(mood_pattern, {})
        return pattern_adjustments.get(time_of_day, 1.0)
    
    async def _get_capacity_timing_adjustment(self, time_of_day: TimeOfDay, 
                                            emotional_capacity: Dict[str, Any]) -> float:
        """Adjust timing based on emotional capacity"""
        
        capacity_level = emotional_capacity.get('level', 'moderate')
        
        # Users with low capacity benefit from gentler times
        if capacity_level in ['very_low', 'low']:
            gentle_times = [TimeOfDay.EARLY_MORNING, TimeOfDay.EVENING, TimeOfDay.NIGHT]
            if time_of_day in gentle_times:
                return 1.2
            else:
                return 0.8
        
        # Users with high capacity can handle any time effectively
        elif capacity_level == 'high':
            return 1.1
        
        # Moderate capacity - standard timing
        return 1.0
    
    async def _get_historical_effectiveness(self, timing_profile: Dict[str, Any], 
                                          time_of_day: TimeOfDay) -> float:
        """Get historical effectiveness for this time period"""
        
        effectiveness_history = timing_profile.get('effectiveness_by_time', {})
        
        # If we have historical data, use it
        if time_of_day.value in effectiveness_history:
            return effectiveness_history[time_of_day.value]
        
        # Otherwise, return neutral
        return 0.7
    
    def _get_time_window_for_period(self, time_of_day: TimeOfDay) -> Tuple[time, time]:
        """Get start and end times for a time period"""
        
        time_windows = {
            TimeOfDay.EARLY_MORNING: (time(5, 0), time(7, 0)),
            TimeOfDay.MORNING: (time(7, 0), time(10, 0)),
            TimeOfDay.LATE_MORNING: (time(10, 0), time(12, 0)),
            TimeOfDay.EARLY_AFTERNOON: (time(12, 0), time(14, 0)),
            TimeOfDay.AFTERNOON: (time(14, 0), time(17, 0)),
            TimeOfDay.EVENING: (time(17, 0), time(20, 0)),
            TimeOfDay.NIGHT: (time(20, 0), time(22, 0)),
            TimeOfDay.LATE_NIGHT: (time(22, 0), time(23, 59))
        }
        
        return time_windows[time_of_day]
    
    async def _apply_cultural_timing_adjustments(self, windows: List[TimingWindow],
                                                user_context: Dict[str, Any]) -> List[TimingWindow]:
        """Apply cultural considerations to timing windows"""
        
        cultural_context = user_context.get('cultural_context', 'western_individualistic')
        cultural_prefs = self.cultural_timing_preferences.get(cultural_context, {})
        
        adjusted_windows = []
        
        for window in windows:
            # Check if this window conflicts with cultural preferences
            conflicts = False
            
            # Check meal times
            meal_times = cultural_prefs.get('meal_times', [])
            for meal_time in meal_times:
                if (meal_time >= window.start_time and meal_time <= window.end_time):
                    window.effectiveness_score *= 0.8  # Reduce effectiveness during meals
            
            # Check work considerations
            if cultural_prefs.get('avoid_peak_work_hours', False):
                work_hours = [time(9, 0), time(10, 0), time(11, 0), time(14, 0), time(15, 0), time(16, 0)]
                if any(work_time >= window.start_time and work_time <= window.end_time 
                       for work_time in work_hours):
                    window.effectiveness_score *= 0.7
            
            # Check quiet hours
            if cultural_prefs.get('respect_quiet_hours', False):
                quiet_start = time(22, 0)
                quiet_end = time(6, 0)
                if window.start_time >= quiet_start or window.end_time <= quiet_end:
                    window.effectiveness_score *= 0.6
            
            adjusted_windows.append(window)
        
        return adjusted_windows
    
    async def _get_fallback_timing(self, user_context: Dict[str, Any]) -> List[TimingWindow]:
        """Get safe fallback timing when calculation fails"""
        
        # Safe, universally appropriate times
        return [
            TimingWindow(
                start_time=time(8, 0),
                end_time=time(9, 0),
                effectiveness_score=0.7,
                reasoning="Safe morning timing",
                priority_level=3,
                user_receptivity=0.8
            ),
            TimingWindow(
                start_time=time(18, 0),
                end_time=time(19, 0),
                effectiveness_score=0.7,
                reasoning="Safe evening timing",
                priority_level=3,
                user_receptivity=0.8
            )
        ]
    
    async def record_timing_feedback(self, user_id: str, timing_window: TimingWindow,
                                   user_response: Dict[str, Any]) -> bool:
        """Record user response to timing for learning and improvement"""
        
        try:
            timing_profile = await self._get_user_timing_profile(user_id)
            
            # Record the feedback
            feedback_record = {
                'timestamp': datetime.utcnow().isoformat(),
                'time_of_day': timing_window.start_time.strftime('%H:%M'),
                'effectiveness_score': timing_window.effectiveness_score,
                'user_rating': user_response.get('rating', 3),
                'user_receptivity': user_response.get('receptivity', 0.5),
                'response_time': user_response.get('response_time_seconds', None),
                'emotional_state_before': user_response.get('emotional_state_before', 'neutral'),
                'emotional_state_after': user_response.get('emotional_state_after', 'neutral')
            }
            
            timing_profile['response_history'].append(feedback_record)
            
            # Update effectiveness scores
            time_period = self._get_time_period_for_time(timing_window.start_time)
            current_effectiveness = timing_profile['effectiveness_by_time'].get(time_period.value, 0.7)
            
            # Weighted average with new data
            new_effectiveness = (user_response.get('rating', 3) / 5.0)
            updated_effectiveness = (current_effectiveness * 0.8) + (new_effectiveness * 0.2)
            
            timing_profile['effectiveness_by_time'][time_period.value] = updated_effectiveness
            timing_profile['last_updated'] = datetime.utcnow()
            
            logger.info(f"Updated timing effectiveness for user {user_id[:8]}...")
            return True
            
        except Exception as e:
            logger.error(f"Error recording timing feedback: {e}")
            return False
    
    def _get_time_period_for_time(self, target_time: time) -> TimeOfDay:
        """Determine which time period a specific time falls into"""
        
        for time_of_day in TimeOfDay:
            start_time, end_time = self._get_time_window_for_period(time_of_day)
            if start_time <= target_time <= end_time:
                return time_of_day
        
        return TimeOfDay.NIGHT  # Default fallback
    
    async def get_next_optimal_delivery_time(self, user_id: str, 
                                           user_context: Dict[str, Any]) -> Optional[datetime]:
        """Get the next optimal time to deliver an affirmation"""
        
        try:
            optimal_windows = await self.calculate_optimal_timing(user_id, user_context)
            
            if not optimal_windows:
                return None
            
            # Get user timezone
            timing_profile = await self._get_user_timing_profile(user_id)
            user_timezone = pytz.timezone(timing_profile.get('timezone', 'UTC'))
            current_time = datetime.now(user_timezone)
            
            # Find the next window that's in the future
            for window in optimal_windows:
                # Convert window time to datetime
                today = current_time.date()
                window_datetime = datetime.combine(today, window.start_time)
                window_datetime = user_timezone.localize(window_datetime)
                
                # If window is today but already passed, try tomorrow
                if window_datetime <= current_time:
                    tomorrow = today + timedelta(days=1)
                    window_datetime = datetime.combine(tomorrow, window.start_time)
                    window_datetime = user_timezone.localize(window_datetime)
                
                return window_datetime
            
            # If no good windows found, default to tomorrow morning
            tomorrow_morning = datetime.combine(
                current_time.date() + timedelta(days=1),
                time(8, 0)
            )
            return user_timezone.localize(tomorrow_morning)
            
        except Exception as e:
            logger.error(f"Error calculating next optimal delivery time: {e}")
            return None
    
    def get_timing_analytics(self, user_id: str) -> Dict[str, Any]:
        """Get analytics about timing effectiveness for a user"""
        
        timing_profile = self.user_timing_profiles.get(user_id, {})
        response_history = timing_profile.get('response_history', [])
        
        if not response_history:
            return {'insufficient_data': True}
        
        # Calculate statistics
        total_responses = len(response_history)
        average_rating = sum(r.get('user_rating', 3) for r in response_history) / total_responses
        
        # Time of day effectiveness
        time_effectiveness = {}
        for time_period in TimeOfDay:
            period_responses = [
                r for r in response_history 
                if self._get_time_period_for_time(
                    datetime.strptime(r['time_of_day'], '%H:%M').time()
                ) == time_period
            ]
            
            if period_responses:
                avg_rating = sum(r.get('user_rating', 3) for r in period_responses) / len(period_responses)
                time_effectiveness[time_period.value] = {
                    'average_rating': avg_rating,
                    'response_count': len(period_responses),
                    'effectiveness': avg_rating / 5.0
                }
        
        return {
            'total_responses': total_responses,
            'average_rating': average_rating,
            'overall_effectiveness': average_rating / 5.0,
            'time_of_day_effectiveness': time_effectiveness,
            'most_effective_time': max(time_effectiveness.items(), 
                                     key=lambda x: x[1]['effectiveness'])[0] if time_effectiveness else None,
            'personalization_confidence': min(1.0, total_responses / 20),  # Confidence increases with data
            'recommendations': await self._generate_timing_recommendations(timing_profile)
        }
    
    async def _generate_timing_recommendations(self, timing_profile: Dict[str, Any]) -> List[str]:
        """Generate recommendations for improving timing effectiveness"""
        
        recommendations = []
        response_history = timing_profile.get('response_history', [])
        
        if len(response_history) < 5:
            recommendations.append("More interaction data would help optimize timing")
        
        effectiveness_by_time = timing_profile.get('effectiveness_by_time', {})
        
        if effectiveness_by_time:
            best_time = max(effectiveness_by_time.items(), key=lambda x: x[1])
            worst_time = min(effectiveness_by_time.items(), key=lambda x: x[1])
            
            if best_time[1] > 0.8:
                recommendations.append(f"Your most effective time is {best_time[0]} - consider focusing there")
            
            if worst_time[1] < 0.4:
                recommendations.append(f"Consider avoiding {worst_time[0]} for better effectiveness")
        
        return recommendations