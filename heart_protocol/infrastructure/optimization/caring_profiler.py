"""
Caring Performance Profiler

Performance profiling system that focuses on user wellbeing impact
rather than just technical performance metrics.
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import logging
import statistics
import json
import time
import traceback
from functools import wraps
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class ProfilerMode(Enum):
    """Profiling modes based on user wellbeing needs"""
    CRISIS_MONITORING = "crisis_monitoring"      # Intensive monitoring for crisis situations
    HEALING_TRACKING = "healing_tracking"       # Track healing journey performance
    GENTLE_PROFILING = "gentle_profiling"       # Minimal overhead for sensitive users
    COMMUNITY_INSIGHTS = "community_insights"   # Community-focused profiling
    DISABLED = "disabled"                       # No profiling for privacy


class WellbeingProfilerLevel(Enum):
    """Levels of profiling detail"""
    MINIMAL = "minimal"                         # Basic response time only
    STANDARD = "standard"                       # Response time + basic metrics
    DETAILED = "detailed"                       # Comprehensive metrics
    THERAPEUTIC = "therapeutic"                 # Includes wellbeing impact metrics


@dataclass
class WellbeingProfile:
    """Profile data focused on wellbeing impact"""
    profile_id: str
    operation: str
    user_id: Optional[str]
    care_context: str
    start_time: datetime
    end_time: Optional[datetime]
    duration_ms: Optional[float]
    wellbeing_impact_score: Optional[float]
    user_experience_rating: Optional[str]
    resource_efficiency: Optional[float]
    emotional_safety_maintained: bool
    healing_progress_supported: bool
    community_connection_facilitated: bool
    gentle_interaction_achieved: bool
    privacy_preserved: bool
    user_consent_respected: bool
    error_occurred: bool
    error_message: Optional[str]
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    therapeutic_notes: str = ""


@dataclass
class OperationPerformancePattern:
    """Pattern analysis for operation performance"""
    operation: str
    care_context: str
    total_executions: int
    average_duration_ms: float
    p50_duration_ms: float
    p95_duration_ms: float
    p99_duration_ms: float
    wellbeing_impact_average: float
    success_rate: float
    user_satisfaction_average: float
    healing_effectiveness_score: float
    optimization_recommendations: List[str]
    last_updated: datetime


class CaringProfiler:
    """
    Performance profiler that focuses on user wellbeing impact and emotional safety.
    
    Core Principles:
    - User wellbeing comes before profiling completeness
    - Crisis situations get priority profiling to ensure help delivery
    - Gentle profiling modes for sensitive or overwhelmed users
    - Profiling respects user privacy and consent preferences
    - Focus on healing outcomes rather than just technical metrics
    - Community care patterns identified and optimized
    - Therapeutic insights from performance data
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.profiler_active = True
        self.profiles: List[WellbeingProfile] = []
        self.active_profiles: Dict[str, WellbeingProfile] = {}
        self.user_profiler_preferences: Dict[str, Dict[str, Any]] = {}
        self.operation_patterns: Dict[str, OperationPerformancePattern] = {}
        
        # Profiling state
        self.wellbeing_callbacks: List[Callable] = []
        self.max_profiles_in_memory = 10000
        
        # Performance overhead tracking
        self.profiler_overhead_tracking = deque(maxlen=100)
        
    def get_user_profiler_mode(self, user_id: Optional[str], care_context: str) -> ProfilerMode:
        """Determine appropriate profiler mode for user"""
        if not user_id:
            return ProfilerMode.GENTLE_PROFILING
        
        user_prefs = self.user_profiler_preferences.get(user_id, {})
        
        # Respect user's profiling preference
        if 'profiler_mode' in user_prefs:
            return ProfilerMode(user_prefs['profiler_mode'])
        
        # Auto-select based on care context
        if care_context in ['immediate_crisis', 'crisis']:
            return ProfilerMode.CRISIS_MONITORING
        elif care_context in ['healing_progress', 'support_seeking']:
            return ProfilerMode.HEALING_TRACKING
        elif care_context in ['community_building']:
            return ProfilerMode.COMMUNITY_INSIGHTS
        else:
            return ProfilerMode.GENTLE_PROFILING
    
    def wellbeing_profile(self, operation: str, care_context: str = "general",
                         user_id: Optional[str] = None,
                         track_wellbeing_impact: bool = True):
        """
        Decorator for wellbeing-focused performance profiling
        
        Args:
            operation: Name of operation being profiled
            care_context: Care context for appropriate profiling level
            user_id: User ID if available
            track_wellbeing_impact: Whether to track wellbeing impact
        """
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await self._profile_async_operation(
                    func, operation, care_context, user_id, track_wellbeing_impact,
                    args, kwargs
                )
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                return self._profile_sync_operation(
                    func, operation, care_context, user_id, track_wellbeing_impact,
                    args, kwargs
                )
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator
    
    async def _profile_async_operation(self, func: Callable, operation: str,
                                     care_context: str, user_id: Optional[str],
                                     track_wellbeing_impact: bool,
                                     args: tuple, kwargs: dict) -> Any:
        """Profile async operation with wellbeing focus"""
        profiler_mode = self.get_user_profiler_mode(user_id, care_context)
        
        if profiler_mode == ProfilerMode.DISABLED:
            return await func(*args, **kwargs)
        
        profile_start = time.perf_counter()
        
        # Start profile
        profile = await self.start_wellbeing_profile(
            operation, user_id, care_context, profiler_mode
        )
        
        try:
            # Execute operation
            result = await func(*args, **kwargs)
            
            # End profile with success
            await self.end_wellbeing_profile(
                profile.profile_id, 
                wellbeing_impact_score=self._calculate_wellbeing_impact(result, care_context),
                error_occurred=False,
                track_wellbeing_impact=track_wellbeing_impact
            )
            
            return result
            
        except Exception as e:
            # End profile with error
            await self.end_wellbeing_profile(
                profile.profile_id,
                error_occurred=True,
                error_message=str(e),
                track_wellbeing_impact=track_wellbeing_impact
            )
            raise
        
        finally:
            # Track profiler overhead
            profiler_overhead = (time.perf_counter() - profile_start) * 1000
            self.profiler_overhead_tracking.append(profiler_overhead)
    
    def _profile_sync_operation(self, func: Callable, operation: str,
                              care_context: str, user_id: Optional[str],
                              track_wellbeing_impact: bool,
                              args: tuple, kwargs: dict) -> Any:
        """Profile sync operation with wellbeing focus"""
        profiler_mode = self.get_user_profiler_mode(user_id, care_context)
        
        if profiler_mode == ProfilerMode.DISABLED:
            return func(*args, **kwargs)
        
        profile_start = time.perf_counter()
        
        # Start profile (sync version)
        profile = self._start_wellbeing_profile_sync(
            operation, user_id, care_context, profiler_mode
        )
        
        try:
            # Execute operation
            result = func(*args, **kwargs)
            
            # End profile with success
            self._end_wellbeing_profile_sync(
                profile.profile_id,
                wellbeing_impact_score=self._calculate_wellbeing_impact(result, care_context),
                error_occurred=False,
                track_wellbeing_impact=track_wellbeing_impact
            )
            
            return result
            
        except Exception as e:
            # End profile with error
            self._end_wellbeing_profile_sync(
                profile.profile_id,
                error_occurred=True,
                error_message=str(e),
                track_wellbeing_impact=track_wellbeing_impact
            )
            raise
        
        finally:
            # Track profiler overhead
            profiler_overhead = (time.perf_counter() - profile_start) * 1000
            self.profiler_overhead_tracking.append(profiler_overhead)
    
    async def start_wellbeing_profile(self, operation: str, user_id: Optional[str],
                                    care_context: str, 
                                    profiler_mode: ProfilerMode) -> WellbeingProfile:
        """Start a wellbeing-focused performance profile"""
        profile_id = f"profile_{operation}_{datetime.utcnow().isoformat()}_{id(self)}"
        
        profile = WellbeingProfile(
            profile_id=profile_id,
            operation=operation,
            user_id=user_id,
            care_context=care_context,
            start_time=datetime.utcnow(),
            end_time=None,
            duration_ms=None,
            wellbeing_impact_score=None,
            user_experience_rating=None,
            resource_efficiency=None,
            emotional_safety_maintained=True,  # Assume true until proven otherwise
            healing_progress_supported=False,
            community_connection_facilitated=False,
            gentle_interaction_achieved=True,
            privacy_preserved=True,
            user_consent_respected=True,
            error_occurred=False,
            error_message=None,
            performance_metrics={
                'profiler_mode': profiler_mode.value,
                'start_timestamp': time.perf_counter()
            }
        )
        
        self.active_profiles[profile_id] = profile
        return profile
    
    def _start_wellbeing_profile_sync(self, operation: str, user_id: Optional[str],
                                    care_context: str, 
                                    profiler_mode: ProfilerMode) -> WellbeingProfile:
        """Sync version of start_wellbeing_profile"""
        profile_id = f"profile_{operation}_{datetime.utcnow().isoformat()}_{id(self)}"
        
        profile = WellbeingProfile(
            profile_id=profile_id,
            operation=operation,
            user_id=user_id,
            care_context=care_context,
            start_time=datetime.utcnow(),
            end_time=None,
            duration_ms=None,
            wellbeing_impact_score=None,
            user_experience_rating=None,
            resource_efficiency=None,
            emotional_safety_maintained=True,
            healing_progress_supported=False,
            community_connection_facilitated=False,
            gentle_interaction_achieved=True,
            privacy_preserved=True,
            user_consent_respected=True,
            error_occurred=False,
            error_message=None,
            performance_metrics={
                'profiler_mode': profiler_mode.value,
                'start_timestamp': time.perf_counter()
            }
        )
        
        self.active_profiles[profile_id] = profile
        return profile
    
    async def end_wellbeing_profile(self, profile_id: str,
                                  wellbeing_impact_score: Optional[float] = None,
                                  user_experience_rating: Optional[str] = None,
                                  error_occurred: bool = False,
                                  error_message: Optional[str] = None,
                                  track_wellbeing_impact: bool = True,
                                  additional_metrics: Optional[Dict[str, Any]] = None):
        """End a wellbeing profile and record results"""
        if profile_id not in self.active_profiles:
            logger.warning(f"Profile {profile_id} not found in active profiles")
            return
        
        profile = self.active_profiles[profile_id]
        end_time = datetime.utcnow()
        
        # Calculate duration
        start_timestamp = profile.performance_metrics.get('start_timestamp', time.perf_counter())
        duration_ms = (time.perf_counter() - start_timestamp) * 1000
        
        # Update profile
        profile.end_time = end_time
        profile.duration_ms = duration_ms
        profile.wellbeing_impact_score = wellbeing_impact_score
        profile.user_experience_rating = user_experience_rating
        profile.error_occurred = error_occurred
        profile.error_message = error_message
        
        # Add additional metrics
        if additional_metrics:
            profile.performance_metrics.update(additional_metrics)
        
        # Calculate wellbeing metrics
        self._calculate_wellbeing_metrics(profile)
        
        # Store completed profile
        self.profiles.append(profile)
        del self.active_profiles[profile_id]
        
        # Update operation patterns
        await self._update_operation_patterns(profile)
        
        # Trigger wellbeing callbacks
        if track_wellbeing_impact:
            for callback in self.wellbeing_callbacks:
                try:
                    await callback(profile)
                except Exception as e:
                    logger.error(f"Wellbeing callback failed: {str(e)}")
        
        # Manage memory usage
        if len(self.profiles) > self.max_profiles_in_memory:
            # Remove oldest profiles
            self.profiles = self.profiles[-self.max_profiles_in_memory:]
        
        logger.debug(f"Completed wellbeing profile for {profile.operation}: {duration_ms:.2f}ms")
    
    def _end_wellbeing_profile_sync(self, profile_id: str,
                                  wellbeing_impact_score: Optional[float] = None,
                                  user_experience_rating: Optional[str] = None,
                                  error_occurred: bool = False,
                                  error_message: Optional[str] = None,
                                  track_wellbeing_impact: bool = True,
                                  additional_metrics: Optional[Dict[str, Any]] = None):
        """Sync version of end_wellbeing_profile"""
        if profile_id not in self.active_profiles:
            logger.warning(f"Profile {profile_id} not found in active profiles")
            return
        
        profile = self.active_profiles[profile_id]
        end_time = datetime.utcnow()
        
        # Calculate duration
        start_timestamp = profile.performance_metrics.get('start_timestamp', time.perf_counter())
        duration_ms = (time.perf_counter() - start_timestamp) * 1000
        
        # Update profile
        profile.end_time = end_time
        profile.duration_ms = duration_ms
        profile.wellbeing_impact_score = wellbeing_impact_score
        profile.user_experience_rating = user_experience_rating
        profile.error_occurred = error_occurred
        profile.error_message = error_message
        
        # Add additional metrics
        if additional_metrics:
            profile.performance_metrics.update(additional_metrics)
        
        # Calculate wellbeing metrics
        self._calculate_wellbeing_metrics(profile)
        
        # Store completed profile
        self.profiles.append(profile)
        del self.active_profiles[profile_id]
        
        # Note: Async operations like updating patterns and callbacks are skipped in sync version
        
        # Manage memory usage
        if len(self.profiles) > self.max_profiles_in_memory:
            self.profiles = self.profiles[-self.max_profiles_in_memory:]
        
        logger.debug(f"Completed wellbeing profile for {profile.operation}: {duration_ms:.2f}ms")
    
    def _calculate_wellbeing_impact(self, result: Any, care_context: str) -> float:
        """Calculate wellbeing impact score based on operation result"""
        try:
            # Default wellbeing impact calculation
            base_score = 7.0  # Neutral positive impact
            
            # Adjust based on care context
            if care_context in ['immediate_crisis', 'crisis']:
                base_score = 9.0  # High impact for crisis help
            elif care_context in ['healing_progress']:
                base_score = 8.0  # High impact for healing
            elif care_context in ['support_seeking']:
                base_score = 7.5  # Good impact for support
            elif care_context in ['community_building']:
                base_score = 6.5  # Moderate impact for community
            
            # Check if result indicates success/failure
            if isinstance(result, dict):
                if result.get('error'):
                    base_score -= 3.0  # Reduce score for errors
                elif result.get('success'):
                    base_score += 1.0  # Boost score for explicit success
            
            return max(1.0, min(10.0, base_score))
            
        except Exception as e:
            logger.error(f"Failed to calculate wellbeing impact: {str(e)}")
            return 5.0  # Return neutral score on error
    
    def _calculate_wellbeing_metrics(self, profile: WellbeingProfile):
        """Calculate comprehensive wellbeing metrics for profile"""
        try:
            # Determine if healing progress was supported
            if profile.care_context in ['healing_progress', 'support_seeking']:
                profile.healing_progress_supported = not profile.error_occurred
            
            # Determine if community connection was facilitated
            if profile.care_context in ['community_building', 'connection_matching']:
                profile.community_connection_facilitated = not profile.error_occurred
            
            # Calculate resource efficiency
            if profile.duration_ms:
                # Simple efficiency based on duration vs. expected times
                expected_times = {
                    'immediate_crisis': 100,  # 100ms expected for crisis
                    'support_seeking': 500,   # 500ms expected for support
                    'healing_progress': 1000, # 1s expected for healing
                    'community_building': 1500, # 1.5s expected for community
                    'general': 2000           # 2s expected for general
                }
                
                expected_ms = expected_times.get(profile.care_context, 2000)
                profile.resource_efficiency = min(1.0, expected_ms / profile.duration_ms)
            
            # Check emotional safety
            if profile.error_occurred and profile.care_context in ['immediate_crisis', 'support_seeking']:
                profile.emotional_safety_maintained = False
            
            # Determine user experience rating
            if profile.wellbeing_impact_score:
                if profile.wellbeing_impact_score >= 8.0:
                    profile.user_experience_rating = "excellent"
                elif profile.wellbeing_impact_score >= 6.0:
                    profile.user_experience_rating = "good"
                elif profile.wellbeing_impact_score >= 4.0:
                    profile.user_experience_rating = "acceptable"
                else:
                    profile.user_experience_rating = "needs_improvement"
            
        except Exception as e:
            logger.error(f"Failed to calculate wellbeing metrics: {str(e)}")
    
    async def _update_operation_patterns(self, profile: WellbeingProfile):
        """Update operation performance patterns"""
        try:
            pattern_key = f"{profile.operation}_{profile.care_context}"
            
            if pattern_key not in self.operation_patterns:
                self.operation_patterns[pattern_key] = OperationPerformancePattern(
                    operation=profile.operation,
                    care_context=profile.care_context,
                    total_executions=0,
                    average_duration_ms=0.0,
                    p50_duration_ms=0.0,
                    p95_duration_ms=0.0,
                    p99_duration_ms=0.0,
                    wellbeing_impact_average=0.0,
                    success_rate=0.0,
                    user_satisfaction_average=0.0,
                    healing_effectiveness_score=0.0,
                    optimization_recommendations=[],
                    last_updated=datetime.utcnow()
                )
            
            pattern = self.operation_patterns[pattern_key]
            
            # Get recent profiles for this operation/context
            recent_profiles = [
                p for p in self.profiles
                if p.operation == profile.operation 
                and p.care_context == profile.care_context
                and p.end_time and p.end_time >= datetime.utcnow() - timedelta(hours=24)
                and p.duration_ms is not None
            ]
            
            if recent_profiles:
                durations = [p.duration_ms for p in recent_profiles]
                wellbeing_scores = [p.wellbeing_impact_score for p in recent_profiles if p.wellbeing_impact_score]
                
                # Update pattern statistics
                pattern.total_executions = len(recent_profiles)
                pattern.average_duration_ms = statistics.mean(durations)
                pattern.p50_duration_ms = statistics.median(durations)
                pattern.p95_duration_ms = statistics.quantiles(durations, n=20)[18] if len(durations) >= 20 else max(durations)
                pattern.p99_duration_ms = statistics.quantiles(durations, n=100)[98] if len(durations) >= 100 else max(durations)
                pattern.success_rate = sum(1 for p in recent_profiles if not p.error_occurred) / len(recent_profiles)
                
                if wellbeing_scores:
                    pattern.wellbeing_impact_average = statistics.mean(wellbeing_scores)
                
                # Calculate healing effectiveness
                healing_profiles = [p for p in recent_profiles if p.healing_progress_supported]
                pattern.healing_effectiveness_score = len(healing_profiles) / len(recent_profiles)
                
                # Generate optimization recommendations
                pattern.optimization_recommendations = self._generate_optimization_recommendations(pattern)
                pattern.last_updated = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Failed to update operation patterns: {str(e)}")
    
    def _generate_optimization_recommendations(self, pattern: OperationPerformancePattern) -> List[str]:
        """Generate optimization recommendations based on pattern analysis"""
        recommendations = []
        
        try:
            # Performance recommendations
            if pattern.p95_duration_ms > 2000:  # P95 > 2 seconds
                recommendations.append(f"P95 latency ({pattern.p95_duration_ms:.0f}ms) is high - consider optimization")
            
            if pattern.success_rate < 0.95:  # Less than 95% success rate
                recommendations.append(f"Success rate ({pattern.success_rate:.1%}) below target - investigate failures")
            
            if pattern.wellbeing_impact_average < 6.0:  # Low wellbeing impact
                recommendations.append("Low wellbeing impact - review operation effectiveness")
            
            # Care context specific recommendations
            if pattern.care_context == 'immediate_crisis':
                if pattern.average_duration_ms > 100:
                    recommendations.append("Crisis response time too slow - prioritize optimization")
            elif pattern.care_context == 'support_seeking':
                if pattern.average_duration_ms > 500:
                    recommendations.append("Support delivery time needs improvement")
            
            # Healing effectiveness recommendations
            if pattern.healing_effectiveness_score < 0.8:
                recommendations.append("Low healing effectiveness - review therapeutic approach")
            
            if not recommendations:
                recommendations.append("Performance meets wellbeing standards")
        
        except Exception as e:
            logger.error(f"Failed to generate optimization recommendations: {str(e)}")
            recommendations.append("Error generating recommendations")
        
        return recommendations
    
    def add_wellbeing_callback(self, callback: Callable[[WellbeingProfile], None]):
        """Add callback for wellbeing profile completion"""
        self.wellbeing_callbacks.append(callback)
    
    async def get_wellbeing_performance_insights(self, time_range_hours: int = 24) -> Dict[str, Any]:
        """Get insights focused on wellbeing performance"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=time_range_hours)
            
            recent_profiles = [
                p for p in self.profiles
                if p.end_time and p.end_time >= cutoff_time
            ]
            
            if not recent_profiles:
                return {'no_data': True, 'time_range_hours': time_range_hours}
            
            # Overall metrics
            total_operations = len(recent_profiles)
            success_rate = sum(1 for p in recent_profiles if not p.error_occurred) / total_operations
            
            # Wellbeing metrics
            wellbeing_scores = [p.wellbeing_impact_score for p in recent_profiles if p.wellbeing_impact_score]
            avg_wellbeing_impact = statistics.mean(wellbeing_scores) if wellbeing_scores else None
            
            # Care context analysis
            care_context_distribution = defaultdict(int)
            care_context_performance = defaultdict(list)
            
            for profile in recent_profiles:
                care_context_distribution[profile.care_context] += 1
                if profile.duration_ms:
                    care_context_performance[profile.care_context].append(profile.duration_ms)
            
            # Crisis response analysis
            crisis_profiles = [p for p in recent_profiles if p.care_context in ['immediate_crisis', 'crisis']]
            crisis_response_time = statistics.mean([p.duration_ms for p in crisis_profiles if p.duration_ms]) if crisis_profiles else None
            
            # Healing support analysis
            healing_profiles = [p for p in recent_profiles if p.healing_progress_supported]
            healing_support_rate = len(healing_profiles) / total_operations
            
            # Community impact analysis
            community_profiles = [p for p in recent_profiles if p.community_connection_facilitated]
            community_impact_rate = len(community_profiles) / total_operations
            
            # Performance overhead analysis
            profiler_overhead = statistics.mean(list(self.profiler_overhead_tracking)) if self.profiler_overhead_tracking else 0
            
            return {
                'time_range_hours': time_range_hours,
                'total_operations': total_operations,
                'overall_success_rate': success_rate,
                'average_wellbeing_impact': avg_wellbeing_impact,
                'care_context_distribution': dict(care_context_distribution),
                'crisis_response_time_ms': crisis_response_time,
                'healing_support_rate': healing_support_rate,
                'community_impact_rate': community_impact_rate,
                'profiler_overhead_ms': profiler_overhead,
                'care_context_performance': {
                    context: {
                        'average_ms': statistics.mean(times),
                        'median_ms': statistics.median(times),
                        'count': len(times)
                    }
                    for context, times in care_context_performance.items()
                    if times
                },
                'operation_patterns': {
                    key: {
                        'total_executions': pattern.total_executions,
                        'average_duration_ms': pattern.average_duration_ms,
                        'success_rate': pattern.success_rate,
                        'wellbeing_impact': pattern.wellbeing_impact_average,
                        'healing_effectiveness': pattern.healing_effectiveness_score,
                        'recommendations': pattern.optimization_recommendations
                    }
                    for key, pattern in self.operation_patterns.items()
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get wellbeing performance insights: {str(e)}")
            return {'error': str(e)}
    
    async def update_user_profiler_preferences(self, user_id: str, 
                                             preferences: Dict[str, Any]) -> bool:
        """Update user's profiler preferences"""
        try:
            valid_keys = [
                'profiler_mode', 'detailed_tracking_consent', 'wellbeing_metrics_sharing',
                'therapeutic_insights_enabled', 'performance_alerts_enabled',
                'privacy_level', 'data_retention_days'
            ]
            
            validated_prefs = {
                key: value for key, value in preferences.items()
                if key in valid_keys
            }
            
            # Validate profiler mode
            if 'profiler_mode' in validated_prefs:
                try:
                    ProfilerMode(validated_prefs['profiler_mode'])
                except ValueError:
                    del validated_prefs['profiler_mode']
            
            self.user_profiler_preferences[user_id] = validated_prefs
            
            logger.info(f"Updated profiler preferences for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update user profiler preferences: {str(e)}")
            return False
    
    def get_operation_recommendations(self, operation: str, care_context: str) -> List[str]:
        """Get optimization recommendations for specific operation"""
        pattern_key = f"{operation}_{care_context}"
        pattern = self.operation_patterns.get(pattern_key)
        
        if pattern:
            return pattern.optimization_recommendations
        else:
            return ["Insufficient data for recommendations - continue monitoring"]