"""
Wellbeing-Focused Performance Monitor

Performance monitoring system that tracks metrics important for user wellbeing
and emotional safety rather than just technical performance metrics.
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import logging
import statistics
import json
from collections import defaultdict, deque
import time

logger = logging.getLogger(__name__)


class WellbeingMetric(Enum):
    """Metrics focused on user wellbeing impact"""
    CARE_DELIVERY_TIME = "care_delivery_time"        # Time to deliver care/support
    CRISIS_RESPONSE_TIME = "crisis_response_time"    # Response time for crisis situations
    USER_OVERWHELM_RATE = "user_overwhelm_rate"      # Rate of user overwhelm incidents
    HEALING_SUPPORT_LATENCY = "healing_support_latency"  # Latency for healing-focused features
    SAFETY_ALERT_TIME = "safety_alert_time"          # Time to process safety alerts
    COMMUNITY_CONNECTION_TIME = "community_connection_time"  # Time to facilitate connections
    CELEBRATION_DELIVERY_TIME = "celebration_delivery_time"  # Time to deliver celebration content
    GENTLE_LOADING_EFFECTIVENESS = "gentle_loading_effectiveness"  # Effectiveness of gentle UX
    ACCESSIBILITY_RESPONSE_TIME = "accessibility_response_time"  # Accessibility feature response
    PRIVACY_OPERATION_TIME = "privacy_operation_time"   # Time for privacy-sensitive operations


class ResponsePriority(Enum):
    """Priority levels for response time monitoring"""
    CRISIS_IMMEDIATE = "crisis_immediate"     # <100ms - Crisis situations
    CARE_URGENT = "care_urgent"              # <500ms - Urgent care needs
    SUPPORT_HIGH = "support_high"            # <1000ms - Support seeking
    HEALING_NORMAL = "healing_normal"        # <2000ms - Healing journey support
    STABLE_STANDARD = "stable_standard"      # <3000ms - Stable user interactions
    LEARNING_GENTLE = "learning_gentle"      # <5000ms - Learning/growth activities


class AlertSeverity(Enum):
    """Severity levels for wellbeing-related performance alerts"""
    CRITICAL = "critical"    # Immediate threat to user wellbeing
    HIGH = "high"           # Significant impact on user experience
    MEDIUM = "medium"       # Moderate impact on wellbeing
    LOW = "low"            # Minor impact
    INFO = "info"          # Informational only


@dataclass
class WellbeingPerformanceThreshold:
    """Performance thresholds based on wellbeing impact"""
    metric: WellbeingMetric
    priority: ResponsePriority
    warning_threshold_ms: float
    critical_threshold_ms: float
    ideal_threshold_ms: float
    user_impact_description: str
    healing_impact: str
    
    
@dataclass
class PerformanceAlert:
    """Alert for wellbeing-impacting performance issues"""
    alert_id: str
    metric: WellbeingMetric
    severity: AlertSeverity
    current_value: float
    threshold_exceeded: float
    user_impact: str
    suggested_actions: List[str]
    timestamp: datetime
    affected_users: List[str]
    care_context_affected: str
    

@dataclass
class WellbeingPerformanceReport:
    """Report on performance from wellbeing perspective"""
    report_id: str
    time_period: str
    care_delivery_summary: Dict[str, float]
    crisis_response_effectiveness: Dict[str, Any]
    user_experience_metrics: Dict[str, float]
    healing_support_performance: Dict[str, Any]
    optimization_recommendations: List[str]
    wellbeing_impact_score: float
    generated_at: datetime


class PerformanceMonitor:
    """
    Performance monitoring system focused on user wellbeing and emotional safety.
    
    Core Principles:
    - User wellbeing impact is primary success metric
    - Crisis response time is most critical measurement
    - Performance degradation alerts based on care impact
    - Gentle monitoring that doesn't overwhelm users
    - Privacy-preserving performance tracking
    - Healing-focused optimization recommendations
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.thresholds = self._initialize_wellbeing_thresholds()
        self.active_alerts: List[PerformanceAlert] = []
        self.metrics_buffer: Dict[WellbeingMetric, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.user_experience_tracking: Dict[str, Dict[str, Any]] = {}
        self.crisis_response_tracking: Dict[str, Dict[str, Any]] = {}
        
        # Real-time monitoring state
        self.monitoring_active = False
        self.alert_callbacks: List[Callable] = []
        
    def _initialize_wellbeing_thresholds(self) -> Dict[WellbeingMetric, WellbeingPerformanceThreshold]:
        """Initialize performance thresholds based on wellbeing impact"""
        return {
            WellbeingMetric.CRISIS_RESPONSE_TIME: WellbeingPerformanceThreshold(
                metric=WellbeingMetric.CRISIS_RESPONSE_TIME,
                priority=ResponsePriority.CRISIS_IMMEDIATE,
                warning_threshold_ms=50.0,
                critical_threshold_ms=100.0,
                ideal_threshold_ms=25.0,
                user_impact_description="Delays in crisis response can be life-threatening",
                healing_impact="Critical for user safety and trust in the system"
            ),
            
            WellbeingMetric.CARE_DELIVERY_TIME: WellbeingPerformanceThreshold(
                metric=WellbeingMetric.CARE_DELIVERY_TIME,
                priority=ResponsePriority.CARE_URGENT,
                warning_threshold_ms=300.0,
                critical_threshold_ms=500.0,
                ideal_threshold_ms=150.0,
                user_impact_description="Slow care delivery reduces effectiveness of support",
                healing_impact="Timely care delivery enhances healing outcomes"
            ),
            
            WellbeingMetric.HEALING_SUPPORT_LATENCY: WellbeingPerformanceThreshold(
                metric=WellbeingMetric.HEALING_SUPPORT_LATENCY,
                priority=ResponsePriority.SUPPORT_HIGH,
                warning_threshold_ms=800.0,
                critical_threshold_ms=1000.0,
                ideal_threshold_ms=400.0,
                user_impact_description="Delays in healing support can interrupt flow states",
                healing_impact="Smooth healing support maintains user engagement"
            ),
            
            WellbeingMetric.SAFETY_ALERT_TIME: WellbeingPerformanceThreshold(
                metric=WellbeingMetric.SAFETY_ALERT_TIME,
                priority=ResponsePriority.CRISIS_IMMEDIATE,
                warning_threshold_ms=75.0,
                critical_threshold_ms=100.0,
                ideal_threshold_ms=30.0,
                user_impact_description="Safety alert delays can allow escalation of dangerous situations",
                healing_impact="Rapid safety alerts prevent harm and build trust"
            ),
            
            WellbeingMetric.COMMUNITY_CONNECTION_TIME: WellbeingPerformanceThreshold(
                metric=WellbeingMetric.COMMUNITY_CONNECTION_TIME,
                priority=ResponsePriority.SUPPORT_HIGH,
                warning_threshold_ms=1500.0,
                critical_threshold_ms=2000.0,
                ideal_threshold_ms=750.0,
                user_impact_description="Slow community connections reduce sense of belonging",
                healing_impact="Quick connections foster healing relationships"
            ),
            
            WellbeingMetric.CELEBRATION_DELIVERY_TIME: WellbeingPerformanceThreshold(
                metric=WellbeingMetric.CELEBRATION_DELIVERY_TIME,
                priority=ResponsePriority.HEALING_NORMAL,
                warning_threshold_ms=1000.0,
                critical_threshold_ms=1500.0,
                ideal_threshold_ms=300.0,
                user_impact_description="Delayed celebrations reduce positive reinforcement",
                healing_impact="Timely celebrations reinforce healing progress"
            ),
            
            WellbeingMetric.USER_OVERWHELM_RATE: WellbeingPerformanceThreshold(
                metric=WellbeingMetric.USER_OVERWHELM_RATE,
                priority=ResponsePriority.CARE_URGENT,
                warning_threshold_ms=0.05,  # 5% overwhelm rate
                critical_threshold_ms=0.10,  # 10% overwhelm rate
                ideal_threshold_ms=0.01,  # 1% overwhelm rate
                user_impact_description="High overwhelm rates indicate system is too demanding",
                healing_impact="Low overwhelm rates create safe, healing environment"
            ),
            
            WellbeingMetric.GENTLE_LOADING_EFFECTIVENESS: WellbeingPerformanceThreshold(
                metric=WellbeingMetric.GENTLE_LOADING_EFFECTIVENESS,
                priority=ResponsePriority.LEARNING_GENTLE,
                warning_threshold_ms=0.70,  # 70% effectiveness
                critical_threshold_ms=0.50,  # 50% effectiveness
                ideal_threshold_ms=0.90,  # 90% effectiveness
                user_impact_description="Poor gentle loading can overwhelm sensitive users",
                healing_impact="Effective gentle loading supports vulnerable users"
            ),
            
            WellbeingMetric.ACCESSIBILITY_RESPONSE_TIME: WellbeingPerformanceThreshold(
                metric=WellbeingMetric.ACCESSIBILITY_RESPONSE_TIME,
                priority=ResponsePriority.SUPPORT_HIGH,
                warning_threshold_ms=1200.0,
                critical_threshold_ms=1500.0,
                ideal_threshold_ms=600.0,
                user_impact_description="Slow accessibility features exclude vulnerable users",
                healing_impact="Fast accessibility ensures inclusive healing environment"
            ),
            
            WellbeingMetric.PRIVACY_OPERATION_TIME: WellbeingPerformanceThreshold(
                metric=WellbeingMetric.PRIVACY_OPERATION_TIME,
                priority=ResponsePriority.STABLE_STANDARD,
                warning_threshold_ms=2000.0,
                critical_threshold_ms=3000.0,
                ideal_threshold_ms=1000.0,
                user_impact_description="Slow privacy operations reduce user control and trust",
                healing_impact="Fast privacy controls empower user autonomy"
            )
        }
    
    async def start_monitoring(self):
        """Start real-time wellbeing-focused performance monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        logger.info("Started wellbeing-focused performance monitoring")
        
        # Start monitoring tasks
        asyncio.create_task(self._monitor_real_time_metrics())
        asyncio.create_task(self._analyze_user_experience_patterns())
        asyncio.create_task(self._check_wellbeing_thresholds())
    
    async def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring_active = False
        logger.info("Stopped wellbeing-focused performance monitoring")
    
    async def record_metric(self, metric: WellbeingMetric, value: float,
                           user_id: Optional[str] = None,
                           care_context: Optional[str] = None,
                           additional_data: Optional[Dict[str, Any]] = None):
        """Record a wellbeing-focused performance metric"""
        try:
            timestamp = datetime.utcnow()
            
            # Store metric in buffer
            metric_data = {
                'value': value,
                'timestamp': timestamp,
                'user_id': user_id,
                'care_context': care_context,
                'additional_data': additional_data or {}
            }
            
            self.metrics_buffer[metric].append(metric_data)
            
            # Check thresholds immediately for critical metrics
            if metric in [WellbeingMetric.CRISIS_RESPONSE_TIME, 
                         WellbeingMetric.SAFETY_ALERT_TIME]:
                await self._check_immediate_threshold(metric, value, user_id, care_context)
            
            # Track user experience
            if user_id:
                await self._update_user_experience_tracking(user_id, metric, value, care_context)
            
            logger.debug(f"Recorded {metric.value}: {value}ms for user {user_id}")
            
        except Exception as e:
            logger.error(f"Failed to record metric {metric.value}: {str(e)}")
    
    async def _check_immediate_threshold(self, metric: WellbeingMetric, value: float,
                                       user_id: Optional[str], care_context: Optional[str]):
        """Check critical thresholds immediately"""
        threshold = self.thresholds.get(metric)
        if not threshold:
            return
        
        severity = None
        if value >= threshold.critical_threshold_ms:
            severity = AlertSeverity.CRITICAL
        elif value >= threshold.warning_threshold_ms:
            severity = AlertSeverity.HIGH
        
        if severity:
            await self._create_alert(metric, value, threshold, severity, 
                                   user_id, care_context)
    
    async def _create_alert(self, metric: WellbeingMetric, value: float,
                           threshold: WellbeingPerformanceThreshold,
                           severity: AlertSeverity, user_id: Optional[str],
                           care_context: Optional[str]):
        """Create a wellbeing-focused performance alert"""
        alert = PerformanceAlert(
            alert_id=f"alert_{metric.value}_{datetime.utcnow().isoformat()}",
            metric=metric,
            severity=severity,
            current_value=value,
            threshold_exceeded=threshold.critical_threshold_ms if severity == AlertSeverity.CRITICAL else threshold.warning_threshold_ms,
            user_impact=threshold.user_impact_description,
            suggested_actions=self._get_suggested_actions(metric, severity),
            timestamp=datetime.utcnow(),
            affected_users=[user_id] if user_id else [],
            care_context_affected=care_context or "unknown"
        )
        
        self.active_alerts.append(alert)
        
        # Trigger alert callbacks
        for callback in self.alert_callbacks:
            try:
                await callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {str(e)}")
        
        logger.warning(f"Performance alert: {metric.value} = {value}ms exceeds {alert.threshold_exceeded}ms threshold")
    
    def _get_suggested_actions(self, metric: WellbeingMetric, severity: AlertSeverity) -> List[str]:
        """Get suggested actions for performance issues"""
        actions = {
            WellbeingMetric.CRISIS_RESPONSE_TIME: [
                "Scale up crisis response servers immediately",
                "Enable crisis cache bypassing",
                "Prioritize crisis requests in queue",
                "Alert on-call crisis response team"
            ],
            WellbeingMetric.CARE_DELIVERY_TIME: [
                "Optimize care content caching",
                "Scale care delivery services",
                "Review care algorithm efficiency",
                "Check for database bottlenecks"
            ],
            WellbeingMetric.HEALING_SUPPORT_LATENCY: [
                "Optimize healing content algorithms",
                "Pre-load commonly accessed healing resources",
                "Scale healing support infrastructure",
                "Review personalization algorithm performance"
            ],
            WellbeingMetric.SAFETY_ALERT_TIME: [
                "Prioritize safety monitoring processes",
                "Scale safety detection services",
                "Optimize safety pattern matching",
                "Review alert routing efficiency"
            ],
            WellbeingMetric.COMMUNITY_CONNECTION_TIME: [
                "Optimize community matching algorithms",
                "Pre-compute compatible user matches",
                "Scale community services",
                "Review connection validation performance"
            ]
        }
        
        return actions.get(metric, ["Review system performance", "Scale relevant services"])
    
    async def _monitor_real_time_metrics(self):
        """Monitor real-time wellbeing metrics"""
        while self.monitoring_active:
            try:
                # Analyze recent metrics for patterns
                for metric, buffer in self.metrics_buffer.items():
                    if len(buffer) >= 10:  # Need enough data points
                        recent_values = [item['value'] for item in list(buffer)[-10:]]
                        avg_value = statistics.mean(recent_values)
                        
                        threshold = self.thresholds.get(metric)
                        if threshold and avg_value >= threshold.warning_threshold_ms:
                            # Check if this is a sustained issue
                            sustained_threshold_violations = sum(
                                1 for value in recent_values 
                                if value >= threshold.warning_threshold_ms
                            )
                            
                            if sustained_threshold_violations >= 7:  # 70% of recent samples
                                await self._create_alert(
                                    metric, avg_value, threshold, 
                                    AlertSeverity.MEDIUM, None, "sustained_degradation"
                                )
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Real-time monitoring error: {str(e)}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _analyze_user_experience_patterns(self):
        """Analyze user experience patterns from wellbeing perspective"""
        while self.monitoring_active:
            try:
                # Analyze user experience patterns every 5 minutes
                await asyncio.sleep(300)
                
                # Identify users experiencing consistent performance issues
                problem_users = []
                for user_id, experience_data in self.user_experience_tracking.items():
                    if self._user_experiencing_performance_issues(experience_data):
                        problem_users.append(user_id)
                
                if problem_users:
                    logger.warning(f"Users experiencing performance issues: {len(problem_users)}")
                    # Could trigger user-specific optimizations here
                
            except Exception as e:
                logger.error(f"User experience analysis error: {str(e)}")
    
    def _user_experiencing_performance_issues(self, experience_data: Dict[str, Any]) -> bool:
        """Determine if user is experiencing performance issues affecting wellbeing"""
        # Check recent response times
        recent_times = experience_data.get('recent_response_times', [])
        if len(recent_times) >= 5:
            avg_time = statistics.mean(recent_times[-5:])
            # Consider user's care context for threshold
            care_context = experience_data.get('primary_care_context', 'stable')
            
            if care_context in ['crisis', 'support_seeking'] and avg_time > 500:
                return True
            elif care_context == 'healing_progress' and avg_time > 1000:
                return True
            elif avg_time > 2000:  # General threshold
                return True
        
        return False
    
    async def _update_user_experience_tracking(self, user_id: str, metric: WellbeingMetric,
                                             value: float, care_context: Optional[str]):
        """Update user experience tracking data"""
        if user_id not in self.user_experience_tracking:
            self.user_experience_tracking[user_id] = {
                'recent_response_times': deque(maxlen=20),
                'metric_history': defaultdict(list),
                'primary_care_context': care_context or 'stable',
                'last_updated': datetime.utcnow()
            }
        
        user_data = self.user_experience_tracking[user_id]
        
        # Update response times for overall experience tracking
        if metric in [WellbeingMetric.CARE_DELIVERY_TIME, WellbeingMetric.HEALING_SUPPORT_LATENCY,
                     WellbeingMetric.COMMUNITY_CONNECTION_TIME]:
            user_data['recent_response_times'].append(value)
        
        # Track specific metric history
        user_data['metric_history'][metric.value].append({
            'value': value,
            'timestamp': datetime.utcnow(),
            'care_context': care_context
        })
        
        # Update care context if provided
        if care_context:
            user_data['primary_care_context'] = care_context
        
        user_data['last_updated'] = datetime.utcnow()
    
    async def _check_wellbeing_thresholds(self):
        """Periodically check all wellbeing thresholds"""
        while self.monitoring_active:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Clean up old alerts (older than 1 hour)
                cutoff_time = datetime.utcnow() - timedelta(hours=1)
                self.active_alerts = [
                    alert for alert in self.active_alerts 
                    if alert.timestamp >= cutoff_time
                ]
                
                # Clean up old user experience data (older than 24 hours)
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                for user_id in list(self.user_experience_tracking.keys()):
                    if self.user_experience_tracking[user_id]['last_updated'] < cutoff_time:
                        del self.user_experience_tracking[user_id]
                
            except Exception as e:
                logger.error(f"Threshold checking error: {str(e)}")
    
    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]):
        """Add callback for performance alerts"""
        self.alert_callbacks.append(callback)
    
    async def get_wellbeing_performance_report(self, time_range_hours: int = 24) -> WellbeingPerformanceReport:
        """Generate comprehensive wellbeing-focused performance report"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=time_range_hours)
            
            # Analyze care delivery performance
            care_delivery_summary = {}
            for metric in [WellbeingMetric.CARE_DELIVERY_TIME, WellbeingMetric.HEALING_SUPPORT_LATENCY,
                          WellbeingMetric.COMMUNITY_CONNECTION_TIME]:
                recent_data = [
                    item for item in self.metrics_buffer[metric]
                    if item['timestamp'] >= cutoff_time
                ]
                if recent_data:
                    values = [item['value'] for item in recent_data]
                    care_delivery_summary[metric.value] = {
                        'average_ms': statistics.mean(values),
                        'median_ms': statistics.median(values),
                        'p95_ms': statistics.quantiles(values, n=20)[18] if len(values) >= 20 else max(values),
                        'count': len(values)
                    }
            
            # Analyze crisis response effectiveness
            crisis_data = [
                item for item in self.metrics_buffer[WellbeingMetric.CRISIS_RESPONSE_TIME]
                if item['timestamp'] >= cutoff_time
            ]
            crisis_response_effectiveness = {}
            if crisis_data:
                crisis_times = [item['value'] for item in crisis_data]
                crisis_response_effectiveness = {
                    'total_crisis_responses': len(crisis_times),
                    'average_response_time_ms': statistics.mean(crisis_times),
                    'responses_under_100ms': sum(1 for t in crisis_times if t < 100),
                    'responses_over_500ms': sum(1 for t in crisis_times if t > 500),
                    'effectiveness_score': sum(1 for t in crisis_times if t < 100) / len(crisis_times)
                }
            
            # Generate optimization recommendations
            recommendations = self._generate_optimization_recommendations(care_delivery_summary, crisis_response_effectiveness)
            
            # Calculate overall wellbeing impact score
            wellbeing_impact_score = self._calculate_wellbeing_impact_score(
                care_delivery_summary, crisis_response_effectiveness
            )
            
            return WellbeingPerformanceReport(
                report_id=f"wellbeing_report_{datetime.utcnow().isoformat()}",
                time_period=f"{time_range_hours} hours",
                care_delivery_summary=care_delivery_summary,
                crisis_response_effectiveness=crisis_response_effectiveness,
                user_experience_metrics={
                    'total_users_tracked': len(self.user_experience_tracking),
                    'users_with_performance_issues': sum(
                        1 for user_data in self.user_experience_tracking.values()
                        if self._user_experiencing_performance_issues(user_data)
                    )
                },
                healing_support_performance={
                    'active_alerts': len(self.active_alerts),
                    'critical_alerts': len([a for a in self.active_alerts if a.severity == AlertSeverity.CRITICAL])
                },
                optimization_recommendations=recommendations,
                wellbeing_impact_score=wellbeing_impact_score,
                generated_at=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Failed to generate wellbeing performance report: {str(e)}")
            return WellbeingPerformanceReport(
                report_id="error_report",
                time_period=f"{time_range_hours} hours",
                care_delivery_summary={},
                crisis_response_effectiveness={},
                user_experience_metrics={},
                healing_support_performance={},
                optimization_recommendations=[f"Error generating report: {str(e)}"],
                wellbeing_impact_score=0.0,
                generated_at=datetime.utcnow()
            )
    
    def _generate_optimization_recommendations(self, care_summary: Dict, crisis_summary: Dict) -> List[str]:
        """Generate optimization recommendations based on wellbeing metrics"""
        recommendations = []
        
        # Crisis response recommendations
        if crisis_summary.get('effectiveness_score', 1.0) < 0.9:
            recommendations.append("Crisis response times need improvement - consider dedicated crisis infrastructure")
        
        if crisis_summary.get('responses_over_500ms', 0) > 0:
            recommendations.append("Some crisis responses exceeded 500ms - investigate database bottlenecks")
        
        # Care delivery recommendations
        for metric, data in care_summary.items():
            if data.get('p95_ms', 0) > self.thresholds[WellbeingMetric(metric)].warning_threshold_ms:
                recommendations.append(f"P95 latency for {metric} exceeds wellbeing threshold - optimize this pathway")
        
        # General recommendations
        if len(self.active_alerts) > 5:
            recommendations.append("High number of active alerts - review system capacity and scaling")
        
        if not recommendations:
            recommendations.append("Performance is meeting wellbeing standards - continue monitoring")
        
        return recommendations
    
    def _calculate_wellbeing_impact_score(self, care_summary: Dict, crisis_summary: Dict) -> float:
        """Calculate overall wellbeing impact score (0-10)"""
        try:
            score = 10.0  # Start with perfect score
            
            # Crisis response impact (40% of score)
            crisis_effectiveness = crisis_summary.get('effectiveness_score', 1.0)
            score -= (1.0 - crisis_effectiveness) * 4.0
            
            # Care delivery impact (40% of score)
            care_scores = []
            for metric, data in care_summary.items():
                threshold = self.thresholds[WellbeingMetric(metric)]
                avg_time = data.get('average_ms', 0)
                if avg_time <= threshold.ideal_threshold_ms:
                    care_scores.append(1.0)
                elif avg_time <= threshold.warning_threshold_ms:
                    care_scores.append(0.8)
                elif avg_time <= threshold.critical_threshold_ms:
                    care_scores.append(0.5)
                else:
                    care_scores.append(0.2)
            
            if care_scores:
                avg_care_score = statistics.mean(care_scores)
                score -= (1.0 - avg_care_score) * 4.0
            
            # Alert impact (20% of score)
            critical_alerts = len([a for a in self.active_alerts if a.severity == AlertSeverity.CRITICAL])
            if critical_alerts > 0:
                score -= critical_alerts * 0.5
            
            return max(0.0, min(10.0, score))
            
        except Exception as e:
            logger.error(f"Failed to calculate wellbeing impact score: {str(e)}")
            return 5.0  # Return neutral score on error