"""
Prometheus Integration

Integration with Prometheus monitoring system specifically configured for
healing-focused metrics, wellbeing tracking, and care effectiveness measurement.
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import logging
import time
import threading
from collections import defaultdict, deque

try:
    from prometheus_client import (
        Counter, Gauge, Histogram, Summary, Info, Enum as PrometheusEnum,
        CollectorRegistry, generate_latest, push_to_gateway,
        start_http_server, MetricWrapperBase
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    # Graceful fallback if prometheus_client not available
    Counter = Gauge = Histogram = Summary = Info = PrometheusEnum = None
    CollectorRegistry = generate_latest = push_to_gateway = None
    start_http_server = MetricWrapperBase = None
    PROMETHEUS_AVAILABLE = False

logger = logging.getLogger(__name__)


class MetricCategory(Enum):
    """Categories of metrics for organization"""
    HEALING = "healing"                                     # Healing-related metrics
    CARE = "care"                                          # Care delivery metrics
    WELLBEING = "wellbeing"                                # Wellbeing measurements
    COMMUNITY = "community"                                # Community health metrics
    CRISIS = "crisis"                                      # Crisis intervention metrics
    SYSTEM = "system"                                      # System performance metrics
    ENGAGEMENT = "engagement"                              # User engagement (healing-focused)
    EFFECTIVENESS = "effectiveness"                        # Care effectiveness metrics


class HealingMetricType(Enum):
    """Types of healing-focused metrics"""
    HEALING_MOMENTS = "healing_moments"                    # Moments of healing facilitated
    CARE_CONNECTIONS = "care_connections"                  # Meaningful connections made
    CRISIS_INTERVENTIONS = "crisis_interventions"          # Crisis interventions provided
    WELLBEING_IMPROVEMENTS = "wellbeing_improvements"      # Wellbeing improvements measured
    COMMUNITY_RESILIENCE = "community_resilience"         # Community resilience indicators
    SUPPORT_EFFECTIVENESS = "support_effectiveness"       # Effectiveness of support provided
    HEALING_JOURNEY_PROGRESS = "healing_journey_progress"  # Progress in healing journeys
    CULTURAL_ADAPTATION_SUCCESS = "cultural_adaptation"   # Cultural adaptation effectiveness


@dataclass
class CustomMetric:
    """Custom metric definition for Heart Protocol"""
    name: str
    metric_type: str  # 'counter', 'gauge', 'histogram', 'summary'
    description: str
    labels: List[str]
    category: MetricCategory
    healing_relevance: float  # 0-1 scale
    privacy_level: str
    cultural_sensitivity: bool
    buckets: Optional[List[float]] = None  # For histograms
    quantiles: Optional[Dict[float, float]] = None  # For summaries
    _metric_instance: Any = None


class HeartProtocolRegistry:
    """
    Custom Prometheus registry for Heart Protocol metrics with healing focus.
    
    Core Principles:
    - Healing outcomes over engagement metrics
    - Privacy-preserving metric collection
    - Cultural sensitivity in measurement
    - Community wellbeing focus
    - Crisis intervention effectiveness
    - Care quality measurement
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.registry = CollectorRegistry() if PROMETHEUS_AVAILABLE else None
        
        # Metric storage and organization
        self.custom_metrics: Dict[str, CustomMetric] = {}
        self.metric_categories: Dict[MetricCategory, List[str]] = defaultdict(list)
        self.healing_metrics: Dict[HealingMetricType, Any] = {}
        
        # Privacy and cultural considerations
        self.privacy_filters: List[Callable] = []
        self.cultural_adaptations: Dict[str, Any] = {}
        self.consent_requirements: Dict[str, bool] = {}
        
        # Prometheus server configuration
        self.metrics_port = config.get('metrics_port', 8000)
        self.metrics_path = config.get('metrics_path', '/metrics')
        self.push_gateway_url = config.get('push_gateway_url')
        self.job_name = config.get('job_name', 'heart_protocol')
        
        # Initialize if Prometheus is available
        if PROMETHEUS_AVAILABLE:
            self._setup_core_metrics()
            self._setup_healing_metrics()
            self._setup_privacy_filters()
            self._setup_cultural_adaptations()
    
    def _setup_core_metrics(self):
        """Setup core Heart Protocol metrics"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        # Healing moments counter
        self.custom_metrics['healing_moments_total'] = CustomMetric(
            name='heart_protocol_healing_moments_total',
            metric_type='counter',
            description='Total number of healing moments facilitated by the system',
            labels=['dimension', 'cultural_context', 'intervention_type', 'outcome'],
            category=MetricCategory.HEALING,
            healing_relevance=1.0,
            privacy_level='aggregated',
            cultural_sensitivity=True
        )
        
        # Care connections counter
        self.custom_metrics['care_connections_total'] = CustomMetric(
            name='heart_protocol_care_connections_total',
            description='Total number of meaningful care connections facilitated',
            metric_type='counter',
            labels=['connection_type', 'support_level', 'cultural_context', 'success'],
            category=MetricCategory.CARE,
            healing_relevance=0.95,
            privacy_level='aggregated',
            cultural_sensitivity=True
        )
        
        # Crisis interventions counter
        self.custom_metrics['crisis_interventions_total'] = CustomMetric(
            name='heart_protocol_crisis_interventions_total',
            description='Total number of crisis interventions provided',
            metric_type='counter',
            labels=['intervention_type', 'urgency_level', 'outcome', 'follow_up_success'],
            category=MetricCategory.CRISIS,
            healing_relevance=1.0,
            privacy_level='aggregated',
            cultural_sensitivity=True
        )
        
        # Community wellbeing gauge
        self.custom_metrics['community_wellbeing_score'] = CustomMetric(
            name='heart_protocol_community_wellbeing_score',
            description='Current community wellbeing score (0-100 scale)',
            metric_type='gauge',
            labels=['community_id', 'cultural_context', 'measurement_method'],
            category=MetricCategory.WELLBEING,
            healing_relevance=1.0,
            privacy_level='community_aggregated',
            cultural_sensitivity=True
        )
        
        # Healing journey progress histogram
        self.custom_metrics['healing_journey_progress'] = CustomMetric(
            name='heart_protocol_healing_journey_progress_duration_seconds',
            description='Time taken for healing journey milestones',
            metric_type='histogram',
            labels=['milestone_type', 'cultural_context', 'support_type'],
            category=MetricCategory.HEALING,
            healing_relevance=1.0,
            privacy_level='anonymized',
            cultural_sensitivity=True,
            buckets=[1*24*3600, 7*24*3600, 30*24*3600, 90*24*3600, 365*24*3600]  # 1d, 1w, 1m, 3m, 1y
        )
        
        # Care effectiveness summary
        self.custom_metrics['care_effectiveness_score'] = CustomMetric(
            name='heart_protocol_care_effectiveness_score',
            description='Effectiveness score of care interventions (0-100 scale)',
            metric_type='summary',
            labels=['care_type', 'cultural_context', 'feedback_source'],
            category=MetricCategory.EFFECTIVENESS,
            healing_relevance=0.9,
            privacy_level='aggregated',
            cultural_sensitivity=True,
            quantiles={0.5: 0.05, 0.9: 0.01, 0.99: 0.001}
        )
        
        # System resource usage for healing capacity
        self.custom_metrics['healing_system_capacity'] = CustomMetric(
            name='heart_protocol_healing_system_capacity_gauge',
            description='Current capacity of healing systems (0-100 scale)',
            metric_type='gauge',
            labels=['system_component', 'resource_type', 'availability_level'],
            category=MetricCategory.SYSTEM,
            healing_relevance=0.7,
            privacy_level='public',
            cultural_sensitivity=False
        )
        
        # Cultural adaptation success rate
        self.custom_metrics['cultural_adaptation_success'] = CustomMetric(
            name='heart_protocol_cultural_adaptation_success_rate',
            description='Success rate of cultural adaptations (0-1 scale)',
            metric_type='gauge',
            labels=['cultural_context', 'adaptation_type', 'measurement_period'],
            category=MetricCategory.EFFECTIVENESS,
            healing_relevance=0.8,
            privacy_level='aggregated',
            cultural_sensitivity=True
        )
    
    def _setup_healing_metrics(self):
        """Setup specific healing-focused metrics"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        # Create Prometheus metric instances
        for metric_name, metric_def in self.custom_metrics.items():
            if metric_def.metric_type == 'counter':
                metric_def._metric_instance = Counter(
                    metric_def.name,
                    metric_def.description,
                    metric_def.labels,
                    registry=self.registry
                )
            elif metric_def.metric_type == 'gauge':
                metric_def._metric_instance = Gauge(
                    metric_def.name,
                    metric_def.description,
                    metric_def.labels,
                    registry=self.registry
                )
            elif metric_def.metric_type == 'histogram':
                metric_def._metric_instance = Histogram(
                    metric_def.name,
                    metric_def.description,
                    metric_def.labels,
                    buckets=metric_def.buckets,
                    registry=self.registry
                )
            elif metric_def.metric_type == 'summary':
                metric_def._metric_instance = Summary(
                    metric_def.name,
                    metric_def.description,
                    metric_def.labels,
                    registry=self.registry
                )
            
            # Categorize metric
            self.metric_categories[metric_def.category].append(metric_name)
        
        # Create healing-specific metrics mapping
        self.healing_metrics = {
            HealingMetricType.HEALING_MOMENTS: self.custom_metrics['healing_moments_total'],
            HealingMetricType.CARE_CONNECTIONS: self.custom_metrics['care_connections_total'],
            HealingMetricType.CRISIS_INTERVENTIONS: self.custom_metrics['crisis_interventions_total'],
            HealingMetricType.WELLBEING_IMPROVEMENTS: self.custom_metrics['community_wellbeing_score'],
            HealingMetricType.SUPPORT_EFFECTIVENESS: self.custom_metrics['care_effectiveness_score'],
            HealingMetricType.HEALING_JOURNEY_PROGRESS: self.custom_metrics['healing_journey_progress'],
            HealingMetricType.CULTURAL_ADAPTATION_SUCCESS: self.custom_metrics['cultural_adaptation_success']
        }
    
    def _setup_privacy_filters(self):
        """Setup privacy filters for metrics"""
        self.privacy_filters = [
            self._filter_personal_identifiers,
            self._anonymize_sensitive_labels,
            self._aggregate_small_populations,
            self._apply_consent_filtering
        ]
    
    def _setup_cultural_adaptations(self):
        """Setup cultural adaptations for metrics"""
        self.cultural_adaptations = {
            'collectivist_cultures': {
                'metric_focus': 'community_over_individual',
                'aggregation_level': 'group_based',
                'privacy_emphasis': 'family_unit_protection'
            },
            'individualist_cultures': {
                'metric_focus': 'individual_outcomes',
                'aggregation_level': 'individual_with_privacy',
                'privacy_emphasis': 'personal_data_protection'
            },
            'high_privacy_cultures': {
                'metric_focus': 'maximum_anonymization',
                'aggregation_level': 'high_level_only',
                'privacy_emphasis': 'minimal_data_collection'
            },
            'trauma_informed_cultures': {
                'metric_focus': 'safety_prioritized',
                'aggregation_level': 'trauma_sensitive',
                'privacy_emphasis': 'healing_context_protection'
            }
        }
    
    async def record_healing_moment(self, dimension: str, cultural_context: str = 'general',
                                  intervention_type: str = 'general', outcome: str = 'positive'):
        """Record a healing moment with privacy protection"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        # Apply privacy filters
        filtered_labels = await self._apply_privacy_filters({
            'dimension': dimension,
            'cultural_context': cultural_context,
            'intervention_type': intervention_type,
            'outcome': outcome
        })
        
        metric = self.healing_metrics[HealingMetricType.HEALING_MOMENTS]
        if metric._metric_instance:
            metric._metric_instance.labels(**filtered_labels).inc()
    
    async def record_care_connection(self, connection_type: str, support_level: str,
                                   cultural_context: str = 'general', success: str = 'true'):
        """Record a care connection with privacy protection"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        filtered_labels = await self._apply_privacy_filters({
            'connection_type': connection_type,
            'support_level': support_level,
            'cultural_context': cultural_context,
            'success': success
        })
        
        metric = self.healing_metrics[HealingMetricType.CARE_CONNECTIONS]
        if metric._metric_instance:
            metric._metric_instance.labels(**filtered_labels).inc()
    
    async def record_crisis_intervention(self, intervention_type: str, urgency_level: str,
                                       outcome: str, follow_up_success: str = 'unknown'):
        """Record a crisis intervention with privacy protection"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        filtered_labels = await self._apply_privacy_filters({
            'intervention_type': intervention_type,
            'urgency_level': urgency_level,
            'outcome': outcome,
            'follow_up_success': follow_up_success
        })
        
        metric = self.healing_metrics[HealingMetricType.CRISIS_INTERVENTIONS]
        if metric._metric_instance:
            metric._metric_instance.labels(**filtered_labels).inc()
    
    async def update_community_wellbeing(self, community_id: str, score: float,
                                       cultural_context: str = 'general',
                                       measurement_method: str = 'assessment'):
        """Update community wellbeing score with privacy protection"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        # Anonymize community ID for privacy
        anonymized_community = await self._anonymize_community_id(community_id)
        
        filtered_labels = await self._apply_privacy_filters({
            'community_id': anonymized_community,
            'cultural_context': cultural_context,
            'measurement_method': measurement_method
        })
        
        metric = self.healing_metrics[HealingMetricType.WELLBEING_IMPROVEMENTS]
        if metric._metric_instance:
            metric._metric_instance.labels(**filtered_labels).set(score)
    
    async def observe_healing_journey_duration(self, duration_seconds: float,
                                             milestone_type: str,
                                             cultural_context: str = 'general',
                                             support_type: str = 'general'):
        """Observe healing journey duration with privacy protection"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        filtered_labels = await self._apply_privacy_filters({
            'milestone_type': milestone_type,
            'cultural_context': cultural_context,
            'support_type': support_type
        })
        
        metric = self.healing_metrics[HealingMetricType.HEALING_JOURNEY_PROGRESS]
        if metric._metric_instance:
            metric._metric_instance.labels(**filtered_labels).observe(duration_seconds)
    
    async def observe_care_effectiveness(self, effectiveness_score: float,
                                       care_type: str, cultural_context: str = 'general',
                                       feedback_source: str = 'user'):
        """Observe care effectiveness score with privacy protection"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        filtered_labels = await self._apply_privacy_filters({
            'care_type': care_type,
            'cultural_context': cultural_context,
            'feedback_source': feedback_source
        })
        
        metric = self.healing_metrics[HealingMetricType.SUPPORT_EFFECTIVENESS]
        if metric._metric_instance:
            metric._metric_instance.labels(**filtered_labels).observe(effectiveness_score)
    
    async def update_system_capacity(self, component: str, resource_type: str,
                                   capacity_percentage: float, availability_level: str = 'normal'):
        """Update system capacity metrics"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        labels = {
            'system_component': component,
            'resource_type': resource_type,
            'availability_level': availability_level
        }
        
        metric = self.custom_metrics['healing_system_capacity']
        if metric._metric_instance:
            metric._metric_instance.labels(**labels).set(capacity_percentage)
    
    async def update_cultural_adaptation_success(self, cultural_context: str,
                                               adaptation_type: str, success_rate: float,
                                               measurement_period: str = 'daily'):
        """Update cultural adaptation success rate"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        filtered_labels = await self._apply_privacy_filters({
            'cultural_context': cultural_context,
            'adaptation_type': adaptation_type,
            'measurement_period': measurement_period
        })
        
        metric = self.healing_metrics[HealingMetricType.CULTURAL_ADAPTATION_SUCCESS]
        if metric._metric_instance:
            metric._metric_instance.labels(**filtered_labels).set(success_rate)
    
    async def _apply_privacy_filters(self, labels: Dict[str, str]) -> Dict[str, str]:
        """Apply privacy filters to metric labels"""
        filtered_labels = labels.copy()
        
        for filter_func in self.privacy_filters:
            filtered_labels = await filter_func(filtered_labels)
        
        return filtered_labels
    
    async def _filter_personal_identifiers(self, labels: Dict[str, str]) -> Dict[str, str]:
        """Filter out personal identifiers from labels"""
        # Remove or anonymize any personal identifiers
        filtered = {}
        for key, value in labels.items():
            if 'user' in key.lower() or 'id' in key.lower():
                if key != 'community_id':  # Community ID is handled separately
                    filtered[key] = '[ANONYMIZED]'
                else:
                    filtered[key] = value
            else:
                filtered[key] = value
        return filtered
    
    async def _anonymize_sensitive_labels(self, labels: Dict[str, str]) -> Dict[str, str]:
        """Anonymize sensitive label values"""
        sensitive_keys = ['intervention_type', 'outcome']
        anonymized = {}
        
        for key, value in labels.items():
            if key in sensitive_keys and 'crisis' in value.lower():
                # Anonymize crisis-related details
                anonymized[key] = 'crisis_intervention'
            else:
                anonymized[key] = value
        
        return anonymized
    
    async def _aggregate_small_populations(self, labels: Dict[str, str]) -> Dict[str, str]:
        """Aggregate labels that might identify small populations"""
        # If cultural context might identify small populations, generalize it
        aggregated = {}
        for key, value in labels.items():
            if key == 'cultural_context':
                # Generalize specific cultural contexts to broader categories
                if 'specific' in value.lower():
                    aggregated[key] = 'cultural_minority'
                else:
                    aggregated[key] = value
            else:
                aggregated[key] = value
        
        return aggregated
    
    async def _apply_consent_filtering(self, labels: Dict[str, str]) -> Dict[str, str]:
        """Apply consent-based filtering to labels"""
        # This would integrate with consent management system
        # For now, just pass through
        return labels
    
    async def _anonymize_community_id(self, community_id: str) -> str:
        """Anonymize community ID while preserving uniqueness"""
        import hashlib
        return hashlib.sha256(f"{community_id}_community_salt".encode()).hexdigest()[:12]
    
    def get_metrics_by_category(self, category: MetricCategory) -> List[str]:
        """Get metrics by category"""
        return self.metric_categories.get(category, [])
    
    def get_healing_focused_metrics(self) -> Dict[str, CustomMetric]:
        """Get metrics with high healing relevance"""
        return {
            name: metric for name, metric in self.custom_metrics.items()
            if metric.healing_relevance >= 0.8
        }
    
    def export_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        if not PROMETHEUS_AVAILABLE or not self.registry:
            return "# Prometheus not available\n"
        
        return generate_latest(self.registry).decode('utf-8')
    
    async def push_metrics_to_gateway(self):
        """Push metrics to Prometheus push gateway"""
        if not PROMETHEUS_AVAILABLE or not self.push_gateway_url:
            logger.warning("Prometheus push gateway not configured")
            return
        
        try:
            push_to_gateway(
                self.push_gateway_url,
                job=self.job_name,
                registry=self.registry
            )
            logger.info("Metrics pushed to gateway successfully")
        except Exception as e:
            logger.error(f"Failed to push metrics to gateway: {e}")
    
    def start_metrics_server(self):
        """Start HTTP server for metrics endpoint"""
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus not available, cannot start metrics server")
            return
        
        try:
            start_http_server(self.metrics_port, registry=self.registry)
            logger.info(f"Metrics server started on port {self.metrics_port}")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
    
    async def generate_healing_metrics_report(self) -> Dict[str, Any]:
        """Generate report of healing-focused metrics"""
        healing_metrics = self.get_healing_focused_metrics()
        
        report = {
            'total_healing_metrics': len(healing_metrics),
            'metrics_by_category': {
                category.value: len(metrics) 
                for category, metrics in self.metric_categories.items()
            },
            'privacy_levels': defaultdict(int),
            'cultural_sensitivity_count': 0,
            'average_healing_relevance': 0.0
        }
        
        total_relevance = 0
        for metric in self.custom_metrics.values():
            report['privacy_levels'][metric.privacy_level] += 1
            if metric.cultural_sensitivity:
                report['cultural_sensitivity_count'] += 1
            total_relevance += metric.healing_relevance
        
        if self.custom_metrics:
            report['average_healing_relevance'] = total_relevance / len(self.custom_metrics)
        
        report['cultural_adaptations_available'] = len(self.cultural_adaptations)
        report['prometheus_available'] = PROMETHEUS_AVAILABLE
        report['report_generated'] = datetime.now().isoformat()
        
        return dict(report)


class PrometheusIntegration:
    """
    Main integration class for Prometheus monitoring in Heart Protocol.
    
    Coordinates between the registry, metrics collection, and reporting
    with a focus on healing outcomes and privacy protection.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.registry = HeartProtocolRegistry(config)
        self.metrics_server_running = False
        self.push_interval = config.get('push_interval', 60)  # seconds
        self.background_tasks: List[asyncio.Task] = []
    
    async def initialize(self):
        """Initialize Prometheus integration"""
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus client not available - metrics collection disabled")
            return
        
        # Start metrics server if configured
        if self.config.get('start_metrics_server', True):
            self.registry.start_metrics_server()
            self.metrics_server_running = True
        
        # Start push gateway task if configured
        if self.registry.push_gateway_url:
            task = asyncio.create_task(self._push_metrics_periodically())
            self.background_tasks.append(task)
    
    async def _push_metrics_periodically(self):
        """Periodically push metrics to gateway"""
        while True:
            try:
                await self.registry.push_metrics_to_gateway()
                await asyncio.sleep(self.push_interval)
            except Exception as e:
                logger.error(f"Error in periodic metrics push: {e}")
                await asyncio.sleep(self.push_interval)
    
    async def record_healing_event(self, event_type: str, **kwargs):
        """Record a healing event with appropriate metric"""
        if event_type == 'healing_moment':
            await self.registry.record_healing_moment(**kwargs)
        elif event_type == 'care_connection':
            await self.registry.record_care_connection(**kwargs)
        elif event_type == 'crisis_intervention':
            await self.registry.record_crisis_intervention(**kwargs)
        elif event_type == 'wellbeing_update':
            await self.registry.update_community_wellbeing(**kwargs)
        elif event_type == 'healing_journey':
            await self.registry.observe_healing_journey_duration(**kwargs)
        elif event_type == 'care_effectiveness':
            await self.registry.observe_care_effectiveness(**kwargs)
    
    def get_metrics_export(self) -> str:
        """Get metrics in Prometheus export format"""
        return self.registry.export_metrics()
    
    async def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive monitoring report"""
        healing_report = await self.registry.generate_healing_metrics_report()
        
        comprehensive_report = {
            'prometheus_integration': {
                'available': PROMETHEUS_AVAILABLE,
                'metrics_server_running': self.metrics_server_running,
                'push_gateway_configured': bool(self.registry.push_gateway_url),
                'background_tasks_running': len(self.background_tasks)
            },
            'healing_metrics': healing_report,
            'configuration': {
                'metrics_port': self.registry.metrics_port,
                'push_interval': self.push_interval,
                'job_name': self.registry.job_name
            },
            'report_timestamp': datetime.now().isoformat()
        }
        
        return comprehensive_report
    
    async def shutdown(self):
        """Shutdown Prometheus integration"""
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        # Final metrics push
        if self.registry.push_gateway_url:
            await self.registry.push_metrics_to_gateway()
        
        logger.info("Prometheus integration shutdown complete")


# Convenience class for easy metric access
class CustomMetrics:
    """Convenience class for accessing custom Heart Protocol metrics"""
    
    def __init__(self, prometheus_integration: PrometheusIntegration):
        self.integration = prometheus_integration
        self.registry = prometheus_integration.registry
    
    async def healing_moment(self, dimension: str = 'general', **kwargs):
        """Record a healing moment"""
        await self.registry.record_healing_moment(dimension=dimension, **kwargs)
    
    async def care_connection(self, connection_type: str = 'support', **kwargs):
        """Record a care connection"""
        await self.registry.record_care_connection(connection_type=connection_type, **kwargs)
    
    async def crisis_intervention(self, intervention_type: str = 'support', **kwargs):
        """Record a crisis intervention"""
        await self.registry.record_crisis_intervention(intervention_type=intervention_type, **kwargs)
    
    async def wellbeing_score(self, community_id: str, score: float, **kwargs):
        """Update wellbeing score"""
        await self.registry.update_community_wellbeing(community_id=community_id, score=score, **kwargs)
    
    async def healing_journey_milestone(self, duration_seconds: float, 
                                      milestone_type: str = 'progress', **kwargs):
        """Record healing journey milestone"""
        await self.registry.observe_healing_journey_duration(
            duration_seconds=duration_seconds, 
            milestone_type=milestone_type, 
            **kwargs
        )
    
    async def care_effectiveness(self, score: float, care_type: str = 'support', **kwargs):
        """Record care effectiveness score"""
        await self.registry.observe_care_effectiveness(
            effectiveness_score=score, 
            care_type=care_type, 
            **kwargs
        )