"""
Wellbeing Metrics Collector

Specialized metrics collection system focused on wellbeing, healing outcomes,
and care effectiveness rather than traditional engagement metrics.
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import logging
import time
import statistics
from collections import defaultdict, deque
import hashlib

try:
    from prometheus_client import Counter, Gauge, Histogram, CollectorRegistry, generate_latest
except ImportError:
    # Graceful fallback if prometheus_client not available
    Counter = Gauge = Histogram = CollectorRegistry = None

logger = logging.getLogger(__name__)


class WellbeingDimension(Enum):
    """Dimensions of wellbeing to measure"""
    EMOTIONAL_SAFETY = "emotional_safety"                   # Feeling emotionally safe
    PSYCHOLOGICAL_WELLBEING = "psychological_wellbeing"     # Mental health indicators
    SOCIAL_CONNECTION = "social_connection"                 # Quality of connections
    HEALING_PROGRESS = "healing_progress"                   # Progress in healing journey
    COMMUNITY_BELONGING = "community_belonging"             # Sense of belonging
    SUPPORT_RECEIVED = "support_received"                   # Support quality received
    CARE_GIVEN = "care_given"                              # Care provided to others
    RESILIENCE_BUILDING = "resilience_building"            # Resilience development
    GROWTH_MINDSET = "growth_mindset"                      # Learning and growth
    AUTHENTIC_EXPRESSION = "authentic_expression"          # Ability to be authentic


class MetricType(Enum):
    """Types of wellbeing metrics"""
    COUNTER = "counter"                                     # Incrementing values
    GAUGE = "gauge"                                         # Current state values
    HISTOGRAM = "histogram"                                 # Distribution of values
    SUMMARY = "summary"                                     # Summary statistics
    HEALING_TRAJECTORY = "healing_trajectory"              # Healing journey tracking


class PrivacyLevel(Enum):
    """Privacy levels for metrics collection"""
    PUBLIC = "public"                                       # Publicly shareable
    COMMUNITY = "community"                                 # Community-level only
    AGGREGATED = "aggregated"                              # Aggregated data only
    ANONYMIZED = "anonymized"                              # Anonymized personal data
    PRIVATE = "private"                                     # Private/confidential


@dataclass
class WellbeingMetric:
    """Individual wellbeing metric measurement"""
    metric_id: str
    dimension: WellbeingDimension
    metric_type: MetricType
    value: float
    labels: Dict[str, str]
    privacy_level: PrivacyLevel
    consent_given: bool
    cultural_context: List[str]
    timestamp: datetime
    user_hash: Optional[str]  # Anonymized user identifier
    healing_context: Dict[str, Any]
    measurement_confidence: float
    data_quality_score: float


@dataclass
class HealingCounter:
    """Counter for healing-related events"""
    name: str
    description: str
    labels: List[str]
    privacy_level: PrivacyLevel
    healing_impact_weight: float
    cultural_sensitivity: bool
    _counter: Any = None
    
    def __post_init__(self):
        if Counter:
            self._counter = Counter(self.name, self.description, self.labels)
    
    def inc(self, amount: float = 1, **labels):
        """Increment counter with healing context"""
        if self._counter:
            self._counter.labels(**labels).inc(amount)


@dataclass
class WellbeingGauge:
    """Gauge for current wellbeing state"""
    name: str
    description: str
    labels: List[str]
    privacy_level: PrivacyLevel
    dimension: WellbeingDimension
    healing_relevance: float
    _gauge: Any = None
    
    def __post_init__(self):
        if Gauge:
            self._gauge = Gauge(self.name, self.description, self.labels)
    
    def set(self, value: float, **labels):
        """Set gauge value with wellbeing context"""
        if self._gauge:
            self._gauge.labels(**labels).set(value)


class WellbeingMetricsCollector:
    """
    Comprehensive wellbeing metrics collection system that prioritizes
    healing outcomes, care effectiveness, and community wellbeing.
    
    Core Principles:
    - Wellbeing metrics over engagement metrics
    - Privacy-preserving measurement
    - Cultural sensitivity in metrics interpretation
    - Healing-focused data collection
    - Trauma-informed measurement practices
    - Community benefit prioritization
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.registry = CollectorRegistry() if CollectorRegistry else None
        
        # Core metrics storage
        self.wellbeing_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.healing_counters: Dict[str, HealingCounter] = {}
        self.wellbeing_gauges: Dict[str, WellbeingGauge] = {}
        self.metric_definitions: Dict[str, Dict[str, Any]] = {}
        
        # Privacy and consent tracking
        self.consent_tracker: Dict[str, Dict[str, bool]] = defaultdict(dict)
        self.privacy_filters: List[Callable] = []
        self.data_anonymizers: List[Callable] = []
        
        # Cultural adaptations
        self.cultural_metric_interpretations: Dict[str, Any] = {}
        self.healing_metric_weights: Dict[WellbeingDimension, float] = {}
        
        # Aggregation and reporting
        self.aggregation_intervals: Dict[str, timedelta] = {}
        self.reporting_callbacks: List[Callable] = []
        
        # Initialize metrics system
        self._setup_core_metrics()
        self._setup_privacy_filters()
        self._setup_cultural_adaptations()
        self._setup_healing_weights()
    
    def _setup_core_metrics(self):
        """Setup core wellbeing metrics"""
        # Healing progress metrics
        self.healing_counters['healing_moments'] = HealingCounter(
            name='heart_protocol_healing_moments_total',
            description='Total number of healing moments facilitated',
            labels=['dimension', 'cultural_context', 'intervention_type'],
            privacy_level=PrivacyLevel.AGGREGATED,
            healing_impact_weight=1.0,
            cultural_sensitivity=True
        )
        
        self.healing_counters['care_connections'] = HealingCounter(
            name='heart_protocol_care_connections_total',
            description='Total number of meaningful care connections made',
            labels=['connection_type', 'cultural_context', 'support_level'],
            privacy_level=PrivacyLevel.AGGREGATED,
            healing_impact_weight=0.9,
            cultural_sensitivity=True
        )
        
        self.healing_counters['crisis_interventions'] = HealingCounter(
            name='heart_protocol_crisis_interventions_total',
            description='Total number of crisis interventions provided',
            labels=['intervention_type', 'outcome', 'urgency_level'],
            privacy_level=PrivacyLevel.AGGREGATED,
            healing_impact_weight=1.0,
            cultural_sensitivity=True
        )
        
        # Current wellbeing state gauges
        self.wellbeing_gauges['community_wellbeing'] = WellbeingGauge(
            name='heart_protocol_community_wellbeing_score',
            description='Current community wellbeing score (0-100)',
            labels=['community_id', 'cultural_context'],
            privacy_level=PrivacyLevel.COMMUNITY,
            dimension=WellbeingDimension.COMMUNITY_BELONGING,
            healing_relevance=1.0
        )
        
        self.wellbeing_gauges['healing_capacity'] = WellbeingGauge(
            name='heart_protocol_healing_capacity_gauge',
            description='Current healing capacity of the system (0-100)',
            labels=['system_component', 'resource_type'],
            privacy_level=PrivacyLevel.PUBLIC,
            dimension=WellbeingDimension.HEALING_PROGRESS,
            healing_relevance=1.0
        )
        
        self.wellbeing_gauges['care_quality'] = WellbeingGauge(
            name='heart_protocol_care_quality_score',
            description='Current care quality score (0-100)',
            labels=['care_type', 'cultural_context', 'feedback_source'],
            privacy_level=PrivacyLevel.AGGREGATED,
            dimension=WellbeingDimension.SUPPORT_RECEIVED,
            healing_relevance=0.9
        )
    
    def _setup_privacy_filters(self):
        """Setup privacy filters for sensitive data"""
        self.privacy_filters = [
            self._filter_personal_identifiers,
            self._filter_sensitive_content,
            self._filter_location_data,
            self._filter_demographic_details,
            self._filter_medical_information
        ]
        
        self.data_anonymizers = [
            self._anonymize_user_identifiers,
            self._anonymize_temporal_patterns,
            self._anonymize_interaction_patterns
        ]
    
    def _setup_cultural_adaptations(self):
        """Setup cultural adaptations for metrics interpretation"""
        self.cultural_metric_interpretations = {
            'collectivist_cultures': {
                'wellbeing_focus': 'community_harmony_metrics',
                'success_indicators': ['group_cohesion', 'collective_support', 'shared_wellbeing'],
                'measurement_approach': 'group_centered',
                'privacy_considerations': 'family_unit_privacy'
            },
            'individualist_cultures': {
                'wellbeing_focus': 'personal_growth_metrics',
                'success_indicators': ['individual_achievement', 'personal_autonomy', 'self_actualization'],
                'measurement_approach': 'individual_centered',
                'privacy_considerations': 'individual_privacy'
            },
            'high_context_cultures': {
                'wellbeing_focus': 'relationship_quality_metrics',
                'success_indicators': ['relationship_harmony', 'respect_maintenance', 'face_saving'],
                'measurement_approach': 'context_sensitive',
                'privacy_considerations': 'indirect_measurement'
            },
            'trauma_informed_cultures': {
                'wellbeing_focus': 'safety_and_healing_metrics',
                'success_indicators': ['psychological_safety', 'trauma_recovery', 'resilience_building'],
                'measurement_approach': 'trauma_sensitive',
                'privacy_considerations': 'maximum_privacy_protection'
            }
        }
    
    def _setup_healing_weights(self):
        """Setup weights for different healing dimensions"""
        self.healing_metric_weights = {
            WellbeingDimension.EMOTIONAL_SAFETY: 1.0,          # Highest priority
            WellbeingDimension.PSYCHOLOGICAL_WELLBEING: 0.95,
            WellbeingDimension.HEALING_PROGRESS: 0.9,
            WellbeingDimension.SOCIAL_CONNECTION: 0.85,
            WellbeingDimension.COMMUNITY_BELONGING: 0.8,
            WellbeingDimension.SUPPORT_RECEIVED: 0.85,
            WellbeingDimension.CARE_GIVEN: 0.8,
            WellbeingDimension.RESILIENCE_BUILDING: 0.9,
            WellbeingDimension.GROWTH_MINDSET: 0.75,
            WellbeingDimension.AUTHENTIC_EXPRESSION: 0.8
        }
    
    async def record_wellbeing_metric(self, dimension: WellbeingDimension,
                                    value: float, user_id: Optional[str] = None,
                                    labels: Optional[Dict[str, str]] = None,
                                    cultural_context: Optional[List[str]] = None,
                                    healing_context: Optional[Dict[str, Any]] = None,
                                    privacy_level: PrivacyLevel = PrivacyLevel.ANONYMIZED) -> str:
        """Record a wellbeing metric with full privacy protection"""
        try:
            # Check consent if user identified
            if user_id and not await self._check_consent(user_id, dimension):
                logger.info(f"Consent not given for user {user_id}, dimension {dimension}")
                return None
            
            # Apply privacy filters
            filtered_labels = await self._apply_privacy_filters(labels or {})
            filtered_context = await self._apply_privacy_filters(healing_context or {})
            
            # Anonymize user identifier
            user_hash = await self._anonymize_user_id(user_id) if user_id else None
            
            # Apply cultural adaptations
            adapted_value = await self._apply_cultural_adaptations(
                dimension, value, cultural_context or []
            )
            
            # Create metric record
            metric_id = f"wb_{dimension.value}_{int(time.time())}_{hash(str(filtered_labels))}"
            metric = WellbeingMetric(
                metric_id=metric_id,
                dimension=dimension,
                metric_type=MetricType.GAUGE,
                value=adapted_value,
                labels=filtered_labels,
                privacy_level=privacy_level,
                consent_given=True if user_id else False,
                cultural_context=cultural_context or [],
                timestamp=datetime.now(),
                user_hash=user_hash,
                healing_context=filtered_context,
                measurement_confidence=await self._calculate_measurement_confidence(dimension, value, filtered_context),
                data_quality_score=await self._assess_data_quality(filtered_labels, filtered_context)
            )
            
            # Store metric
            self.wellbeing_metrics[dimension.value].append(metric)
            
            # Update Prometheus metrics if available
            await self._update_prometheus_metrics(metric)
            
            # Trigger aggregation if needed
            await self._check_aggregation_triggers(dimension)
            
            logger.info(f"Wellbeing metric recorded: {dimension.value}, value: {adapted_value}")
            return metric_id
            
        except Exception as e:
            logger.error(f"Error recording wellbeing metric: {e}")
            return None
    
    async def _check_consent(self, user_id: str, dimension: WellbeingDimension) -> bool:
        """Check if user has given consent for this dimension"""
        user_consents = self.consent_tracker.get(user_id, {})
        return user_consents.get(dimension.value, False)
    
    async def _apply_privacy_filters(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply privacy filters to data"""
        filtered_data = data.copy()
        
        for filter_func in self.privacy_filters:
            filtered_data = await filter_func(filtered_data)
        
        return filtered_data
    
    async def _filter_personal_identifiers(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Filter out personal identifiers"""
        sensitive_keys = ['email', 'phone', 'name', 'address', 'ssn', 'user_id']
        return {k: v for k, v in data.items() if k.lower() not in sensitive_keys}
    
    async def _filter_sensitive_content(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Filter out sensitive content"""
        filtered = {}
        for key, value in data.items():
            if isinstance(value, str):
                # Simple sensitive content detection
                if any(term in value.lower() for term in ['crisis', 'suicide', 'harm', 'abuse']):
                    filtered[key] = '[SENSITIVE_CONTENT_FILTERED]'
                else:
                    filtered[key] = value
            else:
                filtered[key] = value
        return filtered
    
    async def _filter_location_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Filter out precise location data"""
        location_keys = ['lat', 'lng', 'latitude', 'longitude', 'location', 'address']
        filtered = {}
        for key, value in data.items():
            if key.lower() in location_keys:
                # Generalize location to region level
                filtered[key] = '[REGION_GENERALIZED]'
            else:
                filtered[key] = value
        return filtered
    
    async def _filter_demographic_details(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Filter out detailed demographic information"""
        demographic_keys = ['age', 'gender', 'race', 'ethnicity', 'income', 'occupation']
        filtered = {}
        for key, value in data.items():
            if key.lower() in demographic_keys:
                # Generalize demographic data
                filtered[key] = await self._generalize_demographic(key, value)
            else:
                filtered[key] = value
        return filtered
    
    async def _filter_medical_information(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Filter out medical information"""
        medical_keys = ['diagnosis', 'medication', 'therapy', 'medical_history', 'treatment']
        return {k: v for k, v in data.items() if k.lower() not in medical_keys}
    
    async def _anonymize_user_id(self, user_id: str) -> str:
        """Create anonymized but consistent user identifier"""
        return hashlib.sha256(f"{user_id}_heart_protocol_salt".encode()).hexdigest()[:16]
    
    async def _anonymize_user_identifiers(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Anonymize any remaining user identifiers"""
        anonymized = {}
        for key, value in data.items():
            if 'user' in key.lower() or 'id' in key.lower():
                if isinstance(value, str):
                    anonymized[key] = await self._anonymize_user_id(value)
                else:
                    anonymized[key] = value
            else:
                anonymized[key] = value
        return anonymized
    
    async def _anonymize_temporal_patterns(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Anonymize temporal patterns to prevent timing-based identification"""
        temporal_keys = ['timestamp', 'time', 'date', 'created_at', 'updated_at']
        anonymized = {}
        for key, value in data.items():
            if key.lower() in temporal_keys:
                # Round timestamp to hour to prevent precise timing identification
                if isinstance(value, datetime):
                    anonymized[key] = value.replace(minute=0, second=0, microsecond=0)
                else:
                    anonymized[key] = value
            else:
                anonymized[key] = value
        return anonymized
    
    async def _anonymize_interaction_patterns(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Anonymize interaction patterns"""
        # Remove or generalize interaction-specific details that could be identifying
        interaction_keys = ['session_id', 'device_id', 'ip_address', 'user_agent']
        return {k: v for k, v in data.items() if k.lower() not in interaction_keys}
    
    async def _generalize_demographic(self, key: str, value: Any) -> str:
        """Generalize demographic data to protect privacy"""
        if key.lower() == 'age':
            if isinstance(value, int):
                # Age ranges instead of exact age
                if value < 18:
                    return 'under_18'
                elif value < 25:
                    return '18_24'
                elif value < 35:
                    return '25_34'
                elif value < 45:
                    return '35_44'
                elif value < 55:
                    return '45_54'
                elif value < 65:
                    return '55_64'
                else:
                    return '65_plus'
        return '[GENERALIZED]'
    
    async def _apply_cultural_adaptations(self, dimension: WellbeingDimension,
                                        value: float, cultural_context: List[str]) -> float:
        """Apply cultural adaptations to metric values"""
        adapted_value = value
        
        for culture in cultural_context:
            if culture in self.cultural_metric_interpretations:
                interpretation = self.cultural_metric_interpretations[culture]
                
                # Apply cultural weighting
                if dimension == WellbeingDimension.COMMUNITY_BELONGING and culture == 'collectivist_cultures':
                    adapted_value *= 1.2  # Higher weight for community belonging in collectivist cultures
                elif dimension == WellbeingDimension.AUTHENTIC_EXPRESSION and culture == 'individualist_cultures':
                    adapted_value *= 1.1  # Higher weight for authentic expression in individualist cultures
                elif dimension == WellbeingDimension.EMOTIONAL_SAFETY and culture == 'trauma_informed_cultures':
                    adapted_value *= 1.15  # Higher weight for emotional safety in trauma-informed contexts
        
        return max(0.0, min(100.0, adapted_value))  # Keep within 0-100 range
    
    async def _calculate_measurement_confidence(self, dimension: WellbeingDimension,
                                              value: float, context: Dict[str, Any]) -> float:
        """Calculate confidence level for the measurement"""
        base_confidence = 0.7
        
        # Adjust based on context richness
        if context:
            context_richness = len(context) / 10  # Normalize
            base_confidence += min(0.2, context_richness)
        
        # Adjust based on dimension-specific factors
        dimension_confidence_factors = {
            WellbeingDimension.EMOTIONAL_SAFETY: 0.9,  # High confidence dimension
            WellbeingDimension.HEALING_PROGRESS: 0.8,
            WellbeingDimension.SOCIAL_CONNECTION: 0.85,
            WellbeingDimension.COMMUNITY_BELONGING: 0.75,
            WellbeingDimension.AUTHENTIC_EXPRESSION: 0.7  # More subjective
        }
        
        dimension_factor = dimension_confidence_factors.get(dimension, 0.75)
        base_confidence *= dimension_factor
        
        return max(0.1, min(1.0, base_confidence))
    
    async def _assess_data_quality(self, labels: Dict[str, str], 
                                 context: Dict[str, Any]) -> float:
        """Assess quality of the data"""
        quality_score = 0.5  # Base score
        
        # Label completeness
        if labels:
            quality_score += 0.2
        
        # Context richness
        if context:
            quality_score += 0.2
        
        # Data consistency checks would go here
        
        return max(0.0, min(1.0, quality_score))
    
    async def _update_prometheus_metrics(self, metric: WellbeingMetric):
        """Update Prometheus metrics if available"""
        if not self.registry:
            return
        
        # Update appropriate Prometheus metric based on dimension
        if metric.dimension == WellbeingDimension.HEALING_PROGRESS:
            counter = self.healing_counters.get('healing_moments')
            if counter:
                counter.inc(metric.value, dimension=metric.dimension.value, 
                          cultural_context=','.join(metric.cultural_context))
        
        elif metric.dimension == WellbeingDimension.COMMUNITY_BELONGING:
            gauge = self.wellbeing_gauges.get('community_wellbeing')
            if gauge:
                gauge.set(metric.value, cultural_context=','.join(metric.cultural_context))
    
    async def _check_aggregation_triggers(self, dimension: WellbeingDimension):
        """Check if aggregation should be triggered"""
        # Simple time-based aggregation
        last_aggregation_key = f"last_aggregation_{dimension.value}"
        interval = self.aggregation_intervals.get(dimension.value, timedelta(hours=1))
        
        # Would implement actual aggregation logic here
        pass
    
    async def increment_healing_counter(self, counter_name: str, amount: float = 1, **labels):
        """Increment a healing counter"""
        counter = self.healing_counters.get(counter_name)
        if counter:
            # Apply privacy filters to labels
            filtered_labels = await self._apply_privacy_filters(labels)
            counter.inc(amount, **filtered_labels)
    
    async def set_wellbeing_gauge(self, gauge_name: str, value: float, **labels):
        """Set a wellbeing gauge value"""
        gauge = self.wellbeing_gauges.get(gauge_name)
        if gauge:
            # Apply privacy filters to labels
            filtered_labels = await self._apply_privacy_filters(labels)
            gauge.set(value, **filtered_labels)
    
    async def get_wellbeing_summary(self, dimension: Optional[WellbeingDimension] = None,
                                  time_range: Optional[timedelta] = None) -> Dict[str, Any]:
        """Get summary of wellbeing metrics"""
        if time_range:
            cutoff_time = datetime.now() - time_range
        else:
            cutoff_time = None
        
        summary = {}
        
        # Filter dimensions
        dimensions = [dimension] if dimension else list(WellbeingDimension)
        
        for dim in dimensions:
            metrics = self.wellbeing_metrics.get(dim.value, deque())
            
            # Filter by time if specified
            if cutoff_time:
                metrics = [m for m in metrics if m.timestamp >= cutoff_time]
            
            if metrics:
                values = [m.value for m in metrics]
                summary[dim.value] = {
                    'count': len(values),
                    'mean': statistics.mean(values),
                    'median': statistics.median(values),
                    'std_dev': statistics.stdev(values) if len(values) > 1 else 0,
                    'min': min(values),
                    'max': max(values),
                    'latest': values[-1] if values else None,
                    'trend': await self._calculate_trend(values),
                    'healing_weight': self.healing_metric_weights.get(dim, 1.0)
                }
        
        return summary
    
    async def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for values"""
        if len(values) < 3:
            return "insufficient_data"
        
        # Simple trend calculation using linear regression slope
        import numpy as np
        x = list(range(len(values)))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 1:
            return "improving"
        elif slope > 0.1:
            return "slightly_improving"
        elif slope > -0.1:
            return "stable"
        elif slope > -1:
            return "slightly_declining"
        else:
            return "declining"
    
    async def set_user_consent(self, user_id: str, dimension: WellbeingDimension, 
                             consent: bool):
        """Set user consent for metric collection"""
        self.consent_tracker[user_id][dimension.value] = consent
        logger.info(f"Consent set for user {user_id}, dimension {dimension.value}: {consent}")
    
    async def generate_privacy_report(self) -> Dict[str, Any]:
        """Generate privacy compliance report"""
        total_metrics = sum(len(metrics) for metrics in self.wellbeing_metrics.values())
        
        privacy_levels = defaultdict(int)
        consent_given = 0
        
        for metrics in self.wellbeing_metrics.values():
            for metric in metrics:
                privacy_levels[metric.privacy_level.value] += 1
                if metric.consent_given:
                    consent_given += 1
        
        return {
            'total_metrics_collected': total_metrics,
            'privacy_level_distribution': dict(privacy_levels),
            'consent_rate': consent_given / total_metrics if total_metrics > 0 else 0,
            'privacy_filters_active': len(self.privacy_filters),
            'anonymization_methods_active': len(self.data_anonymizers),
            'cultural_adaptations_available': len(self.cultural_metric_interpretations),
            'report_generated': datetime.now().isoformat()
        }
    
    async def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        if not self.registry or not generate_latest:
            return "# Prometheus client not available\n"
        
        return generate_latest(self.registry).decode('utf-8')
    
    async def add_reporting_callback(self, callback: Callable):
        """Add callback for metric reporting"""
        self.reporting_callbacks.append(callback)