"""
A/B Testing Framework for Care Effectiveness

Ethical experimentation framework to measure and improve the caring effectiveness
of Heart Protocol algorithms while maintaining user wellbeing as the primary goal.
"""

import asyncio
import random
import hashlib
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging
import json

logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """Status of care effectiveness experiments"""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    TERMINATED = "terminated"  # Stopped early due to safety concerns


class CareOutcome(Enum):
    """Types of care outcomes we measure"""
    USER_WELLBEING_IMPROVEMENT = "wellbeing_improvement"
    SUCCESSFUL_CONNECTION = "successful_connection"
    CRISIS_INTERVENTION_SUCCESS = "crisis_intervention_success"
    POSITIVE_FEEDBACK = "positive_feedback"
    FEED_SATISFACTION = "feed_satisfaction"
    REDUCED_ISOLATION = "reduced_isolation"
    COMMUNITY_ENGAGEMENT = "community_engagement"
    HELP_SEEKING_SUCCESS = "help_seeking_success"


@dataclass
class ExperimentConfig:
    """Configuration for care effectiveness experiments"""
    name: str
    description: str
    hypothesis: str
    primary_outcome: CareOutcome
    secondary_outcomes: List[CareOutcome]
    target_sample_size: int
    max_duration_days: int
    confidence_level: float = 0.95
    minimum_effect_size: float = 0.05  # Minimum meaningful improvement
    safety_thresholds: Dict[str, float] = field(default_factory=dict)
    ethical_constraints: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.safety_thresholds:
            self.safety_thresholds = {
                'max_negative_feedback_rate': 0.05,  # Stop if >5% negative feedback
                'min_crisis_intervention_success': 0.95,  # Stop if <95% crisis success
                'max_user_distress_increase': 0.02  # Stop if >2% report increased distress
            }
        
        if not self.ethical_constraints:
            self.ethical_constraints = {
                'vulnerable_user_exclusion': True,  # Don't experiment on vulnerable users
                'informed_consent_required': False,  # Users don't need to know about A/B test
                'easy_opt_out': True,  # Users can always opt out
                'human_oversight': True,  # Human review of all experiments
                'no_harmful_interventions': True  # Never test potentially harmful approaches
            }


@dataclass
class ExperimentGroup:
    """Represents a group in an A/B test"""
    name: str
    description: str
    algorithm_config: Dict[str, Any]
    user_ids: List[str] = field(default_factory=list)
    outcomes: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class CareOutcomeEvent:
    """Records a care outcome event"""
    user_id: str
    experiment_id: str
    group_name: str
    outcome_type: CareOutcome
    outcome_value: float  # 0.0 to 1.0, higher is better
    metadata: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'user_id_hash': self._hash_user_id(self.user_id),
            'experiment_id': self.experiment_id,
            'group_name': self.group_name,
            'outcome_type': self.outcome_type.value,
            'outcome_value': self.outcome_value,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat()
        }
    
    def _hash_user_id(self, user_id: str) -> str:
        """Hash user ID for privacy"""
        return hashlib.sha256(user_id.encode()).hexdigest()[:16]


class CareEffectivenessExperiment:
    """
    Framework for running ethical A/B tests on caring algorithms.
    
    Core principles:
    - User wellbeing is always the primary goal
    - No experiments that could cause harm
    - Vulnerable users are protected
    - Statistical rigor with ethical constraints
    - Transparent reporting of results
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.experiment_id = self._generate_experiment_id()
        self.status = ExperimentStatus.DRAFT
        self.groups = {}
        self.start_time = None
        self.end_time = None
        self.safety_monitor = ExperimentSafetyMonitor(config.safety_thresholds)
        self.logger = logging.getLogger(f"{__name__}.{config.name}")
        
        # Results tracking
        self.outcome_events = []
        self.daily_metrics = {}
        self.safety_checks = []
    
    def _generate_experiment_id(self) -> str:
        """Generate unique experiment ID"""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        random_suffix = random.randint(1000, 9999)
        return f"care_exp_{timestamp}_{random_suffix}"
    
    def add_group(self, group: ExperimentGroup) -> bool:
        """Add an experimental group"""
        try:
            if self.status != ExperimentStatus.DRAFT:
                self.logger.error("Cannot add groups to running experiment")
                return False
            
            self.groups[group.name] = group
            self.logger.info(f"Added group '{group.name}' to experiment")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding group: {e}")
            return False
    
    async def assign_user_to_group(self, user_id: str) -> Optional[str]:
        """
        Assign user to experimental group using stable randomization.
        Ensures same user always gets same group assignment.
        """
        try:
            # Check if user should be excluded from experiments
            if await self._should_exclude_user(user_id):
                return None
            
            # Stable hash-based assignment
            user_hash = hashlib.md5(f"{user_id}_{self.experiment_id}".encode()).hexdigest()
            assignment_value = int(user_hash[:8], 16) / (16**8)
            
            # Assign to groups based on equal probability
            group_names = list(self.groups.keys())
            group_index = int(assignment_value * len(group_names))
            assigned_group = group_names[group_index]
            
            # Add user to group
            self.groups[assigned_group].user_ids.append(user_id)
            
            self.logger.info(f"Assigned user to group '{assigned_group}'")
            return assigned_group
            
        except Exception as e:
            self.logger.error(f"Error assigning user to group: {e}")
            return None
    
    async def _should_exclude_user(self, user_id: str) -> bool:
        """
        Check if user should be excluded from experiments.
        Protects vulnerable users and respects preferences.
        """
        if not self.config.ethical_constraints.get('vulnerable_user_exclusion', True):
            return False
        
        # In production, this would check:
        # - User's vulnerability indicators
        # - Recent crisis interventions
        # - Explicit opt-out preferences
        # - Age restrictions
        # - Mental health status
        
        # For now, exclude randomly 10% to simulate vulnerable user protection
        user_hash = hashlib.md5(user_id.encode()).hexdigest()
        exclusion_value = int(user_hash[:2], 16) / 256
        return exclusion_value < 0.1  # Exclude 10% for safety
    
    async def start_experiment(self) -> bool:
        """Start the experiment with safety checks"""
        try:
            # Validate experiment setup
            if not await self._validate_experiment_setup():
                return False
            
            # Ethical review check
            if not await self._pass_ethical_review():
                return False
            
            # Start experiment
            self.status = ExperimentStatus.RUNNING
            self.start_time = datetime.utcnow()
            
            # Schedule safety monitoring
            asyncio.create_task(self._monitor_experiment_safety())
            
            self.logger.info(f"Started experiment '{self.config.name}'")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting experiment: {e}")
            return False
    
    async def _validate_experiment_setup(self) -> bool:
        """Validate experiment configuration"""
        if len(self.groups) < 2:
            self.logger.error("Need at least 2 groups for experiment")
            return False
        
        if self.config.target_sample_size < 100:
            self.logger.error("Sample size too small for meaningful results")
            return False
        
        return True
    
    async def _pass_ethical_review(self) -> bool:
        """Check if experiment passes ethical review"""
        # In production, this would involve human ethical review board
        
        # Check for potentially harmful interventions
        for group in self.groups.values():
            if await self._is_potentially_harmful(group.algorithm_config):
                self.logger.error(f"Group '{group.name}' may be harmful")
                return False
        
        return True
    
    async def _is_potentially_harmful(self, algorithm_config: Dict[str, Any]) -> bool:
        """Check if algorithm configuration could be harmful"""
        # Check for configurations that could harm users
        harmful_patterns = [
            'reduce_crisis_intervention',
            'ignore_help_requests',
            'amplify_negative_content',
            'disable_safety_features'
        ]
        
        config_str = json.dumps(algorithm_config).lower()
        return any(pattern in config_str for pattern in harmful_patterns)
    
    async def record_outcome(self, user_id: str, outcome_type: CareOutcome, 
                           outcome_value: float, metadata: Dict[str, Any] = None) -> bool:
        """Record a care outcome event"""
        try:
            # Find user's group assignment
            user_group = None
            for group_name, group in self.groups.items():
                if user_id in group.user_ids:
                    user_group = group_name
                    break
            
            if not user_group:
                # User not in experiment
                return False
            
            # Create outcome event
            event = CareOutcomeEvent(
                user_id=user_id,
                experiment_id=self.experiment_id,
                group_name=user_group,
                outcome_type=outcome_type,
                outcome_value=outcome_value,
                metadata=metadata or {}
            )
            
            # Store event
            self.outcome_events.append(event)
            self.groups[user_group].outcomes.append(event.to_dict())
            
            # Check for safety concerns
            await self.safety_monitor.check_outcome_event(event)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error recording outcome: {e}")
            return False
    
    async def _monitor_experiment_safety(self):
        """Continuously monitor experiment for safety issues"""
        while self.status == ExperimentStatus.RUNNING:
            try:
                # Check safety thresholds
                safety_check = await self.safety_monitor.run_safety_check(self)
                self.safety_checks.append(safety_check)
                
                if not safety_check['safe_to_continue']:
                    await self._terminate_for_safety(safety_check['reasons'])
                    break
                
                # Check if experiment should end naturally
                if await self._should_end_experiment():
                    await self.complete_experiment()
                    break
                
                # Wait before next check
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                self.logger.error(f"Error in safety monitoring: {e}")
                await asyncio.sleep(300)  # Retry in 5 minutes
    
    async def _terminate_for_safety(self, reasons: List[str]):
        """Terminate experiment due to safety concerns"""
        self.status = ExperimentStatus.TERMINATED
        self.end_time = datetime.utcnow()
        
        self.logger.warning(f"TERMINATED experiment for safety: {reasons}")
        
        # Notify human oversight team
        await self._notify_human_oversight("SAFETY_TERMINATION", {
            'experiment_id': self.experiment_id,
            'reasons': reasons,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    async def _should_end_experiment(self) -> bool:
        """Check if experiment should end naturally"""
        if not self.start_time:
            return False
        
        # Check duration
        duration = datetime.utcnow() - self.start_time
        if duration.days >= self.config.max_duration_days:
            return True
        
        # Check sample size
        total_users = sum(len(group.user_ids) for group in self.groups.values())
        if total_users >= self.config.target_sample_size:
            # Check if we have statistical power
            return await self._has_sufficient_statistical_power()
        
        return False
    
    async def _has_sufficient_statistical_power(self) -> bool:
        """Check if we have enough data for meaningful results"""
        # Simplified statistical power calculation
        # In production, this would be more sophisticated
        
        total_outcomes = len(self.outcome_events)
        return total_outcomes >= 50  # Minimum outcomes for analysis
    
    async def complete_experiment(self) -> Dict[str, Any]:
        """Complete experiment and generate results"""
        try:
            self.status = ExperimentStatus.COMPLETED
            self.end_time = datetime.utcnow()
            
            # Generate results
            results = await self._analyze_results()
            
            # Generate report
            report = await self._generate_experiment_report(results)
            
            self.logger.info(f"Completed experiment '{self.config.name}'")
            return report
            
        except Exception as e:
            self.logger.error(f"Error completing experiment: {e}")
            return {}
    
    async def _analyze_results(self) -> Dict[str, Any]:
        """Analyze experimental results"""
        results = {
            'groups': {},
            'statistical_significance': {},
            'effect_sizes': {},
            'confidence_intervals': {}
        }
        
        # Analyze each group
        for group_name, group in self.groups.items():
            group_outcomes = [event for event in self.outcome_events 
                            if event.group_name == group_name]
            
            results['groups'][group_name] = {
                'sample_size': len(group.user_ids),
                'total_outcomes': len(group_outcomes),
                'outcome_breakdown': self._analyze_group_outcomes(group_outcomes)
            }
        
        # Statistical analysis would be more comprehensive in production
        # For now, provide basic comparison
        if len(self.groups) == 2:
            results['basic_comparison'] = await self._compare_two_groups()
        
        return results
    
    def _analyze_group_outcomes(self, outcomes: List[CareOutcomeEvent]) -> Dict[str, Any]:
        """Analyze outcomes for a specific group"""
        if not outcomes:
            return {'no_data': True}
        
        outcome_by_type = {}
        for outcome in outcomes:
            outcome_type = outcome.outcome_type.value
            if outcome_type not in outcome_by_type:
                outcome_by_type[outcome_type] = []
            outcome_by_type[outcome_type].append(outcome.outcome_value)
        
        analysis = {}
        for outcome_type, values in outcome_by_type.items():
            analysis[outcome_type] = {
                'count': len(values),
                'mean': sum(values) / len(values),
                'min': min(values),
                'max': max(values)
            }
        
        return analysis
    
    async def _compare_two_groups(self) -> Dict[str, Any]:
        """Basic comparison between two groups"""
        group_names = list(self.groups.keys())
        if len(group_names) != 2:
            return {}
        
        group_a_outcomes = [e for e in self.outcome_events if e.group_name == group_names[0]]
        group_b_outcomes = [e for e in self.outcome_events if e.group_name == group_names[1]]
        
        if not group_a_outcomes or not group_b_outcomes:
            return {'insufficient_data': True}
        
        # Simple mean comparison (would be more sophisticated in production)
        mean_a = sum(e.outcome_value for e in group_a_outcomes) / len(group_a_outcomes)
        mean_b = sum(e.outcome_value for e in group_b_outcomes) / len(group_b_outcomes)
        
        return {
            'group_a_mean': mean_a,
            'group_b_mean': mean_b,
            'difference': mean_b - mean_a,
            'better_group': group_names[1] if mean_b > mean_a else group_names[0],
            'note': 'This is a simplified analysis. Production would include proper statistical testing.'
        }
    
    async def _generate_experiment_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive experiment report"""
        return {
            'experiment_info': {
                'id': self.experiment_id,
                'name': self.config.name,
                'description': self.config.description,
                'hypothesis': self.config.hypothesis,
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'end_time': self.end_time.isoformat() if self.end_time else None,
                'status': self.status.value
            },
            'groups': {name: {
                'description': group.description,
                'sample_size': len(group.user_ids),
                'algorithm_config': group.algorithm_config
            } for name, group in self.groups.items()},
            'results': results,
            'safety_summary': {
                'safety_checks_passed': len([c for c in self.safety_checks if c['safe_to_continue']]),
                'safety_concerns': len([c for c in self.safety_checks if not c['safe_to_continue']]),
                'terminated_for_safety': self.status == ExperimentStatus.TERMINATED
            },
            'ethical_compliance': {
                'vulnerable_users_protected': self.config.ethical_constraints.get('vulnerable_user_exclusion', True),
                'human_oversight_maintained': self.config.ethical_constraints.get('human_oversight', True),
                'no_harmful_interventions': True  # Verified during setup
            },
            'recommendations': await self._generate_recommendations(results)
        }
    
    async def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on results"""
        recommendations = []
        
        if 'basic_comparison' in results:
            comparison = results['basic_comparison']
            if 'better_group' in comparison:
                recommendations.append(
                    f"Consider adopting the algorithm configuration from {comparison['better_group']} "
                    f"which showed {comparison['difference']:.3f} improvement in care outcomes."
                )
        
        recommendations.append(
            "Conduct follow-up studies with larger sample sizes for more definitive results."
        )
        
        recommendations.append(
            "Continue monitoring long-term effects of algorithm changes on user wellbeing."
        )
        
        return recommendations
    
    async def _notify_human_oversight(self, event_type: str, data: Dict[str, Any]):
        """Notify human oversight team of important events"""
        # In production, this would send alerts to human reviewers
        self.logger.info(f"Human oversight notification: {event_type} - {data}")


class ExperimentSafetyMonitor:
    """Monitors experiments for safety issues"""
    
    def __init__(self, safety_thresholds: Dict[str, float]):
        self.thresholds = safety_thresholds
        self.logger = logging.getLogger(f"{__name__.SafetyMonitor}")
    
    async def check_outcome_event(self, event: CareOutcomeEvent):
        """Check individual outcome event for safety concerns"""
        # Check for extremely negative outcomes
        if event.outcome_value < 0.1 and event.outcome_type in [
            CareOutcome.USER_WELLBEING_IMPROVEMENT,
            CareOutcome.CRISIS_INTERVENTION_SUCCESS
        ]:
            self.logger.warning(f"Very negative outcome detected: {event.outcome_type.value}")
    
    async def run_safety_check(self, experiment: 'CareEffectivenessExperiment') -> Dict[str, Any]:
        """Run comprehensive safety check on experiment"""
        safety_issues = []
        
        # Check negative feedback rate
        negative_outcomes = [e for e in experiment.outcome_events if e.outcome_value < 0.3]
        total_outcomes = len(experiment.outcome_events)
        
        if total_outcomes > 10:  # Only check if we have enough data
            negative_rate = len(negative_outcomes) / total_outcomes
            if negative_rate > self.thresholds.get('max_negative_feedback_rate', 0.05):
                safety_issues.append(f"High negative feedback rate: {negative_rate:.2%}")
        
        # Check crisis intervention success rate
        crisis_outcomes = [e for e in experiment.outcome_events 
                          if e.outcome_type == CareOutcome.CRISIS_INTERVENTION_SUCCESS]
        if crisis_outcomes:
            crisis_success_rate = sum(e.outcome_value for e in crisis_outcomes) / len(crisis_outcomes)
            min_success_rate = self.thresholds.get('min_crisis_intervention_success', 0.95)
            if crisis_success_rate < min_success_rate:
                safety_issues.append(f"Low crisis intervention success: {crisis_success_rate:.2%}")
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'safe_to_continue': len(safety_issues) == 0,
            'reasons': safety_issues,
            'metrics_checked': list(self.thresholds.keys())
        }


class ExperimentManager:
    """Manages multiple care effectiveness experiments"""
    
    def __init__(self):
        self.experiments = {}
        self.logger = logging.getLogger(f"{__name__}.ExperimentManager")
    
    def create_experiment(self, config: ExperimentConfig) -> CareEffectivenessExperiment:
        """Create a new experiment"""
        experiment = CareEffectivenessExperiment(config)
        self.experiments[experiment.experiment_id] = experiment
        return experiment
    
    async def get_user_experiment_assignment(self, user_id: str) -> Optional[Tuple[str, str]]:
        """Get user's current experiment assignment (experiment_id, group_name)"""
        for experiment in self.experiments.values():
            if experiment.status == ExperimentStatus.RUNNING:
                group_name = await experiment.assign_user_to_group(user_id)
                if group_name:
                    return experiment.experiment_id, group_name
        return None
    
    def get_experiment_results(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get results for completed experiment"""
        if experiment_id in self.experiments:
            experiment = self.experiments[experiment_id]
            if experiment.status == ExperimentStatus.COMPLETED:
                return experiment._generate_experiment_report({})  # Would use cached results
        return None