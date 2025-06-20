"""
Algorithm Utilities for A/B Testing Integration

Utilities for integrating A/B testing into caring feed algorithms
and measuring care effectiveness across different approaches.
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging

from ..base import Post, FeedSkeleton, FeedType, CareLevel
from .ab_testing import (
    ExperimentManager, CareEffectivenessExperiment, ExperimentConfig,
    CareOutcome, ExperimentGroup
)

logger = logging.getLogger(__name__)


class AlgorithmUtils:
    """
    Utilities for integrating A/B testing into caring algorithms
    and measuring the effectiveness of different caring approaches.
    """
    
    def __init__(self, experiment_manager: ExperimentManager):
        self.experiment_manager = experiment_manager
        self.logger = logging.getLogger(f"{__name__}.AlgorithmUtils")
    
    async def get_algorithm_config_for_user(self, user_id: str, 
                                          base_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get algorithm configuration for user, potentially modified by A/B tests.
        """
        try:
            # Check if user is in any running experiments
            assignment = await self.experiment_manager.get_user_experiment_assignment(user_id)
            
            if assignment:
                experiment_id, group_name = assignment
                experiment = self.experiment_manager.experiments[experiment_id]
                
                # Get experimental algorithm config
                group = experiment.groups[group_name]
                experimental_config = group.algorithm_config
                
                # Merge with base config (experimental config takes precedence)
                merged_config = {**base_config, **experimental_config}
                
                self.logger.debug(f"User {user_id[:8]}... using experimental config from {group_name}")
                return merged_config
            
            # Return base configuration if not in experiment
            return base_config
            
        except Exception as e:
            self.logger.error(f"Error getting algorithm config for user: {e}")
            return base_config
    
    async def record_feed_interaction(self, user_id: str, feed_type: FeedType,
                                    interaction_type: str, outcome_value: float,
                                    metadata: Dict[str, Any] = None):
        """
        Record user interaction with feed for A/B testing analysis.
        """
        try:
            # Map interaction types to care outcomes
            outcome_mapping = {
                'feed_satisfaction': CareOutcome.FEED_SATISFACTION,
                'positive_feedback': CareOutcome.POSITIVE_FEEDBACK,
                'successful_connection': CareOutcome.SUCCESSFUL_CONNECTION,
                'help_seeking_success': CareOutcome.HELP_SEEKING_SUCCESS,
                'wellbeing_improvement': CareOutcome.USER_WELLBEING_IMPROVEMENT,
                'community_engagement': CareOutcome.COMMUNITY_ENGAGEMENT,
                'reduced_isolation': CareOutcome.REDUCED_ISOLATION
            }
            
            if interaction_type not in outcome_mapping:
                self.logger.warning(f"Unknown interaction type: {interaction_type}")
                return
            
            outcome_type = outcome_mapping[interaction_type]
            
            # Find all running experiments that include this user
            for experiment in self.experiment_manager.experiments.values():
                if experiment.status.value == "running":
                    # Check if user is in this experiment
                    user_group = None
                    for group_name, group in experiment.groups.items():
                        if user_id in group.user_ids:
                            user_group = group_name
                            break
                    
                    if user_group:
                        # Record outcome for this experiment
                        await experiment.record_outcome(
                            user_id=user_id,
                            outcome_type=outcome_type,
                            outcome_value=outcome_value,
                            metadata={
                                'feed_type': feed_type.value,
                                'interaction_type': interaction_type,
                                **(metadata or {})
                            }
                        )
                        
                        self.logger.debug(f"Recorded {interaction_type} outcome for user in experiment {experiment.experiment_id}")
            
        except Exception as e:
            self.logger.error(f"Error recording feed interaction: {e}")
    
    async def create_care_algorithm_experiment(self, 
                                             name: str,
                                             hypothesis: str,
                                             control_config: Dict[str, Any],
                                             test_configs: List[Dict[str, Any]],
                                             primary_outcome: CareOutcome = CareOutcome.USER_WELLBEING_IMPROVEMENT,
                                             duration_days: int = 30,
                                             target_sample_size: int = 1000) -> CareEffectivenessExperiment:
        """
        Create an A/B test experiment for comparing caring algorithm approaches.
        """
        try:
            # Create experiment configuration
            config = ExperimentConfig(
                name=name,
                description=f"Testing {len(test_configs) + 1} different caring algorithm approaches",
                hypothesis=hypothesis,
                primary_outcome=primary_outcome,
                secondary_outcomes=[
                    CareOutcome.FEED_SATISFACTION,
                    CareOutcome.SUCCESSFUL_CONNECTION,
                    CareOutcome.POSITIVE_FEEDBACK
                ],
                target_sample_size=target_sample_size,
                max_duration_days=duration_days
            )
            
            # Create experiment
            experiment = self.experiment_manager.create_experiment(config)
            
            # Add control group
            control_group = ExperimentGroup(
                name="control",
                description="Current caring algorithm approach",
                algorithm_config=control_config
            )
            experiment.add_group(control_group)
            
            # Add test groups
            for i, test_config in enumerate(test_configs):
                test_group = ExperimentGroup(
                    name=f"test_{i+1}",
                    description=f"Test algorithm variation {i+1}",
                    algorithm_config=test_config
                )
                experiment.add_group(test_group)
            
            self.logger.info(f"Created experiment '{name}' with {len(test_configs) + 1} groups")
            return experiment
            
        except Exception as e:
            self.logger.error(f"Error creating experiment: {e}")
            raise
    
    async def measure_feed_effectiveness(self, feed_posts: List[Post], 
                                       user_context: Dict[str, Any]) -> Dict[str, float]:
        """
        Measure predicted effectiveness of a feed before serving it to user.
        This helps with real-time optimization.
        """
        try:
            effectiveness_metrics = {
                'care_potential': 0.0,
                'emotional_appropriateness': 0.0,
                'diversity_score': 0.0,
                'safety_score': 0.0,
                'personalization_score': 0.0
            }
            
            if not feed_posts:
                return effectiveness_metrics
            
            # Analyze care potential
            care_scores = []
            for post in feed_posts:
                care_score = await self._calculate_post_care_potential(post)
                care_scores.append(care_score)
            effectiveness_metrics['care_potential'] = sum(care_scores) / len(care_scores)
            
            # Analyze emotional appropriateness
            user_emotional_capacity = user_context.get('emotional_capacity', {})
            emotional_scores = []
            for post in feed_posts:
                emotional_score = await self._assess_emotional_appropriateness(post, user_emotional_capacity)
                emotional_scores.append(emotional_score)
            effectiveness_metrics['emotional_appropriateness'] = sum(emotional_scores) / len(emotional_scores)
            
            # Calculate diversity score
            effectiveness_metrics['diversity_score'] = await self._calculate_feed_diversity(feed_posts)
            
            # Calculate safety score
            effectiveness_metrics['safety_score'] = await self._calculate_feed_safety(feed_posts, user_context)
            
            # Calculate personalization score
            effectiveness_metrics['personalization_score'] = await self._calculate_personalization(
                feed_posts, user_context
            )
            
            return effectiveness_metrics
            
        except Exception as e:
            self.logger.error(f"Error measuring feed effectiveness: {e}")
            return {'error': 1.0}
    
    async def _calculate_post_care_potential(self, post: Post) -> float:
        """Calculate how much care potential a post has"""
        text = post.text.lower()
        
        # Caring language indicators
        caring_indicators = [
            'support', 'help', 'love', 'care', 'understanding', 'empathy',
            'here for you', 'not alone', 'proud of you', 'you matter',
            'sending love', 'thinking of you', 'better days', 'hope'
        ]
        
        care_score = 0.0
        for indicator in caring_indicators:
            if indicator in text:
                care_score += 0.1
        
        # Boost for resource sharing
        resource_indicators = ['resource', 'helpful', 'try this', 'worked for me', 'therapy', 'counseling']
        for indicator in resource_indicators:
            if indicator in text:
                care_score += 0.15
        
        return min(1.0, care_score)
    
    async def _assess_emotional_appropriateness(self, post: Post, 
                                              emotional_capacity: Dict[str, float]) -> float:
        """Assess if post is emotionally appropriate for user's current capacity"""
        text = post.text.lower()
        
        # Heavy content indicators
        heavy_indicators = ['trigger warning', 'suicide', 'self harm', 'abuse', 'trauma', 'crisis']
        is_heavy_content = any(indicator in text for indicator in heavy_indicators)
        
        # User's capacity to handle heavy content
        can_handle_heavy = emotional_capacity.get('can_handle_heavy_content', 0.5)
        
        if is_heavy_content:
            return can_handle_heavy  # Appropriate only if user can handle it
        else:
            # Light content is generally appropriate
            return 0.9
    
    async def _calculate_feed_diversity(self, posts: List[Post]) -> float:
        """Calculate diversity score for the feed"""
        if not posts:
            return 0.0
        
        # Check author diversity
        authors = [post.author for post in posts]
        unique_authors = len(set(authors))
        author_diversity = unique_authors / len(posts)
        
        # Check content type diversity (simplified)
        content_types = []
        for post in posts:
            if any(word in post.text.lower() for word in ['question', '?', 'help', 'advice']):
                content_types.append('question')
            elif any(word in post.text.lower() for word in ['grateful', 'thank', 'appreciation']):
                content_types.append('gratitude')
            elif any(word in post.text.lower() for word in ['progress', 'better', 'healing']):
                content_types.append('progress')
            else:
                content_types.append('general')
        
        unique_content_types = len(set(content_types))
        content_diversity = unique_content_types / len(posts)
        
        # Combined diversity score
        return (author_diversity + content_diversity) / 2
    
    async def _calculate_feed_safety(self, posts: List[Post], user_context: Dict[str, Any]) -> float:
        """Calculate safety score for the feed"""
        if not posts:
            return 1.0
        
        safety_scores = []
        user_triggers = user_context.get('care_preferences', {}).get('trigger_warnings', [])
        
        for post in posts:
            post_safety = 1.0
            
            # Check for user's specific triggers
            text = post.text.lower()
            for trigger in user_triggers:
                if trigger.lower() in text:
                    post_safety = 0.0
                    break
            
            # Check for general unsafe content
            unsafe_indicators = ['self harm', 'suicide method', 'how to die', 'end my life']
            for indicator in unsafe_indicators:
                if indicator in text:
                    post_safety = min(post_safety, 0.2)
            
            safety_scores.append(post_safety)
        
        return sum(safety_scores) / len(safety_scores)
    
    async def _calculate_personalization(self, posts: List[Post], 
                                       user_context: Dict[str, Any]) -> float:
        """Calculate how well the feed is personalized for the user"""
        if not posts:
            return 0.0
        
        user_preferences = user_context.get('care_preferences', {})
        preferred_care_types = user_preferences.get('care_types', [])
        
        if not preferred_care_types:
            return 0.5  # Neutral if no preferences known
        
        personalization_scores = []
        
        for post in posts:
            post_score = 0.0
            text = post.text.lower()
            
            # Check alignment with user's preferred care types
            for care_type in preferred_care_types:
                if care_type == 'gentle_reminders' and any(word in text for word in ['you matter', 'worthy', 'enough']):
                    post_score = max(post_score, 0.8)
                elif care_type == 'peer_support' and any(word in text for word in ['been there', 'understand', 'similar']):
                    post_score = max(post_score, 0.8)
                elif care_type == 'resources' and any(word in text for word in ['helpful', 'resource', 'try this']):
                    post_score = max(post_score, 0.8)
            
            personalization_scores.append(post_score)
        
        return sum(personalization_scores) / len(personalization_scores)
    
    async def generate_algorithm_recommendations(self, experiment_results: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations for improving caring algorithms based on A/B test results.
        """
        recommendations = []
        
        try:
            if 'basic_comparison' in experiment_results:
                comparison = experiment_results['basic_comparison']
                
                if 'better_group' in comparison:
                    better_group = comparison['better_group']
                    improvement = comparison.get('difference', 0)
                    
                    recommendations.append(
                        f"Adopt the {better_group} algorithm configuration, which showed "
                        f"{improvement:.1%} improvement in care outcomes."
                    )
            
            # Analyze group-specific insights
            if 'groups' in experiment_results:
                for group_name, group_data in experiment_results['groups'].items():
                    outcome_breakdown = group_data.get('outcome_breakdown', {})
                    
                    # Find the best performing outcome types
                    best_outcomes = []
                    for outcome_type, metrics in outcome_breakdown.items():
                        if isinstance(metrics, dict) and 'mean' in metrics:
                            if metrics['mean'] > 0.7:  # High performing outcome
                                best_outcomes.append(outcome_type)
                    
                    if best_outcomes:
                        recommendations.append(
                            f"The {group_name} group showed particularly strong results in: "
                            f"{', '.join(best_outcomes)}. Consider emphasizing these aspects."
                        )
            
            # General recommendations
            recommendations.extend([
                "Continue monitoring long-term effects on user wellbeing",
                "Test with larger, more diverse user groups",
                "Implement gradual rollout of successful algorithm changes",
                "Maintain human oversight for all algorithm modifications"
            ])
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            recommendations.append("Review experiment data manually due to analysis error")
        
        return recommendations