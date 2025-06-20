"""
Heart Protocol A/B Testing Example

Example of how to set up and run ethical A/B tests for caring algorithms.
"""

import asyncio
from datetime import datetime, timedelta
from heart_protocol.core.feed_generators import (
    ExperimentManager, ExperimentConfig, ExperimentGroup,
    CareOutcome, AlgorithmUtils
)


async def example_caring_algorithm_experiment():
    """
    Example: Testing different approaches to gentle reminder generation
    """
    
    # Create experiment manager
    experiment_manager = ExperimentManager()
    
    # Define experiment configuration
    config = ExperimentConfig(
        name="gentle_reminders_optimization",
        description="Testing different approaches to generating gentle daily reminders",
        hypothesis="More personalized reminders based on user's recent emotional context will be more effective than generic reminders",
        primary_outcome=CareOutcome.USER_WELLBEING_IMPROVEMENT,
        secondary_outcomes=[
            CareOutcome.FEED_SATISFACTION,
            CareOutcome.POSITIVE_FEEDBACK
        ],
        target_sample_size=500,
        max_duration_days=14,  # 2 weeks
        confidence_level=0.95,
        minimum_effect_size=0.05
    )
    
    # Create experiment
    experiment = experiment_manager.create_experiment(config)
    
    # Define control group (current approach)
    control_config = {
        'reminder_style': 'generic',
        'personalization_level': 0.3,
        'emotional_context_weight': 0.2,
        'frequency_per_day': 1,
        'time_sensitivity': False
    }
    
    control_group = ExperimentGroup(
        name="control_generic",
        description="Current generic gentle reminders approach",
        algorithm_config=control_config
    )
    
    # Define test group 1 (personalized approach)
    test1_config = {
        'reminder_style': 'personalized',
        'personalization_level': 0.8,
        'emotional_context_weight': 0.6,
        'frequency_per_day': 1,
        'time_sensitivity': True,
        'user_preference_adaptation': True
    }
    
    test1_group = ExperimentGroup(
        name="test_personalized",
        description="Highly personalized reminders based on emotional context",
        algorithm_config=test1_config
    )
    
    # Define test group 2 (frequency optimization)
    test2_config = {
        'reminder_style': 'adaptive_frequency',
        'personalization_level': 0.5,
        'emotional_context_weight': 0.4,
        'frequency_per_day': 'adaptive',  # 1-3 based on user need
        'time_sensitivity': True,
        'stress_level_adjustment': True
    }
    
    test2_group = ExperimentGroup(
        name="test_adaptive",
        description="Adaptive frequency based on user stress levels",
        algorithm_config=test2_config
    )
    
    # Add groups to experiment
    experiment.add_group(control_group)
    experiment.add_group(test1_group)
    experiment.add_group(test2_group)
    
    # Start experiment
    success = await experiment.start_experiment()
    if not success:
        print("Failed to start experiment")
        return
    
    print(f"Started experiment: {experiment.config.name}")
    print(f"Experiment ID: {experiment.experiment_id}")
    
    # Simulate user interactions over time
    await simulate_user_interactions(experiment)
    
    # Complete experiment and analyze results
    results = await experiment.complete_experiment()
    
    print("\n=== EXPERIMENT RESULTS ===")
    print(f"Status: {results['experiment_info']['status']}")
    print(f"Duration: {results['experiment_info']['start_time']} to {results['experiment_info']['end_time']}")
    
    for group_name, group_data in results['groups'].items():
        print(f"\nGroup: {group_name}")
        print(f"  Sample size: {group_data['sample_size']}")
        print(f"  Algorithm: {group_data['algorithm_config']}")
    
    if 'basic_comparison' in results['results']:
        comparison = results['results']['basic_comparison']
        print(f"\nBest performing group: {comparison.get('better_group', 'unclear')}")
        print(f"Improvement: {comparison.get('difference', 0):.3f}")
    
    print(f"\nRecommendations:")
    for rec in results['recommendations']:
        print(f"  - {rec}")
    
    return results


async def simulate_user_interactions(experiment):
    """
    Simulate user interactions with different algorithm configurations
    """
    print("\nSimulating user interactions...")
    
    # Simulate 100 users over 2 weeks
    for user_id in range(100):
        user_id_str = f"user_{user_id:03d}"
        
        # Assign user to experimental group
        group_name = await experiment.assign_user_to_group(user_id_str)
        if not group_name:
            continue  # User excluded from experiment
        
        # Simulate daily interactions for 14 days
        for day in range(14):
            # Simulate various outcomes based on algorithm configuration
            group_config = experiment.groups[group_name].algorithm_config
            
            # Simulate wellbeing improvement (varies by algorithm)
            if group_config.get('personalization_level', 0) > 0.7:
                wellbeing_score = 0.75 + (day * 0.02)  # Gradual improvement
            elif group_config.get('frequency_per_day') == 'adaptive':
                wellbeing_score = 0.70 + (day * 0.015)
            else:
                wellbeing_score = 0.65 + (day * 0.01)  # Control group
            
            # Add some randomness
            import random
            wellbeing_score += random.uniform(-0.1, 0.1)
            wellbeing_score = max(0, min(1, wellbeing_score))
            
            # Record outcome
            await experiment.record_outcome(
                user_id=user_id_str,
                outcome_type=CareOutcome.USER_WELLBEING_IMPROVEMENT,
                outcome_value=wellbeing_score,
                metadata={'day': day, 'algorithm_config': group_config}
            )
            
            # Occasionally record other outcomes
            if day % 3 == 0:  # Every 3 days
                satisfaction_score = wellbeing_score + random.uniform(-0.05, 0.15)
                satisfaction_score = max(0, min(1, satisfaction_score))
                
                await experiment.record_outcome(
                    user_id=user_id_str,
                    outcome_type=CareOutcome.FEED_SATISFACTION,
                    outcome_value=satisfaction_score,
                    metadata={'day': day}
                )
            
            if day % 7 == 0 and wellbeing_score > 0.7:  # Weekly positive feedback
                await experiment.record_outcome(
                    user_id=user_id_str,
                    outcome_type=CareOutcome.POSITIVE_FEEDBACK,
                    outcome_value=0.8,
                    metadata={'day': day, 'feedback_type': 'weekly_check_in'}
                )
    
    print(f"Simulated interactions for {len(experiment.outcome_events)} outcomes")


async def example_safety_monitoring():
    """
    Example of how safety monitoring works during experiments
    """
    print("\n=== SAFETY MONITORING EXAMPLE ===")
    
    # Create a potentially problematic experiment for demonstration
    experiment_manager = ExperimentManager()
    
    config = ExperimentConfig(
        name="safety_demo",
        description="Demonstration of safety monitoring",
        hypothesis="This is just a demo",
        primary_outcome=CareOutcome.USER_WELLBEING_IMPROVEMENT,
        target_sample_size=50,
        max_duration_days=7,
        safety_thresholds={
            'max_negative_feedback_rate': 0.03,  # Very strict threshold
            'min_crisis_intervention_success': 0.98
        }
    )
    
    experiment = experiment_manager.create_experiment(config)
    
    # Add a control group
    control_group = ExperimentGroup(
        name="control",
        description="Standard approach",
        algorithm_config={'safety_level': 'high'}
    )
    experiment.add_group(control_group)
    
    await experiment.start_experiment()
    
    # Simulate some negative outcomes to trigger safety monitoring
    for i in range(20):
        user_id = f"demo_user_{i}"
        group_name = await experiment.assign_user_to_group(user_id)
        
        if group_name:
            # Simulate negative outcome (low wellbeing score)
            negative_score = 0.2  # Low score indicates negative outcome
            
            await experiment.record_outcome(
                user_id=user_id,
                outcome_type=CareOutcome.USER_WELLBEING_IMPROVEMENT,
                outcome_value=negative_score,
                metadata={'simulated': True}
            )
    
    # Run safety check
    safety_check = await experiment.safety_monitor.run_safety_check(experiment)
    
    print(f"Safety check result: {'SAFE' if safety_check['safe_to_continue'] else 'UNSAFE'}")
    if not safety_check['safe_to_continue']:
        print(f"Safety concerns: {safety_check['reasons']}")
        print("Experiment would be automatically terminated for safety.")
    
    return safety_check


async def main():
    """Run all examples"""
    print("Heart Protocol A/B Testing Examples")
    print("=" * 50)
    
    # Run caring algorithm experiment
    await example_caring_algorithm_experiment()
    
    # Demonstrate safety monitoring
    await example_safety_monitoring()
    
    print("\n=== EXAMPLES COMPLETE ===")
    print("\nKey takeaways:")
    print("1. A/B tests prioritize user wellbeing over statistical significance")
    print("2. Safety monitoring can automatically stop harmful experiments")
    print("3. All experiments include human oversight and ethical review")
    print("4. Results focus on care outcomes, not engagement metrics")


if __name__ == "__main__":
    asyncio.run(main())