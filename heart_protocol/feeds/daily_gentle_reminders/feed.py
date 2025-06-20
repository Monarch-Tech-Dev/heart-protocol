"""
Daily Gentle Reminders Feed - Main Orchestrator

The complete implementation of the first Sacred Feed: personalized affirmations
that remind users of their inherent worth with cultural sensitivity and trauma-informed care.

Core Philosophy: "You are worthy of love, especially when you forget."
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
import json

from ...core.base import FeedItem, CareLevel, FeedGenerator
from ...core.care_detection import CareDetectionEngine
from .content_database import AffirmationDatabase, Affirmation, AffirmationType, CulturalContext
from .personalization import EmotionalCapacityPersonalizer, EmotionalCapacityLevel
from .timing import OptimalTimingEngine, TimingWindow
from .cultural_sensitivity import CulturalSensitivityEngine, CulturalProfile
from .feedback_integration import FeedbackLearningSystem, FeedbackData, FeedbackType

logger = logging.getLogger(__name__)


class DailyGentleRemindersFeed(FeedGenerator):
    """
    The Daily Gentle Reminders Feed generates personalized affirmations that remind
    users of their inherent worth and value.
    
    Key Features:
    - Emotional capacity-aware personalization
    - Cultural sensitivity and inclusivity  
    - Optimal timing based on chronobiology
    - Continuous learning from user feedback
    - Crisis-responsive content adaptation
    - Trauma-informed care principles
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.feed_type = "daily_gentle_reminders"
        
        # Initialize core components
        self.affirmation_db = AffirmationDatabase()
        self.personalizer = EmotionalCapacityPersonalizer(config.get('personalization', {}))
        self.timing_engine = OptimalTimingEngine(config.get('timing', {}))
        self.cultural_engine = CulturalSensitivityEngine(config.get('cultural', {}))
        self.feedback_system = FeedbackLearningSystem(config.get('feedback', {}))
        
        # Cache for user contexts
        self.user_contexts = {}
        self.last_delivery_times = {}
        
        logger.info("Daily Gentle Reminders Feed initialized")
    
    async def generate_feed_items(self, user_id: str, 
                                user_context: Dict[str, Any],
                                care_assessment: Optional[Dict[str, Any]] = None) -> List[FeedItem]:
        """
        Generate personalized gentle reminder feed items for a user.
        
        Args:
            user_id: User identifier
            user_context: User's current context and preferences
            care_assessment: Optional care assessment from CareDetectionEngine
        """
        try:
            # Update user context cache
            self.user_contexts[user_id] = user_context
            
            # Assess current emotional capacity
            emotional_capacity = await self._assess_emotional_capacity(user_id, user_context, care_assessment)
            
            # Check if it's time for a gentle reminder
            should_deliver, timing_context = await self._should_deliver_reminder(user_id, user_context)
            
            if not should_deliver:
                logger.debug(f"Skipping reminder delivery for user {user_id[:8]}... - timing not optimal")
                return []
            
            # Create personalization context
            personalization_context = await self.personalizer.create_personalization_context(
                user_id, emotional_capacity, user_context
            )
            
            # Get cultural profile
            cultural_profile = await self.cultural_engine.get_cultural_profile(user_id)
            
            # Select optimal affirmation
            affirmation = await self._select_optimal_affirmation(
                user_context, personalization_context, cultural_profile
            )
            
            if not affirmation:
                logger.warning(f"No suitable affirmation found for user {user_id[:8]}...")
                return []
            
            # Apply cultural adaptations
            cultural_adaptation = await self.cultural_engine.adapt_affirmation_for_culture(
                affirmation, cultural_profile
            )
            
            # Create feed item
            feed_item = await self._create_feed_item(
                affirmation, cultural_adaptation, emotional_capacity, timing_context, user_context
            )
            
            # Record delivery
            self.last_delivery_times[user_id] = datetime.utcnow()
            
            logger.info(f"Generated gentle reminder for user {user_id[:8]}... "
                       f"(Capacity: {emotional_capacity.value}, Type: {affirmation.type.value})")
            
            return [feed_item]
            
        except Exception as e:
            logger.error(f"Error generating gentle reminders for user {user_id[:8]}...: {e}")
            return []
    
    async def _assess_emotional_capacity(self, user_id: str, 
                                       user_context: Dict[str, Any],
                                       care_assessment: Optional[Dict[str, Any]]) -> EmotionalCapacityLevel:
        """Assess user's current emotional capacity"""
        
        # Build indicators for capacity assessment
        capacity_indicators = {
            'crisis_indicators': care_assessment.get('crisis_detected', False) if care_assessment else False,
            'recent_care_requests': user_context.get('recent_care_seeking_posts', 0),
            'stress_indicators': user_context.get('current_stressors', []),
            'positive_coping_responses': user_context.get('recent_positive_activities', []),
            'social_engagement_level': user_context.get('social_interaction_level', 0.5),
            'user_self_assessment': user_context.get('self_reported_emotional_state')
        }
        
        # If care assessment indicates high need, prioritize that
        if care_assessment:
            care_level = care_assessment.get('care_level', CareLevel.MODERATE)
            if care_level in [CareLevel.HIGH, CareLevel.CRISIS]:
                capacity_indicators['crisis_indicators'] = True
        
        return await self.personalizer.assess_emotional_capacity(user_id, capacity_indicators)
    
    async def _should_deliver_reminder(self, user_id: str, 
                                     user_context: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Determine if it's appropriate to deliver a reminder now"""
        
        # Check basic delivery frequency
        last_delivery = self.last_delivery_times.get(user_id)
        if last_delivery:
            time_since_last = datetime.utcnow() - last_delivery
            min_interval = timedelta(hours=user_context.get('min_reminder_interval_hours', 8))
            
            if time_since_last < min_interval:
                return False, {'reason': 'too_recent'}
        
        # Check optimal timing
        optimal_windows = await self.timing_engine.calculate_optimal_timing(user_id, user_context)
        
        if not optimal_windows:
            return False, {'reason': 'no_optimal_windows'}
        
        # Check if current time falls within optimal windows
        current_time = datetime.now().time()
        
        for window in optimal_windows:
            if window.start_time <= current_time <= window.end_time:
                return True, {
                    'timing_window': window,
                    'effectiveness_score': window.effectiveness_score,
                    'reasoning': window.reasoning
                }
        
        # Check for crisis override
        if user_context.get('crisis_indicators', False):
            return True, {'reason': 'crisis_override', 'immediate_support': True}
        
        return False, {'reason': 'outside_optimal_windows'}
    
    async def _select_optimal_affirmation(self, user_context: Dict[str, Any],
                                        personalization_context: Dict[str, Any],
                                        cultural_profile: CulturalProfile) -> Optional[Affirmation]:
        """Select the most appropriate affirmation for the user"""
        
        # Get personalized affirmation from database
        database_context = {
            'emotional_capacity': personalization_context['emotional_capacity'],
            'cultural_preference': personalization_context['cultural_preference'],
            'preferred_affirmation_types': personalization_context['preferred_affirmation_types'],
            'avoided_types': personalization_context['avoided_types'],
            'current_needs_keywords': personalization_context['current_needs_keywords'],
            'avoided_keywords': personalization_context['avoided_keywords'],
            'crisis_indicators': user_context.get('crisis_indicators', False),
            'recent_affirmations': personalization_context.get('recent_affirmations', []),
            'season': self._get_current_season()
        }
        
        # Get candidate affirmation
        candidate_affirmation = await self.affirmation_db.get_personalized_affirmation(database_context)
        
        if not candidate_affirmation:
            return None
        
        # Use feedback system to predict effectiveness
        effectiveness_prediction = await self.feedback_system.predict_affirmation_effectiveness(
            candidate_affirmation.text, user_context
        )
        
        # If prediction is too low, try to get a better one
        if effectiveness_prediction < 0.4:
            # Try a few more candidates
            for _ in range(3):
                alternative = await self.affirmation_db.get_personalized_affirmation(database_context)
                if alternative:
                    alt_prediction = await self.feedback_system.predict_affirmation_effectiveness(
                        alternative.text, user_context
                    )
                    if alt_prediction > effectiveness_prediction:
                        candidate_affirmation = alternative
                        effectiveness_prediction = alt_prediction
        
        return candidate_affirmation
    
    def _get_current_season(self) -> str:
        """Get current season for seasonal affirmations"""
        month = datetime.now().month
        
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'autumn'
    
    async def _create_feed_item(self, affirmation: Affirmation,
                              cultural_adaptation: Any,
                              emotional_capacity: EmotionalCapacityLevel,
                              timing_context: Dict[str, Any],
                              user_context: Dict[str, Any]) -> FeedItem:
        """Create a feed item from the selected affirmation"""
        
        # Use culturally adapted text if available and confident
        if cultural_adaptation.confidence_score > 0.7:
            affirmation_text = cultural_adaptation.adapted_text
            cultural_notes = cultural_adaptation.reasoning
        else:
            affirmation_text = affirmation.text
            cultural_notes = "Using original text - cultural adaptation uncertain"
        
        # Create metadata
        metadata = {
            'affirmation_type': affirmation.type.value,
            'emotional_intensity': affirmation.emotional_intensity,
            'emotional_capacity_level': emotional_capacity.value,
            'cultural_context': cultural_adaptation.cultural_context.value,
            'cultural_adaptations_made': cultural_adaptation.adaptations_made,
            'cultural_confidence': cultural_adaptation.confidence_score,
            'timing_effectiveness': timing_context.get('effectiveness_score', 0.5),
            'timing_reasoning': timing_context.get('reasoning', ''),
            'keywords': affirmation.keywords,
            'avoid_when': affirmation.avoid_when,
            'usage_count': affirmation.usage_count,
            'effectiveness_score': affirmation.get_effectiveness_score(),
            'generated_at': datetime.utcnow().isoformat(),
            'feed_type': 'daily_gentle_reminders',
            'personalization_confidence': user_context.get('personalization_confidence', 0.5)
        }
        
        # Determine care level based on emotional capacity
        care_level_mapping = {
            EmotionalCapacityLevel.VERY_LOW: CareLevel.CRISIS,
            EmotionalCapacityLevel.LOW: CareLevel.HIGH,
            EmotionalCapacityLevel.MODERATE: CareLevel.MODERATE,
            EmotionalCapacityLevel.GOOD: CareLevel.LOW,
            EmotionalCapacityLevel.HIGH: CareLevel.LOW
        }
        
        care_level = care_level_mapping.get(emotional_capacity, CareLevel.MODERATE)
        
        # Create the feed item
        feed_item = FeedItem(
            content=affirmation_text,
            care_level=care_level,
            source_type="gentle_reminder",
            metadata=metadata,
            reasoning=f"Personalized affirmation for {emotional_capacity.value} emotional capacity. {cultural_notes}",
            confidence_score=min(
                cultural_adaptation.confidence_score,
                affirmation.get_effectiveness_score(),
                0.95  # Always leave room for uncertainty
            ),
            requires_human_review=cultural_adaptation.confidence_score < 0.5 or care_level == CareLevel.CRISIS
        )
        
        return feed_item
    
    async def record_user_feedback(self, user_id: str, feed_item_id: str, 
                                 feedback_data: Dict[str, Any]) -> bool:
        """Record user feedback about a gentle reminder"""
        
        try:
            # Parse feedback data
            feedback = FeedbackData(
                user_id=user_id,
                affirmation_id=feed_item_id,
                affirmation_text=feedback_data['affirmation_text'],
                feedback_type=FeedbackType(feedback_data['feedback_type']),
                rating=feedback_data['rating'],
                mood_before=feedback_data['mood_before'],
                mood_after=feedback_data['mood_after'],
                emotional_capacity_at_time=EmotionalCapacityLevel(feedback_data['emotional_capacity']),
                timing_rating=feedback_data.get('timing_rating', 3),
                cultural_appropriateness=feedback_data.get('cultural_appropriateness', 3),
                personal_relevance=feedback_data.get('personal_relevance', 3),
                written_feedback=feedback_data.get('written_feedback'),
                helpful_keywords=feedback_data.get('helpful_keywords', []),
                unhelpful_keywords=feedback_data.get('unhelpful_keywords', []),
                suggested_improvements=feedback_data.get('suggested_improvements'),
                would_share_with_friend=feedback_data.get('would_share_with_friend', False)
            )
            
            # Record feedback in learning system
            success = await self.feedback_system.record_feedback(feedback)
            
            if success:
                logger.info(f"Recorded feedback from user {user_id[:8]}... "
                           f"(Rating: {feedback.rating}, Type: {feedback.feedback_type.value})")
            
            return success
            
        except Exception as e:
            logger.error(f"Error recording user feedback: {e}")
            return False
    
    async def get_user_personalization_insights(self, user_id: str) -> Dict[str, Any]:
        """Get personalization insights for a user"""
        
        try:
            # Get insights from feedback system
            feedback_insights = await self.feedback_system.get_user_personalization_insights(user_id)
            
            # Get personalization recommendations from personalizer
            personalizer_recommendations = await self.personalizer.get_personalization_recommendations(user_id)
            
            # Get timing analytics
            timing_analytics = self.timing_engine.get_timing_analytics(user_id)
            
            # Combine insights
            combined_insights = {
                'user_id': user_id[:8] + '...',  # Anonymized
                'feedback_insights': feedback_insights,
                'personalization_recommendations': personalizer_recommendations,
                'timing_analytics': timing_analytics,
                'last_reminder_delivered': self.last_delivery_times.get(user_id, {}).get('isoformat', 'never'),
                'total_reminders_delivered': len([t for t in self.last_delivery_times.values() if t]),
                'generated_at': datetime.utcnow().isoformat()
            }
            
            return combined_insights
            
        except Exception as e:
            logger.error(f"Error getting personalization insights for user {user_id[:8]}...: {e}")
            return {'error': 'Unable to generate insights'}
    
    async def get_feed_analytics(self) -> Dict[str, Any]:
        """Get analytics about the gentle reminders feed"""
        
        try:
            # Get feedback analytics
            feedback_analytics = self.feedback_system.get_feedback_analytics()
            
            # Get database statistics
            db_stats = self.affirmation_db.get_database_stats()
            
            # Get cultural sensitivity metrics
            cultural_metrics = self.cultural_engine.get_cultural_sensitivity_metrics()
            
            # Get content recommendations
            content_recommendations = await self.feedback_system.generate_content_recommendations()
            
            # Calculate delivery statistics
            total_users = len(self.user_contexts)
            users_with_deliveries = len(self.last_delivery_times)
            
            analytics = {
                'feed_type': 'daily_gentle_reminders',
                'total_users_served': total_users,
                'users_with_deliveries': users_with_deliveries,
                'delivery_rate': users_with_deliveries / total_users if total_users > 0 else 0,
                'feedback_analytics': feedback_analytics,
                'content_database_stats': db_stats,
                'cultural_sensitivity_metrics': cultural_metrics,
                'content_improvement_recommendations': content_recommendations,
                'system_health': {
                    'affirmation_database_loaded': len(self.affirmation_db.affirmations) > 0,
                    'personalizer_active': True,
                    'timing_engine_active': True,
                    'cultural_engine_active': True,
                    'feedback_system_active': True
                },
                'generated_at': datetime.utcnow().isoformat()
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error generating feed analytics: {e}")
            return {'error': 'Unable to generate analytics'}
    
    async def update_user_cultural_profile(self, user_id: str, 
                                         cultural_data: Dict[str, Any]) -> bool:
        """Update user's cultural profile"""
        
        try:
            # Create or update cultural profile
            cultural_profile = await self.cultural_engine.create_cultural_profile(user_id, cultural_data)
            
            # Update personalization profile if needed
            if 'cultural_context' in cultural_data:
                await self.personalizer.update_user_profile(user_id, {
                    'cultural_context': cultural_data['cultural_context']
                })
            
            logger.info(f"Updated cultural profile for user {user_id[:8]}...")
            return True
            
        except Exception as e:
            logger.error(f"Error updating cultural profile for user {user_id[:8]}...: {e}")
            return False
    
    async def get_next_delivery_time(self, user_id: str, 
                                   user_context: Dict[str, Any]) -> Optional[datetime]:
        """Get the next optimal delivery time for a user"""
        
        try:
            return await self.timing_engine.get_next_optimal_delivery_time(user_id, user_context)
            
        except Exception as e:
            logger.error(f"Error calculating next delivery time for user {user_id[:8]}...: {e}")
            return None
    
    async def handle_crisis_mode(self, user_id: str, 
                               user_context: Dict[str, Any]) -> List[FeedItem]:
        """Generate immediate crisis support reminders"""
        
        try:
            logger.warning(f"Crisis mode activated for user {user_id[:8]}...")
            
            # Override normal timing for crisis
            crisis_context = user_context.copy()
            crisis_context['crisis_indicators'] = True
            crisis_context['emotional_capacity'] = EmotionalCapacityLevel.VERY_LOW
            
            # Generate immediate support
            feed_items = await self.generate_feed_items(user_id, crisis_context)
            
            if feed_items:
                # Mark as requiring human review
                for item in feed_items:
                    item.requires_human_review = True
                    item.metadata['crisis_support'] = True
                    item.metadata['human_escalation_required'] = True
            
            return feed_items
            
        except Exception as e:
            logger.error(f"Error handling crisis mode for user {user_id[:8]}...: {e}")
            return []
    
    async def export_learning_data(self) -> Dict[str, Any]:
        """Export learning data for analysis and improvement"""
        
        try:
            learning_insights = await self.feedback_system.export_learning_insights()
            
            export_data = {
                'export_timestamp': datetime.utcnow().isoformat(),
                'feed_type': 'daily_gentle_reminders',
                'learning_insights': learning_insights,
                'content_performance': [
                    {
                        'content': text[:100] + '...' if len(text) > 100 else text,
                        'metrics': metrics
                    }
                    for text, metrics in self.feedback_system.content_performance.items()
                ],
                'user_segments_learned': list(self.feedback_system.effectiveness_models.keys()),
                'total_feedback_points': len(self.feedback_system.feedback_history),
                'cultural_sensitivity_active': True,
                'trauma_informed_care_active': True
            }
            
            return export_data
            
        except Exception as e:
            logger.error(f"Error exporting learning data: {e}")
            return {'error': 'Unable to export learning data'}