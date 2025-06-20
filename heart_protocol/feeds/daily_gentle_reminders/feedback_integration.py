"""
User Feedback Integration for Content Improvement

Learns from user responses to improve affirmation personalization and effectiveness.
Built on principles of user agency and continuous caring improvement.
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
import logging
import json

from .content_database import Affirmation, AffirmationType, CulturalContext
from .personalization import EmotionalCapacityLevel

logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    """Types of feedback users can provide"""
    HELPFUL = "helpful"           # This helped me feel better
    NOT_HELPFUL = "not_helpful"   # This didn't resonate with me
    INAPPROPRIATE = "inappropriate" # This felt wrong for my situation
    TOO_INTENSE = "too_intense"   # This felt overwhelming
    TOO_GENTLE = "too_gentle"     # This felt insufficient
    WRONG_TIMING = "wrong_timing" # Bad timing for this message
    CULTURAL_MISMATCH = "cultural_mismatch" # Didn't fit my cultural context
    CRISIS_HELPFUL = "crisis_helpful" # This helped during crisis
    TRIGGERING = "triggering"     # This was harmful/triggering


class FeedbackMood(Enum):
    """User's mood before and after the affirmation"""
    MUCH_WORSE = "much_worse"     # 1
    WORSE = "worse"               # 2
    SAME = "same"                 # 3
    BETTER = "better"             # 4
    MUCH_BETTER = "much_better"   # 5


@dataclass
class FeedbackData:
    """Comprehensive feedback data from user"""
    user_id: str
    affirmation_id: str
    affirmation_text: str
    feedback_type: FeedbackType
    rating: int  # 1-5 scale
    mood_before: FeedbackMood
    mood_after: FeedbackMood
    emotional_capacity_at_time: EmotionalCapacityLevel
    timing_rating: int  # 1-5 scale for timing appropriateness
    cultural_appropriateness: int  # 1-5 scale
    personal_relevance: int  # 1-5 scale
    written_feedback: Optional[str] = None
    helpful_keywords: List[str] = None
    unhelpful_keywords: List[str] = None
    suggested_improvements: Optional[str] = None
    would_share_with_friend: bool = False
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.helpful_keywords is None:
            self.helpful_keywords = []
        if self.unhelpful_keywords is None:
            self.unhelpful_keywords = []
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


@dataclass
class LearningInsight:
    """Insight learned from feedback analysis"""
    insight_type: str
    user_segment: str  # e.g., "low_emotional_capacity", "collectivist_culture"
    finding: str
    confidence: float
    sample_size: int
    actionable_recommendation: str
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


class FeedbackLearningSystem:
    """
    Learns from user feedback to continuously improve affirmation effectiveness.
    
    Key Features:
    - Real-time learning from user responses
    - Personalization optimization
    - Cultural sensitivity improvement
    - Crisis response effectiveness tracking
    - Predictive modeling for content selection
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.feedback_history = []  # In production, database-backed
        self.user_preferences = {}  # Learned preferences per user
        self.effectiveness_models = {}  # ML models for effectiveness prediction
        self.learning_insights = []  # Discovered insights
        self.content_performance = {}  # Performance metrics per affirmation
        
        # Learning parameters
        self.min_feedback_for_learning = config.get('min_feedback_for_learning', 5)
        self.learning_confidence_threshold = config.get('learning_confidence_threshold', 0.7)
        self.insight_update_frequency = timedelta(hours=config.get('insight_update_hours', 24))
        
        logger.info("Feedback Learning System initialized")
    
    async def record_feedback(self, feedback: FeedbackData) -> bool:
        """Record user feedback and trigger learning updates"""
        
        try:
            # Validate feedback
            if not await self._validate_feedback(feedback):
                logger.warning(f"Invalid feedback received from user {feedback.user_id[:8]}...")
                return False
            
            # Store feedback
            self.feedback_history.append(feedback)
            
            # Update content performance metrics
            await self._update_content_performance(feedback)
            
            # Update user preference model
            await self._update_user_preferences(feedback)
            
            # Check for immediate learning opportunities
            await self._check_immediate_learning(feedback)
            
            # Update effectiveness models
            await self._update_effectiveness_models(feedback)
            
            logger.info(f"Recorded feedback from user {feedback.user_id[:8]}... "
                       f"(Type: {feedback.feedback_type.value}, Rating: {feedback.rating})")
            
            return True
            
        except Exception as e:
            logger.error(f"Error recording feedback: {e}")
            return False
    
    async def _validate_feedback(self, feedback: FeedbackData) -> bool:
        """Validate feedback data quality and completeness"""
        
        # Check required fields
        if not feedback.user_id or not feedback.affirmation_text:
            return False
        
        # Check rating ranges
        if not (1 <= feedback.rating <= 5):
            return False
        
        if not (1 <= feedback.timing_rating <= 5):
            return False
        
        # Check for spam/abuse patterns
        recent_feedback = [
            f for f in self.feedback_history 
            if f.user_id == feedback.user_id and 
               f.timestamp > datetime.utcnow() - timedelta(minutes=5)
        ]
        
        if len(recent_feedback) > 10:  # Too much feedback too quickly
            logger.warning(f"Potential spam feedback from user {feedback.user_id[:8]}...")
            return False
        
        return True
    
    async def _update_content_performance(self, feedback: FeedbackData) -> None:
        """Update performance metrics for the specific affirmation"""
        
        affirmation_key = feedback.affirmation_text
        
        if affirmation_key not in self.content_performance:
            self.content_performance[affirmation_key] = {
                'total_feedback': 0,
                'average_rating': 0.0,
                'mood_improvement': 0.0,
                'cultural_appropriateness': 0.0,
                'timing_appropriateness': 0.0,
                'effectiveness_by_capacity': {},
                'feedback_types': {},
                'last_updated': datetime.utcnow()
            }
        
        performance = self.content_performance[affirmation_key]
        
        # Update aggregated metrics
        total = performance['total_feedback']
        performance['average_rating'] = (
            (performance['average_rating'] * total + feedback.rating) / (total + 1)
        )
        
        # Calculate mood improvement
        mood_before_score = self._mood_to_score(feedback.mood_before)
        mood_after_score = self._mood_to_score(feedback.mood_after)
        mood_improvement = mood_after_score - mood_before_score
        
        performance['mood_improvement'] = (
            (performance['mood_improvement'] * total + mood_improvement) / (total + 1)
        )
        
        # Update other metrics
        performance['cultural_appropriateness'] = (
            (performance['cultural_appropriateness'] * total + feedback.cultural_appropriateness) / (total + 1)
        )
        
        performance['timing_appropriateness'] = (
            (performance['timing_appropriateness'] * total + feedback.timing_rating) / (total + 1)
        )
        
        # Track effectiveness by emotional capacity
        capacity_key = feedback.emotional_capacity_at_time.value
        if capacity_key not in performance['effectiveness_by_capacity']:
            performance['effectiveness_by_capacity'][capacity_key] = {
                'count': 0,
                'average_rating': 0.0,
                'mood_improvement': 0.0
            }
        
        capacity_metrics = performance['effectiveness_by_capacity'][capacity_key]
        capacity_count = capacity_metrics['count']
        
        capacity_metrics['average_rating'] = (
            (capacity_metrics['average_rating'] * capacity_count + feedback.rating) / (capacity_count + 1)
        )
        
        capacity_metrics['mood_improvement'] = (
            (capacity_metrics['mood_improvement'] * capacity_count + mood_improvement) / (capacity_count + 1)
        )
        
        capacity_metrics['count'] += 1
        
        # Track feedback types
        feedback_type_key = feedback.feedback_type.value
        performance['feedback_types'][feedback_type_key] = \
            performance['feedback_types'].get(feedback_type_key, 0) + 1
        
        performance['total_feedback'] += 1
        performance['last_updated'] = datetime.utcnow()
    
    def _mood_to_score(self, mood: FeedbackMood) -> float:
        """Convert mood enum to numerical score"""
        mood_scores = {
            FeedbackMood.MUCH_WORSE: 1.0,
            FeedbackMood.WORSE: 2.0,
            FeedbackMood.SAME: 3.0,
            FeedbackMood.BETTER: 4.0,
            FeedbackMood.MUCH_BETTER: 5.0
        }
        return mood_scores.get(mood, 3.0)
    
    async def _update_user_preferences(self, feedback: FeedbackData) -> None:
        """Update learned preferences for the specific user"""
        
        user_id = feedback.user_id
        
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {
                'preferred_types': {},
                'avoided_types': {},
                'preferred_keywords': {},
                'avoided_keywords': {},
                'optimal_intensity': 0.5,
                'preferred_timing': {},
                'cultural_preferences': {},
                'mood_patterns': {},
                'feedback_count': 0,
                'last_updated': datetime.utcnow()
            }
        
        prefs = self.user_preferences[user_id]
        
        # Update type preferences based on feedback
        if feedback.rating >= 4:  # Positive feedback
            affirmation_type = await self._infer_affirmation_type(feedback.affirmation_text)
            if affirmation_type:
                prefs['preferred_types'][affirmation_type] = \
                    prefs['preferred_types'].get(affirmation_type, 0) + 1
            
            # Update preferred keywords
            for keyword in feedback.helpful_keywords:
                prefs['preferred_keywords'][keyword] = \
                    prefs['preferred_keywords'].get(keyword, 0) + 1
        
        elif feedback.rating <= 2:  # Negative feedback
            affirmation_type = await self._infer_affirmation_type(feedback.affirmation_text)
            if affirmation_type:
                prefs['avoided_types'][affirmation_type] = \
                    prefs['avoided_types'].get(affirmation_type, 0) + 1
            
            # Update avoided keywords
            for keyword in feedback.unhelpful_keywords:
                prefs['avoided_keywords'][keyword] = \
                    prefs['avoided_keywords'].get(keyword, 0) + 1
        
        # Update timing preferences
        if feedback.timing_rating >= 4:
            hour = feedback.timestamp.hour
            prefs['preferred_timing'][str(hour)] = \
                prefs['preferred_timing'].get(str(hour), 0) + 1
        
        # Update mood pattern tracking
        mood_key = f"{feedback.mood_before.value}_to_{feedback.mood_after.value}"
        prefs['mood_patterns'][mood_key] = \
            prefs['mood_patterns'].get(mood_key, 0) + 1
        
        prefs['feedback_count'] += 1
        prefs['last_updated'] = datetime.utcnow()
    
    async def _infer_affirmation_type(self, affirmation_text: str) -> Optional[str]:
        """Infer affirmation type from text (simplified approach)"""
        
        # Keywords associated with different types
        type_keywords = {
            'self_worth': ['worthy', 'deserve', 'value', 'matter', 'enough'],
            'resilience_building': ['strong', 'overcome', 'survived', 'resilient', 'fighter'],
            'hope_nurturing': ['tomorrow', 'future', 'hope', 'possibilities', 'better'],
            'self_compassion': ['kind', 'gentle', 'forgive', 'rest', 'patience'],
            'healing_support': ['heal', 'recovery', 'time', 'wounds', 'growing'],
            'progress_acknowledgment': ['progress', 'step', 'forward', 'growth', 'change']
        }
        
        text_lower = affirmation_text.lower()
        type_scores = {}
        
        for affirmation_type, keywords in type_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                type_scores[affirmation_type] = score
        
        if type_scores:
            return max(type_scores.items(), key=lambda x: x[1])[0]
        
        return None
    
    async def _check_immediate_learning(self, feedback: FeedbackData) -> None:
        """Check for immediate learning opportunities from this feedback"""
        
        # Check for crisis feedback patterns
        if feedback.emotional_capacity_at_time == EmotionalCapacityLevel.VERY_LOW:
            await self._analyze_crisis_feedback(feedback)
        
        # Check for cultural appropriateness issues
        if feedback.cultural_appropriateness <= 2 or feedback.feedback_type == FeedbackType.CULTURAL_MISMATCH:
            await self._analyze_cultural_feedback(feedback)
        
        # Check for triggering content
        if feedback.feedback_type == FeedbackType.TRIGGERING:
            await self._handle_triggering_content_feedback(feedback)
        
        # Check for timing issues
        if feedback.timing_rating <= 2 or feedback.feedback_type == FeedbackType.WRONG_TIMING:
            await self._analyze_timing_feedback(feedback)
    
    async def _analyze_crisis_feedback(self, feedback: FeedbackData) -> None:
        """Analyze feedback from users in crisis situations"""
        
        if feedback.feedback_type == FeedbackType.CRISIS_HELPFUL:
            # Learn what helps during crisis
            insight = LearningInsight(
                insight_type="crisis_helpful_pattern",
                user_segment="very_low_emotional_capacity",
                finding=f"Affirmation effective during crisis: '{feedback.affirmation_text[:50]}...'",
                confidence=0.8,
                sample_size=1,
                actionable_recommendation="Prioritize similar affirmations for crisis situations"
            )
            self.learning_insights.append(insight)
            
        elif feedback.rating <= 2:
            # Learn what doesn't help during crisis
            insight = LearningInsight(
                insight_type="crisis_unhelpful_pattern",
                user_segment="very_low_emotional_capacity",
                finding=f"Affirmation ineffective during crisis: '{feedback.affirmation_text[:50]}...'",
                confidence=0.8,
                sample_size=1,
                actionable_recommendation="Avoid similar affirmations during crisis"
            )
            self.learning_insights.append(insight)
    
    async def _analyze_cultural_feedback(self, feedback: FeedbackData) -> None:
        """Analyze cultural appropriateness feedback"""
        
        if feedback.written_feedback:
            insight = LearningInsight(
                insight_type="cultural_mismatch",
                user_segment="cultural_sensitivity",
                finding=f"Cultural issue reported: {feedback.written_feedback}",
                confidence=0.9,
                sample_size=1,
                actionable_recommendation="Review cultural adaptation algorithms"
            )
            self.learning_insights.append(insight)
            
            logger.warning(f"Cultural mismatch reported for: '{feedback.affirmation_text[:50]}...'")
    
    async def _handle_triggering_content_feedback(self, feedback: FeedbackData) -> None:
        """Handle feedback about triggering content"""
        
        # Immediately flag content for review
        insight = LearningInsight(
            insight_type="triggering_content",
            user_segment="all_users",
            finding=f"Content reported as triggering: '{feedback.affirmation_text}'",
            confidence=1.0,
            sample_size=1,
            actionable_recommendation="Immediate review and potential removal of content"
        )
        self.learning_insights.append(insight)
        
        logger.error(f"TRIGGERING CONTENT REPORTED: '{feedback.affirmation_text}'")
        
        # In production, this would trigger immediate human review
    
    async def _analyze_timing_feedback(self, feedback: FeedbackData) -> None:
        """Analyze timing appropriateness feedback"""
        
        hour = feedback.timestamp.hour
        insight = LearningInsight(
            insight_type="timing_issue",
            user_segment=f"time_preference_{feedback.user_id[:8]}",
            finding=f"Poor timing at hour {hour} for user",
            confidence=0.7,
            sample_size=1,
            actionable_recommendation=f"Avoid hour {hour} for this user"
        )
        self.learning_insights.append(insight)
    
    async def _update_effectiveness_models(self, feedback: FeedbackData) -> None:
        """Update ML models for predicting affirmation effectiveness"""
        
        # Simplified model update - in production would use proper ML frameworks
        user_segment = self._get_user_segment(feedback)
        
        if user_segment not in self.effectiveness_models:
            self.effectiveness_models[user_segment] = {
                'training_data': [],
                'model_performance': 0.5,
                'last_trained': datetime.utcnow()
            }
        
        # Add training data point
        training_point = {
            'affirmation_features': await self._extract_affirmation_features(feedback.affirmation_text),
            'user_context': {
                'emotional_capacity': feedback.emotional_capacity_at_time.value,
                'mood_before': self._mood_to_score(feedback.mood_before),
                'timestamp_hour': feedback.timestamp.hour
            },
            'effectiveness_score': feedback.rating / 5.0,
            'mood_improvement': (
                self._mood_to_score(feedback.mood_after) - 
                self._mood_to_score(feedback.mood_before)
            ) / 4.0  # Normalize to 0-1
        }
        
        self.effectiveness_models[user_segment]['training_data'].append(training_point)
        
        # Retrain if enough data
        if len(self.effectiveness_models[user_segment]['training_data']) >= self.min_feedback_for_learning:
            await self._retrain_effectiveness_model(user_segment)
    
    def _get_user_segment(self, feedback: FeedbackData) -> str:
        """Get user segment for model training"""
        return f"capacity_{feedback.emotional_capacity_at_time.value}"
    
    async def _extract_affirmation_features(self, affirmation_text: str) -> Dict[str, float]:
        """Extract features from affirmation text for ML"""
        
        # Simplified feature extraction
        features = {
            'length': len(affirmation_text) / 200.0,  # Normalize
            'word_count': len(affirmation_text.split()) / 30.0,  # Normalize
            'contains_you': float('you' in affirmation_text.lower()),
            'contains_strength': float('strong' in affirmation_text.lower() or 'strength' in affirmation_text.lower()),
            'contains_love': float('love' in affirmation_text.lower()),
            'contains_healing': float('heal' in affirmation_text.lower()),
            'contains_future': float(any(word in affirmation_text.lower() for word in ['tomorrow', 'future', 'will'])),
            'emotional_intensity': await self._estimate_emotional_intensity(affirmation_text)
        }
        
        return features
    
    async def _estimate_emotional_intensity(self, text: str) -> float:
        """Estimate emotional intensity of affirmation text"""
        
        # Intensity markers
        high_intensity_words = ['powerful', 'incredible', 'amazing', 'extraordinary', 'magnificent']
        medium_intensity_words = ['strong', 'good', 'worthy', 'capable', 'valuable']
        gentle_words = ['gentle', 'soft', 'quiet', 'peaceful', 'calm']
        
        text_lower = text.lower()
        
        high_count = sum(1 for word in high_intensity_words if word in text_lower)
        medium_count = sum(1 for word in medium_intensity_words if word in text_lower)
        gentle_count = sum(1 for word in gentle_words if word in text_lower)
        
        if high_count > 0:
            return 0.8 + (high_count * 0.1)  # 0.8-1.0
        elif medium_count > 0:
            return 0.5 + (medium_count * 0.1)  # 0.5-0.7
        elif gentle_count > 0:
            return 0.2 + (gentle_count * 0.1)  # 0.2-0.4
        else:
            return 0.5  # Default medium intensity
    
    async def _retrain_effectiveness_model(self, user_segment: str) -> None:
        """Retrain effectiveness prediction model for user segment"""
        
        # Simplified retraining - in production would use proper ML
        training_data = self.effectiveness_models[user_segment]['training_data']
        
        if len(training_data) < self.min_feedback_for_learning:
            return
        
        # Calculate simple effectiveness patterns
        total_improvement = sum(point['mood_improvement'] for point in training_data)
        average_improvement = total_improvement / len(training_data)
        
        # Update model performance estimate
        self.effectiveness_models[user_segment]['model_performance'] = min(0.9, 0.5 + average_improvement)
        self.effectiveness_models[user_segment]['last_trained'] = datetime.utcnow()
        
        logger.info(f"Retrained effectiveness model for {user_segment} with {len(training_data)} data points")
    
    async def predict_affirmation_effectiveness(self, affirmation_text: str, 
                                             user_context: Dict[str, Any]) -> float:
        """Predict how effective an affirmation will be for a user"""
        
        try:
            user_segment = f"capacity_{user_context.get('emotional_capacity', 'moderate')}"
            
            if user_segment not in self.effectiveness_models:
                return 0.5  # Default prediction
            
            model = self.effectiveness_models[user_segment]
            
            if len(model['training_data']) < self.min_feedback_for_learning:
                return 0.5  # Not enough data for prediction
            
            # Extract features for this affirmation
            features = await self._extract_affirmation_features(affirmation_text)
            
            # Simple prediction based on historical patterns
            # In production, would use proper ML models
            training_data = model['training_data']
            
            similar_affirmations = []
            for point in training_data:
                similarity = await self._calculate_feature_similarity(features, point['affirmation_features'])
                if similarity > 0.7:  # Threshold for similarity
                    similar_affirmations.append(point)
            
            if similar_affirmations:
                avg_effectiveness = sum(point['effectiveness_score'] for point in similar_affirmations) / len(similar_affirmations)
                return avg_effectiveness
            
            return model['model_performance']
            
        except Exception as e:
            logger.error(f"Error predicting affirmation effectiveness: {e}")
            return 0.5
    
    async def _calculate_feature_similarity(self, features1: Dict[str, float], 
                                          features2: Dict[str, float]) -> float:
        """Calculate similarity between feature vectors"""
        
        common_keys = set(features1.keys()) & set(features2.keys())
        if not common_keys:
            return 0.0
        
        # Simple cosine similarity
        dot_product = sum(features1[key] * features2[key] for key in common_keys)
        norm1 = sum(features1[key] ** 2 for key in common_keys) ** 0.5
        norm2 = sum(features2[key] ** 2 for key in common_keys) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    async def get_user_personalization_insights(self, user_id: str) -> Dict[str, Any]:
        """Get personalization insights for a specific user"""
        
        if user_id not in self.user_preferences:
            return {'insufficient_data': True}
        
        prefs = self.user_preferences[user_id]
        
        # Get most preferred and avoided types
        preferred_types = sorted(prefs['preferred_types'].items(), key=lambda x: x[1], reverse=True)
        avoided_types = sorted(prefs['avoided_types'].items(), key=lambda x: x[1], reverse=True)
        
        # Get preferred keywords
        preferred_keywords = sorted(prefs['preferred_keywords'].items(), key=lambda x: x[1], reverse=True)[:5]
        avoided_keywords = sorted(prefs['avoided_keywords'].items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Get optimal timing
        preferred_times = sorted(prefs['preferred_timing'].items(), key=lambda x: x[1], reverse=True)[:3]
        
        return {
            'feedback_count': prefs['feedback_count'],
            'most_preferred_types': preferred_types[:3],
            'most_avoided_types': avoided_types[:3],
            'preferred_keywords': preferred_keywords,
            'avoided_keywords': avoided_keywords,
            'optimal_timing_hours': [int(time) for time, _ in preferred_times],
            'personalization_confidence': min(1.0, prefs['feedback_count'] / 20),
            'last_updated': prefs['last_updated'].isoformat()
        }
    
    async def generate_content_recommendations(self) -> List[Dict[str, Any]]:
        """Generate recommendations for content improvements"""
        
        recommendations = []
        
        # Analyze low-performing content
        low_performers = [
            (text, metrics) for text, metrics in self.content_performance.items()
            if metrics['total_feedback'] >= 3 and metrics['average_rating'] < 3.0
        ]
        
        for text, metrics in low_performers[:5]:  # Top 5 worst performers
            recommendations.append({
                'type': 'content_improvement',
                'content': text[:50] + '...',
                'issue': f"Low rating: {metrics['average_rating']:.1f}/5.0",
                'suggestion': 'Consider revising or removing this content',
                'priority': 'high' if metrics['average_rating'] < 2.5 else 'medium'
            })
        
        # Analyze cultural issues
        cultural_issues = [
            insight for insight in self.learning_insights
            if insight.insight_type == 'cultural_mismatch'
        ]
        
        if cultural_issues:
            recommendations.append({
                'type': 'cultural_sensitivity',
                'issue': f"{len(cultural_issues)} cultural mismatch reports",
                'suggestion': 'Review cultural adaptation algorithms',
                'priority': 'high'
            })
        
        # Analyze timing issues
        timing_insights = [
            insight for insight in self.learning_insights
            if insight.insight_type == 'timing_issue'
        ]
        
        if len(timing_insights) > 5:
            recommendations.append({
                'type': 'timing_optimization',
                'issue': f"{len(timing_insights)} timing complaints",
                'suggestion': 'Review and improve timing algorithms',
                'priority': 'medium'
            })
        
        return recommendations
    
    def get_feedback_analytics(self) -> Dict[str, Any]:
        """Get analytics about feedback patterns"""
        
        if not self.feedback_history:
            return {'no_data': True}
        
        total_feedback = len(self.feedback_history)
        
        # Calculate averages
        avg_rating = sum(f.rating for f in self.feedback_history) / total_feedback
        avg_cultural_appropriateness = sum(f.cultural_appropriateness for f in self.feedback_history) / total_feedback
        avg_timing_rating = sum(f.timing_rating for f in self.feedback_history) / total_feedback
        
        # Count feedback types
        feedback_type_counts = {}
        for feedback in self.feedback_history:
            feedback_type_counts[feedback.feedback_type.value] = \
                feedback_type_counts.get(feedback.feedback_type.value, 0) + 1
        
        # Calculate mood improvement
        mood_improvements = []
        for feedback in self.feedback_history:
            improvement = self._mood_to_score(feedback.mood_after) - self._mood_to_score(feedback.mood_before)
            mood_improvements.append(improvement)
        
        avg_mood_improvement = sum(mood_improvements) / len(mood_improvements)
        
        # Recent feedback trends
        recent_feedback = [
            f for f in self.feedback_history
            if f.timestamp > datetime.utcnow() - timedelta(days=7)
        ]
        
        return {
            'total_feedback_received': total_feedback,
            'average_rating': round(avg_rating, 2),
            'average_cultural_appropriateness': round(avg_cultural_appropriateness, 2),
            'average_timing_rating': round(avg_timing_rating, 2),
            'average_mood_improvement': round(avg_mood_improvement, 2),
            'feedback_type_distribution': feedback_type_counts,
            'recent_feedback_count': len(recent_feedback),
            'users_providing_feedback': len(set(f.user_id for f in self.feedback_history)),
            'content_items_rated': len(self.content_performance),
            'learning_insights_generated': len(self.learning_insights),
            'user_segments_learned': len(self.effectiveness_models),
            'learning_confidence': self.learning_confidence_threshold
        }
    
    async def export_learning_insights(self) -> List[Dict[str, Any]]:
        """Export learning insights for human review"""
        
        return [
            {
                'insight_type': insight.insight_type,
                'user_segment': insight.user_segment,
                'finding': insight.finding,
                'confidence': insight.confidence,
                'sample_size': insight.sample_size,
                'recommendation': insight.actionable_recommendation,
                'created_at': insight.created_at.isoformat()
            }
            for insight in self.learning_insights
        ]