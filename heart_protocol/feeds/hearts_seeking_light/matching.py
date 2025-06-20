"""
Connection Matching Algorithm

Intelligently matches support seekers with appropriate support offers
based on compatibility, safety, consent, and caring effectiveness.
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
import logging

from .support_detection import SupportSeeker, SupportOffer, SupportType, SupportUrgency
from .consent_system import ConsentManager, ConsentLevel, ConsentScope
from .safety_protocols import SafetyProtocolEngine, SafetyLevel, SafetyAssessment
from .intervention_timing import InterventionTimingEngine, InterventionTiming

logger = logging.getLogger(__name__)


class MatchQuality(Enum):
    """Quality levels for support matches"""
    EXCELLENT = "excellent"         # 0.9-1.0
    GOOD = "good"                  # 0.7-0.89
    ADEQUATE = "adequate"          # 0.5-0.69
    POOR = "poor"                  # 0.3-0.49
    INCOMPATIBLE = "incompatible"  # 0.0-0.29


class MatchingCriteria(Enum):
    """Criteria used in matching algorithm"""
    SUPPORT_TYPE_COMPATIBILITY = "support_type_compatibility"
    EXPERIENCE_RELEVANCE = "experience_relevance"
    COMMUNICATION_COMPATIBILITY = "communication_compatibility"
    DEMOGRAPHIC_PREFERENCES = "demographic_preferences"
    AVAILABILITY_ALIGNMENT = "availability_alignment"
    CULTURAL_COMPATIBILITY = "cultural_compatibility"
    SAFETY_ASSESSMENT = "safety_assessment"
    CONSENT_COMPATIBILITY = "consent_compatibility"
    CRISIS_APPROPRIATENESS = "crisis_appropriateness"
    HEALING_STAGE_ALIGNMENT = "healing_stage_alignment"


@dataclass
class MatchScore:
    """Detailed scoring for a potential match"""
    overall_score: float
    quality_level: MatchQuality
    criteria_scores: Dict[MatchingCriteria, float]
    safety_assessment: SafetyAssessment
    consent_assessment: Dict[str, Any]
    timing_assessment: InterventionTiming
    match_reasoning: str
    success_probability: float
    care_effectiveness_prediction: float
    recommendations: List[str]
    concerns: List[str]
    
    def is_viable_match(self) -> bool:
        """Check if this is a viable match"""
        return (self.overall_score >= 0.5 and 
                self.safety_assessment.is_connection_safe() and
                self.consent_assessment.get('has_sufficient_consent', False))


@dataclass
class Connection:
    """Represents a matched connection between seeker and supporter"""
    connection_id: str
    seeker: SupportSeeker
    supporter: SupportOffer
    match_score: MatchScore
    created_at: datetime
    status: str  # 'pending', 'active', 'completed', 'terminated'
    interaction_history: List[Dict[str, Any]]
    monitoring_data: Dict[str, Any]
    success_metrics: Dict[str, Any]


class ConnectionMatcher:
    """
    Intelligent matching algorithm that connects support seekers with helpers
    based on compatibility, safety, consent, and potential for positive impact.
    
    Core Principles:
    - Safety first: Never compromise user safety for algorithmic efficiency
    - Consent-driven: Only match when proper consent exists
    - Quality over quantity: Better fewer high-quality matches than many poor ones
    - Care effectiveness: Optimize for healing and growth, not engagement
    - Cultural sensitivity: Respect diverse backgrounds and preferences
    - Trauma-informed: Prevent retraumatization through thoughtful matching
    """
    
    def __init__(self, config: Dict[str, Any], 
                 consent_manager: ConsentManager,
                 safety_engine: SafetyProtocolEngine,
                 timing_engine: InterventionTimingEngine):
        self.config = config
        self.consent_manager = consent_manager
        self.safety_engine = safety_engine
        self.timing_engine = timing_engine
        
        # Matching algorithms and scoring weights
        self.scoring_weights = self._initialize_scoring_weights()
        self.compatibility_algorithms = self._initialize_compatibility_algorithms()
        self.success_prediction_model = self._initialize_success_model()
        
        # Active connections and matching history
        self.active_connections = {}  # connection_id -> Connection
        self.matching_history = []    # Historical matches for learning
        
        # Matching metrics for optimization
        self.matching_metrics = {
            'matches_attempted': 0,
            'matches_successful': 0,
            'matches_declined': 0,
            'connections_completed_successfully': 0,
            'average_match_quality': 0.0,
            'care_effectiveness_achieved': 0.0
        }
        
        logger.info("Connection Matcher initialized")
    
    def _initialize_scoring_weights(self) -> Dict[MatchingCriteria, float]:
        """Initialize weights for different matching criteria"""
        
        return {
            MatchingCriteria.SAFETY_ASSESSMENT: 0.25,         # Highest priority
            MatchingCriteria.SUPPORT_TYPE_COMPATIBILITY: 0.20,
            MatchingCriteria.CONSENT_COMPATIBILITY: 0.15,
            MatchingCriteria.EXPERIENCE_RELEVANCE: 0.12,
            MatchingCriteria.CRISIS_APPROPRIATENESS: 0.10,
            MatchingCriteria.COMMUNICATION_COMPATIBILITY: 0.08,
            MatchingCriteria.AVAILABILITY_ALIGNMENT: 0.05,
            MatchingCriteria.CULTURAL_COMPATIBILITY: 0.03,
            MatchingCriteria.DEMOGRAPHIC_PREFERENCES: 0.02
        }
    
    def _initialize_compatibility_algorithms(self) -> Dict[MatchingCriteria, callable]:
        """Initialize compatibility scoring algorithms"""
        
        return {
            MatchingCriteria.SUPPORT_TYPE_COMPATIBILITY: self._score_support_type_compatibility,
            MatchingCriteria.EXPERIENCE_RELEVANCE: self._score_experience_relevance,
            MatchingCriteria.COMMUNICATION_COMPATIBILITY: self._score_communication_compatibility,
            MatchingCriteria.DEMOGRAPHIC_PREFERENCES: self._score_demographic_preferences,
            MatchingCriteria.AVAILABILITY_ALIGNMENT: self._score_availability_alignment,
            MatchingCriteria.CULTURAL_COMPATIBILITY: self._score_cultural_compatibility,
            MatchingCriteria.CRISIS_APPROPRIATENESS: self._score_crisis_appropriateness,
            MatchingCriteria.HEALING_STAGE_ALIGNMENT: self._score_healing_stage_alignment
        }
    
    def _initialize_success_model(self) -> Dict[str, Any]:
        """Initialize success prediction model"""
        
        return {
            'success_factors': {
                'match_quality_threshold': 0.7,
                'safety_score_weight': 0.3,
                'experience_alignment_weight': 0.25,
                'communication_compatibility_weight': 0.2,
                'support_type_match_weight': 0.15,
                'cultural_compatibility_weight': 0.1
            },
            'risk_factors': {
                'high_vulnerability_penalty': -0.2,
                'inexperienced_supporter_penalty': -0.15,
                'communication_mismatch_penalty': -0.1,
                'timezone_mismatch_penalty': -0.05
            },
            'success_indicators': {
                'similar_experience_bonus': 0.15,
                'qualified_supporter_bonus': 0.1,
                'cultural_match_bonus': 0.05
            }
        }
    
    async def find_matches_for_seeker(self, seeker: SupportSeeker, 
                                    available_offers: List[SupportOffer],
                                    max_matches: int = 3) -> List[MatchScore]:
        """
        Find the best support matches for a person seeking help.
        
        Args:
            seeker: Person seeking support
            available_offers: Available support offers
            max_matches: Maximum number of matches to return
        """
        try:
            self.matching_metrics['matches_attempted'] += 1
            
            candidate_matches = []
            
            # Score each potential match
            for offer in available_offers:
                match_score = await self._calculate_match_score(seeker, offer)
                
                if match_score.is_viable_match():
                    candidate_matches.append(match_score)
            
            # Sort by overall score and quality
            candidate_matches.sort(
                key=lambda m: (m.overall_score, m.success_probability), 
                reverse=True
            )
            
            # Return top matches
            top_matches = candidate_matches[:max_matches]
            
            if top_matches:
                avg_quality = sum(m.overall_score for m in top_matches) / len(top_matches)
                self.matching_metrics['average_match_quality'] = avg_quality
                
                logger.info(f"Found {len(top_matches)} viable matches for seeker {seeker.user_id[:8]}... "
                           f"(avg quality: {avg_quality:.2f})")
            else:
                logger.warning(f"No viable matches found for seeker {seeker.user_id[:8]}...")
            
            return top_matches
            
        except Exception as e:
            logger.error(f"Error finding matches for seeker: {e}")
            return []
    
    async def _calculate_match_score(self, seeker: SupportSeeker, 
                                   offer: SupportOffer) -> MatchScore:
        """Calculate comprehensive match score for seeker-supporter pair"""
        
        try:
            # Calculate scores for each criterion
            criteria_scores = {}
            
            for criterion, algorithm in self.compatibility_algorithms.items():
                score = await algorithm(seeker, offer)
                criteria_scores[criterion] = score
            
            # Assess safety
            safety_assessment = await self.safety_engine.assess_connection_safety(
                seeker, offer, {'assessment_context': 'matching'}
            )
            
            safety_score = 1.0 - safety_assessment.risk_score
            criteria_scores[MatchingCriteria.SAFETY_ASSESSMENT] = safety_score
            
            # Assess consent compatibility
            consent_assessment = await self.consent_manager.assess_consent_for_interaction(
                seeker, offer, {ConsentScope.EMOTIONAL_SUPPORT, ConsentScope.RESOURCE_SHARING}
            )
            
            consent_score = 1.0 if consent_assessment['has_sufficient_consent'] else 0.3
            criteria_scores[MatchingCriteria.CONSENT_COMPATIBILITY] = consent_score
            
            # Calculate weighted overall score
            overall_score = sum(
                criteria_scores[criterion] * self.scoring_weights[criterion]
                for criterion in criteria_scores.keys()
                if criterion in self.scoring_weights
            )
            
            # Determine quality level
            quality_level = self._determine_quality_level(overall_score)
            
            # Calculate success probability
            success_probability = await self._predict_connection_success(
                seeker, offer, criteria_scores, safety_assessment
            )
            
            # Predict care effectiveness
            care_effectiveness = await self._predict_care_effectiveness(
                seeker, offer, criteria_scores
            )
            
            # Generate intervention timing assessment
            timing_assessment = await self.timing_engine.calculate_optimal_intervention_timing(
                seeker, offer, {'assessment_context': 'matching'}
            )
            
            # Generate reasoning and recommendations
            reasoning = await self._generate_match_reasoning(
                seeker, offer, criteria_scores, quality_level
            )
            
            recommendations, concerns = await self._generate_match_recommendations(
                seeker, offer, safety_assessment, criteria_scores
            )
            
            match_score = MatchScore(
                overall_score=overall_score,
                quality_level=quality_level,
                criteria_scores=criteria_scores,
                safety_assessment=safety_assessment,
                consent_assessment=consent_assessment,
                timing_assessment=timing_assessment,
                match_reasoning=reasoning,
                success_probability=success_probability,
                care_effectiveness_prediction=care_effectiveness,
                recommendations=recommendations,
                concerns=concerns
            )
            
            return match_score
            
        except Exception as e:
            logger.error(f"Error calculating match score: {e}")
            return await self._get_fallback_match_score(seeker, offer)
    
    async def _score_support_type_compatibility(self, seeker: SupportSeeker, 
                                              offer: SupportOffer) -> float:
        """Score how well supporter's offerings match seeker's needs"""
        
        needed_types = set(seeker.support_types_needed)
        offered_types = set(offer.support_types_offered)
        
        # Calculate overlap
        matching_types = needed_types & offered_types
        
        if not needed_types:
            return 0.0
        
        # Base score from proportion of needs that can be met
        base_score = len(matching_types) / len(needed_types)
        
        # Bonus for covering high-priority needs
        priority_types = {SupportType.CRISIS_INTERVENTION, SupportType.EMOTIONAL_SUPPORT}
        priority_matches = matching_types & priority_types
        priority_bonus = len(priority_matches) * 0.1
        
        # Bonus for exact match
        exact_match_bonus = 0.2 if matching_types == needed_types else 0.0
        
        return min(1.0, base_score + priority_bonus + exact_match_bonus)
    
    async def _score_experience_relevance(self, seeker: SupportSeeker, 
                                        offer: SupportOffer) -> float:
        """Score relevance of supporter's experience to seeker's situation"""
        
        # Extract situation keywords from seeker's context
        seeker_keywords = set(seeker.support_keywords + seeker.specific_needs)
        
        # Extract experience keywords from supporter
        supporter_keywords = set(offer.experience_areas)
        
        # Calculate keyword overlap
        if not seeker_keywords:
            return 0.5  # Neutral if no specific keywords
        
        overlap = seeker_keywords & supporter_keywords
        keyword_score = len(overlap) / len(seeker_keywords)
        
        # Factor in supporter's success history
        success_bonus = min(0.3, offer.feedback_rating * 0.2)
        
        # Factor in supporter's experience level
        experience_bonus = 0.0
        if 'professional' in ' '.join(offer.credentials).lower():
            experience_bonus = 0.2
        elif 'peer_counselor' in ' '.join(offer.credentials).lower():
            experience_bonus = 0.1
        
        return min(1.0, keyword_score + success_bonus + experience_bonus)
    
    async def _score_communication_compatibility(self, seeker: SupportSeeker, 
                                               offer: SupportOffer) -> float:
        """Score communication style compatibility"""
        
        seeker_prefs = set(seeker.communication_preferences)
        offer_prefs = set(offer.communication_preferences)
        
        # Calculate preference overlap
        if not seeker_prefs or not offer_prefs:
            return 0.5  # Neutral if no preferences specified
        
        overlap = seeker_prefs & offer_prefs
        preference_score = len(overlap) / max(len(seeker_prefs), len(offer_prefs))
        
        # Bonus for compatible timezones
        timezone_bonus = 0.1 if seeker.timezone == offer.timezone else 0.0
        
        # Bonus for overlapping availability
        seeker_times = set(seeker.available_times)
        offer_times = set(offer.available_times)
        
        if 'anytime' in seeker_times or 'anytime' in offer_times:
            availability_bonus = 0.2
        else:
            time_overlap = seeker_times & offer_times
            availability_bonus = 0.1 if time_overlap else 0.0
        
        return min(1.0, preference_score + timezone_bonus + availability_bonus)
    
    async def _score_demographic_preferences(self, seeker: SupportSeeker, 
                                           offer: SupportOffer) -> float:
        """Score demographic compatibility based on seeker preferences"""
        
        seeker_prefs = seeker.preferred_demographics
        offer_demographics = offer.demographic_info
        
        if not seeker_prefs:
            return 1.0  # No preferences means compatible
        
        compatibility_score = 1.0
        
        # Check gender preference
        if 'gender_preference' in seeker_prefs:
            preferred_gender = seeker_prefs['gender_preference']
            offer_gender = offer_demographics.get('gender')
            
            if offer_gender and preferred_gender != offer_gender:
                compatibility_score -= 0.3
        
        # Check age range preference
        if 'age_range' in seeker_prefs:
            preferred_range = seeker_prefs['age_range']
            offer_age = offer_demographics.get('age')
            
            if offer_age:
                age_min, age_max = preferred_range.get('min', 0), preferred_range.get('max', 100)
                if not (age_min <= offer_age <= age_max):
                    compatibility_score -= 0.2
        
        return max(0.0, compatibility_score)
    
    async def _score_availability_alignment(self, seeker: SupportSeeker, 
                                          offer: SupportOffer) -> float:
        """Score alignment of availability and response capabilities"""
        
        # Check supporter availability
        if offer.availability == 'immediate':
            availability_score = 1.0
        elif offer.availability == 'within_hours':
            availability_score = 0.8
        else:  # flexible
            availability_score = 0.6
        
        # Check supporter capacity
        capacity_ratio = offer.current_support_count / offer.max_concurrent_support
        capacity_score = 1.0 - capacity_ratio
        
        # Factor in urgency requirements
        if seeker.urgency == SupportUrgency.CRISIS:
            if offer.availability != 'immediate':
                availability_score *= 0.5
        elif seeker.urgency == SupportUrgency.HIGH:
            if offer.availability == 'flexible':
                availability_score *= 0.7
        
        return (availability_score + capacity_score) / 2
    
    async def _score_cultural_compatibility(self, seeker: SupportSeeker, 
                                          offer: SupportOffer) -> float:
        """Score cultural compatibility and sensitivity"""
        
        # This would integrate with cultural profiles from other components
        # For now, basic compatibility check
        
        seeker_contexts = seeker.cultural_contexts if hasattr(seeker, 'cultural_contexts') else []
        offer_contexts = getattr(offer, 'cultural_contexts', [])
        
        if not seeker_contexts:
            return 1.0  # No specific requirements
        
        # Check for cultural context overlap
        context_overlap = set(seeker_contexts) & set(offer_contexts)
        
        if context_overlap:
            return 1.0  # Perfect cultural match
        
        # Check if supporter has cultural sensitivity training
        if any('cultural' in cred.lower() or 'diversity' in cred.lower() 
               for cred in offer.credentials):
            return 0.8  # Good cultural awareness
        
        return 0.6  # Neutral - no specific incompatibility
    
    async def _score_crisis_appropriateness(self, seeker: SupportSeeker, 
                                          offer: SupportOffer) -> float:
        """Score appropriateness for crisis situations"""
        
        if seeker.urgency != SupportUrgency.CRISIS:
            return 1.0  # Not a crisis, so this criterion doesn't apply
        
        crisis_score = 0.0
        
        # Check for crisis qualifications
        crisis_keywords = ['crisis', 'suicide', 'emergency', 'intervention']
        
        for keyword in crisis_keywords:
            if any(keyword in cred.lower() for cred in offer.credentials):
                crisis_score += 0.25
        
        # Check for professional qualifications
        professional_keywords = ['licensed', 'certified', 'therapist', 'counselor']
        
        for keyword in professional_keywords:
            if any(keyword in cred.lower() for cred in offer.credentials):
                crisis_score += 0.2
        
        # Check availability for crisis
        if offer.availability == 'immediate':
            crisis_score += 0.3
        
        # Check experience with similar situations
        if any(area in ['crisis', 'suicide', 'emergency'] for area in offer.experience_areas):
            crisis_score += 0.2
        
        return min(1.0, crisis_score)
    
    async def _score_healing_stage_alignment(self, seeker: SupportSeeker, 
                                           offer: SupportOffer) -> float:
        """Score alignment of healing stages between seeker and supporter"""
        
        # This is a simplified approach - in production would be more sophisticated
        
        # Someone in crisis needs someone stable and further along in healing
        if seeker.urgency == SupportUrgency.CRISIS:
            if offer.success_stories > 5 and offer.feedback_rating > 4.0:
                return 1.0  # Experienced helper
            elif offer.success_stories > 2:
                return 0.7  # Some experience
            else:
                return 0.3  # Inexperienced for crisis
        
        # Regular support can be more flexible
        if offer.success_stories > 0:
            return 0.8 + min(0.2, offer.success_stories * 0.04)
        
        return 0.6  # Neutral for new supporters
    
    def _determine_quality_level(self, overall_score: float) -> MatchQuality:
        """Determine match quality level from overall score"""
        
        if overall_score >= 0.9:
            return MatchQuality.EXCELLENT
        elif overall_score >= 0.7:
            return MatchQuality.GOOD
        elif overall_score >= 0.5:
            return MatchQuality.ADEQUATE
        elif overall_score >= 0.3:
            return MatchQuality.POOR
        else:
            return MatchQuality.INCOMPATIBLE
    
    async def _predict_connection_success(self, seeker: SupportSeeker, 
                                        offer: SupportOffer,
                                        criteria_scores: Dict[MatchingCriteria, float],
                                        safety_assessment: SafetyAssessment) -> float:
        """Predict probability of successful connection"""
        
        model = self.success_prediction_model
        
        # Base success probability from match quality
        base_probability = sum(
            criteria_scores.get(criterion, 0.5) * weight
            for criterion, weight in model['success_factors'].items()
            if criterion in criteria_scores
        )
        
        # Apply risk factors
        for risk_factor, penalty in model['risk_factors'].items():
            if risk_factor == 'high_vulnerability_penalty' and seeker.urgency == SupportUrgency.CRISIS:
                base_probability += penalty
            elif risk_factor == 'inexperienced_supporter_penalty' and offer.success_stories == 0:
                base_probability += penalty
            elif risk_factor == 'timezone_mismatch_penalty' and seeker.timezone != offer.timezone:
                base_probability += penalty
        
        # Apply success indicators
        for indicator, bonus in model['success_indicators'].items():
            if indicator == 'similar_experience_bonus':
                if set(seeker.support_keywords) & set(offer.experience_areas):
                    base_probability += bonus
            elif indicator == 'qualified_supporter_bonus':
                if any('certified' in cred.lower() for cred in offer.credentials):
                    base_probability += bonus
        
        # Factor in safety assessment
        safety_factor = 1.0 - safety_assessment.risk_score
        base_probability *= safety_factor
        
        return max(0.0, min(1.0, base_probability))
    
    async def _predict_care_effectiveness(self, seeker: SupportSeeker, 
                                        offer: SupportOffer,
                                        criteria_scores: Dict[MatchingCriteria, float]) -> float:
        """Predict how effective this connection will be for care/healing"""
        
        # Weighted combination of key factors for care effectiveness
        effectiveness_factors = {
            MatchingCriteria.SUPPORT_TYPE_COMPATIBILITY: 0.3,
            MatchingCriteria.EXPERIENCE_RELEVANCE: 0.25,
            MatchingCriteria.SAFETY_ASSESSMENT: 0.2,
            MatchingCriteria.COMMUNICATION_COMPATIBILITY: 0.15,
            MatchingCriteria.HEALING_STAGE_ALIGNMENT: 0.1
        }
        
        effectiveness = sum(
            criteria_scores.get(factor, 0.5) * weight
            for factor, weight in effectiveness_factors.items()
        )
        
        # Bonus for supporter track record
        track_record_bonus = min(0.2, offer.feedback_rating / 5.0 * 0.2)
        effectiveness += track_record_bonus
        
        return min(1.0, effectiveness)
    
    async def _generate_match_reasoning(self, seeker: SupportSeeker, 
                                      offer: SupportOffer,
                                      criteria_scores: Dict[MatchingCriteria, float],
                                      quality_level: MatchQuality) -> str:
        """Generate human-readable reasoning for the match"""
        
        reasoning_parts = []
        
        # Overall quality
        reasoning_parts.append(f"Match quality: {quality_level.value}")
        
        # Highlight strongest compatibility areas
        top_scores = sorted(criteria_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        
        for criterion, score in top_scores:
            if score > 0.7:
                criterion_name = criterion.value.replace('_', ' ').title()
                reasoning_parts.append(f"Strong {criterion_name} (score: {score:.2f})")
        
        # Note specific strengths
        if offer.feedback_rating > 4.0:
            reasoning_parts.append(f"Highly rated supporter ({offer.feedback_rating:.1f}/5.0)")
        
        if seeker.urgency == SupportUrgency.CRISIS and offer.availability == 'immediate':
            reasoning_parts.append("Immediate availability for crisis support")
        
        # Note any concerns
        low_scores = [criterion for criterion, score in criteria_scores.items() if score < 0.4]
        if low_scores:
            concern_areas = [c.value.replace('_', ' ') for c in low_scores[:2]]
            reasoning_parts.append(f"Areas for attention: {', '.join(concern_areas)}")
        
        return '; '.join(reasoning_parts)
    
    async def _generate_match_recommendations(self, seeker: SupportSeeker, 
                                            offer: SupportOffer,
                                            safety_assessment: SafetyAssessment,
                                            criteria_scores: Dict[MatchingCriteria, float]) -> Tuple[List[str], List[str]]:
        """Generate recommendations and concerns for the match"""
        
        recommendations = []
        concerns = []
        
        # Safety-based recommendations
        recommendations.extend(safety_assessment.recommendations)
        
        if safety_assessment.safety_level == SafetyLevel.CAUTIOUS:
            concerns.append("Enhanced monitoring recommended due to safety considerations")
        
        # Communication recommendations
        comm_score = criteria_scores.get(MatchingCriteria.COMMUNICATION_COMPATIBILITY, 0.5)
        if comm_score < 0.6:
            recommendations.append("Establish clear communication preferences and boundaries")
        
        # Experience-based recommendations
        exp_score = criteria_scores.get(MatchingCriteria.EXPERIENCE_RELEVANCE, 0.5)
        if exp_score < 0.5:
            recommendations.append("Consider additional resources or supervisor support")
        
        # Crisis-specific recommendations
        if seeker.urgency == SupportUrgency.CRISIS:
            recommendations.append("Activate crisis support protocols")
            recommendations.append("Ensure professional backup is available")
            
            crisis_score = criteria_scores.get(MatchingCriteria.CRISIS_APPROPRIATENESS, 0.5)
            if crisis_score < 0.7:
                concerns.append("Supporter may need additional crisis training")
        
        # Capacity concerns
        if offer.current_support_count >= offer.max_concurrent_support * 0.8:
            concerns.append("Supporter approaching capacity limits")
        
        return recommendations, concerns
    
    async def _get_fallback_match_score(self, seeker: SupportSeeker, 
                                      offer: SupportOffer) -> MatchScore:
        """Get conservative fallback match score when calculation fails"""
        
        return MatchScore(
            overall_score=0.3,
            quality_level=MatchQuality.POOR,
            criteria_scores={},
            safety_assessment=await self.safety_engine._get_fallback_safety_assessment(seeker, offer),
            consent_assessment={'has_sufficient_consent': False},
            timing_assessment=await self.timing_engine._get_fallback_timing(seeker, {}),
            match_reasoning="Match calculation failed - requires manual review",
            success_probability=0.2,
            care_effectiveness_prediction=0.2,
            recommendations=["Manual review required"],
            concerns=["Automated matching failed"]
        )
    
    async def create_connection(self, seeker: SupportSeeker, 
                             supporter: SupportOffer,
                             match_score: MatchScore) -> Optional[Connection]:
        """Create an active connection between seeker and supporter"""
        
        try:
            connection_id = f"conn_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{len(self.active_connections)}"
            
            connection = Connection(
                connection_id=connection_id,
                seeker=seeker,
                supporter=supporter,
                match_score=match_score,
                created_at=datetime.utcnow(),
                status='pending',
                interaction_history=[],
                monitoring_data={
                    'safety_checks': [],
                    'boundary_violations': [],
                    'effectiveness_ratings': []
                },
                success_metrics={
                    'connection_satisfaction': 0.0,
                    'support_effectiveness': 0.0,
                    'safety_maintained': True
                }
            )
            
            self.active_connections[connection_id] = connection
            self.matching_metrics['matches_successful'] += 1
            
            logger.info(f"Created connection {connection_id}: {seeker.user_id[:8]}... -> {supporter.user_id[:8]}...")
            
            return connection
            
        except Exception as e:
            logger.error(f"Error creating connection: {e}")
            return None
    
    def get_matching_analytics(self) -> Dict[str, Any]:
        """Get analytics about matching performance"""
        
        total_attempts = self.matching_metrics['matches_attempted']
        
        if total_attempts == 0:
            return {'no_data': True}
        
        return {
            'matches_attempted': total_attempts,
            'matches_successful': self.matching_metrics['matches_successful'],
            'matches_declined': self.matching_metrics['matches_declined'],
            'connections_completed_successfully': self.matching_metrics['connections_completed_successfully'],
            'success_rate': (self.matching_metrics['matches_successful'] / total_attempts) * 100,
            'completion_rate': (self.matching_metrics['connections_completed_successfully'] / 
                               max(1, self.matching_metrics['matches_successful'])) * 100,
            'average_match_quality': self.matching_metrics['average_match_quality'],
            'care_effectiveness_achieved': self.matching_metrics['care_effectiveness_achieved'],
            'active_connections': len(self.active_connections),
            'matching_algorithm_health': {
                'scoring_weights_loaded': len(self.scoring_weights) > 0,
                'compatibility_algorithms_loaded': len(self.compatibility_algorithms) > 0,
                'success_model_active': len(self.success_prediction_model) > 0,
                'safety_integration_active': True,
                'consent_integration_active': True
            }
        }