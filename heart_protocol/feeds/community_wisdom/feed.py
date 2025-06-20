"""
Community Wisdom Feed - Main Implementation

The fourth of the Four Sacred Feeds that curates healing insights, wisdom,
and knowledge from the community to support collective growth and learning.
Amplifies wisdom that emerges from lived experience and transformation.

Core Philosophy: "In sharing our wisdom, we multiply healing."
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging

from ...core.base import FeedItem, CareLevel, FeedGenerator, Post, CareAssessment
from ...core.care_detection import CareDetectionEngine
from .wisdom_curation import WisdomCurator, WisdomInsight, WisdomCategory, WisdomType
from .insight_validation import InsightValidator, ValidationResult, ValidationStatus, ValidationMethod
from .knowledge_synthesis import KnowledgeSynthesizer, SynthesizedWisdom, SynthesisMethod, SynthesisScope

logger = logging.getLogger(__name__)


class CommunityWisdomFeed(FeedGenerator):
    """
    The Community Wisdom Feed curates and shares healing wisdom from the community.
    
    Key Features:
    - Wisdom curation from community posts
    - Multi-method insight validation
    - Knowledge synthesis into coherent guides
    - Culturally sensitive wisdom sharing
    - Trauma-informed wisdom presentation
    - Evidence-based wisdom validation
    - Privacy-preserving wisdom anonymization
    """
    
    def __init__(self, config: Dict[str, Any], care_detection_engine: CareDetectionEngine):
        super().__init__(config)
        self.feed_type = "community_wisdom"
        self.care_engine = care_detection_engine
        
        # Initialize core components
        self.wisdom_curator = WisdomCurator(config.get('wisdom_curation', {}))
        self.insight_validator = InsightValidator(config.get('insight_validation', {}))
        self.knowledge_synthesizer = KnowledgeSynthesizer(config.get('knowledge_synthesis', {}))
        
        # Feed state management
        self.recent_posts = {}           # post_id -> processed_post_data
        self.pending_validations = []    # insights awaiting validation
        self.featured_wisdom = {}        # featured wisdom collections
        self.daily_wisdom_picks = {}     # daily curated wisdom selections
        
        # Feed configuration
        self.wisdom_rotation_days = config.get('wisdom_rotation_days', 7)
        self.max_wisdom_per_feed = config.get('max_wisdom_per_feed', 3)
        self.validation_threshold = config.get('validation_threshold', 0.7)
        self.synthesis_trigger_threshold = config.get('synthesis_trigger_threshold', 5)  # min insights for synthesis
        
        logger.info("Community Wisdom Feed initialized")
    
    async def generate_feed_items(self, user_id: str, 
                                user_context: Dict[str, Any],
                                care_assessment: Optional[CareAssessment] = None) -> List[FeedItem]:
        """
        Generate Community Wisdom feed items for a user.
        
        This includes:
        - Relevant wisdom insights for user's current needs
        - Synthesized wisdom collections
        - Daily wisdom picks
        - Community knowledge highlights
        """
        try:
            feed_items = []
            
            # Generate relevant wisdom insights
            insight_items = await self._generate_wisdom_insight_items(
                user_id, user_context, care_assessment
            )
            feed_items.extend(insight_items)
            
            # Generate synthesized wisdom collection items
            synthesis_items = await self._generate_synthesis_items(
                user_id, user_context, care_assessment
            )
            feed_items.extend(synthesis_items)
            
            # Generate daily wisdom picks
            daily_items = await self._generate_daily_wisdom_items(
                user_id, user_context, care_assessment
            )
            feed_items.extend(daily_items)
            
            # Generate community knowledge highlights
            community_items = await self._generate_community_knowledge_items(
                user_id, user_context
            )
            feed_items.extend(community_items)
            
            logger.info(f"Generated {len(feed_items)} Community Wisdom items for user {user_id[:8]}...")
            
            return feed_items
            
        except Exception as e:
            logger.error(f"Error generating Community Wisdom feed for user {user_id[:8]}...: {e}")
            return []
    
    async def process_post_for_wisdom_curation(self, post: Post, 
                                             care_assessment: CareAssessment,
                                             user_context: Dict[str, Any]) -> None:
        """Process a post to curate wisdom and insights"""
        
        try:
            # Store post for reference
            self.recent_posts[post.id] = {
                'post': post,
                'care_assessment': care_assessment,
                'user_context': user_context,
                'processed_at': datetime.utcnow()
            }
            
            # Curate wisdom insights from the post
            curated_insights = await self.wisdom_curator.curate_wisdom_from_post(
                post, care_assessment, user_context
            )
            
            # Process each curated insight
            for insight in curated_insights:
                await self._handle_curated_insight(insight, user_context)
            
            # Check if we have enough insights to trigger synthesis
            await self._check_synthesis_triggers()
            
            # Clean up old posts
            await self._cleanup_old_posts()
            
        except Exception as e:
            logger.error(f"Error processing post for wisdom curation: {e}")
    
    async def _handle_curated_insight(self, insight: WisdomInsight, 
                                    user_context: Dict[str, Any]) -> None:
        """Handle a newly curated wisdom insight"""
        
        try:
            logger.debug(f"Processing curated insight {insight.insight_id} "
                        f"(category: {insight.category.value}, score: {insight.wisdom_score:.2f})")
            
            # Validate the insight
            validation_result = await self.insight_validator.validate_insight(
                insight, ValidationMethod.COMMUNITY_FEEDBACK
            )
            
            # Handle validation result
            if validation_result.status == ValidationStatus.VALIDATED:
                await self._handle_validated_insight(insight, validation_result)
            elif validation_result.status == ValidationStatus.EXPERT_REVIEW_NEEDED:
                await self._queue_for_expert_review(insight, validation_result)
            elif validation_result.status == ValidationStatus.CONDITIONALLY_APPROVED:
                await self._handle_conditionally_approved_insight(insight, validation_result)
            
            # Update insight validation score
            insight.validation_score = validation_result.confidence_score
            
        except Exception as e:
            logger.error(f"Error handling curated insight: {e}")
    
    async def _handle_validated_insight(self, insight: WisdomInsight, 
                                      validation_result: ValidationResult) -> None:
        """Handle a successfully validated insight"""
        
        # Add to featured wisdom if high quality
        if validation_result.quality_score >= 0.8 and validation_result.safety_score >= 0.9:
            await self._add_to_featured_wisdom(insight)
        
        # Consider for daily wisdom picks
        if validation_result.applicability_score >= 0.7:
            await self._consider_for_daily_picks(insight)
        
        logger.info(f"Validated insight {insight.insight_id} "
                   f"(quality: {validation_result.quality_score:.2f}, "
                   f"safety: {validation_result.safety_score:.2f})")
    
    async def _queue_for_expert_review(self, insight: WisdomInsight, 
                                     validation_result: ValidationResult) -> None:
        """Queue insight for expert review"""
        
        self.pending_validations.append({
            'insight': insight,
            'validation_result': validation_result,
            'queued_at': datetime.utcnow(),
            'review_type': 'expert_review'
        })
        
        logger.info(f"Queued insight {insight.insight_id} for expert review")
    
    async def _handle_conditionally_approved_insight(self, insight: WisdomInsight, 
                                                   validation_result: ValidationResult) -> None:
        """Handle conditionally approved insight"""
        
        # Add with caveats and additional safety notes
        if validation_result.safety_score >= 0.8:
            # Can be shared with appropriate warnings
            insight.care_considerations.extend(validation_result.improvement_suggestions)
            await self._consider_for_daily_picks(insight)
        
        logger.debug(f"Conditionally approved insight {insight.insight_id}")
    
    async def _check_synthesis_triggers(self) -> None:
        """Check if conditions are met to trigger knowledge synthesis"""
        
        # Group validated insights by category
        category_insights = {}
        
        for category in WisdomCategory:
            insights = self.wisdom_curator.get_wisdom_by_category(category)
            validated_insights = [insight for insight in insights 
                                 if insight.validation_score >= self.validation_threshold]
            
            if len(validated_insights) >= self.synthesis_trigger_threshold:
                category_insights[category] = validated_insights
        
        # Trigger synthesis for categories with enough insights
        for category, insights in category_insights.items():
            await self._trigger_knowledge_synthesis(category, insights)
    
    async def _trigger_knowledge_synthesis(self, category: WisdomCategory, 
                                         insights: List[WisdomInsight]) -> None:
        """Trigger knowledge synthesis for a category"""
        
        try:
            # Check if we've recently synthesized this category
            existing_syntheses = self.knowledge_synthesizer.get_synthesis_by_category(category)
            
            if existing_syntheses:
                latest_synthesis = max(existing_syntheses, key=lambda s: s.created_at)
                time_since_last = datetime.utcnow() - latest_synthesis.created_at
                
                if time_since_last < timedelta(days=30):  # Don't re-synthesize too frequently
                    return
            
            # Determine synthesis method based on category
            synthesis_method = await self._determine_synthesis_method(category, insights)
            
            # Create synthesis
            synthesis = await self.knowledge_synthesizer.synthesize_wisdom_collection(
                insights, synthesis_method, SynthesisScope.CATEGORY_COMPREHENSIVE
            )
            
            # Add to featured content
            await self._add_synthesis_to_featured(synthesis)
            
            logger.info(f"Created knowledge synthesis for {category.value}: {synthesis.title}")
            
        except Exception as e:
            logger.error(f"Error triggering knowledge synthesis: {e}")
    
    async def _determine_synthesis_method(self, category: WisdomCategory, 
                                        insights: List[WisdomInsight]) -> SynthesisMethod:
        """Determine appropriate synthesis method for category"""
        
        method_mapping = {
            WisdomCategory.CRISIS_NAVIGATION: SynthesisMethod.CRISIS_RESOURCE_COMPILATION,
            WisdomCategory.TRAUMA_RECOVERY: SynthesisMethod.TRAUMA_INFORMED_GUIDE,
            WisdomCategory.COPING_STRATEGIES: SynthesisMethod.PRACTICAL_PATHWAY,
            WisdomCategory.HEALING_INSIGHTS: SynthesisMethod.EXPERIENTIAL_JOURNEY,
            WisdomCategory.SELF_CARE_PRACTICES: SynthesisMethod.PRACTICAL_PATHWAY,
            WisdomCategory.EMOTIONAL_REGULATION: SynthesisMethod.PRACTICAL_PATHWAY,
            WisdomCategory.RELATIONSHIP_WISDOM: SynthesisMethod.THEMATIC_CLUSTERING
        }
        
        return method_mapping.get(category, SynthesisMethod.THEMATIC_CLUSTERING)
    
    async def _generate_wisdom_insight_items(self, user_id: str,
                                           user_context: Dict[str, Any],
                                           care_assessment: Optional[CareAssessment]) -> List[FeedItem]:
        """Generate wisdom insight items relevant to user"""
        
        items = []
        
        # Determine relevant categories based on user context
        relevant_categories = await self._determine_relevant_categories(user_context, care_assessment)
        
        # Get insights for relevant categories
        for category in relevant_categories[:2]:  # Top 2 most relevant categories
            insights = self.wisdom_curator.get_wisdom_by_category(category, limit=3)
            validated_insights = [insight for insight in insights 
                                 if insight.validation_score >= self.validation_threshold]
            
            if validated_insights:
                # Create feed item for top insight
                item = await self._create_wisdom_insight_feed_item(validated_insights[0], user_context)
                if item:
                    items.append(item)
        
        return items
    
    async def _generate_synthesis_items(self, user_id: str,
                                      user_context: Dict[str, Any],
                                      care_assessment: Optional[CareAssessment]) -> List[FeedItem]:
        """Generate synthesized wisdom collection items"""
        
        items = []
        
        # Get relevant syntheses
        relevant_categories = await self._determine_relevant_categories(user_context, care_assessment)
        
        for category in relevant_categories[:1]:  # Top category only
            syntheses = self.knowledge_synthesizer.get_synthesis_by_category(category)
            
            if syntheses:
                # Get most recent high-quality synthesis
                quality_syntheses = [s for s in syntheses if s.evidence_strength >= 0.7]
                if quality_syntheses:
                    latest_synthesis = max(quality_syntheses, key=lambda s: s.created_at)
                    
                    item = await self._create_synthesis_feed_item(latest_synthesis, user_context)
                    if item:
                        items.append(item)
        
        return items
    
    async def _generate_daily_wisdom_items(self, user_id: str,
                                         user_context: Dict[str, Any],
                                         care_assessment: Optional[CareAssessment]) -> List[FeedItem]:
        """Generate daily wisdom pick items"""
        
        items = []
        
        # Get today's date for consistent daily picks
        today = datetime.utcnow().date()
        
        # Check if we have daily picks for today
        if today not in self.daily_wisdom_picks:
            await self._select_daily_wisdom_picks(today)
        
        daily_picks = self.daily_wisdom_picks.get(today, [])
        
        # Find picks relevant to user
        relevant_picks = []
        user_needs = await self._determine_user_needs(user_context, care_assessment)
        
        for pick in daily_picks:
            if any(need in pick['applicability_tags'] for need in user_needs):
                relevant_picks.append(pick)
        
        # Create feed item for most relevant pick
        if relevant_picks:
            pick = relevant_picks[0]
            item = await self._create_daily_wisdom_feed_item(pick, user_context)
            if item:
                items.append(item)
        
        return items
    
    async def _generate_community_knowledge_items(self, user_id: str,
                                                user_context: Dict[str, Any]) -> List[FeedItem]:
        """Generate community knowledge highlight items"""
        
        items = []
        
        # Get community knowledge statistics
        curation_metrics = self.wisdom_curator.get_curation_metrics()
        
        # Create community highlight if we have significant activity
        if curation_metrics['insights_curated'] >= 10:
            item = await self._create_community_knowledge_highlight_item(curation_metrics, user_context)
            if item:
                items.append(item)
        
        return items
    
    async def _determine_relevant_categories(self, user_context: Dict[str, Any],
                                           care_assessment: Optional[CareAssessment]) -> List[WisdomCategory]:
        """Determine wisdom categories most relevant to user"""
        
        relevant_categories = []
        
        # Based on care level
        if care_assessment:
            if care_assessment.care_level == CareLevel.CRISIS:
                relevant_categories.append(WisdomCategory.CRISIS_NAVIGATION)
            elif care_assessment.care_level in [CareLevel.HIGH, CareLevel.MODERATE]:
                relevant_categories.extend([
                    WisdomCategory.COPING_STRATEGIES,
                    WisdomCategory.EMOTIONAL_REGULATION,
                    WisdomCategory.SELF_CARE_PRACTICES
                ])
            else:
                relevant_categories.extend([
                    WisdomCategory.HEALING_INSIGHTS,
                    WisdomCategory.MEANING_MAKING,
                    WisdomCategory.COMMUNITY_SUPPORT
                ])
        
        # Based on user interests
        user_interests = user_context.get('healing_interests', [])
        for interest in user_interests:
            try:
                category = WisdomCategory(interest)
                if category not in relevant_categories:
                    relevant_categories.append(category)
            except ValueError:
                pass  # Interest doesn't match a category
        
        # Default categories if none identified
        if not relevant_categories:
            relevant_categories = [
                WisdomCategory.HEALING_INSIGHTS,
                WisdomCategory.SELF_CARE_PRACTICES,
                WisdomCategory.COPING_STRATEGIES
            ]
        
        return relevant_categories
    
    async def _determine_user_needs(self, user_context: Dict[str, Any],
                                  care_assessment: Optional[CareAssessment]) -> List[str]:
        """Determine user's current needs for wisdom matching"""
        
        needs = []
        
        # Based on care assessment
        if care_assessment:
            if care_assessment.care_level == CareLevel.CRISIS:
                needs.extend(['crisis', 'emergency', 'safety'])
            elif care_assessment.care_level == CareLevel.HIGH:
                needs.extend(['anxiety', 'stress', 'support'])
            elif care_assessment.care_level == CareLevel.MODERATE:
                needs.extend(['coping', 'emotional_regulation', 'self_care'])
        
        # Based on user context
        if user_context.get('recent_challenges'):
            needs.extend(user_context['recent_challenges'])
        
        if user_context.get('healing_goals'):
            needs.extend(user_context['healing_goals'])
        
        return needs
    
    async def _create_wisdom_insight_feed_item(self, insight: WisdomInsight,
                                             user_context: Dict[str, Any]) -> Optional[FeedItem]:
        """Create feed item for a wisdom insight"""
        
        try:
            # Anonymize and format the insight
            anonymized_content = await self._anonymize_wisdom_content(insight.content)
            
            content = f"""## 💡 Community Wisdom: {insight.category.value.replace('_', ' ').title()}

{anonymized_content}

**Type:** {insight.wisdom_type.value.replace('_', ' ').title()}

### Why This Helps
{chr(10).join(f'• {evidence}' for evidence in insight.supporting_evidence[:2])}

### Keep in Mind
{chr(10).join(f'• {consideration}' for consideration in insight.care_considerations[:2])}

---
*This wisdom comes from someone in our community who has walked a similar path. Every journey is unique - adapt what resonates and leave what doesn't.*"""
            
            # Determine care level based on insight category
            care_level_mapping = {
                WisdomCategory.CRISIS_NAVIGATION: CareLevel.HIGH,
                WisdomCategory.TRAUMA_RECOVERY: CareLevel.MODERATE,
                WisdomCategory.EMOTIONAL_REGULATION: CareLevel.MODERATE,
                WisdomCategory.HEALING_INSIGHTS: CareLevel.LOW,
                WisdomCategory.SELF_CARE_PRACTICES: CareLevel.LOW
            }
            
            care_level = care_level_mapping.get(insight.category, CareLevel.LOW)
            
            metadata = {
                'feed_type': 'community_wisdom',
                'item_type': 'wisdom_insight',
                'insight_id': insight.insight_id,
                'wisdom_category': insight.category.value,
                'wisdom_type': insight.wisdom_type.value,
                'wisdom_score': insight.wisdom_score,
                'validation_score': insight.validation_score,
                'applicability_tags': insight.applicability_tags,
                'trauma_informed': insight.trauma_informed,
                'anonymization_level': insight.anonymization_level
            }
            
            return FeedItem(
                content=content,
                care_level=care_level,
                source_type="community_wisdom_insight",
                metadata=metadata,
                reasoning=f"Relevant wisdom insight for {insight.category.value.replace('_', ' ')}",
                confidence_score=insight.validation_score,
                requires_human_review=False
            )
            
        except Exception as e:
            logger.error(f"Error creating wisdom insight feed item: {e}")
            return None
    
    async def _create_synthesis_feed_item(self, synthesis: SynthesizedWisdom,
                                        user_context: Dict[str, Any]) -> Optional[FeedItem]:
        """Create feed item for a synthesized wisdom collection"""
        
        try:
            # Create preview of synthesis
            preview_content = synthesis.synthesized_content[:500] + "..." if len(synthesis.synthesized_content) > 500 else synthesis.synthesized_content
            
            content = f"""## 📚 Wisdom Collection: {synthesis.title}

{synthesis.description}

### Preview
{preview_content}

### Key Themes
{chr(10).join(f'• {theme}' for theme in synthesis.key_themes[:3])}

### Practical Applications
{chr(10).join(f'• {app}' for app in synthesis.practical_applications[:2])}

### Safety Guidelines
{chr(10).join(f'• {guideline}' for guideline in synthesis.safety_guidelines[:2])}

**Sources:** {len(synthesis.source_insights)} community insights  
**Evidence Strength:** {synthesis.evidence_strength:.1f}/1.0

---
*This collection represents the collective wisdom of our community, carefully curated and synthesized for your healing journey.*"""
            
            metadata = {
                'feed_type': 'community_wisdom',
                'item_type': 'wisdom_synthesis',
                'synthesis_id': synthesis.synthesis_id,
                'synthesis_method': synthesis.synthesis_method.value,
                'synthesis_scope': synthesis.scope.value,
                'wisdom_category': synthesis.category.value,
                'source_insight_count': len(synthesis.source_insights),
                'evidence_strength': synthesis.evidence_strength,
                'key_themes': synthesis.key_themes,
                'target_audience': synthesis.target_audience
            }
            
            return FeedItem(
                content=content,
                care_level=CareLevel.LOW,  # Synthesized content is generally educational
                source_type="community_wisdom_synthesis",
                metadata=metadata,
                reasoning=f"Comprehensive wisdom collection for {synthesis.category.value.replace('_', ' ')}",
                confidence_score=synthesis.evidence_strength,
                requires_human_review=False
            )
            
        except Exception as e:
            logger.error(f"Error creating synthesis feed item: {e}")
            return None
    
    async def _create_daily_wisdom_feed_item(self, daily_pick: Dict[str, Any],
                                           user_context: Dict[str, Any]) -> Optional[FeedItem]:
        """Create feed item for daily wisdom pick"""
        
        try:
            content = f"""## ⭐ Today's Wisdom

{daily_pick['content']}

**Why we chose this today:** {daily_pick['selection_reason']}

### Reflection Questions
{chr(10).join(f'• {question}' for question in daily_pick.get('reflection_questions', []))}

---
*Each day, we select wisdom that resonates with our community's current journey. Take what serves you, leave what doesn't.*"""
            
            metadata = {
                'feed_type': 'community_wisdom',
                'item_type': 'daily_wisdom_pick',
                'wisdom_category': daily_pick['category'],
                'selection_date': daily_pick['date'].isoformat(),
                'applicability_tags': daily_pick.get('applicability_tags', [])
            }
            
            return FeedItem(
                content=content,
                care_level=CareLevel.LOW,
                source_type="daily_wisdom_pick",
                metadata=metadata,
                reasoning="Daily curated wisdom pick for community",
                confidence_score=0.8,
                requires_human_review=False
            )
            
        except Exception as e:
            logger.error(f"Error creating daily wisdom feed item: {e}")
            return None
    
    async def _create_community_knowledge_highlight_item(self, metrics: Dict[str, Any],
                                                       user_context: Dict[str, Any]) -> Optional[FeedItem]:
        """Create community knowledge highlight item"""
        
        try:
            content = f"""## 🌟 Community Knowledge Highlights

Our healing community continues to grow in wisdom and insight:

**Collective Wisdom Stats:**
• **{metrics['insights_curated']}** pieces of wisdom curated from lived experience
• **{metrics['contributing_users']}** community members sharing their knowledge
• **{metrics['wisdom_categories_active']}** different areas of healing wisdom

**Active Wisdom Categories:**
{chr(10).join(f'• {cat.replace("_", " ").title()}' for cat in list(metrics.get('category_distribution', {}).keys())[:5])}

### What This Means
Every insight represents someone's courage to share what they've learned through their healing journey. This collective wisdom becomes a resource for our entire community.

### How to Contribute
Your experiences and insights matter. When you share what has helped you heal and grow, you contribute to our collective wisdom and help others on similar paths.

---
*Together, we are creating a library of healing wisdom rooted in lived experience and mutual care.*"""
            
            metadata = {
                'feed_type': 'community_wisdom',
                'item_type': 'community_knowledge_highlight',
                'total_insights': metrics['insights_curated'],
                'contributing_users': metrics['contributing_users'],
                'active_categories': metrics['wisdom_categories_active'],
                'community_driven': True
            }
            
            return FeedItem(
                content=content,
                care_level=CareLevel.LOW,
                source_type="community_knowledge_highlight",
                metadata=metadata,
                reasoning="Highlighting growth of community wisdom collection",
                confidence_score=0.9,
                requires_human_review=False
            )
            
        except Exception as e:
            logger.error(f"Error creating community knowledge highlight item: {e}")
            return None
    
    async def _anonymize_wisdom_content(self, content: str) -> str:
        """Anonymize wisdom content for sharing"""
        
        # Remove personal identifiers
        anonymized = content.replace(' I ', ' someone ')
        anonymized = anonymized.replace('I ', 'A person ')
        anonymized = anonymized.replace(' my ', ' their ')
        anonymized = anonymized.replace('My ', 'Their ')
        anonymized = anonymized.replace(' me ', ' them ')
        anonymized = anonymized.replace('Me ', 'They ')
        
        return anonymized
    
    async def _add_to_featured_wisdom(self, insight: WisdomInsight) -> None:
        """Add insight to featured wisdom collection"""
        
        category = insight.category.value
        if category not in self.featured_wisdom:
            self.featured_wisdom[category] = []
        
        self.featured_wisdom[category].append({
            'insight': insight,
            'featured_at': datetime.utcnow(),
            'feature_score': insight.wisdom_score + insight.validation_score
        })
        
        # Keep only top featured items
        self.featured_wisdom[category].sort(key=lambda x: x['feature_score'], reverse=True)
        self.featured_wisdom[category] = self.featured_wisdom[category][:10]
    
    async def _consider_for_daily_picks(self, insight: WisdomInsight) -> None:
        """Consider insight for daily wisdom picks"""
        
        # Add to candidate pool for daily selection
        # This would be used by the daily wisdom selection algorithm
        pass
    
    async def _select_daily_wisdom_picks(self, date: datetime.date) -> None:
        """Select daily wisdom picks for a specific date"""
        
        # Get high-quality insights from different categories
        daily_picks = []
        
        for category in WisdomCategory:
            insights = self.wisdom_curator.get_wisdom_by_category(category, limit=5)
            validated_insights = [insight for insight in insights 
                                 if insight.validation_score >= 0.8]
            
            if validated_insights:
                best_insight = max(validated_insights, key=lambda i: i.wisdom_score + i.validation_score)
                
                daily_picks.append({
                    'content': best_insight.content,
                    'category': category.value,
                    'date': date,
                    'selection_reason': f'High-quality {category.value.replace("_", " ")} wisdom',
                    'applicability_tags': best_insight.applicability_tags,
                    'reflection_questions': [
                        'How might this wisdom apply to your current situation?',
                        'What resonates most with you about this insight?'
                    ]
                })
        
        # Select top 3 picks for the day
        daily_picks.sort(key=lambda x: len(x['applicability_tags']), reverse=True)
        self.daily_wisdom_picks[date] = daily_picks[:3]
    
    async def _add_synthesis_to_featured(self, synthesis: SynthesizedWisdom) -> None:
        """Add synthesis to featured content"""
        
        category = synthesis.category.value
        if f"{category}_syntheses" not in self.featured_wisdom:
            self.featured_wisdom[f"{category}_syntheses"] = []
        
        self.featured_wisdom[f"{category}_syntheses"].append({
            'synthesis': synthesis,
            'featured_at': datetime.utcnow(),
            'feature_score': synthesis.evidence_strength
        })
    
    async def _cleanup_old_posts(self) -> None:
        """Clean up old processed posts"""
        
        cutoff_time = datetime.utcnow() - timedelta(hours=48)
        
        old_posts = [
            post_id for post_id, data in self.recent_posts.items()
            if data['processed_at'] < cutoff_time
        ]
        
        for post_id in old_posts:
            del self.recent_posts[post_id]
        
        # Also cleanup old pending validations
        cutoff_time = datetime.utcnow() - timedelta(days=7)
        self.pending_validations = [
            validation for validation in self.pending_validations
            if validation['queued_at'] > cutoff_time
        ]
    
    async def get_feed_analytics(self) -> Dict[str, Any]:
        """Get analytics for Community Wisdom feed"""
        
        try:
            # Get component analytics
            curation_metrics = self.wisdom_curator.get_curation_metrics()
            validation_metrics = self.insight_validator.get_validation_metrics()
            synthesis_metrics = self.knowledge_synthesizer.get_synthesis_metrics()
            
            # Calculate feed-specific metrics
            featured_wisdom_count = sum(len(items) for items in self.featured_wisdom.values())
            daily_picks_count = len(self.daily_wisdom_picks)
            
            analytics = {
                'feed_type': 'community_wisdom',
                'wisdom_curation': curation_metrics,
                'insight_validation': validation_metrics,
                'knowledge_synthesis': synthesis_metrics,
                'feed_activity': {
                    'featured_wisdom_items': featured_wisdom_count,
                    'daily_wisdom_picks_available': daily_picks_count,
                    'pending_validations': len(self.pending_validations),
                    'recent_posts_processed': len(self.recent_posts)
                },
                'content_quality': {
                    'validation_threshold': self.validation_threshold,
                    'synthesis_trigger_threshold': self.synthesis_trigger_threshold,
                    'average_wisdom_score': self._calculate_average_wisdom_score(),
                    'trauma_informed_percentage': self._calculate_trauma_informed_percentage()
                },
                'system_health': {
                    'wisdom_curator_active': True,
                    'insight_validator_active': True,
                    'knowledge_synthesizer_active': True,
                    'wisdom_generation_healthy': curation_metrics['insights_curated'] > 0
                },
                'generated_at': datetime.utcnow().isoformat()
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error generating Community Wisdom analytics: {e}")
            return {'error': 'Unable to generate analytics'}
    
    def _calculate_average_wisdom_score(self) -> float:
        """Calculate average wisdom score across all curated insights"""
        
        all_insights = []
        for category in WisdomCategory:
            insights = self.wisdom_curator.get_wisdom_by_category(category)
            all_insights.extend(insights)
        
        if not all_insights:
            return 0.0
        
        total_score = sum(insight.wisdom_score for insight in all_insights)
        return total_score / len(all_insights)
    
    def _calculate_trauma_informed_percentage(self) -> float:
        """Calculate percentage of insights that are trauma-informed"""
        
        all_insights = []
        for category in WisdomCategory:
            insights = self.wisdom_curator.get_wisdom_by_category(category)
            all_insights.extend(insights)
        
        if not all_insights:
            return 0.0
        
        trauma_informed_count = sum(1 for insight in all_insights if insight.trauma_informed)
        return (trauma_informed_count / len(all_insights)) * 100