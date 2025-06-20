"""
Community Wisdom Feed - Curated healing insights and resources

Shares validated community healing wisdom, resources, and insights with
trauma-informed, culturally sensitive curation.
"""

import asyncio
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

from ..infrastructure.bluesky_integration.at_protocol_client import ATProtocolClient, RequestPriority

logger = logging.getLogger(__name__)


class CommunityWisdomFeed:
    """
    Community wisdom feed for Heart Protocol
    
    Principles:
    - Evidence-based healing practices and resources
    - Cultural humility in wisdom sharing
    - Trauma-informed approach to advice
    - Community-sourced and professionally validated content
    - Accessibility and inclusion in all resources
    """
    
    def __init__(self, at_client: ATProtocolClient, config: Dict[str, Any]):
        self.at_client = at_client
        self.config = config
        
        # Wisdom collections
        self.healing_wisdom = self._load_healing_wisdom()
        self.coping_strategies = self._load_coping_strategies()
        self.resource_shares = self._load_resource_shares()
        self.community_insights = self._load_community_insights()
        
        # Sharing history and timing
        self.last_wisdom_time = None
        self.wisdom_history = []
        
    def _load_healing_wisdom(self) -> List[Dict[str, str]]:
        """Load evidence-based healing wisdom"""
        return [
            {
                'text': "Grounding technique: Name 5 things you can see, 4 you can touch, 3 you can hear, 2 you can smell, 1 you can taste. This brings you back to the present moment when anxiety overwhelms. 🌱",
                'category': 'grounding',
                'source': 'anxiety_management',
                'evidence_based': True
            },
            {
                'text': "Healing happens in relationship. This can be with a therapist, friend, support group, or even a pet. We heal through connection, not isolation. 🤝",
                'category': 'connection',
                'source': 'trauma_therapy',
                'evidence_based': True
            },
            {
                'text': "Your body holds wisdom about your trauma. Learning to listen to its signals - tension, fatigue, restlessness - can guide your healing process. 💪",
                'category': 'somatic',
                'source': 'body_based_healing',
                'evidence_based': True
            },
            {
                'text': "Trauma responses (fight, flight, freeze, fawn) are adaptive survival mechanisms, not character flaws. Understanding this can reduce self-judgment. 🧠",
                'category': 'trauma_education',
                'source': 'polyvagal_theory',
                'evidence_based': True
            },
            {
                'text': "Depression often lies to us. It says we're worthless, hopeless, burden. These are symptoms, not truths. Your brain chemistry is not your character. 🌦️",
                'category': 'depression',
                'source': 'cognitive_therapy',
                'evidence_based': True
            },
            {
                'text': "Mindfulness isn't about emptying your mind - it's about noticing what's there without judgment. Even 3 minutes of observing your breath can help. 🧘",
                'category': 'mindfulness',
                'source': 'meditation_research',
                'evidence_based': True
            },
            {
                'text': "Boundaries teach people how to treat you. They're not punishment - they're information about what you need to feel safe and respected. 🛡️",
                'category': 'boundaries',
                'source': 'relationship_therapy',
                'evidence_based': True
            },
            {
                'text': "Post-traumatic growth is real, but it doesn't mean you should be grateful for trauma. You can acknowledge growth while still recognizing that you deserved better. 🌱",
                'category': 'growth',
                'source': 'trauma_recovery',
                'evidence_based': True
            }
        ]
    
    def _load_coping_strategies(self) -> List[Dict[str, str]]:
        """Load practical coping strategies"""
        return [
            {
                'text': "The HALT check: Are you Hungry, Angry, Lonely, or Tired? Sometimes addressing basic needs can shift your entire emotional state. 🍎😴🤗",
                'category': 'basic_needs',
                'source': 'addiction_recovery',
                'accessibility': 'universal'
            },
            {
                'text': "Box breathing for panic: Breathe in for 4, hold for 4, out for 4, hold for 4. Repeat. This activates your parasympathetic nervous system. 📦💨",
                'category': 'breathing',
                'source': 'anxiety_management',
                'accessibility': 'universal'
            },
            {
                'text': "Create a crisis plan when you're stable: list warning signs, coping strategies, supportive people, and professional resources. Use it when things get hard. 📋",
                'category': 'crisis_planning',
                'source': 'suicide_prevention',
                'accessibility': 'requires_literacy'
            },
            {
                'text': "The 2-minute rule: If a task takes less than 2 minutes, do it now. This prevents overwhelm from building up and gives you small wins. ⏰",
                'category': 'productivity',
                'source': 'depression_management',
                'accessibility': 'universal'
            },
            {
                'text': "Temperature change for emotional regulation: Cold water on wrists, ice cube in mouth, or warm bath can help reset your nervous system quickly. 🌡️",
                'category': 'nervous_system',
                'source': 'dialectical_behavior_therapy',
                'accessibility': 'requires_resources'
            },
            {
                'text': "The STOP technique: Stop what you're doing, Take a breath, Observe your thoughts/feelings, Proceed with intention. Simple but powerful. ✋",
                'category': 'mindfulness',
                'source': 'mindfulness_therapy',
                'accessibility': 'universal'
            }
        ]
    
    def _load_resource_shares(self) -> List[Dict[str, str]]:
        """Load resource sharing posts"""
        return [
            {
                'text': "🚨 Crisis Resources 🚨\n\nUS: National Suicide Prevention Lifeline - 988\nCrisis Text Line - Text HOME to 741741\nInternational: findahelpline.com\n\nYour life has value. Help is available. 💙",
                'category': 'crisis_resources',
                'urgency': 'high',
                'geographic': 'global'
            },
            {
                'text': "Free therapy resources:\n\n💻 7 Cups - online emotional support\n📱 Crisis Text Line - 24/7 crisis support\n🧠 MindShift app - anxiety management\n📚 Centre for Addiction and Mental Health (CAMH) - self-help resources",
                'category': 'free_resources',
                'urgency': 'medium',
                'geographic': 'english_speaking'
            },
            {
                'text': "Finding culturally competent therapists:\n\n🌍 Psychology Today cultural filters\n🤝 National Queer and Trans Therapists database\n✊ Therapy for Black Girls\n🏳️‍🌈 LGBTQ+ affirming provider directories",
                'category': 'culturally_competent_care',
                'urgency': 'medium',
                'geographic': 'us_focused'
            },
            {
                'text': "Trauma-informed yoga and movement:\n\n🧘 Trauma-Sensitive Yoga International\n💃 Movement Medicine for trauma\n🏃 Mindful movement practices\n\nMovement can be medicine, but honor your body's needs. 💪",
                'category': 'movement_healing',
                'urgency': 'low',
                'geographic': 'global'
            }
        ]
    
    def _load_community_insights(self) -> List[Dict[str, str]]:
        """Load community-sourced insights"""
        return [
            {
                'text': "Community wisdom: 'Healing isn't about becoming who you were before - it's about becoming who you're meant to be.' - Sarah, trauma survivor 🦋",
                'category': 'lived_experience',
                'source': 'community_member',
                'verified': True
            },
            {
                'text': "From our community: 'I used to think asking for help was weak. Now I know it's the strongest thing I've ever done.' - Alex, depression recovery 💪",
                'category': 'help_seeking',
                'source': 'community_member',
                'verified': True
            },
            {
                'text': "Therapist insight: 'Progress in therapy isn't always feeling better - sometimes it's feeling your feelings instead of numbing them.' - Dr. Martinez, LCSW 🌊",
                'category': 'therapy_wisdom',
                'source': 'mental_health_professional',
                'verified': True
            },
            {
                'text': "Peer support insight: 'Your worst day in recovery is still better than your best day in active addiction/mental illness.' - Recovery community wisdom 🌅",
                'category': 'recovery',
                'source': 'peer_support',
                'verified': True
            }
        ]
    
    async def post_wisdom_share(self) -> bool:
        """Post community wisdom"""
        try:
            # Check if we've already posted today
            if self._already_shared_today():
                logger.info("Wisdom already shared today")
                return True
            
            # Select appropriate wisdom based on context
            wisdom = await self._select_contextual_wisdom()
            
            if not wisdom:
                logger.warning("No appropriate wisdom found")
                return False
            
            # Create post with thoughtful formatting
            post_text = self._format_wisdom_post(wisdom)
            
            # Post the wisdom
            post_data = {
                'text': post_text,
                'langs': ['en'],  # TODO: Support multiple languages
                'createdAt': datetime.utcnow().isoformat() + 'Z'
            }
            
            response = await self.at_client.create_post(
                post_data,
                priority=RequestPriority.CARE_DELIVERY
            )
            
            if response.success:
                self.last_wisdom_time = datetime.utcnow()
                self.wisdom_history.append({
                    'wisdom': wisdom,
                    'posted_at': self.last_wisdom_time,
                    'post_uri': response.data.get('uri') if response.data else None
                })
                
                logger.info(f"✨ Posted community wisdom: {wisdom['category']}")
                return True
            else:
                logger.error(f"Failed to post wisdom: {response.error_message}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to post wisdom: {str(e)}")
            return False
    
    def _already_shared_today(self) -> bool:
        """Check if we've already shared wisdom today"""
        if not self.last_wisdom_time:
            return False
        
        today = datetime.utcnow().date()
        last_share_date = self.last_wisdom_time.date()
        
        return today == last_share_date
    
    async def _select_contextual_wisdom(self) -> Optional[Dict[str, str]]:
        """Select appropriate wisdom based on context and timing"""
        try:
            current_hour = datetime.utcnow().hour
            day_of_week = datetime.utcnow().weekday()  # 0=Monday, 6=Sunday
            
            # Monday: Start week with coping strategies
            if day_of_week == 0:
                return random.choice(self.coping_strategies)
            
            # Wednesday: Mid-week healing wisdom
            elif day_of_week == 2:
                return random.choice(self.healing_wisdom)
            
            # Friday: Resource sharing for weekend
            elif day_of_week == 4:
                return random.choice(self.resource_shares)
            
            # Sunday: Community insights for reflection
            elif day_of_week == 6:
                return random.choice(self.community_insights)
            
            # Other days: Mix based on time
            else:
                if 18 <= current_hour <= 22:  # Evening reflection
                    return random.choice(self.healing_wisdom + self.community_insights)
                else:  # Practical support
                    return random.choice(self.coping_strategies + self.resource_shares)
                    
        except Exception as e:
            logger.error(f"Failed to select contextual wisdom: {str(e)}")
            return random.choice(self.healing_wisdom)
    
    def _format_wisdom_post(self, wisdom: Dict[str, str]) -> str:
        """Format wisdom into a thoughtful social media post"""
        
        # Add contextual intro
        if wisdom['category'] in ['crisis_resources', 'free_resources']:
            intro = "📚 Community Resource Share:"
        elif wisdom['category'] in ['lived_experience', 'help_seeking']:
            intro = "💝 Community Wisdom:"
        elif wisdom['category'] in ['grounding', 'breathing', 'mindfulness']:
            intro = "🧘 Healing Practice:"
        else:
            intro = "✨ Wisdom Share:"
        
        # Format the full post
        post_text = f"{intro}\n\n{wisdom['text']}\n\n#CommunityWisdom #HeartProtocol #MentalHealthResources #HealingTogether"
        
        # Ensure post isn't too long for Bluesky
        if len(post_text) > 280:
            # Shorten hashtags if needed
            post_text = f"{intro}\n\n{wisdom['text']}\n\n#CommunityWisdom #HeartProtocol"
            
            # If still too long, truncate wisdom text
            if len(post_text) > 280:
                max_wisdom_length = 280 - len(intro) - len("\n\n\n\n#CommunityWisdom #HeartProtocol")
                truncated_wisdom = wisdom['text'][:max_wisdom_length-3] + "..."
                post_text = f"{intro}\n\n{truncated_wisdom}\n\n#CommunityWisdom #HeartProtocol"
        
        return post_text
    
    async def share_crisis_resources(self) -> bool:
        """Share crisis resources immediately (emergency posting)"""
        try:
            crisis_resources = [w for w in self.resource_shares if w['category'] == 'crisis_resources']
            
            if not crisis_resources:
                logger.error("No crisis resources available")
                return False
            
            resource = random.choice(crisis_resources)
            post_text = self._format_wisdom_post(resource)
            
            post_data = {
                'text': post_text,
                'langs': ['en'],
                'createdAt': datetime.utcnow().isoformat() + 'Z'
            }
            
            response = await self.at_client.create_post(
                post_data,
                priority=RequestPriority.CRISIS_INTERVENTION
            )
            
            if response.success:
                logger.info("🚨 Crisis resources shared")
                return True
            else:
                logger.error(f"Failed to share crisis resources: {response.error_message}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to share crisis resources: {str(e)}")
            return False
    
    async def get_wisdom_by_category(self, category: str) -> Optional[Dict[str, str]]:
        """Get specific wisdom by category"""
        try:
            all_wisdom = (
                self.healing_wisdom + 
                self.coping_strategies + 
                self.resource_shares + 
                self.community_insights
            )
            
            category_wisdom = [w for w in all_wisdom if w['category'] == category]
            
            if category_wisdom:
                return random.choice(category_wisdom)
            else:
                return None
                
        except Exception as e:
            logger.error(f"Failed to get wisdom by category: {str(e)}")
            return None
    
    async def get_wisdom_stats(self) -> Dict[str, Any]:
        """Get statistics about wisdom sharing"""
        try:
            today = datetime.utcnow().date()
            week_ago = datetime.utcnow() - timedelta(days=7)
            
            recent_wisdom = [
                w for w in self.wisdom_history 
                if w['posted_at'] > week_ago
            ]
            
            category_counts = {}
            for wisdom in recent_wisdom:
                category = wisdom['wisdom']['category']
                category_counts[category] = category_counts.get(category, 0) + 1
            
            return {
                'total_wisdom_shared': len(self.wisdom_history),
                'wisdom_this_week': len(recent_wisdom),
                'shared_today': self._already_shared_today(),
                'last_shared': self.last_wisdom_time.isoformat() if self.last_wisdom_time else None,
                'category_distribution': category_counts,
                'available_categories': list(set(w['category'] for w in self.healing_wisdom + self.coping_strategies + self.resource_shares + self.community_insights))
            }
            
        except Exception as e:
            logger.error(f"Failed to get wisdom stats: {str(e)}")
            return {'error': str(e)}
    
    def add_community_wisdom(self, text: str, category: str, source: str = 'community_member', verified: bool = False):
        """Add community-contributed wisdom"""
        try:
            new_wisdom = {
                'text': text,
                'category': category,
                'source': source,
                'verified': verified,
                'added_at': datetime.utcnow().isoformat(),
                'community_contributed': True
            }
            
            # Add to appropriate collection
            if category in ['grounding', 'connection', 'somatic', 'trauma_education']:
                self.healing_wisdom.append(new_wisdom)
            elif category in ['basic_needs', 'breathing', 'crisis_planning']:
                self.coping_strategies.append(new_wisdom)
            elif category in ['crisis_resources', 'free_resources', 'culturally_competent_care']:
                self.resource_shares.append(new_wisdom)
            else:
                self.community_insights.append(new_wisdom)
            
            logger.info(f"Added community wisdom: {category} from {source}")
            
        except Exception as e:
            logger.error(f"Failed to add community wisdom: {str(e)}")
    
    async def validate_resource(self, resource_text: str) -> Dict[str, Any]:
        """Validate a resource for accuracy and safety"""
        try:
            # This would integrate with professional review system
            # For now, basic validation
            validation = {
                'is_safe': True,
                'is_accurate': True,
                'needs_review': False,
                'concerns': [],
                'validated_at': datetime.utcnow().isoformat()
            }
            
            # Check for potential harmful content
            harmful_indicators = ['cure', 'guaranteed', 'miracle', 'instant fix']
            for indicator in harmful_indicators:
                if indicator.lower() in resource_text.lower():
                    validation['needs_review'] = True
                    validation['concerns'].append(f"Contains potentially misleading term: {indicator}")
            
            # Check for crisis resources accuracy
            if '988' in resource_text or '741741' in resource_text:
                validation['is_crisis_resource'] = True
                validation['priority'] = 'high'
            
            return validation
            
        except Exception as e:
            logger.error(f"Failed to validate resource: {str(e)}")
            return {'error': str(e)}