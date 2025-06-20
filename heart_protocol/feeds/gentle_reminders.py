"""
Gentle Reminders Feed - Daily affirmations and wellbeing reminders

Provides trauma-informed, culturally sensitive daily reminders that affirm
user worth and provide gentle encouragement for healing.
"""

import asyncio
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

from ..infrastructure.bluesky_integration.at_protocol_client import ATProtocolClient, RequestPriority

logger = logging.getLogger(__name__)


class GentleRemindersFeed:
    """
    Daily gentle reminders feed for Heart Protocol
    
    Principles:
    - Affirm inherent worth regardless of circumstances
    - Trauma-informed language that never minimizes pain
    - Cultural sensitivity in expressions of care
    - Timing that respects different life rhythms
    - Consent-based personalization over algorithmic forcing
    """
    
    def __init__(self, at_client: ATProtocolClient, config: Dict[str, Any]):
        self.at_client = at_client
        self.config = config
        
        # Gentle reminder collections
        self.affirmations = self._load_affirmations()
        self.encouragements = self._load_encouragements()
        self.care_reminders = self._load_care_reminders()
        self.boundary_reminders = self._load_boundary_reminders()
        
        # Timing and personalization
        self.last_reminder_time = None
        self.reminder_history = []
        
    def _load_affirmations(self) -> List[Dict[str, str]]:
        """Load daily affirmations focused on inherent worth"""
        return [
            {
                'text': "You are worthy of love exactly as you are right now. Not when you're 'better' or 'fixed' - right now. 💙",
                'category': 'worth',
                'tone': 'gentle'
            },
            {
                'text': "Your feelings are valid, even the difficult ones. Especially the difficult ones. You don't have to earn the right to feel. 🤗",
                'category': 'validation',
                'tone': 'accepting'
            },
            {
                'text': "Healing isn't linear. It's okay to have setbacks, rough days, and moments of doubt. Progress isn't always visible. 🌱",
                'category': 'healing',
                'tone': 'patient'
            },
            {
                'text': "You belong here. In this community, in this world, in this moment. Your existence matters and makes a difference. ✨",
                'category': 'belonging',
                'tone': 'affirming'
            },
            {
                'text': "Rest is productive. Rest is healing. Rest is not giving up - it's giving yourself what you need. 😴",
                'category': 'rest',
                'tone': 'permission'
            },
            {
                'text': "You don't have to be grateful for trauma to acknowledge your growth. Your strength is real AND you shouldn't have had to develop it this way. 💪",
                'category': 'trauma',
                'tone': 'validating'
            },
            {
                'text': "Small steps count. Tiny progress counts. Just surviving difficult days counts. You're doing better than you think. 👣",
                'category': 'progress',
                'tone': 'encouraging'
            },
            {
                'text': "It's okay to not be okay. You don't owe anyone constant positivity or happiness. Your struggles are real and valid. 🌧️",
                'category': 'permission',
                'tone': 'understanding'
            },
            {
                'text': "You are not broken. You are not too much. You are not a burden. You are a human being deserving of care and compassion. 💝",
                'category': 'worth',
                'tone': 'powerful'
            },
            {
                'text': "Your pace is perfect for you. You don't have to heal on anyone else's timeline or meet anyone else's expectations. ⏰",
                'category': 'pace',
                'tone': 'accepting'
            }
        ]
    
    def _load_encouragements(self) -> List[Dict[str, str]]:
        """Load encouragements for difficult times"""
        return [
            {
                'text': "This feeling will pass. Not because someone said 'this too shall pass' but because feelings are temporary visitors, not permanent residents. 🌊",
                'category': 'hope',
                'tone': 'realistic'
            },
            {
                'text': "You've survived 100% of your worst days so far. That's a pretty incredible track record. Today might be hard, but you've got this. 🌟",
                'category': 'resilience',
                'tone': 'empowering'
            },
            {
                'text': "Asking for help isn't weakness - it's wisdom. It's courage. It's taking care of yourself in the way you deserve. 🤝",
                'category': 'help',
                'tone': 'encouraging'
            },
            {
                'text': "Your mental health is just as important as your physical health. Taking care of your mind is taking care of yourself. 🧠",
                'category': 'health',
                'tone': 'normalizing'
            },
            {
                'text': "You don't have to earn care or love through suffering. You deserve support simply because you exist. 💙",
                'category': 'deserving',
                'tone': 'affirming'
            }
        ]
    
    def _load_care_reminders(self) -> List[Dict[str, str]]:
        """Load self-care and community care reminders"""
        return [
            {
                'text': "Gentle reminder: Have you eaten something nourishing today? Have you had water? Have you taken any deep breaths? 🍎💧🌬️",
                'category': 'basic_needs',
                'tone': 'caring'
            },
            {
                'text': "Self-care isn't selfish. Taking care of yourself allows you to show up more fully for others when you choose to. 🌸",
                'category': 'self_care',
                'tone': 'permission'
            },
            {
                'text': "Community care is revolutionary. Checking on your people, offering support, creating safety - this is how we heal together. 🌍",
                'category': 'community',
                'tone': 'inspiring'
            },
            {
                'text': "Boundaries are not walls - they're gates with you as the gatekeeper. You get to decide who enters and when. 🚪",
                'category': 'boundaries',
                'tone': 'empowering'
            },
            {
                'text': "Your energy is precious. It's okay to be selective about where and how you spend it. Protect your peace. ⚡",
                'category': 'energy',
                'tone': 'protective'
            }
        ]
    
    def _load_boundary_reminders(self) -> List[Dict[str, str]]:
        """Load boundary and consent reminders"""
        return [
            {
                'text': "No is a complete sentence. You don't need to justify, explain, or apologize for your boundaries. 🛑",
                'category': 'boundaries',
                'tone': 'firm'
            },
            {
                'text': "You can change your mind. You can withdraw consent. You can say no after saying yes. Your autonomy matters. 🔄",
                'category': 'consent',
                'tone': 'empowering'
            },
            {
                'text': "You don't owe anyone access to your trauma story, your body, your time, or your energy. Choose who gets these gifts. 💝",
                'category': 'autonomy',
                'tone': 'protective'
            },
            {
                'text': "Healthy relationships respect your boundaries. If someone pushes against them repeatedly, that tells you something important. 🚨",
                'category': 'relationships',
                'tone': 'warning'
            }
        ]
    
    async def post_daily_reminder(self) -> bool:
        """Post a daily gentle reminder"""
        try:
            # Check if we've already posted today
            if self._already_posted_today():
                logger.info("Daily reminder already posted today")
                return True
            
            # Select appropriate reminder based on time and context
            reminder = await self._select_contextual_reminder()
            
            if not reminder:
                logger.warning("No appropriate reminder found")
                return False
            
            # Create post with gentle formatting
            post_text = self._format_reminder_post(reminder)
            
            # Post the reminder
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
                self.last_reminder_time = datetime.utcnow()
                self.reminder_history.append({
                    'reminder': reminder,
                    'posted_at': self.last_reminder_time,
                    'post_uri': response.data.get('uri') if response.data else None
                })
                
                logger.info(f"✨ Posted daily gentle reminder: {reminder['category']}")
                return True
            else:
                logger.error(f"Failed to post daily reminder: {response.error_message}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to post daily reminder: {str(e)}")
            return False
    
    def _already_posted_today(self) -> bool:
        """Check if we've already posted a reminder today"""
        if not self.last_reminder_time:
            return False
        
        today = datetime.utcnow().date()
        last_post_date = self.last_reminder_time.date()
        
        return today == last_post_date
    
    async def _select_contextual_reminder(self) -> Optional[Dict[str, str]]:
        """Select an appropriate reminder based on context"""
        try:
            current_hour = datetime.utcnow().hour
            
            # Morning affirmations (6-11 AM UTC)
            if 6 <= current_hour <= 11:
                return random.choice(self.affirmations + self.care_reminders)
            
            # Afternoon encouragement (12-17 PM UTC)
            elif 12 <= current_hour <= 17:
                return random.choice(self.encouragements + self.care_reminders)
            
            # Evening validation (18-23 PM UTC)
            elif 18 <= current_hour <= 23:
                return random.choice(self.affirmations + self.boundary_reminders)
            
            # Night support (24-5 AM UTC)
            else:
                return random.choice(self.encouragements + self.affirmations)
                
        except Exception as e:
            logger.error(f"Failed to select contextual reminder: {str(e)}")
            return random.choice(self.affirmations)
    
    def _format_reminder_post(self, reminder: Dict[str, str]) -> str:
        """Format reminder into a gentle social media post"""
        
        # Add gentle intro based on time of day
        current_hour = datetime.utcnow().hour
        
        if 6 <= current_hour <= 11:
            intro = "🌅 Morning gentle reminder:"
        elif 12 <= current_hour <= 17:
            intro = "☀️ Afternoon reflection:"
        elif 18 <= current_hour <= 23:
            intro = "🌙 Evening wisdom:"
        else:
            intro = "✨ Night light:"
        
        # Format the full post
        post_text = f"{intro}\n\n{reminder['text']}\n\n#GentleReminders #HeartProtocol #MentalHealthSupport #SelfCare"
        
        # Ensure post isn't too long for Bluesky (300 chars)
        if len(post_text) > 280:
            # Shorten hashtags if needed
            post_text = f"{intro}\n\n{reminder['text']}\n\n#GentleReminders #HeartProtocol"
        
        return post_text
    
    async def get_reminder_by_category(self, category: str) -> Optional[Dict[str, str]]:
        """Get a specific reminder by category"""
        try:
            all_reminders = (
                self.affirmations + 
                self.encouragements + 
                self.care_reminders + 
                self.boundary_reminders
            )
            
            category_reminders = [r for r in all_reminders if r['category'] == category]
            
            if category_reminders:
                return random.choice(category_reminders)
            else:
                return None
                
        except Exception as e:
            logger.error(f"Failed to get reminder by category: {str(e)}")
            return None
    
    async def get_reminder_stats(self) -> Dict[str, Any]:
        """Get statistics about reminder posting"""
        try:
            today = datetime.utcnow().date()
            week_ago = datetime.utcnow() - timedelta(days=7)
            
            recent_reminders = [
                r for r in self.reminder_history 
                if r['posted_at'] > week_ago
            ]
            
            category_counts = {}
            for reminder in recent_reminders:
                category = reminder['reminder']['category']
                category_counts[category] = category_counts.get(category, 0) + 1
            
            return {
                'total_reminders_posted': len(self.reminder_history),
                'reminders_this_week': len(recent_reminders),
                'posted_today': self._already_posted_today(),
                'last_posted': self.last_reminder_time.isoformat() if self.last_reminder_time else None,
                'category_distribution': category_counts,
                'available_categories': list(set(r['category'] for r in self.affirmations + self.encouragements + self.care_reminders + self.boundary_reminders))
            }
            
        except Exception as e:
            logger.error(f"Failed to get reminder stats: {str(e)}")
            return {'error': str(e)}
    
    def add_custom_reminder(self, text: str, category: str, tone: str = 'gentle'):
        """Add a custom reminder to the collection"""
        try:
            custom_reminder = {
                'text': text,
                'category': category,
                'tone': tone,
                'custom': True
            }
            
            # Add to appropriate collection based on category
            if category in ['worth', 'validation', 'healing', 'belonging']:
                self.affirmations.append(custom_reminder)
            elif category in ['hope', 'resilience', 'help']:
                self.encouragements.append(custom_reminder)
            elif category in ['basic_needs', 'self_care', 'community']:
                self.care_reminders.append(custom_reminder)
            elif category in ['boundaries', 'consent', 'autonomy']:
                self.boundary_reminders.append(custom_reminder)
            else:
                self.affirmations.append(custom_reminder)  # Default to affirmations
            
            logger.info(f"Added custom reminder: {category}")
            
        except Exception as e:
            logger.error(f"Failed to add custom reminder: {str(e)}")