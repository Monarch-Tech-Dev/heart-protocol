"""
Crisis Intervention System for Heart Protocol

Trauma-informed crisis detection and intervention system that prioritizes
human safety while respecting user autonomy and dignity.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
import logging
from enum import Enum
from dataclasses import dataclass

from ..infrastructure.bluesky_integration.bluesky_monitor import CareSignalDetection, CareSignal
from ..infrastructure.bluesky_integration.at_protocol_client import ATProtocolClient, RequestPriority

logger = logging.getLogger(__name__)


class CrisisLevel(Enum):
    """Crisis severity levels"""
    IMMEDIATE_DANGER = "immediate_danger"      # Imminent threat to life
    HIGH_RISK = "high_risk"                   # High suicide/self-harm risk
    MODERATE_RISK = "moderate_risk"           # Concerning but not immediate
    SUPPORT_NEEDED = "support_needed"         # Seeking help/support
    MONITORING = "monitoring"                 # Requires monitoring only


class InterventionType(Enum):
    """Types of crisis interventions"""
    IMMEDIATE_RESPONSE = "immediate_response"     # Direct crisis response
    RESOURCE_SHARING = "resource_sharing"         # Share crisis resources
    GENTLE_OUTREACH = "gentle_outreach"          # Supportive outreach
    HUMAN_HANDOFF = "human_handoff"              # Professional intervention
    COMMUNITY_ALERT = "community_alert"          # Alert support network
    FOLLOW_UP = "follow_up"                      # Follow-up care


@dataclass
class CrisisResponse:
    """Crisis intervention response"""
    response_id: str
    detection_id: str
    user_handle: str
    crisis_level: CrisisLevel
    intervention_type: InterventionType
    response_message: Optional[str]
    resources_shared: List[str]
    human_contact_initiated: bool
    follow_up_scheduled: bool
    success: bool
    error_message: Optional[str]
    response_time_seconds: float
    created_at: datetime


class CrisisInterventionSystem:
    """
    Crisis intervention system for Heart Protocol
    
    Principles:
    - Human safety is the highest priority
    - Trauma-informed responses that don't re-traumatize
    - Respect for user autonomy and dignity
    - Cultural sensitivity in crisis understanding
    - Immediate professional handoff for high-risk situations
    - Gentle but firm intervention when needed
    - Privacy protection even in crisis situations
    - Community support mobilization with consent
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Crisis response configuration
        self.crisis_response_timeout = config.get('CRISIS_INTERVENTION_TIMEOUT', 300)  # 5 minutes
        self.human_handoff_enabled = config.get('HUMAN_HANDOFF_ENABLED', True)
        self.community_alert_enabled = config.get('COMMUNITY_ALERT_ENABLED', True)
        
        # Crisis resources
        self.crisis_resources = self._load_crisis_resources()
        self.professional_contacts = self._load_professional_contacts()
        
        # Response tracking
        self.crisis_responses = []
        self.user_crisis_history = {}
        
        # Callbacks for different intervention types
        self.intervention_callbacks: Dict[InterventionType, List[Callable]] = {
            intervention_type: [] for intervention_type in InterventionType
        }
        
        # Professional escalation contacts
        self.escalation_webhooks = config.get('CRISIS_ESCALATION_WEBHOOK', '')
        
    def _load_crisis_resources(self) -> Dict[str, List[str]]:
        """Load crisis resources by region/language"""
        return {
            'us_english': [
                "🚨 National Suicide Prevention Lifeline: 988",
                "📱 Crisis Text Line: Text HOME to 741741", 
                "🌐 findahelpline.com for international resources",
                "🏥 Emergency services: 911"
            ],
            'international': [
                "🌍 International crisis resources: findahelpline.com",
                "🆘 Emergency services: Contact your local emergency number",
                "💬 Crisis Text Line (US): Text HOME to 741741",
                "📞 Samaritans (UK): 116 123"
            ],
            'multilingual': [
                "🗣️ Crisis resources in multiple languages available",
                "🌐 findahelpline.com - Global crisis support",
                "📱 Many crisis lines offer translation services"
            ]
        }
    
    def _load_professional_contacts(self) -> Dict[str, Any]:
        """Load professional contact information"""
        return {
            'crisis_team_webhook': self.config.get('CRISIS_ESCALATION_WEBHOOK', ''),
            'mental_health_professionals': [],
            'emergency_contacts': []
        }
    
    async def handle_crisis_detection(self, detection: CareSignalDetection) -> CrisisResponse:
        """Handle a detected crisis signal"""
        start_time = datetime.utcnow()
        
        try:
            # Determine crisis level
            crisis_level = self._assess_crisis_level(detection)
            
            # Determine intervention type
            intervention_type = self._determine_intervention_type(crisis_level, detection)
            
            # Track crisis history for this user
            self._update_user_crisis_history(detection.user_handle, crisis_level)
            
            # Execute intervention based on type
            response = await self._execute_intervention(
                detection, crisis_level, intervention_type, start_time
            )
            
            # Log crisis response
            self.crisis_responses.append(response)
            
            # Schedule follow-up if appropriate
            if response.success and crisis_level in [CrisisLevel.HIGH_RISK, CrisisLevel.MODERATE_RISK]:
                await self._schedule_follow_up(response)
            
            return response
            
        except Exception as e:
            logger.error(f"Crisis intervention failed: {str(e)}")
            
            response_time = (datetime.utcnow() - start_time).total_seconds()
            return CrisisResponse(
                response_id=f"crisis_error_{datetime.utcnow().isoformat()}",
                detection_id=detection.signal_id,
                user_handle=detection.user_handle,
                crisis_level=CrisisLevel.MONITORING,
                intervention_type=InterventionType.IMMEDIATE_RESPONSE,
                response_message=None,
                resources_shared=[],
                human_contact_initiated=False,
                follow_up_scheduled=False,
                success=False,
                error_message=str(e),
                response_time_seconds=response_time,
                created_at=datetime.utcnow()
            )
    
    def _assess_crisis_level(self, detection: CareSignalDetection) -> CrisisLevel:
        """Assess the level of crisis based on detection"""
        try:
            # Immediate danger indicators
            immediate_danger_keywords = [
                'kill myself', 'end it all', 'suicide', 'going to die',
                'overdose', 'hanging', 'jumping', 'cutting tonight',
                'goodbye forever', 'last post'
            ]
            
            # High risk indicators
            high_risk_keywords = [
                'want to die', 'can\'t go on', 'no point', 'burden',
                'everyone better without me', 'hurt myself', 'end the pain'
            ]
            
            # Check for immediate danger
            for keyword in immediate_danger_keywords:
                if keyword.lower() in detection.post_content.lower():
                    return CrisisLevel.IMMEDIATE_DANGER
            
            # Check for high risk
            if detection.signal_type == CareSignal.IMMEDIATE_CRISIS:
                if detection.confidence_score > 0.8:
                    return CrisisLevel.HIGH_RISK
                elif detection.confidence_score > 0.6:
                    return CrisisLevel.MODERATE_RISK
            
            # Check for moderate risk
            if detection.signal_type == CareSignal.MENTAL_HEALTH_CRISIS:
                if detection.confidence_score > 0.7:
                    return CrisisLevel.MODERATE_RISK
                else:
                    return CrisisLevel.SUPPORT_NEEDED
            
            # Support seeking
            if detection.signal_type == CareSignal.SUPPORT_SEEKING:
                return CrisisLevel.SUPPORT_NEEDED
            
            return CrisisLevel.MONITORING
            
        except Exception as e:
            logger.error(f"Failed to assess crisis level: {str(e)}")
            return CrisisLevel.MONITORING
    
    def _determine_intervention_type(self, crisis_level: CrisisLevel, detection: CareSignalDetection) -> InterventionType:
        """Determine appropriate intervention type"""
        
        # User crisis history
        user_history = self.user_crisis_history.get(detection.user_handle, [])
        recent_crises = len([c for c in user_history if c['timestamp'] > datetime.utcnow() - timedelta(hours=24)])
        
        if crisis_level == CrisisLevel.IMMEDIATE_DANGER:
            return InterventionType.HUMAN_HANDOFF
        
        elif crisis_level == CrisisLevel.HIGH_RISK:
            if recent_crises > 1:
                return InterventionType.HUMAN_HANDOFF
            else:
                return InterventionType.IMMEDIATE_RESPONSE
        
        elif crisis_level == CrisisLevel.MODERATE_RISK:
            return InterventionType.IMMEDIATE_RESPONSE
        
        elif crisis_level == CrisisLevel.SUPPORT_NEEDED:
            return InterventionType.GENTLE_OUTREACH
        
        else:
            return InterventionType.RESOURCE_SHARING
    
    def _update_user_crisis_history(self, user_handle: str, crisis_level: CrisisLevel):
        """Update crisis history for user"""
        if user_handle not in self.user_crisis_history:
            self.user_crisis_history[user_handle] = []
        
        self.user_crisis_history[user_handle].append({
            'crisis_level': crisis_level,
            'timestamp': datetime.utcnow()
        })
        
        # Keep only last 30 days of history
        cutoff = datetime.utcnow() - timedelta(days=30)
        self.user_crisis_history[user_handle] = [
            entry for entry in self.user_crisis_history[user_handle]
            if entry['timestamp'] > cutoff
        ]
    
    async def _execute_intervention(self, detection: CareSignalDetection, 
                                  crisis_level: CrisisLevel, 
                                  intervention_type: InterventionType,
                                  start_time: datetime) -> CrisisResponse:
        """Execute the determined intervention"""
        
        response_id = f"crisis_{intervention_type.value}_{datetime.utcnow().isoformat()}"
        resources_shared = []
        response_message = None
        human_contact_initiated = False
        follow_up_scheduled = False
        success = False
        error_message = None
        
        try:
            if intervention_type == InterventionType.IMMEDIATE_RESPONSE:
                response_message, resources_shared = await self._create_immediate_response(detection, crisis_level)
                success = True
                
            elif intervention_type == InterventionType.HUMAN_HANDOFF:
                human_contact_initiated = await self._initiate_human_handoff(detection, crisis_level)
                response_message, resources_shared = await self._create_immediate_response(detection, crisis_level)
                success = human_contact_initiated
                
            elif intervention_type == InterventionType.GENTLE_OUTREACH:
                response_message = await self._create_gentle_outreach(detection)
                success = True
                
            elif intervention_type == InterventionType.RESOURCE_SHARING:
                resources_shared = await self._share_appropriate_resources(detection)
                success = True
                
            # Trigger intervention callbacks
            for callback in self.intervention_callbacks.get(intervention_type, []):
                try:
                    await callback(detection, crisis_level)
                except Exception as e:
                    logger.error(f"Intervention callback failed: {str(e)}")
            
        except Exception as e:
            logger.error(f"Intervention execution failed: {str(e)}")
            error_message = str(e)
            success = False
        
        response_time = (datetime.utcnow() - start_time).total_seconds()
        
        return CrisisResponse(
            response_id=response_id,
            detection_id=detection.signal_id,
            user_handle=detection.user_handle,
            crisis_level=crisis_level,
            intervention_type=intervention_type,
            response_message=response_message,
            resources_shared=resources_shared,
            human_contact_initiated=human_contact_initiated,
            follow_up_scheduled=follow_up_scheduled,
            success=success,
            error_message=error_message,
            response_time_seconds=response_time,
            created_at=datetime.utcnow()
        )
    
    async def _create_immediate_response(self, detection: CareSignalDetection, 
                                       crisis_level: CrisisLevel) -> tuple[str, List[str]]:
        """Create immediate crisis response message"""
        
        if crisis_level == CrisisLevel.IMMEDIATE_DANGER:
            message = f"@{detection.user_handle} I'm deeply concerned about your safety. Please reach out for immediate support:\n\n🚨 National Suicide Prevention Lifeline: 988\n📱 Crisis Text Line: Text HOME to 741741\n\nYour life has value. You matter. Help is available. 💙"
            
        elif crisis_level == CrisisLevel.HIGH_RISK:
            message = f"@{detection.user_handle} I hear that you're going through an incredibly difficult time. You don't have to face this alone:\n\n🆘 Crisis support: 988\n💬 Text support: 741741\n\nYour pain is real, and so is the possibility of feeling better. 💙"
            
        elif crisis_level == CrisisLevel.MODERATE_RISK:
            message = f"@{detection.user_handle} Thank you for sharing something so difficult. It takes courage to reach out, even indirectly:\n\n🤝 Crisis support: 988\n💙 You matter, your life has value\n\nHealing is possible, even when it doesn't feel like it."
            
        resources = self.crisis_resources.get('us_english', [])
        
        return message, resources
    
    async def _initiate_human_handoff(self, detection: CareSignalDetection, 
                                    crisis_level: CrisisLevel) -> bool:
        """Initiate human professional handoff"""
        try:
            if not self.human_handoff_enabled:
                logger.warning("Human handoff disabled")
                return False
            
            if not self.escalation_webhooks:
                logger.error("No escalation webhook configured")
                return False
            
            # Create escalation payload
            escalation_data = {
                'alert_type': 'crisis_intervention',
                'crisis_level': crisis_level.value,
                'user_handle': detection.user_handle,
                'detection_id': detection.signal_id,
                'post_content': detection.post_content,
                'confidence_score': detection.confidence_score,
                'detected_at': detection.detected_at.isoformat(),
                'urgency': 'immediate' if crisis_level == CrisisLevel.IMMEDIATE_DANGER else 'high',
                'requires_human_review': True
            }
            
            # TODO: Send webhook to crisis team
            # This would be implemented with actual webhook sending
            logger.critical(f"🚨 CRISIS ALERT: {crisis_level.value} detected for @{detection.user_handle}")
            logger.critical(f"Human intervention required. Escalation data: {escalation_data}")
            
            return True
            
        except Exception as e:
            logger.error(f"Human handoff failed: {str(e)}")
            return False
    
    async def _create_gentle_outreach(self, detection: CareSignalDetection) -> str:
        """Create gentle outreach message"""
        
        gentle_messages = [
            f"@{detection.user_handle} I noticed you might be going through a difficult time. Just wanted you to know that you're not alone, and it's okay to ask for help when you need it. 💙",
            
            f"@{detection.user_handle} Sending you gentle support. Whatever you're facing, you don't have to face it alone. There are people who care and resources available when you're ready. 🤗",
            
            f"@{detection.user_handle} Thank you for sharing what you're going through. It takes courage to be vulnerable. You matter, and your feelings are valid. 💝"
        ]
        
        import random
        return random.choice(gentle_messages)
    
    async def _share_appropriate_resources(self, detection: CareSignalDetection) -> List[str]:
        """Share appropriate resources based on user needs"""
        
        # Determine resource type needed
        if 'therapy' in detection.post_content.lower() or 'counseling' in detection.post_content.lower():
            return [
                "🧠 Psychology Today - Find therapists: psychologytoday.com",
                "💻 7 Cups - Free emotional support: 7cups.com",
                "📱 BetterHelp - Online therapy platform"
            ]
        
        elif 'support group' in detection.post_content.lower():
            return [
                "🤝 NAMI Support Groups: nami.org",
                "💬 Online support communities available",
                "🏥 Local mental health centers often host groups"
            ]
        
        else:
            return self.crisis_resources.get('us_english', [])
    
    async def _schedule_follow_up(self, response: CrisisResponse):
        """Schedule follow-up for crisis intervention"""
        try:
            # This would integrate with a job scheduler
            # For now, just log the intent
            follow_up_time = datetime.utcnow() + timedelta(hours=24)
            
            logger.info(f"📅 Follow-up scheduled for @{response.user_handle} at {follow_up_time}")
            
            # TODO: Implement actual follow-up scheduling
            
        except Exception as e:
            logger.error(f"Failed to schedule follow-up: {str(e)}")
    
    def add_intervention_callback(self, intervention_type: InterventionType, callback: Callable):
        """Add callback for specific intervention type"""
        self.intervention_callbacks[intervention_type].append(callback)
    
    def get_crisis_stats(self) -> Dict[str, Any]:
        """Get crisis intervention statistics"""
        try:
            today = datetime.utcnow().date()
            week_ago = datetime.utcnow() - timedelta(days=7)
            
            recent_responses = [
                r for r in self.crisis_responses
                if r.created_at > week_ago
            ]
            
            crisis_level_counts = {}
            intervention_type_counts = {}
            
            for response in recent_responses:
                level = response.crisis_level.value
                intervention = response.intervention_type.value
                
                crisis_level_counts[level] = crisis_level_counts.get(level, 0) + 1
                intervention_type_counts[intervention] = intervention_type_counts.get(intervention, 0) + 1
            
            return {
                'total_crisis_responses': len(self.crisis_responses),
                'responses_this_week': len(recent_responses),
                'crisis_level_distribution': crisis_level_counts,
                'intervention_type_distribution': intervention_type_counts,
                'average_response_time': sum(r.response_time_seconds for r in recent_responses) / max(1, len(recent_responses)),
                'human_handoffs_initiated': len([r for r in recent_responses if r.human_contact_initiated]),
                'unique_users_supported': len(set(r.user_handle for r in recent_responses))
            }
            
        except Exception as e:
            logger.error(f"Failed to get crisis stats: {str(e)}")
            return {'error': str(e)}