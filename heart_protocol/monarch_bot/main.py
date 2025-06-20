#!/usr/bin/env python3
"""
Monarch Bot - Heart Protocol's Gentle AI Companion for Bluesky

A trauma-informed AI companion that provides:
- Daily gentle reminders
- Crisis intervention support
- Community care amplification
- Resource sharing with consent

Built with love to heal the digital world.
"""

import asyncio
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import signal
import sys

from ..infrastructure.bluesky_integration.at_protocol_client import (
    ATProtocolClient, ATProtocolConfig, AuthMethod, RequestPriority
)
from ..infrastructure.bluesky_integration.bluesky_monitor import (
    BlueSkyMonitor, MonitoringScope, CareSignal, CareSignalDetection
)
from ..core.care_algorithm import CareAlgorithm
from ..core.crisis_intervention import CrisisInterventionSystem
from ..feeds.gentle_reminders import GentleRemindersFeed
from ..feeds.community_wisdom import CommunityWisdomFeed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/var/log/monarch-bot.log')
    ]
)
logger = logging.getLogger(__name__)


class MonarchBot:
    """
    Monarch Bot - Heart Protocol's gentle AI companion for Bluesky
    
    Core Principles:
    - Trauma-informed interactions that never re-traumatize
    - Privacy-first approach with explicit user consent
    - Cultural sensitivity guides all responses
    - Crisis intervention with immediate human handoff
    - Community care amplification over individual engagement
    - Healing-focused metrics rather than engagement metrics
    - User agency preserved in all interactions
    - Gentle presence that respects boundaries
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.running = False
        self.startup_time = None
        
        # Initialize AT Protocol client
        at_config = ATProtocolConfig(
            service_url="https://bsky.social",
            identifier=config.get('BLUESKY_HANDLE', 'monarch.bsky.social'),
            auth_method=AuthMethod.APP_PASSWORD,
            credentials={
                'password': config.get('BLUESKY_APP_PASSWORD', '')
            },
            gentle_mode_enabled=True,
            privacy_protection_level='maximum',
            healing_focus_enabled=True,
            trauma_informed_operations=True
        )
        
        self.at_client = ATProtocolClient(at_config)
        self.monitor = BlueSkyMonitor(self.at_client, config)
        
        # Initialize care systems
        self.care_algorithm = CareAlgorithm(config)
        self.crisis_system = CrisisInterventionSystem(config)
        
        # Initialize feeds
        self.gentle_reminders = GentleRemindersFeed(self.at_client, config)
        self.community_wisdom = CommunityWisdomFeed(self.at_client, config)
        
        # Stats tracking
        self.stats = {
            'care_signals_detected': 0,
            'gentle_interventions': 0,
            'crisis_interventions': 0,
            'community_amplifications': 0,
            'healing_moments_created': 0,
            'users_supported': set(),
            'uptime_start': None
        }
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    async def start(self):
        """Start Monarch Bot with gentle initialization"""
        try:
            logger.info("🦋 Monarch Bot initializing - Heart Protocol v1.0")
            logger.info("💙 Building caring algorithms for digital healing")
            
            self.startup_time = datetime.utcnow()
            self.stats['uptime_start'] = self.startup_time
            self.running = True
            
            # Connect to Bluesky
            logger.info("🔗 Connecting to Bluesky with trauma-informed approach...")
            connection_result = await self.at_client.connect()
            
            if not connection_result.success:
                logger.error(f"❌ Failed to connect to Bluesky: {connection_result.error_message}")
                return
            
            logger.info("✅ Connected to Bluesky successfully")
            
            # Register care signal callbacks
            await self._setup_care_callbacks()
            
            # Start monitoring
            logger.info("👀 Starting gentle monitoring for care signals...")
            monitoring_scopes = [
                MonitoringScope.TARGETED_KEYWORDS,
                MonitoringScope.SEARCH_QUERIES,
                MonitoringScope.COMMUNITY_FEEDS
            ]
            
            # Care-focused keywords to monitor
            care_keywords = [
                "need help", "feeling depressed", "anxiety", "struggling", 
                "support group", "therapy", "mental health", "crisis",
                "healing journey", "recovery", "grateful", "breakthrough"
            ]
            
            # Start monitoring with gentle approach
            monitor_task = asyncio.create_task(
                self.monitor.start_monitoring(monitoring_scopes, keywords=care_keywords)
            )
            
            # Start daily care feeds
            logger.info("🌱 Starting daily care feeds...")
            feeds_task = asyncio.create_task(self._run_care_feeds())
            
            # Start heartbeat and health monitoring
            health_task = asyncio.create_task(self._health_monitor())
            
            # Post gentle introduction
            await self._post_gentle_introduction()
            
            logger.info("🦋 Monarch Bot is now spreading healing across Bluesky")
            logger.info("💫 Serving love, respecting choice, empowering communities")
            
            # Run until shutdown
            await asyncio.gather(
                monitor_task,
                feeds_task, 
                health_task,
                return_exceptions=True
            )
            
        except Exception as e:
            logger.error(f"💔 Monarch Bot startup failed: {str(e)}")
            await self.shutdown()
    
    async def _setup_care_callbacks(self):
        """Setup callbacks for different types of care signals"""
        
        # Crisis intervention callbacks
        self.monitor.add_crisis_callback(self._handle_crisis_signal)
        
        # Support seeking callbacks
        self.monitor.add_care_signal_callback(
            CareSignal.SUPPORT_SEEKING, 
            self._handle_support_seeking
        )
        
        # Healing progress callbacks  
        self.monitor.add_care_signal_callback(
            CareSignal.HEALING_PROGRESS,
            self._handle_healing_progress
        )
        
        # Isolation and loneliness callbacks
        self.monitor.add_care_signal_callback(
            CareSignal.ISOLATION_LONELINESS,
            self._handle_loneliness_signal
        )
        
        # Community care callbacks
        self.monitor.add_community_care_callback(self._amplify_community_care)
    
    async def _handle_crisis_signal(self, detection: CareSignalDetection):
        """Handle crisis signals with immediate human-centered response"""
        try:
            logger.warning(f"🚨 Crisis signal detected from @{detection.user_handle}")
            self.stats['crisis_interventions'] += 1
            self.stats['users_supported'].add(detection.user_handle)
            
            # Immediate crisis response through crisis intervention system
            response = await self.crisis_system.handle_crisis_detection(detection)
            
            if response.success:
                logger.info(f"✅ Crisis intervention initiated for @{detection.user_handle}")
            else:
                logger.error(f"❌ Crisis intervention failed for @{detection.user_handle}")
            
        except Exception as e:
            logger.error(f"Crisis signal handling failed: {str(e)}")
    
    async def _handle_support_seeking(self, detection: CareSignalDetection):
        """Handle support seeking with gentle resource sharing"""
        try:
            logger.info(f"🤝 Support seeking detected from @{detection.user_handle}")
            self.stats['gentle_interventions'] += 1
            self.stats['users_supported'].add(detection.user_handle)
            
            # Generate gentle, helpful response
            care_response = await self.care_algorithm.generate_support_response(detection)
            
            if care_response and care_response.should_respond:
                # Post gentle reply with resources
                await self._post_gentle_reply(detection, care_response.message)
                logger.info(f"💙 Gentle support offered to @{detection.user_handle}")
            
        except Exception as e:
            logger.error(f"Support seeking handling failed: {str(e)}")
    
    async def _handle_healing_progress(self, detection: CareSignalDetection):
        """Handle healing progress with celebration and amplification"""
        try:
            logger.info(f"🌱 Healing progress detected from @{detection.user_handle}")
            self.stats['community_amplifications'] += 1
            self.stats['healing_moments_created'] += 1
            
            # Amplify positive healing content
            amplification = await self.care_algorithm.generate_amplification_response(detection)
            
            if amplification and amplification.should_amplify:
                # Like and gently boost the healing content
                await self._amplify_healing_content(detection, amplification)
                logger.info(f"✨ Healing progress amplified for @{detection.user_handle}")
            
        except Exception as e:
            logger.error(f"Healing progress handling failed: {str(e)}")
    
    async def _handle_loneliness_signal(self, detection: CareSignalDetection):
        """Handle loneliness signals with gentle connection offers"""
        try:
            logger.info(f"💫 Loneliness signal detected from @{detection.user_handle}")
            self.stats['gentle_interventions'] += 1
            self.stats['users_supported'].add(detection.user_handle)
            
            # Generate gentle connection response
            connection_response = await self.care_algorithm.generate_connection_response(detection)
            
            if connection_response and connection_response.should_respond:
                # Post gentle reply offering connection
                await self._post_gentle_reply(detection, connection_response.message)
                logger.info(f"🤗 Gentle connection offered to @{detection.user_handle}")
            
        except Exception as e:
            logger.error(f"Loneliness signal handling failed: {str(e)}")
    
    async def _amplify_community_care(self, detection: CareSignalDetection):
        """Amplify community care and support"""
        try:
            logger.info(f"🌍 Community care detected from @{detection.user_handle}")
            self.stats['community_amplifications'] += 1
            
            # Amplify community care content
            await self._like_post(detection.post_uri)
            
        except Exception as e:
            logger.error(f"Community care amplification failed: {str(e)}")
    
    async def _post_gentle_reply(self, detection: CareSignalDetection, message: str):
        """Post a gentle reply to a user in need"""
        try:
            # Create reply with trauma-informed approach
            reply_data = {
                'text': message,
                'reply': {
                    'root': detection.post_uri,
                    'parent': detection.post_uri
                }
            }
            
            response = await self.at_client.create_post(
                reply_data, 
                priority=RequestPriority.CARE_DELIVERY
            )
            
            return response.success
            
        except Exception as e:
            logger.error(f"Failed to post gentle reply: {str(e)}")
            return False
    
    async def _amplify_healing_content(self, detection: CareSignalDetection, amplification):
        """Amplify healing content with likes and gentle boosts"""
        try:
            # Like the healing post
            await self._like_post(detection.post_uri)
            
            # Optionally repost with gentle affirmation
            if amplification.should_repost:
                repost_data = {
                    'text': amplification.repost_message,
                    'embed': {
                        'record': detection.post_uri
                    }
                }
                
                await self.at_client.create_post(
                    repost_data,
                    priority=RequestPriority.CARE_DELIVERY
                )
            
        except Exception as e:
            logger.error(f"Failed to amplify healing content: {str(e)}")
    
    async def _like_post(self, post_uri: str):
        """Like a post with healing intention"""
        try:
            like_data = {
                'subject': post_uri,
                'createdAt': datetime.utcnow().isoformat() + 'Z'
            }
            
            response = await self.at_client.create_like(
                like_data,
                priority=RequestPriority.CARE_DELIVERY
            )
            
            return response.success
            
        except Exception as e:
            logger.error(f"Failed to like post: {str(e)}")
            return False
    
    async def _run_care_feeds(self):
        """Run daily care feeds with gentle timing"""
        while self.running:
            try:
                current_hour = datetime.utcnow().hour
                
                # Morning gentle reminders (8-10 AM UTC)
                if current_hour in [8, 9]:
                    await self.gentle_reminders.post_daily_reminder()
                    await asyncio.sleep(3600)  # Wait an hour
                
                # Evening community wisdom (7-9 PM UTC)  
                elif current_hour in [19, 20]:
                    await self.community_wisdom.post_wisdom_share()
                    await asyncio.sleep(3600)  # Wait an hour
                
                else:
                    # Check every 30 minutes during other hours
                    await asyncio.sleep(1800)
                
            except Exception as e:
                logger.error(f"Care feeds error: {str(e)}")
                await asyncio.sleep(1800)
    
    async def _health_monitor(self):
        """Monitor bot health and wellbeing metrics"""
        while self.running:
            try:
                # Log health stats every hour
                await self._log_health_stats()
                
                # Health check
                health_status = await self._perform_health_check()
                
                if not health_status['healthy']:
                    logger.warning(f"🔧 Health issues detected: {health_status['issues']}")
                
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                logger.error(f"Health monitoring error: {str(e)}")
                await asyncio.sleep(1800)
    
    async def _log_health_stats(self):
        """Log current health and impact statistics"""
        try:
            uptime = datetime.utcnow() - self.stats['uptime_start'] if self.stats['uptime_start'] else timedelta(0)
            
            logger.info("🦋 Monarch Bot Health Report:")
            logger.info(f"   💙 Uptime: {uptime}")
            logger.info(f"   🔍 Care signals detected: {self.stats['care_signals_detected']}")
            logger.info(f"   🤝 Gentle interventions: {self.stats['gentle_interventions']}")
            logger.info(f"   🚨 Crisis interventions: {self.stats['crisis_interventions']}")
            logger.info(f"   ✨ Community amplifications: {self.stats['community_amplifications']}")
            logger.info(f"   🌱 Healing moments created: {self.stats['healing_moments_created']}")
            logger.info(f"   👥 Unique users supported: {len(self.stats['users_supported'])}")
            
        except Exception as e:
            logger.error(f"Failed to log health stats: {str(e)}")
    
    async def _perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        try:
            health_status = {
                'healthy': True,
                'issues': [],
                'timestamp': datetime.utcnow()
            }
            
            # Check AT Protocol connection
            if not await self.at_client.is_connected():
                health_status['healthy'] = False
                health_status['issues'].append('AT Protocol connection lost')
            
            # Check monitoring status
            monitor_status = self.monitor.get_monitoring_status()
            if not monitor_status['monitoring_active']:
                health_status['healthy'] = False
                health_status['issues'].append('Monitoring not active')
            
            return health_status
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {'healthy': False, 'issues': ['Health check failed'], 'timestamp': datetime.utcnow()}
    
    async def _post_gentle_introduction(self):
        """Post a gentle introduction when starting up"""
        try:
            intro_message = """🦋 Monarch is gently awakening to spread care across Bluesky.

I'm here to offer support, share resources, and amplify healing in our community. 

If you're struggling, you're not alone. If you're healing, your journey inspires others.

Together, we're proving that technology can serve love. 💙

#HeartProtocol #CaringAlgorithms #MentalHealthSupport"""
            
            post_data = {'text': intro_message}
            
            response = await self.at_client.create_post(
                post_data,
                priority=RequestPriority.CARE_DELIVERY
            )
            
            if response.success:
                logger.info("🦋 Gentle introduction posted to Bluesky")
            else:
                logger.warning("⚠️ Failed to post introduction")
                
        except Exception as e:
            logger.error(f"Failed to post introduction: {str(e)}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"🛑 Received signal {signum}, initiating graceful shutdown...")
        asyncio.create_task(self.shutdown())
    
    async def shutdown(self):
        """Gracefully shutdown Monarch Bot"""
        try:
            logger.info("🦋 Monarch Bot shutting down gracefully...")
            self.running = False
            
            # Post gentle goodbye
            await self._post_gentle_goodbye()
            
            # Stop monitoring
            if self.monitor:
                await self.monitor.stop_monitoring()
                logger.info("✅ Monitoring stopped")
            
            # Disconnect from AT Protocol
            if self.at_client:
                await self.at_client.disconnect()
                logger.info("✅ Disconnected from Bluesky")
            
            # Final stats
            await self._log_health_stats()
            
            logger.info("🌟 Monarch Bot shutdown complete. Until we heal again. 💙")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")
    
    async def _post_gentle_goodbye(self):
        """Post a gentle goodbye when shutting down"""
        try:
            goodbye_message = """🦋 Monarch is taking a rest to recharge for more healing.

The care continues even when I'm away - through your kindness, your support for others, and your own healing journey.

Technology that serves love never truly sleeps. 💙

#HeartProtocol #RestAndRecharge"""
            
            post_data = {'text': goodbye_message}
            
            await self.at_client.create_post(
                post_data,
                priority=RequestPriority.CARE_DELIVERY
            )
            
        except Exception as e:
            logger.error(f"Failed to post goodbye: {str(e)}")


async def main():
    """Main entry point for Monarch Bot"""
    try:
        # Load configuration from environment
        config = {
            'BLUESKY_HANDLE': os.getenv('BLUESKY_HANDLE', 'monarch.bsky.social'),
            'BLUESKY_APP_PASSWORD': os.getenv('BLUESKY_APP_PASSWORD', ''),
            'ENVIRONMENT': os.getenv('ENVIRONMENT', 'development'),
            'DEBUG': os.getenv('DEBUG', 'false').lower() == 'true',
            'LOG_LEVEL': os.getenv('LOG_LEVEL', 'INFO'),
        }
        
        # Validate required configuration
        if not config['BLUESKY_APP_PASSWORD']:
            logger.error("❌ BLUESKY_APP_PASSWORD environment variable required")
            return
        
        # Create and start Monarch Bot
        monarch = MonarchBot(config)
        await monarch.start()
        
    except KeyboardInterrupt:
        logger.info("🛑 Keyboard interrupt received")
    except Exception as e:
        logger.error(f"💔 Monarch Bot failed: {str(e)}")
    finally:
        logger.info("🦋 Monarch Bot process ending")


if __name__ == "__main__":
    asyncio.run(main())