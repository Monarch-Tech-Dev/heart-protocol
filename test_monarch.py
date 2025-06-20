#!/usr/bin/env python3
"""
Test script for Monarch Bot Bluesky integration

This script tests the basic functionality of the Monarch Bot
without requiring actual Bluesky credentials.
"""

import asyncio
import os
import sys
import logging
from datetime import datetime

# Add the heart_protocol to path
sys.path.insert(0, '/home/kinglinux101/Heart-Protocol')

from heart_protocol.monarch_bot.main import MonarchBot

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_monarch_initialization():
    """Test Monarch Bot initialization"""
    try:
        logger.info("🧪 Testing Monarch Bot initialization...")
        
        # Test configuration
        test_config = {
            'BLUESKY_HANDLE': 'test.monarch.bsky.social',
            'BLUESKY_APP_PASSWORD': 'test-password-123',
            'ENVIRONMENT': 'test',
            'DEBUG': True,
            'LOG_LEVEL': 'INFO',
        }
        
        # Initialize Monarch Bot
        monarch = MonarchBot(test_config)
        
        logger.info("✅ Monarch Bot initialized successfully")
        
        # Test basic components
        assert monarch.config is not None
        assert monarch.care_algorithm is not None
        assert monarch.crisis_system is not None
        assert monarch.gentle_reminders is not None
        assert monarch.community_wisdom is not None
        
        logger.info("✅ All core components initialized")
        
        # Test stats initialization
        stats = monarch.stats
        assert 'care_signals_detected' in stats
        assert 'gentle_interventions' in stats
        assert 'crisis_interventions' in stats
        
        logger.info("✅ Stats tracking initialized")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Monarch Bot initialization failed: {str(e)}")
        return False


async def test_feed_systems():
    """Test feed system functionality"""
    try:
        logger.info("🧪 Testing feed systems...")
        
        from heart_protocol.feeds.gentle_reminders import GentleRemindersFeed
        from heart_protocol.feeds.community_wisdom import CommunityWisdomFeed
        from heart_protocol.infrastructure.bluesky_integration.at_protocol_client import (
            ATProtocolClient, ATProtocolConfig, AuthMethod
        )
        
        # Create mock AT client
        at_config = ATProtocolConfig(
            service_url="https://test.bsky.social",
            identifier="test.monarch.bsky.social",
            auth_method=AuthMethod.APP_PASSWORD,
            credentials={'password': 'test-password'}
        )
        
        at_client = ATProtocolClient(at_config)
        
        # Test gentle reminders feed
        gentle_feed = GentleRemindersFeed(at_client, {})
        
        # Test reminder selection
        reminder = await gentle_feed._select_contextual_reminder()
        assert reminder is not None
        assert 'text' in reminder
        assert 'category' in reminder
        
        logger.info("✅ Gentle reminders feed working")
        
        # Test community wisdom feed  
        wisdom_feed = CommunityWisdomFeed(at_client, {})
        
        # Test wisdom selection
        wisdom = await wisdom_feed._select_contextual_wisdom()
        assert wisdom is not None
        assert 'text' in wisdom
        assert 'category' in wisdom
        
        logger.info("✅ Community wisdom feed working")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Feed systems test failed: {str(e)}")
        return False


async def test_crisis_system():
    """Test crisis intervention system"""
    try:
        logger.info("🧪 Testing crisis intervention system...")
        
        from heart_protocol.core.crisis_intervention import CrisisInterventionSystem, CrisisLevel
        from heart_protocol.infrastructure.bluesky_integration.bluesky_monitor import (
            CareSignalDetection, CareSignal, SignalConfidence
        )
        
        # Initialize crisis system
        crisis_system = CrisisInterventionSystem({
            'CRISIS_INTERVENTION_TIMEOUT': 300,
            'HUMAN_HANDOFF_ENABLED': True
        })
        
        # Create mock crisis detection
        mock_detection = CareSignalDetection(
            signal_id="test_signal_123",
            user_handle="test_user",
            user_did="did:test:user",
            post_uri="at://test/post/123",
            post_content="I want to die and can't go on anymore",
            signal_type=CareSignal.IMMEDIATE_CRISIS,
            confidence_score=0.9,
            confidence_level=SignalConfidence.VERY_HIGH,
            detected_keywords=["want to die", "can't go on"],
            context_analysis={},
            emotional_intensity=0.9,
            urgency_level="immediate",
            cultural_considerations=[],
            privacy_sensitivity="very_high",
            detected_at=datetime.utcnow(),
            post_created_at=datetime.utcnow(),
            user_profile_context={},
            intervention_recommended=True,
            intervention_type="crisis_intervention",
            gentle_approach_required=True,
            trauma_informed_response=True
        )
        
        # Test crisis level assessment
        crisis_level = crisis_system._assess_crisis_level(mock_detection)
        assert crisis_level in [CrisisLevel.IMMEDIATE_DANGER, CrisisLevel.HIGH_RISK]
        
        logger.info("✅ Crisis level assessment working")
        
        # Test crisis response (without actual posting)
        # This would normally post to Bluesky, but we're just testing the logic
        message, resources = await crisis_system._create_immediate_response(mock_detection, crisis_level)
        assert message is not None
        assert len(resources) > 0
        assert "988" in message  # Crisis hotline number
        
        logger.info("✅ Crisis response generation working")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Crisis system test failed: {str(e)}")
        return False


async def test_monitoring_patterns():
    """Test care signal monitoring patterns"""
    try:
        logger.info("🧪 Testing monitoring patterns...")
        
        from heart_protocol.infrastructure.bluesky_integration.bluesky_monitor import (
            BlueSkyMonitor, CareSignal
        )
        from heart_protocol.infrastructure.bluesky_integration.at_protocol_client import (
            ATProtocolClient, ATProtocolConfig, AuthMethod
        )
        
        # Create mock AT client
        at_config = ATProtocolConfig(
            service_url="https://test.bsky.social",
            identifier="test.monarch.bsky.social", 
            auth_method=AuthMethod.APP_PASSWORD,
            credentials={'password': 'test-password'}
        )
        
        at_client = ATProtocolClient(at_config)
        monitor = BlueSkyMonitor(at_client, {})
        
        # Test pattern initialization
        patterns = monitor.monitoring_patterns
        assert CareSignal.IMMEDIATE_CRISIS in patterns
        assert CareSignal.SUPPORT_SEEKING in patterns
        assert CareSignal.HEALING_PROGRESS in patterns
        
        logger.info("✅ Monitoring patterns initialized")
        
        # Test crisis pattern detection
        crisis_pattern = patterns[CareSignal.IMMEDIATE_CRISIS]
        test_post = "I want to kill myself and end it all"
        
        detection = await monitor._apply_monitoring_pattern(
            crisis_pattern, test_post, {'post': {'record': {}, 'author': {}}}
        )
        
        assert detection is not None
        assert detection.confidence_score > 0.5
        assert len(detection.detected_keywords) > 0
        
        logger.info("✅ Crisis pattern detection working")
        
        # Test support seeking pattern
        support_pattern = patterns[CareSignal.SUPPORT_SEEKING]
        test_post = "I'm struggling with depression and need help finding a therapist"
        
        detection = await monitor._apply_monitoring_pattern(
            support_pattern, test_post, {'post': {'record': {}, 'author': {}}}
        )
        
        assert detection is not None
        assert detection.signal_type == CareSignal.SUPPORT_SEEKING
        
        logger.info("✅ Support seeking pattern working")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Monitoring patterns test failed: {str(e)}")
        return False


async def main():
    """Run all tests"""
    logger.info("🧪 Starting Monarch Bot integration tests...")
    logger.info("💙 Testing Heart Protocol's caring algorithms...")
    
    tests = [
        ("Monarch Bot Initialization", test_monarch_initialization),
        ("Feed Systems", test_feed_systems),
        ("Crisis Intervention", test_crisis_system), 
        ("Monitoring Patterns", test_monitoring_patterns)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        logger.info(f"\n🔍 Running {test_name} test...")
        try:
            result = await test_func()
            if result:
                logger.info(f"✅ {test_name} test PASSED")
                passed += 1
            else:
                logger.error(f"❌ {test_name} test FAILED")
                failed += 1
        except Exception as e:
            logger.error(f"💥 {test_name} test CRASHED: {str(e)}")
            failed += 1
    
    logger.info(f"\n🦋 Test Results:")
    logger.info(f"   ✅ Passed: {passed}")
    logger.info(f"   ❌ Failed: {failed}")
    logger.info(f"   📊 Success Rate: {passed}/{passed+failed} ({100*passed/(passed+failed) if passed+failed > 0 else 0:.1f}%)")
    
    if failed == 0:
        logger.info("🌟 All tests passed! Monarch Bot is ready to spread healing on Bluesky! 💙")
        logger.info("\n🎯 Next steps:")
        logger.info("   1. Create @monarch.bsky.social account on Bluesky")
        logger.info("   2. Generate app password in Bluesky settings")
        logger.info("   3. Set BLUESKY_APP_PASSWORD environment variable")
        logger.info("   4. Run: python -m heart_protocol.monarch_bot.main")
    else:
        logger.error("💔 Some tests failed. Please review the errors above.")
        
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())