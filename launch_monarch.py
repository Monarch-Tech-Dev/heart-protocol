#!/usr/bin/env python3
"""
Simple launcher for Monarch Bot that avoids import conflicts
"""

import os
import sys
import asyncio
import logging
from datetime import datetime

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Simple Monarch Bot implementation
class SimpleMonarchBot:
    """Simplified Monarch Bot for initial launch"""
    
    def __init__(self):
        self.config = {
            'BLUESKY_HANDLE': os.getenv('BLUESKY_HANDLE', 'monarchbot.bsky.social'),
            'BLUESKY_APP_PASSWORD': os.getenv('BLUESKY_APP_PASSWORD', ''),
            'ENVIRONMENT': os.getenv('ENVIRONMENT', 'production')
        }
        
        self.running = False
        
    async def start(self):
        """Start the simple bot"""
        try:
            logger.info("🦋 Monarch Bot starting up...")
            logger.info(f"💙 Handle: @{self.config['BLUESKY_HANDLE']}")
            logger.info(f"🌍 Environment: {self.config['ENVIRONMENT']}")
            
            if not self.config['BLUESKY_APP_PASSWORD']:
                logger.error("❌ BLUESKY_APP_PASSWORD not found in environment")
                return
            
            logger.info("🔑 App password loaded successfully")
            
            # Test AT Protocol connection
            await self.test_bluesky_connection()
            
            # Post introduction
            await self.post_introduction()
            
            logger.info("🌟 Monarch Bot launched successfully!")
            logger.info("💫 Spreading healing across Bluesky...")
            
            # Keep running
            self.running = True
            while self.running:
                await asyncio.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            logger.info("🛑 Graceful shutdown initiated...")
            await self.shutdown()
        except Exception as e:
            logger.error(f"💔 Monarch Bot failed: {str(e)}")
    
    async def test_bluesky_connection(self):
        """Test connection to Bluesky"""
        try:
            import aiohttp
            
            # Test AT Protocol endpoint
            async with aiohttp.ClientSession() as session:
                auth_url = "https://bsky.social/xrpc/com.atproto.server.createSession"
                
                auth_data = {
                    "identifier": self.config['BLUESKY_HANDLE'],
                    "password": self.config['BLUESKY_APP_PASSWORD']
                }
                
                async with session.post(auth_url, json=auth_data) as response:
                    if response.status == 200:
                        auth_result = await response.json()
                        self.access_token = auth_result.get('accessJwt')
                        self.did = auth_result.get('did')
                        logger.info("✅ Successfully authenticated with Bluesky")
                        logger.info(f"🆔 DID: {self.did}")
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ Authentication failed: {response.status} - {error_text}")
                        return False
                        
        except Exception as e:
            logger.error(f"❌ Connection test failed: {str(e)}")
            return False
    
    async def post_introduction(self):
        """Post introduction to Bluesky"""
        try:
            if not hasattr(self, 'access_token'):
                logger.error("❌ Not authenticated - cannot post")
                return
                
            import aiohttp
            
            intro_text = """🦋 Monarch Bot awakening to spread care across Bluesky.

Heart Protocol's gentle AI companion offering support, resources, and healing amplification.

Together, we're proving technology can serve love. 💙

#HeartProtocol #CaringAlgorithms"""

            post_data = {
                "repo": self.did,
                "collection": "app.bsky.feed.post",
                "record": {
                    "text": intro_text,
                    "createdAt": datetime.utcnow().isoformat() + "Z",
                    "$type": "app.bsky.feed.post"
                }
            }
            
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.access_token}",
                    "Content-Type": "application/json"
                }
                
                create_url = "https://bsky.social/xrpc/com.atproto.repo.createRecord"
                
                async with session.post(create_url, json=post_data, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info("🎉 Introduction posted successfully!")
                        logger.info(f"📍 Post URI: {result.get('uri')}")
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ Failed to post introduction: {response.status} - {error_text}")
                        
        except Exception as e:
            logger.error(f"❌ Failed to post introduction: {str(e)}")
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("🦋 Monarch Bot shutting down gracefully...")
        self.running = False
        logger.info("💙 Until we heal again...")


async def main():
    """Main entry point"""
    bot = SimpleMonarchBot()
    await bot.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Keyboard interrupt received")
    except Exception as e:
        logger.error(f"💔 Launch failed: {str(e)}")