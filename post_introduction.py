#!/usr/bin/env python3
"""
Post the introduction for Monarch Bot
"""

import os
import asyncio
import logging
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def post_introduction():
    """Post introduction to Bluesky"""
    try:
        import aiohttp
        
        # Authenticate first
        auth_url = "https://bsky.social/xrpc/com.atproto.server.createSession"
        auth_data = {
            "identifier": os.getenv('BLUESKY_HANDLE'),
            "password": os.getenv('BLUESKY_APP_PASSWORD')
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(auth_url, json=auth_data) as response:
                if response.status != 200:
                    logger.error(f"Auth failed: {response.status}")
                    return
                
                auth_result = await response.json()
                access_token = auth_result.get('accessJwt')
                did = auth_result.get('did')
                
                logger.info("✅ Authenticated successfully")
                
                # Post introduction
                intro_text = """🦋 Monarch Bot awakening to spread care across Bluesky.

Heart Protocol's gentle AI companion offering support, resources, and healing amplification.

Together, we're proving technology can serve love. 💙

#HeartProtocol #CaringAlgorithms"""

                post_data = {
                    "repo": did,
                    "collection": "app.bsky.feed.post",
                    "record": {
                        "text": intro_text,
                        "createdAt": datetime.now().isoformat() + "Z",
                        "$type": "app.bsky.feed.post"
                    }
                }
                
                headers = {
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json"
                }
                
                create_url = "https://bsky.social/xrpc/com.atproto.repo.createRecord"
                
                async with session.post(create_url, json=post_data, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info("🎉 Introduction posted successfully!")
                        logger.info(f"📍 Post URI: {result.get('uri')}")
                        print(f"\n🌟 Success! Check @monarchbot.bsky.social on Bluesky!")
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ Failed to post: {response.status} - {error_text}")
                        
    except Exception as e:
        logger.error(f"❌ Failed: {str(e)}")

if __name__ == "__main__":
    asyncio.run(post_introduction())