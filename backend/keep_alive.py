import asyncio
import aiohttp
import os
from datetime import datetime

async def ping_self():
    """Ping own health endpoint to prevent sleep"""
    port = os.environ.get("PORT", "10000")
    url = f"http://localhost:{port}/health"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    print(f"✅ Self-ping successful at {datetime.now()}")
                else:
                    print(f"⚠️ Self-ping failed with status {response.status}")
    except Exception as e:
        print(f"❌ Self-ping error: {e}")

async def keep_alive_loop():
    """Keep the service alive by pinging every 10 minutes"""
    while True:
        await asyncio.sleep(600)  # 10 minutes
        await ping_self()

if __name__ == "__main__":
    asyncio.run(keep_alive_loop())