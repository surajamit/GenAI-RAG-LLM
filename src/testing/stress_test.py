"""
High-load stress tester
Used for scalability tables.
"""

import asyncio
import aiohttp
import time


async def fire_request(session, url):

    start = time.time()

    async with session.post(url, json={"query": "GraphRAG"}) as resp:
        await resp.text()

    return time.time() - start


async def run_stress(url, n=500):

    async with aiohttp.ClientSession() as session:
        tasks = [fire_request(session, url) for _ in range(n)]
        latencies = await asyncio.gather(*tasks)

    return latencies
