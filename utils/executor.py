import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor()

async def run_in_executor(func):
    return await asyncio.get_event_loop().run_in_executor(executor, func)
