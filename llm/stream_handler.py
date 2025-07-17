from langchain.callbacks.base import BaseCallbackHandler
from asyncio import Queue, get_running_loop, run_coroutine_threadsafe

class SSEHandler(BaseCallbackHandler):
    def __init__(self, queue: Queue, loop=None):
        self.queue = queue
        self.loop = loop or get_running_loop()

    def on_llm_new_token(self, token: str, **kwargs):
        # Schedule coroutine safely from non-async context
        run_coroutine_threadsafe(self.queue.put(token), self.loop)

    def on_llm_end(self, response, **kwargs):
        run_coroutine_threadsafe(self.queue.put("[END]"), self.loop)
