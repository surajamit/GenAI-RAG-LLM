import time
from collections import deque

class DynamicBatcher:
    """
    Combines concurrent requests without latency explosion.
    """

    def __init__(self, max_batch_size=32, timeout_ms=10):
        self.queue = deque()
        self.max_batch_size = max_batch_size
        self.timeout_ms = timeout_ms

    def add_request(self, request):
        self.queue.append((request, time.time()))

    def get_batch(self):
        batch = []

        while len(batch) < self.max_batch_size and self.queue:
            req, t = self.queue.popleft()
            batch.append(req)

        return batch
