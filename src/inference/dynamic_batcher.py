"""
Dynamic batching engine for GPU utilization optimization.
"""

import time
import queue
import threading


class DynamicBatcher:

    def __init__(self, model, max_batch_size=32, timeout_ms=20):
        self.model = model
        self.max_batch_size = max_batch_size
        self.timeout_ms = timeout_ms

        self.request_queue = queue.Queue()
        self.running = True

        self.worker = threading.Thread(target=self._batch_worker)
        self.worker.daemon = True
        self.worker.start()

    def submit(self, request):

        response_event = threading.Event()
        container = {"request": request, "event": response_event, "output": None}

        self.request_queue.put(container)
        response_event.wait()

        return container["output"]

    def _batch_worker(self):

        while self.running:

            batch = []
            start_time = time.time()

            while len(batch) < self.max_batch_size:
                try:
                    item = self.request_queue.get(timeout=self.timeout_ms / 1000)
                    batch.append(item)
                except queue.Empty:
                    break

            if not batch:
                continue

            inputs = [x["request"] for x in batch]
            outputs = self.model.generate_batch(inputs)

            for container, out in zip(batch, outputs):
                container["output"] = out
                container["event"].set()
