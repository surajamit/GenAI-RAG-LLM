"""
Production inference orchestrator
Integrates:
- dynamic batching
- optimized inference
- latency tracking
"""

import time
from .dynamic_batcher import DynamicBatcher
from .triton_wrapper import OptimizedInferenceEngine


class GenAIInferenceServer:

    def __init__(self, model):

        self.engine = OptimizedInferenceEngine(
            model,
            use_tensorrt=True
        )

        self.batcher = DynamicBatcher(
            self.engine,
            max_batch_size=32,
            timeout_ms=25
        )

        self.latencies = []

    def generate(self, request):

        start = time.time()
        output = self.batcher.submit(request)
        latency = time.time() - start

        self.latencies.append(latency)
        return output

    def latency_report(self):

        import numpy as np

        arr = np.array(self.latencies)

        return {
            "P50": float(np.percentile(arr, 50)),
            "P95": float(np.percentile(arr, 95)),
            "Mean": float(arr.mean())
        }
