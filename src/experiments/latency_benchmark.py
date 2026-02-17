import time
import numpy as np
from src.llm.custom_llm import CustomLLM

def measure_latency(n_runs=50):
    model = CustomLLM()

    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = model.generate("Test query for latency")
        end = time.perf_counter()
        latencies.append(end - start)

    return {
        "mean_latency_ms": np.mean(latencies) * 1000,
        "std_latency_ms": np.std(latencies) * 1000,
        "p95_latency_ms": np.percentile(latencies, 95) * 1000,
    }


if __name__ == "__main__":
    stats = measure_latency()
    print(stats)
