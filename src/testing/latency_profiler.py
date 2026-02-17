import numpy as np


def compute_latency_stats(latencies):

    latencies = np.array(latencies)

    return {
        "P50": float(np.percentile(latencies, 50)),
        "P95": float(np.percentile(latencies, 95)),
        "mean": float(np.mean(latencies)),
        "std": float(np.std(latencies))
    }
