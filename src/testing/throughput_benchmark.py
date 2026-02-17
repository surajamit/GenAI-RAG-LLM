import numpy as np


def compute_throughput(latencies):

    latencies = np.array(latencies)

    total_time = latencies.sum()
    rps = len(latencies) / total_time

    return {
        "Requests/sec": float(rps),
        "P50": float(np.percentile(latencies, 50)),
        "P95": float(np.percentile(latencies, 95)),
        "Mean Latency": float(latencies.mean())
    }
