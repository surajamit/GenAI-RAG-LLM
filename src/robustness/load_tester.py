import numpy as np
import pandas as pd


def simulate_load_test():

    """
    Synthetic but structurally correct load model.
    Replace with Locust/JMeter traces in production.
    """

    requests = np.array([50, 100, 250, 500, 1000])

    base_latency = 0.9
    latency = base_latency * (1 + (requests / 1200))
    p95_latency = latency * 1.35
    cpu_util = np.minimum(85, 20 + requests * 0.05)

    df = pd.DataFrame({
        "Requests/sec": requests,
        "P50 Latency (s)": latency.round(3),
        "P95 Latency (s)": p95_latency.round(3),
        "CPU Util (%)": cpu_util.round(1),
    })

    return df
