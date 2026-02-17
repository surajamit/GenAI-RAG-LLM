"""
Unified evaluation runner for all tasks.
"""

import time


def run_inference_with_latency(model, inputs):

    start = time.time()
    outputs = model(inputs)
    latency = time.time() - start

    return outputs, latency


def test_model_on_dataset(model, dataloader):

    latencies = []
    predictions = []

    model.eval()

    for batch in dataloader:

        outputs, latency = run_inference_with_latency(model, batch)

        predictions.append(outputs)
        latencies.append(latency)

    return {
        "predictions": predictions,
        "avg_latency": sum(latencies) / len(latencies),
        "p95_latency": sorted(latencies)[int(0.95 * len(latencies))]
    }
