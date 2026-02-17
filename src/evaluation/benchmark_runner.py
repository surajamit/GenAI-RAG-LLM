BENCHMARKS = {
    "qa": "MS_MARCO",
    "ner": "CONLL2003",
    "t2s": "SPIDER",
    "fact": "FEVER"
}

def run_full_benchmark(model):
    results = {}

    for task, dataset in BENCHMARKS.items():
        score = evaluate_task(model, task, dataset)
        results[task] = score

    return results
