TEMPERATURES = [0.01, 0.03, 0.05, 0.07]

def run_temperature_study(model, dataloader):
    results = []

    for temp in TEMPERATURES:
        precision, recall, mrr = evaluate_retrieval(model, temp)

        results.append({
            "temperature": temp,
            "precision": precision,
            "recall": recall,
            "mrr": mrr
        })

    return results
