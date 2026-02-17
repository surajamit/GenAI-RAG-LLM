#Fact Verification (FEVER)

from src.data.fever_loader import load_fever

def evaluate_factcheck(model):

    dataset = load_fever("validation")

    correct = 0

    for sample in dataset:
        pred = model.verify(sample["claim"])
        if pred == sample["label"]:
            correct += 1

    return {"Accuracy": correct / len(dataset)}
