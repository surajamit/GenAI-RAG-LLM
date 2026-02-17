# NER Evaluation (CoNLL-2003)

from seqeval.metrics import f1_score as ner_f1
from src.data.conll_loader import load_conll2003

def evaluate_ner(model):

    dataset = load_conll2003("validation")

    y_true, y_pred = [], []

    for sample in dataset:
        pred = model.predict(sample["tokens"])
        y_true.append(sample["ner_tags"])
        y_pred.append(pred)

    return {"F1": ner_f1(y_true, y_pred)}
