#Text-to-SQL Evaluation (Spider)

from src.data.spider_loader import load_spider

def execution_accuracy(pred_sql, gold_sql, db):
    try:
        return int(db.execute(pred_sql) == db.execute(gold_sql))
    except Exception:
        return 0


def evaluate_text2sql(model, db):

    dataset = load_spider("validation")

    scores = []

    for sample in dataset:
        pred_sql = model.generate(sample["question"])
        acc = execution_accuracy(pred_sql, sample["sql"], db)
        scores.append(acc)

    return {"Execution Accuracy": sum(scores)/len(scores)}
