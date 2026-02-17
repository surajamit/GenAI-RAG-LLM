class SQLFineTuner:
    """
    Fine-tuning pipeline for GenAI-SQL coder.
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def train_step(self, nl_query, sql_query):
        inputs = self.tokenizer(nl_query, return_tensors="pt")
        labels = self.tokenizer(sql_query, return_tensors="pt").input_ids

        outputs = self.model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        return loss.item()


def execution_accuracy(pred_sql, gold_sql, db_engine):
    """
    Exact execution match metric.
    """

    try:
        pred_res = db_engine.execute(pred_sql).fetchall()
        gold_res = db_engine.execute(gold_sql).fetchall()
        return int(pred_res == gold_res)
    except Exception:
        return 0


