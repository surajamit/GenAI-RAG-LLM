from sklearn.metrics import f1_score


def exact_match(pred: str, truth: str) -> int:
    return int(pred.strip() == truth.strip())


def token_f1(pred_tokens, truth_tokens):
    return f1_score(truth_tokens, pred_tokens, average="macro")
