def exact_match(predictions, references):
    correct = sum(p.strip().lower() == r.strip().lower()
                  for p, r in zip(predictions, references))
    return correct / len(predictions)

def multi_hop_accuracy(tp, fn):
    return tp / (tp + fn + 1e-9)

def ats_alignment(matches, total):
    return matches / total

def execution_accuracy(correct, total):
    return correct / total

def precision_at_k(relevant, retrieved, k=10):
    retrieved_k = retrieved[:k]
    return len(set(retrieved_k) & set(relevant)) / k

def recall_at_k(relevant, retrieved, k=10):
    retrieved_k = retrieved[:k]
    return len(set(retrieved_k) & set(relevant)) / len(relevant)

def mean_reciprocal_rank(ranks):
    return np.mean([1.0 / r for r in ranks if r > 0])

def f1_score(p, r):
    return 2 * p * r / (p + r + 1e-8)

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))