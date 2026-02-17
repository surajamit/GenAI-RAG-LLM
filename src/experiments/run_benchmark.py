# experiments/run_benchmark.py
from src.utils.reproducibility import set_global_seed
from src.evaluation.results_table import build_results_table
from evaluation.retrieval_metrics import precision_at_k
from utils.helpers import mean_reciprocal_rank

def run():
    set_global_seed(42)
    df = build_results_table()
    print(df)
    df.to_csv("results.csv", index=False)

if __name__ == "__main__":
    run()


"""
End-to-end experimental driver
"""

def run_demo():
    relevant = {1, 2, 3}
    retrieved = [3, 4, 2, 8, 1]

    p10 = precision_at_k(relevant, retrieved, k=5)
    mrr = mean_reciprocal_rank([1, 3, 2])

    print("Precision@10:", p10)
    print("MRR:", mrr)


if __name__ == "__main__":
    run_demo()
