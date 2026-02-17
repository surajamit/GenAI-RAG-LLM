import matplotlib.pyplot as plt
from src.evaluation.results_tables import build_results_tables


def plot_retrieval_comparison():

    retrieval_df, _, _ = build_results_tables()

    plt.figure(figsize=(5, 4))

    methods = retrieval_df["Method"]
    precision = retrieval_df["Precision@10"]

    plt.bar(methods, precision)
    plt.ylabel("Precision@10")
    plt.title("GraphRAG vs Vector RAG")

    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig("retrieval_comparison.png", dpi=300)
    plt.close()
