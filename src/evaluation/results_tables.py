import pandas as pd
from experiments.run_full_results import (
    simulate_retrieval,
    simulate_ats,
    simulate_text2sql,
)


def build_results_tables():

    # Retrieval
    ret = simulate_retrieval()

    retrieval_df = pd.DataFrame([
        {
            "Method": "Vector RAG",
            "Precision@10": ret["vector"][0],
            "Recall@10": ret["vector"][1],
            "MRR": ret["vector"][2],
            "Hit@10": ret["vector"][3],
        },
        {
            "Method": "GraphRAG (Proposed)",
            "Precision@10": ret["graph"][0],
            "Recall@10": ret["graph"][1],
            "MRR": ret["graph"][2],
            "Hit@10": ret["graph"][3],
        },
    ])

    # ATS
    mae, corr = simulate_ats()

    ats_df = pd.DataFrame([
        {
            "Model": "Agentic ATS",
            "MAE": mae,
            "Correlation": corr,
        }
    ])

    # Text-to-SQL
    acc, err = simulate_text2sql()

    sql_df = pd.DataFrame([
        {
            "Module": "Schema-aware Text2SQL",
            "Execution Accuracy": acc,
            "Syntax Error Rate": err,
        }
    ])

    return retrieval_df, ats_df, sql_df
