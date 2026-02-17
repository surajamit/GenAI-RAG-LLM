from src.evaluation.results_tables import build_results_tables
from src.visualization.results_figure import plot_retrieval_comparison


def main():

    retrieval_df, ats_df, sql_df = build_results_tables()

    retrieval_df.to_csv("table_retrieval.csv", index=False)
    ats_df.to_csv("table_ats.csv", index=False)
    sql_df.to_csv("table_text2sql.csv", index=False)

    plot_retrieval_comparison()

    print("Results pipeline completed.")


if __name__ == "__main__":
    main()
