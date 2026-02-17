from src.analysis.model_comparison import build_model_comparison_table
from src.robustness.load_tester import simulate_load_test


def export_all_tables():

    model_df = build_model_comparison_table()
    load_df = simulate_load_test()

    model_df.to_csv("table_model_comparison.csv", index=False)
    load_df.to_csv("table_load_test.csv", index=False)

    print("Model comparison tables exported.")


if __name__ == "__main__":
    export_all_tables()
