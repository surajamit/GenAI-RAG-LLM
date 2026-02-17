import pandas as pd
from collections import Counter


class FailureMiner:

    def summarize_failures(self, failure_list):
        counter = Counter([f for f in failure_list if f is not None])

        total = sum(counter.values())

        rows = []
        for k, v in counter.items():
            rows.append({
                "Error Code": k,
                "Count": v,
                "Percentage (%)": round(100 * v / total, 2)
            })

        return pd.DataFrame(rows)

    def export_failure_table(self, df, path="table_failure_analysis.csv"):
        df.to_csv(path, index=False)
