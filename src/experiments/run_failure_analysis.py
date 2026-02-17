import numpy as np

from src.error_analysis.failure_detector import FailureDetector
from src.error_analysis.failure_miner import FailureMiner
from src.error_analysis.hard_case_miner import mine_hard_cases


def simulate_failure_pipeline(n=500):

    detector = FailureDetector()
    miner = FailureMiner()

    failures = []

    for _ in range(n):

        code = detector.detect_hallucination(
            support_score=np.random.rand()
        )

        failures.append(code)

    summary_df = miner.summarize_failures(failures)
    miner.export_failure_table(summary_df)

    # Hard case mining
    scores = np.random.rand(n)
    hard_df = mine_hard_cases(list(range(n)), scores)
    hard_df.to_csv("table_hard_cases.csv", index=False)

    print("Failure analysis completed.")


if __name__ == "__main__":
    simulate_failure_pipeline()
