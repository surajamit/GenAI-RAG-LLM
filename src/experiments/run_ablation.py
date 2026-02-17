from src.ablation.ablation_engine import (
    retrieval_ablation,
    ats_agent_ablation,
    efficiency_ablation,
)


def run_ablation_suite():

    ret_df = retrieval_ablation()
    ats_df = ats_agent_ablation()
    eff_df = efficiency_ablation()

    ret_df.to_csv("ablation_retrieval.csv", index=False)
    ats_df.to_csv("ablation_ats_agents.csv", index=False)
    eff_df.to_csv("ablation_efficiency.csv", index=False)

    print("Ablation study completed.")


if __name__ == "__main__":
    run_ablation_suite()
