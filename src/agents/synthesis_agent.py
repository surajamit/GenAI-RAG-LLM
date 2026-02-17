def synthesize_feedback(agent_outputs: dict) -> dict:
    """
    Final ATS decision fusion.
    """
    final_score = sum(agent_outputs.values()) / len(agent_outputs)
    return {
        "final_score": final_score,
        "decision": "SHORTLIST" if final_score > 0.7 else "REJECT",
    }
