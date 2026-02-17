import numpy as np


class FailureDetector:
    """
    Rule-based failure mining aligned with manuscript tasks.
    """

    def __init__(self):
        self.error_codes = {
            "retrieval_miss": "ER1",
            "graph_failure": "ER2",
            "hallucination": "ER3",
            "sql_error": "ER4",
            "agent_conflict": "ER5",
            "format_error": "ER6",
        }

    # --------------------------------------------------------
    # Retrieval failure
    # --------------------------------------------------------
    def detect_retrieval_failure(self, relevant_docs, retrieved_docs):
        if not set(relevant_docs).intersection(set(retrieved_docs)):
            return "ER1"
        return None

    # --------------------------------------------------------
    # Graph reasoning failure
    # --------------------------------------------------------
    def detect_graph_failure(self, path_length, required_hops):
        if path_length < required_hops:
            return "ER2"
        return None

    # --------------------------------------------------------
    # Hallucination detection (lightweight proxy)
    # --------------------------------------------------------
    def detect_hallucination(self, support_score, threshold=0.5):
        if support_score < threshold:
            return "ER3"
        return None

    # --------------------------------------------------------
    # SQL validation failure
    # --------------------------------------------------------
    def detect_sql_error(self, is_valid_sql):
        if not is_valid_sql:
            return "ER4"
        return None

    # --------------------------------------------------------
    # Multi-agent conflict
    # --------------------------------------------------------
    def detect_agent_conflict(self, agent_scores, tol=15):
        if np.std(agent_scores) > tol:
            return "ER5"
        return None

    # --------------------------------------------------------
    # ATS formatting failure
    # --------------------------------------------------------
    def detect_format_error(self, format_score, threshold=0.6):
        if format_score < threshold:
            return "ER6"
        return None
