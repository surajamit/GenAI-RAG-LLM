from typing import Dict, List
import numpy as np


class HTNNode:
    """
    Represents a task in Hierarchical Task Network.
    """

    def __init__(self, name: str, subtasks: List["HTNNode"] = None):
        self.name = name
        self.subtasks = subtasks or []


class HTNEvaluator:
    """
    Evaluates HTN execution quality.

    Metric:
        HTN Score = Σ w_i * success_i / N
    """

    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {}

    def evaluate_execution(self, task_success: Dict[str, int]) -> float:
        scores = []
        for task, success in task_success.items():
            w = self.weights.get(task, 1.0)
            scores.append(w * success)

        return float(np.mean(scores))

    def hierarchical_depth(self, root: HTNNode) -> int:
        if not root.subtasks:
            return 1
        return 1 + max(self.hierarchical_depth(s) for s in root.subtasks)
