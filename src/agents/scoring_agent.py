def score_resume(features: dict) -> float:
    """
    Weighted ATS scoring.

    Formula:
        S = Σ w_i f_i
    """
    weights = {
        "skills": 0.35,
        "experience": 0.30,
        "education": 0.20,
        "format": 0.15,
    }

    return sum(weights[k] * features.get(k, 0) for k in weights)
