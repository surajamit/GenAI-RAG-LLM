def job_match_score(resume_keywords, job_keywords):
    inter = len(set(resume_keywords) & set(job_keywords))
    return inter / (len(job_keywords) + 1e-9)
