def ats_score(resume_text, job_description):
    overlap = len(set(resume_text.lower().split()) &
                  set(job_description.lower().split()))
    return overlap / (len(job_description.split()) + 1e-9)
