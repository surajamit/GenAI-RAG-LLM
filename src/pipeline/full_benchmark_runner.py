from src.evaluation.evaluate_retrieval import evaluate_retrieval
from src.evaluation.evaluate_ner import evaluate_ner
from src.evaluation.evaluate_text2sql import evaluate_text2sql
from src.evaluation.evaluate_factcheck import evaluate_factcheck

def run_full_benchmark(system):

    results = {}

    results["QA Retrieval"] = evaluate_retrieval(system)
    results["NER"] = evaluate_ner(system.ner_model)
    results["Text2SQL"] = evaluate_text2sql(system.sql_model, system.db)
    results["Fact Verification"] = evaluate_factcheck(system.fact_model)

    return results
