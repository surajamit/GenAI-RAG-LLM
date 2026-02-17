def graph_context_score(query_entities, graph):
    score = 0
    for ent in query_entities:
        if ent in graph.nodes:
            score += 1
    return score / (len(query_entities) + 1e-9)


def fused_score(vector_score, graph_score, alpha=0.6):
    beta = 1 - alpha
    return alpha * vector_score + beta * graph_score