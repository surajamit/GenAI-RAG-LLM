def build_prompt(context, graph_context, query, history=None):
    return f"""
Context:
{context}

Graph Context:
{graph_context}

Query:
{query}

Answer clearly and cite sources.
"""
