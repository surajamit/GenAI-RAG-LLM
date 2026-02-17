import networkx as nx


def build_knowledge_graph(chunks):
    """
    Builds co-occurrence graph.
    """
    G = nx.Graph()

    for chunk in chunks:
        ents = list(extract_entities(chunk))
        for i in range(len(ents)):
            for j in range(i + 1, len(ents)):
                G.add_edge(ents[i], ents[j])

    return G
