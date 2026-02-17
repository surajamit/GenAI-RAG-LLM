import networkx as nx

def build_graph(entities):
    G = nx.Graph()
    for i in range(len(entities)-1):
        G.add_edge(entities[i][0], entities[i+1][0])
    return G
