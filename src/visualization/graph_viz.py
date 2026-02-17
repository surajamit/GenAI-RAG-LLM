import networkx as nx
import matplotlib.pyplot as plt

def visualize_graph(G, save_path="knowledge_graph.png"):
    plt.figure(figsize=(6, 5))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=800,
        font_size=8
    )
    plt.title("GraphRAG Knowledge Graph")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
