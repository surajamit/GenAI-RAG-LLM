import networkx as nx
import community as community_louvain


def detect_communities(edge_list):
    """
    Louvain modularity optimization.

    Returns:
        node -> community_id
    """

    G = nx.Graph()
    G.add_edges_from(edge_list)

    partition = community_louvain.best_partition(G)

    return partition
