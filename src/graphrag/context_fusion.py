import torch


def fuse_context(dense_ctx, graph_ctx, alpha=0.6):
    """
    Hybrid fusion:

    C = α C_dense + (1-α) C_graph
    """

    return alpha * dense_ctx + (1 - alpha) * graph_ctx
