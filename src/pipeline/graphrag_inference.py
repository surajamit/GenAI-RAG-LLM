def run_graphrag_inference(
    query_emb,
    dense_ctx,
    graph_ctx,
    llm,
    temperature=0.7,
):

    # Hybrid fusion
    fused_ctx = fuse_context(dense_ctx, graph_ctx)

    # LLM forward
    logits = llm(query_emb, fused_ctx)

    # Decode
    token = sample_with_temperature(logits, temperature)

    return token
