WITH RECURSIVE graph_path AS (
    SELECT
        src_entity,
        dst_entity,
        relation_type,
        1 AS depth
    FROM relation
    WHERE src_entity = :start_id

    UNION ALL

    SELECT
        r.src_entity,
        r.dst_entity,
        r.relation_type,
        gp.depth + 1
    FROM relation r
    JOIN graph_path gp
      ON r.src_entity = gp.dst_entity
    WHERE gp.depth < :max_hops
)
SELECT * FROM graph_path;
