-- ===============================
-- Entity Table
-- ===============================
CREATE TABLE IF NOT EXISTS entity (
    entity_id SERIAL PRIMARY KEY,
    entity_name TEXT,
    entity_type TEXT,
    properties JSONB
);

-- GIN index for fast lookup
CREATE INDEX idx_entity_name_gin
ON entity USING GIN (to_tsvector('english', entity_name));

-- ===============================
-- Relation Table
-- ===============================
CREATE TABLE IF NOT EXISTS relation (
    relation_id SERIAL PRIMARY KEY,
    src_entity INT,
    dst_entity INT,
    relation_type TEXT,
    weight FLOAT
);

CREATE INDEX idx_relation_type_gin
ON relation USING GIN (to_tsvector('english', relation_type));

-- ===============================
-- Entity Property Table
-- ===============================
CREATE TABLE IF NOT EXISTS entity_property (
    entity_id INT,
    key TEXT,
    value TEXT
);
