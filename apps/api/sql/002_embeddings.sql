-- bge-m3 embeddings (1024 dims) for tariff lines and rulings, plus the eval split.
ALTER TABLE hts_line ADD COLUMN IF NOT EXISTS embedding vector(1024);
CREATE INDEX IF NOT EXISTS hts_line_embedding_idx
    ON hts_line USING hnsw (embedding vector_cosine_ops);

-- query_text is the product description distilled from a ruling, with the
-- decided tariff codes masked so retrieval cannot cheat off them.
ALTER TABLE cross_ruling ADD COLUMN IF NOT EXISTS query_text text;
ALTER TABLE cross_ruling ADD COLUMN IF NOT EXISTS embedding vector(1024);
CREATE INDEX IF NOT EXISTS cross_ruling_embedding_idx
    ON cross_ruling USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS eval_split (
    ruling_number text PRIMARY KEY REFERENCES cross_ruling (ruling_number),
    split         text NOT NULL,
    gold_code     text NOT NULL
);
