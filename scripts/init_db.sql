CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS trials (
    nct_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    status TEXT,
    conditions TEXT[] NOT NULL DEFAULT '{}',
    interventions TEXT[] NOT NULL DEFAULT '{}',
    eligibility_criteria TEXT,
    source JSONB NOT NULL DEFAULT '{}'::jsonb,
    search_document TSVECTOR GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(array_to_string(conditions, ' '), '')), 'B') ||
        setweight(to_tsvector('english', coalesce(array_to_string(interventions, ' '), '')), 'B') ||
        setweight(to_tsvector('english', coalesce(eligibility_criteria, '')), 'C')
    ) STORED
);

CREATE INDEX IF NOT EXISTS trials_search_document_idx ON trials USING GIN (search_document);
CREATE INDEX IF NOT EXISTS trials_conditions_idx ON trials USING GIN (conditions);
