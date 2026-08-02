CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS trials (
    nct_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    status TEXT,
    conditions TEXT[] NOT NULL DEFAULT '{}',
    interventions TEXT[] NOT NULL DEFAULT '{}',
    eligibility_criteria TEXT,
    source JSONB NOT NULL DEFAULT '{}'::jsonb,
    search_document TSVECTOR NOT NULL DEFAULT ''::tsvector
);

CREATE OR REPLACE FUNCTION update_trials_search_document()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.search_document :=
        setweight(to_tsvector('english', coalesce(NEW.title, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(array_to_string(NEW.conditions, ' '), '')), 'B') ||
        setweight(to_tsvector('english', coalesce(array_to_string(NEW.interventions, ' '), '')), 'B') ||
        setweight(to_tsvector('english', coalesce(NEW.eligibility_criteria, '')), 'C');
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trials_search_document_update ON trials;
CREATE TRIGGER trials_search_document_update
BEFORE INSERT OR UPDATE OF title, conditions, interventions, eligibility_criteria
ON trials
FOR EACH ROW
EXECUTE FUNCTION update_trials_search_document();

CREATE INDEX IF NOT EXISTS trials_search_document_idx ON trials USING GIN (search_document);
CREATE INDEX IF NOT EXISTS trials_conditions_idx ON trials USING GIN (conditions);
