CREATE EXTENSION IF NOT EXISTS vector;

-- One row per import of the USITC schedule, so revisions can be tracked later.
CREATE TABLE IF NOT EXISTS hts_release (
    id          serial PRIMARY KEY,
    fetched_at  timestamptz NOT NULL DEFAULT now(),
    source      text NOT NULL,
    row_count   int NOT NULL
);

-- Every row of the schedule export, in schedule order. Header rows have an
-- empty htsno; full_path joins the row's ancestors so "Other" becomes readable.
CREATE TABLE IF NOT EXISTS hts_line (
    id                serial PRIMARY KEY,
    ordinal           int NOT NULL,
    htsno             text NOT NULL DEFAULT '',
    digits            int NOT NULL,
    indent            int NOT NULL,
    description       text NOT NULL,
    full_path         text NOT NULL,
    superior          boolean NOT NULL DEFAULT false,
    units             text[] NOT NULL DEFAULT '{}',
    general           text NOT NULL DEFAULT '',
    special           text NOT NULL DEFAULT '',
    other             text NOT NULL DEFAULT '',
    footnotes         jsonb NOT NULL DEFAULT '[]',
    quota_quantity    text NOT NULL DEFAULT '',
    additional_duties text NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS hts_line_htsno_idx ON hts_line (htsno);
CREATE INDEX IF NOT EXISTS hts_line_fts_idx ON hts_line USING gin (to_tsvector('english', full_path));

-- CBP CROSS rulings. Metadata comes from the search API; text is filled in
-- later by a resumable fetch and stays NULL until then.
CREATE TABLE IF NOT EXISTS cross_ruling (
    ruling_number text PRIMARY KEY,
    collection    text NOT NULL,
    subject       text NOT NULL DEFAULT '',
    categories    text NOT NULL DEFAULT '',
    ruling_date   date,
    tariffs       text NOT NULL DEFAULT '',
    tariff_codes  text[] NOT NULL DEFAULT '{}',
    revoked       boolean NOT NULL DEFAULT false,
    raw           jsonb NOT NULL,
    text          text,
    fetched_at    timestamptz
);
CREATE INDEX IF NOT EXISTS cross_ruling_date_idx ON cross_ruling (ruling_date DESC);
CREATE INDEX IF NOT EXISTS cross_ruling_codes_idx ON cross_ruling USING gin (tariff_codes);
