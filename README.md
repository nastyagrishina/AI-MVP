# AI Insurance & Finance Agent

A LangGraph agent that answers questions about insurance and financial documents. PDFs are ingested into a Supabase pgvector store with automatic PII redaction and metadata extraction. At runtime the agent retrieves relevant chunks via semantic search and can also query a local MCP tool server for structured policy data.

## Prerequisites

- Node.js 18+
- OpenAI API key
- Anthropic API key (rate-limit fallback)
- A [Supabase](https://supabase.com) project with the schema below applied

## One-time Supabase setup

Run this SQL once in your Supabase project's **SQL Editor**:

```sql
create extension if not exists vector;

create table if not exists documents (
  id        bigserial primary key,
  content   text,
  metadata  jsonb,
  embedding vector(1536)
);

-- Grant the service_role (used by the secret/service key) full access.
-- Without this the REST API returns 403 even though RLS is off.
grant select, insert, update, delete on documents to service_role;

create or replace function match_documents (
  query_embedding vector(1536),
  match_count     int     default null,
  filter          jsonb   default '{}'
) returns table (
  id bigint, content text, metadata jsonb, similarity float
)
language plpgsql as $$
begin
  return query
  select
    documents.id,
    documents.content,
    documents.metadata,
    1 - (documents.embedding <=> query_embedding) as similarity
  from documents
  where documents.metadata @> filter
  order by documents.embedding <=> query_embedding
  limit match_count;
end;
$$;
-- Note: all columns are qualified with the table name (documents.id, documents.content, etc.)
-- to avoid PL/pgSQL error 42702 "column reference is ambiguous" — PL/pgSQL treats
-- every name in the RETURNS TABLE(...) clause as an output variable, so an unqualified
-- column name like `id` is ambiguous between the output variable and the table column.

grant execute on function match_documents to service_role;
```

## Setup

```bash
npm install
```

Copy `.env.example` to `.env` and fill in all five values:

```
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
SUPABASE_URL=
SUPABASE_PRIVATE_KEY=   # secret key (new format: sb_secret_…) or legacy service_role JWT — not the anon/publishable key
```

## Usage

### 1. Ingest documents

Drop PDF files into the `data/` folder, then run:

```bash
npm run ingest
```

The ingest pipeline applies a three-layer PII strategy before anything reaches Supabase:

1. **Regex pre-scrub** — strips structured PII (Czech birth numbers, phone numbers, IBANs, emails, policy numbers) from the raw text before any data leaves your system.
2. **Metadata extraction** (Pass 1, `gpt-4o-mini`) — classifies the document and detects whether it contains personal data, using only the pre-scrubbed first page.
3. **LLM redaction** (Pass 2, `gpt-4o-mini`, parallel chunks) — only runs for personal documents (contracts, proposals). Handles unstructured PII — names and addresses — that regex cannot reliably catch. Public documents (general terms and conditions, brochures) skip this step entirely.
4. **Post-LLM regex sweep** — a final safety-net pass to catch any structured PII the LLM may have missed.

The `document_type`, `provider`, `product_name`, and `is_personal` fields are stored as metadata on every chunk. Already-ingested filenames are skipped on re-run.

### 2. Chat

```bash
npm run chat
```

An interactive terminal session. Ask questions in plain language — the agent searches the ingested documents and, when relevant, calls the MCP refund-policy tool. You can narrow a search to a specific provider, e.g.:

> *"What does VZP cover for property damage?"*

Type `exit` or press `Ctrl+D` to quit.

## How it works

```
PDF files
    │  npm run ingest
    ▼
[pdf-parse]
    ▼
[regex pre-scrub]          ← strips birth numbers, phones, IBANs, emails, policy numbers
    ▼
[gpt-4o-mini: metadata]    ← Pass 1, first 3k chars only, already pre-scrubbed
    ▼
[is_personal?]
  ├─ NO  → store as-is (public terms/conditions — no LLM redaction needed)
  └─ YES → [gpt-4o-mini: redaction, parallel chunks]   ← Pass 2, names & addresses only
                ▼
           [post-LLM regex sweep]   ← safety net
    ▼
[Supabase pgvector]

User message
    │  npm run chat
    ▼
[AgentNode: gpt-4o] ←→ [search_insurance_and_financial_docs  (Supabase RAG)]
                    ←→ [get_refund_policy  (MCP stdio server)]
```

## Stack

| | |
|---|---|
| Orchestration | LangGraph |
| LLMs | OpenAI `gpt-4o` / `gpt-4o-mini`, Anthropic Claude (fallback) |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector store | Supabase pgvector |
| Tool protocol | MCP (`@modelcontextprotocol/sdk`) |
| Language | TypeScript, ESM, `module: nodenext` |
