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

alter table documents enable row level security;
create policy "service role full access" on documents
  using (true) with check (true);

create or replace function match_documents (
  query_embedding vector(1536),
  match_count     int     default null,
  filter          jsonb   default '{}'
) returns table (
  id bigint, content text, metadata jsonb, embedding jsonb, similarity float
)
language plpgsql as $$
#variable_conflict use_column
begin
  return query
  select id, content, metadata, (embedding::text)::jsonb,
         1 - (documents.embedding <=> query_embedding)
  from documents
  where metadata @> filter
  order by documents.embedding <=> query_embedding
  limit match_count;
end;
$$;
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
SUPABASE_PRIVATE_KEY=   # service role key (not the anon key)
```

## Usage

### 1. Ingest documents

Drop PDF files into the `data/` folder, then run:

```bash
npm run ingest
```

Each PDF is hashed for deduplication, so re-running is safe — already-ingested files are skipped. The ingest pipeline redacts PII and extracts `document_type`, `provider`, and `product_name` metadata from each document before uploading chunks to Supabase.

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
[sha256 dedup] → [pdf-parse] → [gpt-4o-mini: PII redact + metadata] → [Supabase pgvector]

User message
    │  npm run chat
    ▼
[RedactNode: email regex + gpt-4o-mini name redaction]
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
