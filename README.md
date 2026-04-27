# Local TypeScript AI Agent MVP

A local AI agent built with LangGraph, LangChain, and the Model Context Protocol (MCP). The agent redacts PII from user input before passing it to a tool-enabled GPT-4o agent that can query an in-memory vector store (RAG) or a local MCP tool server.

## Architecture

```
User message
    │
    ▼
┌─────────────┐    stdio subprocess
│  RedactNode │──────────────────────────────────────────┐
│  (PII scrub)│                                          │
└──────┬──────┘                              ┌───────────┴────────────┐
       │                                     │  src/mcp.ts            │
       ▼                                     │  MCP Server (stdio)    │
┌─────────────┐   tool calls                 │  Tool: get_refund_policy│
│  AgentNode  │◄────────────────────────────►└────────────────────────┘
│  (gpt-4o)   │
└──────┬──────┘   tool calls
       │◄────────────────────────────────────┐
       ▼                                     │
┌─────────────┐                   ┌──────────┴──────────┐
│  ToolNode   │                   │  src/rag.ts          │
│  (executor) │                   │  MemoryVectorStore   │
└─────────────┘                   │  Tool: search_company│
                                  └─────────────────────-┘
```

### Files

| File | Role |
|------|------|
| `src/mcp.ts` | MCP stdio server — exposes `get_refund_policy` tool |
| `src/rag.ts` | In-memory RAG — embeds 3 company-history docs, exports `search_company_history` tool |
| `src/graph.ts` | LangGraph orchestrator — PII redaction → agent loop → tool execution |

## Prerequisites

- Node.js 18+
- An OpenAI API key

## Setup

```bash
npm install
```

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
```

## Running

```bash
npm run dev
```

This runs `src/graph.ts` directly via `tsx`. It will:

1. Spawn `src/mcp.ts` as a child process (the MCP server) over stdio
2. Build the in-memory vector store by embedding 3 company-history strings via `text-embedding-3-small`
3. Invoke the graph with a hardcoded test message:
   > "Hi, my name is John Doe and my email is test@test.com. Should I use RAG for company history, or the MCP for the refund policy?"
4. Print the agent's final response

### Expected output

```
Building graph and connecting to MCP server…

=== Agent Final Response ===
For company history, use the RAG tool (search_company_history) — it searches
Acme Corp's internal knowledge base. For the refund policy, use the MCP tool
(get_refund_policy) — it returns the official policy directly.
```

The PII in the test message (`John Doe`, `test@test.com`) is scrubbed by the
RedactNode before the agent ever sees it.

### Smoke-test the MCP server in isolation

```bash
npm run mcp
```

The server starts and listens on stdin for MCP protocol messages. Press `Ctrl-C` to exit.

## Models used

| Step | Model | Purpose |
|------|-------|---------|
| Embeddings | `text-embedding-3-small` | Build the RAG vector store at startup |
| RedactNode | `gpt-4o-mini` | Fast PII name-scrubbing |
| AgentNode | `gpt-4o` | Main reasoning + tool selection |

## Stack

- **TypeScript** (ESM, `module: nodenext`)
- **LangGraph** (`@langchain/langgraph`) — state machine orchestration
- **LangChain** (`@langchain/core`, `@langchain/openai`, `@langchain/classic`) — LLMs, tools, vector store
- **MCP SDK** (`@modelcontextprotocol/sdk`) — stdio tool server
- **`@langchain/mcp-adapters`** — bridges MCP tools into LangChain's tool interface
- **Zod** — tool input schemas
