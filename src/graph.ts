import "dotenv/config";

import { fileURLToPath } from "url";
import { dirname, join } from "path";
import type { BaseMessage } from "@langchain/core/messages";
import {
  HumanMessage,
  RemoveMessage,
  SystemMessage,
} from "@langchain/core/messages";
import { ChatOpenAI } from "@langchain/openai";
import { ChatAnthropic } from "@langchain/anthropic";
import { END, MessagesAnnotation, START, StateGraph } from "@langchain/langgraph";
import { ToolNode, toolsCondition } from "@langchain/langgraph/prebuilt";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { loadMcpTools } from "@langchain/mcp-adapters";

import { createSearchCompanyHistoryTool } from "./rag.js";


const EMAIL_RE = /\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b/g;


export async function buildGraph() {
  const __dirname = dirname(fileURLToPath(import.meta.url));
  const mcpScriptPath = join(__dirname, "mcp.ts");
  const transport = new StdioClientTransport({
    command: "npx",
    args: ["tsx", mcpScriptPath],
  });
  const client = new Client({ name: "refund-client", version: "1.0.0" });
  await client.connect(transport);
  const mcpTools = await loadMcpTools("refund", client, {
    throwOnLoadError: true,
  });

  const ragTool = await createSearchCompanyHistoryTool();

  const allTools = [...mcpTools, ragTool];

  // Returns true for OpenAI 429 rate-limit errors (LangChain tags them with
  // lc_error_code: "MODEL_RATE_LIMIT").
  const isRateLimit = (e: unknown): boolean =>
    !!e &&
    typeof e === "object" &&
    "lc_error_code" in e &&
    (e as { lc_error_code: string }).lc_error_code === "MODEL_RATE_LIMIT";

  const miniPrimary = new ChatOpenAI({ model: "gpt-4o-mini", temperature: 0 });
  const miniFallback = new ChatAnthropic({ model: "claude-haiku-4-5", temperature: 0 });

  const agentPrimary = new ChatOpenAI({ model: "gpt-4o", temperature: 0 }).bindTools(allTools);
  const agentFallback = new ChatAnthropic({ model: "claude-sonnet-4-5", temperature: 0 }).bindTools(allTools);

  // Thin wrappers that catch rate-limit errors and retry on the backup provider.
  const miniLLM = {
    invoke: async (messages: Parameters<typeof miniPrimary.invoke>[0]) => {
      try {
        return await miniPrimary.invoke(messages);
      } catch (e) {
        if (!isRateLimit(e)) throw e;
        console.error("  [fallback] gpt-4o-mini rate-limited → switching to claude-haiku-4-5");
        return miniFallback.invoke(messages);
      }
    },
  };

  const agentLLM = {
    invoke: async (messages: Parameters<typeof agentPrimary.invoke>[0]) => {
      try {
        return await agentPrimary.invoke(messages);
      } catch (e) {
        if (!isRateLimit(e)) throw e;
        console.error("  [fallback] gpt-4o rate-limited → switching to claude-sonnet-4-5");
        return agentFallback.invoke(messages);
      }
    },
  };

  // RedactNode: Hybrid PII scrubbing
  //   1. Regex replaces email addresses → [REDACTED_EMAIL]
  //   2. gpt-4o-mini replaces personal names → [REDACTED_NAME]
  const redactNode = async (state: typeof MessagesAnnotation.State) => {
    const lastHuman = [...state.messages]
      .reverse()
      .find((m) => m._getType() === "human");

    const msgId = lastHuman?.id;
    if (!lastHuman || !msgId) return {};

    const rawText =
      typeof lastHuman.content === "string"
        ? lastHuman.content
        : JSON.stringify(lastHuman.content);

    // Step 1: deterministic email redaction
    const emailRedacted = rawText.replace(EMAIL_RE, "[REDACTED_EMAIL]");

    // Step 2: LLM-based name redaction
    const response = await miniLLM.invoke([
      new SystemMessage(
        "You are a PII redaction filter. Remove all personal names from the text, replacing each with [REDACTED_NAME]. " +
          "Output ONLY the scrubbed text — no commentary, no explanation.",
      ),
      new HumanMessage(emailRedacted),
    ]);

    const scrubbed =
      typeof response.content === "string"
        ? response.content
        : JSON.stringify(response.content);

    return {
      messages: [
        // Remove the original user turn, then append the scrubbed replacement.
        new RemoveMessage({ id: msgId }),
        new HumanMessage(scrubbed),
      ],
    };
  };

  const AGENT_SYSTEM = new SystemMessage(
    "You are a helpful assistant with access to two tools: " +
      "a company knowledge base (RAG) and a refund policy tool (MCP). " +
      "Always use the available tools to answer questions. " +
      "If the tools return no relevant information, or if the question is outside " +
      "the scope of what the tools can answer, reply with exactly: " +
      '"[No Answer] Unfortunately, there\'s no information present for this question in the documents." ' +
      "Do not invent facts or answer from general knowledge.",
  );

  // AgentNode:
  // Receives the already-scrubbed message list. Calls gpt-4o with all tools
  // bound; the model decides whether to invoke a tool or respond directly.
  const agentNode = async (state: typeof MessagesAnnotation.State) => {
    const response = await agentLLM.invoke([AGENT_SYSTEM, ...state.messages]);
    return { messages: [response] };
  };

  // Graph wiring:
  // START → redact → agent ⟵→ tools → END
  const toolNode = new ToolNode(allTools);

  const graph = new StateGraph(MessagesAnnotation)
    .addNode("redact", redactNode)
    .addNode("agent", agentNode)
    .addNode("tools", toolNode)
    .addEdge(START, "redact")
    .addEdge("redact", "agent")
    .addConditionalEdges("agent", toolsCondition, {
      tools: "tools",
      [END]: END,
    })
    .addEdge("tools", "agent")
    .compile();

  return { graph, client };
}

// Test harness:
async function main() {
  console.log("Building graph and connecting to MCP server…");
  const { graph, client } = await buildGraph();

  try {
    const result = await graph.invoke({
      messages: [
        new HumanMessage(
          "Hi, my name is John Doe and my email is test@test.com. " +
            "Should I use RAG for company history, or the MCP for the refund policy?",
        ),
      ],
    });

    const msgs = result.messages as BaseMessage[];
    const finalMsg = [...msgs].reverse().find((m) => m._getType() === "ai");

    console.log("\n=== Agent Final Response ===");
    console.log(finalMsg?.content ?? "(no response)");
  } finally {
    await client.close();
  }
}

// Only run the one-shot demo when this file is the direct entry point
// (i.e. `tsx src/graph.ts`). When chat.ts imports buildGraph, main() is skipped.
const __filename = fileURLToPath(import.meta.url);
if (process.argv[1] === __filename) {
  void main();
}
