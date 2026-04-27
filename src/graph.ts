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
import { END, MessagesAnnotation, START, StateGraph } from "@langchain/langgraph";
import { ToolNode, toolsCondition } from "@langchain/langgraph/prebuilt";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { loadMcpTools } from "@langchain/mcp-adapters";

import { createSearchCompanyHistoryTool } from "./rag.js";


const EMAIL_RE = /\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b/g;


async function buildGraph() {
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

  // LLMs:
  // Fast model used only for PII name-redaction in RedactNode
  const miniLLM = new ChatOpenAI({ model: "gpt-4o-mini", temperature: 0 });
  // Main agent model with all tools bound
  const agentLLM = new ChatOpenAI({ model: "gpt-4o", temperature: 0 }).bindTools(
    allTools,
  );

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

  // AgentNode:
  // Receives the already-scrubbed message list. Calls gpt-4o with all tools
  // bound; the model decides whether to invoke a tool or respond directly.
  const agentNode = async (state: typeof MessagesAnnotation.State) => {
    const response = await agentLLM.invoke(state.messages);
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

void main();
