// Interactive CLI chat — one isolated turn per user message (no cross-turn memory).
// Run: npm run chat
// Type a question and press Enter. Type "exit" / "quit" or Ctrl+D to leave.
// Requires OPENAI_API_KEY in .env

import "dotenv/config";

import { createInterface } from "readline/promises";
import type { AIMessage, ToolMessage } from "@langchain/core/messages";
import { HumanMessage } from "@langchain/core/messages";

import { buildGraph } from "./graph.js";

// Tool activity logger:
// Scans result.messages after each invoke and prints minimal tool visibility.
function logToolActivity(messages: unknown[]): void {
  let logged = false;

  for (const msg of messages) {
    const m = msg as { _getType?: () => string };
    if (typeof m._getType !== "function") continue;
    const type = m._getType();

    if (type === "ai") {
      const ai = m as AIMessage;
      const calls = ai.tool_calls;
      if (calls && calls.length > 0) {
        const names = calls.map((c) => c.name).join(", ");
        console.log(`  [tools called] ${names}`);
        logged = true;
      }
    }

    if (type === "tool") {
      const tm = m as ToolMessage;
      const name = tm.name ?? "tool";
      const raw = typeof tm.content === "string" ? tm.content : JSON.stringify(tm.content);
      const preview = raw.length > 120 ? raw.slice(0, 120) + "…" : raw;
      console.log(`  [tool result]  ${name} → ${preview}`);
      logged = true;
    }
  }

  if (logged) console.log(); // blank line after tool block for readability
}

// Main chat loop:
async function chat(): Promise<void> {
  console.log("Starting up — connecting to MCP server and building RAG index…");
  const { graph, client } = await buildGraph();
  console.log("Ready. Type a question or \"exit\" to quit.\n");

  // No prompt string on createInterface — we write "Your Question: " manually so it
  // only appears after the previous agent reply has been fully printed.
  const rl = createInterface({ input: process.stdin, output: process.stdout });

  const shutdown = async () => {
    rl.close();
    await client.close();
    process.exit(0);
  };

  process.on("SIGINT", () => void shutdown());

  try {
    // eslint-disable-next-line no-constant-condition
    while (true) {
      let line: string;
      try {
        line = await rl.question("You: ");
      } catch {
        // EOF (Ctrl+D)
        break;
      }

      const trimmed = line.trim();
      if (!trimmed) continue;
      if (trimmed === "exit" || trimmed === "quit") break;

      process.stdout.write("Thinking…\n");

      const result = await graph.invoke({
        messages: [new HumanMessage(trimmed)],
      });

      const msgs = result.messages as unknown[];

      // Print minimal tool transparency before the answer
      logToolActivity(msgs);

      // Print the final AI reply
      const finalMsg = [...msgs]
        .reverse()
        .find((m) => {
          const t = m as { _getType?: () => string };
          return typeof t._getType === "function" && t._getType() === "ai";
        }) as AIMessage | undefined;

      const content = finalMsg?.content;
      const text = typeof content === "string" ? content : JSON.stringify(content);
      console.log(`Agent: ${text}\n`);
    }
  } finally {
    rl.close();
    await client.close();
  }
}

void chat();
