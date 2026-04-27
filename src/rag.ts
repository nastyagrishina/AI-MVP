import "dotenv/config";

import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { OpenAIEmbeddings } from "@langchain/openai";
import { tool } from "@langchain/core/tools";
import { z } from "zod";

const COMPANY_HISTORY = [
  "Acme Corp was founded in 1998 by Alice and Bob to build industrial-grade widgets.",
  "In 2005, Acme Corp expanded globally, opening offices in London, Tokyo, and São Paulo.",
  "Acme Corp launched its first AI-powered product line in 2021, doubling revenue within two years.",
];

export async function createSearchCompanyHistoryTool() {
  const embeddings = new OpenAIEmbeddings({ model: "text-embedding-3-small" });
  const store = await MemoryVectorStore.fromTexts(
    COMPANY_HISTORY,
    COMPANY_HISTORY.map((_, i) => ({ index: i })),
    embeddings,
  );

  return tool(
    async ({ query }: { query: string }) => {
      const results = await store.similaritySearch(query, 3);
      return results.map((doc) => doc.pageContent).join("\n\n");
    },
    {
      name: "search_company_history",
      description:
        "Search Acme Corp's company history. Use this tool when the user asks about the company's background, founding, expansion, or product launches.",
      schema: z.object({
        query: z.string().describe("The search query about company history"),
      }),
    },
  );
}
