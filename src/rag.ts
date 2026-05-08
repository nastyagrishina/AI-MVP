import "dotenv/config";

import { createClient } from "@supabase/supabase-js";
import { SupabaseVectorStore } from "@langchain/community/vectorstores/supabase";
import { OpenAIEmbeddings } from "@langchain/openai";
import { tool } from "@langchain/core/tools";
import { z } from "zod";

export async function createSearchPolicyDocsTool() {
  const { SUPABASE_URL, SUPABASE_PRIVATE_KEY } = process.env;
  if (!SUPABASE_URL || !SUPABASE_PRIVATE_KEY) {
    throw new Error("SUPABASE_URL and SUPABASE_PRIVATE_KEY must be set in .env");
  }

  const supabase = createClient(SUPABASE_URL, SUPABASE_PRIVATE_KEY);
  const embeddings = new OpenAIEmbeddings({ model: "text-embedding-3-small" });

  const store = new SupabaseVectorStore(embeddings, {
    client: supabase,
    tableName: "documents",
    queryName: "match_documents",
  });

  return tool(
    async ({
      query,
      filter,
    }: {
      query: string;
      filter?: Record<string, string>;
    }) => {
      const results = await store.similaritySearch(query, 10, filter);
      if (results.length === 0) {
        return "NO_RESULTS: No relevant documents found.";
      }
      return results
        .map((doc) => {
          const meta = doc.metadata as Record<string, string>;
          const header = [meta.provider, meta.document_type, meta.product_name]
            .filter(Boolean)
            .join(" | ");
          return header ? `[${header}]\n${doc.pageContent}` : doc.pageContent;
        })
        .join("\n\n---\n\n");
    },
    {
      name: "search_insurance_and_financial_docs",
      description:
        "Search ingested insurance and financial documents. " +
        "Use this tool when the user asks about policy details, coverage, premiums, " +
        "credit card benefits, loan conditions, or any other product information. " +
        'Optionally pass a filter object (e.g. { "provider": "VZP" } or { "document_type": "insurance" }) ' +
        "to narrow results to a specific provider or document type.",
      schema: z.object({
        query: z.string().describe("The search query"),
        filter: z
          .record(z.string(), z.string())
          .optional()
          .describe(
            'Optional metadata filter, e.g. { "provider": "VZP" } or { "document_type": "credit_card" }',
          ),
      }),
    },
  );
}
