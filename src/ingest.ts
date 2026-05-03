import "dotenv/config";

import { createHash } from "crypto";
import { readdir, readFile } from "fs/promises";
import { join, basename, dirname } from "path";
import { fileURLToPath } from "url";

import { createRequire } from "module";
const require = createRequire(import.meta.url);
// pdf-parse is CJS-only; load it through require so ESM interop works
const pdfParse = require("pdf-parse") as (
  buf: Buffer,
) => Promise<{ text: string; numpages: number; info: unknown }>;
import { createClient } from "@supabase/supabase-js";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { SupabaseVectorStore } from "@langchain/community/vectorstores/supabase";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { Document } from "@langchain/core/documents";
import { z } from "zod";

// ---------------------------------------------------------------------------
// Structured output schema for PII redaction + metadata extraction
// ---------------------------------------------------------------------------
const DocAnalysisSchema = z.object({
  redacted_text: z
    .string()
    .describe(
      "The full document text with all personal names, addresses, and policy numbers replaced with [REDACTED].",
    ),
  document_type: z
    .string()
    .describe(
      "Category of the document, e.g. insurance, credit_card, loan, mortgage.",
    ),
  provider: z
    .string()
    .describe(
      "The company or institution that issued the document, e.g. VZP, KB, Allianz, CSOB.",
    ),
  product_name: z
    .string()
    .describe(
      "The specific product or plan name, e.g. 'Flexi Life', 'Gold Credit Card'.",
    ),
});

type DocAnalysis = z.infer<typeof DocAnalysisSchema>;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function sha256(buf: Buffer): string {
  return createHash("sha256").update(buf).digest("hex");
}

async function isAlreadyIngested(
  supabase: ReturnType<typeof createClient>,
  hash: string,
): Promise<boolean> {
  const { data, error } = await supabase
    .from("documents")
    .select("id")
    .filter("metadata->>file_hash", "eq", hash)
    .limit(1);

  if (error) throw new Error(`Dedup check failed: ${error.message}`);
  return (data?.length ?? 0) > 0;
}

async function analyzeDocument(
  llm: ChatOpenAI,
  rawText: string,
): Promise<DocAnalysis> {
  const structured = llm.withStructuredOutput(DocAnalysisSchema);
  return structured.invoke([
    {
      role: "system",
      content:
        "You are a document analysis assistant. " +
        "Given the raw text of a financial or insurance document, you must:\n" +
        "1. Replace every personal name, postal address, and policy/contract number with [REDACTED].\n" +
        "2. Classify the document and extract provider and product metadata.\n" +
        "Return your analysis in the requested structured format.",
    },
    {
      role: "user",
      content: `Document text:\n\n${rawText}`,
    },
  ]);
}

// ---------------------------------------------------------------------------
// Main ingest pipeline
// ---------------------------------------------------------------------------
async function ingest() {
  const { SUPABASE_URL, SUPABASE_PRIVATE_KEY, OPENAI_API_KEY } = process.env;
  if (!SUPABASE_URL || !SUPABASE_PRIVATE_KEY) {
    throw new Error("SUPABASE_URL and SUPABASE_PRIVATE_KEY must be set in .env");
  }
  if (!OPENAI_API_KEY) {
    throw new Error("OPENAI_API_KEY must be set in .env");
  }

  const __dirname = dirname(fileURLToPath(import.meta.url));
  const dataDir = join(__dirname, "..", "data");

  // Cast required: @langchain/community's SupabaseVectorStore uses a narrower
  // internal generic than createClient<any> produces at the call site.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const supabase = createClient(SUPABASE_URL, SUPABASE_PRIVATE_KEY) as any;
  const embeddings = new OpenAIEmbeddings({ model: "text-embedding-3-small" });
  const llm = new ChatOpenAI({ model: "gpt-4o-mini", temperature: 0 });
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 150,
  });

  // Discover all PDF files in data/
  const allFiles = await readdir(dataDir);
  const pdfFiles = allFiles.filter((f) => f.toLowerCase().endsWith(".pdf"));

  if (pdfFiles.length === 0) {
    console.log("No PDF files found in data/. Add PDFs and re-run.");
    return;
  }

  console.log(`Found ${pdfFiles.length} PDF file(s).\n`);

  for (const filename of pdfFiles) {
    const filePath = join(dataDir, filename);
    console.log(`Processing: ${filename}`);

    // 1. Hash the raw file bytes for deduplication
    const fileBuffer = await readFile(filePath);
    const fileHash = sha256(fileBuffer);

    // 2. Skip if already ingested
    if (await isAlreadyIngested(supabase, fileHash)) {
      console.log(`  → Already ingested (hash ${fileHash.slice(0, 12)}…), skipping.\n`);
      continue;
    }

    // 3. Parse PDF text
    const parsed = await pdfParse(fileBuffer);
    const rawText = parsed.text.trim();
    if (!rawText) {
      console.warn(`  → No extractable text found, skipping.\n`);
      continue;
    }
    console.log(`  → Extracted ${rawText.length} characters from PDF.`);

    // 4. PII redaction + metadata extraction via gpt-4o-mini
    console.log("  → Calling gpt-4o-mini for PII redaction and metadata extraction…");
    const analysis = await analyzeDocument(llm, rawText);
    console.log(
      `  → document_type="${analysis.document_type}", provider="${analysis.provider}", product_name="${analysis.product_name}"`,
    );

    // 5. Chunk the anonymised text
    const chunks = await splitter.createDocuments(
      [analysis.redacted_text],
      [
        {
          source: basename(filename),
          file_hash: fileHash,
          document_type: analysis.document_type,
          provider: analysis.provider,
          product_name: analysis.product_name,
        },
      ],
    );
    console.log(`  → Created ${chunks.length} chunk(s).`);

    // 6. Embed + upload to Supabase
    console.log("  → Embedding and uploading to Supabase…");
    await SupabaseVectorStore.fromDocuments(chunks, embeddings, {
      client: supabase,
      tableName: "documents",
      queryName: "match_documents",
    });

    console.log(`  ✓ Done.\n`);
  }

  console.log("Ingest complete.");
}

ingest().catch((err) => {
  console.error("Ingest failed:", err);
  process.exit(1);
});
