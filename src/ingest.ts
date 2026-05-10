import "dotenv/config";

import { readdir, readFile } from "fs/promises";
import { join, basename, dirname } from "path";
import { fileURLToPath } from "url";

import { createRequire } from "module";
const require = createRequire(import.meta.url);
// pdf-parse v2 is CJS-only; load it through require so ESM interop works
const { PDFParse } = require("pdf-parse") as {
  PDFParse: new (opts: { data: Buffer }) => { getText: () => Promise<{ text: string }> };
};
import { createClient } from "@supabase/supabase-js";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { SupabaseVectorStore } from "@langchain/community/vectorstores/supabase";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { z } from "zod";

// ---------------------------------------------------------------------------
// Schemas
// ---------------------------------------------------------------------------

// Pass 1 — fast metadata extraction from first page only
const DocMetadataSchema = z.object({
  document_type: z
    .string()
    .describe(
      "Category of the document, e.g. insurance, credit_card, loan, mortgage, terms_and_conditions.",
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
  is_personal: z
    .boolean()
    .describe(
      "true ONLY if the document contains actual personal identifiers belonging to a specific real person: " +
      "a named individual, their birth number, home address, phone, email, or a policy/contract number " +
      "pre-filled for them. Examples: a signed insurance contract, a loan agreement, an account statement. " +
      "false if the document is a generic public document that merely references 'the policyholder', " +
      "'the client', or 'the insured' in the abstract — even if it is branded or product-specific. " +
      "Examples: general terms and conditions, product brochures, rate schedules, coverage handbooks.",
    ),
});

type DocMetadata = z.infer<typeof DocMetadataSchema>;

// Pass 2 — redaction of a single chunk
const RedactionSchema = z.object({
  redacted_text: z
    .string()
    .describe(
      "The input text with every personal name, postal address, birth number, " +
      "phone number, email address, and policy/contract number replaced with [REDACTED].",
    ),
});

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

// Only send the first N chars to the LLM for metadata — the header is always there
const METADATA_CHARS = 3_000;

// Max chars per redaction chunk — ~1.5k tokens, fast and parallel-friendly
const REDACTION_CHUNK_CHARS = 6_000;

// Max concurrent redaction LLM calls to stay within rate limits
const MAX_PARALLEL_REDACTION = 5;

// ---------------------------------------------------------------------------
// Czech PII regex patterns — post-LLM safety sweep
// ---------------------------------------------------------------------------
const PII_PATTERNS: [RegExp, string][] = [
  // Rodné číslo (birth number): 123456/7890
  [/\b\d{6}\/\d{3,4}\b/g, "[REDACTED]"],
  // Czech/Slovak phone: +420 123 456 789 or 123 456 789
  [/(\+420[\s-]?)?\b\d{3}[\s-]?\d{3}[\s-]?\d{3}\b/g, "[REDACTED]"],
  // IBAN
  [/\b[A-Z]{2}\d{2}[\s]?(\d{4}[\s]?){4,6}\b/g, "[REDACTED]"],
  // Email
  [/[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}/g, "[REDACTED]"],
  // Policy/contract numbers: long digit sequences (10+ digits) not part of dates
  [/\b\d{10,}\b/g, "[REDACTED]"],
];

function regexSweep(text: string): string {
  let result = text;
  for (const [pattern, replacement] of PII_PATTERNS) {
    result = result.replace(pattern, replacement);
  }
  return result;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// eslint-disable-next-line @typescript-eslint/no-explicit-any
async function isAlreadyIngested(supabase: any, filename: string): Promise<boolean> {
  const { data, error } = await supabase
    .from("documents")
    .select("id")
    .eq("metadata->>source", filename)
    .limit(1);
  if (error) throw new Error(`Dedup check failed: ${error.message}`);
  return data.length > 0;
}

function elapsed(startMs: number): string {
  return `${((Date.now() - startMs) / 1000).toFixed(1)}s`;
}

function chunkText(text: string, size: number): string[] {
  const chunks: string[] = [];
  for (let i = 0; i < text.length; i += size) {
    chunks.push(text.slice(i, i + size));
  }
  return chunks;
}

async function runWithLimit<T>(
  tasks: (() => Promise<T>)[],
  limit: number,
): Promise<T[]> {
  const results: T[] = new Array(tasks.length);
  let idx = 0;

  async function worker() {
    while (idx < tasks.length) {
      const i = idx++;
      results[i] = await tasks[i]!();
    }
  }

  await Promise.all(Array.from({ length: Math.min(limit, tasks.length) }, worker));
  return results;
}

// Pass 1: extract metadata from first page only
async function extractMetadata(
  llm: ChatOpenAI,
  filename: string,
  firstPageText: string,
): Promise<DocMetadata> {
  const structured = llm.withStructuredOutput(DocMetadataSchema);
  return structured.invoke([
    {
      role: "system",
      content:
        "You are a document analysis assistant. " +
        "Given the filename and opening text of a financial or insurance document, classify it. " +
        "For is_personal: look for concrete personal identifiers — a real person's name, " +
        "birth number, home address, pre-filled contract/policy number, or similar PII. " +
        "Generic phrases like 'the policyholder', 'the client', or 'pojistník' alone do NOT make a document personal. " +
        "When in doubt, prefer false.",
    },
    {
      role: "user",
      content: `Filename: ${filename}\n\nDocument opening:\n\n${firstPageText}`,
    },
  ]);
}

// Pass 2: redact a single chunk
async function redactChunk(
  llm: ChatOpenAI,
  chunk: string,
  chunkIdx: number,
  totalChunks: number,
): Promise<string> {
  const structured = llm.withStructuredOutput(RedactionSchema);
  try {
    const result = await structured.invoke([
      {
        role: "system",
        content:
          "You are a PII redaction assistant. Replace every personal name, " +
          "postal address, birth number, phone number, email, and policy/contract " +
          "number with [REDACTED]. Return only the redacted text, preserving all " +
          "other content exactly.",
      },
      {
        role: "user",
        content: chunk,
      },
    ]);
    return result.redacted_text;
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    const isRateLimit =
      msg.includes("429") ||
      msg.toLowerCase().includes("rate limit") ||
      msg.toLowerCase().includes("too many requests");
    if (isRateLimit) {
      throw new Error(
        `Rate limit hit on chunk ${chunkIdx + 1}/${totalChunks}. ` +
        `Wait a minute and re-run. (OpenAI: ${msg})`,
      );
    }
    throw new Error(`Chunk ${chunkIdx + 1}/${totalChunks} failed: ${msg}`);
  }
}

async function redactDocument(llm: ChatOpenAI, text: string): Promise<string> {
  const chunks = chunkText(text, REDACTION_CHUNK_CHARS);
  const tasks = chunks.map((chunk, i) => () => redactChunk(llm, chunk, i, chunks.length));
  const redacted = await runWithLimit(tasks, MAX_PARALLEL_REDACTION);
  return redacted.join("");
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

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const supabase = createClient(SUPABASE_URL, SUPABASE_PRIVATE_KEY) as any;
  const embeddings = new OpenAIEmbeddings({ model: "text-embedding-3-small" });
  const llm = new ChatOpenAI({ model: "gpt-4o-mini", temperature: 0 });
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 150,
  });

  const allFiles = await readdir(dataDir);
  const pdfFiles = allFiles.filter((f) => f.toLowerCase().endsWith(".pdf"));

  if (pdfFiles.length === 0) {
    console.log("No PDF files found in data/. Add PDFs and re-run.");
    return;
  }

  console.log(`Found ${pdfFiles.length} PDF file(s).\n`);

  const totalStart = Date.now();

  for (const filename of pdfFiles) {
    const filePath = join(dataDir, filename);
    const fileStart = Date.now();
    console.log(`Processing: ${filename}`);

    // 1. Parse PDF text
    let t = Date.now();
    const fileBuffer = await readFile(filePath);

    if (await isAlreadyIngested(supabase, filename)) {
      console.log(`  → Already ingested, skipping.\n`);
      continue;
    }

    const parser = new PDFParse({ data: fileBuffer });
    const parsed = await parser.getText();
    const rawText = parsed.text.trim();
    if (!rawText) {
      console.warn(`  → No extractable text found, skipping.\n`);
      continue;
    }
    console.log(`  → Extracted ${rawText.length.toLocaleString()} chars. (${elapsed(t)})`);

    // 2. Regex pre-scrub — strip structured PII before any LLM call
    const preScrubbed = regexSweep(rawText);
    const preScrubCount = (preScrubbed.match(/\[REDACTED\]/g) ?? []).length;
    if (preScrubCount > 0) {
      console.log(`  → Regex pre-scrub: ${preScrubCount} item(s) redacted before LLM.`);
    }

    // 3. Pass 1 — metadata from first page only (fast, pre-scrubbed)
    t = Date.now();
    console.log(`  → Pass 1: extracting metadata from first ${METADATA_CHARS} chars…`);
    const metadata = await extractMetadata(llm, filename, preScrubbed.slice(0, METADATA_CHARS));
    console.log(`  → document_type="${metadata.document_type}", provider="${metadata.provider}", product_name="${metadata.product_name}", is_personal=${metadata.is_personal} (${elapsed(t)})`);

    // 4. Pass 2 — LLM redaction of names/addresses (only for personal documents)
    //    Input is already pre-scrubbed so only unstructured PII (names, addresses) remains.
    let finalText: string;
    if (!metadata.is_personal) {
      console.log(`  → Public document — skipping LLM redaction.`);
      finalText = preScrubbed;
    } else {
      t = Date.now();
      const chunks = chunkText(preScrubbed, REDACTION_CHUNK_CHARS);
      console.log(`  → Pass 2: redacting ${chunks.length} chunk(s) in parallel (≤${MAX_PARALLEL_REDACTION} at a time)…`);
      const heartbeat = setInterval(() => {
        process.stdout.write(`     still redacting… (${elapsed(t)})\n`);
      }, 10_000);
      try {
        finalText = await redactDocument(llm, preScrubbed);
      } finally {
        clearInterval(heartbeat);
      }
      console.log(`  → LLM redaction done. (${elapsed(t)})`);

      // 5. Post-LLM regex sweep — safety net for anything the LLM missed
      const before = (finalText.match(/\[REDACTED\]/g) ?? []).length;
      finalText = regexSweep(finalText);
      const after = (finalText.match(/\[REDACTED\]/g) ?? []).length;
      if (after - before > 0) {
        console.log(`  → Post-LLM regex sweep: ${after - before} additional item(s) caught.`);
      }
    }

    // 7. Chunk for embeddings
    t = Date.now();
    const embeddingChunks = await splitter.createDocuments(
      [finalText],
      [
        {
          source: basename(filename),
          document_type: metadata.document_type,
          provider: metadata.provider,
          product_name: metadata.product_name,
          is_personal: metadata.is_personal,
        },
      ],
    );
    console.log(`  → Created ${embeddingChunks.length} embedding chunk(s). (${elapsed(t)})`);

    // 8. Embed + upload to Supabase
    t = Date.now();
    console.log("  → Embedding and uploading to Supabase…");
    const embedHeartbeat = setInterval(() => {
      process.stdout.write(`     still embedding… (${elapsed(t)})\n`);
    }, 10_000);
    try {
      await SupabaseVectorStore.fromDocuments(embeddingChunks, embeddings, {
        client: supabase,
        tableName: "documents",
        queryName: "match_documents",
      });
    } finally {
      clearInterval(embedHeartbeat);
    }
    console.log(`  → Uploaded to Supabase. (${elapsed(t)})`);

    console.log(`  ✓ Done. (file total: ${elapsed(fileStart)})\n`);
  }

  console.log(`Ingest complete. (total: ${elapsed(totalStart)})`);
}

ingest().catch((err) => {
  console.error("Ingest failed:", err);
  process.exit(1);
});
