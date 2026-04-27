import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";

const server = new McpServer({
  name: "refund-policy-server",
  version: "1.0.0",
});

server.registerTool(
  "get_refund_policy",
  {
    description: "Returns the company refund policy.",
  },
  async () => ({
    content: [
      {
        type: "text" as const,
        text: '{ "policy": "Refunds allowed within 30 days" }',
      },
    ],
  }),
);

async function startServer(): Promise<void> {
  const transport = new StdioServerTransport();
  await server.connect(transport);
}

void startServer();
