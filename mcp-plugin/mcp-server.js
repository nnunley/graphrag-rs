#!/usr/bin/env node
/**
 * MCP Server for GraphRAG - Knowledge Graph Memory
 *
 * Wraps the nu_plugin_graphrag nushell plugin to provide:
 * - Semantic search with graph expansion
 * - Entity and relation queries
 * - Community summaries
 */

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from '@modelcontextprotocol/sdk/types.js';
import { spawn } from 'child_process';

// Configuration
const DEFAULT_STORE = process.env.GRAPHRAG_STORE || 'conversations';
const NU_PATH = process.env.NU_PATH || 'nu';

/**
 * Execute a nushell command and return the result
 */
async function runNu(command) {
  return new Promise((resolve, reject) => {
    const proc = spawn(NU_PATH, ['-c', command], {
      env: { ...process.env },
      stdio: ['pipe', 'pipe', 'pipe'],
    });

    let stdout = '';
    let stderr = '';

    proc.stdout.on('data', (data) => {
      stdout += data.toString();
    });

    proc.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    proc.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(`nu exited with code ${code}: ${stderr}`));
      } else {
        resolve(stdout.trim());
      }
    });

    proc.on('error', (err) => {
      reject(new Error(`Failed to spawn nu: ${err.message}`));
    });
  });
}

/**
 * Search memory using graphrag query with embedding from ollama
 * Falls back to entity search if embedding fails
 */
async function searchMemory(query, store, top = 5) {
  // Try to get embedding from ollama and search
  try {
    const embedCmd = `http post -t application/json http://ndn.local:11434/api/embeddings { model: "nomic-embed-text", prompt: "${query.replace(/"/g, '\\"')}" } | get embedding | to json`;
    const embeddingJson = await runNu(embedCmd);
    const embedding = JSON.parse(embeddingJson);

    // Query graphrag with embedding
    const queryCmd = `graphrag query ${store} [${embedding.join(',')}] --top ${top} --expand 1 | to json`;
    const result = await runNu(queryCmd);
    return JSON.parse(result);
  } catch (error) {
    // Fallback to listing chunks (no semantic search)
    throw new Error(`Embedding failed: ${error.message}. Ensure ollama is running with nomic-embed-text model.`);
  }
}

/**
 * Format search results as markdown
 */
function formatSearchResults(results) {
  if (!results.chunks || results.chunks.length === 0) {
    return 'No relevant memories found.';
  }

  let text = '## Relevant Memories\n\n';
  for (const chunk of results.chunks) {
    const source = chunk.source || 'unknown';
    const fromGraph = chunk.from_graph ? ' (via graph)' : '';
    text += `### [${source}]${fromGraph}\n\n${chunk.content}\n\n---\n\n`;
  }

  if (results.entities && results.entities.length > 0) {
    text += '\n## Related Entities\n\n';
    for (const entity of results.entities) {
      text += `- **${entity.name}** (${entity.entity_type || 'unknown'})\n`;
    }
  }

  return text;
}

/**
 * List entities in a store
 */
async function listEntities(store) {
  const cmd = `graphrag entities ${store} | to json`;
  const result = await runNu(cmd);
  return JSON.parse(result);
}

/**
 * Get relations for an entity
 */
async function getRelations(store, entityName) {
  const cmd = `graphrag relations ${store} "${entityName.replace(/"/g, '\\"')}" | to json`;
  const result = await runNu(cmd);
  return JSON.parse(result);
}

/**
 * List communities with summaries
 */
async function listCommunities(store) {
  const cmd = `graphrag list-communities ${store} | to json`;
  const result = await runNu(cmd);
  return JSON.parse(result);
}

/**
 * Get store statistics
 */
async function getStats(store) {
  const cmd = `graphrag list | where name == "${store}" | first | to json`;
  const result = await runNu(cmd);
  return JSON.parse(result);
}

// Create MCP Server
const server = new Server(
  {
    name: 'graphrag',
    version: '0.1.0',
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

// Register Tools
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: 'recall',
        description: `Search knowledge graph memory for relevant context. Returns semantically similar content plus related entities discovered through graph traversal. Use this to recover context about past work, decisions, and patterns.`,
        inputSchema: {
          type: 'object',
          properties: {
            query: {
              type: 'string',
              minLength: 2,
              description: 'What to search for in memory',
            },
            top: {
              type: 'number',
              minimum: 1,
              maximum: 20,
              default: 5,
              description: 'Number of results to return (default: 5)',
            },
            store: {
              type: 'string',
              default: DEFAULT_STORE,
              description: `Store name (default: ${DEFAULT_STORE})`,
            },
          },
          required: ['query'],
          additionalProperties: false,
        },
      },
      {
        name: 'entities',
        description: `List all entities in the knowledge graph. Returns entity names, types (Person, Software, Concept, etc.), and IDs.`,
        inputSchema: {
          type: 'object',
          properties: {
            store: {
              type: 'string',
              default: DEFAULT_STORE,
              description: `Store name (default: ${DEFAULT_STORE})`,
            },
          },
          additionalProperties: false,
        },
      },
      {
        name: 'relations',
        description: `Get all relations involving an entity. Shows how entities connect (e.g., "uses", "implements", "authored_by").`,
        inputSchema: {
          type: 'object',
          properties: {
            entity: {
              type: 'string',
              minLength: 1,
              description: 'Entity name to get relations for',
            },
            store: {
              type: 'string',
              default: DEFAULT_STORE,
              description: `Store name (default: ${DEFAULT_STORE})`,
            },
          },
          required: ['entity'],
          additionalProperties: false,
        },
      },
      {
        name: 'communities',
        description: `List detected communities (clusters of related entities) with their summaries. Useful for understanding topic areas.`,
        inputSchema: {
          type: 'object',
          properties: {
            store: {
              type: 'string',
              default: DEFAULT_STORE,
              description: `Store name (default: ${DEFAULT_STORE})`,
            },
          },
          additionalProperties: false,
        },
      },
      {
        name: 'stats',
        description: `Get statistics about a knowledge graph store (chunk count, entity count, relation count, dimension).`,
        inputSchema: {
          type: 'object',
          properties: {
            store: {
              type: 'string',
              default: DEFAULT_STORE,
              description: `Store name (default: ${DEFAULT_STORE})`,
            },
          },
          additionalProperties: false,
        },
      },
    ],
  };
});

// Handle Tool Calls
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  try {
    const { name, arguments: args } = request.params;

    if (name === 'recall') {
      const query = args?.query || '';
      const top = args?.top || 5;
      const store = args?.store || DEFAULT_STORE;

      const results = await searchMemory(query, store, top);
      const text = formatSearchResults(results);

      return {
        content: [{ type: 'text', text }],
      };
    }

    if (name === 'entities') {
      const store = args?.store || DEFAULT_STORE;
      const entities = await listEntities(store);

      // Format as markdown table
      let text = `## Entities in "${store}"\n\n`;
      text += `| Name | Type |\n|------|------|\n`;
      for (const e of entities) {
        text += `| ${e.name} | ${e.type || '-'} |\n`;
      }
      text += `\n*Total: ${entities.length} entities*`;

      return {
        content: [{ type: 'text', text }],
      };
    }

    if (name === 'relations') {
      const store = args?.store || DEFAULT_STORE;
      const entity = args?.entity || '';

      const relations = await getRelations(store, entity);

      let text = `## Relations for "${entity}"\n\n`;
      text += `| Direction | Relation | Other Entity | Type |\n|-----------|----------|--------------|------|\n`;
      for (const r of relations) {
        text += `| ${r.direction} | ${r.relation} | ${r.other_entity} | ${r.other_type || '-'} |\n`;
      }
      text += `\n*Total: ${relations.length} relations*`;

      return {
        content: [{ type: 'text', text }],
      };
    }

    if (name === 'communities') {
      const store = args?.store || DEFAULT_STORE;
      const communities = await listCommunities(store);

      let text = `## Communities in "${store}"\n\n`;
      for (const c of communities) {
        text += `### Community ${c.id} (Level ${c.level})\n`;
        text += `- **Size**: ${c.size} entities\n`;
        if (c.parent_id) {
          text += `- **Parent**: ${c.parent_id}\n`;
        }
        if (c.summary) {
          text += `- **Summary**: ${c.summary}\n`;
        }
        text += `- **Entities**: ${c.entities.join(', ')}\n\n`;
      }

      return {
        content: [{ type: 'text', text }],
      };
    }

    if (name === 'stats') {
      const store = args?.store || DEFAULT_STORE;
      const stats = await getStats(store);

      const text = `## Store "${store}" Statistics\n\n` +
        `- **Dimension**: ${stats.dim}\n` +
        `- **Chunks**: ${stats.chunks}\n` +
        `- **Entities**: ${stats.entities}\n` +
        `- **Relations**: ${stats.relations}\n`;

      return {
        content: [{ type: 'text', text }],
      };
    }

    throw new Error(`Unknown tool: ${name}`);
  } catch (error) {
    return {
      content: [
        {
          type: 'text',
          text: `Error: ${error instanceof Error ? error.message : String(error)}`,
        },
      ],
      isError: true,
    };
  }
});

// Main
async function main() {
  console.error('GraphRAG MCP server running via stdio');
  const transport = new StdioServerTransport();
  await server.connect(transport);
}

main().catch((error) => {
  console.error('Server error:', error);
  process.exit(1);
});
