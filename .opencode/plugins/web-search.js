/**
 * Nova web search — opencode plugin.
 *
 * Registers a `web_search` tool backed by Grok's agentic search
 * (`/v1/responses`), reached through Sidekick's local injector proxy so no auth
 * token is handled here: the proxy strips the dummy key and injects the user's
 * Nova bearer per request.
 *
 * Credit: the search core (request body, SSE parse, early-abort, web/X modes,
 * tool schema, and output shape) is adapted from the GenAI team's
 * `tesla-web-search` MCP server (v3.1.4) —
 * https://github-it.tesla.com/bottle-rocket/claude-code-tesla/tree/main/plugins/tesla-web-search
 * This port swaps its transport (stdio JSON-RPC → opencode plugin tool) and its
 * auth (env token → dummy key replaced by the injector proxy), and adds
 * injector-URL discovery. Thanks to bottle-rocket / the GenAI team.
 *
 * @typedef {{
 *   query: string,
 *   max_searches?: number,
 *   from_date?: string,
 *   to_date?: string,
 *   twitter?: boolean,
 *   allowed_domains?: string[],
 *   excluded_domains?: string[],
 *   timeout_ms?: number,
 * }} SearchArgs
 */

// The injector's OpenAI-compatible route prepends the gateway's `/openai`
// prefix, which already maps to xAI's `/v1` base — so the suffix is `/responses`
// (NOT `/v1/responses`, which would double the `v1` and 404). A direct call to
// inference.tesla.com uses `/v1/responses`; through the injector it's `/responses`.
const RESPONSES_PATH = "/responses";
const DEFAULT_MODEL = "grok-build-latest";
const DEFAULT_MAX_SEARCHES = 1;
const DEFAULT_REQUEST_TIMEOUT_MS =
  Number.parseInt(process.env.NOVA_WEB_SEARCH_TIMEOUT_MS || "", 10) || 60_000;

function buildInstructions() {
  const today = new Date().toISOString().split("T")[0];
  return `Today is ${today}. Find web sources for the user's query with one focused search.`;
}

function buildXInstructions() {
  const today = new Date().toISOString().split("T")[0];
  return (
    `Today is ${today}. Search X/Twitter for the user's query and synthesize what people are saying — ` +
    "notable claims, sentiment, key accounts — citing the posts you use."
  );
}

// --- Injector-proxy + model discovery ---------------------------------------
//
// The injector's loopback URL is ephemeral (rewritten on every catalog reload),
// so it's resolved per call rather than cached across the plugin's lifetime.
// Order: a URL captured from a live model request (exact, if one has fired) →
// the Grok provider's baseURL from opencode's own config.

/** @returns {boolean} true for a Grok provider/model id. */
function isGrok(id) {
  return typeof id === "string" && /grok/i.test(id);
}

/**
 * Finds the Grok provider in opencode's resolved config. Returns the provider's
 * injector baseURL and the Grok model id to search with.
 *
 * @param {any} client opencode SDK client from the plugin context.
 * @returns {Promise<{ baseURL: string, model: string }>}
 */
async function resolveGrokProvider(client) {
  const res = await client.config.providers();
  const providers = res?.data?.providers ?? res?.providers ?? [];

  for (const p of providers) {
    const models = p?.models ?? {};
    const grokModel = Object.keys(models).find(isGrok);
    const baseURL = p?.options?.baseURL;
    if (grokModel && baseURL) return { baseURL, model: grokModel };
  }
  throw new Error(
    "No Grok provider found in the agent's configuration; web search is unavailable."
  );
}

/**
 * Resolves the model id: explicit env override → discovered Grok model →
 * hardcoded default. The bundle owns the default; the desktop overrides via
 * NOVA_WEB_SEARCH_MODEL on the process env.
 *
 * @param {string} discovered model id found in the Grok provider.
 * @returns {string}
 */
function resolveModel(discovered) {
  return process.env.NOVA_WEB_SEARCH_MODEL || discovered || DEFAULT_MODEL;
}

// --- Search ------------------------------------------------------------------

/**
 * Returns source URLs (web) or a synthesized answer (X) for the query.
 *
 * Web mode: the `web_search` tool exposes its hits inline via
 * `web_search_call.action.sources`, so we grab the URLs and abort before the
 * model synthesizes (~1.5s). URLs are fetchable, so the agent reads them itself.
 *
 * X mode (`twitter: true`): X posts are login-gated, so their URLs are useless
 * as fetch targets. The only usable artifact is the model's synthesis; we run
 * to completion and return the answer plus citations (~5-10s).
 *
 * @param {SearchArgs} args
 * @param {string} endpoint injector baseURL + /v1/responses
 * @param {string} model Grok model id
 * @returns {Promise<object>}
 */
async function search(args, endpoint, model) {
  const maxSearches = Math.max(1, Math.min(5, args.max_searches ?? DEFAULT_MAX_SEARCHES));
  const timeoutMs = Math.max(1, args.timeout_ms ?? DEFAULT_REQUEST_TIMEOUT_MS);
  const twitter = !!args.twitter;

  /** @type {Record<string, any>} */
  const tool = { type: twitter ? "x_search" : "web_search" };
  if (args.from_date) tool.from_date = args.from_date;
  if (args.to_date) tool.to_date = args.to_date;
  if (!twitter && args.allowed_domains?.length) tool.allowed_domains = args.allowed_domains;
  if (!twitter && args.excluded_domains?.length) tool.excluded_domains = args.excluded_domains;

  const body = {
    model,
    store: false,
    stream: true,
    instructions: twitter ? buildXInstructions() : buildInstructions(),
    input: [{ role: "user", content: args.query }],
    tools: [tool],
    tool_choice: "required",
  };

  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(endpoint, {
      method: "POST",
      headers: {
        // Dummy key: the injector strips it and injects the real Nova bearer.
        Authorization: "Bearer no-key-needed",
        "Content-Type": "application/json",
        Accept: "text/event-stream",
      },
      body: JSON.stringify(body),
      signal: ctrl.signal,
    });
    if (!res.ok) {
      throw new Error("HTTP " + res.status + ": " + (await res.text()).slice(0, 300));
    }

    /** @type {Array<{source: string, query: string, urls: string[]}>} */
    const searches = [];
    const seen = new Set();
    const dec = new TextDecoder();
    let buf = "";
    let summary = "";

    outer: for await (const chunk of res.body) {
      buf += dec.decode(chunk, { stream: true });
      let i;
      while ((i = buf.indexOf("\n\n")) >= 0) {
        const block = buf.slice(0, i);
        buf = buf.slice(i + 2);
        const dm = block.match(/data: (\{[\s\S]*)/);
        if (!dm) continue;
        let d;
        try {
          d = JSON.parse(dm[1]);
        } catch {
          continue;
        }

        if (!twitter) {
          // Web: hits arrive inline before synthesis — grab and bail.
          if (d.type === "response.output_item.done" && d.item && d.item.type === "web_search_call") {
            const action = d.item.action || {};
            const urls = (action.sources || []).map((s) => s.url).filter(Boolean);
            const fresh = urls.filter((u) => !seen.has(u));
            fresh.forEach((u) => seen.add(u));
            searches.push({ source: "web", query: action.query || "", urls: fresh });
            if (searches.length >= maxSearches) break outer;
          }
        } else {
          // X: posts are login-gated; the synthesized answer is the deliverable.
          if (d.type === "response.output_text.delta") {
            summary += d.delta || "";
          } else if (d.type === "response.output_text.annotation.added") {
            const u = d.annotation && d.annotation.url;
            if (u) seen.add(u);
          } else if (d.type === "response.custom_tool_call_input.done") {
            let q = d.input;
            try {
              q = JSON.parse(d.input).query || d.input;
            } catch {
              /* keep raw */
            }
            searches.push({ source: "x", query: q || "", urls: [] });
          }
        }

        if (d.type === "response.completed") break outer;
        if (d.type === "response.failed" || d.type === "response.incomplete") {
          const r = d.response || {};
          const detail = (r.incomplete_details && r.incomplete_details.reason) || r.error;
          throw new Error(
            "Response " + d.type + (detail ? ": " + (typeof detail === "string" ? detail : JSON.stringify(detail)) : "")
          );
        }
        if (d.type === "error" || d.error) {
          throw new Error(typeof d.error === "string" ? d.error : JSON.stringify(d.error || d));
        }
      }
    }

    if (twitter) {
      const citations = [...seen];
      const note = "X posts are login-gated and unfetchable. Use the summary; don't WebFetch the citations.";
      return { result_type: "synthesized_answer", source: "x", note, query: args.query, summary: summary.trim(), citations, searches };
    }

    const urls = [...seen];
    const note = urls.length
      ? "Source URLs only (not fetched). WebFetch the relevant ones to read them."
      : "No results — try a broader or reworded query.";
    return { result_type: "source_urls", source: "web", note, query: args.query, urls, searches };
  } catch (e) {
    if (e.name === "AbortError") {
      throw new Error("Search timed out after " + timeoutMs + "ms. Retry with a larger timeout_ms.");
    }
    throw e;
  } finally {
    clearTimeout(timer);
    ctrl.abort();
  }
}

const TOOL_DESCRIPTION =
  "Fast web search. Returns source URLs (not page content) — WebFetch the relevant ones to read them. " +
  "Set twitter:true to search X/Twitter instead (returns a synthesized summary, since X posts can't be fetched).";

// --- Plugin ------------------------------------------------------------------
//
// Targets the @opencode-ai/plugin v1 API (novacode ships 1.17.x): a plugin is a
// factory `async ({ client }) => Hooks` that returns a `tool` map. Tools are
// built with the `tool()` helper + zod args (not JSON Schema). This is the same
// shape the working opencode-cron plugin uses.

/**
 * Builds the web_search tool bound to the opencode client.
 *
 * @param {any} tool the tool() helper from @opencode-ai/plugin.
 * @param {any} z zod (tool.schema).
 * @param {any} client opencode SDK client.
 */
function buildWebSearchTool(tool, z, client) {
  return tool({
    description: TOOL_DESCRIPTION,
    args: {
      query: z.string().describe("The search query."),
      max_searches: z
        .number()
        .int()
        .min(1)
        .max(5)
        .optional()
        .describe("Web only. Search rounds to aggregate (default 1, fastest). Higher = broader, slower."),
      from_date: z
        .string()
        .regex(/^\d{4}-\d{2}-\d{2}$/)
        .optional()
        .describe("Earliest source date (YYYY-MM-DD)."),
      to_date: z
        .string()
        .regex(/^\d{4}-\d{2}-\d{2}$/)
        .optional()
        .describe("Latest source date (YYYY-MM-DD)."),
      twitter: z
        .boolean()
        .optional()
        .describe("Search X/Twitter instead of the web. Returns a synthesized summary + citations (X is login-gated). Slower (~5-10s)."),
      allowed_domains: z
        .array(z.string())
        .optional()
        .describe('Web only. Restrict to these domains (e.g., ["github.com", "arxiv.org"]).'),
      excluded_domains: z.array(z.string()).optional().describe("Web only. Exclude these domains."),
      timeout_ms: z.number().int().min(1).optional().describe("Client-side timeout in ms (default 60000)."),
    },
    async execute(args) {
      const grok = await resolveGrokProvider(client);
      const model = resolveModel(grok.model);
      const endpoint = grok.baseURL.replace(/\/+$/, "") + RESPONSES_PATH;

      const result = await search(args, endpoint, model);
      return JSON.stringify(result, null, 2);
    },
  });
}

/**
 * opencode plugin entrypoint (v1 API). File-discovered from
 * `.opencode/plugins/`; novacode resolves the `@opencode-ai/plugin` import from
 * its own node_modules at load time.
 */
const NovaWebSearchPlugin = async ({ client }) => {
  const { tool } = await import("@opencode-ai/plugin");
  const z = tool.schema;

  return {
    tool: {
      web_search: buildWebSearchTool(tool, z, client),
    },
  };
};

export default NovaWebSearchPlugin;
