import {
	defaultLocalMaxTokens,
	isLocalConcurrency,
	isLocalMaxTokens,
	isLocalModelContextWindow,
	isLocalRetries,
	isLocalTimeoutSeconds,
	LOCAL_MODEL_CONCURRENCY_LIMITS,
	LOCAL_MODEL_CONTEXT_LIMITS,
	LOCAL_MODEL_RETRY_LIMITS,
	LOCAL_MODEL_SERVER_KINDS,
	LOCAL_MODEL_TIMEOUT_LIMITS,
	type LocalModelEndpoint,
	type LocalModelEndpointDraft,
	type LocalModelEntry,
	type LocalModelProbe,
	type LocalModelProbedModel,
	type LocalModelProbeRequest,
	type LocalModelServerKind,
	type LocalModelTestResult,
} from "@thinkrail/contracts";
import { applyEdits, getNodeValue, modify, type Node } from "jsonc-parser";
import { usePiRuntime } from "./agentSessionManager";
import { readLocalEndpointSettings, writeLocalEndpointSettings } from "./localModelSettings";
import {
	MODELS_JSON_FORMATTING,
	modelsPath,
	parseModelsConfig,
	publishModelsFileChange,
	readModelsSource,
	serializeModelsWrite,
	splitModelsSource,
	writeModelsFile,
} from "./modelsFile";

const API = "openai-completions";
const NO_KEY = "thinkrail-local";
const PROBE_TIMEOUT_MS = 5_000;
const MAX_DETAIL_PROBES = 64;
const ID_PREFIX = "local-";
const ZERO_COST = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 };
const LOCAL_COMPAT = {
	supportsStore: false,
	supportsDeveloperRole: false,
	supportsReasoningEffort: false,
	supportsStrictMode: false,
	maxTokensField: "max_tokens",
} as const;
const THINKING_TEMPLATE_KINDS: ReadonlySet<LocalModelServerKind> = new Set(["llama.cpp", "vllm"]);

type Json = Record<string, unknown>;

function isObject(value: unknown): value is Json {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

function positiveInt(value: unknown): number | null {
	return typeof value === "number" && Number.isSafeInteger(value) && value > 0 ? value : null;
}

export function normalizeLocalBaseUrl(kind: LocalModelServerKind, raw: string): string {
	let url: URL;
	try {
		url = new URL(raw.trim());
	} catch {
		throw new Error("Enter a full server URL, such as http://127.0.0.1:8080/v1");
	}
	if (url.protocol !== "http:" && url.protocol !== "https:")
		throw new Error("The server URL must start with http:// or https://");
	url.search = "";
	url.hash = "";
	let path = url.pathname.replace(/\/+$/, "");
	path = path.replace(/\/(chat\/completions|completions|models)$/, "");
	if (kind !== "openai-compatible" && !/\/v1$/.test(path)) path = `${path}/v1`;
	url.pathname = path || "/";
	return url.toString().replace(/\/+$/, "");
}

function serverRoot(baseUrl: string): string {
	return baseUrl.replace(/\/v1$/, "");
}

/** Resolves a probe key the way pi resolves `apiKey`, except `!command` keys, which a probe never runs. */
export function resolveProbeKey(apiKey: string | undefined): string | undefined {
	const key = apiKey?.trim();
	if (!key || key === NO_KEY || key.startsWith("!")) return undefined;
	const env = /^\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?$/.exec(key);
	if (env?.[1]) return process.env[env[1]] || undefined;
	return key;
}

async function fetchJson(
	url: string,
	apiKey: string | undefined,
	init: { method?: "GET" | "POST"; body?: unknown } = {},
): Promise<unknown> {
	const headers: Record<string, string> = { accept: "application/json" };
	if (apiKey) headers.authorization = `Bearer ${apiKey}`;
	if (init.body !== undefined) headers["content-type"] = "application/json";
	const response = await fetch(url, {
		method: init.method ?? "GET",
		headers,
		...(init.body === undefined ? {} : { body: JSON.stringify(init.body) }),
		signal: AbortSignal.timeout(PROBE_TIMEOUT_MS),
	});
	if (!response.ok) throw new Error(`HTTP ${response.status}`);
	return response.json();
}

function llamaContext(entry: Json): number | null {
	const meta = isObject(entry.meta) ? entry.meta : {};
	const runtime = positiveInt(meta.n_ctx);
	if (runtime) return runtime;
	const status = isObject(entry.status) ? entry.status : {};
	const args = Array.isArray(status.args) ? status.args : [];
	for (let index = 0; index < args.length - 1; index++) {
		if (args[index] !== "-c" && args[index] !== "--ctx-size" && args[index] !== "-ctx") continue;
		const value = positiveInt(Number(args[index + 1]));
		if (value) return value;
	}
	return positiveInt(meta.n_ctx_train);
}

function modalities(entry: Json, key: "input_modalities" | "output_modalities"): string[] {
	const architecture = isObject(entry.architecture) ? entry.architecture : {};
	const value = architecture[key];
	return Array.isArray(value)
		? value.filter((item): item is string => typeof item === "string")
		: [];
}

/** Reads an OpenAI `GET /models` listing, plus the context hints vLLM and llama.cpp attach to it. */
export function parseModelList(payload: unknown): LocalModelProbedModel[] {
	const data = isObject(payload) && Array.isArray(payload.data) ? payload.data : [];
	const seen = new Set<string>();
	const models: LocalModelProbedModel[] = [];
	for (const entry of data) {
		if (!isObject(entry) || typeof entry.id !== "string" || entry.id === "") continue;
		if (seen.has(entry.id)) continue;
		const output = modalities(entry, "output_modalities");
		if (output.length > 0 && !output.includes("text")) continue;
		seen.add(entry.id);
		models.push({
			id: entry.id,
			contextWindow: positiveInt(entry.max_model_len) ?? llamaContext(entry),
			reasoning: false,
			vision: modalities(entry, "input_modalities").includes("image"),
			tools: null,
		});
	}
	return models;
}

/** llama.cpp `/props`: per-slot context, slot count, and what the loaded chat template supports. */
async function enrichLlamaCpp(
	root: string,
	key: string | undefined,
	models: LocalModelProbedModel[],
): Promise<{ models: LocalModelProbedModel[]; slots: number | null }> {
	const props = await fetchJson(`${root}/props`, key).catch(() => undefined);
	if (!isObject(props)) return { models, slots: null };
	const settings = isObject(props.default_generation_settings)
		? props.default_generation_settings
		: {};
	const context = positiveInt(settings.n_ctx);
	const reasoning =
		typeof props.chat_template === "string" && props.chat_template.includes("enable_thinking");
	const caps = isObject(props.chat_template_caps) ? props.chat_template_caps : {};
	const tools = typeof caps.supports_tools === "boolean" ? caps.supports_tools : null;
	return {
		slots: positiveInt(props.total_slots),
		models: models.map((model) => ({
			...model,
			contextWindow: model.contextWindow ?? context,
			reasoning: model.reasoning || reasoning,
			tools: model.tools ?? tools,
		})),
	};
}

async function enrichOllama(
	root: string,
	key: string | undefined,
	models: LocalModelProbedModel[],
) {
	return Promise.all(
		models.map(async (model, index) => {
			if (index >= MAX_DETAIL_PROBES) return model;
			const show = await fetchJson(`${root}/api/show`, key, {
				method: "POST",
				body: { model: model.id },
			}).catch(() => undefined);
			if (!isObject(show)) return model;
			const info = isObject(show.model_info) ? show.model_info : {};
			const contextKey = Object.keys(info).find((name) => name.endsWith(".context_length"));
			const capabilities = Array.isArray(show.capabilities) ? show.capabilities : [];
			return {
				...model,
				contextWindow: model.contextWindow ?? (contextKey ? positiveInt(info[contextKey]) : null),
				reasoning: capabilities.includes("thinking"),
				vision: model.vision || capabilities.includes("vision"),
				tools: capabilities.length > 0 ? capabilities.includes("tools") : model.tools,
			};
		}),
	);
}

async function enrichLmStudio(
	root: string,
	key: string | undefined,
	models: LocalModelProbedModel[],
) {
	const listing = await fetchJson(`${root}/api/v0/models`, key).catch(() => undefined);
	const data = isObject(listing) && Array.isArray(listing.data) ? listing.data : [];
	const details = new Map<string, Json>();
	for (const entry of data)
		if (isObject(entry) && typeof entry.id === "string") details.set(entry.id, entry);
	return models.flatMap((model) => {
		const detail = details.get(model.id);
		if (!detail) return [model];
		if (detail.type === "embeddings") return [];
		return [
			{
				...model,
				contextWindow:
					model.contextWindow ??
					positiveInt(detail.loaded_context_length) ??
					positiveInt(detail.max_context_length),
				vision: model.vision || detail.type === "vlm",
				tools: Array.isArray(detail.capabilities)
					? detail.capabilities.includes("tool_use")
					: model.tools,
			},
		];
	});
}

function describeFailure(error: unknown): string {
	if (error instanceof Error && error.name === "TimeoutError") return "timed out";
	if (error instanceof Error && /^HTTP \d+$/.test(error.message)) return error.message;
	return "connection failed";
}

function readEndpointKey(id: string): string | undefined {
	const config = parseModelsConfig(readModelsSource(modelsPath()));
	const provider = config && providerValue(config, id);
	return provider && typeof provider.apiKey === "string" ? provider.apiKey : undefined;
}

export async function probeLocalModels(request: LocalModelProbeRequest): Promise<LocalModelProbe> {
	if (!LOCAL_MODEL_SERVER_KINDS.includes(request.kind)) throw new Error("Unknown server kind");
	const baseUrl = normalizeLocalBaseUrl(request.kind, request.baseUrl);
	const rawKey =
		request.apiKey === undefined && request.endpointId
			? readEndpointKey(request.endpointId)
			: request.apiKey;
	const key = resolveProbeKey(rawKey);
	let listing: unknown;
	try {
		listing = await fetchJson(`${baseUrl}/models`, key);
	} catch (error) {
		throw new Error(`Couldn't list models at ${baseUrl}/models (${describeFailure(error)}).`);
	}
	let models = parseModelList(listing);
	let slots: number | null = null;
	const root = serverRoot(baseUrl);
	if (request.kind === "llama.cpp") ({ models, slots } = await enrichLlamaCpp(root, key, models));
	else if (request.kind === "ollama") models = await enrichOllama(root, key, models);
	else if (request.kind === "lmstudio") models = await enrichLmStudio(root, key, models);
	return { baseUrl, slots, models };
}

function providerValue(config: Node, id: string): Json | undefined {
	const root = getNodeValue(config);
	const providers = isObject(root) && isObject(root.providers) ? root.providers : {};
	const provider = providers[id];
	return isObject(provider) ? provider : undefined;
}

function toEntry(model: unknown): LocalModelEntry | undefined {
	if (!isObject(model) || typeof model.id !== "string" || model.id === "") return undefined;
	const input = Array.isArray(model.input) ? model.input : [];
	const contextWindow = positiveInt(model.contextWindow) ?? LOCAL_MODEL_CONTEXT_LIMITS.fallback;
	return {
		id: model.id,
		contextWindow,
		maxTokens: positiveInt(model.maxTokens) ?? defaultLocalMaxTokens(contextWindow),
		reasoning: model.reasoning === true,
		vision: input.includes("image"),
	};
}

/** Every models.json provider that serves chat completions from its own base URL. */
export function readLocalEndpoints(config: Node | undefined): LocalModelEndpoint[] {
	if (!config) return [];
	const root = getNodeValue(config);
	const providers = isObject(root) && isObject(root.providers) ? root.providers : {};
	const settings = readLocalEndpointSettings();
	return Object.entries(providers).flatMap(([id, provider]) => {
		if (!isObject(provider) || typeof provider.baseUrl !== "string") return [];
		if (provider.api !== API || !Array.isArray(provider.models)) return [];
		const models = provider.models.flatMap((model) => toEntry(model) ?? []);
		const apiKey = typeof provider.apiKey === "string" ? provider.apiKey : undefined;
		return [
			{
				id,
				name: typeof provider.name === "string" && provider.name ? provider.name : id,
				baseUrl: provider.baseUrl,
				hasApiKey: apiKey !== undefined && apiKey !== NO_KEY,
				kind: settings[id]?.kind ?? null,
				maxConcurrency: settings[id]?.maxConcurrency ?? null,
				timeoutSeconds: settings[id]?.timeoutSeconds ?? null,
				maxRetries: settings[id]?.maxRetries ?? null,
				models,
			},
		];
	});
}

export function listLocalEndpoints(): Promise<LocalModelEndpoint[]> {
	return Promise.resolve(readLocalEndpoints(parseModelsConfig(readModelsSource(modelsPath()))));
}

function slug(name: string): string {
	return (
		name
			.toLowerCase()
			.normalize("NFKD")
			.replace(/[^a-z0-9.]+/g, "-")
			.replace(/^-+|-+$/g, "")
			.slice(0, 40) || "server"
	);
}

function newProviderId(name: string, config: Node | undefined): string {
	const root = config ? getNodeValue(config) : undefined;
	const taken = new Set(
		isObject(root) && isObject(root.providers) ? Object.keys(root.providers) : [],
	);
	const base = `${ID_PREFIX}${slug(name)}`;
	let id = base;
	for (let n = 2; taken.has(id); n++) id = `${base}-${n}`;
	return id;
}

function validateDraft(draft: LocalModelEndpointDraft): void {
	if (!LOCAL_MODEL_SERVER_KINDS.includes(draft.kind)) throw new Error("Unknown server kind");
	if (!draft.name.trim()) throw new Error("Name the server.");
	if (draft.models.length === 0) throw new Error("Pick at least one model.");
	if (draft.maxConcurrency !== null && !isLocalConcurrency(draft.maxConcurrency))
		throw new Error(
			`Parallel requests must be a whole number between ${LOCAL_MODEL_CONCURRENCY_LIMITS.min} and ${LOCAL_MODEL_CONCURRENCY_LIMITS.max}, or no limit.`,
		);
	if (draft.timeoutSeconds !== null && !isLocalTimeoutSeconds(draft.timeoutSeconds))
		throw new Error(
			`The timeout must be a whole number of seconds between ${LOCAL_MODEL_TIMEOUT_LIMITS.min} and ${LOCAL_MODEL_TIMEOUT_LIMITS.max}, or pi's default.`,
		);
	if (draft.maxRetries !== null && !isLocalRetries(draft.maxRetries))
		throw new Error(
			`Retries must be a whole number between ${LOCAL_MODEL_RETRY_LIMITS.min} and ${LOCAL_MODEL_RETRY_LIMITS.max}, or pi's default.`,
		);
	const ids = new Set<string>();
	for (const model of draft.models) {
		if (!model.id || ids.has(model.id)) throw new Error("Model ids must be unique and non-empty.");
		ids.add(model.id);
		if (!isLocalModelContextWindow(model.contextWindow))
			throw new Error(
				`Context windows must be whole numbers between ${LOCAL_MODEL_CONTEXT_LIMITS.min.toLocaleString("en-US")} and ${LOCAL_MODEL_CONTEXT_LIMITS.max.toLocaleString("en-US")} tokens.`,
			);
		if (!isLocalMaxTokens(model.maxTokens, model.contextWindow))
			throw new Error(
				"Max output must be a whole number of at least 256 tokens and fit the context.",
			);
	}
}

function modelConfig(
	kind: LocalModelServerKind,
	model: LocalModelEntry,
	previous: Json | undefined,
): Json {
	const thinking = model.reasoning && THINKING_TEMPLATE_KINDS.has(kind);
	const next: Json = {
		...(previous ?? {}),
		id: model.id,
		name: typeof previous?.name === "string" ? previous.name : model.id,
		reasoning: model.reasoning,
		input: model.vision ? ["text", "image"] : ["text"],
		cost: isObject(previous?.cost) ? previous.cost : ZERO_COST,
		contextWindow: model.contextWindow,
		maxTokens: model.maxTokens,
	};
	if (thinking) {
		next.thinkingLevelMap = {
			off: "off",
			minimal: null,
			low: null,
			medium: "medium",
			high: null,
			xhigh: null,
		};
		next.compat = {
			...(isObject(previous?.compat) ? previous.compat : {}),
			thinkingFormat: "qwen-chat-template",
		};
	} else if (!model.reasoning) {
		delete next.thinkingLevelMap;
		if (isObject(next.compat)) {
			const { thinkingFormat: _dropped, ...rest } = next.compat;
			if (Object.keys(rest).length > 0) next.compat = rest;
			else delete next.compat;
		}
	}
	return next;
}

export function buildProviderConfig(
	draft: LocalModelEndpointDraft,
	previous: Json | undefined,
): Json {
	const baseUrl = normalizeLocalBaseUrl(draft.kind, draft.baseUrl);
	const previousKey = typeof previous?.apiKey === "string" ? previous.apiKey : undefined;
	const apiKey =
		draft.apiKey === undefined ? (previousKey ?? NO_KEY) : draft.apiKey.trim() || NO_KEY;
	const previousModels = new Map<string, Json>();
	if (Array.isArray(previous?.models)) {
		for (const model of previous.models)
			if (isObject(model) && typeof model.id === "string") previousModels.set(model.id, model);
	}
	return {
		...(previous ?? {}),
		name: draft.name.trim(),
		baseUrl,
		api: API,
		apiKey,
		compat: { ...LOCAL_COMPAT, ...(isObject(previous?.compat) ? previous.compat : {}) },
		models: draft.models.map((model) =>
			modelConfig(draft.kind, model, previousModels.get(model.id)),
		),
	};
}

async function commit(
	edit: (body: string, config: Node | undefined) => { body: string; settings: () => void },
) {
	return usePiRuntime(async (runtime) => {
		const path = modelsPath();
		const source = readModelsSource(path);
		const config = parseModelsConfig(source);
		const { bom, body } = splitModelsSource(source);
		const result = edit(body, config);
		const next = bom + result.body;
		if (next !== source) writeModelsFile(path, next);
		result.settings();
		if (next !== source) {
			await runtime.refresh({ allowNetwork: false });
			publishModelsFileChange();
		}
		return readLocalEndpoints(parseModelsConfig(next));
	});
}

export function saveLocalEndpoint(draft: LocalModelEndpointDraft): Promise<LocalModelEndpoint[]> {
	try {
		validateDraft(draft);
	} catch (error) {
		return Promise.reject(error);
	}
	return serializeModelsWrite(() =>
		commit((body, config) => {
			const existing = draft.id && config ? providerValue(config, draft.id) : undefined;
			if (draft.id && (!existing || !readLocalEndpoints(config).some((e) => e.id === draft.id)))
				throw new Error("That server is no longer in models.json. Reload and retry.");
			const id = draft.id ?? newProviderId(draft.name, config);
			const value = buildProviderConfig(draft, existing);
			return {
				body: applyEdits(body, modify(body, ["providers", id], value, MODELS_JSON_FORMATTING)),
				settings: () =>
					writeLocalEndpointSettings(id, {
						kind: draft.kind,
						...(draft.maxConcurrency === null ? {} : { maxConcurrency: draft.maxConcurrency }),
						...(draft.timeoutSeconds === null ? {} : { timeoutSeconds: draft.timeoutSeconds }),
						...(draft.maxRetries === null ? {} : { maxRetries: draft.maxRetries }),
					}),
			};
		}),
	);
}

export function removeLocalEndpoint(id: string): Promise<LocalModelEndpoint[]> {
	return serializeModelsWrite(() =>
		commit((body, config) => {
			if (!readLocalEndpoints(config).some((endpoint) => endpoint.id === id))
				return { body, settings: () => {} };
			return {
				body: applyEdits(body, modify(body, ["providers", id], undefined, MODELS_JSON_FORMATTING)),
				settings: () => writeLocalEndpointSettings(id, null),
			};
		}),
	);
}

const TEST_TIMEOUT_MS = 180_000;
const TEST_TOOL = {
	name: "get_current_time",
	description: "Returns the current time in the given IANA timezone.",
	parameters: {
		type: "object",
		properties: { timezone: { type: "string", description: "IANA timezone, e.g. UTC" } },
		required: ["timezone"],
		additionalProperties: false,
	},
};

function failureText(message: { stopReason: string; errorMessage?: string }): string | null {
	if (message.stopReason !== "error" && message.stopReason !== "aborted") return null;
	return message.errorMessage?.trim() || "The request failed without an error message.";
}

/**
 * Runs the two requests an agent turn depends on through pi's own dispatch — so models.json compat,
 * credentials and the endpoint's parallel-request limit all apply — and reports each outcome.
 */
export function testLocalModel(id: string, modelId: string): Promise<LocalModelTestResult> {
	return usePiRuntime(async (runtime) => {
		await runtime.refresh({ allowNetwork: false });
		const model = runtime.getAvailableSnapshot().find((m) => m.provider === id && m.id === modelId);
		if (!model) throw new Error("That model is not available. Save the server and retry.");
		const saved = readLocalEndpointSettings()[id];
		const budgetMs =
			Math.max(TEST_TIMEOUT_MS, (saved?.timeoutSeconds ?? 0) * 1_000) *
			((saved?.maxRetries ?? 0) + 1);
		const options = { maxTokens: 256, signal: AbortSignal.timeout(budgetMs) };
		const started = performance.now();
		const chat = await runtime.completeSimple(
			model,
			{
				systemPrompt: "You are a connectivity check. Reply with the single word OK.",
				messages: [{ role: "user", content: "Reply with OK.", timestamp: Date.now() }],
			},
			options,
		);
		const elapsed = performance.now() - started;
		const chatFailure = failureText(chat);
		if (chatFailure)
			return {
				chat: "failed",
				tools: "skipped",
				latencyMs: Math.round(elapsed),
				outputTokensPerSecond: null,
				message: chatFailure,
			};
		const seconds = elapsed / 1000;
		const outputTokensPerSecond =
			chat.usage.output > 0 && seconds > 0
				? Math.round((chat.usage.output / seconds) * 10) / 10
				: null;
		const agent = await runtime.completeSimple(
			model,
			{
				systemPrompt:
					"You are an agent with tools. When a tool can answer, call it instead of replying in text.",
				messages: [
					{ role: "user", content: "What time is it in UTC? Use the tool.", timestamp: Date.now() },
				],
				tools: [TEST_TOOL as never],
			},
			options,
		);
		const toolFailure = failureText(agent);
		const called = agent.content.some(
			(block) => block.type === "toolCall" && block.name === TEST_TOOL.name,
		);
		return {
			chat: "ok",
			tools: called ? "ok" : "failed",
			latencyMs: Math.round(elapsed),
			outputTokensPerSecond,
			message: toolFailure,
		};
	});
}
