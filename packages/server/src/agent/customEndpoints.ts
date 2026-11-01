import { randomBytes } from "node:crypto";
import { readFileSync, statSync } from "node:fs";
import {
	CUSTOM_ENDPOINT_HEADER_LIMITS,
	CUSTOM_ENDPOINT_ID_PREFIX,
	type CustomEndpoint,
	type CustomEndpointCost,
	type CustomEndpointDraft,
	type CustomEndpointHeader,
	type CustomEndpointKeyStatus,
	type CustomEndpointModel,
	type CustomEndpointNetwork,
	type CustomEndpointProbe,
	type CustomEndpointProbeFailure,
	type CustomEndpointProbeRequest,
	DEFAULT_ENDPOINT_NETWORK,
	isCustomEndpointApi,
	isCustomEndpointId,
	isEndpointConcurrency,
	isEndpointHeaderName,
	isEndpointHeaderValue,
	isEndpointProxyUrl,
	isEndpointRetries,
	isEndpointTag,
	isEndpointTimeoutSeconds,
} from "@thinkrail/contracts";
import { applyEdits, getNodeValue, modify, type Node } from "jsonc-parser";
import { logger } from "../log";
import { usePiRuntime } from "./agentSessionManager";
import {
	type Access,
	commandHash,
	commandToken,
	isReference,
	networkFetch,
	requestHeaders,
	resolveValue,
	sendsCredentials,
} from "./endpointAccess";
import {
	type EndpointLimits,
	endpointGatewayOf,
	endpointLimitsOf,
	endpointNetworkOf,
	nextEndpointSettings,
	readEndpointLimits,
	sanitizeNetwork,
	writeEndpointLimits,
} from "./endpointLimits";
import { applyLiteLlmModelInfo, fetchLiteLlmKeyStatus, fetchLiteLlmModelInfo } from "./liteLlm";
import { type ManagedServer, managedInfo, managedServer, readManagedConfig } from "./managedConfig";
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
import { forgetRequestGate, setEndpointDirectory } from "./requestLimits";

const log = logger("agent");

const API = "openai-completions";
const NO_KEY = "thinkrail-no-key";
const LIST_TIMEOUT_MS = 5_000;
const DETAIL_TIMEOUT_MS = 3_000;
const MAX_DETAIL_PROBES = 64;
const MAX_MODELS = 500;
const ZERO_COST = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 };
const COMPAT = {
	supportsStore: false,
	supportsDeveloperRole: false,
	maxTokensField: "max_tokens",
} as const;
const OUTPUT_LIMITS = { min: 256, max: 16_384 } as const;
const CHAT_MODES = new Set(["chat", "completion", "responses"]);
const MANAGED_MODEL_FIELDS = ["contextWindow", "maxTokens", "reasoning", "input"] as const;
const CHAT_TEMPLATE_THINKING = "qwen-chat-template";

type Json = Record<string, unknown>;

function isObject(value: unknown): value is Json {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

function positiveInt(value: unknown): number | null {
	return typeof value === "number" && Number.isSafeInteger(value) && value > 0 ? value : null;
}

function strings(value: unknown): string[] {
	return Array.isArray(value)
		? value.filter((item): item is string => typeof item === "string")
		: [];
}

export function parseBaseUrl(raw: string): string | null {
	let url: URL;
	try {
		url = new URL(raw.trim());
	} catch {
		return null;
	}
	if (url.protocol !== "http:" && url.protocol !== "https:") return null;
	if (url.username || url.password || url.search) return null;
	url.hash = "";
	url.pathname =
		url.pathname.replace(/\/+$/, "").replace(/\/(chat\/completions|completions|models)$/, "") ||
		"/";
	return url.toString().replace(/\/+$/, "");
}

export function defaultMaxTokens(contextWindow: number): number {
	return Math.max(OUTPUT_LIMITS.min, Math.min(OUTPUT_LIMITS.max, Math.floor(contextWindow / 4)));
}

const ENV_REFERENCE = /^\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))$/;

export function resolveProbeKey(apiKey: string | undefined): string | undefined {
	const key = apiKey?.trim();
	if (!key || key === NO_KEY || key.startsWith("!")) return undefined;
	const reference = ENV_REFERENCE.exec(key);
	if (!reference) return key;
	return process.env[reference[1] ?? reference[2] ?? ""] || undefined;
}

const CERTIFICATE_ERRORS = new Set([
	"SELF_SIGNED_CERT_IN_CHAIN",
	"DEPTH_ZERO_SELF_SIGNED_CERT",
	"UNABLE_TO_VERIFY_LEAF_SIGNATURE",
	"UNABLE_TO_GET_ISSUER_CERT",
	"UNABLE_TO_GET_ISSUER_CERT_LOCALLY",
	"CERT_UNTRUSTED",
	"CERT_HAS_EXPIRED",
	"ERR_TLS_CERT_ALTNAME_INVALID",
]);

function fetchFailure(error: unknown): CustomEndpointProbeFailure {
	if (!(error instanceof Error)) return "unreachable";
	if (error.name === "TimeoutError") return "timeout";
	const code = "code" in error && typeof error.code === "string" ? error.code : "";
	if (CERTIFICATE_ERRORS.has(code) || /certificate/i.test(error.message)) return "certificate";
	return "unreachable";
}

type Fetched =
	| { ok: true; json: unknown }
	| { ok: false; failure: CustomEndpointProbeFailure; status?: number };

async function fetchJson(
	url: string,
	access: Access,
	timeout: number,
	body?: unknown,
): Promise<Fetched> {
	const headers = requestHeaders(access, {
		accept: "application/json",
		...(body === undefined ? {} : { "content-type": "application/json" }),
	});
	let response: Response;
	try {
		response = await access.fetch(url, {
			method: body === undefined ? "GET" : "POST",
			headers,
			...(body === undefined ? {} : { body: JSON.stringify(body) }),
			redirect: sendsCredentials(access) ? "error" : "follow",
			signal: AbortSignal.timeout(timeout),
		});
	} catch (error) {
		return { ok: false, failure: fetchFailure(error) };
	}
	if (response.status === 401 || response.status === 403)
		return { ok: false, failure: "unauthorized", status: response.status };
	if (!response.ok) return { ok: false, failure: "not-openai", status: response.status };
	try {
		return { ok: true, json: await response.json() };
	} catch {
		return { ok: false, failure: "not-openai" };
	}
}

async function detail(url: string, access: Access, body?: unknown): Promise<unknown> {
	const result = await fetchJson(url, access, DETAIL_TIMEOUT_MS, body);
	return result.ok ? result.json : undefined;
}

export function parseModelList(payload: unknown): CustomEndpointModel[] | null {
	if (!isObject(payload) || !Array.isArray(payload.data)) return null;
	const seen = new Set<string>();
	const models: CustomEndpointModel[] = [];
	for (const entry of payload.data) {
		if (!isObject(entry) || typeof entry.id !== "string" || entry.id === "") continue;
		if (seen.has(entry.id)) continue;
		if (typeof entry.mode === "string" && !CHAT_MODES.has(entry.mode)) continue;
		const architecture = isObject(entry.architecture) ? entry.architecture : {};
		const output = strings(architecture.output_modalities);
		if (output.length > 0 && !output.includes("text")) continue;
		const meta = isObject(entry.meta) ? entry.meta : {};
		const input = strings(architecture.input_modalities);
		seen.add(entry.id);
		models.push({
			id: entry.id,
			contextWindow:
				positiveInt(entry.max_model_len) ??
				positiveInt(entry.max_input_tokens) ??
				positiveInt(entry.context_length) ??
				positiveInt(meta.n_ctx),
			maxTokens: positiveInt(entry.max_output_tokens),
			reasoning: null,
			vision: input.length > 0 ? input.includes("image") : null,
			tools: null,
			chatTemplateThinking: null,
		});
	}
	return models;
}

async function enrichLlamaCpp(root: string, access: Access) {
	const props = await detail(`${root}/props`, access);
	if (!isObject(props) || !isObject(props.default_generation_settings)) return undefined;
	const caps = isObject(props.chat_template_caps) ? props.chat_template_caps : {};
	const templateThinking =
		typeof props.chat_template === "string" && props.chat_template.includes("enable_thinking");
	return {
		contextWindow: positiveInt(props.default_generation_settings.n_ctx),
		reasoning: templateThinking ? true : null,
		chatTemplateThinking: templateThinking ? true : null,
		tools: typeof caps.supports_tools === "boolean" ? caps.supports_tools : null,
	};
}

function ollamaContext(parameters: unknown): number | null {
	if (typeof parameters !== "string") return null;
	const match = /^\s*num_ctx\s+(\d+)\s*$/m.exec(parameters);
	return match ? positiveInt(Number(match[1])) : null;
}

async function enrichOllama(
	root: string,
	access: Access,
	models: CustomEndpointModel[],
): Promise<CustomEndpointModel[] | undefined> {
	const version = await detail(`${root}/api/version`, access);
	if (!isObject(version) || typeof version.version !== "string") return undefined;
	const shown = await Promise.all(
		models.map(async (model, index) => {
			if (index >= MAX_DETAIL_PROBES) return model;
			const show = await detail(`${root}/api/show`, access, { model: model.id });
			if (!isObject(show) || !Array.isArray(show.capabilities)) return model;
			const capabilities = strings(show.capabilities);
			if (!capabilities.includes("completion")) return null;
			return {
				...model,
				contextWindow: model.contextWindow ?? ollamaContext(show.parameters),
				reasoning: capabilities.includes("thinking"),
				vision: capabilities.includes("vision"),
				tools: capabilities.includes("tools"),
			};
		}),
	);
	return shown.filter((model): model is CustomEndpointModel => model !== null);
}

async function enrichLmStudio(root: string, access: Access) {
	const listing = await detail(`${root}/api/v0/models`, access);
	if (!isObject(listing) || !Array.isArray(listing.data)) return undefined;
	const details = new Map<string, Json>();
	for (const entry of listing.data)
		if (isObject(entry) && typeof entry.id === "string") details.set(entry.id, entry);
	return details;
}

async function enrich(
	baseUrl: string,
	access: Access,
	models: CustomEndpointModel[],
): Promise<CustomEndpointModel[]> {
	const root = baseUrl.replace(/\/v1$/, "");
	const [llama, ollama, lmStudio] = await Promise.all([
		enrichLlamaCpp(root, access),
		enrichOllama(root, access, models),
		enrichLmStudio(root, access),
	]);
	let next = ollama ?? models;
	if (llama)
		next = next.map((model) => ({
			...model,
			contextWindow: llama.contextWindow ?? model.contextWindow,
			reasoning: model.reasoning ?? llama.reasoning,
			chatTemplateThinking: model.chatTemplateThinking ?? llama.chatTemplateThinking,
			tools: model.tools ?? llama.tools,
		}));
	if (lmStudio)
		next = next.flatMap((model) => {
			const entry = lmStudio.get(model.id);
			if (!entry) return [model];
			if (entry.type === "embeddings") return [];
			return [
				{
					...model,
					contextWindow: model.contextWindow ?? positiveInt(entry.loaded_context_length),
					vision:
						entry.type === "vlm" || entry.type === "llm" ? entry.type === "vlm" : model.vision,
					tools: Array.isArray(entry.capabilities)
						? entry.capabilities.includes("tool_use")
						: model.tools,
				},
			];
		});
	return next;
}

let providerCache: { path: string; mtimeMs: number; providers: Json } | null = null;

/** models.json's providers, re-read only when the file changes: the request path asks per request. */
function currentProviders(): Json {
	const path = modelsPath();
	let mtimeMs: number;
	try {
		mtimeMs = statSync(path).mtimeMs;
	} catch {
		return {};
	}
	if (providerCache?.path === path && providerCache.mtimeMs === mtimeMs)
		return providerCache.providers;
	const providers = providersOf(parseModelsConfig(readModelsSource(path)));
	providerCache = { path, mtimeMs, providers };
	return providers;
}

function savedProvider(id: string): Json | undefined {
	if (!isCustomEndpointId(id)) return undefined;
	const provider = currentProviders()[id];
	return isObject(provider) ? provider : undefined;
}

setEndpointDirectory({
	name: (id) => {
		const name = savedProvider(id)?.name;
		return typeof name === "string" && name ? name : id;
	},
	keyCommand: (id) => savedCommands(savedProvider(id))[0],
	keyStatus: (id) => customEndpointKeyStatus(id),
});

function headerRecord(provider: Json | undefined): Record<string, string> {
	const headers: Record<string, string> = {};
	if (!provider || !isObject(provider.headers)) return headers;
	for (const [name, value] of Object.entries(provider.headers))
		if (typeof value === "string") headers[name] = value;
	return headers;
}

function savedHeader(provider: Json | undefined, name: string): string | undefined {
	const wanted = name.toLowerCase();
	return Object.entries(headerRecord(provider)).find(([key]) => key.toLowerCase() === wanted)?.[1];
}

/** The `!command` this provider's key runs, as pi would run it (pi doesn't trim before the `!`). */
function savedCommands(provider: Json | undefined): string[] {
	const key = provider?.apiKey;
	return typeof key === "string" && key.startsWith("!") ? [key] : [];
}

function commandsHash(commands: string[]): string | null {
	return commands.length === 0 ? null : commandHash(JSON.stringify([...commands].sort()));
}

/**
 * Fork: saved `!commands` run unprompted (Detect from an edit, key status) only when the user saved
 * exactly these commands through the form or the organization ships them; a command written into
 * models.json by anything else waits for the user to run it from the dialog.
 */
function commandsTrusted(id: string, provider: Json | undefined): boolean {
	const commands = savedCommands(provider);
	const hash = commandsHash(commands);
	if (hash === null) return true;
	const shipped = managedServer(id)?.apiKey;
	if (shipped !== undefined && commands.every((command) => command === shipped)) return true;
	return readEndpointLimits()[id]?.trustedKeyCommand === hash;
}

async function resolveHeaders(
	headers: Record<string, string>,
	allowCommand: boolean,
): Promise<Record<string, string>> {
	const resolved: Record<string, string> = {};
	for (const [name, value] of Object.entries(headers)) {
		const next = await resolveValue(value, allowCommand && !value.startsWith("!"));
		if (next !== undefined) resolved[name] = next;
	}
	return resolved;
}

async function resolveKey(raw: unknown, allowCommand: boolean): Promise<string | undefined> {
	if (typeof raw !== "string" || raw.trim() === NO_KEY) return undefined;
	if (raw.trim().startsWith("!")) return allowCommand ? commandToken(raw.trim()) : undefined;
	return resolveProbeKey(raw);
}

/** The saved key and headers for an endpoint's own unchanged address, and its network settings. */
async function savedAccess(id: string, baseUrl: string | null): Promise<Access> {
	const provider = savedProvider(id);
	const fetch = networkFetch(endpointNetworkOf(id));
	if (
		!provider ||
		typeof provider.baseUrl !== "string" ||
		baseUrl === null ||
		parseBaseUrl(provider.baseUrl) !== baseUrl
	)
		return { key: undefined, headers: {}, fetch };
	const trusted = commandsTrusted(id, provider);
	return {
		key: await resolveKey(provider.apiKey, trusted),
		headers: await resolveHeaders(headerRecord(provider), trusted),
		fetch,
	};
}

function requestNetwork(value: unknown, id: string | undefined): CustomEndpointNetwork {
	if (value !== undefined) return sanitizeNetwork(value) ?? DEFAULT_ENDPOINT_NETWORK;
	return id ? endpointNetworkOf(id) : DEFAULT_ENDPOINT_NETWORK;
}

async function probeAccess(request: CustomEndpointProbeRequest, baseUrl: string): Promise<Access> {
	const id = typeof request.endpointId === "string" ? request.endpointId : undefined;
	const saved = id ? await savedAccess(id, baseUrl) : undefined;
	const provider = id ? savedProvider(id) : undefined;
	const savedKeyRaw = typeof provider?.apiKey === "string" ? provider.apiKey : undefined;
	const typedKey = request.apiKey === savedKeyRaw ? undefined : request.apiKey;
	const usesSavedKey = typedKey === undefined;
	const key =
		typedKey === undefined
			? saved?.key
			: typedKey.startsWith("!")
				? await commandToken(typedKey)
				: resolveProbeKey(typedKey);
	let headers = saved?.headers ?? {};
	let usesSavedHeader = !Array.isArray(request.headers) && saved !== undefined;
	if (Array.isArray(request.headers)) {
		const typed: Record<string, string> = {};
		for (const header of request.headers) {
			if (!isObject(header) || !isEndpointHeaderName(header.name)) continue;
			const savedRaw = savedHeader(provider, header.name);
			if (header.value === null || (savedRaw !== undefined && header.value === savedRaw)) {
				const kept = savedHeaderResolved(saved, provider, header.name);
				if (kept !== undefined) {
					typed[header.name] = kept;
					usesSavedHeader = true;
				}
			} else if (typeof header.value === "string" && !header.value.startsWith("!")) {
				const value = await resolveValue(header.value, false);
				if (value !== undefined) typed[header.name] = value;
			}
		}
		headers = typed;
	}
	const savedCredentials =
		id !== undefined && ((usesSavedKey && saved?.key !== undefined) || usesSavedHeader);
	const network = savedCredentials ? endpointNetworkOf(id) : requestNetwork(request.network, id);
	return { key, headers, fetch: networkFetch(network) };
}

function savedHeaderResolved(
	saved: Access | undefined,
	provider: Json | undefined,
	name: string,
): string | undefined {
	if (!saved || !provider) return undefined;
	const original = Object.keys(headerRecord(provider)).find(
		(key) => key.toLowerCase() === name.toLowerCase(),
	);
	return original === undefined ? undefined : saved.headers[original];
}

export async function probeCustomEndpoint(
	request: CustomEndpointProbeRequest,
): Promise<CustomEndpointProbe> {
	if (!isObject(request) || typeof request.baseUrl !== "string")
		throw new Error("The server address must be text.");
	if (request.apiKey !== undefined && typeof request.apiKey !== "string")
		throw new Error("The API key must be text.");
	let baseUrl = parseBaseUrl(request.baseUrl);
	if (!baseUrl) return { ok: false, reason: "invalid-url" };
	let access: Access;
	try {
		access = await probeAccess(request, baseUrl);
	} catch {
		return { ok: false, reason: "certificate" };
	}
	let listing = await fetchJson(`${baseUrl}/models`, access, LIST_TIMEOUT_MS);
	if (!listing.ok && listing.status === 404 && !baseUrl.endsWith("/v1")) {
		const versioned = await fetchJson(`${baseUrl}/v1/models`, access, LIST_TIMEOUT_MS);
		if (versioned.ok) baseUrl = `${baseUrl}/v1`;
		if (versioned.ok || versioned.status !== 404) listing = versioned;
	}
	if (!listing.ok)
		return {
			ok: false,
			reason: listing.failure,
			...(listing.status === undefined ? {} : { status: listing.status }),
		};
	const models = parseModelList(listing.json);
	if (models === null) return { ok: false, reason: "not-openai" };
	if (models.length === 0) return { ok: false, reason: "no-models" };
	const [enriched, liteLlm] = await Promise.all([
		enrich(baseUrl, access, models),
		fetchLiteLlmModelInfo(baseUrl, access),
	]);
	if (!liteLlm) return { ok: true, baseUrl, models: enriched };
	const gatewayModels = applyLiteLlmModelInfo(enriched, liteLlm);
	if (gatewayModels.length === 0) return { ok: false, reason: "no-models" };
	return {
		ok: true,
		baseUrl,
		models: gatewayModels,
		gateway: { kind: "litellm", key: await fetchLiteLlmKeyStatus(baseUrl, access) },
	};
}

/** Fork: the saved key's own LiteLLM status, for an endpoint saved as a LiteLLM gateway. */
export async function customEndpointKeyStatus(id: string): Promise<CustomEndpointKeyStatus | null> {
	if (typeof id !== "string" || endpointGatewayOf(id)?.kind !== "litellm") return null;
	const provider = savedProvider(id);
	if (!provider || typeof provider.baseUrl !== "string") return null;
	const baseUrl = parseBaseUrl(provider.baseUrl);
	if (!baseUrl) return null;
	try {
		return await fetchLiteLlmKeyStatus(baseUrl, await savedAccess(id, baseUrl));
	} catch {
		return null;
	}
}

function readModel(raw: unknown, overrides: Json): CustomEndpointModel | undefined {
	if (!isObject(raw) || typeof raw.id !== "string" || raw.id === "") return undefined;
	const override = overrides[raw.id];
	const model = isObject(override) ? { ...raw, ...override } : raw;
	const compat = isObject(override) && isObject(override.compat) ? override.compat : raw.compat;
	const cost = readCost(model.cost);
	const api = isCustomEndpointApi(raw.api) && raw.api !== API ? raw.api : null;
	return {
		id: raw.id,
		...(cost ? { cost } : {}),
		...(api ? { api } : {}),
		contextWindow: positiveInt(model.contextWindow),
		maxTokens: positiveInt(model.maxTokens),
		reasoning: typeof model.reasoning === "boolean" ? model.reasoning : null,
		vision: Array.isArray(model.input) ? strings(model.input).includes("image") : null,
		tools: null,
		chatTemplateThinking:
			isObject(compat) && compat.thinkingFormat === CHAT_TEMPLATE_THINKING ? true : null,
	};
}

function readCost(value: unknown): CustomEndpointCost | null {
	if (!isObject(value)) return null;
	const { input, output, cacheRead, cacheWrite } = value;
	if (!isPrice(input) || !isPrice(output) || !isPrice(cacheRead) || !isPrice(cacheWrite))
		return null;
	return input > 0 || output > 0 ? { input, output, cacheRead, cacheWrite } : null;
}

function isPrice(value: unknown): value is number {
	return typeof value === "number" && Number.isFinite(value) && value >= 0;
}

function isCost(value: unknown): boolean {
	if (value === undefined || value === null) return true;
	if (!isObject(value)) return false;
	return [value.input, value.output, value.cacheRead, value.cacheWrite].every(isPrice);
}

function providersOf(config: Node | undefined): Json {
	const root = config ? getNodeValue(config) : undefined;
	return isObject(root) && isObject(root.providers) ? root.providers : {};
}

function readEndpoints(config: Node | undefined): CustomEndpoint[] {
	return Object.entries(providersOf(config)).flatMap(([id, provider]) => {
		if (!isCustomEndpointId(id) || !isObject(provider)) return [];
		if (provider.api !== API || typeof provider.baseUrl !== "string") return [];
		if (!parseBaseUrl(provider.baseUrl) || !Array.isArray(provider.models)) return [];
		const overrides = isObject(provider.modelOverrides) ? provider.modelOverrides : {};
		const apiKey = typeof provider.apiKey === "string" ? provider.apiKey : undefined;
		return [
			{
				id,
				name: typeof provider.name === "string" && provider.name ? provider.name : id,
				baseUrl: provider.baseUrl,
				hasApiKey: apiKey !== undefined && apiKey !== NO_KEY,
				limits: endpointLimitsOf(id),
				gateway: endpointGatewayOf(id),
				apiKeyReference:
					apiKey !== undefined && apiKey !== NO_KEY && isReference(apiKey) ? apiKey : null,
				headers: Object.entries(headerRecord(provider)).map(([name, value]) => ({
					name,
					value: isReference(value) ? value : null,
				})),
				network: endpointNetworkOf(id),
				managed: managedInfo(id),
				models: provider.models.flatMap((model) => readModel(model, overrides) ?? []),
			},
		];
	});
}

export async function listCustomEndpoints(): Promise<CustomEndpoint[]> {
	await syncManagedEndpoints().catch((error: unknown) =>
		log.warn("syncing the organization's managed servers failed", error),
	);
	return readEndpoints(parseModelsConfig(readModelsSource(modelsPath())));
}

function slug(text: string): string {
	return text
		.toLowerCase()
		.normalize("NFKD")
		.replace(/[^a-z0-9]+/g, "-")
		.slice(0, 40)
		.replace(/^-+|-+$/g, "");
}

function newEndpointId(name: string, baseUrl: string, config: Node | undefined): string {
	const taken = new Set(Object.keys(providersOf(config)));
	const host = parseBaseUrl(baseUrl);
	const fallback = host ? slug(new URL(host).hostname) : "";
	const base = `${CUSTOM_ENDPOINT_ID_PREFIX}${slug(name) || fallback || "server"}`;
	let id: string;
	do id = `${base}-${randomBytes(3).toString("hex")}`;
	while (taken.has(id));
	return id;
}

function isCount(value: unknown): boolean {
	return value === null || positiveInt(value) !== null;
}

function isFlag(value: unknown): boolean {
	return value === null || typeof value === "boolean";
}

function validateDraft(draft: CustomEndpointDraft): void {
	if (!isObject(draft) || typeof draft.name !== "string" || typeof draft.baseUrl !== "string")
		throw new Error("Name and server address must be text.");
	if (draft.id !== undefined && typeof draft.id !== "string")
		throw new Error("That server is no longer in models.json. Reload and retry.");
	if (draft.apiKey !== undefined && typeof draft.apiKey !== "string")
		throw new Error("The API key must be text.");
	if (!draft.name.trim()) throw new Error("Name the server.");
	if (draft.limits !== undefined) {
		const limits = draft.limits;
		if (
			!isObject(limits) ||
			(limits.maxConcurrency !== null && !isEndpointConcurrency(limits.maxConcurrency)) ||
			(limits.timeoutSeconds !== null && !isEndpointTimeoutSeconds(limits.timeoutSeconds)) ||
			(limits.maxRetries !== null && !isEndpointRetries(limits.maxRetries))
		)
			throw new Error("Parallel requests, timeout and retries must be whole numbers in range.");
	}
	if (draft.gateway !== undefined && draft.gateway !== null) {
		const gateway = draft.gateway;
		if (
			!isObject(gateway) ||
			gateway.kind !== "litellm" ||
			typeof gateway.sessionAffinity !== "boolean" ||
			!Array.isArray(gateway.tags) ||
			!gateway.tags.every(isEndpointTag)
		)
			throw new Error("Gateway tags may use letters, digits and . _ : / @ - only.");
	}
	validateHeaders(draft.headers);
	validateNetwork(draft.network);
	if (!parseBaseUrl(draft.baseUrl))
		throw new Error("Enter a full http:// or https:// address without credentials or a query.");
	if (!Array.isArray(draft.models) || draft.models.length === 0)
		throw new Error("Pick at least one model.");
	if (draft.models.length > MAX_MODELS) throw new Error("Pick fewer models.");
	const ids = new Set<string>();
	for (const model of draft.models) {
		if (!isObject(model) || typeof model.id !== "string" || !model.id.trim() || ids.has(model.id))
			throw new Error("Model ids must be unique and non-empty.");
		ids.add(model.id);
		if (!isCount(model.contextWindow) || !isCount(model.maxTokens))
			throw new Error("Model limits must be whole numbers.");
		if (
			!isFlag(model.reasoning) ||
			!isFlag(model.vision) ||
			!isFlag(model.chatTemplateThinking ?? null)
		)
			throw new Error("Model capabilities must be on or off.");
		if (!isCost(model.cost)) throw new Error("Model prices must be non-negative numbers.");
		if (model.api !== undefined && model.api !== null && !isCustomEndpointApi(model.api))
			throw new Error("Pick chat completions, Responses or Anthropic Messages for each model.");
	}
}

function validateHeaders(headers: unknown): void {
	if (headers === undefined) return;
	if (!Array.isArray(headers) || headers.length > CUSTOM_ENDPOINT_HEADER_LIMITS.count)
		throw new Error(`Send at most ${CUSTOM_ENDPOINT_HEADER_LIMITS.count} headers.`);
	const names = new Set<string>();
	for (const header of headers) {
		if (!isObject(header) || !isEndpointHeaderName(header.name))
			throw new Error("Header names may use letters, digits and - _ . only.");
		if (header.value !== null && !isEndpointHeaderValue(header.value))
			throw new Error(`Give header ${header.name} a value on one line.`);
		if (typeof header.value === "string" && header.value.startsWith("!"))
			throw new Error(
				`Header ${header.name} can't run a command; use the API key field or an environment variable.`,
			);
		if (names.has(header.name.toLowerCase()))
			throw new Error(`Header ${header.name} appears twice.`);
		names.add(header.name.toLowerCase());
	}
}

function validateNetwork(network: unknown): void {
	if (network === undefined) return;
	if (!isObject(network) || typeof network.systemCertificates !== "boolean")
		throw new Error("Network settings are incomplete.");
	if (network.proxy !== null && !isEndpointProxyUrl(network.proxy))
		throw new Error("Enter the proxy as http(s)://host:port, without a user name or password.");
	if (network.caFile !== null) {
		if (typeof network.caFile !== "string" || !network.caFile.trim())
			throw new Error("Enter the certificate file's path.");
		let pem: string;
		try {
			pem = readFileSync(network.caFile.trim(), "utf8");
		} catch {
			throw new Error(`The certificate file ${network.caFile} can't be read.`);
		}
		if (!pem.includes("-----BEGIN CERTIFICATE-----"))
			throw new Error(`The certificate file ${network.caFile} has no PEM certificate.`);
	}
}

function maxTokensFor(model: CustomEndpointModel): number | null {
	const context = model.contextWindow;
	if (model.maxTokens !== null && (context === null || model.maxTokens < context))
		return model.maxTokens;
	return context === null ? null : defaultMaxTokens(context);
}

/** The base URL a model's API expects: Anthropic's SDK appends `/v1/messages` itself. */
function apiBaseUrl(api: string, baseUrl: string): string | undefined {
	return api === "anthropic-messages" ? baseUrl.replace(/\/v1$/, "") : undefined;
}

function applyModelApi(next: Json, model: CustomEndpointModel, baseUrl: string): void {
	if (model.api === undefined) return;
	if (
		next.api === "anthropic-messages" &&
		typeof next.baseUrl === "string" &&
		originOf(next.baseUrl) === originOf(baseUrl)
	)
		delete next.baseUrl;
	if (model.api === null || model.api === API) {
		delete next.api;
		return;
	}
	next.api = model.api;
	const root = apiBaseUrl(model.api, baseUrl);
	if (root !== undefined) next.baseUrl = root;
}

function modelConfig(
	model: CustomEndpointModel,
	previous: Json | undefined,
	baseUrl: string,
): Json {
	const next: Json = {
		...(previous ?? {}),
		id: model.id,
		name: typeof previous?.name === "string" ? previous.name : model.id,
		cost: model.cost ? model.cost : isObject(previous?.cost) ? previous.cost : ZERO_COST,
	};
	if (model.reasoning === null) delete next.reasoning;
	else next.reasoning = model.reasoning;
	if (model.vision === null) delete next.input;
	else next.input = model.vision ? ["text", "image"] : ["text"];
	const maxTokens = maxTokensFor(model);
	if (model.contextWindow === null) delete next.contextWindow;
	else next.contextWindow = model.contextWindow;
	if (maxTokens === null) delete next.maxTokens;
	else next.maxTokens = maxTokens;
	const compat = isObject(next.compat) ? { ...next.compat } : {};
	if (model.chatTemplateThinking === true) compat.thinkingFormat = CHAT_TEMPLATE_THINKING;
	else if (model.chatTemplateThinking === false && compat.thinkingFormat === CHAT_TEMPLATE_THINKING)
		delete compat.thinkingFormat;
	if (Object.keys(compat).length > 0) next.compat = compat;
	else delete next.compat;
	applyModelApi(next, model, baseUrl);
	return next;
}

function withoutManagedOverrides(
	overrides: unknown,
	ids: Set<string>,
	thinkingIds: Set<string>,
): Json | undefined {
	if (!isObject(overrides)) return undefined;
	const next: Json = {};
	for (const [id, override] of Object.entries(overrides)) {
		if (!ids.has(id) || !isObject(override)) {
			next[id] = override;
			continue;
		}
		const kept: Json = { ...override };
		for (const field of MANAGED_MODEL_FIELDS) delete kept[field];
		if (thinkingIds.has(id) && isObject(kept.compat)) {
			const { thinkingFormat: _managed, ...compat } = kept.compat;
			if (Object.keys(compat).length > 0) kept.compat = compat;
			else delete kept.compat;
		}
		if (Object.keys(kept).length > 0) next[id] = kept;
	}
	return Object.keys(next).length > 0 ? next : undefined;
}

function originOf(url: string): string | null {
	try {
		return new URL(url).origin;
	} catch {
		return null;
	}
}

function routesElsewhere(provider: Json | undefined, baseUrl: string): boolean {
	if (!provider) return false;
	const origin = originOf(baseUrl);
	const models = Array.isArray(provider.models) ? provider.models : [];
	const overrides = isObject(provider.modelOverrides) ? Object.values(provider.modelOverrides) : [];
	return [...models, ...overrides].some(
		(entry) =>
			isObject(entry) && typeof entry.baseUrl === "string" && originOf(entry.baseUrl) !== origin,
	);
}

function nextHeaders(
	headers: CustomEndpointHeader[] | undefined,
	previous: Json | undefined,
): Record<string, string> | undefined {
	if (headers === undefined) return undefined;
	const next: Record<string, string> = {};
	for (const header of headers) {
		const value = header.value === null ? savedHeader(previous, header.name) : header.value.trim();
		if (value === undefined)
			throw new Error(`Enter a value for header ${header.name}; it isn't saved yet.`);
		next[header.name] = value;
	}
	return next;
}

function providerConfig(draft: CustomEndpointDraft, previous: Json | undefined): Json {
	const previousUrl = typeof previous?.baseUrl === "string" ? previous.baseUrl : undefined;
	const baseUrl =
		previousUrl !== undefined && draft.baseUrl.trim() === previousUrl
			? previousUrl
			: (parseBaseUrl(draft.baseUrl) ?? draft.baseUrl.trim());
	const previousKey = typeof previous?.apiKey === "string" ? previous.apiKey : undefined;
	if (previousUrl !== undefined && originOf(previousUrl) !== originOf(baseUrl))
		throw new Error("A server keeps its address. Add a different server as a new one.");
	if (routesElsewhere(previous, baseUrl))
		throw new Error(
			"Some of this server's models are routed to another address in models.json. Edit it there.",
		);
	const previousModels = new Map<string, Json>();
	for (const model of Array.isArray(previous?.models) ? previous.models : [])
		if (isObject(model) && typeof model.id === "string") previousModels.set(model.id, model);
	const next: Json = {
		...(previous ?? {}),
		name: draft.name.trim(),
		baseUrl,
		api: API,
		apiKey: draft.apiKey === undefined ? (previousKey ?? NO_KEY) : draft.apiKey.trim() || NO_KEY,
		compat: previous === undefined ? { ...COMPAT } : previous.compat,
		models: draft.models.map((model) => modelConfig(model, previousModels.get(model.id), baseUrl)),
	};
	if (next.compat === undefined) delete next.compat;
	const headers = nextHeaders(draft.headers, previous);
	if (headers === undefined) {
		// keep what is saved
	} else if (Object.keys(headers).length > 0) next.headers = headers;
	else delete next.headers;
	const overrides = withoutManagedOverrides(
		previous?.modelOverrides,
		new Set(draft.models.map((model) => model.id)),
		new Set(
			draft.models
				.filter((model) => typeof model.chatTemplateThinking === "boolean")
				.map((m) => m.id),
		),
	);
	if (overrides) next.modelOverrides = overrides;
	else delete next.modelOverrides;
	return next;
}

type Edit = { body: string; id: string; limits?: EndpointLimits | null };

function commit(edit: (body: string, config: Node | undefined) => Edit | null) {
	return serializeModelsWrite(() =>
		usePiRuntime(async (runtime) => {
			const path = modelsPath();
			const source = readModelsSource(path);
			const config = parseModelsConfig(source);
			const { bom, body } = splitModelsSource(source);
			const result = edit(body, config);
			if (result === null) return readEndpoints(config);
			const next = bom + result.body;
			const previousLimits = readEndpointLimits()[result.id] ?? null;
			const limitsChanged =
				result.limits !== undefined &&
				JSON.stringify(previousLimits) !== JSON.stringify(result.limits);
			if (limitsChanged) writeEndpointLimits(result.id, result.limits ?? null);
			if (next !== source) {
				try {
					writeModelsFile(path, next);
					providerCache = null;
				} catch (error) {
					if (limitsChanged) writeEndpointLimits(result.id, previousLimits);
					throw error;
				}
				await runtime
					.refresh({ allowNetwork: false })
					.catch((error: unknown) =>
						log.warn("pi runtime refresh after a custom endpoint save failed", error),
					);
			}
			if (next !== source || limitsChanged) publishModelsFileChange();
			if (result.limits === null) forgetRequestGate(result.id);
			return readEndpoints(parseModelsConfig(next));
		}),
	);
}

/**
 * Fork: what of a managed server's saved provider survives a sync or save: only the user's own key
 * (when the file ships none). Address, headers, compat and model routing always come from the file,
 * so a hand-edited models.json can't move the organization's server or keep its old address.
 */
function managedPrevious(server: ManagedServer, previous: unknown): Json {
	const kept: Json = { compat: { ...COMPAT } };
	if (server.apiKey === undefined && isObject(previous) && typeof previous.apiKey === "string")
		kept.apiKey = previous.apiKey;
	return kept;
}

/** Fork: an organization-managed server keeps everything its file sets; the user adds only the rest. */
function managedDraft(server: ManagedServer, draft: CustomEndpointDraft): CustomEndpointDraft {
	return {
		id: server.id,
		name: server.name,
		baseUrl: server.baseUrl,
		...(server.apiKey === undefined
			? draft.apiKey === undefined
				? {}
				: { apiKey: draft.apiKey }
			: { apiKey: server.apiKey }),
		headers: server.headers,
		limits: server.limits,
		gateway: server.gateway,
		network: server.network ?? DEFAULT_ENDPOINT_NETWORK,
		models: server.models ?? draft.models,
	};
}

export function saveCustomEndpoint(input: CustomEndpointDraft): Promise<CustomEndpoint[]> {
	try {
		validateDraft(input);
	} catch (error) {
		return Promise.reject(error);
	}
	const server = input.id === undefined ? undefined : managedServer(input.id);
	const draft = server ? managedDraft(server, input) : input;
	return commit((body, config) => {
		if (draft.id !== undefined && !readEndpoints(config).some((e) => e.id === draft.id))
			throw new Error("That server is no longer in models.json. Reload and retry.");
		const id = draft.id ?? newEndpointId(draft.name, draft.baseUrl, config);
		const previousRaw = draft.id === undefined ? undefined : providersOf(config)[draft.id];
		const previous = server
			? managedPrevious(server, previousRaw)
			: isObject(previousRaw)
				? previousRaw
				: undefined;
		const value = providerConfig(draft, previous);
		return {
			body: applyEdits(body, modify(body, ["providers", id], value, MODELS_JSON_FORMATTING)),
			id,
			...settingsEdit(id, draft, value, server !== undefined, input.headers !== undefined),
		};
	});
}

function settingsEdit(
	id: string,
	draft: CustomEndpointDraft,
	provider: Json,
	managed: boolean,
	showsReferences: boolean,
): { limits?: EndpointLimits | null } {
	const next = nextEndpointSettings(id, {
		limits: draft.limits,
		gateway: draft.gateway,
		network: draft.network,
		// Only a client that shows the saved command (v105+) can have the user vouch for it.
		...(showsReferences ? { trustedKeyCommand: commandsHash(savedCommands(provider)) } : {}),
		...(managed ? { managed: true } : {}),
	});
	return next === undefined ? {} : { limits: next };
}

/**
 * Fork: brings the organization's managed servers into models.json and the sidecar, and drops the
 * ones its file no longer lists. A server whose file sets no key keeps the key the user entered; one
 * that lists no models keeps the user's picks. A no-op when everything already matches.
 */
export function syncManagedEndpoints(): Promise<void> {
	const managed = readManagedConfig();
	const wanted = new Set(managed.servers.map((server) => server.id));
	const stale = managed.degraded
		? []
		: Object.entries(readEndpointLimits())
				.filter(([id, entry]) => entry.managed === true && !wanted.has(id))
				.map(([id]) => id);
	if (managed.servers.length === 0 && stale.length === 0) return Promise.resolve();
	return serializeModelsWrite(() =>
		usePiRuntime(async (runtime) => {
			const path = modelsPath();
			const source = readModelsSource(path);
			const config = parseModelsConfig(source);
			const { bom, body: original } = splitModelsSource(source);
			let body = original;
			const settings = new Map<string, EndpointLimits | null>();
			for (const server of managed.servers) {
				const previous = managedPrevious(server, providersOf(config)[server.id]);
				const previousModels = readEndpoints(config).find((e) => e.id === server.id)?.models ?? [];
				const draft = managedDraft(server, {
					name: server.name,
					baseUrl: server.baseUrl,
					models: previousModels,
				});
				let value: Json;
				try {
					value = providerConfig(draft, previous);
				} catch (error) {
					log.warn(`managed server ${server.id} wasn't applied`, error);
					continue;
				}
				body = applyEdits(
					body,
					modify(body, ["providers", server.id], value, MODELS_JSON_FORMATTING),
				);
				const next = nextEndpointSettings(server.id, {
					limits: server.limits,
					gateway: server.gateway,
					network: server.network ?? DEFAULT_ENDPOINT_NETWORK,
					managed: true,
				});
				if (next !== undefined) settings.set(server.id, next);
			}
			for (const id of stale) {
				body = applyEdits(body, modify(body, ["providers", id], undefined, MODELS_JSON_FORMATTING));
				settings.set(id, null);
			}
			const sidecar = readEndpointLimits();
			let settingsChanged = false;
			for (const [id, next] of settings) {
				if (JSON.stringify(sidecar[id] ?? null) === JSON.stringify(next)) continue;
				writeEndpointLimits(id, next);
				settingsChanged = true;
				if (next === null) forgetRequestGate(id);
			}
			const next = bom + body;
			if (next !== source) {
				writeModelsFile(path, next);
				providerCache = null;
				await runtime
					.refresh({ allowNetwork: false })
					.catch((error: unknown) =>
						log.warn("pi runtime refresh after a managed sync failed", error),
					);
			}
			if (next !== source || settingsChanged) publishModelsFileChange();
		}),
	);
}

export function removeCustomEndpoint(id: string): Promise<CustomEndpoint[]> {
	if (managedServer(id))
		return Promise.reject(
			new Error("Your organization manages this server, so it can't be removed here."),
		);
	return commit((body, config) => {
		if (!readEndpoints(config).some((endpoint) => endpoint.id === id)) return null;
		return {
			body: applyEdits(body, modify(body, ["providers", id], undefined, MODELS_JSON_FORMATTING)),
			id,
			limits: null,
		};
	});
}
