import type {
	CustomEndpointCost,
	CustomEndpointKeyStatus,
	CustomEndpointModel,
} from "@thinkrail/contracts";

const DETAIL_TIMEOUT_MS = 3_000;
const CHAT_MODES = new Set(["chat", "completion", "responses"]);
const PER_MILLION = 1_000_000;

type Json = Record<string, unknown>;

function isObject(value: unknown): value is Json {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

function positiveInt(value: unknown): number | null {
	return typeof value === "number" && Number.isSafeInteger(value) && value > 0 ? value : null;
}

function nonNegative(value: unknown): number | null {
	return typeof value === "number" && Number.isFinite(value) && value >= 0 ? value : null;
}

function text(value: unknown): string | null {
	return typeof value === "string" && value.trim() !== "" ? value : null;
}

async function getJson(url: string, key: string | undefined): Promise<unknown> {
	const headers: Record<string, string> = { accept: "application/json" };
	if (key) headers.authorization = `Bearer ${key}`;
	try {
		const response = await fetch(url, {
			headers,
			redirect: key ? "error" : "follow",
			signal: AbortSignal.timeout(DETAIL_TIMEOUT_MS),
		});
		return response.ok ? await response.json() : undefined;
	} catch {
		return undefined;
	}
}

/** The proxy root LiteLLM serves its management routes from: the base URL without its `/v1`. */
export function liteLlmRoot(baseUrl: string): string {
	return baseUrl.replace(/\/v1$/, "");
}

export interface LiteLlmModelInfo {
	mode: string | null;
	contextWindow: number | null;
	maxTokens: number | null;
	reasoning: boolean | null;
	vision: boolean | null;
	tools: boolean | null;
	cost: CustomEndpointCost | null;
}

function flag(info: Json, field: string): boolean | null {
	const value = info[field];
	return typeof value === "boolean" ? value : null;
}

function costOf(info: Json): CustomEndpointCost | null {
	const input = nonNegative(info.input_cost_per_token);
	const output = nonNegative(info.output_cost_per_token);
	if (input === null && output === null) return null;
	const perMillion = (value: number | null) =>
		value === null ? 0 : Math.round(value * PER_MILLION * 1e6) / 1e6;
	const cost = {
		input: perMillion(input),
		output: perMillion(output),
		cacheRead: perMillion(nonNegative(info.cache_read_input_token_cost)),
		cacheWrite: perMillion(nonNegative(info.cache_creation_input_token_cost)),
	};
	return cost.input > 0 || cost.output > 0 ? cost : null;
}

function smallest(a: number | null, b: number | null): number | null {
	if (a === null) return b;
	if (b === null) return a;
	return Math.min(a, b);
}

function both(a: boolean | null, b: boolean | null): boolean | null {
	if (a === null) return b;
	if (b === null) return a;
	return a && b;
}

/**
 * Reads LiteLLM's `/model/info`. A model group can have several deployments; the group gets the
 * smallest limits and only the capabilities every reporting deployment has, since a request may land
 * on any of them. Returns undefined when the server isn't a LiteLLM proxy.
 */
export function parseLiteLlmModelInfo(payload: unknown): Map<string, LiteLlmModelInfo> | undefined {
	if (!isObject(payload) || !Array.isArray(payload.data)) return undefined;
	const groups = new Map<string, LiteLlmModelInfo>();
	let recognized = false;
	for (const entry of payload.data) {
		if (!isObject(entry) || typeof entry.model_name !== "string" || !isObject(entry.model_info))
			continue;
		recognized = true;
		const info = entry.model_info;
		const next: LiteLlmModelInfo = {
			mode: text(info.mode),
			contextWindow: positiveInt(info.max_input_tokens),
			maxTokens: positiveInt(info.max_output_tokens),
			reasoning: flag(info, "supports_reasoning"),
			vision: flag(info, "supports_vision"),
			tools: flag(info, "supports_function_calling"),
			cost: costOf(info),
		};
		const previous = groups.get(entry.model_name);
		groups.set(
			entry.model_name,
			previous
				? {
						mode: previous.mode ?? next.mode,
						contextWindow: smallest(previous.contextWindow, next.contextWindow),
						maxTokens: smallest(previous.maxTokens, next.maxTokens),
						reasoning: both(previous.reasoning, next.reasoning),
						vision: both(previous.vision, next.vision),
						tools: both(previous.tools, next.tools),
						cost: previous.cost ?? next.cost,
					}
				: next,
		);
	}
	return recognized ? groups : undefined;
}

/**
 * Applies a LiteLLM proxy's own model facts to the listed models: its configured limits win over
 * listing guesses, wildcard routes and non-chat routes (embeddings, audio, images) are dropped.
 */
export function applyLiteLlmModelInfo(
	models: CustomEndpointModel[],
	info: Map<string, LiteLlmModelInfo>,
): CustomEndpointModel[] {
	return models.flatMap((model) => {
		if (model.id.includes("*")) return [];
		const group = info.get(model.id);
		if (!group) return [model];
		if (group.mode !== null && !CHAT_MODES.has(group.mode)) return [];
		return [
			{
				...model,
				contextWindow: group.contextWindow ?? model.contextWindow,
				maxTokens: group.maxTokens ?? model.maxTokens,
				reasoning: group.reasoning ?? model.reasoning,
				vision: group.vision ?? model.vision,
				tools: group.tools ?? model.tools,
				cost: group.cost,
			},
		];
	});
}

export async function fetchLiteLlmModelInfo(
	baseUrl: string,
	key: string | undefined,
): Promise<Map<string, LiteLlmModelInfo> | undefined> {
	return parseLiteLlmModelInfo(await getJson(`${liteLlmRoot(baseUrl)}/model/info`, key));
}

export function parseLiteLlmKeyInfo(payload: unknown): CustomEndpointKeyStatus | null {
	if (!isObject(payload) || !isObject(payload.info)) return null;
	const info = payload.info;
	return {
		alias: text(info.key_alias) ?? text(info.key_name),
		spend: nonNegative(info.spend),
		maxBudget: nonNegative(info.max_budget),
		budgetResetAt: text(info.budget_reset_at),
		expiresAt: text(info.expires),
		rpmLimit: positiveInt(info.rpm_limit),
		tpmLimit: positiveInt(info.tpm_limit),
		maxParallelRequests: positiveInt(info.max_parallel_requests),
	};
}

/** The calling key's own `/key/info`; LiteLLM answers it for the key in the Authorization header. */
export async function fetchLiteLlmKeyStatus(
	baseUrl: string,
	key: string | undefined,
): Promise<CustomEndpointKeyStatus | null> {
	if (!key) return null;
	return parseLiteLlmKeyInfo(await getJson(`${liteLlmRoot(baseUrl)}/key/info`, key));
}
