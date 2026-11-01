import {
	CUSTOM_ENDPOINT_CORPORATE_PROTOCOL_VERSION,
	CUSTOM_ENDPOINT_GATEWAY_PROTOCOL_VERSION,
	CUSTOM_ENDPOINT_HEADER_LIMITS,
	CUSTOM_ENDPOINT_LIMITS_PROTOCOL_VERSION,
	CUSTOM_ENDPOINT_TAG_LIMITS,
	type CustomEndpoint,
	type CustomEndpointDraft,
	type CustomEndpointGateway,
	type CustomEndpointHeader,
	type CustomEndpointKeyStatus,
	type CustomEndpointLimits,
	type CustomEndpointManaged,
	type CustomEndpointModel,
	type CustomEndpointNetwork,
	type CustomEndpointProbe,
	isEndpointConcurrency,
	isEndpointHeaderName,
	isEndpointHeaderValue,
	isEndpointProxyUrl,
	isEndpointRetries,
	isEndpointTag,
	isEndpointTimeoutSeconds,
} from "@thinkrail/contracts";
import type { Translator } from "@thinkrail/locales/react";

export interface EndpointModelRow extends CustomEndpointModel {
	selected: boolean;
}

export interface GatewayDraft {
	sessionAffinity: boolean;
	tagsText: string;
}

export const DEFAULT_GATEWAY_TAGS = "thinkrail";

export interface HeaderRow {
	key: number;
	name: string;
	value: string;
	/** The host keeps a value for this header; an empty field keeps it. */
	saved: boolean;
	/** The saved `$VAR` reference the field starts with; while it still shows it, nothing was typed. */
	reference: string | null;
}

export interface EndpointDraft {
	id?: string;
	name: string;
	baseUrl: string;
	apiKey: string;
	hasSavedKey: boolean;
	removeKey: boolean;
	savedBaseUrl: string | null;
	concurrencyText: string;
	timeoutText: string;
	retriesText: string;
	gateway: GatewayDraft | null;
	/** Fork: the saved key when it's a `$VAR` or `!command`; while the field still shows it, nothing was typed. */
	savedKeyReference: string | null;
	headers: HeaderRow[];
	systemCertificates: boolean;
	caFile: string;
	proxy: string;
	managed: CustomEndpointManaged | null;
	models: EndpointModelRow[];
}

export const LOCAL_SERVER_PRESETS = [
	{ name: "Ollama", baseUrl: "http://127.0.0.1:11434/v1" },
	{ name: "LM Studio", baseUrl: "http://127.0.0.1:1234/v1" },
	{ name: "vLLM", baseUrl: "http://127.0.0.1:8000/v1" },
	{ name: "llama.cpp", baseUrl: "http://127.0.0.1:8080/v1" },
] as const;

function numberText(value: number | null): string {
	return value === null ? "" : String(value);
}

function wholeNumber(text: string): number | null {
	return /^\d+$/.test(text.trim()) ? Number(text.trim()) : null;
}

function optionalValid(text: string, valid: (value: number) => boolean): boolean {
	if (text.trim() === "") return true;
	const value = wholeNumber(text);
	return value !== null && valid(value);
}

export interface LimitIssues {
	concurrency: boolean;
	timeout: boolean;
	retries: boolean;
	tags: boolean;
}

export function parseTags(text: string): string[] {
	return [...new Set(text.split(/[\s,]+/).filter((tag) => tag !== ""))];
}

function tagsInvalid(gateway: GatewayDraft | null): boolean {
	if (!gateway) return false;
	const tags = parseTags(gateway.tagsText);
	return tags.length > CUSTOM_ENDPOINT_TAG_LIMITS.count || !tags.every(isEndpointTag);
}

export function draftGateway(draft: EndpointDraft): CustomEndpointGateway | null {
	if (!draft.gateway) return null;
	return {
		kind: "litellm",
		sessionAffinity: draft.gateway.sessionAffinity,
		tags: parseTags(draft.gateway.tagsText),
	};
}

export function limitIssues(draft: EndpointDraft): LimitIssues {
	return {
		concurrency: !optionalValid(draft.concurrencyText, isEndpointConcurrency),
		timeout: !optionalValid(draft.timeoutText, isEndpointTimeoutSeconds),
		retries: !optionalValid(draft.retriesText, isEndpointRetries),
		tags: tagsInvalid(draft.gateway),
	};
}

export function draftLimits(draft: EndpointDraft): CustomEndpointLimits {
	return {
		maxConcurrency: wholeNumber(draft.concurrencyText),
		timeoutSeconds: wholeNumber(draft.timeoutText),
		maxRetries: wholeNumber(draft.retriesText),
	};
}

let nextHeaderKey = 1;

function headerRow(name = "", reference: string | null = null, saved = false): HeaderRow {
	return { key: nextHeaderKey++, name, value: reference ?? "", saved, reference };
}

export function addHeaderRow(draft: EndpointDraft): EndpointDraft {
	return { ...draft, headers: [...draft.headers, headerRow()] };
}

export function updateHeaderRow(
	draft: EndpointDraft,
	key: number,
	patch: Partial<Pick<HeaderRow, "name" | "value">>,
): EndpointDraft {
	return {
		...draft,
		headers: draft.headers.map((row) => (row.key === key ? { ...row, ...patch } : row)),
	};
}

export function removeHeaderRow(draft: EndpointDraft, key: number): EndpointDraft {
	return { ...draft, headers: draft.headers.filter((row) => row.key !== key) };
}

export interface ConnectionIssues {
	headers: boolean;
	proxy: boolean;
}

export function connectionIssues(draft: EndpointDraft): ConnectionIssues {
	const names = new Set<string>();
	const headers =
		draft.headers.length > CUSTOM_ENDPOINT_HEADER_LIMITS.count ||
		draft.headers.some((row) => {
			const name = row.name.trim();
			const duplicate = names.has(name.toLowerCase());
			names.add(name.toLowerCase());
			const valueMissing = row.value.trim() === "" && !row.saved;
			const value = row.value.trim();
			return (
				duplicate ||
				!isEndpointHeaderName(name) ||
				valueMissing ||
				(value !== "" && (!isEndpointHeaderValue(value) || value.startsWith("!")))
			);
		});
	return { headers, proxy: draft.proxy.trim() !== "" && !isEndpointProxyUrl(draft.proxy.trim()) };
}

export function draftHeaders(draft: EndpointDraft): CustomEndpointHeader[] {
	return draft.headers.map((row) => {
		const value = row.value.trim();
		const kept = row.saved && (value === "" || value === row.reference);
		return { name: row.name.trim(), value: kept ? null : value };
	});
}

export function draftNetwork(draft: EndpointDraft): CustomEndpointNetwork {
	return {
		systemCertificates: draft.systemCertificates,
		caFile: draft.caFile.trim() || null,
		proxy: draft.proxy.trim() || null,
	};
}

export function newEndpointDraft(): EndpointDraft {
	return {
		name: "",
		baseUrl: "",
		apiKey: "",
		hasSavedKey: false,
		removeKey: false,
		savedBaseUrl: null,
		concurrencyText: "",
		timeoutText: "",
		retriesText: "",
		gateway: null,
		savedKeyReference: null,
		headers: [],
		systemCertificates: false,
		caFile: "",
		proxy: "",
		managed: null,
		models: [],
	};
}

export function editEndpointDraft(endpoint: CustomEndpoint): EndpointDraft {
	return {
		id: endpoint.id,
		name: endpoint.name,
		baseUrl: endpoint.baseUrl,
		apiKey: endpoint.apiKeyReference ?? "",
		hasSavedKey: endpoint.hasApiKey,
		removeKey: false,
		savedBaseUrl: endpoint.baseUrl,
		concurrencyText: numberText(endpoint.limits.maxConcurrency),
		timeoutText: numberText(endpoint.limits.timeoutSeconds),
		retriesText: numberText(endpoint.limits.maxRetries),
		gateway: endpoint.gateway
			? {
					sessionAffinity: endpoint.gateway.sessionAffinity,
					tagsText: endpoint.gateway.tags.join(", "),
				}
			: null,
		savedKeyReference: endpoint.apiKeyReference,
		headers: endpoint.headers.map((header) => headerRow(header.name, header.value, true)),
		systemCertificates: endpoint.network.systemCertificates,
		caFile: endpoint.network.caFile ?? "",
		proxy: endpoint.network.proxy ?? "",
		managed: endpoint.managed,
		models: endpoint.models.map((model) => ({ ...model, selected: true })),
	};
}

export function applyPreset(
	draft: EndpointDraft,
	preset: (typeof LOCAL_SERVER_PRESETS)[number],
): EndpointDraft {
	const presetName = LOCAL_SERVER_PRESETS.some((other) => other.name === draft.name.trim());
	return {
		...draft,
		baseUrl: preset.baseUrl,
		name: draft.name.trim() === "" || presetName ? preset.name : draft.name,
	};
}

export function addManualModel(draft: EndpointDraft, rawId: string): EndpointDraft {
	const id = rawId.trim();
	if (!id || draft.models.some((model) => model.id === id)) return draft;
	const row: EndpointModelRow = {
		id,
		contextWindow: null,
		maxTokens: null,
		reasoning: null,
		vision: null,
		tools: null,
		chatTemplateThinking: null,
		selected: true,
	};
	return { ...draft, models: [...draft.models, row] };
}

function withCost(cost: CustomEndpointModel["cost"]): Pick<CustomEndpointModel, "cost"> {
	return cost ? { cost } : {};
}

function withApi(api: CustomEndpointModel["api"]): Pick<CustomEndpointModel, "api"> {
	return api === undefined ? {} : { api };
}

export function mergeProbe(
	draft: EndpointDraft,
	probe: Extract<CustomEndpointProbe, { ok: true }>,
): EndpointDraft {
	const known = new Map(draft.models.map((model) => [model.id, model]));
	const firstDetect = draft.models.length === 0;
	const found: EndpointModelRow[] = probe.models.map((model) => {
		const previous = known.get(model.id);
		if (!previous) return { ...model, selected: firstDetect };
		const { api: _suggested, ...reported } = model;
		return {
			...reported,
			contextWindow: model.contextWindow ?? previous.contextWindow,
			maxTokens:
				model.contextWindow === null ? (model.maxTokens ?? previous.maxTokens) : model.maxTokens,
			reasoning: model.reasoning ?? previous.reasoning,
			vision: model.vision ?? previous.vision,
			chatTemplateThinking: model.chatTemplateThinking ?? previous.chatTemplateThinking,
			...withCost(model.cost ?? previous.cost),
			...withApi(previous.api),
			selected: previous.selected,
		};
	});
	const kept = draft.models.filter((model) => !probe.models.some((found) => found.id === model.id));
	const baseUrl = draft.baseUrl === draft.savedBaseUrl ? draft.baseUrl : probe.baseUrl;
	const gateway =
		draft.gateway ??
		(probe.gateway ? { sessionAffinity: true, tagsText: DEFAULT_GATEWAY_TAGS } : null);
	const keyParallel = probe.gateway?.key?.maxParallelRequests ?? null;
	const concurrencyText =
		draft.concurrencyText.trim() === "" &&
		keyParallel !== null &&
		isEndpointConcurrency(keyParallel)
			? String(keyParallel)
			: draft.concurrencyText;
	return { ...draft, baseUrl, gateway, concurrencyText, models: [...found, ...kept] };
}

function keyTyped(draft: EndpointDraft): boolean {
	const key = draft.apiKey.trim();
	return key !== "" && key !== draft.savedKeyReference;
}

export function probeTarget(draft: EndpointDraft): { apiKey?: string; endpointId?: string } {
	if (draft.removeKey) return {};
	if (keyTyped(draft) || !draft.id) return { apiKey: draft.apiKey };
	return { endpointId: draft.id };
}

/** Fork: the headers and network settings Detect sends; saved header values stay on the host. */
export function probeExtras(
	draft: EndpointDraft,
	protocolVersion: number | null,
): { endpointId?: string; headers?: CustomEndpointHeader[]; network?: CustomEndpointNetwork } {
	if ((protocolVersion ?? 0) < CUSTOM_ENDPOINT_CORPORATE_PROTOCOL_VERSION) return {};
	return {
		...(draft.id ? { endpointId: draft.id } : {}),
		headers: draftHeaders(draft),
		network: draftNetwork(draft),
	};
}

export function draftParams(
	draft: EndpointDraft,
	protocolVersion: number | null,
): CustomEndpointDraft {
	const version = protocolVersion ?? 0;
	const key = draft.removeKey ? "" : draft.apiKey.trim() || (draft.id ? undefined : "");
	return {
		...(draft.id ? { id: draft.id } : {}),
		name: draft.name.trim(),
		baseUrl: draft.baseUrl.trim(),
		...(key === undefined ? {} : { apiKey: key }),
		...(version >= CUSTOM_ENDPOINT_LIMITS_PROTOCOL_VERSION ? { limits: draftLimits(draft) } : {}),
		...(version >= CUSTOM_ENDPOINT_GATEWAY_PROTOCOL_VERSION
			? { gateway: draftGateway(draft) }
			: {}),
		...(version >= CUSTOM_ENDPOINT_CORPORATE_PROTOCOL_VERSION
			? { headers: draftHeaders(draft), network: draftNetwork(draft) }
			: {}),
		models: draft.models
			.filter((model) => model.selected)
			.map(({ selected: _selected, api, ...model }) =>
				version >= CUSTOM_ENDPOINT_CORPORATE_PROTOCOL_VERSION
					? { ...model, ...withApi(api) }
					: model,
			),
	};
}

function originOf(address: string): string | null {
	try {
		return new URL(address.trim()).origin;
	} catch {
		return null;
	}
}

export function addressMoved(draft: EndpointDraft): boolean {
	return draft.savedBaseUrl !== null && originOf(draft.baseUrl) !== originOf(draft.savedBaseUrl);
}

export function canSave(draft: EndpointDraft): boolean {
	const issues = limitIssues(draft);
	const connection = connectionIssues(draft);
	return (
		!connection.headers &&
		!connection.proxy &&
		!addressMoved(draft) &&
		!issues.concurrency &&
		!issues.timeout &&
		!issues.retries &&
		!issues.tags &&
		draft.name.trim() !== "" &&
		draft.baseUrl.trim() !== "" &&
		draft.models.some((model) => model.selected)
	);
}

type ProbeFailure = Extract<CustomEndpointProbe, { ok: false }>;

export function probeFailureText(t: Translator, failure: ProbeFailure): string {
	switch (failure.reason) {
		case "invalid-url":
			return t("settings.customEndpoints.failure.invalidUrl");
		case "unauthorized":
			return t("settings.customEndpoints.failure.unauthorized");
		case "timeout":
			return t("settings.customEndpoints.failure.timeout");
		case "unreachable":
			return t("settings.customEndpoints.failure.unreachable");
		case "no-models":
			return t("settings.customEndpoints.failure.noModels");
		case "certificate":
			return t("settings.customEndpoints.failure.certificate");
		case "not-openai":
			return failure.status === undefined
				? t("settings.customEndpoints.failure.notOpenai")
				: t("settings.customEndpoints.failure.http", { status: failure.status });
	}
}

const EXPIRY_WARNING_MS = 7 * 24 * 60 * 60 * 1000;
const BUDGET_WARNING_SHARE = 0.9;

export interface KeyAlerts {
	budget: "near" | "exhausted" | null;
	expiry: "soon" | "expired" | null;
}

export function keyAlerts(status: CustomEndpointKeyStatus, now: number): KeyAlerts {
	let budget: KeyAlerts["budget"] = null;
	if (status.maxBudget !== null && status.maxBudget > 0 && status.spend !== null) {
		if (status.spend >= status.maxBudget) budget = "exhausted";
		else if (status.spend >= status.maxBudget * BUDGET_WARNING_SHARE) budget = "near";
	}
	const expires = status.expiresAt === null ? Number.NaN : Date.parse(status.expiresAt);
	let expiry: KeyAlerts["expiry"] = null;
	if (Number.isFinite(expires)) {
		if (expires <= now) expiry = "expired";
		else if (expires - now <= EXPIRY_WARNING_MS) expiry = "soon";
	}
	return { budget, expiry };
}
