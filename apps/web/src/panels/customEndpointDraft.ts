import {
	type CustomEndpoint,
	type CustomEndpointDraft,
	type CustomEndpointLimits,
	type CustomEndpointModel,
	type CustomEndpointProbe,
	isEndpointConcurrency,
	isEndpointRetries,
	isEndpointTimeoutSeconds,
} from "@thinkrail/contracts";
import type { Translator } from "@thinkrail/locales/react";

export interface EndpointModelRow extends CustomEndpointModel {
	selected: boolean;
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
}

export function limitIssues(draft: EndpointDraft): LimitIssues {
	return {
		concurrency: !optionalValid(draft.concurrencyText, isEndpointConcurrency),
		timeout: !optionalValid(draft.timeoutText, isEndpointTimeoutSeconds),
		retries: !optionalValid(draft.retriesText, isEndpointRetries),
	};
}

export function draftLimits(draft: EndpointDraft): CustomEndpointLimits {
	return {
		maxConcurrency: wholeNumber(draft.concurrencyText),
		timeoutSeconds: wholeNumber(draft.timeoutText),
		maxRetries: wholeNumber(draft.retriesText),
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
		models: [],
	};
}

export function editEndpointDraft(endpoint: CustomEndpoint): EndpointDraft {
	return {
		id: endpoint.id,
		name: endpoint.name,
		baseUrl: endpoint.baseUrl,
		apiKey: "",
		hasSavedKey: endpoint.hasApiKey,
		removeKey: false,
		savedBaseUrl: endpoint.baseUrl,
		concurrencyText: numberText(endpoint.limits.maxConcurrency),
		timeoutText: numberText(endpoint.limits.timeoutSeconds),
		retriesText: numberText(endpoint.limits.maxRetries),
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

export function mergeProbe(
	draft: EndpointDraft,
	probe: Extract<CustomEndpointProbe, { ok: true }>,
): EndpointDraft {
	const known = new Map(draft.models.map((model) => [model.id, model]));
	const firstDetect = draft.models.length === 0;
	const found: EndpointModelRow[] = probe.models.map((model) => {
		const previous = known.get(model.id);
		if (!previous) return { ...model, selected: firstDetect };
		return {
			...model,
			contextWindow: model.contextWindow ?? previous.contextWindow,
			maxTokens:
				model.contextWindow === null ? (model.maxTokens ?? previous.maxTokens) : model.maxTokens,
			reasoning: model.reasoning ?? previous.reasoning,
			vision: model.vision ?? previous.vision,
			chatTemplateThinking: model.chatTemplateThinking ?? previous.chatTemplateThinking,
			selected: previous.selected,
		};
	});
	const kept = draft.models.filter((model) => !probe.models.some((found) => found.id === model.id));
	const baseUrl = draft.baseUrl === draft.savedBaseUrl ? draft.baseUrl : probe.baseUrl;
	return { ...draft, baseUrl, models: [...found, ...kept] };
}

export function probeTarget(draft: EndpointDraft): { apiKey?: string; endpointId?: string } {
	if (draft.removeKey) return {};
	if (draft.apiKey.trim() || !draft.id) return { apiKey: draft.apiKey };
	return { endpointId: draft.id };
}

export function draftParams(draft: EndpointDraft, withLimits: boolean): CustomEndpointDraft {
	const key = draft.removeKey ? "" : draft.apiKey.trim() || (draft.id ? undefined : "");
	return {
		...(draft.id ? { id: draft.id } : {}),
		name: draft.name.trim(),
		baseUrl: draft.baseUrl.trim(),
		...(key === undefined ? {} : { apiKey: key }),
		...(withLimits ? { limits: draftLimits(draft) } : {}),
		models: draft.models
			.filter((model) => model.selected)
			.map(({ selected: _selected, ...model }) => model),
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
	return (
		!addressMoved(draft) &&
		!issues.concurrency &&
		!issues.timeout &&
		!issues.retries &&
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
		case "not-openai":
			return failure.status === undefined
				? t("settings.customEndpoints.failure.notOpenai")
				: t("settings.customEndpoints.failure.http", { status: failure.status });
	}
}
