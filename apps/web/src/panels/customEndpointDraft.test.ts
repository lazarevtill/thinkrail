import { expect, test } from "bun:test";
import type { CustomEndpointModel } from "@thinkrail/contracts";
import {
	addHeaderRow,
	addManualModel,
	addressMoved,
	applyPreset,
	canSave,
	connectionIssues,
	draftParams,
	editEndpointDraft,
	keyAlerts,
	LOCAL_SERVER_PRESETS,
	limitIssues,
	mergeProbe,
	newEndpointDraft,
	parseTags,
	probeExtras,
	probeTarget,
	removeHeaderRow,
	updateHeaderRow,
} from "./customEndpointDraft";

function model(id: string, patch: Partial<CustomEndpointModel> = {}): CustomEndpointModel {
	return {
		id,
		contextWindow: null,
		maxTokens: null,
		reasoning: null,
		vision: null,
		tools: null,
		chatTemplateThinking: null,
		...patch,
	};
}

const saved = {
	id: "custom-gw",
	name: "Gateway",
	baseUrl: "https://gw.example/v1",
	hasApiKey: true,
	limits: { maxConcurrency: null, timeoutSeconds: null, maxRetries: null },
	gateway: null,
	apiKeyReference: null,
	headers: [],
	network: { systemCertificates: false, caFile: null, proxy: null },
	managed: null,
	models: [model("a", { contextWindow: 8_000, maxTokens: 2_000 }), model("manual")],
};

test("a preset fills the local address and only replaces an empty or preset name", () => {
	const [ollama, lmStudio] = LOCAL_SERVER_PRESETS;
	const filled = applyPreset(newEndpointDraft(), ollama);
	expect(filled).toMatchObject({ name: "Ollama", baseUrl: "http://127.0.0.1:11434/v1" });
	expect(applyPreset(filled, lmStudio).name).toBe("LM Studio");
	expect(applyPreset({ ...filled, name: "My box" }, lmStudio).name).toBe("My box");
});

test("the first detection selects every model and later ones keep the user's choice", () => {
	const first = mergeProbe(newEndpointDraft(), {
		ok: true,
		baseUrl: "http://h/v1",
		models: [model("a"), model("b")],
	});
	expect(first.models.map((row) => row.selected)).toEqual([true, true]);
	const unselected = {
		...first,
		models: first.models.map((row) => ({ ...row, selected: row.id === "a" })),
	};
	const again = mergeProbe(unselected, {
		ok: true,
		baseUrl: "http://h/v1",
		models: [model("a"), model("b"), model("c")],
	});
	expect(again.models.map((row) => [row.id, row.selected])).toEqual([
		["a", true],
		["b", false],
		["c", false],
	]);
});

test("re-detecting keeps saved facts the server doesn't report and refreshes the ones it does", () => {
	const draft = editEndpointDraft(saved);
	const merged = mergeProbe(draft, {
		ok: true,
		baseUrl: "https://gw.example/v1",
		models: [model("a", { tools: true })],
	});
	expect(merged.models).toEqual([
		{ ...model("a", { contextWindow: 8_000, maxTokens: 2_000, tools: true }), selected: true },
		{ ...model("manual"), selected: true },
	]);
	const smaller = mergeProbe(draft, {
		ok: true,
		baseUrl: "https://gw.example/v1",
		models: [model("a", { contextWindow: 4_096 })],
	});
	expect(smaller.models[0]).toMatchObject({ contextWindow: 4_096, maxTokens: null });
});

test("an unchanged saved address stays as written while a typed one takes the probed form", () => {
	const draft = editEndpointDraft(saved);
	expect(
		mergeProbe(draft, { ok: true, baseUrl: "https://gw.example/v1", models: [] }).baseUrl,
	).toBe("https://gw.example/v1");
	const typed = { ...newEndpointDraft(), baseUrl: "http://127.0.0.1:11434" };
	expect(
		mergeProbe(typed, { ok: true, baseUrl: "http://127.0.0.1:11434/v1", models: [] }).baseUrl,
	).toBe("http://127.0.0.1:11434/v1");
});

test("Detect sends a typed key, else the saved endpoint, and never a key marked for removal", () => {
	const draft = editEndpointDraft(saved);
	expect(probeTarget(draft)).toEqual({ endpointId: "custom-gw" });
	expect(probeTarget({ ...draft, apiKey: "sk-new" })).toEqual({ apiKey: "sk-new" });
	expect(probeTarget({ ...draft, removeKey: true })).toEqual({});
	expect(probeTarget(newEndpointDraft())).toEqual({ apiKey: "" });
});

test("save sends selected models, keeps a saved key unless replaced or removed", () => {
	const draft = {
		...editEndpointDraft(saved),
		models: editEndpointDraft(saved).models.map((row) => ({ ...row, selected: row.id === "a" })),
	};
	expect(draftParams(draft, 0)).toEqual({
		id: "custom-gw",
		name: "Gateway",
		baseUrl: "https://gw.example/v1",
		models: [model("a", { contextWindow: 8_000, maxTokens: 2_000 })],
	});
	expect(draftParams({ ...draft, apiKey: " sk-new " }, 0).apiKey).toBe("sk-new");
	expect(draftParams({ ...draft, removeKey: true }, 0).apiKey).toBe("");
	expect(draftParams(newEndpointDraft(), 0).apiKey).toBe("");
});

test("manual ids are trimmed, deduplicated and selected; saving needs a name, address and a model", () => {
	let draft = addManualModel(newEndpointDraft(), "  qwen3-coder ");
	draft = addManualModel(draft, "qwen3-coder");
	expect(draft.models).toEqual([{ ...model("qwen3-coder"), selected: true }]);
	expect(canSave(draft)).toBe(false);
	expect(canSave({ ...draft, name: "GW", baseUrl: "https://gw.example/v1" })).toBe(true);
	expect(
		canSave({
			...draft,
			name: "GW",
			baseUrl: "https://gw.example/v1",
			models: draft.models.map((row) => ({ ...row, selected: false })),
		}),
	).toBe(false);
});

test("re-detecting keeps an unreported capability and clears one the server now denies", () => {
	const draft = editEndpointDraft({
		...saved,
		models: [model("a", { reasoning: true, vision: true })],
	});
	const unreported = mergeProbe(draft, {
		ok: true,
		baseUrl: "https://gw.example/v1",
		models: [model("a")],
	});
	expect(unreported.models[0]).toMatchObject({ reasoning: true, vision: true });
	const denied = mergeProbe(draft, {
		ok: true,
		baseUrl: "https://gw.example/v1",
		models: [model("a", { reasoning: false, vision: false })],
	});
	expect(denied.models[0]).toMatchObject({ reasoning: false, vision: false });
});

test("an edit can change the path but not the origin of a saved server", () => {
	const draft = editEndpointDraft(saved);
	expect(addressMoved({ ...draft, baseUrl: "https://gw.example/openai/v1" })).toBe(false);
	expect(addressMoved({ ...draft, baseUrl: "https://other.example/v1" })).toBe(true);
	expect(canSave({ ...draft, baseUrl: "https://other.example/v1" })).toBe(false);
	expect(addressMoved({ ...newEndpointDraft(), baseUrl: "https://other.example/v1" })).toBe(false);
});

test("a re-detect keeps a chat-template thinking fact the server stopped reporting", () => {
	const draft = editEndpointDraft({
		...saved,
		models: [model("hybrid", { reasoning: true, chatTemplateThinking: true })],
	});
	const merged = mergeProbe(draft, {
		ok: true,
		baseUrl: saved.baseUrl,
		models: [model("hybrid", { contextWindow: 16_384 })],
	});
	expect(merged.models[0]).toMatchObject({ chatTemplateThinking: true, contextWindow: 16_384 });
	expect(draftParams(merged, 0).models[0]?.chatTemplateThinking).toBe(true);
	expect(addManualModel(merged, "new").models.at(-1)?.chatTemplateThinking).toBeNull();
});

test("fork: request limits load from the endpoint, validate as whole numbers and save only when sent", () => {
	const draft = editEndpointDraft({
		...saved,
		limits: { maxConcurrency: 2, timeoutSeconds: null, maxRetries: 0 },
	});
	expect(draft).toMatchObject({ concurrencyText: "2", timeoutText: "", retriesText: "0" });
	expect(draftParams(draft, 103).limits).toEqual({
		maxConcurrency: 2,
		timeoutSeconds: null,
		maxRetries: 0,
	});
	expect(draftParams(draft, 0)).not.toHaveProperty("limits");
	const cleared = { ...draft, concurrencyText: " ", retriesText: "" };
	expect(draftParams(cleared, 103).limits).toEqual({
		maxConcurrency: null,
		timeoutSeconds: null,
		maxRetries: null,
	});
	for (const [field, text] of [
		["concurrencyText", "0"],
		["concurrencyText", "1.5"],
		["timeoutText", "5"],
		["timeoutText", "abc"],
		["retriesText", "11"],
	] as const) {
		const bad = { ...draft, [field]: text };
		expect(Object.values(limitIssues(bad))).toContain(true);
		expect(canSave(bad)).toBe(false);
	}
	expect(limitIssues(draft)).toEqual({
		concurrency: false,
		timeout: false,
		retries: false,
		tags: false,
	});
	expect(canSave(draft)).toBe(true);
});

const LITELLM_PROBE = {
	ok: true as const,
	baseUrl: "https://llm.corp.example/v1",
	models: [
		model("qwen", {
			contextWindow: 131_072,
			tools: true,
			cost: { input: 0.2, output: 0.6, cacheRead: 0, cacheWrite: 0 },
		}),
	],
	gateway: {
		kind: "litellm" as const,
		key: {
			alias: "till-laptop",
			spend: 1.5,
			maxBudget: 50,
			budgetResetAt: null,
			expiresAt: null,
			rpmLimit: 60,
			tpmLimit: null,
			maxParallelRequests: 4,
		},
	},
};

test("fork: a LiteLLM probe turns on gateway options, fills parallel requests from the key and keeps prices", () => {
	const merged = mergeProbe(newEndpointDraft(), LITELLM_PROBE);
	expect(merged.gateway).toEqual({ sessionAffinity: true, tagsText: "thinkrail" });
	expect(merged.concurrencyText).toBe("4");
	expect(merged.models[0]?.cost).toEqual({ input: 0.2, output: 0.6, cacheRead: 0, cacheWrite: 0 });
	const typed = mergeProbe({ ...newEndpointDraft(), concurrencyText: "2" }, LITELLM_PROBE);
	expect(typed.concurrencyText).toBe("2");
	const edited = mergeProbe(
		{ ...merged, gateway: { sessionAffinity: false, tagsText: "team:ai" } },
		LITELLM_PROBE,
	);
	expect(edited.gateway).toEqual({ sessionAffinity: false, tagsText: "team:ai" });
	const plain = mergeProbe(newEndpointDraft(), {
		ok: true,
		baseUrl: "http://h/v1",
		models: [model("a")],
	});
	expect(plain.gateway).toBeNull();
	const params = draftParams({ ...merged, name: "Gateway" }, 104);
	expect(params.gateway).toEqual({ kind: "litellm", sessionAffinity: true, tags: ["thinkrail"] });
	expect(draftParams({ ...merged, name: "Gateway" }, 103)).not.toHaveProperty("gateway");
	expect(draftParams({ ...plain, name: "Local" }, 104).gateway).toBeNull();
});

test("fork: gateway tags split on commas and spaces, dedupe, and block a save when invalid", () => {
	expect(parseTags(" team:ai, thinkrail  env/prod,,thinkrail ")).toEqual([
		"team:ai",
		"thinkrail",
		"env/prod",
	]);
	const draft = mergeProbe({ ...newEndpointDraft(), name: "Gateway" }, LITELLM_PROBE);
	expect(canSave(draft)).toBe(true);
	const bad = { ...draft, gateway: { sessionAffinity: true, tagsText: "ok, not;ok" } };
	expect(limitIssues(bad).tags).toBe(true);
	expect(canSave(bad)).toBe(false);
	const many = {
		...draft,
		gateway: {
			sessionAffinity: true,
			tagsText: Array.from({ length: 11 }, (_, i) => `t${i}`).join(","),
		},
	};
	expect(limitIssues(many).tags).toBe(true);
});

test("fork: key alerts flag a nearly spent or spent budget and a key close to or past expiry", () => {
	const now = Date.parse("2026-11-01T12:00:00Z");
	const key = LITELLM_PROBE.gateway.key;
	expect(keyAlerts(key, now)).toEqual({ budget: null, expiry: null });
	expect(keyAlerts({ ...key, spend: 45 }, now).budget).toBe("near");
	expect(keyAlerts({ ...key, spend: 50 }, now).budget).toBe("exhausted");
	expect(keyAlerts({ ...key, maxBudget: null, spend: 900 }, now).budget).toBeNull();
	expect(keyAlerts({ ...key, expiresAt: "2026-11-05T00:00:00Z" }, now).expiry).toBe("soon");
	expect(keyAlerts({ ...key, expiresAt: "2026-12-30T00:00:00Z" }, now).expiry).toBeNull();
	expect(keyAlerts({ ...key, expiresAt: "2026-10-30T00:00:00Z" }, now).expiry).toBe("expired");
	expect(keyAlerts({ ...key, expiresAt: "not a date" }, now).expiry).toBeNull();
});

const corporate = {
	...saved,
	apiKeyReference: "!az account get-access-token --query accessToken -o tsv",
	headers: [
		{ name: "Ocp-Apim-Subscription-Key", value: null },
		{ name: "X-Org", value: "$ORG_ID" },
	],
	network: { systemCertificates: true, caFile: null, proxy: "http://proxy.corp:3128" },
};

test("fork: an edit shows a saved key reference but only a changed key counts as typed", () => {
	const draft = editEndpointDraft(corporate);
	expect(draft.apiKey).toBe(corporate.apiKeyReference);
	expect(probeTarget(draft)).toEqual({ endpointId: "custom-gw" });
	expect(probeTarget({ ...draft, apiKey: "!gcloud auth print-access-token" })).toEqual({
		apiKey: "!gcloud auth print-access-token",
	});
	expect(draftParams(draft, 105).apiKey).toBe(corporate.apiKeyReference);
});

test("fork: headers keep saved secrets as null, validate, and ride Detect and save from v105", () => {
	let draft = editEndpointDraft(corporate);
	expect(draft.headers.map((row) => row.value)).toEqual(["", "$ORG_ID"]);
	expect(draftParams(draft, 105).headers).toEqual([
		{ name: "Ocp-Apim-Subscription-Key", value: null },
		{ name: "X-Org", value: null },
	]);
	const retyped = updateHeaderRow(draft, draft.headers[1]?.key ?? 0, { value: "$OTHER_ORG" });
	expect(draftParams(retyped, 105).headers?.[1]).toEqual({ name: "X-Org", value: "$OTHER_ORG" });
	expect(draftParams(draft, 105).network).toEqual(corporate.network);
	expect(probeExtras(draft, 105)).toEqual({
		endpointId: "custom-gw",
		headers: [
			{ name: "Ocp-Apim-Subscription-Key", value: null },
			{ name: "X-Org", value: null },
		],
		network: corporate.network,
	});
	expect(probeExtras(draft, 104)).toEqual({});
	expect(draftParams(draft, 104)).not.toHaveProperty("headers");
	expect(connectionIssues(draft)).toEqual({ headers: false, proxy: false });
	draft = addHeaderRow(draft);
	const added = draft.headers.at(-1);
	if (!added) throw new Error("no row");
	expect(connectionIssues(draft).headers).toBe(true);
	expect(canSave(draft)).toBe(false);
	draft = updateHeaderRow(draft, added.key, { name: "x-org", value: "dup" });
	expect(connectionIssues(draft).headers).toBe(true);
	draft = updateHeaderRow(draft, added.key, { name: "CF-Access-Client-Id", value: " id-1 " });
	expect(connectionIssues(draft).headers).toBe(false);
	expect(draftParams(draft, 105).headers?.at(-1)).toEqual({
		name: "CF-Access-Client-Id",
		value: "id-1",
	});
	draft = removeHeaderRow(draft, added.key);
	expect(draft.headers).toHaveLength(2);
	expect(connectionIssues({ ...draft, proxy: "http://u:p@proxy:1" }).proxy).toBe(true);
	const withCommand = addHeaderRow(draft);
	const commandRow = withCommand.headers.at(-1);
	if (!commandRow) throw new Error("no row");
	expect(
		connectionIssues(updateHeaderRow(withCommand, commandRow.key, { name: "X-Tok", value: "!id" }))
			.headers,
	).toBe(true);
	expect(canSave({ ...draft, proxy: "socks5://proxy:1" })).toBe(false);
});

test("fork: a re-detect keeps the API the user picked and older hosts never get one", () => {
	const draft = editEndpointDraft({
		...corporate,
		models: [model("claude", { api: "openai-responses" })],
	});
	const merged = mergeProbe(draft, {
		ok: true,
		baseUrl: "https://gw.example/v1",
		models: [
			model("claude", { api: "anthropic-messages" }),
			model("new", { api: "anthropic-messages" }),
		],
	});
	expect(merged.models.map((row) => row.api)).toEqual(["openai-responses", "anthropic-messages"]);
	const completions = mergeProbe(editEndpointDraft({ ...corporate, models: [model("claude")] }), {
		ok: true,
		baseUrl: "https://gw.example/v1",
		models: [model("claude", { api: "anthropic-messages" })],
	});
	expect(completions.models[0]?.api).toBeUndefined();
	const params = draftParams(
		{ ...merged, models: merged.models.map((row) => ({ ...row, selected: true })) },
		104,
	);
	expect(params.models.every((row) => !("api" in row))).toBe(true);
	expect(draftParams(merged, 105).models[0]?.api).toBe("openai-responses");
});
