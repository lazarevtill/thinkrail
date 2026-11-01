import { afterEach, beforeEach, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { InMemoryCredentialStore } from "@earendil-works/pi-ai";
import { ModelRuntime } from "@earendil-works/pi-coding-agent";
import type {
	CustomEndpointDraft,
	CustomEndpointGateway,
	CustomEndpointModel,
} from "@thinkrail/contracts";
import {
	customEndpointKeyStatus,
	listCustomEndpoints,
	probeCustomEndpoint,
	saveCustomEndpoint,
} from "./customEndpoints";
import { parseLiteLlmKeyInfo, parseLiteLlmModelInfo } from "./liteLlm";
import { setModelsFilePublisher } from "./modelsFile";
import { configurePiRuntime } from "./piRuntime";

const KEY = "sk-litellm-virtual";

let directory: string;
let path: string;
let priorAgentDir: string | undefined;
let priorOffline: string | undefined;
let priorDataDir: string | undefined;
let server: ReturnType<typeof Bun.serve> | undefined;

beforeEach(() => {
	directory = mkdtempSync(join(tmpdir(), "thinkrail-litellm-"));
	path = join(directory, "models.json");
	priorAgentDir = process.env.PI_CODING_AGENT_DIR;
	priorOffline = process.env.PI_OFFLINE;
	priorDataDir = process.env.THINKRAIL_DATA_DIR;
	process.env.THINKRAIL_DATA_DIR = join(directory, "data");
	process.env.PI_CODING_AGENT_DIR = directory;
	process.env.PI_OFFLINE = "1";
});

afterEach(() => {
	server?.stop(true);
	server = undefined;
	setModelsFilePublisher(null);
	configurePiRuntime(null);
	if (priorAgentDir === undefined) delete process.env.PI_CODING_AGENT_DIR;
	else process.env.PI_CODING_AGENT_DIR = priorAgentDir;
	if (priorDataDir === undefined) delete process.env.THINKRAIL_DATA_DIR;
	else process.env.THINKRAIL_DATA_DIR = priorDataDir;
	if (priorOffline === undefined) delete process.env.PI_OFFLINE;
	else process.env.PI_OFFLINE = priorOffline;
	rmSync(directory, { recursive: true, force: true });
});

async function runtime() {
	const result = await ModelRuntime.create({
		credentials: new InMemoryCredentialStore(),
		allowModelNetwork: false,
	});
	await result.refresh({ allowNetwork: false });
	configurePiRuntime(result);
	return result;
}

const MODEL_INFO = {
	data: [
		{
			model_name: "qwen3-coder",
			litellm_params: { model: "hosted_vllm/qwen" },
			model_info: {
				mode: "chat",
				max_input_tokens: 131_072,
				max_output_tokens: 16_384,
				supports_function_calling: true,
				supports_vision: false,
				supports_reasoning: true,
				input_cost_per_token: 2e-7,
				output_cost_per_token: 6e-7,
			},
		},
		{
			model_name: "qwen3-coder",
			litellm_params: { model: "hosted_vllm/qwen-replica" },
			model_info: {
				mode: "chat",
				max_input_tokens: 65_536,
				supports_function_calling: true,
				supports_vision: true,
			},
		},
		{
			model_name: "claude-sonnet",
			model_info: {
				mode: "chat",
				max_input_tokens: 200_000,
				max_output_tokens: 64_000,
				supports_function_calling: true,
				supports_vision: true,
				input_cost_per_token: 3e-6,
				output_cost_per_token: 1.5e-5,
				cache_read_input_token_cost: 3e-7,
				cache_creation_input_token_cost: 3.75e-6,
			},
		},
		{ model_name: "text-embedding", model_info: { mode: "embedding", max_input_tokens: 8_192 } },
	],
};

const KEY_INFO = {
	key: "hashed",
	info: {
		key_alias: "till-thinkrail",
		spend: 12.5,
		max_budget: 50,
		budget_reset_at: "2026-12-01T00:00:00Z",
		expires: null,
		rpm_limit: 60,
		tpm_limit: 200_000,
		max_parallel_requests: 4,
	},
};

const chatHeaders: Record<string, string | null>[] = [];

function sse(chunks: unknown[]): string {
	return `${chunks.map((chunk) => `data: ${JSON.stringify(chunk)}\n\n`).join("")}data: [DONE]\n\n`;
}

function fakeLiteLlm() {
	chatHeaders.length = 0;
	server = Bun.serve({
		port: 0,
		hostname: "127.0.0.1",
		async fetch(request) {
			if (request.headers.get("authorization") !== `Bearer ${KEY}`)
				return Response.json({ error: { message: "Authentication Error" } }, { status: 401 });
			const route = `${request.method} ${new URL(request.url).pathname}`;
			if (route === "GET /v1/models")
				return Response.json({
					object: "list",
					data: ["qwen3-coder", "claude-sonnet", "text-embedding", "openai/*"].map((id) => ({
						id,
						object: "model",
						owned_by: "openai",
					})),
				});
			if (route === "GET /model/info") return Response.json(MODEL_INFO);
			if (route === "GET /key/info") return Response.json(KEY_INFO);
			if (route === "POST /v1/chat/completions") {
				chatHeaders.push({
					session: request.headers.get("x-litellm-session-id"),
					tags: request.headers.get("x-litellm-tags"),
				});
				const base = {
					id: "c1",
					object: "chat.completion.chunk",
					created: 1,
					model: "qwen3-coder",
				};
				return new Response(
					sse([
						{ ...base, choices: [{ index: 0, delta: { role: "assistant", content: "OK" } }] },
						{ ...base, choices: [{ index: 0, delta: {}, finish_reason: "stop" }] },
					]),
					{ headers: { "content-type": "text/event-stream" } },
				);
			}
			return new Response("missing", { status: 404 });
		},
	});
	return `http://127.0.0.1:${server.port}`;
}

function draft(
	baseUrl: string,
	models: CustomEndpointModel[],
	gateway: CustomEndpointGateway | null,
): CustomEndpointDraft {
	return { name: "Corp LiteLLM", baseUrl, apiKey: KEY, models, gateway };
}

test("a deployment group takes the smallest limits and only capabilities every deployment has", () => {
	const groups = parseLiteLlmModelInfo(MODEL_INFO);
	expect(groups?.get("qwen3-coder")).toEqual({
		mode: "chat",
		contextWindow: 65_536,
		maxTokens: 16_384,
		reasoning: true,
		vision: false,
		tools: true,
		cost: { input: 0.2, output: 0.6, cacheRead: 0, cacheWrite: 0 },
		anthropic: false,
	});
	expect(groups?.get("claude-sonnet")?.cost).toEqual({
		input: 3,
		output: 15,
		cacheRead: 0.3,
		cacheWrite: 3.75,
	});
	expect(parseLiteLlmModelInfo({ data: [{ id: "x" }] })).toBeUndefined();
	expect(parseLiteLlmModelInfo({ object: "list" })).toBeUndefined();
	expect(parseLiteLlmKeyInfo({ info: { spend: -1, max_budget: "lots", rpm_limit: 0 } })).toEqual({
		alias: null,
		spend: null,
		maxBudget: null,
		budgetResetAt: null,
		expiresAt: null,
		rpmLimit: null,
		tpmLimit: null,
		maxParallelRequests: null,
	});
});

test("Detect on a LiteLLM proxy reads its model info and the key's own status", async () => {
	const origin = fakeLiteLlm();
	const probe = await probeCustomEndpoint({ baseUrl: origin, apiKey: KEY });
	if (!probe.ok) throw new Error(probe.reason);
	expect(probe.baseUrl).toBe(`${origin}/v1`);
	expect(probe.models.map((model) => model.id)).toEqual(["qwen3-coder", "claude-sonnet"]);
	expect(probe.models[1]).toMatchObject({
		contextWindow: 200_000,
		maxTokens: 64_000,
		tools: true,
		vision: true,
		cost: { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 },
	});
	expect(probe.gateway).toEqual({
		kind: "litellm",
		key: {
			alias: "till-thinkrail",
			spend: 12.5,
			maxBudget: 50,
			budgetResetAt: "2026-12-01T00:00:00Z",
			expiresAt: null,
			rpmLimit: 60,
			tpmLimit: 200_000,
			maxParallelRequests: 4,
		},
	});
	const unauthorized = await probeCustomEndpoint({ baseUrl: origin, apiKey: "wrong" });
	expect(unauthorized).toEqual({ ok: false, reason: "unauthorized", status: 401 });
});

test("a saved gateway keeps its options and prices, reports the saved key's status, and is refused bad tags", async () => {
	const origin = fakeLiteLlm();
	await runtime();
	const probe = await probeCustomEndpoint({ baseUrl: origin, apiKey: KEY });
	if (!probe.ok) throw new Error(probe.reason);
	const endpoints = await saveCustomEndpoint(
		draft(probe.baseUrl, probe.models, {
			kind: "litellm",
			sessionAffinity: true,
			tags: ["thinkrail", "team:ai"],
		}),
	);
	const saved = endpoints[0];
	if (!saved) throw new Error("not saved");
	expect(saved.gateway).toEqual({
		kind: "litellm",
		sessionAffinity: true,
		tags: ["thinkrail", "team:ai"],
	});
	expect(saved.models[1]?.cost).toEqual({ input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 });
	const file = JSON.parse(readFileSync(path, "utf8"));
	expect(file.providers[saved.id].models[1].cost).toEqual({
		input: 3,
		output: 15,
		cacheRead: 0.3,
		cacheWrite: 3.75,
	});
	expect(JSON.stringify(file)).not.toMatch(/gateway|sessionAffinity|team:ai/);
	expect(await customEndpointKeyStatus(saved.id)).toMatchObject({ spend: 12.5, maxBudget: 50 });

	const { gateway: _kept, ...withoutGateway } = draft(probe.baseUrl, saved.models, null);
	await saveCustomEndpoint({ ...withoutGateway, id: saved.id, name: "Renamed" });
	expect((await listCustomEndpoints())[0]?.gateway?.tags).toEqual(["thinkrail", "team:ai"]);

	await expect(
		saveCustomEndpoint({
			...draft(probe.baseUrl, saved.models, {
				kind: "litellm",
				sessionAffinity: false,
				tags: ["has space"],
			}),
			id: saved.id,
		}),
	).rejects.toThrow("Gateway tags");

	await saveCustomEndpoint({ ...draft(probe.baseUrl, saved.models, null), id: saved.id });
	expect((await listCustomEndpoints())[0]?.gateway).toBeNull();
	expect(await customEndpointKeyStatus(saved.id)).toBeNull();
	expect(await customEndpointKeyStatus("openai")).toBeNull();
});

test("model requests carry the chat's session id and the tags, and a caller's header wins", async () => {
	const origin = fakeLiteLlm();
	const pi = await runtime();
	const probe = await probeCustomEndpoint({ baseUrl: origin, apiKey: KEY });
	if (!probe.ok) throw new Error(probe.reason);
	const [saved] = await saveCustomEndpoint(
		draft(probe.baseUrl, probe.models, {
			kind: "litellm",
			sessionAffinity: true,
			tags: ["thinkrail", "team:ai"],
		}),
	);
	if (!saved) throw new Error("not saved");
	const model = pi.getAvailableSnapshot().find((entry) => entry.provider === saved.id);
	if (!model) throw new Error("missing model");
	const messages = [{ role: "user" as const, content: "hi", timestamp: Date.now() }];
	await pi.completeSimple(model, { messages }, { sessionId: "chat-1" });
	await pi.completeSimple(model, { messages });
	await pi.completeSimple(model, { messages }, { headers: { "x-litellm-tags": "override" } });
	await saveCustomEndpoint({
		...draft(probe.baseUrl, saved.models, { kind: "litellm", sessionAffinity: false, tags: [] }),
		id: saved.id,
	});
	await pi.completeSimple(model, { messages }, { sessionId: "chat-2" });
	expect(chatHeaders).toEqual([
		{ session: "chat-1", tags: "thinkrail,team:ai" },
		{ session: null, tags: "thinkrail,team:ai" },
		{ session: null, tags: "override" },
		{ session: null, tags: null },
	]);
});
