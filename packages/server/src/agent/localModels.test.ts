import { afterEach, beforeEach, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { InMemoryCredentialStore } from "@earendil-works/pi-ai";
import { ModelRuntime } from "@earendil-works/pi-coding-agent";
import { parse } from "jsonc-parser";
import {
	listLocalEndpoints,
	normalizeLocalBaseUrl,
	parseModelList,
	probeLocalModels,
	removeLocalEndpoint,
	resolveProbeKey,
	saveLocalEndpoint,
	testLocalModel,
} from "./localModels";
import { setModelsFilePublisher } from "./modelsFile";
import { configurePiRuntime, settledAvailableModels } from "./piRuntime";
import { requestLoad } from "./requestLimits";

let directory: string;
let path: string;
let priorAgentDir: string | undefined;
let priorOffline: string | undefined;
let priorDataDir: string | undefined;
let server: ReturnType<typeof Bun.serve> | undefined;

beforeEach(() => {
	directory = mkdtempSync(join(tmpdir(), "thinkrail-local-models-"));
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

function serve(routes: Record<string, (request: Request) => unknown>, auth?: string) {
	server = Bun.serve({
		port: 0,
		hostname: "127.0.0.1",
		async fetch(request) {
			if (auth && request.headers.get("authorization") !== `Bearer ${auth}`)
				return new Response("no", { status: 401 });
			const route = routes[`${request.method} ${new URL(request.url).pathname}`];
			if (!route) return new Response("missing", { status: 404 });
			return Response.json(await route(request));
		},
	});
	return `http://127.0.0.1:${server.port}`;
}

test("base URLs gain /v1 for known servers and lose endpoint suffixes", () => {
	expect(normalizeLocalBaseUrl("llama.cpp", "http://127.0.0.1:8080")).toBe(
		"http://127.0.0.1:8080/v1",
	);
	expect(normalizeLocalBaseUrl("vllm", "http://gpu:8000/v1/models/")).toBe("http://gpu:8000/v1");
	expect(normalizeLocalBaseUrl("ollama", " http://localhost:11434/v1/chat/completions ")).toBe(
		"http://localhost:11434/v1",
	);
	expect(normalizeLocalBaseUrl("openai-compatible", "https://proxy.lan/openai")).toBe(
		"https://proxy.lan/openai",
	);
	expect(() => normalizeLocalBaseUrl("vllm", "localhost:8000")).toThrow();
	expect(() => normalizeLocalBaseUrl("vllm", "file:///etc/passwd")).toThrow();
});

test("probe keys resolve env references and never run commands", () => {
	process.env.THINKRAIL_TEST_KEY = "from-env";
	try {
		expect(resolveProbeKey("$THINKRAIL_TEST_KEY")).toBe("from-env");
		expect(resolveProbeKey(["$", "{THINKRAIL_TEST_KEY}"].join(""))).toBe("from-env");
		expect(resolveProbeKey("!cat secret")).toBeUndefined();
		expect(resolveProbeKey("  ")).toBeUndefined();
		expect(resolveProbeKey("literal")).toBe("literal");
	} finally {
		delete process.env.THINKRAIL_TEST_KEY;
	}
});

test("model listings read vLLM and llama.cpp context hints and skip non-text models", () => {
	expect(
		parseModelList({
			data: [
				{ id: "Qwen/Qwen3-Coder-30B", max_model_len: 65_536 },
				{
					id: "gemma",
					meta: { n_ctx: 32_768, n_ctx_train: 131_072 },
					architecture: { input_modalities: ["text", "image"] },
				},
				{ id: "router-model", status: { value: "loaded", args: ["--port", "1", "-c", "16384"] } },
				{ id: "judge", architecture: { output_modalities: ["decisions"] } },
				{ id: "plain" },
				{ id: "plain" },
				{ nope: true },
			],
		}),
	).toEqual([
		{
			id: "Qwen/Qwen3-Coder-30B",
			contextWindow: 65_536,
			reasoning: false,
			vision: false,
			tools: null,
		},
		{ id: "gemma", contextWindow: 32_768, reasoning: false, vision: true, tools: null },
		{ id: "router-model", contextWindow: 16_384, reasoning: false, vision: false, tools: null },
		{ id: "plain", contextWindow: null, reasoning: false, vision: false, tools: null },
	]);
	expect(parseModelList("garbage")).toEqual([]);
});

test("a llama.cpp probe reads /props for context and thinking support", async () => {
	const base = serve(
		{
			"GET /v1/models": () => ({ data: [{ id: "qwen3-8b" }] }),
			"GET /props": () => ({
				default_generation_settings: { n_ctx: 40_960 },
				total_slots: 4,
				chat_template_caps: { supports_tools: true },
				chat_template: "{% if enable_thinking %}...",
			}),
		},
		"sekret",
	);
	expect(await probeLocalModels({ kind: "llama.cpp", baseUrl: base, apiKey: "sekret" })).toEqual({
		baseUrl: `${base}/v1`,
		slots: 4,
		models: [
			{ id: "qwen3-8b", contextWindow: 40_960, reasoning: true, vision: false, tools: true },
		],
	});
	await expect(probeLocalModels({ kind: "llama.cpp", baseUrl: base })).rejects.toThrow("HTTP 401");
});

test("an Ollama probe reads /api/show capabilities", async () => {
	const base = serve({
		"GET /v1/models": () => ({ data: [{ id: "qwen3:8b" }] }),
		"POST /api/show": () => ({
			model_info: { "qwen3.context_length": 40_960 },
			capabilities: ["completion", "tools", "thinking"],
		}),
	});
	expect((await probeLocalModels({ kind: "ollama", baseUrl: base })).models).toEqual([
		{ id: "qwen3:8b", contextWindow: 40_960, reasoning: true, vision: false, tools: true },
	]);
});

test("an unreachable server reports a short failure", async () => {
	const base = serve({});
	server?.stop(true);
	server = undefined;
	await expect(probeLocalModels({ kind: "vllm", baseUrl: base })).rejects.toThrow(
		/Couldn't list models at .*\/v1\/models/,
	);
});

test("saving writes a chat-completions provider pi serves, keeps unrelated config, and removes cleanly", async () => {
	writeFileSync(
		path,
		'﻿{\n  // keep me\n  "providers": {\n    "openai": { "modelOverrides": { "gpt-5.5": { "contextWindow": 1000000 } } }\n  }\n}\n',
	);
	const pi = await runtime();
	let published = 0;
	setModelsFilePublisher(() => {
		published++;
	});

	const saved = await saveLocalEndpoint({
		kind: "vllm",
		name: "GPU box vLLM",
		baseUrl: "http://gpu.lan:8000",
		maxConcurrency: 2,
		timeoutSeconds: null,
		maxRetries: null,
		models: [
			{
				id: "Qwen/Qwen3-Coder-30B",
				contextWindow: 65_536,
				maxTokens: 16384,
				reasoning: true,
				vision: false,
			},
			{ id: "small", contextWindow: 8_192, maxTokens: 2048, reasoning: false, vision: true },
		],
	});
	expect(saved).toEqual([
		{
			id: "local-gpu-box-vllm",
			name: "GPU box vLLM",
			baseUrl: "http://gpu.lan:8000/v1",
			hasApiKey: false,
			kind: "vllm",
			maxConcurrency: 2,
			timeoutSeconds: null,
			maxRetries: null,
			models: [
				{
					id: "Qwen/Qwen3-Coder-30B",
					contextWindow: 65_536,
					maxTokens: 16384,
					reasoning: true,
					vision: false,
				},
				{ id: "small", contextWindow: 8_192, maxTokens: 2048, reasoning: false, vision: true },
			],
		},
	]);
	expect(published).toBe(1);

	const source = readFileSync(path, "utf8");
	expect(source.startsWith("﻿")).toBe(true);
	expect(source).toContain("// keep me");
	const config = parse(source.slice(1), [], { allowTrailingComma: true });
	expect(config.providers.openai.modelOverrides["gpt-5.5"].contextWindow).toBe(1_000_000);
	const provider = config.providers["local-gpu-box-vllm"];
	expect(provider).toMatchObject({
		api: "openai-completions",
		compat: { supportsDeveloperRole: false, maxTokensField: "max_tokens" },
	});
	expect(provider.models[0]).toMatchObject({
		maxTokens: 16_384,
		compat: { thinkingFormat: "qwen-chat-template" },
	});
	expect(provider.models[1]).toMatchObject({ input: ["text", "image"], maxTokens: 2_048 });

	const available = settledAvailableModels(pi).filter((m) => m.provider === "local-gpu-box-vllm");
	expect(available.map((m) => [m.id, m.contextWindow, m.baseUrl])).toEqual([
		["Qwen/Qwen3-Coder-30B", 65_536, "http://gpu.lan:8000/v1"],
		["small", 8_192, "http://gpu.lan:8000/v1"],
	]);

	await saveLocalEndpoint({
		kind: "vllm",
		name: "GPU box vLLM",
		baseUrl: "http://gpu.lan:8000/v1",
		maxConcurrency: null,
		timeoutSeconds: null,
		maxRetries: null,
		models: [
			{ id: "small", contextWindow: 8_192, maxTokens: 2048, reasoning: false, vision: false },
		],
	});
	expect((await listLocalEndpoints()).map((endpoint) => endpoint.id)).toEqual([
		"local-gpu-box-vllm",
		"local-gpu-box-vllm-2",
	]);

	const edited = await saveLocalEndpoint({
		id: "local-gpu-box-vllm",
		kind: "vllm",
		name: "GPU box",
		baseUrl: "http://gpu.lan:8000/v1",
		apiKey: "$VLLM_KEY",
		maxConcurrency: null,
		timeoutSeconds: null,
		maxRetries: null,
		models: [
			{ id: "small", contextWindow: 8_192, maxTokens: 2048, reasoning: false, vision: true },
		],
	});
	expect(edited[0]).toMatchObject({
		id: "local-gpu-box-vllm",
		name: "GPU box",
		hasApiKey: true,
		maxConcurrency: null,
		timeoutSeconds: null,
		maxRetries: null,
	});
	expect(edited[1]).toMatchObject({
		id: "local-gpu-box-vllm-2",
		kind: "vllm",
		maxConcurrency: null,
		timeoutSeconds: null,
		maxRetries: null,
	});

	await removeLocalEndpoint("local-gpu-box-vllm");
	await removeLocalEndpoint("local-gpu-box-vllm-2");
	await removeLocalEndpoint("openai");
	expect(await listLocalEndpoints()).toEqual([]);
	const after = parse(readFileSync(path, "utf8").slice(1), [], { allowTrailingComma: true });
	expect(Object.keys(after.providers)).toEqual(["openai"]);
	expect(settledAvailableModels(pi).some((m) => m.provider.startsWith("local-"))).toBe(false);
	expect(JSON.parse(readFileSync(join(directory, "data", "local-models.json"), "utf8"))).toEqual({
		version: 1,
		endpoints: {},
	});
});

test("drafts are validated before anything is written", async () => {
	await runtime();
	const draft = {
		kind: "llama.cpp" as const,
		name: "x",
		baseUrl: "http://127.0.0.1:8080",
		maxConcurrency: null,
		timeoutSeconds: null,
		maxRetries: null,
		models: [{ id: "m", contextWindow: 100, maxTokens: 1024, reasoning: false, vision: false }],
	};
	await expect(saveLocalEndpoint(draft)).rejects.toThrow("Context windows");
	await expect(saveLocalEndpoint({ ...draft, models: [] })).rejects.toThrow("at least one");
	const ok = { id: "m", contextWindow: 4096, maxTokens: 1024, reasoning: false, vision: false };
	await expect(saveLocalEndpoint({ ...draft, models: [ok], maxConcurrency: 0 })).rejects.toThrow(
		"Parallel requests",
	);
	await expect(
		saveLocalEndpoint({ ...draft, models: [{ ...ok, maxTokens: 8192 }] }),
	).rejects.toThrow("Max output");
	await expect(saveLocalEndpoint({ ...draft, models: [ok], timeoutSeconds: 1 })).rejects.toThrow(
		"timeout",
	);
	await expect(saveLocalEndpoint({ ...draft, models: [ok], maxRetries: 99 })).rejects.toThrow(
		"Retries",
	);
	await expect(
		saveLocalEndpoint({
			...draft,
			id: "local-missing",
			models: [{ id: "m", contextWindow: 4096, maxTokens: 1024, reasoning: false, vision: false }],
		}),
	).rejects.toThrow("no longer");
	expect(() => statSync(path)).toThrow();
});

function sse(chunks: unknown[]): string {
	return `${chunks.map((chunk) => `data: ${JSON.stringify(chunk)}\n\n`).join("")}data: [DONE]\n\n`;
}

const chunk = (choices: unknown[], extra: Record<string, unknown> = {}) => ({
	id: "c1",
	object: "chat.completion.chunk",
	created: 1,
	model: "m",
	choices,
	...extra,
});

function fakeChatServer(
	onChat: (body: Record<string, unknown>) => Promise<string | Response> | string | Response,
) {
	server = Bun.serve({
		port: 0,
		hostname: "127.0.0.1",
		async fetch(request) {
			const url = new URL(request.url);
			if (request.method === "POST" && url.pathname === "/v1/chat/completions") {
				const body = (await request.json()) as Record<string, unknown>;
				const reply = await onChat(body);
				if (reply instanceof Response) return reply;
				return new Response(reply, { headers: { "content-type": "text/event-stream" } });
			}
			return new Response("missing", { status: 404 });
		},
	});
	return `http://127.0.0.1:${server.port}`;
}

const textReply = (text: string) =>
	sse([
		chunk([{ index: 0, delta: { role: "assistant", content: text } }]),
		chunk([{ index: 0, delta: {}, finish_reason: "stop" }]),
		chunk([], { usage: { prompt_tokens: 5, completion_tokens: 2, total_tokens: 7 } }),
	]);

test("the agent check reports a chat reply and a real tool call through pi", async () => {
	const requests: Record<string, unknown>[] = [];
	const base = fakeChatServer((body) => {
		requests.push(body);
		if (!Array.isArray(body.tools)) return textReply("OK");
		return sse([
			chunk([
				{
					index: 0,
					delta: {
						role: "assistant",
						tool_calls: [
							{
								index: 0,
								id: "call_1",
								type: "function",
								function: { name: "get_current_time", arguments: '{"timezone":"UTC"}' },
							},
						],
					},
				},
			]),
			chunk([{ index: 0, delta: {}, finish_reason: "tool_calls" }]),
		]);
	});
	await runtime();
	await saveLocalEndpoint({
		kind: "vllm",
		name: "box",
		baseUrl: base,
		maxConcurrency: null,
		timeoutSeconds: null,
		maxRetries: null,
		models: [{ id: "m", contextWindow: 8_192, maxTokens: 2_048, reasoning: false, vision: false }],
	});
	const result = await testLocalModel("local-box", "m");
	expect(result).toMatchObject({ chat: "ok", tools: "ok", message: null });
	expect(result.latencyMs).toBeGreaterThanOrEqual(0);
	expect(requests).toHaveLength(2);
	expect(requests[0]).toMatchObject({ model: "m", max_tokens: 256 });
	expect(requests[0]).not.toHaveProperty("store");
	expect(requests[1]?.tools).toEqual([
		expect.objectContaining({
			type: "function",
			function: expect.objectContaining({ name: "get_current_time" }),
		}),
	]);
});

test("the agent check tells a text-only answer and a server error apart", async () => {
	let fail = false;
	const base = fakeChatServer(() => {
		if (fail) return Response.json({ error: { message: "model not loaded" } }, { status: 503 });
		return textReply("It is noon.");
	});
	await runtime();
	await saveLocalEndpoint({
		kind: "llama.cpp",
		name: "box",
		baseUrl: base,
		maxConcurrency: null,
		timeoutSeconds: null,
		maxRetries: null,
		models: [{ id: "m", contextWindow: 8_192, maxTokens: 2_048, reasoning: false, vision: false }],
	});
	expect(await testLocalModel("local-box", "m")).toMatchObject({
		chat: "ok",
		tools: "failed",
		message: null,
	});
	fail = true;
	const failed = await testLocalModel("local-box", "m");
	expect(failed).toMatchObject({ chat: "failed", tools: "skipped" });
	expect(failed.message).toContain("model not loaded");
	await expect(testLocalModel("local-box", "missing")).rejects.toThrow("not available");
});

test("an endpoint's parallel-request limit queues the extra requests", async () => {
	let active = 0;
	let peak = 0;
	const releases: (() => void)[] = [];
	const base = fakeChatServer(async () => {
		active++;
		peak = Math.max(peak, active);
		await new Promise<void>((resolve) => releases.push(resolve));
		active--;
		return textReply("OK");
	});
	const pi = await runtime();
	await saveLocalEndpoint({
		kind: "vllm",
		name: "box",
		baseUrl: base,
		maxConcurrency: 1,
		timeoutSeconds: null,
		maxRetries: null,
		models: [{ id: "m", contextWindow: 8_192, maxTokens: 2_048, reasoning: false, vision: false }],
	});
	const model = pi.getAvailableSnapshot().find((m) => m.provider === "local-box");
	if (!model) throw new Error("missing model");
	const ask = () =>
		pi.completeSimple(model, {
			messages: [{ role: "user", content: "hi", timestamp: Date.now() }],
		});
	const runs = [ask(), ask(), ask()];
	await new Promise((resolve) => setTimeout(resolve, 100));
	expect(requestLoad("local-box")).toEqual({ active: 1, queued: 2 });
	while (releases.length > 0 || requestLoad("local-box").active > 0) {
		releases.shift()?.();
		await new Promise((resolve) => setTimeout(resolve, 20));
	}
	const results = await Promise.all(runs);
	expect(results.map((message) => message.stopReason)).toEqual(["stop", "stop", "stop"]);
	expect(peak).toBe(1);
	expect(requestLoad("local-box")).toEqual({ active: 0, queued: 0 });
});

test("an endpoint's saved retries and timeout reach pi's request options", async () => {
	let calls = 0;
	const base = fakeChatServer(() => {
		calls++;
		if (calls < 3) return Response.json({ error: { message: "loading model" } }, { status: 503 });
		return textReply("OK");
	});
	const pi = await runtime();
	await saveLocalEndpoint({
		kind: "llama.cpp",
		name: "box",
		baseUrl: base,
		maxConcurrency: null,
		timeoutSeconds: 30,
		maxRetries: 2,
		models: [{ id: "m", contextWindow: 8_192, maxTokens: 2_048, reasoning: false, vision: false }],
	});
	expect((await listLocalEndpoints())[0]).toMatchObject({ timeoutSeconds: 30, maxRetries: 2 });
	const model = pi.getAvailableSnapshot().find((m) => m.provider === "local-box");
	if (!model) throw new Error("missing model");
	const reply = await pi.completeSimple(model, {
		messages: [{ role: "user", content: "hi", timestamp: Date.now() }],
	});
	expect(reply.stopReason).toBe("stop");
	expect(calls).toBe(3);

	calls = 0;
	await saveLocalEndpoint({
		id: "local-box",
		kind: "llama.cpp",
		name: "box",
		baseUrl: base,
		maxConcurrency: null,
		timeoutSeconds: null,
		maxRetries: 0,
		models: [{ id: "m", contextWindow: 8_192, maxTokens: 2_048, reasoning: false, vision: false }],
	});
	const failed = await pi.completeSimple(model, {
		messages: [{ role: "user", content: "hi", timestamp: Date.now() }],
	});
	expect(failed.stopReason).toBe("error");
	expect(calls).toBe(1);
}, 20_000);
