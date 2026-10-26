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
} from "./localModels";
import { setModelsFilePublisher } from "./modelsFile";
import { configurePiRuntime, settledAvailableModels } from "./piRuntime";

let directory: string;
let path: string;
let priorAgentDir: string | undefined;
let priorOffline: string | undefined;
let server: ReturnType<typeof Bun.serve> | undefined;

beforeEach(() => {
	directory = mkdtempSync(join(tmpdir(), "thinkrail-local-models-"));
	path = join(directory, "models.json");
	priorAgentDir = process.env.PI_CODING_AGENT_DIR;
	priorOffline = process.env.PI_OFFLINE;
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
		{ id: "Qwen/Qwen3-Coder-30B", contextWindow: 65_536, reasoning: false, vision: false },
		{ id: "gemma", contextWindow: 32_768, reasoning: false, vision: true },
		{ id: "router-model", contextWindow: 16_384, reasoning: false, vision: false },
		{ id: "plain", contextWindow: null, reasoning: false, vision: false },
	]);
	expect(parseModelList("garbage")).toEqual([]);
});

test("a llama.cpp probe reads /props for context and thinking support", async () => {
	const base = serve(
		{
			"GET /v1/models": () => ({ data: [{ id: "qwen3-8b" }] }),
			"GET /props": () => ({
				default_generation_settings: { n_ctx: 40_960 },
				chat_template: "{% if enable_thinking %}...",
			}),
		},
		"sekret",
	);
	expect(await probeLocalModels({ kind: "llama.cpp", baseUrl: base, apiKey: "sekret" })).toEqual({
		baseUrl: `${base}/v1`,
		models: [{ id: "qwen3-8b", contextWindow: 40_960, reasoning: true, vision: false }],
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
		{ id: "qwen3:8b", contextWindow: 40_960, reasoning: true, vision: false },
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
		models: [
			{ id: "Qwen/Qwen3-Coder-30B", contextWindow: 65_536, reasoning: true, vision: false },
			{ id: "small", contextWindow: 8_192, reasoning: false, vision: true },
		],
	});
	expect(saved).toEqual([
		{
			id: "local-gpu-box-vllm",
			name: "GPU box vLLM",
			baseUrl: "http://gpu.lan:8000/v1",
			hasApiKey: false,
			models: [
				{ id: "Qwen/Qwen3-Coder-30B", contextWindow: 65_536, reasoning: true, vision: false },
				{ id: "small", contextWindow: 8_192, reasoning: false, vision: true },
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
		models: [{ id: "small", contextWindow: 8_192, reasoning: false, vision: false }],
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
		models: [{ id: "small", contextWindow: 8_192, reasoning: false, vision: true }],
	});
	expect(edited[0]).toMatchObject({ id: "local-gpu-box-vllm", name: "GPU box", hasApiKey: true });

	await removeLocalEndpoint("local-gpu-box-vllm");
	await removeLocalEndpoint("local-gpu-box-vllm-2");
	await removeLocalEndpoint("openai");
	expect(await listLocalEndpoints()).toEqual([]);
	const after = parse(readFileSync(path, "utf8").slice(1), [], { allowTrailingComma: true });
	expect(Object.keys(after.providers)).toEqual(["openai"]);
	expect(settledAvailableModels(pi).some((m) => m.provider.startsWith("local-"))).toBe(false);
});

test("drafts are validated before anything is written", async () => {
	await runtime();
	const draft = {
		kind: "llama.cpp" as const,
		name: "x",
		baseUrl: "http://127.0.0.1:8080",
		models: [{ id: "m", contextWindow: 100, reasoning: false, vision: false }],
	};
	await expect(saveLocalEndpoint(draft)).rejects.toThrow("Context windows");
	await expect(saveLocalEndpoint({ ...draft, models: [] })).rejects.toThrow("at least one");
	await expect(
		saveLocalEndpoint({
			...draft,
			id: "local-missing",
			models: [{ id: "m", contextWindow: 4096, reasoning: false, vision: false }],
		}),
	).rejects.toThrow("no longer");
	expect(() => statSync(path)).toThrow();
});
