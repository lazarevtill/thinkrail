import { afterEach, beforeEach, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { InMemoryCredentialStore } from "@earendil-works/pi-ai";
import { ModelRuntime } from "@earendil-works/pi-coding-agent";
import type {
	CustomEndpointDraft,
	CustomEndpointLimits,
	CustomEndpointModel,
} from "@thinkrail/contracts";
import { listCustomEndpoints, removeCustomEndpoint, saveCustomEndpoint } from "./customEndpoints";
import { testCustomEndpoint } from "./endpointTest";
import { setModelsFilePublisher } from "./modelsFile";
import { configurePiRuntime } from "./piRuntime";
import { limitedFetch, requestLoad, withEndpointOptions } from "./requestLimits";

let directory: string;
let path: string;
let priorAgentDir: string | undefined;
let priorOffline: string | undefined;
let priorDataDir: string | undefined;
let server: ReturnType<typeof Bun.serve> | undefined;

beforeEach(() => {
	directory = mkdtempSync(join(tmpdir(), "thinkrail-endpoint-limits-"));
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

const NO_LIMITS: CustomEndpointLimits = {
	maxConcurrency: null,
	timeoutSeconds: null,
	maxRetries: null,
};

const MODEL: CustomEndpointModel = {
	id: "m",
	contextWindow: 8_192,
	maxTokens: 2_048,
	reasoning: false,
	vision: false,
	tools: null,
	chatTemplateThinking: null,
};

function draft(baseUrl: string, limits: Partial<CustomEndpointLimits> = {}): CustomEndpointDraft {
	return { name: "box", baseUrl, apiKey: "", limits: { ...NO_LIMITS, ...limits }, models: [MODEL] };
}

async function save(next: CustomEndpointDraft): Promise<string> {
	const endpoints = await saveCustomEndpoint(next);
	const saved = endpoints.find((endpoint) => endpoint.name === next.name);
	if (!saved) throw new Error("endpoint was not saved");
	return saved.id;
}

function serve(routes: Record<string, () => unknown>) {
	server = Bun.serve({
		port: 0,
		hostname: "127.0.0.1",
		async fetch(request) {
			const route = routes[`${request.method} ${new URL(request.url).pathname}`];
			if (!route) return new Response("missing", { status: 404 });
			return Response.json(await route());
		},
	});
	return `http://127.0.0.1:${server.port}`;
}

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

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

function fakeChatServer(
	onChat: (body: Record<string, unknown>) => Promise<string | Response> | string | Response,
) {
	server = Bun.serve({
		port: 0,
		hostname: "127.0.0.1",
		async fetch(request) {
			const url = new URL(request.url);
			if (request.method === "POST" && url.pathname === "/v1/chat/completions") {
				const body: unknown = await request.json();
				const reply = await onChat(isRecord(body) ? body : {});
				if (reply instanceof Response) return reply;
				return new Response(reply, { headers: { "content-type": "text/event-stream" } });
			}
			return new Response("missing", { status: 404 });
		},
	});
	return `http://127.0.0.1:${server.port}/v1`;
}

const textReply = (text: string) =>
	sse([
		chunk([{ index: 0, delta: { role: "assistant", content: text } }]),
		chunk([{ index: 0, delta: {}, finish_reason: "stop" }]),
		chunk([], { usage: { prompt_tokens: 5, completion_tokens: 2, total_tokens: 7 } }),
	]);

const toolReply = () =>
	sse([
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

function modelOf(pi: ModelRuntime, provider: string) {
	const found = pi.getAvailableSnapshot().find((model) => model.provider === provider);
	if (!found) throw new Error("missing model");
	return found;
}

const ask = (pi: ModelRuntime, provider: string) =>
	pi.completeSimple(modelOf(pi, provider), {
		messages: [{ role: "user", content: "hi", timestamp: Date.now() }],
	});

test("limits round-trip through the sidecar, survive an edit without them, and leave with the server", async () => {
	await runtime();
	const id = await save(draft("http://127.0.0.1:1/v1", { maxConcurrency: 2, maxRetries: 0 }));
	expect((await listCustomEndpoints())[0]?.limits).toEqual({
		maxConcurrency: 2,
		timeoutSeconds: null,
		maxRetries: 0,
	});
	const sidecar = join(directory, "data", "local-models.json");
	expect(JSON.parse(readFileSync(sidecar, "utf8"))).toEqual({
		version: 1,
		endpoints: { [id]: { maxConcurrency: 2, maxRetries: 0 } },
	});
	expect(readFileSync(path, "utf8")).not.toContain("maxConcurrency");
	const { limits: _limits, ...withoutLimits } = draft("http://127.0.0.1:1/v1");
	await saveCustomEndpoint({ ...withoutLimits, id, name: "renamed" });
	expect((await listCustomEndpoints())[0]).toMatchObject({
		name: "renamed",
		limits: { maxConcurrency: 2, maxRetries: 0 },
	});
	await saveCustomEndpoint({ ...draft("http://127.0.0.1:1/v1"), id });
	expect((await listCustomEndpoints())[0]?.limits).toEqual(NO_LIMITS);
	await saveCustomEndpoint({ ...draft("http://127.0.0.1:1/v1", { timeoutSeconds: 60 }), id });
	await removeCustomEndpoint(id);
	expect(JSON.parse(readFileSync(sidecar, "utf8")).endpoints).toEqual({});
});

test("out-of-range limits are refused before anything is written", async () => {
	await runtime();
	for (const limits of [{ maxConcurrency: 0 }, { timeoutSeconds: 5 }, { maxRetries: 11 }]) {
		await expect(saveCustomEndpoint(draft("http://127.0.0.1:1/v1", limits))).rejects.toThrow();
	}
	expect(await listCustomEndpoints()).toEqual([]);
});

test("a failed limits write leaves models.json untouched, so a retry adds no duplicate", async () => {
	const blocker = join(directory, "not-a-dir");
	writeFileSync(blocker, "");
	process.env.THINKRAIL_DATA_DIR = join(blocker, "data");
	writeFileSync(path, '{ "providers": {} }');
	await runtime();
	let published = 0;
	setModelsFilePublisher(() => {
		published++;
	});
	const next = draft("http://127.0.0.1:1/v1", { maxConcurrency: 2 });
	await expect(saveCustomEndpoint(next)).rejects.toThrow("request limits");
	expect(readFileSync(path, "utf8")).toBe('{ "providers": {} }');
	expect(published).toBe(0);
	process.env.THINKRAIL_DATA_DIR = join(directory, "data");
	await save(next);
	expect(await listCustomEndpoints()).toHaveLength(1);
	expect(published).toBe(1);
});

test("a limits-only change still notifies other clients", async () => {
	await runtime();
	const id = await save(draft("http://127.0.0.1:1/v1", { maxConcurrency: 2 }));
	const before = readFileSync(path, "utf8");
	let published = 0;
	setModelsFilePublisher(() => {
		published++;
	});
	await saveCustomEndpoint({ ...draft("http://127.0.0.1:1/v1", { maxConcurrency: 3 }), id });
	expect(readFileSync(path, "utf8")).toBe(before);
	expect(published).toBe(1);
	await saveCustomEndpoint({ ...draft("http://127.0.0.1:1/v1", { maxConcurrency: 3 }), id });
	expect(published).toBe(1);
	expect((await listCustomEndpoints())[0]?.limits.maxConcurrency).toBe(3);
});

test("an unreadable limits file is never overwritten", async () => {
	mkdirSync(join(directory, "data"), { recursive: true });
	writeFileSync(join(directory, "data", "local-models.json"), "{ not json");
	await runtime();
	await expect(
		saveCustomEndpoint(draft("http://127.0.0.1:1/v1", { maxConcurrency: 1 })),
	).rejects.toThrow("isn't valid JSON");
	expect(readFileSync(join(directory, "data", "local-models.json"), "utf8")).toBe("{ not json");
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
	const id = await save(draft(base, { maxConcurrency: 1 }));
	const runs = [ask(pi, id), ask(pi, id), ask(pi, id)];
	await new Promise((resolve) => setTimeout(resolve, 100));
	expect(requestLoad(id)).toEqual({ active: 1, queued: 2 });
	while (releases.length > 0 || requestLoad(id).active > 0) {
		releases.shift()?.();
		await new Promise((resolve) => setTimeout(resolve, 20));
	}
	const results = await Promise.all(runs);
	expect(results.map((message) => message.stopReason)).toEqual(["stop", "stop", "stop"]);
	expect(peak).toBe(1);
	expect(requestLoad(id)).toEqual({ active: 0, queued: 0 });
});

test("an endpoint's saved retries and timeout reach pi's request options", async () => {
	let calls = 0;
	const base = fakeChatServer(() => {
		calls++;
		if (calls < 3) return Response.json({ error: { message: "loading model" } }, { status: 503 });
		return textReply("OK");
	});
	const pi = await runtime();
	const id = await save(draft(base, { timeoutSeconds: 30, maxRetries: 2 }));
	expect((await ask(pi, id)).stopReason).toBe("stop");
	expect(calls).toBe(3);

	calls = 0;
	await saveCustomEndpoint({ ...draft(base, { maxRetries: 0 }), id });
	expect((await ask(pi, id)).stopReason).toBe("error");
	expect(calls).toBe(1);
}, 20_000);

test("queue wait does not count against an endpoint's request timeout", async () => {
	const base = serve({
		"GET /slow": async () => {
			await new Promise((resolve) => setTimeout(resolve, 150));
			return { ok: true };
		},
		"GET /hang": async () => {
			await new Promise((resolve) => setTimeout(resolve, 1_000));
			return { ok: true };
		},
	});
	await runtime();
	const id = await save(draft(`${base}/v1`, { maxConcurrency: 1 }));
	const fetch = limitedFetch(id, 400);
	const queued = await Promise.all(
		[1, 2, 3, 4].map(async () => (await fetch(`${base}/slow`)).json()),
	);
	expect(queued).toEqual([{ ok: true }, { ok: true }, { ok: true }, { ok: true }]);
	await expect(fetch(`${base}/hang`)).rejects.toThrow("timed out");
	expect(requestLoad(id)).toEqual({ active: 0, queued: 0 });
});

test("gated requests move the timeout off the SDK, nested dispatch is not re-gated, and a caller's 0 retries wins", async () => {
	await runtime();
	const id = await save(
		draft("http://127.0.0.1:1/v1", { maxConcurrency: 1, timeoutSeconds: 30, maxRetries: 2 }),
	);
	const options: { fetch?: typeof fetch; timeoutMs?: number; maxRetries?: number } = {
		timeoutMs: 5_000,
	};
	const outer = withEndpointOptions(id, options);
	expect(outer.timeoutMs).toBe(2_147_483_647);
	expect(outer.maxRetries).toBe(2);
	expect(typeof outer.fetch).toBe("function");
	const nested = withEndpointOptions(id, outer);
	expect(nested.fetch).toBe(outer.fetch);
	expect(nested.timeoutMs).toBe(2_147_483_647);
	expect(withEndpointOptions("openai", { timeoutMs: 5_000 })).toEqual({ timeoutMs: 5_000 });
	const noRetry: { fetch?: typeof fetch; timeoutMs?: number; maxRetries?: number } = {
		maxRetries: 0,
	};
	expect(withEndpointOptions(id, noRetry).maxRetries).toBe(0);
});

test("a queued request can be aborted and a stalled or cancelled body frees its slot", async () => {
	const base = serve({
		"GET /hold": async () => {
			await new Promise((resolve) => setTimeout(resolve, 200));
			return { ok: true };
		},
	});
	const stall = Bun.serve({
		port: 0,
		hostname: "127.0.0.1",
		fetch() {
			return new Response(
				new ReadableStream({
					start(controller) {
						controller.enqueue(new TextEncoder().encode("data: first\n\n"));
					},
				}),
				{ headers: { "content-type": "text/event-stream" } },
			);
		},
	});
	try {
		await runtime();
		const id = await save(draft(`${base}/v1`, { maxConcurrency: 1 }));
		const fetch = limitedFetch(id, 300);
		const holding = fetch(`${base}/hold`);
		const abort = new AbortController();
		const queued = fetch(`${base}/hold`, { signal: abort.signal });
		await new Promise((resolve) => setTimeout(resolve, 20));
		expect(requestLoad(id)).toEqual({ active: 1, queued: 1 });
		abort.abort(new Error("stop"));
		await expect(queued).rejects.toThrow("stop");
		expect(requestLoad(id)).toEqual({ active: 1, queued: 0 });
		await (await holding).json();
		expect(requestLoad(id).active).toBe(0);

		const stalled = await fetch(`http://127.0.0.1:${stall.port}/`);
		await expect(stalled.text()).rejects.toThrow("timed out");
		expect(requestLoad(id).active).toBe(0);

		const cancelled = await fetch(`http://127.0.0.1:${stall.port}/`);
		await cancelled.body?.cancel();
		expect(requestLoad(id).active).toBe(0);
	} finally {
		stall.stop(true);
	}
});

test("the agent check reports a chat reply and a real tool call through pi", async () => {
	const requests: Record<string, unknown>[] = [];
	const base = fakeChatServer((body) => {
		requests.push(body);
		return Array.isArray(body.tools) ? toolReply() : textReply("OK");
	});
	await runtime();
	const id = await save(draft(base));
	const result = await testCustomEndpoint(id, "m");
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

test("the agent check tells a text-only answer, a server error and a missing model apart", async () => {
	let fail = false;
	const base = fakeChatServer(() => {
		if (fail) return Response.json({ error: { message: "model not loaded" } }, { status: 503 });
		return textReply("It is noon.");
	});
	await runtime();
	const id = await save(draft(base));
	expect(await testCustomEndpoint(id, "m")).toMatchObject({
		chat: "ok",
		tools: "failed",
		message: null,
	});
	fail = true;
	const failed = await testCustomEndpoint(id, "m");
	expect(failed).toMatchObject({ chat: "failed", tools: "skipped" });
	expect(failed.message).toContain("model not loaded");
	await expect(testCustomEndpoint(id, "missing")).rejects.toThrow("not available");
	await expect(testCustomEndpoint("openai", "gpt-5")).rejects.toThrow();
});

test("the agent check reports a busy endpoint instead of a failure", async () => {
	const releases: (() => void)[] = [];
	const base = fakeChatServer(async () => {
		await new Promise<void>((resolve) => releases.push(resolve));
		return textReply("OK");
	});
	const pi = await runtime();
	const id = await save(draft(base, { maxConcurrency: 1 }));
	const running = ask(pi, id);
	await new Promise((resolve) => setTimeout(resolve, 100));
	expect(await testCustomEndpoint(id, "m")).toEqual({
		chat: "skipped",
		tools: "skipped",
		latencyMs: null,
		outputTokensPerSecond: null,
		message: null,
	});
	releases.shift()?.();
	await running;
});
