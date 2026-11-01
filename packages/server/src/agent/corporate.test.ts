import { afterEach, beforeEach, expect, test } from "bun:test";
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { InMemoryCredentialStore } from "@earendil-works/pi-ai";
import { ModelRuntime } from "@earendil-works/pi-coding-agent";
import type {
	CustomEndpointDraft,
	CustomEndpointModel,
	CustomEndpointNetwork,
} from "@thinkrail/contracts";
import {
	listCustomEndpoints,
	probeCustomEndpoint,
	removeCustomEndpoint,
	saveCustomEndpoint,
	syncManagedEndpoints,
} from "./customEndpoints";
import { commandToken, jwtExpiry } from "./endpointAccess";
import { organizationPolicy, providerAllowed } from "./managedConfig";
import { accountAvailableModels } from "./modelAvailability";
import { setModelsFilePublisher } from "./modelsFile";
import { configurePiRuntime } from "./piRuntime";

const FIXTURES = join(import.meta.dir, "__fixtures__", "corp-tls");
const NO_NETWORK: CustomEndpointNetwork = { systemCertificates: false, caFile: null, proxy: null };

let directory: string;
let path: string;
const prior: Record<string, string | undefined> = {};
const servers: { stop(force?: boolean): void }[] = [];

beforeEach(() => {
	directory = mkdtempSync(join(tmpdir(), "thinkrail-corp-"));
	path = join(directory, "models.json");
	for (const name of [
		"PI_CODING_AGENT_DIR",
		"PI_OFFLINE",
		"THINKRAIL_DATA_DIR",
		"THINKRAIL_MANAGED_CONFIG",
		"CORP_APIM_KEY",
	])
		prior[name] = process.env[name];
	process.env.THINKRAIL_DATA_DIR = join(directory, "data");
	process.env.PI_CODING_AGENT_DIR = directory;
	process.env.PI_OFFLINE = "1";
	process.env.THINKRAIL_MANAGED_CONFIG = join(directory, "managed.json");
});

afterEach(() => {
	for (const server of servers.splice(0)) server.stop(true);
	setModelsFilePublisher(null);
	configurePiRuntime(null);
	for (const [name, value] of Object.entries(prior)) {
		if (value === undefined) delete process.env[name];
		else process.env[name] = value;
	}
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

function model(id: string, patch: Partial<CustomEndpointModel> = {}): CustomEndpointModel {
	return {
		id,
		contextWindow: 8_192,
		maxTokens: 2_048,
		reasoning: false,
		vision: false,
		tools: null,
		chatTemplateThinking: null,
		...patch,
	};
}

function sse(chunks: unknown[]): string {
	return `${chunks.map((chunk) => `data: ${JSON.stringify(chunk)}\n\n`).join("")}data: [DONE]\n\n`;
}

const base = { id: "c1", object: "chat.completion.chunk", created: 1, model: "m" };
const textReply = () =>
	new Response(
		sse([
			{ ...base, choices: [{ index: 0, delta: { role: "assistant", content: "OK" } }] },
			{ ...base, choices: [{ index: 0, delta: {}, finish_reason: "stop" }] },
		]),
		{ headers: { "content-type": "text/event-stream" } },
	);

const listing = () => Response.json({ object: "list", data: [{ id: "m", object: "model" }] });

interface Seen {
	path: string;
	host: string;
	headers: Headers;
}

function serve(
	handle: (request: Request, seen: Seen) => Response | Promise<Response>,
	tls?: { cert: string; key: string },
) {
	const seen: Seen[] = [];
	const server = Bun.serve({
		port: 0,
		hostname: "127.0.0.1",
		...(tls ? { tls } : {}),
		fetch(request) {
			const url = new URL(request.url);
			const entry = { path: url.pathname, host: url.host, headers: request.headers };
			seen.push(entry);
			return handle(request, entry);
		},
	});
	servers.push(server);
	return { url: `${tls ? "https://localhost" : "http://127.0.0.1"}:${server.port}`, seen };
}

function draft(baseUrl: string, patch: Partial<CustomEndpointDraft> = {}): CustomEndpointDraft {
	return { name: "Corp", baseUrl, apiKey: "", headers: [], models: [model("m")], ...patch };
}

function ask(pi: ModelRuntime, provider: string, modelId = "m") {
	const found = pi.getAvailableSnapshot().find((m) => m.provider === provider && m.id === modelId);
	if (!found) throw new Error(`missing ${provider}/${modelId}`);
	return pi.completeSimple(
		found,
		{ messages: [{ role: "user", content: "hi", timestamp: Date.now() }] },
		{ maxRetries: 0 },
	);
}

test("headers: saved to pi, secret literals hidden, kept on edit, sent by Detect and by chats", async () => {
	process.env.CORP_APIM_KEY = "apim-secret";
	const gateway = serve((request) => {
		if (request.headers.get("ocp-apim-subscription-key") !== "apim-secret")
			return new Response("no", { status: 401 });
		return new URL(request.url).pathname === "/v1/models" ? listing() : textReply();
	});
	const pi = await runtime();
	const probe = await probeCustomEndpoint({
		baseUrl: `${gateway.url}/v1`,
		headers: [{ name: "Ocp-Apim-Subscription-Key", value: "$CORP_APIM_KEY" }],
	});
	expect(probe.ok).toBe(true);
	const [saved] = await saveCustomEndpoint(
		draft(`${gateway.url}/v1`, {
			headers: [
				{ name: "Ocp-Apim-Subscription-Key", value: "$CORP_APIM_KEY" },
				{ name: "X-Org", value: "plata-literal" },
			],
		}),
	);
	if (!saved) throw new Error("not saved");
	expect(saved.headers).toEqual([
		{ name: "Ocp-Apim-Subscription-Key", value: "$CORP_APIM_KEY" },
		{ name: "X-Org", value: null },
	]);
	expect(JSON.parse(readFileSync(path, "utf8")).providers[saved.id].headers).toEqual({
		"Ocp-Apim-Subscription-Key": "$CORP_APIM_KEY",
		"X-Org": "plata-literal",
	});
	await saveCustomEndpoint({
		...draft(`${gateway.url}/v1`, { headers: saved.headers }),
		id: saved.id,
	});
	expect(JSON.parse(readFileSync(path, "utf8")).providers[saved.id].headers["X-Org"]).toBe(
		"plata-literal",
	);
	expect(
		(await probeCustomEndpoint({ baseUrl: `${gateway.url}/v1`, endpointId: saved.id })).ok,
	).toBe(true);
	expect((await ask(pi, saved.id)).stopReason).toBe("stop");
	expect(gateway.seen.at(-1)?.headers.get("x-org")).toBe("plata-literal");
	await expect(
		saveCustomEndpoint({
			...draft(`${gateway.url}/v1`, { headers: [{ name: "X-New", value: null }] }),
			id: saved.id,
		}),
	).rejects.toThrow("Enter a value for header X-New");
	await expect(
		saveCustomEndpoint(draft(`${gateway.url}/v1`, { headers: [{ name: "Host", value: "x" }] })),
	).rejects.toThrow("Header names");
});

test("network: an untrusted company CA fails Detect as a certificate problem until its PEM is trusted", async () => {
	const secure = serve(
		(request) => (new URL(request.url).pathname === "/v1/models" ? listing() : textReply()),
		{
			cert: readFileSync(join(FIXTURES, "server.pem"), "utf8"),
			key: readFileSync(join(FIXTURES, "server.key"), "utf8"),
		},
	);
	const pi = await runtime();
	expect(await probeCustomEndpoint({ baseUrl: `${secure.url}/v1` })).toEqual({
		ok: false,
		reason: "certificate",
	});
	const network = { ...NO_NETWORK, caFile: join(FIXTURES, "ca.pem") };
	expect((await probeCustomEndpoint({ baseUrl: `${secure.url}/v1`, network })).ok).toBe(true);
	await expect(
		saveCustomEndpoint(draft(`${secure.url}/v1`, { network: { ...network, caFile: "/nope.pem" } })),
	).rejects.toThrow("can't be read");
	const [saved] = await saveCustomEndpoint(draft(`${secure.url}/v1`, { network }));
	if (!saved) throw new Error("not saved");
	expect(saved.network).toEqual(network);
	expect((await ask(pi, saved.id)).stopReason).toBe("stop");
	expect(
		(await probeCustomEndpoint({ baseUrl: `${secure.url}/v1`, endpointId: saved.id })).ok,
	).toBe(true);
});

test("network: a proxy carries Detect and chats to a host this machine can't resolve", async () => {
	const proxy = serve((request, seen) => {
		if (seen.host !== "llm.corp.invalid") return new Response("wrong host", { status: 502 });
		return new URL(request.url).pathname === "/v1/models" ? listing() : textReply();
	});
	const pi = await runtime();
	const network = { ...NO_NETWORK, proxy: proxy.url };
	const target = "http://llm.corp.invalid/v1";
	expect((await probeCustomEndpoint({ baseUrl: target })).ok).toBe(false);
	expect((await probeCustomEndpoint({ baseUrl: target, network })).ok).toBe(true);
	const [saved] = await saveCustomEndpoint(draft(target, { network }));
	if (!saved) throw new Error("not saved");
	expect((await ask(pi, saved.id)).stopReason).toBe("stop");
	expect(proxy.seen.map((entry) => entry.path)).toContain("/v1/chat/completions");
	await expect(
		saveCustomEndpoint(draft(target, { network: { ...network, proxy: "http://u:p@proxy:1" } })),
	).rejects.toThrow("proxy");
});

test("command keys: Detect runs a typed command, a form-saved command is trusted, a hand-edited one isn't", async () => {
	const counter = join(directory, "runs");
	const command = `!printf x >> '${counter}' && echo tok-123`;
	const gateway = serve((request) =>
		request.headers.get("authorization") === "Bearer tok-123"
			? new URL(request.url).pathname === "/v1/models"
				? listing()
				: textReply()
			: new Response("no", { status: 401 }),
	);
	await runtime();
	expect((await probeCustomEndpoint({ baseUrl: `${gateway.url}/v1`, apiKey: command })).ok).toBe(
		true,
	);
	const [saved] = await saveCustomEndpoint(draft(`${gateway.url}/v1`, { apiKey: command }));
	if (!saved) throw new Error("not saved");
	expect(saved.apiKeyReference).toBe(command);
	expect(
		(await probeCustomEndpoint({ baseUrl: `${gateway.url}/v1`, endpointId: saved.id })).ok,
	).toBe(true);
	const file = JSON.parse(readFileSync(path, "utf8"));
	file.providers[saved.id].apiKey = `!printf y >> '${counter}' && echo tok-123`;
	writeFileSync(path, JSON.stringify(file));
	expect(
		await probeCustomEndpoint({ baseUrl: `${gateway.url}/v1`, endpointId: saved.id }),
	).toMatchObject({ ok: false, reason: "unauthorized" });
	expect(readFileSync(counter, "utf8")).not.toContain("y");
});

test("command keys: chats reuse the token until the server refuses it, then fetch a fresh one", async () => {
	const counter = join(directory, "runs");
	let refuseNext = false;
	const gateway = serve((request) => {
		if (new URL(request.url).pathname === "/v1/models") return listing();
		if (refuseNext) {
			refuseNext = false;
			return Response.json({ error: { message: "token expired" } }, { status: 401 });
		}
		return textReply();
	});
	const pi = await runtime();
	const [saved] = await saveCustomEndpoint(
		draft(`${gateway.url}/v1`, { apiKey: `!printf x >> '${counter}' && echo tok-cache` }),
	);
	if (!saved) throw new Error("not saved");
	const runs = () => (existsSync(counter) ? readFileSync(counter, "utf8").length : 0);
	await ask(pi, saved.id);
	await ask(pi, saved.id);
	expect(runs()).toBe(1);
	expect(gateway.seen.at(-1)?.headers.get("authorization")).toBe("Bearer tok-cache");
	refuseNext = true;
	expect((await ask(pi, saved.id)).stopReason).toBe("error");
	await ask(pi, saved.id);
	expect(runs()).toBe(2);
});

test("command keys: a JWT is cached until shortly before its exp claim", async () => {
	const exp = Math.floor(Date.parse("2026-11-01T13:00:00Z") / 1000);
	const jwt = [
		Buffer.from('{"alg":"none"}').toString("base64url"),
		Buffer.from(JSON.stringify({ exp })).toString("base64url"),
		"sig",
	].join(".");
	expect(jwtExpiry(jwt)).toBe(exp * 1000);
	expect(jwtExpiry("not-a-jwt")).toBeNull();
	const counter = join(directory, "jwt-runs");
	const command = `!printf x >> '${counter}' && echo ${jwt}`;
	let now = Date.parse("2026-11-01T12:00:00Z");
	expect(await commandToken(command, () => now)).toBe(jwt);
	now = Date.parse("2026-11-01T12:58:00Z");
	await commandToken(command, () => now);
	expect(readFileSync(counter, "utf8")).toBe("x");
	now = Date.parse("2026-11-01T12:59:30Z");
	await commandToken(command, () => now);
	expect(readFileSync(counter, "utf8")).toBe("xx");
});

test("per-model API: Anthropic Messages goes to the root with x-api-key, Responses to /v1/responses", async () => {
	const gateway = serve(async (request) => {
		const url = new URL(request.url);
		if (url.pathname === "/v1/messages") {
			const events = [
				[
					"message_start",
					{
						type: "message_start",
						message: {
							id: "msg_1",
							type: "message",
							role: "assistant",
							model: "claude",
							content: [],
							stop_reason: null,
							stop_sequence: null,
							usage: { input_tokens: 5, output_tokens: 1 },
						},
					},
				],
				[
					"content_block_start",
					{ type: "content_block_start", index: 0, content_block: { type: "text", text: "" } },
				],
				[
					"content_block_delta",
					{ type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "Hi" } },
				],
				["content_block_stop", { type: "content_block_stop", index: 0 }],
				[
					"message_delta",
					{
						type: "message_delta",
						delta: { stop_reason: "end_turn", stop_sequence: null },
						usage: { output_tokens: 2 },
					},
				],
				["message_stop", { type: "message_stop" }],
			] as const;
			return new Response(
				events
					.map(([event, data]) => `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`)
					.join(""),
				{ headers: { "content-type": "text/event-stream" } },
			);
		}
		return Response.json({ error: { message: "not here" } }, { status: 404 });
	});
	const pi = await runtime();
	const [saved] = await saveCustomEndpoint(
		draft(`${gateway.url}/v1`, {
			apiKey: "sk-corp",
			models: [
				model("claude", { api: "anthropic-messages" }),
				model("gpt", { api: "openai-responses" }),
			],
		}),
	);
	if (!saved) throw new Error("not saved");
	const written = JSON.parse(readFileSync(path, "utf8")).providers[saved.id].models;
	expect(written[0]).toMatchObject({ api: "anthropic-messages", baseUrl: gateway.url });
	expect(written[1]).toMatchObject({ api: "openai-responses" });
	expect(written[1]).not.toHaveProperty("baseUrl");
	expect(saved.models.map((m) => m.api)).toEqual(["anthropic-messages", "openai-responses"]);
	const reply = await ask(pi, saved.id, "claude");
	expect(reply.stopReason).toBe("stop");
	const messages = gateway.seen.find((entry) => entry.path === "/v1/messages");
	expect(messages?.headers.get("x-api-key")).toBe("sk-corp");
	await ask(pi, saved.id, "gpt");
	expect(gateway.seen.map((entry) => entry.path)).toContain("/v1/responses");
	await saveCustomEndpoint({
		...draft(`${gateway.url}/v1`, {
			models: [model("claude", { api: null }), model("gpt", { api: null })],
		}),
		id: saved.id,
	});
	const reset = JSON.parse(readFileSync(path, "utf8")).providers[saved.id].models;
	expect(reset[0]).not.toHaveProperty("api");
	expect(reset[0]).not.toHaveProperty("baseUrl");
});

test("LiteLLM quota and key refusals read in plain words with the key's budget", async () => {
	let reply: () => Response = textReply;
	const gateway = serve((request) => {
		const url = new URL(request.url);
		if (url.pathname === "/v1/models") return listing();
		if (url.pathname === "/model/info")
			return Response.json({ data: [{ model_name: "m", model_info: { mode: "chat" } }] });
		if (url.pathname === "/key/info")
			return Response.json({
				info: { spend: 50.2, max_budget: 50, budget_reset_at: "2026-12-01T00:00:00Z" },
			});
		return reply();
	});
	const pi = await runtime();
	const [saved] = await saveCustomEndpoint(
		draft(`${gateway.url}/v1`, {
			apiKey: "sk-v",
			gateway: { kind: "litellm", sessionAffinity: false, tags: [] },
		}),
	);
	if (!saved) throw new Error("not saved");
	reply = () =>
		Response.json(
			{
				error: {
					message: "ExceededBudget: Key over budget. Spend=50.2, Budget=50",
					type: "budget_exceeded",
					code: "400",
				},
			},
			{ status: 400 },
		);
	const budget = await ask(pi, saved.id);
	expect(budget.errorMessage).toContain("The budget for your key on Corp is used up");
	expect(budget.errorMessage).toContain("$50.2 of $50 spent");
	expect(budget.errorMessage).toContain("It resets on 2026-12-01");
	reply = () =>
		Response.json(
			{ error: { message: "RPM limit=60", type: "rate_limit_error", code: 429 } },
			{ status: 429, headers: { "retry-after": "12" } },
		);
	expect((await ask(pi, saved.id)).errorMessage).toContain("rate-limiting your key");
	reply = () =>
		Response.json(
			{ detail: "Authentication Error - Expired Key. Key Expiry time 2026-10-01" },
			{
				status: 401,
			},
		);
	expect((await ask(pi, saved.id)).errorMessage).toContain("Your key for Corp has expired");
});

test("managed servers sync from the organization's file, lock what it sets, and leave with it", async () => {
	const gateway = serve((request) =>
		new URL(request.url).pathname === "/v1/models" ? listing() : textReply(),
	);
	const managedPath = join(directory, "managed.json");
	writeFileSync(
		managedPath,
		JSON.stringify({
			servers: [
				{
					id: "plata",
					name: "Plata AI Gateway",
					baseUrl: `${gateway.url}/v1`,
					headers: { "X-Org": "plata" },
					limits: { maxConcurrency: 4 },
					gateway: { kind: "litellm", sessionAffinity: true, tags: ["thinkrail"] },
				},
				{ id: "bad", name: "No URL" },
			],
			policy: { restrictModels: "managed" },
		}),
	);
	const pi = await runtime();
	await syncManagedEndpoints();
	const [managed] = await listCustomEndpoints();
	expect(managed).toMatchObject({
		id: "custom-org-plata",
		name: "Plata AI Gateway",
		hasApiKey: false,
		managed: { key: false, models: false },
		limits: { maxConcurrency: 4 },
		gateway: { tags: ["thinkrail"] },
		headers: [{ name: "X-Org", value: null }],
	});
	expect(organizationPolicy()).toMatchObject({ restrictModels: "managed", managedServers: 1 });
	expect(organizationPolicy().error).toContain("server bad needs an http(s) baseUrl");

	await saveCustomEndpoint({
		...draft("http://elsewhere.invalid/v1", { name: "Renamed", apiKey: "sk-mine" }),
		id: "custom-org-plata",
	});
	const [edited] = await listCustomEndpoints();
	expect(edited).toMatchObject({
		name: "Plata AI Gateway",
		baseUrl: `${gateway.url}/v1`,
		hasApiKey: true,
	});
	await expect(removeCustomEndpoint("custom-org-plata")).rejects.toThrow("organization manages");

	const [own] = (await saveCustomEndpoint(draft(`${gateway.url}/v1`, { name: "Mine" }))).filter(
		(endpoint) => endpoint.name === "Mine",
	);
	if (!own) throw new Error("not saved");
	expect(providerAllowed("custom-org-plata")).toBe(true);
	expect(providerAllowed(own.id)).toBe(false);
	expect(providerAllowed("openai")).toBe(false);
	expect(accountAvailableModels(pi).map((m) => m.provider)).toEqual(["custom-org-plata"]);
	const refused = await ask(pi, own.id);
	expect(refused.stopReason).toBe("error");
	expect(refused.errorMessage).toContain("policy allows only its own model servers");
	expect((await ask(pi, "custom-org-plata")).stopReason).toBe("stop");
	expect(gateway.seen.at(-1)?.headers.get("x-org")).toBe("plata");

	writeFileSync(managedPath, JSON.stringify({ servers: [] }));
	await syncManagedEndpoints();
	expect((await listCustomEndpoints()).map((endpoint) => endpoint.id)).toEqual([own.id]);
	expect(providerAllowed(own.id)).toBe(true);
});

test("review: a managed server follows its file's new address and a hand-edited address is repaired or refused", async () => {
	const oldGateway = serve(() => listing());
	const newGateway = serve((request) =>
		new URL(request.url).pathname === "/v1/models" ? listing() : textReply(),
	);
	const managedPath = join(directory, "managed.json");
	const write = (url: string) =>
		writeFileSync(
			managedPath,
			JSON.stringify({
				servers: [{ id: "corp", name: "Corp", baseUrl: `${url}/v1`, models: ["m"] }],
				policy: { restrictModels: "managed" },
			}),
		);
	write(oldGateway.url);
	const pi = await runtime();
	await syncManagedEndpoints();
	await saveCustomEndpoint({
		...draft(`${oldGateway.url}/v1`, { apiKey: "sk-user" }),
		id: "custom-org-corp",
	});
	write(newGateway.url);
	await syncManagedEndpoints();
	const [moved] = await listCustomEndpoints();
	expect(moved).toMatchObject({ baseUrl: `${newGateway.url}/v1`, hasApiKey: true });
	const file = JSON.parse(readFileSync(path, "utf8"));
	file.providers["custom-org-corp"].baseUrl = "https://evil.invalid/v1";
	writeFileSync(path, JSON.stringify(file));
	await pi.refresh({ allowNetwork: false });
	const refused = await ask(pi, "custom-org-corp");
	expect(refused.errorMessage).toContain("must be reached at");
	await syncManagedEndpoints();
	expect((await listCustomEndpoints())[0]?.baseUrl).toBe(`${newGateway.url}/v1`);
	expect((await ask(pi, "custom-org-corp")).stopReason).toBe("stop");
});

test("review: a broken managed file or entry never deletes synced servers or the user's key", async () => {
	const gateway = serve(() => listing());
	const managedPath = join(directory, "managed.json");
	writeFileSync(
		managedPath,
		JSON.stringify({
			servers: [{ id: "corp", name: "Corp", baseUrl: `${gateway.url}/v1`, models: ["m"] }],
			policy: { restrictModels: "managed" },
		}),
	);
	await runtime();
	await syncManagedEndpoints();
	await saveCustomEndpoint({
		...draft(`${gateway.url}/v1`, { apiKey: "sk-user" }),
		id: "custom-org-corp",
	});
	writeFileSync(
		managedPath,
		JSON.stringify({
			servers: [{ id: "corp", name: "Corp", baseUrl: "typo" }],
			policy: { restrictModels: "managed" },
		}),
	);
	await syncManagedEndpoints();
	expect((await listCustomEndpoints())[0]).toMatchObject({
		id: "custom-org-corp",
		hasApiKey: true,
	});
	writeFileSync(managedPath, "{ half-written");
	await syncManagedEndpoints();
	expect((await listCustomEndpoints()).map((e) => e.id)).toEqual(["custom-org-corp"]);
	expect(organizationPolicy().error).toContain("isn't valid JSON");
	expect(providerAllowed("custom-org-corp")).toBe(true);
	expect(providerAllowed("openai")).toBe(false);
	writeFileSync(
		managedPath,
		JSON.stringify({ servers: [], policy: { restrictModels: "Managed" } }),
	);
	expect(organizationPolicy().restrictModels).toBe("managed");
});

test("review: header commands never run, and planted key commands stay untrusted on managed servers", async () => {
	const marker = join(directory, "ran");
	const gateway = serve(() => listing());
	await runtime();
	const [saved] = await saveCustomEndpoint(draft(`${gateway.url}/v1`));
	if (!saved) throw new Error("not saved");
	const file = JSON.parse(readFileSync(path, "utf8"));
	file.providers[saved.id].headers = { "X-Tok": `!touch '${marker}'` };
	writeFileSync(path, JSON.stringify(file));
	await probeCustomEndpoint({
		baseUrl: `${gateway.url}/v1`,
		endpointId: saved.id,
		headers: [{ name: "X-Tok", value: `!touch '${marker}'` }],
	});
	await probeCustomEndpoint({ baseUrl: `${gateway.url}/v1`, endpointId: saved.id });
	expect(existsSync(marker)).toBe(false);
	await expect(
		saveCustomEndpoint(draft(`${gateway.url}/v1`, { headers: [{ name: "X-Tok", value: "!id" }] })),
	).rejects.toThrow("can't run a command");

	writeFileSync(
		join(directory, "managed.json"),
		JSON.stringify({ servers: [{ id: "corp", name: "Corp", baseUrl: `${gateway.url}/v1` }] }),
	);
	await syncManagedEndpoints();
	const planted = JSON.parse(readFileSync(path, "utf8"));
	planted.providers["custom-org-corp"].apiKey = `!touch '${marker}'`;
	writeFileSync(path, JSON.stringify(planted));
	await syncManagedEndpoints();
	await probeCustomEndpoint({ baseUrl: `${gateway.url}/v1`, endpointId: "custom-org-corp" });
	expect(existsSync(marker)).toBe(false);
});

test("review: literals with $ stay secret, saved credentials keep their own network, an old client vouches for nothing", async () => {
	const gateway = serve(() => listing());
	const trap = serve(() => listing());
	await runtime();
	const [saved] = await saveCustomEndpoint(
		draft(`${gateway.url}/v1`, {
			apiKey: "sk-abc$$def",
			headers: [{ name: "Authorization-Extra", value: "Bearer s3cr3t-$TENANT" }],
		}),
	);
	if (!saved) throw new Error("not saved");
	expect(saved.apiKeyReference).toBeNull();
	expect(saved.headers).toEqual([{ name: "Authorization-Extra", value: null }]);
	await probeCustomEndpoint({
		baseUrl: `${gateway.url}/v1`,
		endpointId: saved.id,
		network: { ...NO_NETWORK, proxy: trap.url },
	});
	expect(trap.seen).toHaveLength(0);

	const marker = join(directory, "old-client");
	const { headers: _modern, ...old } = draft(`${gateway.url}/v1`, { apiKey: `!touch '${marker}'` });
	const [legacy] = (await saveCustomEndpoint({ ...old, name: "Legacy" })).filter(
		(e) => e.name === "Legacy",
	);
	if (!legacy) throw new Error("not saved");
	await probeCustomEndpoint({ baseUrl: `${gateway.url}/v1`, endpointId: legacy.id });
	expect(existsSync(marker)).toBe(false);
});

test("review: switching a model off Anthropic Messages drops its root baseUrl even when the path changed", async () => {
	const gateway = serve(() => listing());
	await runtime();
	const [saved] = await saveCustomEndpoint(
		draft(`${gateway.url}/v1`, { models: [model("claude", { api: "anthropic-messages" })] }),
	);
	if (!saved) throw new Error("not saved");
	await saveCustomEndpoint({
		...draft(`${gateway.url}/litellm/v1`, { models: [model("claude", { api: null })] }),
		id: saved.id,
	});
	const written = JSON.parse(readFileSync(path, "utf8")).providers[saved.id].models[0];
	expect(written).not.toHaveProperty("baseUrl");
	expect(written).not.toHaveProperty("api");
});
