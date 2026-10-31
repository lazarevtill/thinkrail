import { readFileSync, writeFileSync } from "node:fs";
import { createServer, type IncomingMessage, type Server } from "node:http";
import { join } from "node:path";
import { expect, test } from "@playwright/test";
import {
	expandModelGroup,
	openWorkspaceChat,
	revealModelGroups,
	waitForAgentSettled,
} from "./fixtures/app";
import { openProviders } from "./fixtures/jbcentral";
import { E2E_PI_AGENT_DIR } from "./fixtures/paths";
import { shot } from "./fixtures/screenshots";

const modelsPath = join(E2E_PI_AGENT_DIR, "models.json");
const KEY = "sk-litellm-e2e";
const MODEL = "qwen3-coder";
const REPLY = "LITELLM_GATEWAY_REPLY_SENTINEL";
let server: Server;
let origin: string;
const chatHeaders: { session: string | null; tags: string | null }[] = [];

function header(request: IncomingMessage, name: string): string | null {
	const value = request.headers[name];
	return typeof value === "string" ? value : null;
}

function json(response: import("node:http").ServerResponse, body: unknown) {
	response.setHeader("content-type", "application/json");
	response.end(JSON.stringify(body));
}

function sse(chunks: unknown[]): string {
	return `${chunks.map((chunk) => `data: ${JSON.stringify(chunk)}\n\n`).join("")}data: [DONE]\n\n`;
}

test.beforeAll(async () => {
	server = createServer((request, response) => {
		if (request.headers.authorization !== `Bearer ${KEY}`) {
			response.statusCode = 401;
			json(response, { error: { message: "Authentication Error" } });
			return;
		}
		const route = `${request.method} ${request.url}`;
		if (route === "GET /v1/models") {
			json(response, {
				object: "list",
				data: [MODEL, "text-embedding", "openai/*"].map((id) => ({ id, object: "model" })),
			});
			return;
		}
		if (route === "GET /model/info") {
			json(response, {
				data: [
					{
						model_name: MODEL,
						model_info: {
							mode: "chat",
							max_input_tokens: 131_072,
							max_output_tokens: 16_384,
							supports_function_calling: true,
							supports_reasoning: true,
							input_cost_per_token: 2e-7,
							output_cost_per_token: 6e-7,
						},
					},
					{ model_name: "text-embedding", model_info: { mode: "embedding" } },
				],
			});
			return;
		}
		if (route === "GET /key/info") {
			json(response, {
				info: {
					key_alias: "e2e-laptop",
					spend: 46,
					max_budget: 50,
					rpm_limit: 60,
					max_parallel_requests: 4,
				},
			});
			return;
		}
		if (route === "POST /v1/chat/completions") {
			chatHeaders.push({
				session: header(request, "x-litellm-session-id"),
				tags: header(request, "x-litellm-tags"),
			});
			request.resume();
			request.on("end", () => {
				const base = { id: "c1", object: "chat.completion.chunk", created: 1, model: MODEL };
				response.setHeader("content-type", "text/event-stream");
				response.end(
					sse([
						{ ...base, choices: [{ index: 0, delta: { role: "assistant", content: REPLY } }] },
						{ ...base, choices: [{ index: 0, delta: {}, finish_reason: "stop" }] },
						{
							...base,
							choices: [],
							usage: { prompt_tokens: 10, completion_tokens: 3, total_tokens: 13 },
						},
					]),
				);
			});
			return;
		}
		response.statusCode = 404;
		response.end();
	});
	await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
	const address = server.address();
	if (address === null || typeof address === "string")
		throw new Error("The fake server has no port.");
	origin = `http://127.0.0.1:${address.port}`;
});

test.afterAll(async () => {
	writeFileSync(modelsPath, JSON.stringify({ providers: {} }));
	await new Promise<void>((resolve) => server.close(() => resolve()));
});

test("fork: a LiteLLM gateway is detected from its model info, shows its key, and tags every chat request", async ({
	page,
}) => {
	writeFileSync(modelsPath, JSON.stringify({ providers: {} }));
	await openWorkspaceChat(page);
	await openProviders(page);
	const group = page.getByTestId("settings-custom-endpoints");
	await group.getByTestId("custom-endpoints-add").click();
	const dialog = page.getByTestId("custom-endpoints-dialog");
	await dialog.getByTestId("custom-endpoints-name").fill("Corp LiteLLM");
	await dialog.getByTestId("custom-endpoints-url").fill(origin);
	await dialog.getByTestId("custom-endpoints-key").fill(KEY);
	await dialog.getByTestId("custom-endpoints-detect").click();
	await expect(dialog.getByTestId("custom-endpoints-status")).toHaveAttribute("data-tone", "ok");

	await expect(dialog.getByTestId("custom-endpoints-row")).toHaveCount(1);
	const row = dialog.locator(`[data-testid="custom-endpoints-row"][data-model="${MODEL}"]`);
	await expect(row).toContainText("131K");
	await expect(row).toContainText("$0.2 in / $0.6 out per 1M tokens");
	const gateway = dialog.getByTestId("custom-endpoints-gateway");
	await expect(gateway).toBeVisible();
	const probeKey = gateway.getByTestId("custom-endpoints-probe-key");
	await expect(probeKey).toContainText("LiteLLM key e2e-laptop");
	await expect(probeKey).toContainText("$46 of $50 spent");
	await expect(probeKey).toHaveAttribute("data-tone", "warning");
	await expect(dialog.getByTestId("custom-endpoints-parallel")).toHaveValue("4");
	await expect(gateway.getByTestId("custom-endpoints-session")).toHaveAttribute(
		"aria-checked",
		"true",
	);
	const tags = gateway.getByTestId("custom-endpoints-tags");
	await expect(tags).toHaveValue("thinkrail");
	await tags.fill("thinkrail, bad tag;");
	await expect(tags).toHaveAttribute("aria-invalid", "true");
	await expect(dialog.getByTestId("custom-endpoints-save")).toBeDisabled();
	await tags.fill("thinkrail, team:ai");
	await shot(dialog, "providers-settings", "custom-endpoint-litellm-dialog");
	await dialog.getByTestId("custom-endpoints-save").click();
	await expect(dialog).toHaveCount(0);

	const endpoint = group.locator(
		'[data-testid="custom-endpoints-endpoint"][data-endpoint^="custom-corp-litellm-"]',
	);
	await expect(endpoint).toContainText("4 requests at a time");
	await expect(endpoint.getByTestId("custom-endpoints-key-status")).toContainText(
		"$46 of $50 spent · 60 req/min · 4 parallel requests",
	);
	const providerId = (await endpoint.getAttribute("data-endpoint")) ?? "";
	const saved = JSON.parse(readFileSync(modelsPath, "utf8"));
	expect(saved.providers[providerId].models[0]).toMatchObject({
		id: MODEL,
		contextWindow: 131_072,
		maxTokens: 16_384,
		cost: { input: 0.2, output: 0.6 },
	});
	await shot(group, "providers-settings", "custom-endpoint-litellm");

	await page.keyboard.press("Escape");
	await page.getByTestId("model-selector").last().click();
	await revealModelGroups(page);
	await expandModelGroup(page, providerId);
	await page
		.locator(
			`[data-testid="model-option"][data-provider="${providerId}"][data-model-id="${MODEL}"]`,
		)
		.click();
	await page.getByTestId("chat-input").fill("Say hello");
	await page.getByTestId("chat-send").click();
	await waitForAgentSettled(page);
	await expect(
		page.locator('[data-testid="chat-message"][data-role="assistant"]').last(),
	).toContainText(REPLY);
	const sent = chatHeaders.at(-1);
	expect(sent?.tags).toBe("thinkrail,team:ai");
	expect(sent?.session).toMatch(/\S/);

	await openProviders(page);
	await endpoint.getByTestId("custom-endpoints-remove").click();
	await page.getByTestId("custom-endpoints-remove-confirm").click();
	await expect(endpoint).toHaveCount(0);
});
