import { rmSync, writeFileSync } from "node:fs";
import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";
import { join } from "node:path";
import { expect, test } from "@playwright/test";
import {
	expandModelGroup,
	openWorkspaceChat,
	revealModelGroups,
	waitForAgentSettled,
} from "./fixtures/app";
import { openProviders } from "./fixtures/jbcentral";
import { E2E_MANAGED_CONFIG, E2E_PI_AGENT_DIR } from "./fixtures/paths";
import { shot } from "./fixtures/screenshots";
import { E2eWire } from "./fixtures/wire";

const modelsPath = join(E2E_PI_AGENT_DIR, "models.json");
const KEY = "sk-corp-e2e";
const PROVIDER = "custom-org-plata";
const CLAUDE = "claude-sonnet";
const QWEN = "qwen3-coder";
const REPLY = "CORPORATE_ANTHROPIC_REPLY_SENTINEL";
let server: Server;
let origin: string;
const requests: { path: string; key: string | null; org: string | null }[] = [];

function header(request: IncomingMessage, name: string): string | null {
	const value = request.headers[name];
	return typeof value === "string" ? value : null;
}

function json(response: ServerResponse, body: unknown, status = 200) {
	response.statusCode = status;
	response.setHeader("content-type", "application/json");
	response.end(JSON.stringify(body));
}

function anthropicStream(response: ServerResponse) {
	const events: [string, unknown][] = [
		[
			"message_start",
			{
				type: "message_start",
				message: {
					id: "msg_1",
					type: "message",
					role: "assistant",
					model: CLAUDE,
					content: [],
					stop_reason: null,
					stop_sequence: null,
					usage: { input_tokens: 10, output_tokens: 1 },
				},
			},
		],
		[
			"content_block_start",
			{ type: "content_block_start", index: 0, content_block: { type: "text", text: "" } },
		],
		[
			"content_block_delta",
			{ type: "content_block_delta", index: 0, delta: { type: "text_delta", text: REPLY } },
		],
		["content_block_stop", { type: "content_block_stop", index: 0 }],
		[
			"message_delta",
			{
				type: "message_delta",
				delta: { stop_reason: "end_turn", stop_sequence: null },
				usage: { output_tokens: 3 },
			},
		],
		["message_stop", { type: "message_stop" }],
	];
	response.setHeader("content-type", "text/event-stream");
	response.end(
		events.map(([event, data]) => `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`).join(""),
	);
}

test.beforeAll(async () => {
	server = createServer((request, response) => {
		const key =
			header(request, "x-api-key") ??
			header(request, "authorization")?.replace("Bearer ", "") ??
			null;
		const org = header(request, "x-org");
		const path = new URL(request.url ?? "/", "http://gateway").pathname;
		requests.push({ path, key, org });
		if (org !== "plata") return json(response, { error: { message: "missing org" } }, 403);
		if (key !== KEY)
			return json(response, { detail: "Authentication Error, Invalid API key provided." }, 401);
		request.resume();
		if (path === "/v1/models")
			return json(response, {
				object: "list",
				data: [CLAUDE, QWEN].map((id) => ({ id, object: "model" })),
			});
		if (path === "/model/info")
			return json(response, {
				data: [
					{
						model_name: CLAUDE,
						litellm_params: { model: "anthropic/claude-sonnet-5" },
						model_info: { mode: "chat", max_input_tokens: 200_000, max_output_tokens: 64_000 },
					},
					{
						model_name: QWEN,
						litellm_params: { model: "hosted_vllm/qwen" },
						model_info: { mode: "chat", max_input_tokens: 131_072 },
					},
				],
			});
		if (path === "/key/info") return json(response, { info: { spend: 1, max_budget: 100 } });
		if (path === "/v1/messages") return request.on("end", () => anthropicStream(response));
		return json(response, { error: { message: "not here" } }, 404);
	});
	await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
	const address = server.address();
	if (address === null || typeof address === "string")
		throw new Error("The fake gateway has no port.");
	origin = `http://127.0.0.1:${address.port}`;
});

test.afterAll(async () => {
	rmSync(E2E_MANAGED_CONFIG, { force: true });
	writeFileSync(modelsPath, JSON.stringify({ providers: {} }));
	await new Promise<void>((resolve) => server.close(() => resolve()));
});

test("fork: an organization-managed gateway is locked, takes the user's key, routes Claude natively and limits chats to it", async ({
	page,
}) => {
	writeFileSync(modelsPath, JSON.stringify({ providers: {} }));
	writeFileSync(
		E2E_MANAGED_CONFIG,
		JSON.stringify({
			servers: [
				{
					id: "plata",
					name: "Plata AI Gateway",
					baseUrl: `${origin}/v1`,
					headers: { "X-Org": "plata" },
					gateway: { kind: "litellm", sessionAffinity: true, tags: ["thinkrail"] },
				},
			],
			policy: { restrictModels: "managed" },
		}),
	);
	await openWorkspaceChat(page);
	await openProviders(page);
	const group = page.getByTestId("settings-custom-endpoints");
	await expect(group.getByTestId("custom-endpoints-policy")).toContainText(
		"Your organization allows chats only on its own servers.",
	);
	const endpoint = group.locator(
		`[data-testid="custom-endpoints-endpoint"][data-endpoint="${PROVIDER}"]`,
	);
	await expect(endpoint.getByTestId("custom-endpoints-managed")).toBeVisible();
	await expect(endpoint.getByTestId("custom-endpoints-needs-key")).toBeVisible();
	await expect(endpoint.getByTestId("custom-endpoints-remove")).toHaveCount(0);

	await endpoint.getByTestId("custom-endpoints-edit").click();
	const dialog = page.getByTestId("custom-endpoints-dialog");
	await expect(dialog.getByTestId("custom-endpoints-managed-note")).toBeVisible();
	await expect(dialog.getByTestId("custom-endpoints-name")).toBeDisabled();
	await expect(dialog.getByTestId("custom-endpoints-url")).toBeDisabled();
	await expect(dialog.getByTestId("custom-endpoints-header-name")).toBeDisabled();
	await expect(dialog.getByTestId("custom-endpoints-proxy")).toBeDisabled();
	await dialog.getByTestId("custom-endpoints-key").fill(KEY);
	await dialog.getByTestId("custom-endpoints-detect").click();
	await expect(dialog.getByTestId("custom-endpoints-status")).toHaveAttribute("data-tone", "ok");
	const claudeRow = dialog.locator(`[data-testid="custom-endpoints-row"][data-model="${CLAUDE}"]`);
	await expect(claudeRow.getByTestId("custom-endpoints-api")).toHaveValue("anthropic-messages");
	const qwenRow = dialog.locator(`[data-testid="custom-endpoints-row"][data-model="${QWEN}"]`);
	await expect(qwenRow.getByTestId("custom-endpoints-api")).toHaveValue("openai-completions");
	await shot(dialog, "providers-settings", "custom-endpoint-managed-dialog");
	await dialog.getByTestId("custom-endpoints-save").click();
	await expect(dialog).toHaveCount(0);
	await expect(endpoint.getByTestId("custom-endpoints-needs-key")).toHaveCount(0);
	await shot(group, "providers-settings", "custom-endpoint-managed");

	const wire = await E2eWire.connect(Number(new URL(page.url()).port));
	try {
		await expect
			.poll(async () => new Set((await wire.request("model.list", {})).map((m) => m.provider)))
			.toEqual(new Set([PROVIDER]));
	} finally {
		wire.close();
	}

	await page.keyboard.press("Escape");
	await page.getByTestId("model-selector").last().click();
	await revealModelGroups(page);
	await expandModelGroup(page, PROVIDER);
	await page
		.locator(`[data-testid="model-option"][data-provider="${PROVIDER}"][data-model-id="${CLAUDE}"]`)
		.click();
	await page.getByTestId("chat-input").fill("Say hello");
	await page.getByTestId("chat-send").click();
	await waitForAgentSettled(page);
	await expect(
		page.locator('[data-testid="chat-message"][data-role="assistant"]').last(),
	).toContainText(REPLY);
	expect(requests.filter((entry) => entry.path === "/v1/messages").at(-1)).toEqual({
		path: "/v1/messages",
		key: KEY,
		org: "plata",
	});

	rmSync(E2E_MANAGED_CONFIG, { force: true });
	await openProviders(page);
	await page.getByTestId("providers-refresh").click();
	await expect(endpoint).toHaveCount(0);
	await expect(group.getByTestId("custom-endpoints-policy")).toHaveCount(0);
});

test("fork: a new server's connection settings validate the proxy and headers before saving", async ({
	page,
}) => {
	writeFileSync(modelsPath, JSON.stringify({ providers: {} }));
	await openWorkspaceChat(page);
	await openProviders(page);
	const group = page.getByTestId("settings-custom-endpoints");
	await group.getByTestId("custom-endpoints-add").click();
	const dialog = page.getByTestId("custom-endpoints-dialog");
	await dialog.getByTestId("custom-endpoints-name").fill("Gateway");
	await dialog.getByTestId("custom-endpoints-url").fill(`${origin}/v1`);
	await dialog.getByTestId("custom-endpoints-key").fill(KEY);
	await dialog.getByTestId("custom-endpoints-detect").click();
	await expect(dialog.getByTestId("custom-endpoints-status")).toHaveAttribute("data-tone", "error");
	await dialog.getByTestId("custom-endpoints-header-add").click();
	await dialog.getByTestId("custom-endpoints-header-name").fill("X-Org");
	await dialog.getByTestId("custom-endpoints-header-value").fill("plata");
	await dialog.getByTestId("custom-endpoints-detect").click();
	await expect(dialog.getByTestId("custom-endpoints-status")).toHaveAttribute("data-tone", "ok");
	const save = dialog.getByTestId("custom-endpoints-save");
	await dialog.getByTestId("custom-endpoints-proxy").fill("http://user:pw@proxy:3128");
	await expect(dialog.getByTestId("custom-endpoints-proxy")).toHaveAttribute(
		"aria-invalid",
		"true",
	);
	await expect(save).toBeDisabled();
	await dialog.getByTestId("custom-endpoints-proxy").fill("");
	await expect(save).toBeEnabled();
	await shot(dialog, "providers-settings", "custom-endpoint-connection");
	await save.click();
	await expect(dialog).toHaveCount(0);
	const endpoint = group.locator(
		'[data-testid="custom-endpoints-endpoint"][data-endpoint^="custom-gateway-"]',
	);
	await endpoint.getByTestId("custom-endpoints-edit").click();
	await expect(dialog.getByTestId("custom-endpoints-header-name")).toHaveValue("X-Org");
	await expect(dialog.getByTestId("custom-endpoints-header-value")).toHaveValue("");
	await expect(dialog.getByTestId("custom-endpoints-status")).toHaveAttribute("data-tone", "ok");
	await page.keyboard.press("Escape");
	await endpoint.getByTestId("custom-endpoints-remove").click();
	await page.getByTestId("custom-endpoints-remove-confirm").click();
	await expect(endpoint).toHaveCount(0);
});
