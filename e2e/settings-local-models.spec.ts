import { readFileSync, writeFileSync } from "node:fs";
import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
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
import { E2eWire } from "./fixtures/wire";

const modelsPath = join(E2E_PI_AGENT_DIR, "models.json");
const MODEL = "qwen3-coder-local";
const REPLY = "LOCAL_MODEL_REPLY_SENTINEL";
let server: Server;
let baseUrl: string;
const chatRequests: Record<string, unknown>[] = [];

function sse(chunks: unknown[]): string {
	return `${chunks.map((chunk) => `data: ${JSON.stringify(chunk)}\n\n`).join("")}data: [DONE]\n\n`;
}

test.beforeAll(async () => {
	server = createServer((request, response) => {
		if (request.method === "GET" && request.url === "/v1/models") {
			response.setHeader("content-type", "application/json");
			response.end(
				JSON.stringify({
					object: "list",
					data: [{ id: MODEL, object: "model", max_model_len: 65_536 }],
				}),
			);
			return;
		}
		if (request.method === "POST" && request.url === "/v1/chat/completions") {
			let body = "";
			request.on("data", (chunk) => {
				body += chunk;
			});
			request.on("end", () => {
				chatRequests.push(JSON.parse(body));
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
	baseUrl = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
});

test.afterAll(async () => {
	writeFileSync(modelsPath, JSON.stringify({ providers: {} }));
	await new Promise<void>((resolve) => server.close(() => resolve()));
});

test("a local vLLM server is detected, saved to models.json, chatted with, and removed", async ({
	page,
}) => {
	writeFileSync(modelsPath, JSON.stringify({ providers: {} }));
	await openWorkspaceChat(page);
	await openProviders(page);
	const group = page.getByTestId("settings-local-models");
	await expect(group).toBeVisible();

	await group.getByTestId("local-models-add").click();
	const dialog = page.getByTestId("local-models-dialog");
	await dialog.getByTestId("local-models-kind-vllm").click();
	await dialog.getByTestId("local-models-url").fill(baseUrl);
	await dialog.getByTestId("local-models-detect").click();
	const row = dialog.locator(`[data-testid="local-models-row"][data-model="${MODEL}"]`);
	await expect(row).toBeVisible();
	await expect(row.getByRole("textbox")).toHaveValue("65536");
	await dialog.getByTestId("local-models-save").click();
	await expect(dialog).toHaveCount(0);

	const endpoint = group.locator(
		'[data-testid="local-models-endpoint"][data-endpoint="local-vllm"]',
	);
	await expect(endpoint).toContainText(`${baseUrl}/v1`);
	const saved = JSON.parse(readFileSync(modelsPath, "utf8"));
	expect(saved.providers["local-vllm"]).toMatchObject({
		api: "openai-completions",
		baseUrl: `${baseUrl}/v1`,
		models: [{ id: MODEL, contextWindow: 65_536 }],
	});

	const wire = await E2eWire.connect(Number(new URL(page.url()).port));
	try {
		await expect
			.poll(async () =>
				(await wire.request("model.list", {})).some(
					(model) => model.provider === "local-vllm" && model.id === MODEL,
				),
			)
			.toBe(true);
	} finally {
		wire.close();
	}

	await page.keyboard.press("Escape");
	await page.getByTestId("model-selector").last().click();
	await revealModelGroups(page);
	await expandModelGroup(page, "local-vllm");
	await page
		.locator(`[data-testid="model-option"][data-provider="local-vllm"][data-model-id="${MODEL}"]`)
		.click();
	await page.getByTestId("chat-input").fill("Say hello");
	await page.getByTestId("chat-send").click();
	await waitForAgentSettled(page);
	await expect(
		page.locator('[data-testid="chat-message"][data-role="assistant"]').last(),
	).toContainText(REPLY);
	expect(chatRequests.at(-1)).toMatchObject({ model: MODEL, stream: true });
	expect(chatRequests.at(-1)).not.toHaveProperty("store");

	await openProviders(page);
	await endpoint.getByTestId("local-models-remove").click();
	await page.getByTestId("local-models-remove-confirm").click();
	await expect(endpoint).toHaveCount(0);
	expect(JSON.parse(readFileSync(modelsPath, "utf8")).providers).toEqual({});
});
