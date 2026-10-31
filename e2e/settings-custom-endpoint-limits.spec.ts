import { writeFileSync } from "node:fs";
import { createServer, type Server } from "node:http";
import { join } from "node:path";
import { expect, test } from "@playwright/test";
import { openWorkspaceChat } from "./fixtures/app";
import { openProviders } from "./fixtures/jbcentral";
import { E2E_PI_AGENT_DIR } from "./fixtures/paths";
import { shot } from "./fixtures/screenshots";
import { E2eWire } from "./fixtures/wire";

const modelsPath = join(E2E_PI_AGENT_DIR, "models.json");
const MODEL = "qwen3-coder-local";
let server: Server;
let origin: string;
const chatRequests: Record<string, unknown>[] = [];

function sse(chunks: unknown[]): string {
	return `${chunks.map((chunk) => `data: ${JSON.stringify(chunk)}\n\n`).join("")}data: [DONE]\n\n`;
}

function callsTimeTool(body: unknown): boolean {
	if (typeof body !== "object" || body === null || !("tools" in body)) return false;
	return Array.isArray(body.tools) && JSON.stringify(body.tools).includes("get_current_time");
}

test.beforeAll(async () => {
	server = createServer((request, response) => {
		if (request.method === "GET" && request.url === "/v1/models") {
			response.setHeader("content-type", "application/json");
			response.end(
				JSON.stringify({
					object: "list",
					data: [{ id: MODEL, object: "model", owned_by: "vllm", max_model_len: 65_536 }],
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
				const parsed: unknown = JSON.parse(body);
				if (typeof parsed === "object" && parsed !== null)
					chatRequests.push(Object.fromEntries(Object.entries(parsed)));
				const base = { id: "c1", object: "chat.completion.chunk", created: 1, model: MODEL };
				response.setHeader("content-type", "text/event-stream");
				if (callsTimeTool(parsed)) {
					response.end(
						sse([
							{
								...base,
								choices: [
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
								],
							},
							{ ...base, choices: [{ index: 0, delta: {}, finish_reason: "tool_calls" }] },
						]),
					);
					return;
				}
				response.end(
					sse([
						{ ...base, choices: [{ index: 0, delta: { role: "assistant", content: "OK" } }] },
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

test("fork: a server's request limits are validated, saved, shown and cleared, and Test checks chat and tools", async ({
	page,
}) => {
	writeFileSync(modelsPath, JSON.stringify({ providers: {} }));
	await openWorkspaceChat(page);
	await openProviders(page);
	const group = page.getByTestId("settings-custom-endpoints");
	await group.getByTestId("custom-endpoints-add").click();
	const dialog = page.getByTestId("custom-endpoints-dialog");
	await dialog.getByTestId("custom-endpoints-name").fill("Local vLLM");
	await dialog.getByTestId("custom-endpoints-url").fill(origin);
	await dialog.getByTestId("custom-endpoints-detect").click();
	await expect(dialog.getByTestId("custom-endpoints-status")).toHaveAttribute("data-tone", "ok");

	const save = dialog.getByTestId("custom-endpoints-save");
	await dialog.getByTestId("custom-endpoints-parallel").fill("0");
	await expect(dialog.getByTestId("custom-endpoints-parallel")).toHaveAttribute(
		"aria-invalid",
		"true",
	);
	await expect(save).toBeDisabled();
	await dialog.getByTestId("custom-endpoints-parallel").fill("2");
	await dialog.getByTestId("custom-endpoints-timeout").fill("600");
	await dialog.getByTestId("custom-endpoints-retries").fill("3");
	await expect(save).toBeEnabled();
	await shot(dialog, "providers-settings", "custom-endpoint-limits");
	await save.click();
	await expect(dialog).toHaveCount(0);

	const endpoint = group.locator(
		'[data-testid="custom-endpoints-endpoint"][data-endpoint^="custom-local-vllm-"]',
	);
	await expect(endpoint).toContainText("2 requests at a time · 600 s timeout · 3 retries");
	const providerId = (await endpoint.getAttribute("data-endpoint")) ?? "";

	const wire = await E2eWire.connect(Number(new URL(page.url()).port));
	try {
		const listed = await wire.request("customEndpoints.list", {});
		expect(listed.find((entry) => entry.id === providerId)?.limits).toEqual({
			maxConcurrency: 2,
			timeoutSeconds: 600,
			maxRetries: 3,
		});
	} finally {
		wire.close();
	}

	const tested = endpoint.locator(`[data-testid="custom-endpoints-model"][data-model="${MODEL}"]`);
	await tested.getByTestId("custom-endpoints-test").click();
	await expect(tested).toHaveAttribute("data-test-state", "ok", { timeout: 30_000 });
	await expect(tested.getByTestId("custom-endpoints-test-result")).toContainText("Tool calls OK");
	expect(chatRequests.some(callsTimeTool)).toBe(true);
	await shot(group, "providers-settings", "custom-endpoint-tested");

	await endpoint.getByTestId("custom-endpoints-edit").click();
	await expect(dialog.getByTestId("custom-endpoints-parallel")).toHaveValue("2");
	await expect(dialog.getByTestId("custom-endpoints-timeout")).toHaveValue("600");
	await expect(dialog.getByTestId("custom-endpoints-retries")).toHaveValue("3");
	await dialog.getByTestId("custom-endpoints-parallel").fill("");
	await dialog.getByTestId("custom-endpoints-timeout").fill("");
	await dialog.getByTestId("custom-endpoints-retries").fill("");
	await dialog.getByTestId("custom-endpoints-save").click();
	await expect(dialog).toHaveCount(0);
	await expect(endpoint).toContainText("no parallel limit");
	await expect(endpoint).not.toContainText("timeout");

	await endpoint.getByTestId("custom-endpoints-remove").click();
	await page.getByTestId("custom-endpoints-remove-confirm").click();
	await expect(endpoint).toHaveCount(0);
});
