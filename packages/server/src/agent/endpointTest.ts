import {
	type CustomEndpointTestResult,
	customEndpointTestBudgetMs,
	isCustomEndpointId,
} from "@thinkrail/contracts";
import { Type } from "typebox";
import { usePiRuntime } from "./agentSessionManager";
import { endpointLimitsOf } from "./endpointLimits";
import { requestLoad } from "./requestLimits";

const TEST_TOOL = {
	name: "get_current_time",
	description: "Returns the current time in the given IANA timezone.",
	parameters: Type.Object({
		timezone: Type.String({ description: "IANA timezone, e.g. UTC" }),
	}),
};

function failureText(message: { stopReason: string; errorMessage?: string }): string | null {
	if (message.stopReason !== "error" && message.stopReason !== "aborted") return null;
	return message.errorMessage?.trim() || "The request failed without an error message.";
}

export function testCustomEndpoint(id: string, modelId: string): Promise<CustomEndpointTestResult> {
	if (typeof id !== "string" || !isCustomEndpointId(id) || typeof modelId !== "string")
		return Promise.reject(new Error("That model is not available. Save the server and retry."));
	return usePiRuntime(async (runtime) => {
		await runtime.refresh({ allowNetwork: false });
		const model = runtime.getAvailableSnapshot().find((m) => m.provider === id && m.id === modelId);
		if (!model) throw new Error("That model is not available. Save the server and retry.");
		const limits = endpointLimitsOf(id);
		if (limits.maxConcurrency !== null && requestLoad(id).active >= limits.maxConcurrency)
			return {
				chat: "skipped",
				tools: "skipped",
				latencyMs: null,
				outputTokensPerSecond: null,
				message: null,
			};
		const options = {
			maxTokens: 256,
			signal: AbortSignal.timeout(customEndpointTestBudgetMs(limits)),
		};
		const started = performance.now();
		const chat = await runtime.completeSimple(
			model,
			{
				systemPrompt: "You are a connectivity check. Reply with the single word OK.",
				messages: [{ role: "user", content: "Reply with OK.", timestamp: Date.now() }],
			},
			options,
		);
		const elapsed = performance.now() - started;
		const chatFailure = failureText(chat);
		if (chatFailure)
			return {
				chat: "failed",
				tools: "skipped",
				latencyMs: Math.round(elapsed),
				outputTokensPerSecond: null,
				message: chatFailure,
			};
		const seconds = elapsed / 1000;
		const outputTokensPerSecond =
			chat.usage.output > 0 && seconds > 0
				? Math.round((chat.usage.output / seconds) * 10) / 10
				: null;
		const agent = await runtime.completeSimple(
			model,
			{
				systemPrompt:
					"You are an agent with tools. When a tool can answer, call it instead of replying in text.",
				messages: [
					{ role: "user", content: "What time is it in UTC? Use the tool.", timestamp: Date.now() },
				],
				tools: [TEST_TOOL],
			},
			options,
		);
		const called = agent.content.some(
			(block) => block.type === "toolCall" && block.name === TEST_TOOL.name,
		);
		return {
			chat: "ok",
			tools: called ? "ok" : "failed",
			latencyMs: Math.round(elapsed),
			outputTokensPerSecond,
			message: failureText(agent),
		};
	});
}
