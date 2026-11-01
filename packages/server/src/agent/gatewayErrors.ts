import type { CustomEndpointKeyStatus } from "@thinkrail/contracts";

type Json = Record<string, unknown>;

function isObject(value: unknown): value is Json {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

export type GatewayErrorKind = "budget" | "rate-limit" | "expired-key" | "invalid-key";

/** The message and LiteLLM error type of a failed response body, in either OpenAI or Anthropic shape. */
export function gatewayError(body: unknown): { message: string; type: string | null } | null {
	if (!isObject(body)) return null;
	const error = isObject(body.error) ? body.error : null;
	if (error && typeof error.message === "string")
		return { message: error.message, type: typeof error.type === "string" ? error.type : null };
	if (typeof body.detail === "string") return { message: body.detail, type: null };
	return null;
}

export function classifyGatewayError(
	status: number,
	error: { message: string; type: string | null },
): GatewayErrorKind | null {
	const text = error.message.toLowerCase();
	if (error.type === "budget_exceeded" || /exceeded ?budget|budget has been exceeded/.test(text))
		return "budget";
	if (status === 401 || status === 403) {
		if (text.includes("not allowed") || text.includes("access to model")) return null;
		if (text.includes("expired")) return "expired-key";
		if (text.includes("authentication error") || text.includes("invalid")) return "invalid-key";
		return null;
	}
	if (status === 429) return "rate-limit";
	return null;
}

function money(value: number): string {
	return `$${value.toFixed(value < 1 ? 4 : 2).replace(/\.?0+$/, "")}`;
}

function day(iso: string | null): string | null {
	if (!iso) return null;
	const time = Date.parse(iso);
	return Number.isFinite(time) ? new Date(time).toISOString().slice(0, 10) : null;
}

/** Plain words for a LiteLLM refusal, naming the server and, for a budget, what the key reports. */
export function gatewayErrorMessage(input: {
	kind: GatewayErrorKind;
	server: string;
	original: string;
	key: CustomEndpointKeyStatus | null;
	retryAfterSeconds: number | null;
}): string {
	const { kind, server, original, key } = input;
	switch (kind) {
		case "budget": {
			const spend =
				key?.spend !== null && key?.spend !== undefined && key.maxBudget !== null
					? ` (${money(key.spend)} of ${money(key.maxBudget)} spent)`
					: "";
			const reset = day(key?.budgetResetAt ?? null);
			return `The budget for your key on ${server} is used up${spend}. ${
				reset ? `It resets on ${reset}.` : "Ask your platform team to raise it."
			} Gateway said: ${original}`;
		}
		case "rate-limit": {
			const wait =
				input.retryAfterSeconds !== null ? ` Try again in ${input.retryAfterSeconds}s.` : "";
			return `${server} is rate-limiting your key; ThinkRail already retried.${wait} Lower Parallel requests for this server if it keeps happening. Gateway said: ${original}`;
		}
		case "expired-key":
			return `Your key for ${server} has expired. Create a new one and paste it in Settings → Providers → Custom servers → ${server}. Gateway said: ${original}`;
		case "invalid-key":
			return `${server} rejected your key. Check it in Settings → Providers → Custom servers → ${server}. Gateway said: ${original}`;
	}
}

/** Rebuilds an error body with a new message, keeping the rest of its shape for the SDK to parse. */
export function rewriteErrorBody(body: unknown, message: string): Json {
	if (isObject(body) && isObject(body.error)) return { ...body, error: { ...body.error, message } };
	return { error: { message, type: "gateway_error" } };
}

export function retryAfterSeconds(headers: Headers): number | null {
	const value = headers.get("retry-after");
	if (!value) return null;
	const seconds = Number(value);
	if (Number.isFinite(seconds) && seconds >= 0) return Math.ceil(seconds);
	const date = Date.parse(value);
	return Number.isFinite(date) ? Math.max(0, Math.ceil((date - Date.now()) / 1000)) : null;
}
