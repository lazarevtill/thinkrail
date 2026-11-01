import type { AnyModel } from "@earendil-works/pi-ai";
import type { ModelRuntime, ModelRuntimeAuthOverrides } from "@earendil-works/pi-coding-agent";
import {
	CUSTOM_ENDPOINT_ID_PREFIX,
	type CustomEndpointGateway,
	type CustomEndpointKeyStatus,
} from "@thinkrail/contracts";
import { commandToken, forgetCommandToken, networkFetch } from "./endpointAccess";
import { type EndpointLimits, readEndpointLimits } from "./endpointLimits";
import {
	classifyGatewayError,
	gatewayError,
	gatewayErrorMessage,
	retryAfterSeconds,
	rewriteErrorBody,
} from "./gatewayErrors";
import { managedOrigin, policyRefusal, providerAllowed, readManagedConfig } from "./managedConfig";

type Fetch = typeof globalThis.fetch;
type Release = () => void;

const LIMITED = Symbol("thinkrail.requestLimit");
const INSTALLED = Symbol("thinkrail.requestLimitsInstalled");

class Gate {
	active = 0;
	private readonly waiting: { start: () => void; signal?: AbortSignal | undefined }[] = [];

	constructor(private readonly limit: () => number | undefined) {}

	acquire(signal?: AbortSignal | null): Promise<Release> {
		if (signal?.aborted) return Promise.reject(signal.reason);
		return new Promise((resolve, reject) => {
			const entry = {
				signal: signal ?? undefined,
				start: () => {
					signal?.removeEventListener("abort", onAbort);
					this.active++;
					let released = false;
					resolve(() => {
						if (released) return;
						released = true;
						this.active--;
						this.drain();
					});
				},
			};
			const onAbort = () => {
				const index = this.waiting.indexOf(entry);
				if (index >= 0) this.waiting.splice(index, 1);
				reject(signal?.reason);
			};
			signal?.addEventListener("abort", onAbort, { once: true });
			this.waiting.push(entry);
			this.drain();
		});
	}

	get queued(): number {
		return this.waiting.length;
	}

	private drain(): void {
		while (this.waiting.length > 0) {
			const limit = this.limit();
			if (limit !== undefined && this.active >= limit) return;
			this.waiting.shift()?.start();
		}
	}
}

const gates = new Map<string, Gate>();

function limitFor(providerId: string): number | undefined {
	return readEndpointLimits()[providerId]?.maxConcurrency;
}

function gateFor(providerId: string): Gate {
	let gate = gates.get(providerId);
	if (!gate) {
		gate = new Gate(() => limitFor(providerId));
		gates.set(providerId, gate);
	}
	return gate;
}

export function forgetRequestGate(providerId: string): void {
	const gate = gates.get(providerId);
	if (gate && gate.active === 0 && gate.queued === 0) gates.delete(providerId);
}

export function requestLoad(providerId: string): { active: number; queued: number } {
	const gate = gates.get(providerId);
	return { active: gate?.active ?? 0, queued: gate?.queued ?? 0 };
}

const OPENAI_SDK_DEFAULT_TIMEOUT_MS = 600_000;
const UNBOUNDED_TIMEOUT_MS = 2_147_483_647;

function timeoutError(timeoutMs: number): Error {
	const error = new Error(`Request timed out after ${Math.round(timeoutMs / 1000)}s`);
	error.name = "TimeoutError";
	return error;
}

export function limitedFetch(
	providerId: string,
	headerTimeoutMs: number,
	base: Fetch = globalThis.fetch,
): Fetch {
	const gate = gateFor(providerId);
	const wrapped = (async (input: Parameters<Fetch>[0], init?: Parameters<Fetch>[1]) => {
		const release = await gate.acquire(init?.signal);
		const timer = new AbortController();
		const timeout = setTimeout(() => timer.abort(timeoutError(headerTimeoutMs)), headerTimeoutMs);
		const signal = init?.signal ? AbortSignal.any([init.signal, timer.signal]) : timer.signal;
		let response: Response;
		try {
			response = await base(input, { ...init, signal });
		} catch (error) {
			release();
			throw timer.signal.aborted ? timer.signal.reason : error;
		} finally {
			clearTimeout(timeout);
		}
		if (!response.body) {
			release();
			return response;
		}
		const reader = response.body.getReader();
		const body = new ReadableStream<Uint8Array>({
			async pull(controller) {
				let idle: ReturnType<typeof setTimeout> | undefined;
				const stalled = new Promise<never>((_, reject) => {
					idle = setTimeout(() => reject(timeoutError(headerTimeoutMs)), headerTimeoutMs);
				});
				try {
					const chunk = await Promise.race([reader.read(), stalled]);
					if (chunk.done) {
						release();
						controller.close();
					} else controller.enqueue(chunk.value);
				} catch (error) {
					release();
					reader.cancel(error).catch(() => {});
					controller.error(error);
				} finally {
					clearTimeout(idle);
				}
			},
			cancel(reason) {
				release();
				return reader.cancel(reason);
			},
		});
		return new Response(body, {
			status: response.status,
			statusText: response.statusText,
			headers: response.headers,
		});
	}) as Fetch & { [LIMITED]?: true };
	wrapped[LIMITED] = true;
	return wrapped;
}

type RequestOptions =
	| {
			fetch?: Fetch;
			timeoutMs?: number;
			maxRetries?: number;
			sessionId?: string;
			headers?: Record<string, string | null>;
	  }
	| undefined;

export const LITELLM_SESSION_HEADER = "x-litellm-session-id";
export const LITELLM_TAGS_HEADER = "x-litellm-tags";

function gatewayHeaders(
	gateway: CustomEndpointGateway | undefined,
	sessionId: string | undefined,
): Record<string, string> {
	const headers: Record<string, string> = {};
	if (gateway?.kind !== "litellm") return headers;
	if (gateway.sessionAffinity && sessionId) headers[LITELLM_SESSION_HEADER] = sessionId;
	if (gateway.tags.length > 0) headers[LITELLM_TAGS_HEADER] = gateway.tags.join(",");
	return headers;
}

/** Fork: what the request layer needs to know about a saved endpoint, supplied by customEndpoints. */
export interface EndpointDirectory {
	name(providerId: string): string;
	keyCommand(providerId: string): string | undefined;
	keyStatus(providerId: string): Promise<CustomEndpointKeyStatus | null>;
}

let directory: EndpointDirectory | null = null;

export function setEndpointDirectory(next: EndpointDirectory | null): void {
	directory = next;
}

function isEndpoint(providerId: string): boolean {
	return providerId.startsWith(CUSTOM_ENDPOINT_ID_PREFIX);
}

async function explainGatewayFailure(providerId: string, response: Response): Promise<Response> {
	const text = await response.text();
	const rebuilt = (body: string) =>
		new Response(body, {
			status: response.status,
			statusText: response.statusText,
			headers: (() => {
				const headers = new Headers(response.headers);
				headers.delete("content-length");
				headers.delete("content-encoding");
				return headers;
			})(),
		});
	let body: unknown;
	try {
		body = JSON.parse(text);
	} catch {
		return rebuilt(text);
	}
	const error = gatewayError(body);
	const kind = error ? classifyGatewayError(response.status, error) : null;
	if (!error || !kind) return rebuilt(text);
	const key =
		kind === "budget" ? await (directory?.keyStatus(providerId).catch(() => null) ?? null) : null;
	const message = gatewayErrorMessage({
		kind,
		server: directory?.name(providerId) ?? providerId,
		original: error.message,
		key,
		retryAfterSeconds: retryAfterSeconds(response.headers),
	});
	return rebuilt(JSON.stringify(rewriteErrorBody(body, message)));
}

/**
 * Fork: a saved endpoint's fetch — its certificate and proxy settings, a fresh token after the
 * server refuses a cached one, and plain-words LiteLLM quota and key errors.
 */
function endpointFetch(
	providerId: string,
	settings: EndpointLimits | undefined,
	base?: Fetch,
): Fetch {
	const inner = settings?.network
		? networkFetch(settings.network, base)
		: (base ?? globalThis.fetch);
	const wrapped = Object.assign(
		async (input: Parameters<Fetch>[0], init?: Parameters<Fetch>[1]) => {
			const response = await inner(input, init);
			if (response.ok) return response;
			if (response.status === 401 || response.status === 403) {
				const command = directory?.keyCommand(providerId);
				if (command) forgetCommandToken(command);
			}
			return settings?.gateway?.kind === "litellm"
				? explainGatewayFailure(providerId, response)
				: response;
		},
		{ preconnect: inner.preconnect, [LIMITED]: true as const },
	);
	return wrapped;
}

export function withEndpointOptions<T extends RequestOptions>(providerId: string, options: T): T {
	const settings = readEndpointLimits()[providerId];
	if (!settings && !isEndpoint(providerId)) return options;
	const next: NonNullable<RequestOptions> = { ...(options ?? {}) };
	if (!settings) {
		const current = next.fetch as (Fetch & { [LIMITED]?: true }) | undefined;
		if (!current?.[LIMITED]) next.fetch = endpointFetch(providerId, undefined, current);
		return next as T;
	}
	const added = gatewayHeaders(settings.gateway, next.sessionId);
	if (Object.keys(added).length > 0) next.headers = { ...added, ...(next.headers ?? {}) };
	if (settings.maxRetries !== undefined && next.maxRetries !== 0)
		next.maxRetries = settings.maxRetries;
	const timeoutMs =
		settings.timeoutSeconds !== undefined ? settings.timeoutSeconds * 1_000 : next.timeoutMs;
	const current = next.fetch as (Fetch & { [LIMITED]?: true }) | undefined;
	if (current?.[LIMITED]) return next as T;
	const base = endpointFetch(providerId, settings, current);
	if (settings.maxConcurrency === undefined) {
		next.fetch = base;
		if (timeoutMs !== undefined) next.timeoutMs = timeoutMs;
		return next as T;
	}
	next.fetch = limitedFetch(providerId, timeoutMs ?? OPENAI_SDK_DEFAULT_TIMEOUT_MS, base);
	next.timeoutMs = UNBOUNDED_TIMEOUT_MS;
	return next as T;
}

type AuthOverrides = ModelRuntimeAuthOverrides | undefined;

/**
 * Fork: every model request resolves its auth here. A refused provider fails before anything is sent;
 * an endpoint whose key is a `!command` gets the cached token instead of pi running the command on
 * every request.
 */
function originOf(url: unknown): string | null {
	if (typeof url !== "string") return null;
	try {
		return new URL(url).origin;
	} catch {
		return null;
	}
}

async function endpointAuthOverrides(
	model: AnyModel,
	overrides: AuthOverrides,
): Promise<AuthOverrides> {
	const providerId = model.provider;
	if (!providerAllowed(providerId)) throw new Error(policyRefusal(providerId));
	const origin = managedOrigin(providerId);
	if (
		origin !== null &&
		readManagedConfig().restrictModels !== null &&
		originOf(model.baseUrl) !== origin
	)
		throw new Error(
			`The organization's server "${providerId}" must be reached at ${origin}; its saved address was changed. Open Settings → Providers to restore it.`,
		);
	if (overrides?.apiKey !== undefined || !isEndpoint(providerId)) return overrides;
	const command = directory?.keyCommand(providerId);
	if (!command) return overrides;
	const token = await commandToken(command);
	return token === undefined ? overrides : { ...overrides, apiKey: token };
}

export function installRequestLimits(runtime: ModelRuntime): void {
	const target = runtime as ModelRuntime & { [INSTALLED]?: true };
	if (!Object.isExtensible(target) || target[INSTALLED]) return;
	target[INSTALLED] = true;
	if (typeof runtime.getAuth === "function") {
		const getAuth = runtime.getAuth.bind(runtime);
		runtime.getAuth = async (target: string | AnyModel, overrides?: AuthOverrides) => {
			if (typeof target === "string") return getAuth(target, overrides);
			return getAuth(target, await endpointAuthOverrides(target, overrides));
		};
	}
	if (typeof runtime.stream === "function") {
		const stream = runtime.stream.bind(runtime);
		runtime.stream = (model, context, options) =>
			stream(model, context, withEndpointOptions(model.provider, options));
	}
	if (typeof runtime.streamSimple === "function") {
		const streamSimple = runtime.streamSimple.bind(runtime);
		runtime.streamSimple = (model, context, options) =>
			streamSimple(model, context, withEndpointOptions(model.provider, options));
	}
	if (typeof runtime.complete === "function") {
		const complete = runtime.complete.bind(runtime);
		runtime.complete = (model, context, options) =>
			complete(model, context, withEndpointOptions(model.provider, options));
	}
	if (typeof runtime.completeSimple === "function") {
		const completeSimple = runtime.completeSimple.bind(runtime);
		runtime.completeSimple = (model, context, options) =>
			completeSimple(model, context, withEndpointOptions(model.provider, options));
	}
}
