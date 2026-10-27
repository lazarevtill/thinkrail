import type { ModelRuntime } from "@earendil-works/pi-coding-agent";
import { readLocalEndpointSettings } from "./localModelSettings";

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
	return readLocalEndpointSettings()[providerId]?.maxConcurrency;
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

type RequestOptions = { fetch?: Fetch; timeoutMs?: number; maxRetries?: number } | undefined;

export function withEndpointOptions<T extends RequestOptions>(providerId: string, options: T): T {
	const settings = readLocalEndpointSettings()[providerId];
	if (!settings) return options;
	const next: NonNullable<RequestOptions> = { ...(options ?? {}) };
	if (settings.maxRetries !== undefined && next.maxRetries !== 0)
		next.maxRetries = settings.maxRetries;
	const timeoutMs =
		settings.timeoutSeconds !== undefined ? settings.timeoutSeconds * 1_000 : next.timeoutMs;
	const current = next.fetch as (Fetch & { [LIMITED]?: true }) | undefined;
	if (current?.[LIMITED]) return next as T;
	if (settings.maxConcurrency === undefined) {
		if (timeoutMs !== undefined) next.timeoutMs = timeoutMs;
		return next as T;
	}
	next.fetch = limitedFetch(providerId, timeoutMs ?? OPENAI_SDK_DEFAULT_TIMEOUT_MS, current);
	next.timeoutMs = UNBOUNDED_TIMEOUT_MS;
	return next as T;
}

export function installRequestLimits(runtime: ModelRuntime): void {
	const target = runtime as ModelRuntime & { [INSTALLED]?: true };
	if (!Object.isExtensible(target) || target[INSTALLED]) return;
	target[INSTALLED] = true;
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
