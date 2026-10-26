import type { ModelRuntime } from "@earendil-works/pi-coding-agent";
import { readLocalEndpointSettings } from "./localModelSettings";

type Fetch = typeof globalThis.fetch;
type Release = () => void;

const LIMITED = Symbol("thinkrail.requestLimit");
const INSTALLED = Symbol("thinkrail.requestLimitsInstalled");

/** A per-provider counting gate whose limit is re-read on every acquire and release. */
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

/** In-flight and queued HTTP requests ThinkRail holds for one provider (diagnostics and tests). */
export function requestLoad(providerId: string): { active: number; queued: number } {
	const gate = gates.get(providerId);
	return { active: gate?.active ?? 0, queued: gate?.queued ?? 0 };
}

/** Holds a slot from the request until its response body is fully read, cancelled or fails. */
export function limitedFetch(providerId: string, base: Fetch = globalThis.fetch): Fetch {
	const gate = gateFor(providerId);
	const wrapped = (async (input: Parameters<Fetch>[0], init?: Parameters<Fetch>[1]) => {
		const release = await gate.acquire(init?.signal);
		let response: Response;
		try {
			response = await base(input, init);
		} catch (error) {
			release();
			throw error;
		}
		if (!response.body) {
			release();
			return response;
		}
		const reader = response.body.getReader();
		const body = new ReadableStream<Uint8Array>({
			async pull(controller) {
				try {
					const chunk = await reader.read();
					if (chunk.done) {
						release();
						controller.close();
					} else controller.enqueue(chunk.value);
				} catch (error) {
					release();
					controller.error(error);
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

/** Applies the endpoint's saved limit, timeout and retries; saved values win over pi's global settings. */
function withEndpointOptions<T extends RequestOptions>(providerId: string, options: T): T {
	const settings = readLocalEndpointSettings()[providerId];
	if (!settings) return options;
	const next: NonNullable<RequestOptions> = { ...(options ?? {}) };
	if (settings.timeoutSeconds !== undefined) next.timeoutMs = settings.timeoutSeconds * 1_000;
	if (settings.maxRetries !== undefined) next.maxRetries = settings.maxRetries;
	const current = next.fetch as (Fetch & { [LIMITED]?: true }) | undefined;
	if (settings.maxConcurrency !== undefined && !current?.[LIMITED])
		next.fetch = limitedFetch(providerId, current);
	return next as T;
}

/**
 * Routes every model request of `runtime` through the per-endpoint options saved for local servers —
 * the parallel-request limit (a per-request `fetch`), the request timeout and provider retries. Every
 * pi caller — agent turns, subagents, compaction and one-shot completions — dispatches through these.
 */
export function installRequestLimits(runtime: ModelRuntime): void {
	const target = runtime as ModelRuntime & { [INSTALLED]?: true };
	if (target[INSTALLED]) return;
	target[INSTALLED] = true;
	const stream = runtime.stream.bind(runtime);
	const streamSimple = runtime.streamSimple.bind(runtime);
	const complete = runtime.complete.bind(runtime);
	const completeSimple = runtime.completeSimple.bind(runtime);
	runtime.stream = (model, context, options) =>
		stream(model, context, withEndpointOptions(model.provider, options));
	runtime.streamSimple = (model, context, options) =>
		streamSimple(model, context, withEndpointOptions(model.provider, options));
	runtime.complete = (model, context, options) =>
		complete(model, context, withEndpointOptions(model.provider, options));
	runtime.completeSimple = (model, context, options) =>
		completeSimple(model, context, withEndpointOptions(model.provider, options));
}
