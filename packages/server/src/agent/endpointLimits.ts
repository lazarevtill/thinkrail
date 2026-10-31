import { mkdirSync, readFileSync, renameSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import {
	CUSTOM_ENDPOINT_TAG_LIMITS,
	type CustomEndpointGateway,
	type CustomEndpointLimits,
	isEndpointConcurrency,
	isEndpointRetries,
	isEndpointTag,
	isEndpointTimeoutSeconds,
} from "@thinkrail/contracts";
import { dataDir } from "../persistence";

export interface EndpointLimits {
	maxConcurrency?: number;
	timeoutSeconds?: number;
	maxRetries?: number;
	gateway?: CustomEndpointGateway;
}

let cache: { path: string; endpoints: Record<string, EndpointLimits>; unreadable: boolean } | null =
	null;

function limitsPath(): string {
	return join(dataDir(), "local-models.json");
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

function sanitize(value: unknown): EndpointLimits {
	if (!isRecord(value)) return {};
	const limits: EndpointLimits = {};
	if (isEndpointConcurrency(value.maxConcurrency)) limits.maxConcurrency = value.maxConcurrency;
	if (isEndpointTimeoutSeconds(value.timeoutSeconds)) limits.timeoutSeconds = value.timeoutSeconds;
	if (isEndpointRetries(value.maxRetries)) limits.maxRetries = value.maxRetries;
	const gateway = sanitizeGateway(value.gateway);
	if (gateway) limits.gateway = gateway;
	return limits;
}

export function sanitizeGateway(value: unknown): CustomEndpointGateway | null {
	if (!isRecord(value) || value.kind !== "litellm") return null;
	const tags = Array.isArray(value.tags) ? value.tags.filter(isEndpointTag) : [];
	return {
		kind: "litellm",
		sessionAffinity: value.sessionAffinity === true,
		tags: [...new Set(tags)].slice(0, CUSTOM_ENDPOINT_TAG_LIMITS.count),
	};
}

export function readEndpointLimits(): Readonly<Record<string, EndpointLimits>> {
	const path = limitsPath();
	if (cache?.path === path) return cache.endpoints;
	const endpoints: Record<string, EndpointLimits> = {};
	let unreadable = false;
	try {
		const parsed: unknown = JSON.parse(readFileSync(path, "utf8"));
		if (isRecord(parsed) && isRecord(parsed.endpoints))
			for (const [id, value] of Object.entries(parsed.endpoints)) endpoints[id] = sanitize(value);
	} catch (error) {
		unreadable = !(
			error instanceof Error &&
			"code" in error &&
			(error.code === "ENOENT" || error.code === "ENOTDIR")
		);
	}
	cache = { path, endpoints, unreadable };
	return endpoints;
}

export function endpointLimitsOf(id: string): CustomEndpointLimits {
	const saved = readEndpointLimits()[id];
	return {
		maxConcurrency: saved?.maxConcurrency ?? null,
		timeoutSeconds: saved?.timeoutSeconds ?? null,
		maxRetries: saved?.maxRetries ?? null,
	};
}

export function endpointGatewayOf(id: string): CustomEndpointGateway | null {
	return readEndpointLimits()[id]?.gateway ?? null;
}

/** The sidecar entry a save leaves: `undefined` parts keep what is saved, so `undefined` overall means no change. */
export function nextEndpointSettings(
	id: string,
	limits: CustomEndpointLimits | undefined,
	gateway: CustomEndpointGateway | null | undefined,
): EndpointLimits | null | undefined {
	if (limits === undefined && gateway === undefined) return undefined;
	const saved = readEndpointLimits()[id] ?? {};
	const { gateway: savedGateway, ...savedLimits } = saved;
	const next = sanitize({
		...(limits === undefined ? savedLimits : limits),
		gateway: gateway === undefined ? savedGateway : gateway,
	});
	return Object.keys(next).length > 0 ? next : null;
}

export function writeEndpointLimits(id: string, limits: EndpointLimits | null): void {
	const endpoints = { ...readEndpointLimits() };
	if (cache?.unreadable)
		throw new Error(
			"ThinkRail's local-models.json isn't valid JSON. Fix or remove the file, then retry.",
		);
	if (limits) endpoints[id] = sanitize(limits);
	else delete endpoints[id];
	const path = limitsPath();
	const temp = `${path}.${process.pid}.tmp`;
	try {
		mkdirSync(dirname(path), { recursive: true, mode: 0o700 });
		writeFileSync(temp, `${JSON.stringify({ version: 1, endpoints }, null, 2)}\n`, { mode: 0o600 });
		renameSync(temp, path);
	} catch {
		try {
			rmSync(temp, { force: true });
		} catch {}
		throw new Error(
			"Couldn't save the server's request limits. Check ThinkRail's data folder and retry.",
		);
	}
	cache = { path, endpoints, unreadable: false };
}
