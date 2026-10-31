import { mkdirSync, readFileSync, renameSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import {
	type CustomEndpointLimits,
	isEndpointConcurrency,
	isEndpointRetries,
	isEndpointTimeoutSeconds,
} from "@thinkrail/contracts";
import { dataDir } from "../persistence";

export interface EndpointLimits {
	maxConcurrency?: number;
	timeoutSeconds?: number;
	maxRetries?: number;
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
	return limits;
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

export function toEndpointLimits(limits: CustomEndpointLimits): EndpointLimits | null {
	const next = sanitize(limits);
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
