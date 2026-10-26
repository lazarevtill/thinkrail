import { mkdirSync, readFileSync, renameSync, rmSync, statSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import {
	isLocalConcurrency,
	isLocalRetries,
	isLocalTimeoutSeconds,
	LOCAL_MODEL_SERVER_KINDS,
	type LocalModelServerKind,
} from "@thinkrail/contracts";
import { dataDir } from "../persistence";

/** What ThinkRail keeps about a local endpoint beyond what pi's models.json can hold. */
export interface LocalEndpointSettings {
	kind?: LocalModelServerKind;
	maxConcurrency?: number;
	timeoutSeconds?: number;
	maxRetries?: number;
}

type SettingsFile = { version: 1; endpoints: Record<string, LocalEndpointSettings> };

let cache: {
	path: string;
	mtimeMs: number;
	endpoints: Record<string, LocalEndpointSettings>;
} | null = null;

function settingsPath(): string {
	return join(dataDir(), "local-models.json");
}

function sanitize(value: unknown): LocalEndpointSettings {
	if (typeof value !== "object" || value === null) return {};
	const raw = value as Record<string, unknown>;
	const settings: LocalEndpointSettings = {};
	if (LOCAL_MODEL_SERVER_KINDS.includes(raw.kind as LocalModelServerKind))
		settings.kind = raw.kind as LocalModelServerKind;
	if (isLocalConcurrency(raw.maxConcurrency)) settings.maxConcurrency = raw.maxConcurrency;
	if (isLocalTimeoutSeconds(raw.timeoutSeconds)) settings.timeoutSeconds = raw.timeoutSeconds;
	if (isLocalRetries(raw.maxRetries)) settings.maxRetries = raw.maxRetries;
	return settings;
}

export function readLocalEndpointSettings(): Readonly<Record<string, LocalEndpointSettings>> {
	const path = settingsPath();
	let mtimeMs: number;
	try {
		mtimeMs = statSync(path).mtimeMs;
	} catch {
		cache = { path, mtimeMs: -1, endpoints: {} };
		return cache.endpoints;
	}
	if (cache?.path === path && cache.mtimeMs === mtimeMs) return cache.endpoints;
	const endpoints: Record<string, LocalEndpointSettings> = {};
	try {
		const parsed = JSON.parse(readFileSync(path, "utf8")) as Partial<SettingsFile>;
		if (parsed && typeof parsed.endpoints === "object" && parsed.endpoints !== null) {
			for (const [id, value] of Object.entries(parsed.endpoints)) endpoints[id] = sanitize(value);
		}
	} catch {}
	cache = { path, mtimeMs, endpoints };
	return endpoints;
}

export function writeLocalEndpointSettings(
	id: string,
	settings: LocalEndpointSettings | null,
): void {
	const endpoints = { ...readLocalEndpointSettings() };
	if (settings) endpoints[id] = sanitize(settings);
	else delete endpoints[id];
	const path = settingsPath();
	const file: SettingsFile = { version: 1, endpoints };
	mkdirSync(dirname(path), { recursive: true, mode: 0o700 });
	const temp = `${path}.${process.pid}.tmp`;
	try {
		writeFileSync(temp, `${JSON.stringify(file, null, 2)}\n`, { mode: 0o600 });
		renameSync(temp, path);
	} catch {
		rmSync(temp, { force: true });
		throw new Error("Couldn't save local model settings. Check ThinkRail's data folder and retry.");
	}
	cache = null;
}
