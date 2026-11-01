import { existsSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import {
	CUSTOM_ENDPOINT_ID_PREFIX,
	type CustomEndpointGateway,
	type CustomEndpointHeader,
	type CustomEndpointLimits,
	type CustomEndpointManaged,
	type CustomEndpointModel,
	type CustomEndpointNetwork,
	isCustomEndpointApi,
	isEndpointHeaderName,
	isEndpointHeaderValue,
	type OrganizationPolicy,
} from "@thinkrail/contracts";
import {
	readEndpointLimits,
	sanitizeGateway,
	sanitizeLimits,
	sanitizeNetwork,
} from "./endpointLimits";

/** The id prefix of servers the organization ships; the rest of the id comes from the managed file. */
export const MANAGED_ID_PREFIX = `${CUSTOM_ENDPOINT_ID_PREFIX}org-`;

export interface ManagedServer {
	id: string;
	name: string;
	baseUrl: string;
	/** Absent: each user enters their own key. */
	apiKey: string | undefined;
	headers: CustomEndpointHeader[];
	/** Absent: each user detects and picks the models. */
	models: CustomEndpointModel[] | undefined;
	limits: CustomEndpointLimits;
	gateway: CustomEndpointGateway | null;
	network: CustomEndpointNetwork | null;
}

export interface ManagedConfig {
	path: string | null;
	servers: ManagedServer[];
	restrictModels: OrganizationPolicy["restrictModels"];
	error: string | null;
	/**
	 * The file exists but couldn't be read or applied in full, so nothing synced from it is removed and
	 * the servers synced earlier still count as the organization's.
	 */
	degraded: boolean;
}

const EMPTY: ManagedConfig = {
	path: null,
	servers: [],
	restrictModels: null,
	error: null,
	degraded: false,
};

/**
 * The organization's managed file: the machine-wide path only an administrator can write
 * (`%ProgramData%\ThinkRail`, `/Library/Application Support/ThinkRail`, `/etc/thinkrail`). Only when
 * that file doesn't exist does `THINKRAIL_MANAGED_CONFIG` name another one, so a user's own
 * environment can't switch an organization's policy off.
 */
export function machineConfigPath(): string {
	if (process.platform === "win32")
		return join(process.env.ProgramData || "C:\\ProgramData", "ThinkRail", "managed.json");
	if (process.platform === "darwin") return "/Library/Application Support/ThinkRail/managed.json";
	return "/etc/thinkrail/managed.json";
}

export function managedConfigPath(): string {
	const machine = machineConfigPath();
	if (existsSync(machine)) return machine;
	return process.env.THINKRAIL_MANAGED_CONFIG?.trim() || machine;
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

function slug(text: string): string {
	return text
		.toLowerCase()
		.replace(/[^a-z0-9]+/g, "-")
		.replace(/^-+|-+$/g, "")
		.slice(0, 40);
}

function count(value: unknown): number | null {
	return typeof value === "number" && Number.isSafeInteger(value) && value > 0 ? value : null;
}

function flag(value: unknown): boolean | null {
	return typeof value === "boolean" ? value : null;
}

function price(value: unknown): number {
	return typeof value === "number" && Number.isFinite(value) && value >= 0 ? value : 0;
}

function parseModel(value: unknown): CustomEndpointModel | null {
	const entry = typeof value === "string" ? { id: value } : value;
	if (!isRecord(entry) || typeof entry.id !== "string" || !entry.id.trim()) return null;
	const cost = isRecord(entry.cost)
		? {
				input: price(entry.cost.input),
				output: price(entry.cost.output),
				cacheRead: price(entry.cost.cacheRead),
				cacheWrite: price(entry.cost.cacheWrite),
			}
		: null;
	return {
		id: entry.id.trim(),
		contextWindow: count(entry.contextWindow),
		maxTokens: count(entry.maxTokens),
		reasoning: flag(entry.reasoning),
		vision: flag(entry.vision),
		tools: null,
		chatTemplateThinking: flag(entry.chatTemplateThinking),
		...(cost && (cost.input > 0 || cost.output > 0) ? { cost } : {}),
		...(isCustomEndpointApi(entry.api) ? { api: entry.api } : {}),
	};
}

function parseHeaders(value: unknown): CustomEndpointHeader[] | string {
	if (value === undefined) return [];
	if (!isRecord(value)) return "headers must be an object of name to value";
	const headers: CustomEndpointHeader[] = [];
	for (const [name, header] of Object.entries(value)) {
		if (!isEndpointHeaderName(name) || !isEndpointHeaderValue(header))
			return `header ${name} has an invalid name or value`;
		if (header.startsWith("!"))
			return `header ${name} can't be a command; use the apiKey or an environment variable`;
		headers.push({ name, value: header });
	}
	return headers;
}

function parseServer(value: unknown, index: number): ManagedServer | string {
	if (!isRecord(value)) return `servers[${index}] is not an object`;
	const key = typeof value.id === "string" ? slug(value.id) : "";
	if (!key) return `servers[${index}] needs an id of letters and digits`;
	if (typeof value.name !== "string" || !value.name.trim()) return `server ${key} needs a name`;
	if (typeof value.baseUrl !== "string" || !/^https?:\/\//.test(value.baseUrl))
		return `server ${key} needs an http(s) baseUrl`;
	if (value.apiKey !== undefined && (typeof value.apiKey !== "string" || !value.apiKey.trim()))
		return `server ${key} has an empty apiKey`;
	const headers = parseHeaders(value.headers);
	if (typeof headers === "string") return `server ${key}: ${headers}`;
	let models: CustomEndpointModel[] | undefined;
	if (value.models !== undefined) {
		if (!Array.isArray(value.models)) return `server ${key}: models must be a list`;
		models = value.models.flatMap((model) => parseModel(model) ?? []);
		if (models.length === 0) return `server ${key} lists no valid models`;
	}
	const limits = sanitizeLimits(value.limits);
	return {
		id: `${MANAGED_ID_PREFIX}${key}`,
		name: value.name.trim(),
		baseUrl: value.baseUrl.trim(),
		apiKey: typeof value.apiKey === "string" ? value.apiKey.trim() : undefined,
		headers,
		models,
		limits: {
			maxConcurrency: limits.maxConcurrency ?? null,
			timeoutSeconds: limits.timeoutSeconds ?? null,
			maxRetries: limits.maxRetries ?? null,
		},
		gateway: sanitizeGateway(value.gateway),
		network: sanitizeNetwork(value.network),
	};
}

export function parseManagedConfig(raw: unknown, path: string | null): ManagedConfig {
	if (!isRecord(raw))
		return {
			...EMPTY,
			path,
			restrictModels: "managed",
			error: "the file must hold a JSON object",
			degraded: true,
		};
	const errors: string[] = [];
	const servers: ManagedServer[] = [];
	const seen = new Set<string>();
	for (const [index, value] of (Array.isArray(raw.servers) ? raw.servers : []).entries()) {
		const server = parseServer(value, index);
		if (typeof server === "string") errors.push(server);
		else if (seen.has(server.id)) errors.push(`server id ${server.id} appears twice`);
		else {
			seen.add(server.id);
			servers.push(server);
		}
	}
	const policy = isRecord(raw.policy) ? raw.policy : {};
	const restrict = policy.restrictModels;
	if (
		restrict !== undefined &&
		restrict !== null &&
		restrict !== "managed" &&
		restrict !== "custom"
	)
		errors.push('policy.restrictModels must be "managed", "custom" or null');
	return {
		path,
		servers,
		restrictModels:
			restrict === "custom"
				? "custom"
				: restrict === undefined || restrict === null
					? null
					: "managed",
		error: errors.length > 0 ? errors.join("; ") : null,
		degraded: errors.length > 0,
	};
}

let cache: { path: string; mtimeMs: number; config: ManagedConfig } | null = null;

export function readManagedConfig(): ManagedConfig {
	const path = managedConfigPath();
	let mtimeMs: number;
	try {
		mtimeMs = statSync(path).mtimeMs;
	} catch {
		cache = null;
		return EMPTY;
	}
	if (cache?.path === path && cache.mtimeMs === mtimeMs) return cache.config;
	let config: ManagedConfig;
	try {
		config = parseManagedConfig(JSON.parse(readFileSync(path, "utf8")), path);
	} catch {
		const previous = cache?.path === path ? cache.config : null;
		config = {
			...(previous ?? EMPTY),
			path,
			restrictModels: previous ? previous.restrictModels : "managed",
			error: previous
				? "the file can't be read or isn't valid JSON, so the previous settings stay in force"
				: "the file can't be read or isn't valid JSON, so only the servers it set up before stay usable",
			degraded: true,
		};
	}
	cache = { path, mtimeMs, config };
	return config;
}

export function managedServer(id: string): ManagedServer | undefined {
	return readManagedConfig().servers.find((server) => server.id === id);
}

export function managedInfo(id: string): CustomEndpointManaged | null {
	const server = managedServer(id);
	return server ? { key: server.apiKey !== undefined, models: server.models !== undefined } : null;
}

export function organizationPolicy(): OrganizationPolicy {
	const config = readManagedConfig();
	return {
		restrictModels: config.restrictModels,
		managedFile: config.path,
		managedServers: config.servers.length,
		error: config.error,
	};
}

/** Whether the organization's policy lets a model of this provider be used. */
export function providerAllowed(providerId: string): boolean {
	const config = readManagedConfig();
	if (config.restrictModels === null) return true;
	if (config.restrictModels === "custom") return providerId.startsWith(CUSTOM_ENDPOINT_ID_PREFIX);
	if (config.servers.some((server) => server.id === providerId)) return true;
	return config.degraded && readEndpointLimits()[providerId]?.managed === true;
}

export function policyRefusal(providerId: string): string {
	return `Your organization's policy allows only its own model servers, and "${providerId}" isn't one of them. Pick one of its servers in the model picker.`;
}

/** The origin a managed server's requests must go to while a policy is in force. */
export function managedOrigin(providerId: string): string | null {
	const server = managedServer(providerId);
	if (!server) return null;
	try {
		return new URL(server.baseUrl).origin;
	} catch {
		return null;
	}
}
