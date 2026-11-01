import { createHash } from "node:crypto";
import { readFileSync, statSync } from "node:fs";
import { getCACertificates } from "node:tls";
import type { CustomEndpointNetwork } from "@thinkrail/contracts";

export type Fetch = typeof globalThis.fetch;

/** How a probe reaches a server: its resolved key and headers and a fetch with its network settings. */
export interface Access {
	key: string | undefined;
	headers: Record<string, string>;
	fetch: Fetch;
}

export function requestHeaders(
	access: Access,
	extra: Record<string, string> = {},
): Record<string, string> {
	const headers: Record<string, string> = { ...access.headers, ...extra };
	if (!access.key) return headers;
	for (const name of Object.keys(headers))
		if (name.toLowerCase() === "authorization") delete headers[name];
	headers.authorization = `Bearer ${access.key}`;
	return headers;
}

export function sendsCredentials(access: Access): boolean {
	return access.key !== undefined || Object.keys(access.headers).length > 0;
}

const COMMAND_TIMEOUT_MS = 15_000;
const COMMAND_OUTPUT_LIMIT = 64 * 1024;
const DEFAULT_TOKEN_TTL_MS = 5 * 60 * 1000;
const TOKEN_EXPIRY_MARGIN_MS = 60 * 1000;

const ENV_TEMPLATE = /\$\$|\$!|\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)/g;
const WHOLE_REFERENCE = /^\$(?:\{[A-Za-z_][A-Za-z0-9_]*\}|[A-Za-z_][A-Za-z0-9_]*)$/;

/**
 * pi's `$VAR` / `${VAR}` template rules for a config value: `$$` is a literal `$` and `$!` a literal
 * `!`; a missing variable makes the whole value unresolved. `!command` values aren't templates.
 */
export function resolveTemplate(value: string): string | undefined {
	if (value.startsWith("!")) return undefined;
	let missing = false;
	const resolved = value.replace(ENV_TEMPLATE, (match, braced?: string, bare?: string) => {
		if (match === "$$") return "$";
		if (match === "$!") return "!";
		const found = process.env[braced ?? bare ?? ""];
		if (!found) missing = true;
		return found ?? "";
	});
	return missing ? undefined : resolved;
}

/**
 * A value that names where a secret comes from rather than holding one: a `!command` or exactly one
 * `$VAR` / `${VAR}`. Anything with literal parts may carry a secret, so it is never sent to a client.
 */
export function isReference(value: string): boolean {
	return value.startsWith("!") || WHOLE_REFERENCE.test(value);
}

export function commandHash(command: string): string {
	return createHash("sha256").update(command).digest("hex");
}

function shellFor(command: string): string[] {
	return process.platform === "win32"
		? [process.env.ComSpec || "cmd.exe", "/d", "/s", "/c", `"${command}"`]
		: ["/bin/sh", "-c", command];
}

/** Runs a `!command` key the way pi does (its trimmed stdout), but asynchronously and bounded. */
export async function runKeyCommand(command: string): Promise<string | undefined> {
	const body = command.startsWith("!") ? command.slice(1) : command;
	if (!body.trim()) return undefined;
	let timer: ReturnType<typeof setTimeout> | undefined;
	try {
		const child = Bun.spawn(shellFor(body), {
			stdin: "ignore",
			stdout: "pipe",
			stderr: "ignore",
			windowsHide: true,
			windowsVerbatimArguments: true,
		});
		const finished = Promise.all([new Response(child.stdout).text(), child.exited]);
		const timedOut = new Promise<null>((resolve) => {
			timer = setTimeout(() => resolve(null), COMMAND_TIMEOUT_MS);
		});
		const result = await Promise.race([finished, timedOut]);
		if (result === null) {
			child.kill();
			return undefined;
		}
		const [output, code] = result;
		if (code !== 0 || output.length > COMMAND_OUTPUT_LIMIT) return undefined;
		return output.trim() || undefined;
	} catch {
		return undefined;
	} finally {
		clearTimeout(timer);
	}
}

/** The `exp` claim of a JWT access token, in ms; tokens from az, gcloud and most SSO tools are JWTs. */
export function jwtExpiry(token: string): number | null {
	const payload = token.split(".")[1];
	if (!payload || token.split(".").length !== 3) return null;
	try {
		const claims: unknown = JSON.parse(Buffer.from(payload, "base64url").toString("utf8"));
		if (typeof claims !== "object" || claims === null || !("exp" in claims)) return null;
		return typeof claims.exp === "number" && Number.isFinite(claims.exp) ? claims.exp * 1000 : null;
	} catch {
		return null;
	}
}

const tokens = new Map<string, { value: string; expiresAt: number }>();
const pending = new Map<string, Promise<string | undefined>>();

/**
 * A command key's token, cached until shortly before its JWT expiry (or for five minutes), so a
 * short-lived SSO token is fetched once per lifetime instead of once per request as pi would.
 */
export function commandToken(
	command: string,
	now: () => number = Date.now,
): Promise<string | undefined> {
	const cached = tokens.get(command);
	if (cached && cached.expiresAt > now()) return Promise.resolve(cached.value);
	const running = pending.get(command);
	if (running) return running;
	const next = runKeyCommand(command)
		.then((value) => {
			if (value === undefined) {
				tokens.delete(command);
				return undefined;
			}
			const expiry = jwtExpiry(value);
			tokens.set(command, {
				value,
				expiresAt: expiry === null ? now() + DEFAULT_TOKEN_TTL_MS : expiry - TOKEN_EXPIRY_MARGIN_MS,
			});
			return value;
		})
		.finally(() => pending.delete(command));
	pending.set(command, next);
	return next;
}

export function forgetCommandToken(command: string): void {
	tokens.delete(command);
}

/**
 * Resolves a key or header value. A `!command` runs only when `allowCommand` (a user action, or a
 * command the user saved through the form or the organization shipped).
 */
export async function resolveValue(
	value: string | undefined,
	allowCommand: boolean,
): Promise<string | undefined> {
	if (!value?.trim()) return undefined;
	if (value.startsWith("!")) return allowCommand ? commandToken(value) : undefined;
	return resolveTemplate(value.trim());
}

const caFiles = new Map<string, { mtimeMs: number; pem: string }>();

function caFile(path: string): string {
	let mtimeMs: number;
	try {
		mtimeMs = statSync(path).mtimeMs;
	} catch {
		throw new Error(`The certificate file ${path} can't be read.`);
	}
	const cached = caFiles.get(path);
	if (cached?.mtimeMs === mtimeMs) return cached.pem;
	const pem = readFileSync(path, "utf8");
	if (!pem.includes("-----BEGIN CERTIFICATE-----"))
		throw new Error(`The certificate file ${path} has no PEM certificate.`);
	caFiles.set(path, { mtimeMs, pem });
	return pem;
}

/** Bun fetch options for a server's network settings; empty when the defaults apply. */
export function networkInit(network: CustomEndpointNetwork): {
	tls?: { ca: string[] };
	proxy?: string;
} {
	const init: { tls?: { ca: string[] }; proxy?: string } = {};
	if (network.systemCertificates || network.caFile) {
		const ca = [...getCACertificates("bundled")];
		if (network.systemCertificates) ca.push(...getCACertificates("system"));
		if (network.caFile) ca.push(caFile(network.caFile));
		init.tls = { ca };
	}
	if (network.proxy) init.proxy = network.proxy;
	return init;
}

export function hasNetworkSettings(network: CustomEndpointNetwork): boolean {
	return network.systemCertificates || network.caFile !== null || network.proxy !== null;
}

/** A fetch that reaches the server with its certificate and proxy settings. */
export function networkFetch(
	network: CustomEndpointNetwork,
	base: Fetch = globalThis.fetch,
): Fetch {
	if (!hasNetworkSettings(network)) return base;
	return Object.assign(
		(input: Parameters<Fetch>[0], init?: Parameters<Fetch>[1]) =>
			base(input, { ...init, ...networkInit(network) }),
		{ preconnect: base.preconnect },
	);
}
