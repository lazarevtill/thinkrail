import {
	RiPlayLine as Play,
	RiAddLine as Plus,
	RiSearchLine as Search,
	RiDeleteBin6Line as Trash2,
} from "@remixicon/react";
import {
	CUSTOM_ENDPOINT_CONCURRENCY_LIMITS,
	CUSTOM_ENDPOINT_GATEWAY_PROTOCOL_VERSION,
	CUSTOM_ENDPOINT_LIMITS_PROTOCOL_VERSION,
	CUSTOM_ENDPOINT_RETRY_LIMITS,
	CUSTOM_ENDPOINT_TAG_LIMITS,
	CUSTOM_ENDPOINT_TIMEOUT_LIMITS,
	CUSTOM_ENDPOINTS_PROTOCOL_VERSION,
	type CustomEndpoint,
	type CustomEndpointKeyStatus,
	type CustomEndpointModel,
	type CustomEndpointTestResult,
	customEndpointTestBudgetMs,
} from "@thinkrail/contracts";
import { localeOf, type Translator, useT } from "@thinkrail/locales/react";
import { Button, IconButton } from "@thinkrail/ui/button";
import {
	Dialog,
	DialogContent,
	DialogDescription,
	DialogFooter,
	DialogHeader,
	DialogTitle,
} from "@thinkrail/ui/dialog";
import { cn } from "@thinkrail/ui/utils";
import { type ReactNode, useCallback, useEffect, useId, useRef, useState } from "react";
import { formatContext } from "@/chat/modelPicker";
import { toast, useAppStore } from "@/store";
import { errorText, getTransport } from "@/transport";
import { ConfirmDialog } from "./ConfirmDialog";
import {
	addManualModel,
	addressMoved,
	applyPreset,
	canSave,
	draftParams,
	type EndpointDraft,
	editEndpointDraft,
	type GatewayDraft,
	keyAlerts,
	LOCAL_SERVER_PRESETS,
	limitIssues,
	mergeProbe,
	newEndpointDraft,
	probeFailureText,
	probeTarget,
} from "./customEndpointDraft";
import { SettingsGroup } from "./SettingsGroup";
import { SettingsSwitch } from "./SettingsSwitch";

const INPUT_CLASS =
	"h-32 w-full min-w-0 rounded-sm border border-control-border-default bg-control-bg px-8 tr-text-ui text-text-default outline-none transition-colors placeholder:text-text-muted focus:border-control-border-active disabled:border-control-disabled-border disabled:bg-control-disabled-bg disabled:text-control-disabled-text";

type Status = { tone: "ok" | "error"; text: string } | null;

const TEST_REPLY_MARGIN_MS = 15_000;

function supported(protocolVersion: number | null): boolean {
	return protocolVersion !== null && protocolVersion >= CUSTOM_ENDPOINTS_PROTOCOL_VERSION;
}

function useGatewaySupported(): boolean {
	const protocolVersion = useAppStore((state) => state.protocolVersion);
	return protocolVersion !== null && protocolVersion >= CUSTOM_ENDPOINT_GATEWAY_PROTOCOL_VERSION;
}

function usd(t: Translator, value: number): string {
	return new Intl.NumberFormat(localeOf(t), {
		style: "currency",
		currency: "USD",
		minimumFractionDigits: 0,
		maximumFractionDigits: value < 1 ? 4 : 2,
	}).format(value);
}

function day(t: Translator, iso: string): string | null {
	const time = Date.parse(iso);
	return Number.isFinite(time)
		? new Date(time).toLocaleDateString(localeOf(t), { month: "short", day: "numeric" })
		: null;
}

function keyStatusText(t: Translator, status: CustomEndpointKeyStatus, now: number): string {
	const alerts = keyAlerts(status, now);
	const reset = status.budgetResetAt === null ? null : day(t, status.budgetResetAt);
	const expiry = status.expiresAt === null ? null : day(t, status.expiresAt);
	const parts = [
		status.alias === null
			? t("settings.customEndpoints.key.title")
			: t("settings.customEndpoints.key.titleAlias", { alias: status.alias }),
		status.spend === null
			? null
			: status.maxBudget === null
				? t("settings.customEndpoints.key.spend", { spend: usd(t, status.spend) })
				: t("settings.customEndpoints.key.spendOf", {
						spend: usd(t, status.spend),
						budget: usd(t, status.maxBudget),
					}),
		alerts.budget === "exhausted" ? t("settings.customEndpoints.key.budgetSpent") : null,
		reset === null ? null : t("settings.customEndpoints.key.resets", { date: reset }),
		status.rpmLimit === null
			? null
			: t("settings.customEndpoints.key.rpm", { count: status.rpmLimit }),
		status.tpmLimit === null
			? null
			: t("settings.customEndpoints.key.tpm", { count: status.tpmLimit }),
		status.maxParallelRequests === null
			? null
			: t("settings.customEndpoints.key.parallel", { count: status.maxParallelRequests }),
		expiry === null
			? null
			: alerts.expiry === "expired"
				? t("settings.customEndpoints.key.expired", { date: expiry })
				: t("settings.customEndpoints.key.expires", { date: expiry }),
	];
	return parts.filter((part): part is string => part !== null).join(" · ");
}

function KeyStatusLine({ status, testId }: { status: CustomEndpointKeyStatus; testId: string }) {
	const t = useT();
	const [now] = useState(() => Date.now());
	const alerts = keyAlerts(status, now);
	const warn = alerts.budget !== null || alerts.expiry !== null;
	return (
		<p
			data-testid={testId}
			data-tone={warn ? "warning" : "ok"}
			className={cn(
				"break-words tr-text-metadata",
				warn ? "text-feedback-warning" : "text-text-muted",
			)}
		>
			{keyStatusText(t, status, now)}
		</p>
	);
}

function EndpointKeyStatus({ endpoint }: { endpoint: CustomEndpoint }) {
	const [status, setStatus] = useState<CustomEndpointKeyStatus | null>(null);
	const gateway = endpoint.gateway?.kind ?? null;
	const { id, baseUrl, hasApiKey } = endpoint;
	useEffect(() => {
		setStatus(null);
		if (gateway !== "litellm" || !hasApiKey || !baseUrl) return;
		let live = true;
		void getTransport()
			.request("customEndpoints.keyStatus", { id })
			.then((next) => {
				if (live) setStatus(next);
			})
			.catch(() => {});
		return () => {
			live = false;
		};
	}, [id, baseUrl, hasApiKey, gateway]);
	return status ? <KeyStatusLine status={status} testId="custom-endpoints-key-status" /> : null;
}

function GatewayFields({
	gateway,
	keyStatus,
	invalid,
	disabled,
	onChange,
}: {
	gateway: GatewayDraft;
	keyStatus: CustomEndpointKeyStatus | null;
	invalid: boolean;
	disabled: boolean;
	onChange: (gateway: GatewayDraft) => void;
}) {
	const t = useT();
	const id = useId();
	return (
		<fieldset
			data-testid="custom-endpoints-gateway"
			className="m-0 flex min-w-0 flex-col gap-8 border-0 p-0"
		>
			<legend className="mb-4 tr-title-compact text-text-default">
				{t("settings.customEndpoints.gateway.title")}
			</legend>
			<p className="tr-text-metadata text-text-muted">
				{t("settings.customEndpoints.gateway.detected")}
			</p>
			{keyStatus ? <KeyStatusLine status={keyStatus} testId="custom-endpoints-probe-key" /> : null}
			<div className="flex items-center gap-8">
				<span className="flex min-w-0 flex-1 flex-col gap-2">
					<span className="tr-text-ui text-text-default">
						{t("settings.customEndpoints.gateway.session")}
					</span>
					<span className="tr-text-metadata text-text-muted">
						{t("settings.customEndpoints.gateway.sessionHint")}
					</span>
				</span>
				<SettingsSwitch
					checked={gateway.sessionAffinity}
					disabled={disabled}
					label={t("settings.customEndpoints.gateway.session")}
					testId="custom-endpoints-session"
					onChange={(sessionAffinity) => onChange({ ...gateway, sessionAffinity })}
				/>
			</div>
			<div className="flex flex-col gap-4">
				<label htmlFor={`${id}-tags`} className="tr-text-metadata text-text-muted">
					{t("settings.customEndpoints.gateway.tags")}
				</label>
				<input
					id={`${id}-tags`}
					data-testid="custom-endpoints-tags"
					className={cn(INPUT_CLASS, "tr-code-text", invalid && "border-feedback-error")}
					value={gateway.tagsText}
					disabled={disabled}
					spellCheck={false}
					aria-invalid={invalid}
					onChange={(event) => onChange({ ...gateway, tagsText: event.currentTarget.value })}
				/>
				<p className="tr-text-metadata text-text-muted">
					{t("settings.customEndpoints.gateway.tagsHint")}
				</p>
				{invalid ? (
					<p className="tr-text-metadata text-feedback-error">
						{t("settings.customEndpoints.gateway.tagsInvalid", CUSTOM_ENDPOINT_TAG_LIMITS)}
					</p>
				) : null}
			</div>
		</fieldset>
	);
}

function useLimitsSupported(): boolean {
	const protocolVersion = useAppStore((state) => state.protocolVersion);
	return protocolVersion !== null && protocolVersion >= CUSTOM_ENDPOINT_LIMITS_PROTOCOL_VERSION;
}

function LimitInput({
	id,
	label,
	testId,
	value,
	invalid,
	disabled,
	placeholder,
	onChange,
}: {
	id: string;
	label: string;
	testId: string;
	value: string;
	invalid: boolean;
	disabled: boolean;
	placeholder: string;
	onChange: (value: string) => void;
}) {
	return (
		<div className="flex w-160 flex-col gap-4">
			<label htmlFor={id} className="tr-text-metadata text-text-muted">
				{label}
			</label>
			<input
				id={id}
				data-testid={testId}
				inputMode="numeric"
				className={cn(INPUT_CLASS, invalid && "border-feedback-error")}
				value={value}
				disabled={disabled}
				aria-invalid={invalid}
				placeholder={placeholder}
				onChange={(event) => onChange(event.currentTarget.value)}
			/>
		</div>
	);
}

function LimitFields({
	draft,
	disabled,
	onChange,
}: {
	draft: EndpointDraft;
	disabled: boolean;
	onChange: (draft: EndpointDraft) => void;
}) {
	const t = useT();
	const id = useId();
	const issues = limitIssues(draft);
	return (
		<fieldset className="m-0 flex min-w-0 flex-col gap-8 border-0 p-0">
			<legend className="mb-4 tr-title-compact text-text-default">
				{t("settings.customEndpoints.limits")}
			</legend>
			<div className="flex flex-wrap gap-12">
				<LimitInput
					id={`${id}-parallel`}
					label={t("settings.customEndpoints.parallel")}
					testId="custom-endpoints-parallel"
					value={draft.concurrencyText}
					invalid={issues.concurrency}
					disabled={disabled}
					placeholder={t("settings.customEndpoints.parallelNoLimit")}
					onChange={(concurrencyText) => onChange({ ...draft, concurrencyText })}
				/>
				<LimitInput
					id={`${id}-timeout`}
					label={t("settings.customEndpoints.timeout")}
					testId="custom-endpoints-timeout"
					value={draft.timeoutText}
					invalid={issues.timeout}
					disabled={disabled}
					placeholder={t("settings.customEndpoints.piDefault")}
					onChange={(timeoutText) => onChange({ ...draft, timeoutText })}
				/>
				<LimitInput
					id={`${id}-retries`}
					label={t("settings.customEndpoints.retries")}
					testId="custom-endpoints-retries"
					value={draft.retriesText}
					invalid={issues.retries}
					disabled={disabled}
					placeholder={t("settings.customEndpoints.piDefault")}
					onChange={(retriesText) => onChange({ ...draft, retriesText })}
				/>
			</div>
			<p className="tr-text-metadata text-text-muted">
				{t("settings.customEndpoints.parallelHint")}
			</p>
			<p className="tr-text-metadata text-text-muted">
				{t("settings.customEndpoints.timeoutHint")}
			</p>
			{issues.concurrency ? (
				<p className="tr-text-metadata text-feedback-error">
					{t("settings.customEndpoints.parallelInvalid", CUSTOM_ENDPOINT_CONCURRENCY_LIMITS)}
				</p>
			) : null}
			{issues.timeout ? (
				<p className="tr-text-metadata text-feedback-error">
					{t("settings.customEndpoints.timeoutInvalid", CUSTOM_ENDPOINT_TIMEOUT_LIMITS)}
				</p>
			) : null}
			{issues.retries ? (
				<p className="tr-text-metadata text-feedback-error">
					{t("settings.customEndpoints.retriesInvalid", CUSTOM_ENDPOINT_RETRY_LIMITS)}
				</p>
			) : null}
		</fieldset>
	);
}

function Field({
	label,
	htmlFor,
	children,
}: {
	label: string;
	htmlFor: string;
	children: ReactNode;
}) {
	return (
		<div className="flex flex-col gap-4">
			<label htmlFor={htmlFor} className="tr-title-compact text-text-default">
				{label}
			</label>
			{children}
		</div>
	);
}

function ModelFacts({ model }: { model: CustomEndpointModel }) {
	const t = useT();
	const facts = [
		model.contextWindow === null
			? null
			: t("settings.customEndpoints.factContext", { tokens: formatContext(model.contextWindow) }),
		model.reasoning ? t("settings.customEndpoints.factThinking") : null,
		model.vision ? t("settings.customEndpoints.factImages") : null,
		model.tools === true ? t("settings.customEndpoints.factTools") : null,
		model.cost
			? t("settings.customEndpoints.factPrice", {
					input: usd(t, model.cost.input),
					output: usd(t, model.cost.output),
				})
			: null,
	].filter((fact): fact is string => fact !== null);
	if (facts.length === 0 && model.tools !== false) return null;
	return (
		<span className="truncate tr-text-metadata text-text-muted">
			{facts.join(" · ")}
			{model.tools === false ? (
				<span data-testid="custom-endpoints-no-tools" className="text-feedback-warning">
					{facts.length > 0 ? " · " : ""}
					{t("settings.customEndpoints.factNoTools")}
				</span>
			) : null}
		</span>
	);
}

function EndpointDialog({
	initial,
	onClose,
	onSaved,
}: {
	initial: EndpointDraft;
	onClose: () => void;
	onSaved: (endpoints: CustomEndpoint[]) => void;
}) {
	const t = useT();
	const id = useId();
	const [draft, setDraft] = useState(initial);
	const [manualId, setManualId] = useState("");
	const [probing, setProbing] = useState(false);
	const [saving, setSaving] = useState(false);
	const [status, setStatus] = useState<Status>(null);
	const busy = probing || saving;
	const autoDetected = useRef(false);
	const limitsSupported = useLimitsSupported();
	const gatewaySupported = useGatewaySupported();
	const protocolVersion = useAppStore((state) => state.protocolVersion);
	const [probeKey, setProbeKey] = useState<CustomEndpointKeyStatus | null>(null);

	const detect = useCallback(
		(current: EndpointDraft) => {
			setProbing(true);
			setStatus(null);
			return getTransport()
				.request("customEndpoints.probe", { baseUrl: current.baseUrl, ...probeTarget(current) })
				.then((probe) => {
					if (!probe.ok) {
						setStatus({ tone: "error", text: probeFailureText(t, probe) });
						return;
					}
					setDraft((latest) => mergeProbe(latest, probe));
					setProbeKey(probe.gateway?.key ?? null);
					setStatus({
						tone: "ok",
						text: t("settings.customEndpoints.found", { count: probe.models.length }),
					});
				})
				.catch((error: unknown) => setStatus({ tone: "error", text: errorText(error) }))
				.finally(() => setProbing(false));
		},
		[t],
	);

	useEffect(() => {
		if (autoDetected.current || !initial.id) return;
		autoDetected.current = true;
		void detect(initial);
	}, [detect, initial]);

	const save = () => {
		if (!canSave(draft)) return;
		setSaving(true);
		setStatus(null);
		void getTransport()
			.request("customEndpoints.save", draftParams(draft, protocolVersion))
			.then((endpoints) => {
				onSaved(endpoints);
				onClose();
			})
			.catch((error: unknown) => setStatus({ tone: "error", text: errorText(error) }))
			.finally(() => setSaving(false));
	};

	const addManual = () => {
		setDraft((current) => addManualModel(current, manualId));
		setManualId("");
	};

	return (
		<Dialog open onOpenChange={(open) => !open && !busy && onClose()}>
			<DialogContent
				data-testid="custom-endpoints-dialog"
				className="flex max-h-[85vh] max-w-[36rem] flex-col"
			>
				<DialogHeader>
					<DialogTitle>
						{draft.id
							? t("settings.customEndpoints.editTitle")
							: t("settings.customEndpoints.addTitle")}
					</DialogTitle>
					<DialogDescription>{t("settings.customEndpoints.dialogDescription")}</DialogDescription>
				</DialogHeader>
				<div className="-mx-16 flex min-h-0 flex-1 flex-col gap-12 overflow-y-auto px-16">
					<Field label={t("settings.customEndpoints.name")} htmlFor={`${id}-name`}>
						<input
							id={`${id}-name`}
							data-testid="custom-endpoints-name"
							className={INPUT_CLASS}
							value={draft.name}
							disabled={busy}
							onChange={(event) => setDraft({ ...draft, name: event.currentTarget.value })}
						/>
					</Field>
					<Field label={t("settings.customEndpoints.url")} htmlFor={`${id}-url`}>
						<input
							id={`${id}-url`}
							data-testid="custom-endpoints-url"
							className={cn(INPUT_CLASS, "tr-code-text")}
							value={draft.baseUrl}
							disabled={busy}
							spellCheck={false}
							placeholder={t("settings.customEndpoints.urlPlaceholder")}
							onChange={(event) => setDraft({ ...draft, baseUrl: event.currentTarget.value })}
						/>
						{addressMoved(draft) ? (
							<p
								data-testid="custom-endpoints-address-moved"
								className="tr-text-metadata text-feedback-warning"
							>
								{t("settings.customEndpoints.addressMoved")}
							</p>
						) : null}
						<div className="flex flex-wrap items-center gap-4">
							<span className="tr-text-metadata text-text-muted">
								{t("settings.customEndpoints.local")}
							</span>
							{LOCAL_SERVER_PRESETS.map((preset) => (
								<Button
									key={preset.name}
									variant="outline"
									size="s"
									data-testid="custom-endpoints-preset"
									data-preset={preset.name}
									disabled={busy}
									selected={draft.baseUrl === preset.baseUrl}
									onClick={() => setDraft(applyPreset(draft, preset))}
								>
									{preset.name}
								</Button>
							))}
						</div>
					</Field>
					<Field label={t("settings.customEndpoints.apiKey")} htmlFor={`${id}-key`}>
						<div className="flex items-center gap-8">
							<input
								id={`${id}-key`}
								data-testid="custom-endpoints-key"
								type="password"
								autoComplete="new-password"
								className={INPUT_CLASS}
								value={draft.apiKey}
								disabled={busy || draft.removeKey}
								placeholder={
									draft.removeKey
										? t("settings.customEndpoints.apiKeyRemoved")
										: draft.hasSavedKey
											? t("settings.customEndpoints.apiKeySaved")
											: t("settings.customEndpoints.apiKeyOptional")
								}
								onChange={(event) => setDraft({ ...draft, apiKey: event.currentTarget.value })}
							/>
							{draft.hasSavedKey ? (
								<Button
									variant="ghost"
									size="m"
									data-testid="custom-endpoints-remove-key"
									disabled={busy}
									selected={draft.removeKey}
									onClick={() => setDraft({ ...draft, removeKey: !draft.removeKey, apiKey: "" })}
								>
									{draft.removeKey
										? t("settings.customEndpoints.keepKey")
										: t("settings.customEndpoints.removeKey")}
								</Button>
							) : null}
						</div>
					</Field>
					{limitsSupported ? (
						<LimitFields draft={draft} disabled={busy} onChange={setDraft} />
					) : null}
					{gatewaySupported && draft.gateway ? (
						<GatewayFields
							gateway={draft.gateway}
							keyStatus={probeKey}
							invalid={limitIssues(draft).tags}
							disabled={busy}
							onChange={(gateway) => setDraft((current) => ({ ...current, gateway }))}
						/>
					) : null}
					<div className="flex flex-col gap-4">
						<div className="flex items-center justify-between gap-8">
							<span className="tr-title-compact text-text-default">
								{t("settings.customEndpoints.models")}
							</span>
							<Button
								variant="outline"
								size="m"
								iconLeft={Search}
								loading={probing}
								data-testid="custom-endpoints-detect"
								disabled={busy || !draft.baseUrl.trim()}
								onClick={() => void detect(draft)}
							>
								{t("settings.customEndpoints.detect")}
							</Button>
						</div>
						{draft.models.length > 0 ? (
							<ul className="flex flex-col" data-testid="custom-endpoints-models">
								{draft.models.map((model, index) => (
									<li
										key={model.id}
										data-testid="custom-endpoints-row"
										data-model={model.id}
										className={cn(
											"flex items-center gap-8 py-8",
											index > 0 && "border-border-muted border-t",
										)}
									>
										<span className="flex min-w-0 flex-1 flex-col">
											<span
												className={cn(
													"truncate tr-text-ui",
													model.selected ? "text-text-default" : "text-text-muted",
												)}
												title={model.id}
											>
												{model.id}
											</span>
											<ModelFacts model={model} />
										</span>
										<SettingsSwitch
											checked={model.selected}
											disabled={busy}
											label={t("settings.customEndpoints.use", { name: model.id })}
											testId="custom-endpoints-use"
											onChange={(selected) =>
												setDraft((current) => ({
													...current,
													models: current.models.map((row) =>
														row.id === model.id ? { ...row, selected } : row,
													),
												}))
											}
										/>
									</li>
								))}
							</ul>
						) : null}
						<div className="flex items-center gap-8">
							<input
								data-testid="custom-endpoints-manual"
								aria-label={t("settings.customEndpoints.manualLabel")}
								className={cn(INPUT_CLASS, "tr-code-text")}
								value={manualId}
								disabled={busy}
								spellCheck={false}
								placeholder={t("settings.customEndpoints.manualPlaceholder")}
								onChange={(event) => setManualId(event.currentTarget.value)}
								onKeyDown={(event) => {
									if (event.key !== "Enter") return;
									event.preventDefault();
									addManual();
								}}
							/>
							<Button
								variant="outline"
								size="m"
								iconLeft={Plus}
								data-testid="custom-endpoints-manual-add"
								disabled={busy || !manualId.trim()}
								onClick={addManual}
							>
								{t("common.add")}
							</Button>
						</div>
					</div>
					{status ? (
						<p
							role={status.tone === "error" ? "alert" : "status"}
							data-testid="custom-endpoints-status"
							data-tone={status.tone}
							className={cn(
								"tr-text-metadata",
								status.tone === "error" ? "text-feedback-error" : "text-text-muted",
							)}
						>
							{status.text}
						</p>
					) : null}
				</div>
				<DialogFooter>
					<Button variant="outline" disabled={busy} onClick={onClose}>
						{t("common.cancel")}
					</Button>
					<Button
						data-testid="custom-endpoints-save"
						loading={saving}
						disabled={busy || !canSave(draft)}
						onClick={save}
					>
						{t("common.save")}
					</Button>
				</DialogFooter>
			</DialogContent>
		</Dialog>
	);
}

type TestState =
	| { running: true }
	| { running: false; result: CustomEndpointTestResult | null; error: string | null };

function testSummary(t: Translator, result: CustomEndpointTestResult): string {
	if (result.chat === "skipped") return t("settings.customEndpoints.test.busy");
	const parts = [
		result.chat === "ok"
			? t("settings.customEndpoints.test.chatOk")
			: t("settings.customEndpoints.test.chatFailed"),
	];
	if (result.tools === "ok") parts.push(t("settings.customEndpoints.test.toolsOk"));
	else if (result.tools === "failed") parts.push(t("settings.customEndpoints.test.toolsFailed"));
	if (result.latencyMs !== null)
		parts.push(
			t("settings.customEndpoints.test.latency", { seconds: (result.latencyMs / 1000).toFixed(1) }),
		);
	if (result.outputTokensPerSecond !== null)
		parts.push(t("settings.customEndpoints.test.speed", { rate: result.outputTokensPerSecond }));
	return parts.join(" · ");
}

function testStateName(state: TestState | undefined): string {
	if (!state) return "idle";
	if (state.running) return "running";
	if (state.result?.chat === "skipped") return "busy";
	return state.result?.chat === "ok" && state.result.tools === "ok" ? "ok" : "failed";
}

function ModelTestRow({
	model,
	state,
	onTest,
}: {
	model: CustomEndpointModel;
	state: TestState | undefined;
	onTest: () => void;
}) {
	const t = useT();
	const result = state && !state.running ? state.result : null;
	const failure = state && !state.running ? state.error : null;
	const name = testStateName(state);
	return (
		<li
			data-testid="custom-endpoints-model"
			data-model={model.id}
			data-test-state={name}
			className="flex flex-col gap-2 py-4 pl-12"
		>
			<div className="flex items-center gap-8">
				<span className="flex min-w-0 flex-1 flex-col">
					<span className="truncate tr-text-metadata text-text-default" title={model.id}>
						{model.id}
					</span>
					<ModelFacts model={model} />
				</span>
				<Button
					variant="ghost"
					size="m"
					iconLeft={Play}
					data-testid="custom-endpoints-test"
					loading={state?.running === true}
					disabled={state?.running === true}
					aria-label={t("settings.customEndpoints.test.label", { name: model.id })}
					onClick={onTest}
				>
					{state?.running
						? t("settings.customEndpoints.test.running")
						: t("settings.customEndpoints.test.run")}
				</Button>
			</div>
			{result ? (
				<p
					data-testid="custom-endpoints-test-result"
					className={cn(
						"tr-text-metadata",
						name === "failed" ? "text-feedback-error" : "text-text-muted",
					)}
				>
					{testSummary(t, result)}
				</p>
			) : null}
			{result?.message || failure ? (
				<p className="break-words tr-text-metadata text-feedback-error">
					{result?.message ?? failure}
				</p>
			) : null}
		</li>
	);
}

function limitsSummary(t: Translator, endpoint: CustomEndpoint): string {
	const { maxConcurrency, timeoutSeconds, maxRetries } = endpoint.limits;
	return [
		maxConcurrency === null
			? t("settings.customEndpoints.parallelUnlimited")
			: t("settings.customEndpoints.parallelSummary", { count: maxConcurrency }),
		timeoutSeconds === null
			? null
			: t("settings.customEndpoints.timeoutSummary", { seconds: timeoutSeconds }),
		maxRetries === null
			? null
			: t("settings.customEndpoints.retriesSummary", { count: maxRetries }),
	]
		.filter((part): part is string => part !== null)
		.join(" · ");
}

export interface CustomEndpointsState {
	endpoints: CustomEndpoint[] | null;
	failed: boolean;
	load: () => Promise<void>;
	replace: (next: CustomEndpoint[]) => void;
}

export function useCustomEndpoints(): CustomEndpointsState | null {
	const protocolVersion = useAppStore((state) => state.protocolVersion);
	const providerVersion = useAppStore((state) => state.providerVersion);
	const [endpoints, setEndpoints] = useState<CustomEndpoint[] | null>(null);
	const [failed, setFailed] = useState(false);
	const sequence = useRef(0);

	const load = useCallback(async () => {
		if (!supported(protocolVersion)) return;
		const current = ++sequence.current;
		try {
			const next = await getTransport().request("customEndpoints.list", {});
			if (current !== sequence.current) return;
			setEndpoints(next);
			setFailed(false);
		} catch {
			if (current === sequence.current) setFailed(true);
		}
	}, [protocolVersion]);

	useEffect(() => {
		void load();
	}, [load, providerVersion]);

	const replace = useCallback((next: CustomEndpoint[]) => {
		sequence.current++;
		setEndpoints(next);
	}, []);

	if (!supported(protocolVersion)) return null;
	return { endpoints, failed, load, replace };
}

export function CustomEndpointsSettings({
	state: { endpoints, failed, load, replace },
}: {
	state: CustomEndpointsState;
}) {
	const t = useT();
	const [dialog, setDialog] = useState<EndpointDraft | null>(null);
	const [removing, setRemoving] = useState<CustomEndpoint | null>(null);
	const [tests, setTests] = useState<Record<string, TestState>>({});
	const limitsSupported = useLimitsSupported();

	const remove = async (endpoint: CustomEndpoint) => {
		try {
			replace(await getTransport().request("customEndpoints.remove", { id: endpoint.id }));
		} catch (error) {
			toast.error(errorText(error), t("settings.customEndpoints.removeFailed"));
		}
	};

	const runTest = (endpoint: CustomEndpoint, model: CustomEndpointModel) => {
		const key = `${endpoint.id}/${model.id}`;
		const settle = (next: TestState) => setTests((current) => ({ ...current, [key]: next }));
		settle({ running: true });
		void getTransport()
			.request(
				"customEndpoints.test",
				{ id: endpoint.id, modelId: model.id },
				{ timeoutMs: customEndpointTestBudgetMs(endpoint.limits) + TEST_REPLY_MARGIN_MS },
			)
			.then((result) => settle({ running: false, result, error: null }))
			.catch((error: unknown) => settle({ running: false, result: null, error: errorText(error) }));
	};

	return (
		<SettingsGroup
			title={t("settings.customEndpoints.title")}
			description={t("settings.customEndpoints.description")}
			testId="settings-custom-endpoints"
		>
			{failed ? (
				<div className="flex items-center gap-8 border-border-muted border-b py-12" role="alert">
					<p className="tr-text-ui text-text-muted">{t("settings.customEndpoints.loadFailed")}</p>
					<Button variant="ghost" size="m" onClick={() => void load()}>
						{t("common.retry")}
					</Button>
				</div>
			) : (
				(endpoints ?? []).map((endpoint) => (
					<div
						key={endpoint.id}
						data-row
						data-testid="custom-endpoints-endpoint"
						data-endpoint={endpoint.id}
						className="flex flex-col gap-4 border-border-muted border-b py-12"
					>
						<div className="flex items-center gap-8">
							<div className="flex min-w-0 flex-1 flex-col gap-2">
								<span className="truncate tr-text-ui text-text-default">{endpoint.name}</span>
								<span
									className="truncate tr-text-metadata text-text-muted"
									title={endpoint.baseUrl}
								>
									{t("settings.customEndpoints.endpointSummary", {
										url: endpoint.baseUrl,
										count: endpoint.models.length,
									})}
									{limitsSupported ? ` · ${limitsSummary(t, endpoint)}` : null}
								</span>
								{endpoint.gateway ? <EndpointKeyStatus endpoint={endpoint} /> : null}
							</div>
							<Button
								variant="ghost"
								size="m"
								data-testid="custom-endpoints-edit"
								aria-label={t("settings.customEndpoints.editLabel", { name: endpoint.name })}
								onClick={() => setDialog(editEndpointDraft(endpoint))}
							>
								{t("common.edit")}
							</Button>
							<IconButton
								icon={Trash2}
								label={t("settings.customEndpoints.removeLabel", { name: endpoint.name })}
								data-testid="custom-endpoints-remove"
								onClick={() => setRemoving(endpoint)}
							/>
						</div>
						{limitsSupported ? (
							<ul className="flex flex-col">
								{endpoint.models.map((model) => (
									<ModelTestRow
										key={model.id}
										model={model}
										state={tests[`${endpoint.id}/${model.id}`]}
										onTest={() => runTest(endpoint, model)}
									/>
								))}
							</ul>
						) : null}
					</div>
				))
			)}
			<div className="py-12">
				<Button
					variant="outline"
					size="m"
					iconLeft={Plus}
					data-testid="custom-endpoints-add"
					onClick={() => setDialog(newEndpointDraft())}
				>
					{t("settings.customEndpoints.add")}
				</Button>
			</div>
			{dialog ? (
				<EndpointDialog initial={dialog} onClose={() => setDialog(null)} onSaved={replace} />
			) : null}
			<ConfirmDialog
				open={removing !== null}
				onOpenChange={(open) => !open && setRemoving(null)}
				title={t("settings.customEndpoints.removeTitle", { name: removing?.name ?? "" })}
				description={t("settings.customEndpoints.removeDescription")}
				confirmLabel={t("common.remove")}
				destructive
				confirmTestId="custom-endpoints-remove-confirm"
				onConfirm={() => {
					if (removing) void remove(removing);
				}}
			/>
		</SettingsGroup>
	);
}
