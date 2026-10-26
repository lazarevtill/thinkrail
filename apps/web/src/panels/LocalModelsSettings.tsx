import {
	RiPencilLine as Pencil,
	RiAddLine as Plus,
	RiSearchLine as Search,
	RiDeleteBin6Line as Trash2,
} from "@remixicon/react";
import {
	isLocalModelContextWindow,
	LOCAL_MODEL_CONTEXT_LIMITS,
	LOCAL_MODEL_DEFAULT_URLS,
	LOCAL_MODEL_SERVER_KINDS,
	LOCAL_MODELS_PROTOCOL_VERSION,
	type LocalModelEndpoint,
	type LocalModelEndpointDraft,
	type LocalModelEntry,
	type LocalModelProbe,
	type LocalModelServerKind,
} from "@thinkrail/contracts";
import type { MessageKey } from "@thinkrail/locales";
import { localeOf, type Translator, useT } from "@thinkrail/locales/react";
import { Button } from "@thinkrail/ui/button";
import {
	Dialog,
	DialogContent,
	DialogDescription,
	DialogFooter,
	DialogHeader,
	DialogTitle,
} from "@thinkrail/ui/dialog";
import { ToggleSegment } from "@thinkrail/ui/toggle-segment";
import { useCallback, useEffect, useId, useRef, useState } from "react";
import { toast, useAppStore } from "@/store";
import { errorText, getTransport } from "@/transport";
import { ConfirmDialog } from "./ConfirmDialog";
import { SettingsGroup } from "./SettingsGroup";
import { SettingsSwitch } from "./SettingsSwitch";

const INPUT_CLASS =
	"w-full min-w-0 rounded-sm border border-control-border-default bg-control-bg px-8 py-4 tr-text-ui text-text-default outline-none placeholder:text-text-muted focus-visible:border-control-border-active disabled:border-control-disabled-border disabled:bg-control-disabled-bg disabled:text-control-disabled-text";

const KIND_LABEL: Readonly<Record<LocalModelServerKind, MessageKey>> = {
	"llama.cpp": "settings.localModels.kind.llamaCpp",
	vllm: "settings.localModels.kind.vllm",
	ollama: "settings.localModels.kind.ollama",
	lmstudio: "settings.localModels.kind.lmstudio",
	"openai-compatible": "settings.localModels.kind.other",
};

const PORT_KIND: Readonly<Record<string, LocalModelServerKind>> = {
	"8080": "llama.cpp",
	"8000": "vllm",
	"11434": "ollama",
	"1234": "lmstudio",
};

interface ModelRow extends LocalModelEntry {
	selected: boolean;
	contextText: string;
	detected: boolean;
}

interface Draft {
	id?: string;
	kind: LocalModelServerKind;
	name: string;
	nameTouched: boolean;
	baseUrl: string;
	apiKey: string;
	hasSavedKey: boolean;
	models: ModelRow[];
}

function supported(protocolVersion: number | null): boolean {
	return protocolVersion !== null && protocolVersion >= LOCAL_MODELS_PROTOCOL_VERSION;
}

function guessKind(baseUrl: string): LocalModelServerKind {
	try {
		return PORT_KIND[new URL(baseUrl).port] ?? "openai-compatible";
	} catch {
		return "openai-compatible";
	}
}

function newDraft(t: Translator): Draft {
	return {
		kind: "llama.cpp",
		name: t(KIND_LABEL["llama.cpp"]),
		nameTouched: false,
		baseUrl: LOCAL_MODEL_DEFAULT_URLS["llama.cpp"],
		apiKey: "",
		hasSavedKey: false,
		models: [],
	};
}

function editDraft(endpoint: LocalModelEndpoint): Draft {
	return {
		id: endpoint.id,
		kind: guessKind(endpoint.baseUrl),
		name: endpoint.name,
		nameTouched: true,
		baseUrl: endpoint.baseUrl,
		apiKey: "",
		hasSavedKey: endpoint.hasApiKey,
		models: endpoint.models.map((model) => ({
			...model,
			selected: true,
			contextText: String(model.contextWindow),
			detected: false,
		})),
	};
}

const tokens = (t: Translator, value: number) => value.toLocaleString(localeOf(t));

function contextValid(text: string): boolean {
	return /^\d+$/.test(text.trim()) && isLocalModelContextWindow(Number(text.trim()));
}

function mergeProbe(current: Draft, probe: LocalModelProbe): Draft {
	const known = new Map(current.models.map((model) => [model.id, model]));
	const fresh: ModelRow[] = probe.models.map((model) => {
		const previous = known.get(model.id);
		const contextWindow =
			model.contextWindow ?? previous?.contextWindow ?? LOCAL_MODEL_CONTEXT_LIMITS.fallback;
		return {
			id: model.id,
			contextWindow,
			contextText: previous && !previous.detected ? previous.contextText : String(contextWindow),
			reasoning: previous?.reasoning ?? model.reasoning,
			vision: model.vision,
			selected: previous ? previous.selected : current.models.length === 0,
			detected: model.contextWindow !== null,
		};
	});
	const missing = current.models.filter(
		(model) => !probe.models.some((found) => found.id === model.id),
	);
	return { ...current, baseUrl: probe.baseUrl, models: [...fresh, ...missing] };
}

function LocalServerDialog({
	initial,
	onOpenChange,
	onSaved,
}: {
	initial: Draft;
	onOpenChange: (open: boolean) => void;
	onSaved: (endpoints: LocalModelEndpoint[]) => void;
}) {
	const t = useT();
	const id = useId();
	const [draft, setDraft] = useState(initial);
	const [probing, setProbing] = useState(false);
	const [saving, setSaving] = useState(false);
	const [error, setError] = useState<string | null>(null);
	const busy = probing || saving;
	const selected = draft.models.filter((model) => model.selected);
	const invalid = selected.some((model) => !contextValid(model.contextText));
	const range = `${tokens(t, LOCAL_MODEL_CONTEXT_LIMITS.min)}–${tokens(t, LOCAL_MODEL_CONTEXT_LIMITS.max)}`;

	const setKind = (kind: LocalModelServerKind) =>
		setDraft((current) => ({
			...current,
			kind,
			name: current.nameTouched ? current.name : t(KIND_LABEL[kind]),
			baseUrl: Object.values(LOCAL_MODEL_DEFAULT_URLS).includes(current.baseUrl)
				? LOCAL_MODEL_DEFAULT_URLS[kind]
				: current.baseUrl,
		}));

	const updateModel = (modelId: string, patch: Partial<ModelRow>) =>
		setDraft((current) => ({
			...current,
			models: current.models.map((model) =>
				model.id === modelId ? { ...model, ...patch } : model,
			),
		}));

	const detect = async () => {
		setProbing(true);
		setError(null);
		const keyEntered = draft.apiKey.trim() !== "";
		await getTransport()
			.request("localModels.probe", {
				kind: draft.kind,
				baseUrl: draft.baseUrl,
				...(keyEntered || !draft.id ? { apiKey: draft.apiKey } : { endpointId: draft.id }),
			})
			.then((probe) => {
				setDraft((current) => mergeProbe(current, probe));
				if (probe.models.length === 0) setError(t("settings.localModels.noModelsFound"));
			})
			.catch((err: unknown) => setError(errorText(err)))
			.finally(() => setProbing(false));
	};

	const save = async () => {
		if (selected.length === 0 || invalid) return;
		setSaving(true);
		setError(null);
		const params: LocalModelEndpointDraft = {
			...(draft.id ? { id: draft.id } : {}),
			kind: draft.kind,
			name: draft.name,
			baseUrl: draft.baseUrl,
			...(draft.apiKey.trim() || !draft.id ? { apiKey: draft.apiKey } : {}),
			models: selected.map((model) => ({
				id: model.id,
				contextWindow: Number(model.contextText.trim()),
				reasoning: model.reasoning,
				vision: model.vision,
			})),
		};
		await getTransport()
			.request("localModels.save", params)
			.then((endpoints) => {
				onSaved(endpoints);
				onOpenChange(false);
			})
			.catch((err: unknown) => setError(errorText(err)))
			.finally(() => setSaving(false));
	};

	return (
		<Dialog open onOpenChange={(next) => !busy && onOpenChange(next)}>
			<DialogContent
				data-testid="local-models-dialog"
				className="flex max-h-[85vh] max-w-[40rem] flex-col"
			>
				<DialogHeader>
					<DialogTitle>
						{draft.id ? t("settings.localModels.editTitle") : t("settings.localModels.addTitle")}
					</DialogTitle>
					<DialogDescription>{t("settings.localModels.dialogDescription")}</DialogDescription>
				</DialogHeader>
				<div className="-mx-16 flex min-h-0 flex-1 flex-col gap-12 overflow-y-auto px-16">
					<div className="flex flex-col gap-4">
						<span className="tr-text-metadata text-text-muted">
							{t("settings.localModels.serverType")}
						</span>
						<div className="flex flex-wrap gap-4" data-testid="local-models-kind">
							{LOCAL_MODEL_SERVER_KINDS.map((kind) => (
								<ToggleSegment
									key={kind}
									testid={`local-models-kind-${kind}`}
									label={t(KIND_LABEL[kind])}
									active={draft.kind === kind}
									disabled={busy}
									onClick={() => setKind(kind)}
								/>
							))}
						</div>
					</div>
					<label className="flex flex-col gap-4" htmlFor={`${id}-name`}>
						<span className="tr-text-metadata text-text-muted">
							{t("settings.localModels.name")}
						</span>
						<input
							id={`${id}-name`}
							data-testid="local-models-name"
							className={INPUT_CLASS}
							value={draft.name}
							disabled={busy}
							onChange={(event) =>
								setDraft({ ...draft, name: event.currentTarget.value, nameTouched: true })
							}
						/>
					</label>
					<label className="flex flex-col gap-4" htmlFor={`${id}-url`}>
						<span className="tr-text-metadata text-text-muted">
							{t("settings.localModels.url")}
						</span>
						<input
							id={`${id}-url`}
							data-testid="local-models-url"
							className={INPUT_CLASS}
							value={draft.baseUrl}
							disabled={busy}
							spellCheck={false}
							placeholder={LOCAL_MODEL_DEFAULT_URLS[draft.kind]}
							onChange={(event) => setDraft({ ...draft, baseUrl: event.currentTarget.value })}
						/>
					</label>
					<label className="flex flex-col gap-4" htmlFor={`${id}-key`}>
						<span className="tr-text-metadata text-text-muted">
							{t("settings.localModels.apiKey")}
						</span>
						<input
							id={`${id}-key`}
							data-testid="local-models-key"
							type="password"
							autoComplete="off"
							className={INPUT_CLASS}
							value={draft.apiKey}
							disabled={busy}
							placeholder={
								draft.hasSavedKey
									? t("settings.localModels.apiKeyKeep")
									: t("settings.localModels.apiKeyPlaceholder")
							}
							onChange={(event) => setDraft({ ...draft, apiKey: event.currentTarget.value })}
						/>
					</label>
					<div className="flex flex-wrap items-center justify-between gap-8">
						<span className="tr-text-ui text-text-default">{t("settings.localModels.models")}</span>
						<Button
							variant="outline"
							size="sm"
							data-testid="local-models-detect"
							disabled={busy || !draft.baseUrl.trim()}
							onClick={() => void detect()}
						>
							<Search className="size-14" />
							{probing ? t("settings.localModels.detecting") : t("settings.localModels.detect")}
						</Button>
					</div>
					{draft.models.length === 0 ? (
						<p className="rounded-sm bg-control-bg px-12 py-8 tr-text-metadata text-text-muted">
							{t("settings.localModels.detectHint")}
						</p>
					) : (
						<ul className="flex flex-col" data-testid="local-models-list">
							{draft.models.map((model) => {
								const bad = model.selected && !contextValid(model.contextText);
								return (
									<li
										key={model.id}
										data-testid="local-models-row"
										data-model={model.id}
										className="flex flex-wrap items-center gap-8 border-border-muted border-b py-8"
									>
										<input
											type="checkbox"
											checked={model.selected}
											disabled={busy}
											aria-label={t("settings.localModels.useModel", { name: model.id })}
											onChange={(event) =>
												updateModel(model.id, { selected: event.currentTarget.checked })
											}
										/>
										<span
											className="min-w-0 flex-1 truncate tr-text-ui text-text-default"
											title={model.id}
										>
											{model.id}
										</span>
										<input
											inputMode="numeric"
											className={`${INPUT_CLASS} w-100 ${bad ? "border-feedback-error" : ""}`}
											value={model.contextText}
											disabled={busy || !model.selected}
											aria-invalid={bad}
											aria-label={t("settings.localModels.contextFor", { name: model.id })}
											onChange={(event) =>
												updateModel(model.id, { contextText: event.currentTarget.value })
											}
										/>
										<span className="tr-text-metadata text-text-muted">
											{t("settings.localModels.thinking")}
										</span>
										<SettingsSwitch
											checked={model.reasoning}
											disabled={busy || !model.selected}
											label={t("settings.localModels.thinkingFor", { name: model.id })}
											testId={`local-models-thinking-${model.id}`}
											onChange={(reasoning) => updateModel(model.id, { reasoning })}
										/>
									</li>
								);
							})}
						</ul>
					)}
					{invalid ? (
						<p className="tr-text-metadata text-feedback-error">
							{t("settings.localModels.contextInvalid", { range })}
						</p>
					) : null}
					<p className="tr-text-metadata text-text-muted">{t("settings.localModels.note")}</p>
					{error ? (
						<p
							role="alert"
							data-testid="local-models-error"
							className="tr-text-metadata text-feedback-error"
						>
							{error}
						</p>
					) : null}
				</div>
				<DialogFooter>
					<Button variant="outline" disabled={busy} onClick={() => onOpenChange(false)}>
						{t("common.cancel")}
					</Button>
					<Button
						data-testid="local-models-save"
						disabled={busy || selected.length === 0 || invalid || !draft.name.trim()}
						onClick={() => void save()}
					>
						{t("common.save")}
					</Button>
				</DialogFooter>
			</DialogContent>
		</Dialog>
	);
}

export function LocalModelsSettings() {
	const t = useT();
	const protocolVersion = useAppStore((state) => state.protocolVersion);
	const providerVersion = useAppStore((state) => state.providerVersion);
	const [endpoints, setEndpoints] = useState<LocalModelEndpoint[] | null>(null);
	const [failed, setFailed] = useState(false);
	const [dialog, setDialog] = useState<Draft | null>(null);
	const [removing, setRemoving] = useState<LocalModelEndpoint | null>(null);
	const sequence = useRef(0);

	const load = useCallback(async () => {
		if (!supported(protocolVersion)) return;
		const current = ++sequence.current;
		try {
			const next = await getTransport().request("localModels.list", {});
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

	if (!supported(protocolVersion)) return null;

	const remove = async (endpoint: LocalModelEndpoint) => {
		try {
			setEndpoints(await getTransport().request("localModels.remove", { id: endpoint.id }));
		} catch {
			toast.error(t("settings.localModels.removeFailed"));
		}
	};

	return (
		<SettingsGroup
			title={t("settings.localModels.title")}
			description={t("settings.localModels.description")}
			testId="settings-local-models"
		>
			{failed ? (
				<div className="flex items-center gap-8 border-border-muted border-b py-12" role="alert">
					<p className="tr-text-ui text-text-muted">{t("settings.localModels.loadFailed")}</p>
					<Button variant="ghost" size="sm" onClick={() => void load()}>
						{t("common.retry")}
					</Button>
				</div>
			) : (
				(endpoints ?? []).map((endpoint) => (
					<div
						key={endpoint.id}
						data-testid="local-models-endpoint"
						data-endpoint={endpoint.id}
						className="flex items-center gap-8 border-border-muted border-b py-12"
					>
						<div className="flex min-w-0 flex-1 flex-col gap-2">
							<span className="truncate tr-text-ui text-text-default">{endpoint.name}</span>
							<span className="truncate tr-text-metadata text-text-muted" title={endpoint.baseUrl}>
								{t("settings.localModels.endpointSummary", {
									url: endpoint.baseUrl,
									count: endpoint.models.length,
								})}
							</span>
						</div>
						<Button
							variant="ghost"
							size="sm"
							data-testid="local-models-edit"
							aria-label={t("settings.localModels.editLabel", { name: endpoint.name })}
							onClick={() => setDialog(editDraft(endpoint))}
						>
							<Pencil className="size-14" />
							{t("common.edit")}
						</Button>
						<Button
							variant="ghost"
							size="sm"
							data-testid="local-models-remove"
							aria-label={t("settings.localModels.removeLabel", { name: endpoint.name })}
							onClick={() => setRemoving(endpoint)}
						>
							<Trash2 className="size-14" />
							{t("common.remove")}
						</Button>
					</div>
				))
			)}
			<div className="py-12">
				<Button
					variant="outline"
					size="sm"
					data-testid="local-models-add"
					onClick={() => setDialog(newDraft(t))}
				>
					<Plus className="size-14" />
					{t("settings.localModels.add")}
				</Button>
			</div>
			{dialog ? (
				<LocalServerDialog
					initial={dialog}
					onOpenChange={(open) => !open && setDialog(null)}
					onSaved={setEndpoints}
				/>
			) : null}
			<ConfirmDialog
				open={removing !== null}
				onOpenChange={(open) => !open && setRemoving(null)}
				title={t("settings.localModels.removeTitle", { name: removing?.name ?? "" })}
				description={t("settings.localModels.removeDescription")}
				confirmLabel={t("common.remove")}
				destructive
				confirmTestId="local-models-remove-confirm"
				onConfirm={() => {
					if (removing) void remove(removing);
					setRemoving(null);
				}}
			/>
		</SettingsGroup>
	);
}
