# Corporate setup (fork)

This fork adds what a company needs to run ThinkRail against its own LLM gateway: org-managed servers,
a corporate-only model policy, SSO tokens, custom headers, company certificates and proxies, native
APIs per model, and LiteLLM budgets and errors that read in plain words. Every server is still a
`custom-*` provider in pi's `~/.pi/agent/models.json`, so the pi CLI sees the same models.

## Managed file

The platform team ships one JSON file per machine (MDM, GPO, Ansible, an image). ThinkRail reads it
at startup and whenever the Providers settings open, so edits apply without a reinstall.

| OS | Path |
|---|---|
| Windows | `%ProgramData%\ThinkRail\managed.json` |
| macOS | `/Library/Application Support/ThinkRail/managed.json` |
| Linux | `/etc/thinkrail/managed.json` |

Keep the file writable by administrators only: it decides which servers developers can use and which
commands run for their keys. `THINKRAIL_MANAGED_CONFIG` names another file only when the machine-wide
one doesn't exist, so a user's own environment can't switch an organization's policy off.

```json
{
  "servers": [
    {
      "id": "plata",
      "name": "Plata AI Gateway",
      "baseUrl": "https://llm.plata.internal/v1",
      "headers": { "X-Org": "plata" },
      "network": { "systemCertificates": true, "caFile": null, "proxy": null },
      "limits": { "maxConcurrency": 4, "timeoutSeconds": 600, "maxRetries": 2 },
      "gateway": { "kind": "litellm", "sessionAffinity": true, "tags": ["thinkrail"] },
      "models": [
        { "id": "qwen3-coder", "contextWindow": 131072, "maxTokens": 16384 },
        { "id": "claude-sonnet", "api": "anthropic-messages", "reasoning": true }
      ]
    }
  ],
  "policy": { "restrictModels": "managed" }
}
```

- `id` becomes the provider id `custom-org-<id>`. Keep it stable: chats remember their model by it.
- `apiKey` is optional. Set it (`$ENV_VAR` or `!command`) to ship the key; leave it out and every
  developer adds their own key, for example a personal LiteLLM virtual key.
- `models` is optional. List them to lock the set; leave it out and developers use **Find models** and
  pick their own. Each model takes `contextWindow`, `maxTokens`, `reasoning`, `vision`, `api`
  (`openai-completions`, `openai-responses` or `anthropic-messages`) and `cost` (USD per million tokens).
- Everything the file sets is locked in the dialog and rewritten on every sync, so a hand-edited
  models.json can't move a managed server; change `baseUrl` in the file to migrate every machine.
  Managed servers can't be removed in the app; take one out of the file and it disappears from every
  machine on the next start or settings visit.
- `policy.restrictModels`: `"managed"` allows chats, subagents, reviews and one-shot calls only on the
  file's servers; `"custom"` allows any custom server; leave it out for no restriction. The model
  picker hides everything else, and a request to another provider fails before it is sent.

A server the file describes badly is skipped, and Settings shows why; the rest still apply, and nothing
synced earlier is removed while the file has an error. If the file exists but can't be read or parsed,
ThinkRail keeps the last good settings; when it has none, it fails closed: only the servers the file
set up before stay usable. An unknown `restrictModels` value counts as `"managed"`.

## Keys and SSO tokens

The key field takes a literal key, `$ENV_VAR`, or a `!command` whose output is the key. Commands run
on the developer's machine and their token is reused until shortly before its JWT `exp` (five minutes
for non-JWT output). A 401 or 403 from the server drops the token, so the next request fetches a
fresh one. Examples:

- Azure AD / Entra: `!az account get-access-token --resource api://llm-gateway --query accessToken -o tsv`
- Google: `!gcloud auth print-identity-token`
- Vault: `!vault kv get -field=key secret/llm/thinkrail`
- 1Password: `!op read op://Engineering/LiteLLM/credential`

A command the user saved through the dialog, or the managed file ships as the server's `apiKey`, also
runs for **Find models** and the key status. A command written into models.json by anything else runs
only for chats, the same as in pi, until someone saves it from the dialog. Commands time out after 15
seconds; on Windows they run in `cmd.exe`.

## Headers, certificates and proxies

- **Headers** go with every request: an Azure API Management `Ocp-Apim-Subscription-Key`, Cloudflare
  Access `CF-Access-Client-Id` / `CF-Access-Client-Secret`, an org id. Values take `$ENV_VAR` or a
  literal; they can't be commands (pi would run a header command on every request), so put a token
  command in the key field. Only a value that is exactly one `$VAR` is shown again; anything else is a
  secret, and an empty field keeps the saved value.
- **Trust system certificates** adds the operating system's store to the trusted roots, so a company
  root CA pushed by IT works without a file. **CA certificate file** adds a PEM bundle instead.
- **Proxy** sends that server's requests through `http(s)://host:port`. Leave it blank to use
  `HTTPS_PROXY` / `NO_PROXY` from the environment. Proxies with credentials aren't accepted.

## Native APIs per model

LiteLLM serves Anthropic's `/v1/messages` and OpenAI's `/v1/responses` next to chat completions. Claude
models keep their thinking and prompt caching only through Messages, so Find models picks Anthropic
Messages for models LiteLLM routes to Anthropic, Bedrock or Vertex Claude. Each model's API can be
changed in the dialog.

## LiteLLM

When Find models sees a LiteLLM proxy it reads `/model/info` for each model's context, output limit,
tools, vision and thinking support and price, and the key's own `/key/info` for spend, budget, reset,
expiry and rate limits; the server row shows the key's status and warns at 90% of the budget or a week
before expiry. Requests carry `x-litellm-session-id` (one per chat, for sticky routing and Langfuse
sessions) and `x-litellm-tags`. Budget-exceeded, rate-limit and expired or invalid key errors are
rewritten into plain words in the chat, with the key's spend and reset date where LiteLLM reports them.

For the most accurate Find models, set `model_info` (`max_input_tokens`, `max_output_tokens`,
`supports_function_calling`, `supports_vision`, `supports_reasoning`) on every route in the proxy's
config, and enable session affinity on the router if you want prompt-cache reuse.
