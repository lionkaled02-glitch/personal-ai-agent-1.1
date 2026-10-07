# CONTRIBUTING

Thanks for helping build the personal AI agent. This is a deliberately
**incremental** project: small, well-tested, modular changes on top of the
existing foundation. Read [AGENTS.md](AGENTS.md) (engineering rules) and
[ARCHITECTURE.md](ARCHITECTURE.md) before writing code.

---

## Setup

Python **3.11+**.

```bash
git clone <repo>
cd personal-ai-agent

python3 -m venv .venv
source .venv/bin/activate

pip install -e packages/agent-core   # the core library (editable)
pip install pytest ruff mypy          # dev tools

# only if you will develop/test against a real OpenAI model:
pip install -e "packages/agent-core[openai]"
```

Configuration is environment-based. If you want to override defaults, copy
`.env.example` to `.env` (which is git-ignored). **The default (mock) needs
no API keys.** Real-provider settings: `MODEL_PROVIDER`, `MODEL_NAME`,
`MODEL_TIMEOUT_S`, `MODEL_MAX_RETRIES`, and `OPENAI_API_KEY` (env only,
never committed).

---

## Run the quality gates

All four must pass before a change is considered complete:

```bash
ruff format .     # format
ruff check .      # lint
mypy              # type-check (strict)
pytest            # tests (offline, deterministic)
```

Run the end-to-end demo to sanity-check behavior:

```bash
python apps/backend/src/main.py "Run the demo tool."
```

---

## Testing conventions

- **Deterministic and offline.** Tests must not require real external AI APIs
  or network access. Use `MockModelProvider` and the injectable clock
  (see `packages/agent-core/tests/conftest.py`).
- **Test important behavior:** state transitions, permission decisions, tool
  validation, the planner, and the end-to-end flow are all covered. If you add
  behavior, add a test.
- **Document tests use generated fixtures.** `packages/agent-core/tests/
  document_fixtures.py` builds deterministic TXT/MD/PDF/DOCX/PPTX/XLSX
  fixtures in temp directories (the PDF is generated structurally). Never
  test against real user files.
- Place core tests in `packages/agent-core/tests/` and app-level tests in
  `apps/backend/tests/`.

---

## Adding a tool

The Tool Runtime (Phase 2) handles permission gating, input/output
validation, metadata, and events — a well-behaved tool is just a spec plus a
pure `run`.

1. Create a class with a `spec: ToolSpec` and a `run(input) -> ToolResult`
   method (see `demo_tools.py` and `builtin_tools/` for patterns).
2. Fill in the **full** spec: `name`, `description`, `input_schema`,
   `output_schema`, `permission_level`, `version`, and `deterministic`
   (declare `False` if the tool reads the clock or any external state).
3. Declare the correct `permission_level` (`LOW` / `MEDIUM` / `HIGH`) —
   see [SECURITY.md](SECURITY.md).
4. **Validate & bound input yourself too.** The runtime checks the input
   against `input_schema`, but also enforce sensible bounds (e.g. max
   lengths) and return a *structured* failure — `ToolResult(ok=False,
   error=..., error_code="...")` — rather than raising. Never `eval`/`exec`,
   never call `subprocess`, never touch the filesystem/network unless that is
   the tool's explicit, permissioned purpose.
5. **Return output that matches `output_schema`.** A mismatch is a structured
   `TOOL_OUTPUT_INVALID` failure, not a value handed to the Agent.
6. Register it in a `ToolRegistry` (or extend `register_default_tools` for a
   built-in).
7. Add tests: registration, valid input, invalid input, permission path,
   and the failure/`error_code` behavior.
8. No core changes should be needed. If they are, stop and reconsider —
   that's a smell.

Do **not** add tools with shell access, unrestricted filesystem or network
access, or broad side effects before the relevant roadmap phase.

### Adding a workspace tool (Phase 3)

Workspace filesystem tools follow the same spec+`run` contract, plus
workspace-specific rules:

1. **Construct the tool with the `Workspace`** it operates on (see
   `workspace_tools/`) — never with a raw path and never without one.
2. **Resolve every path through `Workspace.resolve`** (or the `require_file`
   / `require_directory` / `require_source` helpers in
   `workspace_tools/_common.py`). Never build host paths yourself, and never
   rely on string-prefix checks for containment.
3. **Never coerce input.** Use `require_str_field` for required string
   inputs; malformed types fail closed with a structured error.
4. **Bound the operation** with the configured limits
   (`WorkspaceLimits`): file sizes for read/copy/write, entry counts for
   listing/search, path length. Report truncation with a `truncated` flag.
5. **Return the documented stable error codes** (`path_outside_workspace`,
   `path_not_found`, `source_not_found`, `target_exists`, `not_a_file`,
   `not_a_directory`, `file_too_large`, `content_too_large`, `decode_error`,
   `invalid_encoding`, `invalid_path`, `invalid_content`,
   `unsupported_operation`, `filesystem_error`, `security_violation`, …).
   Messages must contain only the workspace-relative path — never the
   absolute host path, never file contents.
6. **Keep it inside the boundary:** no `subprocess`, no `shutil`-style
   helpers that could follow links out, no directory deletion, no implicit
   `mkdir` on write/copy/move, atomic writes where practical.
7. **Register via `register_workspace_tools`** (keeps the workspace tool
   set auditable as one unit) and add tests for the happy path, every
   error code you can trigger, and at least one escape attempt.

### Adding a document tool or parser (Phase 4)

Document tools and parsers follow the same spec+`run` contract, plus
document-specific rules:

1. **Document content is untrusted data.** Parsers extract text/structure
   only — they must never execute anything a document contains (macros,
   scripts, formulas, links) and must never treat document text as
   instructions. XLSX stays `read_only` + `data_only` (cached values, no
   formula evaluation).
2. **Parsers implement the interface, not a format leak.** Subclass
   `DocumentParser` (`documents/parsers/base.py`), declare
   `supported_extensions` / `supported_media_types`, implement `_extract`
   (return `ExtractedContent`). The shared `normalize` handles limits,
   truncation reporting, stats, and deterministic ids — don't re-implement
   it. Binary parsers import their library **lazily inside `_extract`** and
   raise `DocumentError(PARSER_UNAVAILABLE, …)` on `ImportError` so the
   package stays importable without the `docs` extra.
3. **New third-party imports in the document layer are blocked by a static
   test** (`test_security_boundaries.py` whitelists stdlib + the four parser
   libraries + pydantic). A genuinely new parsing library needs the
   whitelist, the `docs` extra, a py.typed/mypy audit, and a security review.
4. **Respect `DocumentLimits`** (`documents/limits.py`). Size violations are
   structured errors; capacity violations produce an explicitly reported
   truncation (`truncated=True` + warning) — never a silent cut, never
   unbounded memory.
5. **Tools resolve paths through the Phase 3 `Workspace`** (via
   `load_document` in `document_tools/_common.py`) and keep the stable
   `DocumentError` codes in error messages (workspace-relative paths only,
   no host paths, no document content in inspect-style metadata).
6. **Permissions:** reading/extraction/searching documents is LOW;
   `index_document` is MEDIUM (internal knowledge mutation) and must go
   through the fail-safe approval path — a denied index performs no store
   mutation. Keep it that way unless the mutation grows.
7. **Retrieval stays deterministic and provider-neutral.** Don't add
   embeddings or external models to `KnowledgeStore`; if you need semantic
   search, implement the `RetrievalIndex` protocol with a new backend.
8. **Register via `register_document_tools`** and add tests: happy path per
   tool, every structured error code, permission behavior (LOW runs without
   approval; MEDIUM denied ⇒ no mutation), boundary escapes, and at least
   one injection-style document proving content stays data.

### Adding a memory tool, store, or retriever (Phase 5)

Memory tools, stores, and retrievers follow the same spec+`run` contract,
plus memory-specific rules:

1. **Memory content is untrusted data.** Retrieved memory and document
   content must never be executed or interpreted as instructions — not by
   tools, the context builder, or the agent. Injection-style content is a
   required test case for any new memory-facing code.
2. **Creation stays explicit.** Never add implicit memory capture
   (auto-saving messages, model outputs, or tool results). New memories
   come through `remember` (MEDIUM, approval-gated) or a clearly defined
   trusted internal pathway — document the pathway if you add one.
3. **Policy is enforced at the store layer.** Enforce `MemoryLimits`
   (allowed types, content chars, metadata bytes, item cap, recall cap) in
   the store itself so no code path can bypass the caps; tool-layer
   validation is UX, the store is the guarantee. Failures raise
   `MemoryStoreError` with the stable codes — never partial writes.
4. **Preserve determinism.** Stable ids (`make_memory_id`), ordering
   (`created_at`, then `memory_id`), and id tie-breaks in ranking. If you
   change recall ranking, update the determinism tests.
5. **Preserve expiration semantics.** Only `short_term`/`working` may get
   an implicit TTL; `long_term`/`knowledge` must never implicitly expire or
   be purged. Expired/soft-forgotten memories are never "active".
6. **Deletion stays safe.** `forget` soft-deactivates by default; hard
   delete is an explicit opt-in behind HIGH-permission approval and affects
   exactly one memory. Don't introduce recursive/batch deletion.
7. **No new capability in the memory layer.** The `memory/`,
   `memory_tools/`, and `rag/` packages are stdlib + pydantic only — a
   static whitelist test enforces it. No network, no database drivers, no
   subprocess/shell/eval/exec. A durable or vector backend implements the
   `MemoryStore` / `MemoryRetriever` protocols as a *new* backend (see
   ARCHITECTURE.md §6), not an addition to these modules.
8. **Privacy in events.** Keep confirmations metadata-only (no content in
   `remember` output/events); rely on `bounded_value` for the rest. Full
   memory content must not appear in event payloads by default.
9. **Register via `register_memory_tools`** and add tests: happy path per
   tool, every structured error code, permission behavior (LOW runs without
   approval; MEDIUM/HIGH denied ⇒ no mutation), immutable identity fields,
   soft vs hard forget, expiration, limits, and at least one injection
   proving content stays data.

## Adding a model provider

Follow the `OpenAIProvider` pattern (Phase 1) so the core stays
vendor-agnostic:

1. Subclass `ModelProvider` in `providers/<name>_provider.py`; implement
   `complete()` (optionally `stream()` / `embed()`); set `name` and declare
   `capabilities`.
2. **Lazy vendor import** — import the SDK inside methods (client
   construction / request time), never at module top level, so
   `import agent_core` works without the SDK installed.
3. Add an **optional extra** in `packages/agent-core/pyproject.toml`
   (e.g. `[project.optional-dependencies] <name> = [...]`) and a mypy
   `ignore_missing_imports` override for the SDK in the root `pyproject.toml`.
4. **Credentials from the environment only.** Read the key inside
   `__init__`; raise `ProviderConfigurationError` (naming the variable, never
   a value) when it is missing. Never store the key on the instance beyond
   what the SDK client needs, and never put it in logs/errors.
5. **Normalize errors** to the project hierarchy: timeouts/5xx/429/connection
   → `TransientProviderError` (the gateway retries these); auth/config
   problems → `ProviderConfigurationError`; everything else → `ProviderError`
   with bounded, secret-free messages. Disable the SDK's own retry layer so
   the `ModelGateway` owns retry policy.
6. **Register the provider** in `providers/factory.py`
   (`create_provider` + `SUPPORTED_PROVIDERS`). Nothing upstream (gateway,
   planner, agent) changes.
7. **Tests:** offline unit tests with a stub client (inject it via the
   provider's `client`/constructor hook) covering request mapping, the error
   map, and key-leakage; plus an opt-in live test gated on the env key
   (skipped by default).
8. Update `.env.example` (placeholders only) and the docs
   (ARCHITECTURE.md, ROADMAP.md, SECURITY.md, README.md).

## Changing the core

Prefer additive changes. If a public signature must change, note it in the PR
and update the docs. Keep dependencies one-way:
`apps/ → agent_core → (stdlib + pydantic)`.

---

## Documentation

Docs must stay **truthful**. Use exactly these labels:

- **IMPLEMENTED** / **PLANNED** / **NOT IMPLEMENTED**

When you change code, update [ARCHITECTURE.md](ARCHITECTURE.md) and
[ROADMAP.md](ROADMAP.md) so the labels match reality.

---

## Commit & PR guidelines

- Small, focused changes. One concern per change.
- Conventional-commit style subject (`feat:`, `fix:`, `docs:`, `test:`,
  `chore:`, `refactor:`).
- In the PR description: what changed, why, which quality gates you ran, and
  any doc updates.
- Never commit `.env`, credentials, or secrets.
- Do not force-push to shared branches.

---

## Definition of done

A change is done when:

- [ ] `ruff format .`, `ruff check .`, `mypy`, and `pytest` all pass.
- [ ] No secrets/credentials introduced.
- [ ] No scratch or unnecessary files.
- [ ] Docs (labels + descriptions) match the implementation.
- [ ] No unrelated changes included.
