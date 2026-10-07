# AGENTS.md — Engineering Rules

Read this **before touching any code**. These rules apply to every change in
this repository, whether made by a human or an AI agent.

The project is building a modular personal autonomous AI agent **incrementally**.
The guiding principle is: **build only what the current phase needs, keep every
boundary replaceable, and never fake capability that does not exist.**

---

## Ground rules (project-wide)

1. **Inspect existing code before editing.** Read the module and its callers.
   Understand what is already there before changing anything.
2. **Do not rewrite unrelated code.** A fix stays a fix. Refactoring is a
   separate, deliberate change.
3. **Do not create duplicate abstractions.** If a concept (tool, provider,
   task, permission) already has a home, extend it — do not fork it.
4. **Use a modular architecture.** Each concern lives in one module with a
   small, explicit interface. See [ARCHITECTURE.md](ARCHITECTURE.md).
5. **Keep providers replaceable.** The core depends on the `ModelProvider`
   *interface*, never on a specific vendor SDK. New vendors are new adapters,
   not edits to the core.
6. **Use interfaces/protocols where appropriate.** Boundaries (`Planner`,
   `Tool`, `ModelProvider`, `Verifier`, `ApprovalCallback`) are declared as
   `Protocol`s or ABCs so implementations are swappable.
7. **Use strong typing.** All public code is fully type-annotated. `mypy` runs
   in `strict` mode and must pass. Avoid `Any` at boundaries.
8. **Write tests for important behavior.** State transitions, permission
   decisions, tool validation, and the end-to-end flow are all tested. New
   behavior ⇒ new test.
9. **Never hard-code secrets.** No API keys, tokens, or passwords in code,
   docs, or examples.
10. **Never commit credentials.** `.env` (and any real config) is git-ignored.
    Only `.env.example` with placeholders is committed.
11. **Never expose secrets in logs or events.** Event payloads and logs carry
    concise *operational* data only — never model chain-of-thought, raw
    prompts, or credentials.
12. **Use environment variables for configuration.** All tunables flow through
    `agent_core.config.Settings` (env-based). See [SECURITY.md](SECURITY.md).
13. **Maintain backwards compatibility when possible.** Prefer additive
    changes. If a public signature must change, say so explicitly.
14. **Document important architectural decisions.** Record *why* in
    [ARCHITECTURE.md](ARCHITECTURE.md) (the "Decisions" section), not just
    *what*.
15. **Do not implement future features prematurely.** If a capability belongs
    to a later phase, add an interface stub *at most* — no half-wired
    implementations. Distinguish IMPLEMENTED / PLANNED / NOT IMPLEMENTED in
    docs.

---

## Where things live

| Path | Purpose | Rules |
| --- | --- | --- |
| `packages/agent-core/src/agent_core/` | The core library. | The only importable package. Must not import from `apps/`. |
| `apps/backend/src/` | Entry points / scripts. | May import `agent_core`. No business logic — orchestration lives in the core. |
| `packages/agent-core/tests/`, `apps/backend/tests/` | Tests. | Deterministic, offline. No real external AI APIs. |
| `data/` | Runtime data (uploads, generated, workspace). | Content is git-ignored. Never read/write outside `DATA_ROOT` without an explicit, approved reason. |
| `*.md` (root) | Documentation. | Must match the code. Keep IMPLEMENTED / PLANNED / NOT IMPLEMENTED labels accurate. |

**Dependency direction is one-way:** `apps/ → agent_core → (stdlib + pydantic)`.
The core must never depend on an app, a UI, or a vendor SDK.

---

## Verification checklist (run before finishing)

```bash
ruff format .        # 1. formatting
ruff check .         # 2. linting
mypy                 # 3. type checking (strict)
pytest               # 4. full test suite
```

Additionally, inspect the final diff and confirm:
- No accidental secrets or credentials.
- No unnecessary or scratch files.
- Documentation still matches the implementation.
- No unrelated changes were introduced.

---

## Explicitly out of scope for the current phase

Do **not** add, even as a "quick win":

- Real model-provider adapters (OpenAI/Anthropic/… calls).
- Real tools with side effects (shell, filesystem writes, network).
- Computer control, browser automation, or desktop operation.
- Real microphone capture, real/cloud STT or TTS providers, external voice
  services, image/video generation, or media pipelines. Phase 8 permits only
  the bounded provider-neutral voice foundation and deterministic local fakes.
- A user interface or HTTP/WebSocket API server.

These are tracked as future phases in [ROADMAP.md](ROADMAP.md). Premature
implementations violate rule 15.

---

## Status labels

Documentation in this repo must use exactly these three labels so the state of
every feature is unambiguous:

- **IMPLEMENTED** — code exists, is wired in, and is tested.
- **PLANNED** — designed and on the roadmap, not yet built.
- **NOT IMPLEMENTED** — desired, but with no design or code yet.

If you change code, update the labels in
[ARCHITECTURE.md](ARCHITECTURE.md) and [ROADMAP.md](ROADMAP.md) to stay truthful.
