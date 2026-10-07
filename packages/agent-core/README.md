# agent-core

The only importable package in this repository. It contains the Phase 0
foundation, the Phase 1 model layer, the Phase 2 tool layer, the Phase 3
workspace layer, the Phase 4 document layer, the Phase 5 memory & RAG layer,
the Phase 6 safe computer foundation, the Phase 7 Vision & Visual
Verification layer, the Phase 8 Voice Agent Foundation, the Phase 9
Browser Agent Foundation, and the Phase 10 Coding Agent core foundation of
the personal AI agent. It includes the task state model,
planner, executor, tool system with registry, permission manager, structured
event bus, a
vendor-neutral `ModelProvider` interface with a `ModelGateway` (error
normalization + safe retry), a configuration-driven provider factory, a
deterministic mock provider (offline default), an OpenAI provider adapter
(optional `openai` extra, lazy SDK import), the **Tool Runtime** —
permission-gated, schema-validated, metadata-carrying tool execution — safe,
deterministic built-in tools (calculator, date/time, text utils, JSON
utils), the **workspace filesystem layer** (a fail-closed `Workspace`
boundary plus nine scoped, permission-gated file tools; no shell or
subprocess), the **document processing & knowledge foundation**:

- **Parsers** — a `DocumentParser` interface + registry with built-ins for
  TXT, Markdown, PDF (pypdf), DOCX (python-docx), PPTX (python-pptx), and
  XLSX (openpyxl). The four binary parsers use an optional `docs` extra and
  import their library lazily (missing library ⇒ structured
  `parser_unavailable`). Parsers extract text/structure only — they never
  execute anything a document contains, and document text is never treated
  as instructions.
- **Normalized model** — `Document` / `DocumentSection` / `DocumentChunk`
  (Pydantic, deterministic ids, serializable) with page/slide/sheet
  locations, headings, deterministic table rendering, extraction warnings,
  stats, and an explicit truncation flag.
- **Chunking** — deterministic, bounded (configurable size + overlap,
  count-capped, metadata-preserving). No embeddings.
- **Retrieval** — a provider-neutral, deterministic **lexical**
  `KnowledgeStore` behind the `RetrievalIndex` protocol (a future vector
  store implements the same protocol). No external model.
- **Tools** — `inspect_document` (LOW), `extract_document` (LOW),
  `index_document` (MEDIUM), `search_documents` (LOW), all run through the
  Tool Runtime and read only through the Phase 3 `Workspace` boundary.

and the **memory & RAG foundation** (Phase 5):

- **Memory model** — `Memory` (Pydantic, deterministic `mem-…` ids):
  lifecycle types (`short_term`/`working`/`long_term`/`knowledge`),
  provenance (`user_explicit`/`task`/`agent`/`document`/`system` + source
  ref), confidence, timestamps, expiration, soft-delete flag.
- **Storage** — `MemoryStore` protocol + `InMemoryMemoryStore`:
  remember/get/update/forget/list/recall/purge_expired; policy enforced at
  the store layer (`MemoryLimits` from `MEMORY_*` settings); deterministic
  ordering; `forget` soft-deactivates by default (hard delete opt-in). No
  external database, no network.
- **Retrieval** — provider-neutral, deterministic **lexical**
  `MemoryRetriever` (`LexicalMemoryRetriever`; no embeddings, no external
  model) — the seam a future semantic/vector retriever implements.
- **Tools** — `remember` (MEDIUM), `update_memory` (MEDIUM), `forget`
  (HIGH), `recall` (LOW), `list_memories` (LOW), all through the Tool
  Runtime; creation is explicit (never automatic); content is untrusted
  data, never instructions.
- **RAG context** — `ContextBuilder` combines retrieved memories + document
  chunks into a structured, bounded, provenance-labeled `Context`
  (MEMORY vs DOCUMENT items, locations, explicit omission reporting).
  Assembly only — it never generates answers.
- **Safe Computer Agent foundation (Phase 6)** — provider-neutral
  `ComputerProvider`, bounded observation/action models and runtime, strict
  verification/recovery, structured events, and opt-in tool registration.
  Six LOW observation tools and eight explicit MEDIUM interaction tools use
  the existing permission system; there is no generic arbitrary-action tool.
  `WindowsComputerProvider` is an optional, Windows-only UI Automation
  adapter. UI/window text is untrusted data; screenshots and typed text are
  sensitive and event-redacted; screenshots are bounded and never persisted
  by the core. Timeouts are cooperative (synchronous OS calls cannot be
  forcibly interrupted).
- **Vision & Visual Verification (Phase 7)** — provider-neutral Pydantic
  models and a deterministic offline provider reuse the Phase 6 screenshot
  path for metadata-only observation and bounded local RGB comparisons. The
  `vision_analyze_screenshot` tool is registered only with an explicit
  computer provider and returns no screenshot bytes. Visual results use
  `VERIFIED` / `FAILED` / `UNCERTAIN`; a pixel difference is never proof of a
  click or other semantic action, and a successful action still requires a
  non-screenshot Phase 6 postcondition. On uncertainty, only bounded
  screenshot refreshes can retry—actions are never blindly repeated. No OCR,
  semantic vision model, remote API/cloud upload, or persistent screenshot
  storage is included.
- **Voice Agent Foundation (Phase 8)** — provider-neutral `STTProvider` /
  `TTSProvider` protocols; strict bounded PCM, transcription, synthesis,
  observation, and response models; deterministic local mocks; NFC and
  whitespace normalization; explicit confidence/uncertainty; safe
  `VOICE_*` limits, cooperative timeouts, and bounded retry. A confident
  transcript is handed once to the existing `Agent.run` flow, so voice cannot
  bypass HIGH/MEDIUM permissions or approvals. Voice events contain metadata
  only; audio input/output is ephemeral and excluded from repr/serialization.
  Use `VoiceRuntime.from_settings(provider, settings=Settings.from_env())` to
  apply the validated `VOICE_*` bounds. No microphone hardware, external/cloud
  provider, API key, or audio storage is included. The mock TTS output is
  silent PCM test data, not speech.

- **Browser Agent Foundation (Phase 9)** — provider-neutral `BrowserProvider`,
  strict bounded models/limits, `BrowserRuntime`, safe HTTP(S) URL validation,
  redaction, verification/recovery, deterministic offline mock, and optional
  lazy Playwright adapter. Nineteen fixed tools are registered only when an
  explicit provider is supplied to `Agent.create_demo` or
  `Agent.create_configured`. Each operation requires explicit session/page
  IDs; no active tab is inferred. Observation/read/wait tools are LOW,
  ordinary navigation and interaction are generally MEDIUM, and form submits
  or externally consequential controls require HIGH approval. Sensitive
  controls cannot be filled/selected; no raw HTML, screenshot bytes, or page
  text is sent in normal events. `VERIFIED`/`FAILED`/`UNCERTAIN` results are
  preserved, and uncertain actions do not become tool success. Page content
  is untrusted data, never instructions. This is not unrestricted autonomous
  browsing, CAPTCHA/anti-bot bypass, credential harvesting, arbitrary
  JavaScript, or profile reuse.
- **Coding Agent core foundation (Phase 10, Steps 1–3)** — `agent_core.coding`
  provides strict bounded project/file/region, analysis, diagnostic, structural
  patch, test-plan, and metadata-only observation models; a provider-neutral
  `CodingProvider`; `CODING_*` limits; content-safe errors; and an offline
  deterministic mock. `CodingAnalysisRuntime` performs bounded read-only
  discovery through `Workspace`; `CodeDiagnosticsEngine` adds fixed, bounded
  Python, JavaScript, and TypeScript syntax/style findings from the validated
  snapshots. Python uses AST parsing without execution; JavaScript/TypeScript
  diagnostics are shallow and rule-based; JSON/TOML/Markdown/YAML remain
  metadata-only. Findings carry stable code/category, severity, path, and
  optional line/column. Repository/config/source text remains untrusted data.
  No patch is applied; Steps 2–3 add no shell/process, compiler/build/test
  execution, package installation, arbitrary code execution, automatic writes,
  network requests, or real-provider integration.

To opt in with the Windows provider, install the extra on Windows and pass
an explicit provider. MEDIUM interactions are denied unless the existing
approval callback returns `True`:

```bash
pip install -e "packages/agent-core[computer-windows]"
```

```python
from agent_core import Agent, WindowsComputerProvider

agent = Agent.create_configured(
    approval=your_approval_callback,
    computer_provider=WindowsComputerProvider(),
)
```

Without an explicit provider, browser, computer, and vision tools are not
registered. With a computer provider, `vision_analyze_screenshot` reuses its
permission-gated Phase 6 screenshot path. Metadata-only analysis needs no image
extra; local comparison of non-identical images uses the optional
`vision-image` extra. The core remains importable off Windows; constructing the
Windows adapter elsewhere raises a structured `UnsupportedPlatformError`.

To opt in to Playwright, install its Python extra and Chromium separately;
core installation and offline CI do not install browser binaries:

```bash
pip install -e "packages/agent-core[browser-playwright]"
playwright install chromium
```

```python
from agent_core import Agent, PlaywrightBrowserProvider

agent = Agent.create_configured(
    approval=your_approval_callback,
    browser_provider=PlaywrightBrowserProvider(),
)
```

No provider launches until an explicit browser session is opened. The
Playwright adapter uses new ephemeral contexts, closes popups, disables
downloads, and does not load/save browser profiles. Its HTTP(S) URL checks
are syntactic; deployment egress/network controls should still restrict local
and private networks if that matters to the application.

## Extras

- `agent-core[openai]` — the OpenAI provider SDK.
- `agent-core[docs]` — the four document parser libraries (pypdf,
  python-docx, python-pptx, openpyxl). Without it, the binary parsers fail
  with a structured `parser_unavailable` error at parse time; TXT and
  Markdown always work.
- `agent-core[vision-image]` — optional Pillow support for local deterministic
  RGB comparison; imported lazily. Metadata-only observation and exact
  screenshot-payload matches work without it.
- `agent-core[browser-playwright]` — optional Playwright Python provider; it
  does not install Chromium. Run `playwright install chromium` separately only
  when you explicitly choose to use the real provider.
- `agent-core[computer-windows]` — Windows-only UI Automation dependencies
  (pywinauto, pywin32, Pillow), installed only on Windows. The module remains
  importable without this extra; provider construction then raises a
  structured `provider_unavailable` error.

- Architecture: see [`ARCHITECTURE.md`](../../ARCHITECTURE.md)
- Engineering rules: see [`AGENTS.md`](../../AGENTS.md)
- Security model: see [`SECURITY.md`](../../SECURITY.md)

No API keys or network access are required to use or test this package; a
real provider is opt-in via `MODEL_PROVIDER=openai` + `OPENAI_API_KEY`.
