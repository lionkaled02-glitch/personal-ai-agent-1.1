# Personal AI Agent 1.0.0

A modular local personal AI agent for Windows with planning, tools, documents, memory, browser/computer automation, coding, creation, approvals, verification and durable task state.

## Included

- **Model gateway:** mock/offline provider plus OpenAI-compatible provider and streaming.
- **Agent core:** planner → permission check → approval → tool execution → verification → durable checkpoint.
- **Safety:** workspace boundary, bounded inputs/outputs, HIGH-risk approval, no unrestricted shell, prompt-injection boundary for retrieved content, sensitive-field redaction.
- **Tasks:** SQLite persistence, idempotency keys, cancellation, resume, worker lifecycle, resource locks, audit/event cursor.
- **Knowledge:** TXT/Markdown/PDF/DOCX/PPTX/XLSX parsing, chunking, retrieval, durable SQLite index.
- **Memory:** durable SQLite memory with TTL/soft-delete and RAG context.
- **Computer:** optional Windows UI Automation provider.
- **Browser:** optional Playwright provider with bounded sessions/actions and sensitive-input protections.
- **Vision:** bounded screenshot analysis/comparison and verification.
- **Voice:** provider-neutral STT/TTS runtime and transport.
- **Coding:** project analysis, search/navigation, diagnostics and safe patch workflow.
- **Creation:** scripts, storyboards, subtitles and media provider boundaries.
- **Application:** FastAPI + WebSocket + Arabic RTL web UI, approvals, task monitoring and document index/search API.

## Optional integrations

The safe default is offline/mock mode. Enable integrations explicitly in `.env`:

```text
MODEL_PROVIDER=mock
COMPUTER_PROVIDER=none
BROWSER_PROVIDER=none
```

For Windows computer control use `COMPUTER_PROVIDER=windows` and install the optional Windows dependencies. For browser automation use `BROWSER_PROVIDER=playwright` and install Playwright plus its browser separately.

## Windows

Use the supplied PowerShell scripts:

- `scripts/install_windows.ps1`
- `scripts/test_windows.ps1`
- `scripts/run_server.ps1`

The final external validation depends on the actual Windows machine, installed applications, optional browser binaries, API credentials, microphone/audio devices and media tools.
