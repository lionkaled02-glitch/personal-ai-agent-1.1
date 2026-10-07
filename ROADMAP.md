# Roadmap

## Completed application milestone — 1.0.0

The core architecture and the local application integration are implemented: planning, execution, permissions, approvals, durable state, memory, document knowledge, browser/computer adapters, vision, voice contracts, coding, creation, API and UI.

## Runtime validation after installation

These are environment-dependent validation steps rather than missing core architecture:

1. Windows UI Automation with `COMPUTER_PROVIDER=windows`.
2. Playwright/Chromium with `BROWSER_PROVIDER=playwright`.
3. A real model provider such as OpenAI with its API key.
4. Real microphone/STT/TTS provider configuration if voice is enabled.
5. FFmpeg/media rendering where enabled.

The application remains fail-safe: optional integrations are disabled by default, sensitive/destructive operations require approval, and untrusted web/document content is never treated as instructions.
