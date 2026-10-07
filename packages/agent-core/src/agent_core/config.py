"""Environment-based configuration.

The core reads plain environment variables via :meth:`Settings.from_env`.
Settings holds **non-secret** values only. Provider *credentials*
(``OPENAI_API_KEY``) are read from the environment by the provider factory
at construction time — never stored in Settings and never logged
(see SECURITY.md).

Variables:

- ``AGENT_NAME`` (str, default ``personal-agent``)
- ``LOG_LEVEL`` (str, default ``INFO``)
- ``DATA_ROOT`` (path, default ``data``)
- ``MODEL_PROVIDER`` (str, default ``mock`` — provider selection, Phase 1)
- ``MODEL_NAME`` (str, default ``""`` — model for the selected provider)
- ``MODEL_TIMEOUT_S`` (float, default ``60`` — provider request timeout)
- ``MODEL_MAX_RETRIES`` (int, default ``2`` — gateway retries after the
  first attempt for transient failures)
- ``WORKSPACE_ROOT`` (path, default ``data/workspace`` — root of the
  workspace boundary for the filesystem tools, Phase 3)
- ``WORKSPACE_MAX_READ_BYTES`` (int, default ``1048576`` — largest file the
  read/copy tools will process)
- ``WORKSPACE_MAX_WRITE_BYTES`` (int, default ``1048576`` — largest
  content the write tool will accept)
- ``WORKSPACE_MAX_LIST_ENTRIES`` (int, default ``500`` — listing cap)
- ``WORKSPACE_MAX_SEARCH_RESULTS`` (int, default ``200`` — search cap)
- ``WORKSPACE_MAX_PATH_LENGTH`` (int, default ``512`` — max resolved
  workspace-relative path length)
- ``DOCUMENT_MAX_INPUT_BYTES`` (int, default ``10485760`` — max raw document
  size, Phase 4)
- ``DOCUMENT_MAX_EXTRACTED_CHARS`` (int, default ``500000`` — total
  extracted text budget)
- ``DOCUMENT_MAX_PAGES`` / ``DOCUMENT_MAX_SLIDES`` /
  ``DOCUMENT_MAX_SHEETS`` (int, defaults ``200`` / ``100`` / ``20`` —
  container caps for PDF / PPTX / XLSX)
- ``DOCUMENT_MAX_SECTIONS`` (int, default ``500`` — section cap)
- ``DOCUMENT_MAX_CHUNKS`` (int, default ``500`` — chunk count cap)
- ``DOCUMENT_CHUNK_SIZE`` (int, default ``800`` — max chunk characters)
- ``DOCUMENT_CHUNK_OVERLAP`` (int, default ``100`` — chunk overlap)
- ``DOCUMENT_MAX_SEARCH_RESULTS`` (int, default ``10`` — retrieval cap)
- ``DOCUMENT_MAX_QUERY_CHARS`` (int, default ``500`` — max query length)
- ``MEMORY_MAX_ITEMS`` (int, default ``1000`` — max stored memories, Phase 5)
- ``MEMORY_MAX_CONTENT_CHARS`` (int, default ``4000`` — max memory content
  length; also bounds recall/context queries)
- ``MEMORY_MAX_METADATA_BYTES`` (int, default ``4096`` — max serialized
  metadata size per memory)
- ``MEMORY_MAX_RECALL_RESULTS`` (int, default ``10`` — recall/list cap)
- ``MEMORY_MAX_CONTEXT_CHARS`` (int, default ``8000`` — RAG context budget)
- ``MEMORY_MAX_CONTEXT_ITEMS`` (int, default ``20`` — RAG context item cap)
- ``MEMORY_SHORT_TERM_TTL_S`` (int, default ``3600`` — short_term TTL)
- ``MEMORY_WORKING_TTL_S`` (int, default ``86400`` — working TTL)
- ``COMPUTER_MAX_ACTIONS_PER_TASK`` (int, default ``20``)
- ``COMPUTER_ACTION_TIMEOUT_S`` (float, default ``5``)
- ``COMPUTER_MAX_TEXT_INPUT_CHARS`` (int, default ``256``)
- ``COMPUTER_MAX_SCREENSHOT_BYTES`` (int, default ``1048576``)
- ``COMPUTER_MAX_WINDOWS`` (int, default ``50``)
- ``COMPUTER_MAX_UI_ELEMENTS`` (int, default ``100``)
- ``COMPUTER_MAX_RETRIES`` (int, default ``1``)
- ``COMPUTER_MOUSE_MOVE_DURATION_S`` (float, default ``0.5``)
- ``COMPUTER_CURSOR_TOLERANCE_PX`` (int, default ``2``)
- ``VISION_MAX_IMAGE_BYTES`` / ``VISION_MAX_IMAGE_WIDTH`` /
  ``VISION_MAX_IMAGE_HEIGHT`` / ``VISION_MAX_IMAGE_PIXELS`` (int; screenshot
  and decode limits, Phase 7)
- ``VISION_MAX_REGIONS`` / ``VISION_MAX_LABEL_CHARS`` /
  ``VISION_MAX_SUMMARY_CHARS`` (int; provider output bounds)
- ``VISION_MAX_COMPARISON_PIXELS`` (int; deterministic pixel-work cap)
- ``VISION_MAX_OBSERVATION_RETRIES`` (int; screenshot-only uncertainty refreshes)
- ``VISION_MAX_OPERATION_SECONDS`` (float; cooperative elapsed-time limit)
- ``VOICE_MAX_AUDIO_BYTES`` / ``VOICE_MAX_DURATION_S`` (input/output bounds)
- ``VOICE_MAX_TEXT_CHARS`` / ``VOICE_MAX_LANGUAGE_CHARS`` /
  ``VOICE_MAX_VOICE_NAME_CHARS`` / ``VOICE_MAX_SYNTHESIS_TEXT_CHARS``
- ``VOICE_MAX_TRANSCRIPTION_TIME_S`` / ``VOICE_MAX_SYNTHESIS_TIME_S`` /
  ``VOICE_MAX_RETRIES`` / ``VOICE_MIN_TRANSCRIPTION_CONFIDENCE``
- ``CODING_MAX_PROJECT_FILES`` / ``CODING_MAX_FILE_SIZE_BYTES`` /
  ``CODING_MAX_SOURCE_CHARS`` / ``CODING_MAX_PATCH_SIZE_BYTES`` /
  ``CODING_MAX_CHANGED_FILES`` / ``CODING_MAX_SYMBOLS`` /
  ``CODING_MAX_REGIONS`` / ``CODING_MAX_DIAGNOSTICS`` /
  ``CODING_MAX_ANALYSIS_TIME_S`` / ``CODING_MAX_TEST_DURATION_S`` /
  ``CODING_MAX_OUTPUT_BYTES`` (Phase 10 bounded coding foundation)

Voice time limits are cooperative because synchronous provider calls cannot
be forcibly interrupted. Phase 8 adds no microphone hardware, API keys, or
external STT/TTS provider. Phase 10 adds bounded coding data contracts, a
local mock, read-only analysis, and deterministic diagnostics; it adds no
source writes, command execution, or real provider.

``DATA_ROOT`` holds logs and task artifacts; it is independent of the
workspace boundary, which the Phase 3 filesystem tools enforce strictly.
Document tools (Phase 4) read only through that same boundary.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel


class Settings(BaseModel):
    agent_name: str = "personal-agent"
    log_level: str = "INFO"
    data_root: Path = Path("data")
    # Model gateway (Phase 1) — non-secret provider configuration.
    model_provider: str = "mock"
    model_name: str = ""
    model_timeout_s: float = 60.0
    model_max_retries: int = 2
    computer_provider: str = "none"
    browser_provider: str = "none"
    # Workspace filesystem tools (Phase 3) — boundary + limits.
    workspace_root: Path = Path("data/workspace")
    workspace_max_read_bytes: int = 1_048_576
    workspace_max_write_bytes: int = 1_048_576
    workspace_max_list_entries: int = 500
    workspace_max_search_results: int = 200
    workspace_max_path_length: int = 512
    # Document processing & knowledge foundation (Phase 4) — limits.
    document_max_input_bytes: int = 10_485_760
    document_max_extracted_chars: int = 500_000
    document_max_pages: int = 200
    document_max_slides: int = 100
    document_max_sheets: int = 20
    document_max_sections: int = 500
    document_max_chunks: int = 500
    document_chunk_size: int = 800
    document_chunk_overlap: int = 100
    document_max_search_results: int = 10
    document_max_query_chars: int = 500
    # Memory layer (Phase 5) — limits + policy.
    memory_max_items: int = 1_000
    memory_max_content_chars: int = 4_000
    memory_max_metadata_bytes: int = 4_096
    memory_max_recall_results: int = 10
    memory_max_context_chars: int = 8_000
    memory_max_context_items: int = 20
    memory_short_term_ttl_s: int = 3_600
    memory_working_ttl_s: int = 86_400
    # Computer Agent Foundation (Phase 6) — conservative runtime bounds.
    computer_max_actions_per_task: int = 20
    computer_action_timeout_s: float = 5.0
    computer_max_text_input_chars: int = 256
    computer_max_screenshot_bytes: int = 1_048_576
    computer_max_windows: int = 50
    computer_max_ui_elements: int = 100
    computer_max_retries: int = 1
    computer_mouse_move_duration_s: float = 0.5
    computer_cursor_tolerance_px: int = 2
    # Vision & visual verification (Phase 7) — local processing bounds only.
    vision_max_image_bytes: int = 1_048_576
    vision_max_image_width: int = 4_096
    vision_max_image_height: int = 4_096
    vision_max_image_pixels: int = 16_777_216
    vision_max_regions: int = 100
    vision_max_label_chars: int = 128
    vision_max_summary_chars: int = 512
    vision_max_comparison_pixels: int = 1_048_576
    vision_max_observation_retries: int = 1
    vision_max_operation_seconds: float = 5.0
    # Voice Agent Foundation (Phase 8) — provider-neutral bounds only.
    voice_max_audio_bytes: int = 1_048_576
    voice_max_duration_s: float = 30.0
    voice_max_text_chars: int = 4_000
    voice_max_language_chars: int = 35
    voice_max_voice_name_chars: int = 64
    voice_max_synthesis_text_chars: int = 2_000
    voice_max_transcription_time_s: float = 15.0
    voice_max_synthesis_time_s: float = 15.0
    voice_max_retries: int = 1
    voice_min_transcription_confidence: float = 0.6
    # Browser Agent Foundation (Phase 9) — provider-neutral bounds only.
    browser_max_url_chars: int = 2_048
    browser_max_title_chars: int = 256
    browser_max_text_chars: int = 4_000
    browser_max_elements: int = 100
    browser_max_element_text_chars: int = 256
    browser_max_attributes: int = 12
    browser_max_attribute_chars: int = 128
    browser_max_fill_chars: int = 1_024
    browser_max_screenshot_bytes: int = 1_048_576
    browser_max_sessions: int = 5
    browser_max_pages_per_session: int = 10
    browser_max_navigation_time_s: float = 10.0
    browser_max_action_time_s: float = 5.0
    browser_max_wait_time_s: float = 5.0
    browser_max_retries: int = 1
    # Coding Agent foundation (Phase 10, Steps 1-3) — data/plan/analysis/diagnostic bounds.
    coding_max_project_files: int = 100
    coding_max_file_size_bytes: int = 524_288
    coding_max_source_chars: int = 500_000
    coding_max_patch_size_bytes: int = 262_144
    coding_max_changed_files: int = 20
    coding_max_symbols: int = 200
    coding_max_regions: int = 100
    coding_max_diagnostics: int = 100
    coding_max_analysis_time_s: float = 20.0
    coding_max_test_duration_s: float = 120.0
    coding_max_output_bytes: int = 262_144
    # Human approval broker bounds.
    approval_timeout_s: float = 300.0
    approval_max_pending: int = 32

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        """Build settings from an environment mapping (defaults: os.environ)."""
        source: Mapping[str, str] = os.environ if env is None else env
        defaults = cls()
        return cls(
            agent_name=source.get("AGENT_NAME", defaults.agent_name),
            log_level=source.get("LOG_LEVEL", defaults.log_level),
            data_root=Path(source.get("DATA_ROOT", str(defaults.data_root))),
            model_provider=source.get("MODEL_PROVIDER", defaults.model_provider),
            model_name=source.get("MODEL_NAME", defaults.model_name),
            model_timeout_s=float(source.get("MODEL_TIMEOUT_S", defaults.model_timeout_s)),
            model_max_retries=int(source.get("MODEL_MAX_RETRIES", defaults.model_max_retries)),
            computer_provider=source.get("COMPUTER_PROVIDER", defaults.computer_provider)
            .strip()
            .lower(),
            browser_provider=source.get("BROWSER_PROVIDER", defaults.browser_provider)
            .strip()
            .lower(),
            workspace_root=Path(source.get("WORKSPACE_ROOT", str(defaults.workspace_root))),
            workspace_max_read_bytes=int(
                source.get("WORKSPACE_MAX_READ_BYTES", defaults.workspace_max_read_bytes)
            ),
            workspace_max_write_bytes=int(
                source.get("WORKSPACE_MAX_WRITE_BYTES", defaults.workspace_max_write_bytes)
            ),
            workspace_max_list_entries=int(
                source.get("WORKSPACE_MAX_LIST_ENTRIES", defaults.workspace_max_list_entries)
            ),
            workspace_max_search_results=int(
                source.get("WORKSPACE_MAX_SEARCH_RESULTS", defaults.workspace_max_search_results)
            ),
            workspace_max_path_length=int(
                source.get("WORKSPACE_MAX_PATH_LENGTH", defaults.workspace_max_path_length)
            ),
            document_max_input_bytes=int(
                source.get("DOCUMENT_MAX_INPUT_BYTES", defaults.document_max_input_bytes)
            ),
            document_max_extracted_chars=int(
                source.get("DOCUMENT_MAX_EXTRACTED_CHARS", defaults.document_max_extracted_chars)
            ),
            document_max_pages=int(source.get("DOCUMENT_MAX_PAGES", defaults.document_max_pages)),
            document_max_slides=int(
                source.get("DOCUMENT_MAX_SLIDES", defaults.document_max_slides)
            ),
            document_max_sheets=int(
                source.get("DOCUMENT_MAX_SHEETS", defaults.document_max_sheets)
            ),
            document_max_sections=int(
                source.get("DOCUMENT_MAX_SECTIONS", defaults.document_max_sections)
            ),
            document_max_chunks=int(
                source.get("DOCUMENT_MAX_CHUNKS", defaults.document_max_chunks)
            ),
            document_chunk_size=int(
                source.get("DOCUMENT_CHUNK_SIZE", defaults.document_chunk_size)
            ),
            document_chunk_overlap=int(
                source.get("DOCUMENT_CHUNK_OVERLAP", defaults.document_chunk_overlap)
            ),
            document_max_search_results=int(
                source.get("DOCUMENT_MAX_SEARCH_RESULTS", defaults.document_max_search_results)
            ),
            document_max_query_chars=int(
                source.get("DOCUMENT_MAX_QUERY_CHARS", defaults.document_max_query_chars)
            ),
            memory_max_items=int(source.get("MEMORY_MAX_ITEMS", defaults.memory_max_items)),
            memory_max_content_chars=int(
                source.get("MEMORY_MAX_CONTENT_CHARS", defaults.memory_max_content_chars)
            ),
            memory_max_metadata_bytes=int(
                source.get("MEMORY_MAX_METADATA_BYTES", defaults.memory_max_metadata_bytes)
            ),
            memory_max_recall_results=int(
                source.get("MEMORY_MAX_RECALL_RESULTS", defaults.memory_max_recall_results)
            ),
            memory_max_context_chars=int(
                source.get("MEMORY_MAX_CONTEXT_CHARS", defaults.memory_max_context_chars)
            ),
            memory_max_context_items=int(
                source.get("MEMORY_MAX_CONTEXT_ITEMS", defaults.memory_max_context_items)
            ),
            memory_short_term_ttl_s=int(
                source.get("MEMORY_SHORT_TERM_TTL_S", defaults.memory_short_term_ttl_s)
            ),
            memory_working_ttl_s=int(
                source.get("MEMORY_WORKING_TTL_S", defaults.memory_working_ttl_s)
            ),
            computer_max_actions_per_task=int(
                source.get("COMPUTER_MAX_ACTIONS_PER_TASK", defaults.computer_max_actions_per_task)
            ),
            computer_action_timeout_s=float(
                source.get("COMPUTER_ACTION_TIMEOUT_S", defaults.computer_action_timeout_s)
            ),
            computer_max_text_input_chars=int(
                source.get("COMPUTER_MAX_TEXT_INPUT_CHARS", defaults.computer_max_text_input_chars)
            ),
            computer_max_screenshot_bytes=int(
                source.get("COMPUTER_MAX_SCREENSHOT_BYTES", defaults.computer_max_screenshot_bytes)
            ),
            computer_max_windows=int(
                source.get("COMPUTER_MAX_WINDOWS", defaults.computer_max_windows)
            ),
            computer_max_ui_elements=int(
                source.get("COMPUTER_MAX_UI_ELEMENTS", defaults.computer_max_ui_elements)
            ),
            computer_max_retries=int(
                source.get("COMPUTER_MAX_RETRIES", defaults.computer_max_retries)
            ),
            computer_mouse_move_duration_s=float(
                source.get(
                    "COMPUTER_MOUSE_MOVE_DURATION_S", defaults.computer_mouse_move_duration_s
                )
            ),
            computer_cursor_tolerance_px=int(
                source.get("COMPUTER_CURSOR_TOLERANCE_PX", defaults.computer_cursor_tolerance_px)
            ),
            vision_max_image_bytes=int(
                source.get("VISION_MAX_IMAGE_BYTES", defaults.vision_max_image_bytes)
            ),
            vision_max_image_width=int(
                source.get("VISION_MAX_IMAGE_WIDTH", defaults.vision_max_image_width)
            ),
            vision_max_image_height=int(
                source.get("VISION_MAX_IMAGE_HEIGHT", defaults.vision_max_image_height)
            ),
            vision_max_image_pixels=int(
                source.get("VISION_MAX_IMAGE_PIXELS", defaults.vision_max_image_pixels)
            ),
            vision_max_regions=int(source.get("VISION_MAX_REGIONS", defaults.vision_max_regions)),
            vision_max_label_chars=int(
                source.get("VISION_MAX_LABEL_CHARS", defaults.vision_max_label_chars)
            ),
            vision_max_summary_chars=int(
                source.get("VISION_MAX_SUMMARY_CHARS", defaults.vision_max_summary_chars)
            ),
            vision_max_comparison_pixels=int(
                source.get("VISION_MAX_COMPARISON_PIXELS", defaults.vision_max_comparison_pixels)
            ),
            vision_max_observation_retries=int(
                source.get(
                    "VISION_MAX_OBSERVATION_RETRIES", defaults.vision_max_observation_retries
                )
            ),
            vision_max_operation_seconds=float(
                source.get("VISION_MAX_OPERATION_SECONDS", defaults.vision_max_operation_seconds)
            ),
            voice_max_audio_bytes=int(
                source.get("VOICE_MAX_AUDIO_BYTES", defaults.voice_max_audio_bytes)
            ),
            voice_max_duration_s=float(
                source.get("VOICE_MAX_DURATION_S", defaults.voice_max_duration_s)
            ),
            voice_max_text_chars=int(
                source.get("VOICE_MAX_TEXT_CHARS", defaults.voice_max_text_chars)
            ),
            voice_max_language_chars=int(
                source.get("VOICE_MAX_LANGUAGE_CHARS", defaults.voice_max_language_chars)
            ),
            voice_max_voice_name_chars=int(
                source.get("VOICE_MAX_VOICE_NAME_CHARS", defaults.voice_max_voice_name_chars)
            ),
            voice_max_synthesis_text_chars=int(
                source.get(
                    "VOICE_MAX_SYNTHESIS_TEXT_CHARS", defaults.voice_max_synthesis_text_chars
                )
            ),
            voice_max_transcription_time_s=float(
                source.get(
                    "VOICE_MAX_TRANSCRIPTION_TIME_S", defaults.voice_max_transcription_time_s
                )
            ),
            voice_max_synthesis_time_s=float(
                source.get("VOICE_MAX_SYNTHESIS_TIME_S", defaults.voice_max_synthesis_time_s)
            ),
            voice_max_retries=int(source.get("VOICE_MAX_RETRIES", defaults.voice_max_retries)),
            voice_min_transcription_confidence=float(
                source.get(
                    "VOICE_MIN_TRANSCRIPTION_CONFIDENCE",
                    defaults.voice_min_transcription_confidence,
                )
            ),
            browser_max_url_chars=int(
                source.get("BROWSER_MAX_URL_CHARS", defaults.browser_max_url_chars)
            ),
            browser_max_title_chars=int(
                source.get("BROWSER_MAX_TITLE_CHARS", defaults.browser_max_title_chars)
            ),
            browser_max_text_chars=int(
                source.get("BROWSER_MAX_TEXT_CHARS", defaults.browser_max_text_chars)
            ),
            browser_max_elements=int(
                source.get("BROWSER_MAX_ELEMENTS", defaults.browser_max_elements)
            ),
            browser_max_element_text_chars=int(
                source.get(
                    "BROWSER_MAX_ELEMENT_TEXT_CHARS", defaults.browser_max_element_text_chars
                )
            ),
            browser_max_attributes=int(
                source.get("BROWSER_MAX_ATTRIBUTES", defaults.browser_max_attributes)
            ),
            browser_max_attribute_chars=int(
                source.get("BROWSER_MAX_ATTRIBUTE_CHARS", defaults.browser_max_attribute_chars)
            ),
            browser_max_fill_chars=int(
                source.get("BROWSER_MAX_FILL_CHARS", defaults.browser_max_fill_chars)
            ),
            browser_max_screenshot_bytes=int(
                source.get("BROWSER_MAX_SCREENSHOT_BYTES", defaults.browser_max_screenshot_bytes)
            ),
            browser_max_sessions=int(
                source.get("BROWSER_MAX_SESSIONS", defaults.browser_max_sessions)
            ),
            browser_max_pages_per_session=int(
                source.get("BROWSER_MAX_PAGES_PER_SESSION", defaults.browser_max_pages_per_session)
            ),
            browser_max_navigation_time_s=float(
                source.get("BROWSER_MAX_NAVIGATION_TIME_S", defaults.browser_max_navigation_time_s)
            ),
            browser_max_action_time_s=float(
                source.get("BROWSER_MAX_ACTION_TIME_S", defaults.browser_max_action_time_s)
            ),
            browser_max_wait_time_s=float(
                source.get("BROWSER_MAX_WAIT_TIME_S", defaults.browser_max_wait_time_s)
            ),
            browser_max_retries=int(
                source.get("BROWSER_MAX_RETRIES", defaults.browser_max_retries)
            ),
            coding_max_project_files=int(
                source.get("CODING_MAX_PROJECT_FILES", defaults.coding_max_project_files)
            ),
            coding_max_file_size_bytes=int(
                source.get("CODING_MAX_FILE_SIZE_BYTES", defaults.coding_max_file_size_bytes)
            ),
            coding_max_source_chars=int(
                source.get("CODING_MAX_SOURCE_CHARS", defaults.coding_max_source_chars)
            ),
            coding_max_patch_size_bytes=int(
                source.get("CODING_MAX_PATCH_SIZE_BYTES", defaults.coding_max_patch_size_bytes)
            ),
            coding_max_changed_files=int(
                source.get("CODING_MAX_CHANGED_FILES", defaults.coding_max_changed_files)
            ),
            coding_max_symbols=int(source.get("CODING_MAX_SYMBOLS", defaults.coding_max_symbols)),
            coding_max_regions=int(source.get("CODING_MAX_REGIONS", defaults.coding_max_regions)),
            coding_max_diagnostics=int(
                source.get("CODING_MAX_DIAGNOSTICS", defaults.coding_max_diagnostics)
            ),
            coding_max_analysis_time_s=float(
                source.get("CODING_MAX_ANALYSIS_TIME_S", defaults.coding_max_analysis_time_s)
            ),
            coding_max_test_duration_s=float(
                source.get("CODING_MAX_TEST_DURATION_S", defaults.coding_max_test_duration_s)
            ),
            coding_max_output_bytes=int(
                source.get("CODING_MAX_OUTPUT_BYTES", defaults.coding_max_output_bytes)
            ),
            approval_timeout_s=float(source.get("APPROVAL_TIMEOUT_S", defaults.approval_timeout_s)),
            approval_max_pending=int(
                source.get("APPROVAL_MAX_PENDING", defaults.approval_max_pending)
            ),
        )

    def configure_logging(self) -> None:
        """Configure the root logger to the configured level."""
        logging.basicConfig(level=self.log_level.upper())
