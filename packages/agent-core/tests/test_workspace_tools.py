"""Workspace filesystem tools: behavior, limits, and security tests (Phase 3).

Covers all nine tools (list_directory, read_text_file, write_text_file,
create_directory, copy_file, move_file, delete_file, file_info,
search_files): normal operation, structured error codes, size/count limits,
path-escape rejection, permission behavior (LOW/MEDIUM/HIGH with approval),
and tool-runtime / event integration (including bounded event payloads and
no filesystem action after a denial).

All tests are deterministic and run in temporary workspaces; no network, no
external APIs.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from agent_core import (
    Agent,
    ApprovalCallback,
    EventBus,
    EventType,
    MockModelProvider,
    ModelPlanner,
    PermissionDecision,
    PermissionManager,
    PermissionPolicy,
    Settings,
    StepStatus,
    TaskState,
    Tool,
    ToolInvocation,
    ToolRegistry,
    ToolResult,
    ToolRuntime,
    Workspace,
    WorkspaceLimits,
    register_workspace_tools,
)
from conftest import FIXED_NOW

FIXED: datetime = FIXED_NOW

SmallLimits = WorkspaceLimits(
    max_read_bytes=64,
    max_write_bytes=64,
    max_list_entries=3,
    max_search_results=3,
    max_path_length=512,
)


@pytest.fixture
def workspace_root(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    root.mkdir()
    return root


@pytest.fixture
def workspace(workspace_root: Path) -> Workspace:
    return Workspace(workspace_root)


@pytest.fixture
def small_workspace(workspace_root: Path) -> Workspace:
    return Workspace(workspace_root, SmallLimits)


@pytest.fixture
def tools(small_workspace: Workspace) -> dict[str, Tool]:
    """All nine tools bound to a small-limit workspace, name-keyed."""
    registry: ToolRegistry = ToolRegistry()
    register_workspace_tools(registry, small_workspace)
    by_name: dict[str, Tool] = {}
    for spec in registry.list_tools():
        tool = registry.get(spec.name)
        assert tool is not None
        by_name[spec.name] = tool
    return by_name


def call(tools: dict[str, Tool], name: str, payload: dict[str, Any]) -> ToolResult:
    return tools[name].run(payload)


def out(result: ToolResult) -> dict[str, Any]:
    """Assert a successful structured output and return it as a dict."""
    assert result.ok, result.error
    assert isinstance(result.output, dict)
    return result.output


# ---------------------------------------------------------------------------
# write_text_file / read_text_file / file_info
# ---------------------------------------------------------------------------


class TestWriteTextFile:
    def test_creates_file(self, tools: dict[str, Tool], workspace_root: Path) -> None:
        result = call(tools, "write_text_file", {"path": "a.txt", "content": "hello"})
        assert result.ok
        assert result.output == {"path": "a.txt", "bytes_written": 5, "created": True}
        assert (workspace_root / "a.txt").read_text(encoding="utf-8") == "hello"

    def test_missing_parent_directory_is_path_not_found(self, tools: dict[str, Tool]) -> None:
        result = call(tools, "write_text_file", {"path": "missing/a.txt", "content": "x"})
        assert not result.ok and result.error_code == "path_not_found"

    def test_existing_without_overwrite_is_target_exists(
        self, tools: dict[str, Tool], workspace_root: Path
    ) -> None:
        (workspace_root / "a.txt").write_text("old", encoding="utf-8")
        result = call(tools, "write_text_file", {"path": "a.txt", "content": "new"})
        assert not result.ok and result.error_code == "target_exists"
        assert (workspace_root / "a.txt").read_text(encoding="utf-8") == "old"

    def test_overwrite_replaces_content(self, tools: dict[str, Tool], workspace_root: Path) -> None:
        (workspace_root / "a.txt").write_text("old", encoding="utf-8")
        result = call(
            tools, "write_text_file", {"path": "a.txt", "content": "newer", "overwrite": True}
        )
        assert result.ok
        assert out(result)["created"] is False
        assert (workspace_root / "a.txt").read_text(encoding="utf-8") == "newer"

    def test_content_too_large(self, tools: dict[str, Tool]) -> None:
        result = call(tools, "write_text_file", {"path": "big.txt", "content": "x" * 65})
        assert not result.ok and result.error_code == "content_too_large"

    def test_invalid_encoding(self, tools: dict[str, Tool]) -> None:
        result = call(
            tools,
            "write_text_file",
            {"path": "a.txt", "content": "x", "encoding": "no-such-charset"},
        )
        assert not result.ok and result.error_code == "invalid_encoding"

    def test_path_outside_workspace(self, tools: dict[str, Tool]) -> None:
        result = call(tools, "write_text_file", {"path": "../evil.txt", "content": "x"})
        assert not result.ok and result.error_code == "path_outside_workspace"

    def test_write_into_symlink_escape_rejected(
        self, tools: dict[str, Tool], workspace_root: Path, tmp_path: Path
    ) -> None:
        outside = tmp_path / "outside.txt"
        link = workspace_root / "out.txt"
        try:
            os.symlink(outside, link)
        except (OSError, NotImplementedError, ValueError):
            pytest.skip("symlinks not supported in this environment")
        result = call(tools, "write_text_file", {"path": "out.txt", "content": "x"})
        assert not result.ok and result.error_code == "path_outside_workspace"
        assert not outside.exists()

    def test_non_string_content_rejected(self, tools: dict[str, Tool]) -> None:
        result = call(tools, "write_text_file", {"path": "a.txt", "content": 123})
        assert not result.ok
        assert result.error_code == "invalid_content"
        assert result.output is None


class TestReadTextFile:
    def test_roundtrip(self, tools: dict[str, Tool]) -> None:
        call(tools, "write_text_file", {"path": "a.txt", "content": "héllo"})
        result = call(tools, "read_text_file", {"path": "a.txt"})
        assert result.ok
        # "héllo" is 6 bytes in UTF-8 (é is two bytes).
        assert result.output == {
            "path": "a.txt",
            "content": "héllo",
            "encoding": "utf-8",
            "size_bytes": 6,
        }

    def test_missing_file(self, tools: dict[str, Tool]) -> None:
        result = call(tools, "read_text_file", {"path": "nope.txt"})
        assert not result.ok and result.error_code == "path_not_found"

    def test_directory_is_not_a_file(self, tools: dict[str, Tool], workspace_root: Path) -> None:
        (workspace_root / "d").mkdir()
        result = call(tools, "read_text_file", {"path": "d"})
        assert not result.ok and result.error_code == "not_a_file"

    def test_file_too_large(self, tools: dict[str, Tool], workspace_root: Path) -> None:
        (workspace_root / "big.bin").write_bytes(b"0" * 65)
        result = call(tools, "read_text_file", {"path": "big.bin"})
        assert not result.ok and result.error_code == "file_too_large"

    def test_decode_error(self, tools: dict[str, Tool], workspace_root: Path) -> None:
        (workspace_root / "bin.dat").write_bytes(b"\xff\xfe\x00\x01")
        result = call(tools, "read_text_file", {"path": "bin.dat"})
        assert not result.ok and result.error_code == "decode_error"

    def test_explicit_encoding(self, tools: dict[str, Tool], workspace_root: Path) -> None:
        (workspace_root / "l.txt").write_bytes("café".encode("latin-1"))
        result = call(tools, "read_text_file", {"path": "l.txt", "encoding": "latin-1"})
        assert result.ok
        assert out(result)["content"] == "café"
        assert out(result)["encoding"] == "latin-1"

    def test_invalid_encoding(self, tools: dict[str, Tool], workspace_root: Path) -> None:
        (workspace_root / "a.txt").write_text("x", encoding="utf-8")
        result = call(tools, "read_text_file", {"path": "a.txt", "encoding": "no-such-charset"})
        assert not result.ok and result.error_code == "invalid_encoding"

    def test_path_outside_workspace(self, tools: dict[str, Tool]) -> None:
        result = call(tools, "read_text_file", {"path": "/etc/passwd"})
        assert not result.ok and result.error_code == "path_outside_workspace"


class TestFileInfo:
    def test_missing_path_is_success_with_exists_false(self, tools: dict[str, Tool]) -> None:
        result = call(tools, "file_info", {"path": "ghost.txt"})
        assert result.ok
        assert result.output == {
            "path": "ghost.txt",
            "exists": False,
            "is_file": False,
            "is_directory": False,
            "size_bytes": 0,
        }

    def test_file_metadata(self, tools: dict[str, Tool]) -> None:
        call(tools, "write_text_file", {"path": "a.txt", "content": "12345"})
        result = call(tools, "file_info", {"path": "a.txt"})
        assert result.ok
        assert out(result)["exists"] is True
        assert out(result)["is_file"] is True
        assert out(result)["is_directory"] is False
        assert out(result)["size_bytes"] == 5

    def test_directory_metadata(self, tools: dict[str, Tool], workspace_root: Path) -> None:
        (workspace_root / "d").mkdir()
        result = call(tools, "file_info", {"path": "d"})
        assert result.ok
        assert out(result)["is_directory"] is True
        assert out(result)["is_file"] is False

    def test_root_metadata(self, tools: dict[str, Tool]) -> None:
        result = call(tools, "file_info", {"path": ""})
        assert result.ok
        assert out(result)["path"] == ""
        assert out(result)["is_directory"] is True

    def test_path_outside_workspace(self, tools: dict[str, Tool]) -> None:
        result = call(tools, "file_info", {"path": "../x"})
        assert not result.ok and result.error_code == "path_outside_workspace"


# ---------------------------------------------------------------------------
# list_directory / create_directory
# ---------------------------------------------------------------------------


class TestListDirectory:
    def test_lists_root_sorted_with_metadata(
        self, tools: dict[str, Tool], workspace_root: Path
    ) -> None:
        (workspace_root / "b.txt").write_text("11", encoding="utf-8")
        (workspace_root / "a.txt").write_text("123", encoding="utf-8")
        (workspace_root / "d").mkdir()
        result = call(tools, "list_directory", {})
        assert result.ok
        names = [e["name"] for e in out(result)["entries"]]
        assert names == ["a.txt", "b.txt", "d"]
        by_name = {e["name"]: e for e in out(result)["entries"]}
        assert by_name["a.txt"]["type"] == "file"
        assert by_name["a.txt"]["size_bytes"] == 3
        assert by_name["a.txt"]["path"] == "a.txt"
        assert by_name["d"]["type"] == "directory"
        assert out(result)["count"] == 3
        assert out(result)["truncated"] is False

    def test_missing_directory(self, tools: dict[str, Tool]) -> None:
        result = call(tools, "list_directory", {"path": "nope"})
        assert not result.ok and result.error_code == "path_not_found"

    def test_listing_a_file(self, tools: dict[str, Tool]) -> None:
        call(tools, "write_text_file", {"path": "a.txt", "content": "x"})
        result = call(tools, "list_directory", {"path": "a.txt"})
        assert not result.ok and result.error_code == "not_a_directory"

    def test_truncation_flag(self, tools: dict[str, Tool], workspace_root: Path) -> None:
        for i in range(5):
            (workspace_root / f"f{i}.txt").write_text("x", encoding="utf-8")
        result = call(tools, "list_directory", {})
        assert result.ok
        assert out(result)["count"] == 3  # max_list_entries
        assert out(result)["truncated"] is True
        assert [e["name"] for e in out(result)["entries"]] == ["f0.txt", "f1.txt", "f2.txt"]

    def test_path_outside_workspace(self, tools: dict[str, Tool]) -> None:
        result = call(tools, "list_directory", {"path": ".."})
        assert not result.ok and result.error_code == "path_outside_workspace"

    def test_symlink_listed_as_symlink(
        self, tools: dict[str, Tool], workspace_root: Path, tmp_path: Path
    ) -> None:
        outside = tmp_path / "outside.txt"
        outside.write_text("s", encoding="utf-8")
        try:
            os.symlink(outside, workspace_root / "link.txt")
        except (OSError, NotImplementedError, ValueError):
            pytest.skip("symlinks not supported in this environment")
        result = call(tools, "list_directory", {})
        assert result.ok
        by_name = {e["name"]: e for e in out(result)["entries"]}
        assert by_name["link.txt"]["type"] == "symlink"


class TestCreateDirectory:
    def test_creates_nested(self, tools: dict[str, Tool], workspace_root: Path) -> None:
        result = call(tools, "create_directory", {"path": "a/b/c"})
        assert result.ok
        assert result.output == {"path": "a/b/c", "created": True}
        assert (workspace_root / "a/b/c").is_dir()

    def test_idempotent_when_exists(self, tools: dict[str, Tool], workspace_root: Path) -> None:
        (workspace_root / "d").mkdir()
        result = call(tools, "create_directory", {"path": "d"})
        assert result.ok
        assert out(result)["created"] is False

    def test_parents_false_requires_existing_parent(self, tools: dict[str, Tool]) -> None:
        result = call(tools, "create_directory", {"path": "x/y", "parents": False})
        assert not result.ok and result.error_code == "path_not_found"

    def test_path_is_file(self, tools: dict[str, Tool], workspace_root: Path) -> None:
        (workspace_root / "f.txt").write_text("x", encoding="utf-8")
        result = call(tools, "create_directory", {"path": "f.txt"})
        assert not result.ok and result.error_code == "not_a_directory"

    def test_root_rejected(self, tools: dict[str, Tool]) -> None:
        result = call(tools, "create_directory", {"path": ""})
        assert not result.ok and result.error_code == "unsupported_operation"

    def test_path_outside_workspace(self, tools: dict[str, Tool]) -> None:
        result = call(tools, "create_directory", {"path": "../outside"})
        assert not result.ok and result.error_code == "path_outside_workspace"


# ---------------------------------------------------------------------------
# copy_file / move_file / delete_file
# ---------------------------------------------------------------------------


class TestCopyFile:
    def test_copies(self, tools: dict[str, Tool], workspace_root: Path) -> None:
        (workspace_root / "a.txt").write_text("data", encoding="utf-8")
        result = call(tools, "copy_file", {"source": "a.txt", "destination": "b.txt"})
        assert result.ok
        assert result.output == {
            "source": "a.txt",
            "destination": "b.txt",
            "bytes_copied": 4,
            "overwritten": False,
        }
        assert (workspace_root / "b.txt").read_text(encoding="utf-8") == "data"
        assert (workspace_root / "a.txt").exists()

    def test_existing_destination_without_overwrite(
        self, tools: dict[str, Tool], workspace_root: Path
    ) -> None:
        (workspace_root / "a.txt").write_text("a", encoding="utf-8")
        (workspace_root / "b.txt").write_text("b", encoding="utf-8")
        result = call(tools, "copy_file", {"source": "a.txt", "destination": "b.txt"})
        assert not result.ok and result.error_code == "target_exists"
        assert (workspace_root / "b.txt").read_text(encoding="utf-8") == "b"

    def test_overwrite_true(self, tools: dict[str, Tool], workspace_root: Path) -> None:
        (workspace_root / "a.txt").write_text("newer", encoding="utf-8")
        (workspace_root / "b.txt").write_text("old", encoding="utf-8")
        result = call(
            tools,
            "copy_file",
            {"source": "a.txt", "destination": "b.txt", "overwrite": True},
        )
        assert result.ok
        assert out(result)["overwritten"] is True
        assert (workspace_root / "b.txt").read_text(encoding="utf-8") == "newer"

    def test_source_missing(self, tools: dict[str, Tool]) -> None:
        result = call(tools, "copy_file", {"source": "ghost.txt", "destination": "b.txt"})
        assert not result.ok and result.error_code == "source_not_found"

    def test_source_is_directory(self, tools: dict[str, Tool], workspace_root: Path) -> None:
        (workspace_root / "d").mkdir()
        result = call(tools, "copy_file", {"source": "d", "destination": "b.txt"})
        assert not result.ok and result.error_code == "not_a_file"

    def test_destination_directory_missing(
        self, tools: dict[str, Tool], workspace_root: Path
    ) -> None:
        (workspace_root / "a.txt").write_text("x", encoding="utf-8")
        result = call(tools, "copy_file", {"source": "a.txt", "destination": "nope/b.txt"})
        assert not result.ok and result.error_code == "path_not_found"

    def test_same_source_and_destination(
        self, tools: dict[str, Tool], workspace_root: Path
    ) -> None:
        (workspace_root / "a.txt").write_text("x", encoding="utf-8")
        result = call(tools, "copy_file", {"source": "a.txt", "destination": "a.txt"})
        assert not result.ok and result.error_code == "invalid_path"

    def test_source_too_large(self, tools: dict[str, Tool], workspace_root: Path) -> None:
        (workspace_root / "big.bin").write_bytes(b"0" * 65)
        result = call(tools, "copy_file", {"source": "big.bin", "destination": "b.txt"})
        assert not result.ok and result.error_code == "file_too_large"

    def test_destination_outside_workspace(
        self, tools: dict[str, Tool], workspace_root: Path
    ) -> None:
        (workspace_root / "a.txt").write_text("x", encoding="utf-8")
        result = call(tools, "copy_file", {"source": "a.txt", "destination": "../evil.txt"})
        assert not result.ok and result.error_code == "path_outside_workspace"


class TestMoveFile:
    def test_moves(self, tools: dict[str, Tool], workspace_root: Path) -> None:
        (workspace_root / "a.txt").write_text("payload", encoding="utf-8")
        result = call(tools, "move_file", {"source": "a.txt", "destination": "b.txt"})
        assert result.ok
        assert result.output == {
            "source": "a.txt",
            "destination": "b.txt",
            "bytes_moved": 7,
            "overwritten": False,
        }
        assert not (workspace_root / "a.txt").exists()
        assert (workspace_root / "b.txt").read_text(encoding="utf-8") == "payload"

    def test_existing_destination_without_overwrite(
        self, tools: dict[str, Tool], workspace_root: Path
    ) -> None:
        (workspace_root / "a.txt").write_text("a", encoding="utf-8")
        (workspace_root / "b.txt").write_text("b", encoding="utf-8")
        result = call(tools, "move_file", {"source": "a.txt", "destination": "b.txt"})
        assert not result.ok and result.error_code == "target_exists"
        assert (workspace_root / "a.txt").exists()  # source untouched

    def test_overwrite_true(self, tools: dict[str, Tool], workspace_root: Path) -> None:
        (workspace_root / "a.txt").write_text("newer", encoding="utf-8")
        (workspace_root / "b.txt").write_text("old", encoding="utf-8")
        result = call(
            tools,
            "move_file",
            {"source": "a.txt", "destination": "b.txt", "overwrite": True},
        )
        assert result.ok
        assert out(result)["overwritten"] is True
        assert (workspace_root / "b.txt").read_text(encoding="utf-8") == "newer"

    def test_source_missing(self, tools: dict[str, Tool]) -> None:
        result = call(tools, "move_file", {"source": "ghost.txt", "destination": "b.txt"})
        assert not result.ok and result.error_code == "source_not_found"

    def test_destination_directory_missing(
        self, tools: dict[str, Tool], workspace_root: Path
    ) -> None:
        (workspace_root / "a.txt").write_text("x", encoding="utf-8")
        result = call(tools, "move_file", {"source": "a.txt", "destination": "nope/b.txt"})
        assert not result.ok and result.error_code == "path_not_found"

    def test_same_source_and_destination(
        self, tools: dict[str, Tool], workspace_root: Path
    ) -> None:
        (workspace_root / "a.txt").write_text("x", encoding="utf-8")
        result = call(tools, "move_file", {"source": "a.txt", "destination": "a.txt"})
        assert not result.ok and result.error_code == "invalid_path"

    def test_destination_outside_workspace(
        self, tools: dict[str, Tool], workspace_root: Path
    ) -> None:
        (workspace_root / "a.txt").write_text("x", encoding="utf-8")
        result = call(tools, "move_file", {"source": "a.txt", "destination": "../evil.txt"})
        assert not result.ok and result.error_code == "path_outside_workspace"
        assert (workspace_root / "a.txt").exists()


class TestDeleteFile:
    def test_deletes(self, tools: dict[str, Tool], workspace_root: Path) -> None:
        (workspace_root / "a.txt").write_text("12345", encoding="utf-8")
        result = call(tools, "delete_file", {"path": "a.txt"})
        assert result.ok
        assert result.output == {"path": "a.txt", "bytes_deleted": 5}
        assert not (workspace_root / "a.txt").exists()

    def test_missing_path(self, tools: dict[str, Tool]) -> None:
        result = call(tools, "delete_file", {"path": "ghost.txt"})
        assert not result.ok and result.error_code == "path_not_found"

    def test_directory_never_deleted(self, tools: dict[str, Tool], workspace_root: Path) -> None:
        (workspace_root / "d").mkdir()
        (workspace_root / "d" / "x.txt").write_text("keep", encoding="utf-8")
        result = call(tools, "delete_file", {"path": "d"})
        assert not result.ok and result.error_code == "unsupported_operation"
        assert (workspace_root / "d" / "x.txt").exists()  # nothing removed

    def test_root_never_deleted(self, tools: dict[str, Tool]) -> None:
        result = call(tools, "delete_file", {"path": ""})
        assert not result.ok and result.error_code == "unsupported_operation"

    def test_path_outside_workspace(self, tools: dict[str, Tool]) -> None:
        result = call(tools, "delete_file", {"path": "../outside.txt"})
        assert not result.ok and result.error_code == "path_outside_workspace"


# ---------------------------------------------------------------------------
# search_files
# ---------------------------------------------------------------------------


class TestSearchFiles:
    def test_finds_by_pattern(self, tools: dict[str, Tool], workspace_root: Path) -> None:
        (workspace_root / "a.txt").write_text("1", encoding="utf-8")
        (workspace_root / "sub").mkdir()
        (workspace_root / "sub" / "b.txt").write_text("22", encoding="utf-8")
        (workspace_root / "c.md").write_text("333", encoding="utf-8")
        result = call(tools, "search_files", {"pattern": "*.txt"})
        assert result.ok
        paths = sorted(e["path"] for e in out(result)["entries"])
        assert paths == ["a.txt", "sub/b.txt"]
        assert out(result)["count"] == 2
        assert out(result)["truncated"] is False
        assert all(e["type"] == "file" for e in out(result)["entries"])

    def test_star_crosses_slashes(self, tools: dict[str, Tool], workspace_root: Path) -> None:
        """Glob ``*`` matches across directory separators (fnmatch semantics)."""
        (workspace_root / "sub").mkdir()
        (workspace_root / "sub" / "deep").mkdir()
        (workspace_root / "sub" / "deep" / "x.txt").write_text("x", encoding="utf-8")
        result = call(tools, "search_files", {"pattern": "sub/*.txt"})
        assert result.ok
        assert [e["path"] for e in out(result)["entries"]] == ["sub/deep/x.txt"]

    def test_non_recursive(self, tools: dict[str, Tool], workspace_root: Path) -> None:
        (workspace_root / "a.txt").write_text("1", encoding="utf-8")
        (workspace_root / "sub").mkdir()
        (workspace_root / "sub" / "b.txt").write_text("2", encoding="utf-8")
        result = call(tools, "search_files", {"pattern": "*.txt", "recursive": False})
        assert result.ok
        assert [e["path"] for e in out(result)["entries"]] == ["a.txt"]

    def test_scoped_to_path(self, tools: dict[str, Tool], workspace_root: Path) -> None:
        (workspace_root / "a.txt").write_text("1", encoding="utf-8")
        (workspace_root / "sub").mkdir()
        (workspace_root / "sub" / "b.txt").write_text("2", encoding="utf-8")
        result = call(tools, "search_files", {"pattern": "*.txt", "path": "sub"})
        assert result.ok
        assert out(result)["path"] == "sub"
        assert [e["path"] for e in out(result)["entries"]] == ["sub/b.txt"]

    def test_no_matches(self, tools: dict[str, Tool]) -> None:
        result = call(tools, "search_files", {"pattern": "*.nope"})
        assert result.ok
        assert out(result)["entries"] == []
        assert out(result)["count"] == 0

    def test_truncated_flag(self, tools: dict[str, Tool], workspace_root: Path) -> None:
        for i in range(5):
            (workspace_root / f"f{i}.txt").write_text("x", encoding="utf-8")
        result = call(tools, "search_files", {"pattern": "*.txt"})
        assert result.ok
        assert out(result)["count"] == 3  # max_search_results
        assert out(result)["truncated"] is True

    def test_pattern_is_case_sensitive(self, tools: dict[str, Tool], workspace_root: Path) -> None:
        (workspace_root / "A.TXT").write_text("x", encoding="utf-8")
        result = call(tools, "search_files", {"pattern": "*.txt"})
        assert result.ok
        assert out(result)["entries"] == []

    def test_missing_pattern(self, tools: dict[str, Tool]) -> None:
        result = call(tools, "search_files", {})
        assert not result.ok and result.error_code == "invalid_path"

    def test_empty_pattern(self, tools: dict[str, Tool]) -> None:
        result = call(tools, "search_files", {"pattern": ""})
        assert not result.ok and result.error_code == "invalid_path"

    def test_overlong_pattern(self, tools: dict[str, Tool]) -> None:
        result = call(tools, "search_files", {"pattern": "a" * 201})
        assert not result.ok and result.error_code == "invalid_path"

    def test_search_path_is_file(self, tools: dict[str, Tool]) -> None:
        call(tools, "write_text_file", {"path": "a.txt", "content": "x"})
        result = call(tools, "search_files", {"pattern": "*", "path": "a.txt"})
        assert not result.ok and result.error_code == "not_a_directory"

    def test_search_path_missing(self, tools: dict[str, Tool]) -> None:
        result = call(tools, "search_files", {"pattern": "*", "path": "nope"})
        assert not result.ok and result.error_code == "path_not_found"

    def test_search_never_escapes_workspace(
        self, tools: dict[str, Tool], workspace_root: Path, tmp_path: Path
    ) -> None:
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.txt").write_text("s", encoding="utf-8")
        try:
            os.symlink(outside, workspace_root / "dirlink", target_is_directory=True)
        except (OSError, NotImplementedError, ValueError):
            pytest.skip("symlinks not supported in this environment")
        result = call(tools, "search_files", {"pattern": "*.txt"})
        assert result.ok
        assert all("secret" not in e["path"] for e in out(result)["entries"])


# ---------------------------------------------------------------------------
# Permission behavior + agent integration
# ---------------------------------------------------------------------------


def _plan_json(tool_name: str, tool_input: dict[str, Any]) -> str:
    return json.dumps(
        {
            "steps": [
                {"tool_name": tool_name, "description": f"run {tool_name}", "input": tool_input}
            ]
        }
    )


Clock = Callable[[], datetime]


def build_workspace_agent(
    workspace: Workspace,
    plan_json: str,
    *,
    approval: ApprovalCallback | None = None,
    policy: PermissionPolicy | None = None,
) -> Agent:
    registry: ToolRegistry = ToolRegistry()
    register_workspace_tools(registry, workspace)
    return Agent(
        planner=ModelPlanner(MockModelProvider(responses=[plan_json])),
        registry=registry,
        permissions=PermissionManager(policy=policy, approval=approval),
        events=EventBus(clock=lambda: FIXED),
        clock=lambda: FIXED,
    )


class TestPermissionBehavior:
    def test_low_permission_runs_without_approval(
        self, workspace: Workspace, workspace_root: Path
    ) -> None:
        (workspace_root / "a.txt").write_text("data", encoding="utf-8")
        agent = build_workspace_agent(workspace, _plan_json("read_text_file", {"path": "a.txt"}))
        task = agent.run("read it")
        assert task.state is TaskState.COMPLETED
        step = task.steps[0]
        assert step.status is StepStatus.COMPLETED
        assert step.output is not None
        assert step.output["content"] == "data"
        # No approval was needed.
        assert agent.events.events_of_type(EventType.APPROVAL_REQUIRED) == []

    def test_medium_write_with_approval_completes(
        self, workspace: Workspace, workspace_root: Path
    ) -> None:
        agent = build_workspace_agent(
            workspace,
            _plan_json("write_text_file", {"path": "n.txt", "content": "hello"}),
            approval=lambda _r: True,
        )
        task = agent.run("write it")
        assert task.state is TaskState.COMPLETED
        assert (workspace_root / "n.txt").read_text(encoding="utf-8") == "hello"
        approval = agent.events.events_of_type(EventType.APPROVAL_REQUIRED)[0]
        assert approval.data["tool_name"] == "write_text_file"
        assert approval.data["permission_level"] == "MEDIUM"

    def test_medium_write_without_approval_denied_and_no_file_written(
        self, workspace: Workspace, workspace_root: Path
    ) -> None:
        agent = build_workspace_agent(
            workspace,
            _plan_json("write_text_file", {"path": "n.txt", "content": "hello"}),
            approval=lambda _r: False,
        )
        task = agent.run("write it")
        assert task.state is TaskState.CANCELLED
        assert not (workspace_root / "n.txt").exists()  # NO filesystem action after denial
        denied = agent.events.events_of_type(EventType.TOOL_DENIED)[0]
        assert denied.data["reason"] == "approval_denied"

    def test_high_delete_with_approval_completes(
        self, workspace: Workspace, workspace_root: Path
    ) -> None:
        (workspace_root / "a.txt").write_text("12345", encoding="utf-8")
        agent = build_workspace_agent(
            workspace, _plan_json("delete_file", {"path": "a.txt"}), approval=lambda _r: True
        )
        task = agent.run("delete it")
        assert task.state is TaskState.COMPLETED
        assert not (workspace_root / "a.txt").exists()
        approval = agent.events.events_of_type(EventType.APPROVAL_REQUIRED)[0]
        assert approval.data["permission_level"] == "HIGH"

    def test_high_delete_without_channel_fail_safe_no_file_deleted(
        self, workspace: Workspace, workspace_root: Path
    ) -> None:
        (workspace_root / "a.txt").write_text("keep", encoding="utf-8")
        agent = build_workspace_agent(workspace, _plan_json("delete_file", {"path": "a.txt"}))
        task = agent.run("delete it")
        assert task.state is TaskState.CANCELLED
        assert (workspace_root / "a.txt").read_text(encoding="utf-8") == "keep"

    def test_high_delete_denied_by_policy_even_with_approval(
        self, workspace: Workspace, workspace_root: Path
    ) -> None:
        (workspace_root / "a.txt").write_text("keep", encoding="utf-8")
        policy = PermissionPolicy(high=PermissionDecision.DENIED)
        agent = build_workspace_agent(
            workspace,
            _plan_json("delete_file", {"path": "a.txt"}),
            approval=lambda _r: True,
            policy=policy,
        )
        task = agent.run("delete it")
        assert task.state is TaskState.CANCELLED
        assert (workspace_root / "a.txt").read_text(encoding="utf-8") == "keep"
        denied = agent.events.events_of_type(EventType.TOOL_DENIED)[0]
        assert denied.data["reason"] == "denied_by_policy"


class TestFactoryWiring:
    def test_create_configured_registers_workspace_tools(self, tmp_path: Path) -> None:
        settings = Settings(workspace_root=tmp_path / "ws")
        agent = Agent.create_configured(settings=settings)
        names = {spec.name for spec in agent.registry.list_tools()}
        # Phase 4 topology change (documented): the configured agent also
        # registers the 4 document tools.
        # Phase 5 topology change (documented): the configured agent also
        # registers the 5 memory tools.
        expected = {
            "list_directory",
            "read_text_file",
            "write_text_file",
            "create_directory",
            "copy_file",
            "move_file",
            "delete_file",
            "file_info",
            "search_files",
            "demo_tool",
            "calculator",
            "datetime",
            "json_utils",
            "text_utils",
            "inspect_document",
            "extract_document",
            "index_document",
            "search_documents",
            "remember",
            "recall",
            "update_memory",
            "forget",
            "list_memories",
        }
        assert names == expected

    def test_create_demo_registers_workspace_tools(self) -> None:
        agent = Agent.create_demo()
        names = {spec.name for spec in agent.registry.list_tools()}
        assert "delete_file" in names
        assert "search_files" in names
        assert "demo_tool" in names


# ---------------------------------------------------------------------------
# Tool runtime + event bounding integration
# ---------------------------------------------------------------------------


class TestRuntimeIntegration:
    def test_lifecycle_events_with_bounded_payload(
        self, workspace: Workspace, workspace_root: Path
    ) -> None:
        registry: ToolRegistry = ToolRegistry()
        register_workspace_tools(registry, workspace)
        events = EventBus(clock=lambda: FIXED)
        runtime = ToolRuntime(registry, events)
        big = "x" * 3000
        (workspace_root / "big.txt").write_text(big, encoding="utf-8")
        invocation = ToolInvocation(
            task_id="t1",
            step_id="s1",
            tool_name="read_text_file",
            input={"path": "big.txt"},
        )
        result = runtime.execute(invocation, decision=PermissionDecision.ALLOWED)
        assert result.ok
        # Full content is returned to the model via the result...
        assert out(result)["content"] == big
        # ...but the event payload keeps only a bounded excerpt.
        completed = events.events_of_type(EventType.TOOL_COMPLETED)[0]
        payload = completed.data["output"]
        assert isinstance(payload["content"], str)
        assert len(payload["content"]) <= 200
        assert payload["content"] != big
        started = events.events_of_type(EventType.TOOL_STARTED)[0]
        assert started.data["tool_name"] == "read_text_file"

    def test_denied_decision_never_executes(self, workspace: Workspace) -> None:
        registry: ToolRegistry = ToolRegistry()
        register_workspace_tools(registry, workspace)
        events = EventBus(clock=lambda: FIXED)
        runtime = ToolRuntime(registry, events)
        (workspace.root / "a.txt").write_text("x", encoding="utf-8")
        invocation = ToolInvocation(
            task_id="t1", step_id="s1", tool_name="delete_file", input={"path": "a.txt"}
        )
        from agent_core.errors import PermissionDeniedError

        with pytest.raises(PermissionDeniedError):
            runtime.execute(invocation, decision=PermissionDecision.REQUIRES_APPROVAL)
        assert (workspace.root / "a.txt").exists()  # nothing deleted
        assert events.history == []

    def test_tool_failure_is_structured_not_fatal(self, workspace: Workspace) -> None:
        registry: ToolRegistry = ToolRegistry()
        register_workspace_tools(registry, workspace)
        events = EventBus(clock=lambda: FIXED)
        runtime = ToolRuntime(registry, events)
        invocation = ToolInvocation(
            task_id="t1", step_id="s1", tool_name="read_text_file", input={"path": "ghost.txt"}
        )
        result = runtime.execute(invocation, decision=PermissionDecision.ALLOWED)
        assert not result.ok
        assert result.error_code == "path_not_found"
        failed = events.events_of_type(EventType.TOOL_FAILED)[0]
        assert failed.data["error_code"] == "path_not_found"


# ---------------------------------------------------------------------------
# Spec permission-level contract (regression guard)
# ---------------------------------------------------------------------------


class TestPermissionLevelsContract:
    def test_levels_match_spec(self, workspace: Workspace) -> None:
        from agent_core import PermissionLevel

        registry: ToolRegistry = ToolRegistry()
        register_workspace_tools(registry, workspace)
        levels = {spec.name: spec.permission_level for spec in registry.list_tools()}
        assert levels["list_directory"] is PermissionLevel.LOW
        assert levels["read_text_file"] is PermissionLevel.LOW
        assert levels["file_info"] is PermissionLevel.LOW
        assert levels["search_files"] is PermissionLevel.LOW
        assert levels["create_directory"] is PermissionLevel.MEDIUM
        assert levels["write_text_file"] is PermissionLevel.MEDIUM
        assert levels["copy_file"] is PermissionLevel.MEDIUM
        assert levels["move_file"] is PermissionLevel.MEDIUM
        assert levels["delete_file"] is PermissionLevel.HIGH
