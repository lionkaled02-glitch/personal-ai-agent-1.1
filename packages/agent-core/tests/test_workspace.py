"""Workspace boundary security tests (Phase 3).

Proves the :class:`~agent_core.workspace.Workspace` fails closed: every
escape vector (absolute paths, ``../`` traversal, symlink/junction redirects,
malformed paths) is rejected with a stable error code, and error messages
never leak the absolute host path.

Also covers the Phase 3 settings/env wiring and the ``bounded_value`` event
helper. All tests are deterministic and use temporary directories.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from agent_core import Settings, Workspace, WorkspaceError, WorkspaceLimits, bounded_value
from agent_core.config import Settings as _Settings

# ---------------------------------------------------------------------------
# Workspace.resolve()
# ---------------------------------------------------------------------------


@pytest.fixture
def root(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    root.mkdir()
    return root


@pytest.fixture
def ws(root: Path) -> Workspace:
    return Workspace(root)


class TestResolveNormal:
    def test_empty_path_is_root(self, ws: Workspace, root: Path) -> None:
        assert ws.resolve("") == root
        assert ws.resolve("   ") == root

    def test_relative_path_resolves_inside(self, ws: Workspace, root: Path) -> None:
        (root / "a").mkdir()
        (root / "a" / "b.txt").write_text("x", encoding="utf-8")
        assert ws.resolve("a/b.txt") == root / "a" / "b.txt"

    def test_dot_segments_collapse(self, ws: Workspace, root: Path) -> None:
        (root / "a").mkdir()
        assert ws.resolve("a/./b/../c.txt") == root / "a" / "c.txt"

    def test_relative_to_root_gives_posix_path(self, ws: Workspace, root: Path) -> None:
        resolved = ws.resolve("a/b.txt")
        assert ws.relative_to_root(resolved) == "a/b.txt"

    def test_relative_to_root_of_root_is_empty(self, ws: Workspace, root: Path) -> None:
        assert ws.relative_to_root(root) == ""

    def test_dotdot_cancelling_inside_is_safe(self, ws: Workspace, root: Path) -> None:
        """``a/..`` resolves to the workspace root — inside the boundary."""
        (root / "a").mkdir()
        assert ws.resolve("a/..") == root

    def test_host_path_never_leaks_in_relative(self, ws: Workspace, root: Path) -> None:
        resolved = ws.resolve("a/b.txt")
        assert str(root) not in ws.relative_to_root(resolved)
        assert "\\\\" not in ws.relative_to_root(resolved)


class TestResolveRejections:
    @pytest.mark.parametrize(
        "raw",
        [
            "../outside.txt",
            "a/../../outside.txt",
            "..",
        ],
    )
    def test_dotdot_traversal_rejected(self, ws: Workspace, raw: str) -> None:
        with pytest.raises(WorkspaceError) as exc:
            ws.resolve(raw)
        assert exc.value.code == "path_outside_workspace"

    @pytest.mark.parametrize(
        "raw",
        [
            "/etc/passwd",
            "/etc/hostname",
        ],
    )
    def test_absolute_posix_paths_rejected(self, ws: Workspace, raw: str) -> None:
        with pytest.raises(WorkspaceError) as exc:
            ws.resolve(raw)
        assert exc.value.code == "path_outside_workspace"

    def test_absolute_path_inside_root_still_rejected(self, ws: Workspace, root: Path) -> None:
        """Even an absolute path that lands inside the root breaks the
        workspace-relative contract and must be rejected."""
        with pytest.raises(WorkspaceError) as exc:
            ws.resolve(str(root / "inside.txt"))
        assert exc.value.code == "path_outside_workspace"

    def test_windows_absolute_forms_rejected(self, ws: Workspace) -> None:
        for raw in ("C:\\Users\\x", "C:/Users/x", "\\\\server\\share"):
            with pytest.raises(WorkspaceError) as exc:
                ws.resolve(raw)
            assert exc.value.code == "path_outside_workspace"

    def test_nul_byte_rejected(self, ws: Workspace) -> None:
        with pytest.raises(WorkspaceError) as exc:
            ws.resolve("a\x00b.txt")
        assert exc.value.code == "invalid_path"

    def test_overlong_path_rejected(self, ws: Workspace) -> None:
        raw = "a" * 600
        with pytest.raises(WorkspaceError) as exc:
            ws.resolve(raw)
        assert exc.value.code == "invalid_path"

    def test_non_string_rejected(self, ws: Workspace) -> None:
        with pytest.raises(WorkspaceError) as exc:
            ws.resolve(123)  # type: ignore[arg-type]
        assert exc.value.code == "invalid_path"
        with pytest.raises(WorkspaceError) as exc:
            ws.resolve(None)  # type: ignore[arg-type]
        assert exc.value.code == "invalid_path"

    def test_message_never_contains_host_path(self, ws: Workspace, root: Path) -> None:
        with pytest.raises(WorkspaceError) as exc:
            ws.resolve("../outside.txt")
        assert str(root) not in exc.value.message

    def test_error_code_is_stable_attribute(self, ws: Workspace) -> None:
        with pytest.raises(WorkspaceError) as exc:
            ws.resolve("../x")
        assert isinstance(exc.value.code, str)
        assert exc.value.code == "path_outside_workspace"


class TestSymlinkEscapes:
    """Symlink / junction / reparse-point redirects must not escape the
    boundary. ``Path.resolve()`` follows the whole chain, so the final
    containment check catches these. Skipped on platforms where creating
    symlinks is not permitted."""

    @pytest.fixture
    def escape_root(self, tmp_path: Path) -> tuple[Workspace, Path, Path]:
        root = tmp_path / "ws"
        root.mkdir()
        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()
        (outside_dir / "secret.txt").write_text("secret", encoding="utf-8")
        file_link = root / "link.txt"
        try:
            os.symlink(outside_dir / "secret.txt", file_link)
        except (OSError, NotImplementedError, ValueError):
            pytest.skip("symlinks not supported in this environment")
        ws = Workspace(root)
        return ws, outside_dir, file_link

    def test_file_symlink_outside_rejected(self, escape_root: tuple[Workspace, Path, Path]) -> None:
        ws, _outside_dir, _file_link = escape_root
        with pytest.raises(WorkspaceError) as exc:
            ws.resolve("link.txt")
        assert exc.value.code == "path_outside_workspace"

    def test_directory_symlink_outside_rejected(
        self, escape_root: tuple[Workspace, Path, Path]
    ) -> None:
        ws, outside_dir, _file_link = escape_root
        dir_link = ws.root / "dirlink"
        os.symlink(outside_dir, dir_link, target_is_directory=True)
        with pytest.raises(WorkspaceError) as exc:
            ws.resolve("dirlink/secret.txt")
        assert exc.value.code == "path_outside_workspace"

    def test_walk_files_never_follows_symlinks(
        self, escape_root: tuple[Workspace, Path, Path]
    ) -> None:
        ws, outside_dir, _file_link = escape_root
        dir_link = ws.root / "dirlink"
        os.symlink(outside_dir, dir_link, target_is_directory=True)
        walked = list(ws.walk_files(ws.root))
        # The symlinks themselves are not regular files and must not yield
        # the file beyond the link.
        assert all(p == ws.root / "link.txt" or ws.root in p.parents for p in walked)
        assert not any("secret.txt" in p.name for p in walked)

    def test_walk_files_applies_directory_exclusions_and_file_cap(self, root: Path) -> None:
        (root / "src").mkdir()
        (root / "vendor").mkdir()
        (root / "src" / "a.py").write_text("a = 1", encoding="utf-8")
        (root / "src" / "b.py").write_text("b = 2", encoding="utf-8")
        (root / "vendor" / "ignored.py").write_text("x = 3", encoding="utf-8")
        ws = Workspace(root)

        walked = list(ws.walk_files(root, skip_directories={"VENDOR"}, max_files=1))

        assert len(walked) == 1
        assert walked[0].parent == root / "src"

    def test_walk_files_fails_closed_when_entry_budget_is_exceeded(self, root: Path) -> None:
        (root / "one").mkdir()
        (root / "two").mkdir()
        ws = Workspace(root)

        with pytest.raises(WorkspaceError) as exc:
            list(ws.walk_files(root, max_entries=1))

        assert exc.value.code == "traversal_limit"


class TestLimitsConfig:
    def test_custom_limits(self, root: Path) -> None:
        limits = WorkspaceLimits(max_read_bytes=10, max_path_length=20)
        ws = Workspace(root, limits)
        with pytest.raises(WorkspaceError) as exc:
            ws.resolve("a" * 25)
        assert exc.value.code == "invalid_path"

    def test_from_settings_maps_all_fields(self, tmp_path: Path) -> None:
        settings = Settings(
            workspace_root=tmp_path / "ws2",
            workspace_max_read_bytes=111,
            workspace_max_write_bytes=222,
            workspace_max_list_entries=33,
            workspace_max_search_results=44,
            workspace_max_path_length=55,
        )
        ws = Workspace.from_settings(settings)
        assert ws.root == (tmp_path / "ws2").resolve()
        assert ws.limits.max_read_bytes == 111
        assert ws.limits.max_write_bytes == 222
        assert ws.limits.max_list_entries == 33
        assert ws.limits.max_search_results == 44
        assert ws.limits.max_path_length == 55

    def test_settings_env_wiring(self, tmp_path: Path) -> None:
        env = {
            "WORKSPACE_ROOT": str(tmp_path / "envws"),
            "WORKSPACE_MAX_READ_BYTES": "1234",
            "WORKSPACE_MAX_WRITE_BYTES": "4321",
            "WORKSPACE_MAX_LIST_ENTRIES": "7",
            "WORKSPACE_MAX_SEARCH_RESULTS": "9",
            "WORKSPACE_MAX_PATH_LENGTH": "32",
        }
        settings = _Settings.from_env(env)
        assert settings.workspace_root == tmp_path / "envws"
        assert settings.workspace_max_read_bytes == 1234
        assert settings.workspace_max_write_bytes == 4321
        assert settings.workspace_max_list_entries == 7
        assert settings.workspace_max_search_results == 9
        assert settings.workspace_max_path_length == 32

    def test_settings_defaults(self) -> None:
        settings = Settings()
        assert settings.workspace_root == Path("data/workspace")
        assert settings.workspace_max_read_bytes == 1_048_576
        assert settings.workspace_max_write_bytes == 1_048_576
        assert settings.workspace_max_list_entries == 500
        assert settings.workspace_max_search_results == 200
        assert settings.workspace_max_path_length == 512


# ---------------------------------------------------------------------------
# bounded_value (event payload bounding, Phase 3)
# ---------------------------------------------------------------------------


class TestBoundedValue:
    def test_short_values_pass_through_unchanged(self) -> None:
        value = {"a": 1, "b": True, "c": None, "d": "short", "e": [1, 2]}
        assert bounded_value(value) == value

    def test_long_string_truncated(self) -> None:
        result = bounded_value("x" * 500)
        assert isinstance(result, str)
        assert len(result) == 200
        assert result.endswith("…")

    def test_dict_recurses_into_values(self) -> None:
        result = bounded_value({"content": "y" * 300, "n": 5})
        assert len(result["content"]) == 200
        assert result["n"] == 5

    def test_list_capped_with_marker(self) -> None:
        result = bounded_value(list(range(15)), max_items=10)
        assert len(result) == 11
        assert result[10] == "… (5 more)"
        assert result[:10] == list(range(10))

    def test_nested_list_in_dict(self) -> None:
        result = bounded_value({"items": ["v" * 300] * 5}, max_items=2)
        assert result["items"][0].endswith("…")
        assert len(result["items"]) == 3
        assert result["items"][2] == "… (3 more)"
