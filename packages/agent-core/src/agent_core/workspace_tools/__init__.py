"""Workspace filesystem tools (Phase 3).

Nine provider-agnostic tools operating only inside the configured
workspace boundary:

- LOW:    list_directory, read_text_file, file_info, search_files
- MEDIUM: create_directory, write_text_file, copy_file, move_file
- HIGH:   delete_file (requires explicit approval)

Registration requires an explicit :class:`~agent_core.workspace.Workspace`;
there is no implicit or host-root fallback.
"""

from __future__ import annotations

from ..tools import ToolRegistry
from ..workspace import Workspace
from .directory_tools import (
    CREATE_DIR_TOOL_NAME,
    LIST_TOOL_NAME,
    CreateDirectoryTool,
    ListDirectoryTool,
)
from .file_tools import (
    INFO_TOOL_NAME,
    READ_TOOL_NAME,
    WRITE_TOOL_NAME,
    FileInfoTool,
    ReadTextFileTool,
    WriteTextFileTool,
)
from .search_tools import SEARCH_TOOL_NAME, SearchFilesTool
from .transfer_tools import (
    COPY_TOOL_NAME,
    DELETE_TOOL_NAME,
    MOVE_TOOL_NAME,
    CopyFileTool,
    DeleteFileTool,
    MoveFileTool,
)

__all__ = [
    "COPY_TOOL_NAME",
    "CREATE_DIR_TOOL_NAME",
    "DELETE_TOOL_NAME",
    "INFO_TOOL_NAME",
    "LIST_TOOL_NAME",
    "MOVE_TOOL_NAME",
    "READ_TOOL_NAME",
    "SEARCH_TOOL_NAME",
    "WORKSPACE_TOOL_NAMES",
    "WRITE_TOOL_NAME",
    "CopyFileTool",
    "CreateDirectoryTool",
    "DeleteFileTool",
    "FileInfoTool",
    "ListDirectoryTool",
    "MoveFileTool",
    "ReadTextFileTool",
    "SearchFilesTool",
    "WriteTextFileTool",
    "register_workspace_tools",
]

WORKSPACE_TOOL_NAMES: tuple[str, ...] = (
    COPY_TOOL_NAME,
    CREATE_DIR_TOOL_NAME,
    DELETE_TOOL_NAME,
    INFO_TOOL_NAME,
    LIST_TOOL_NAME,
    MOVE_TOOL_NAME,
    READ_TOOL_NAME,
    SEARCH_TOOL_NAME,
    WRITE_TOOL_NAME,
)


def register_workspace_tools(registry: ToolRegistry, workspace: Workspace) -> None:
    """Registers all nine workspace tools against the given workspace."""
    registry.register(ListDirectoryTool(workspace))
    registry.register(ReadTextFileTool(workspace))
    registry.register(WriteTextFileTool(workspace))
    registry.register(CreateDirectoryTool(workspace))
    registry.register(CopyFileTool(workspace))
    registry.register(MoveFileTool(workspace))
    registry.register(DeleteFileTool(workspace))
    registry.register(FileInfoTool(workspace))
    registry.register(SearchFilesTool(workspace))
