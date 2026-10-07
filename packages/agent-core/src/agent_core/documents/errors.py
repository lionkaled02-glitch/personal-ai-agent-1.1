"""Document processing errors (Phase 4).

Document failures are structured: a stable machine-readable ``code`` plus a
concise message. Messages contain only workspace-relative paths — never the
absolute host path, never document content.
"""

from __future__ import annotations

from ..errors import AgentCoreError

# The stable document error codes (Phase 4). Tool layers and parsers use
# these; extensions beyond the Phase 4 spec are marked as such in docs.
DOCUMENT_NOT_FOUND = "document_not_found"
UNSUPPORTED_DOCUMENT_TYPE = "unsupported_document_type"
DOCUMENT_TOO_LARGE = "document_too_large"
DOCUMENT_CORRUPT = "document_corrupt"
EXTRACTION_FAILED = "extraction_failed"
DECODE_FAILED = "decode_failed"
PARSER_LIMIT_EXCEEDED = "parser_limit_exceeded"
INVALID_DOCUMENT = "invalid_document"
SECURITY_VIOLATION = "security_violation"
PARSER_UNAVAILABLE = "parser_unavailable"  # extension: optional parser library missing
INVALID_QUERY = "invalid_query"  # extension: malformed search query input
INVALID_INPUT = "invalid_input"  # extension: malformed scalar tool input
DOCUMENT_NOT_INDEXED = "document_not_indexed"  # extension: unknown document id
CHUNK_NOT_FOUND = "chunk_not_found"  # extension: unknown chunk id


class DocumentError(AgentCoreError):
    """A document operation was rejected or failed.

    ``code`` is a stable machine-readable error class; ``message`` is concise
    and never leaks host paths or document content.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
