"""User-facing error types.

These carry messages that are safe to show directly in the CLI or the Streamlit UI: a
plain-language explanation and, where useful, a recovery hint. They never contain stack
traces or secrets. Front-ends should catch :class:`PaperTrailError` and render
``str(exc)`` as a clean error message instead of letting a raw exception break the page.
"""

from __future__ import annotations


class PaperTrailError(Exception):
    """Base class for predictable, user-facing errors."""


class IngestionError(PaperTrailError):
    """A paper or repository could not be ingested.

    Raised for predictable failures (bad URL, clone failure, unreadable PDF, no
    supported files) with a message the user can act on.
    """


class ConfigurationError(PaperTrailError):
    """A required credential or setting is missing/invalid (e.g. no Voyage/Bedrock key)."""
